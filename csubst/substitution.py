import numpy
import pandas
import scipy.sparse

import os
import time
from collections import defaultdict

from csubst import table
from csubst import parallel
from csubst import substitution_cy
from csubst import substitution_sparse

def resolve_sub_tensor_backend(g):
    if 'resolved_sub_tensor_backend' in g.keys():
        return g['resolved_sub_tensor_backend']
    requested = str(g.get('sub_tensor_backend', 'auto')).lower()
    if requested not in ['auto', 'dense', 'sparse']:
        raise ValueError('Invalid sub_tensor_backend: {}'.format(requested))
    if requested in ['auto', 'dense']:
        resolved = 'dense'
    elif requested == 'sparse':
        resolved = 'sparse'
    g['sub_tensor_backend'] = requested
    g['resolved_sub_tensor_backend'] = resolved
    txt = 'Substitution tensor backend: requested={}, resolved={}'
    print(txt.format(requested, resolved), flush=True)
    return resolved

def dense_to_sparse_sub_tensor(sub_tensor, tol=0):
    return substitution_sparse.dense_to_sparse_substitution_tensor(sub_tensor=sub_tensor, tol=tol)

def sparse_to_dense_sub_tensor(sparse_sub_tensor):
    return substitution_sparse.sparse_to_dense_substitution_tensor(sparse_tensor=sparse_sub_tensor)

def _is_sparse_sub_tensor(sub_tensor):
    return isinstance(sub_tensor, substitution_sparse.SparseSubstitutionTensor)

def estimate_sub_tensor_density(sub_tensor, tol=0):
    if _is_sparse_sub_tensor(sub_tensor):
        return sub_tensor.density
    arr = numpy.asarray(sub_tensor)
    if arr.size == 0:
        return 0.0
    if tol > 0:
        nnz = numpy.count_nonzero(numpy.abs(arr) > tol)
    else:
        nnz = numpy.count_nonzero(arr)
    return nnz / arr.size

def resolve_reducer_backend(g, sub_tensor=None, label=''):
    requested = str(g.get('sub_tensor_backend', 'auto')).lower()
    if requested not in ['auto', 'dense', 'sparse']:
        raise ValueError('Invalid sub_tensor_backend: {}'.format(requested))
    if requested in ['dense', 'sparse']:
        resolved = requested
    else:
        if sub_tensor is None:
            resolved = 'dense'
        else:
            tol = float(g.get('float_tol', 0))
            cutoff = float(g.get('sub_tensor_sparse_density_cutoff', 0.15))
            density = estimate_sub_tensor_density(sub_tensor=sub_tensor, tol=tol)
            resolved = 'sparse' if (density <= cutoff) else 'dense'
            txt = 'Auto-selected substitution reducer backend{}: density={:.6f}, cutoff={:.6f}, resolved={}'
            lbl = '' if label == '' else ' for {}'.format(label)
            print(txt.format(lbl, density, cutoff, resolved), flush=True)
    if 'resolved_reducer_backend' not in g.keys():
        g['resolved_reducer_backend'] = dict()
    if label != '':
        g['resolved_reducer_backend'][label] = resolved
    return resolved

def get_reducer_sub_tensor(sub_tensor, g, label=''):
    if _is_sparse_sub_tensor(sub_tensor):
        return sub_tensor
    resolved = resolve_reducer_backend(g=g, sub_tensor=sub_tensor, label=label)
    if resolved != 'sparse':
        return sub_tensor
    if 'reducer_sub_tensor_cache' not in g.keys():
        g['reducer_sub_tensor_cache'] = dict()
    if label in g['reducer_sub_tensor_cache']:
        return g['reducer_sub_tensor_cache'][label]
    tol = float(g.get('float_tol', 0))
    sparse_sub_tensor = dense_to_sparse_sub_tensor(sub_tensor=sub_tensor, tol=tol)
    txt = 'Converted substitution tensor{} to sparse: density={:.6f} ({:,}/{:,})'
    lbl = '' if label == '' else ' for {}'.format(label)
    print(txt.format(lbl, sparse_sub_tensor.density, sparse_sub_tensor.nnz, sparse_sub_tensor.size), flush=True)
    if label != '':
        g['reducer_sub_tensor_cache'][label] = sparse_sub_tensor
    return sparse_sub_tensor

def _get_sparse_group_block_index(sub_tensor):
    cache = getattr(sub_tensor, '_group_block_index', None)
    if cache is not None:
        return cache
    index = [list() for _ in range(sub_tensor.num_group)]
    for (group_id, a, d), mat in sub_tensor.blocks.items():
        index[int(group_id)].append((int(a), int(d), mat))
    cache = tuple(index)
    setattr(sub_tensor, '_group_block_index', cache)
    return cache


def _get_sparse_row_indices_and_data(mat, branch_id, row_cache=None):
    branch_id = int(branch_id)
    if row_cache is not None:
        key = (id(mat), branch_id)
        cached = row_cache.get(key)
        if cached is not None:
            return cached
    start = mat.indptr[branch_id]
    end = mat.indptr[branch_id + 1]
    indices = mat.indices[start:end]
    data = mat.data[start:end]
    if row_cache is not None:
        row_cache[key] = (indices, data)
    return indices, data


def _get_sparse_combination_group_tensor(sub_tensor, branch_ids, sg, data_type=None, group_block_index=None, row_cache=None):
    if data_type is None:
        data_type = sub_tensor.dtype
    arity = len(branch_ids)
    out = numpy.zeros(
        shape=(arity, sub_tensor.num_site, sub_tensor.num_state_from, sub_tensor.num_state_to),
        dtype=data_type,
    )
    if group_block_index is None:
        group_blocks = _get_sparse_group_block_index(sub_tensor)[int(sg)]
    else:
        group_blocks = group_block_index[int(sg)]
    for a, d, mat in group_blocks:
        for bi, bid in enumerate(branch_ids):
            indices, data = _get_sparse_row_indices_and_data(
                mat=mat,
                branch_id=bid,
                row_cache=row_cache,
            )
            if data.shape[0] == 0:
                continue
            out[bi, indices, a, d] = data
    return out


def _build_sparse_substitution_tensor(state_tensor, state_tensor_anc, mode, g):
    dtype = state_tensor.dtype
    num_branch = state_tensor.shape[0]
    num_site = state_tensor.shape[1]
    if mode == 'asis':
        num_syngroup = 1
        num_state = state_tensor.shape[2]
    elif mode == 'syn':
        num_syngroup = len(g['amino_acid_orders'])
        num_state = g['max_synonymous_size']
    else:
        raise ValueError('Unsupported mode for sparse substitution tensor: {}'.format(mode))
    sparse_entries = defaultdict(lambda: [list(), list(), list()])  # key -> [rows, cols, data]
    for node in g['tree'].traverse():
        if node.is_root():
            continue
        child = node.numerical_label
        parent = node.up.numerical_label
        if state_tensor_anc[parent, :, :].sum() < g['float_tol']:
            continue
        parent_matrix = state_tensor_anc[parent, :, :]
        child_matrix = state_tensor[child, :, :]
        if mode == 'asis':
            for a in range(num_state):
                parent_state = parent_matrix[:, a]
                if not numpy.any(parent_state):
                    continue
                for d in range(num_state):
                    if a == d:
                        continue
                    vals = parent_state * child_matrix[:, d]
                    nz = numpy.where(vals != 0)[0]
                    if nz.shape[0] == 0:
                        continue
                    rows, cols, data = sparse_entries[(0, a, d)]
                    rows.extend([child] * nz.shape[0])
                    cols.extend(nz.tolist())
                    data.extend(vals[nz].tolist())
        elif mode == 'syn':
            for sg, aa in enumerate(g['amino_acid_orders']):
                ind = numpy.array(g['synonymous_indices'][aa], dtype=numpy.int64)
                size = ind.shape[0]
                if size <= 1:
                    continue
                # Keep indexing semantics consistent with dense path: [state, site].
                parent_sub = state_tensor_anc[parent, :, ind]
                child_sub = state_tensor[child, :, ind]
                for a in range(size):
                    parent_state = parent_sub[a, :]
                    if not numpy.any(parent_state):
                        continue
                    for d in range(size):
                        if a == d:
                            continue
                        vals = parent_state * child_sub[d, :]
                        nz = numpy.where(vals != 0)[0]
                        if nz.shape[0] == 0:
                            continue
                        rows, cols, data = sparse_entries[(sg, a, d)]
                        rows.extend([child] * nz.shape[0])
                        cols.extend(nz.tolist())
                        data.extend(vals[nz].tolist())
    blocks = dict()
    shape = (num_branch, num_site)
    for key, (rows, cols, data) in sparse_entries.items():
        if len(data) == 0:
            continue
        mat = scipy.sparse.coo_matrix(
            (numpy.asarray(data, dtype=dtype), (numpy.asarray(rows), numpy.asarray(cols))),
            shape=shape,
            dtype=dtype,
        ).tocsr()
        mat.sum_duplicates()
        mat.eliminate_zeros()
        if mat.nnz > 0:
            blocks[key] = mat
    tensor_shape = (num_branch, num_site, num_syngroup, num_state, num_state)
    out = substitution_sparse.SparseSubstitutionTensor(shape=tensor_shape, dtype=dtype, blocks=blocks)
    txt = 'Generated sparse substitution tensor: shape={}, density={:.6f} ({:,}/{:,})'
    print(txt.format(out.shape, out.density, out.nnz, out.size), flush=True)
    return out


def get_branch_sub_counts(sub_tensor):
    if _is_sparse_sub_tensor(sub_tensor):
        out = numpy.zeros(shape=(sub_tensor.num_branch,), dtype=numpy.float64)
        for mat in sub_tensor.blocks.values():
            out += numpy.asarray(mat.sum(axis=1)).reshape(-1)
        return out
    return sub_tensor.sum(axis=(1, 2, 3, 4))


def get_site_sub_counts(sub_tensor):
    if _is_sparse_sub_tensor(sub_tensor):
        out = numpy.zeros(shape=(sub_tensor.num_site,), dtype=numpy.float64)
        for mat in sub_tensor.blocks.values():
            out += numpy.asarray(mat.sum(axis=0)).reshape(-1)
        return out
    return sub_tensor.sum(axis=(0, 2, 3, 4))


def get_branch_site_sub_counts(sub_tensor, branch_id):
    if _is_sparse_sub_tensor(sub_tensor):
        out = numpy.zeros(shape=(sub_tensor.num_site,), dtype=numpy.float64)
        for mat in sub_tensor.blocks.values():
            row = mat.getrow(int(branch_id))
            if row.nnz == 0:
                continue
            out[row.indices] += row.data
        return out
    return sub_tensor[branch_id, :, :, :, :].sum(axis=(1, 2, 3))


def get_total_substitution(sub_tensor):
    if _is_sparse_sub_tensor(sub_tensor):
        return float(sum([mat.data.sum() for mat in sub_tensor.blocks.values()]))
    return float(sub_tensor.sum())


def get_group_state_totals(sub_tensor):
    if _is_sparse_sub_tensor(sub_tensor):
        gad = numpy.zeros(
            shape=(sub_tensor.num_group, sub_tensor.num_state_from, sub_tensor.num_state_to),
            dtype=numpy.float64,
        )
        for (sg, a, d), mat in sub_tensor.blocks.items():
            gad[sg, a, d] += float(mat.data.sum())
    else:
        gad = sub_tensor.sum(axis=(0, 1))
    ga = gad.sum(axis=2)
    gd = gad.sum(axis=1)
    return gad, ga, gd


def _get_sparse_branch_tensor(sub_tensor, branch_id):
    out = numpy.zeros(
        shape=(sub_tensor.num_site, sub_tensor.num_group, sub_tensor.num_state_from, sub_tensor.num_state_to),
        dtype=sub_tensor.dtype,
    )
    for (sg, a, d), mat in sub_tensor.blocks.items():
        row = mat.getrow(int(branch_id))
        if row.nnz == 0:
            continue
        out[row.indices, sg, a, d] = row.data
    return out


def _get_sparse_site_vectors(sub_tensor, branch_ids, data_type=numpy.float64, group_block_index=None, row_cache=None):
    num_site = sub_tensor.num_site
    any2any = numpy.zeros(shape=(num_site,), dtype=data_type)
    spe2any = numpy.zeros(shape=(num_site,), dtype=data_type)
    any2spe = numpy.zeros(shape=(num_site,), dtype=data_type)
    spe2spe = numpy.zeros(shape=(num_site,), dtype=data_type)
    for sg in range(sub_tensor.num_group):
        sub_sg = _get_sparse_combination_group_tensor(
            sub_tensor=sub_tensor,
            branch_ids=branch_ids,
            sg=int(sg),
            data_type=data_type,
            group_block_index=group_block_index,
            row_cache=row_cache,
        )
        any2any += sub_sg.sum(axis=(2, 3)).prod(axis=0)
        spe2any += sub_sg.sum(axis=3).prod(axis=0).sum(axis=1)
        any2spe += sub_sg.sum(axis=2).prod(axis=0).sum(axis=1)
        spe2spe += sub_sg.prod(axis=0).sum(axis=(1, 2))
    return any2any, spe2any, any2spe, spe2spe


def get_cs_sparse(id_combinations, sub_tensor, attr):
    num_site = sub_tensor.shape[1]
    df = numpy.zeros([num_site, 5], dtype=numpy.float64)
    df[:, 0] = numpy.arange(num_site)
    group_block_index = _get_sparse_group_block_index(sub_tensor)
    row_cache = dict()
    for i in numpy.arange(id_combinations.shape[0]):
        any2any, spe2any, any2spe, spe2spe = _get_sparse_site_vectors(
            sub_tensor=sub_tensor,
            branch_ids=id_combinations[i, :],
            data_type=numpy.float64,
            group_block_index=group_block_index,
            row_cache=row_cache,
        )
        df[:, 1] += any2any
        df[:, 2] += spe2any
        df[:, 3] += any2spe
        df[:, 4] += spe2spe
    cn = ['site',] + [ 'OC'+attr+subs for subs in ["any2any","spe2any","any2spe","spe2spe"] ]
    df = pandas.DataFrame(df, columns=cn)
    df = table.set_substitution_dtype(df=df)
    return df

def initialize_substitution_tensor(state_tensor, mode, g, mmap_attr, dtype=None):
    if dtype is None:
        dtype = state_tensor.dtype
    num_branch = state_tensor.shape[0]
    num_site = state_tensor.shape[1]
    if mode=='asis':
        num_syngroup = 1
        num_state = state_tensor.shape[2]
    elif mode=='syn':
        num_syngroup = len(g['amino_acid_orders'])
        num_state = g['max_synonymous_size']
    axis = (num_branch,num_site,num_syngroup,num_state,num_state) # axis = [branch,site,matrix_group,state_from,state_to]
    mmap_tensor = os.path.join(os.getcwd(), 'tmp.csubst.sub_tensor.'+mmap_attr+'.mmap')
    if os.path.exists(mmap_tensor): os.unlink(mmap_tensor)
    txt = 'Generating memory map: dtype={}, axis={}, path={}'
    print(txt.format(state_tensor.dtype, axis, mmap_tensor), flush=True)
    sub_tensor = numpy.memmap(mmap_tensor, dtype=dtype, shape=axis, mode='w+')
    return sub_tensor

def get_substitution_tensor(state_tensor, state_tensor_anc=None, mode='', g={}, mmap_attr=''):
    backend = resolve_sub_tensor_backend(g)
    if state_tensor_anc is None:
        state_tensor_anc = state_tensor
    if backend == 'sparse':
        return _build_sparse_substitution_tensor(
            state_tensor=state_tensor,
            state_tensor_anc=state_tensor_anc,
            mode=mode,
            g=g,
        )
    sub_tensor = initialize_substitution_tensor(state_tensor, mode, g, mmap_attr)
    if (g['ml_anc']=='no'):
        sub_tensor[:,:,:,:,:] = numpy.nan
    if mode=='asis':
        num_state = state_tensor.shape[2]
        diag_zero = numpy.diag([-1] * num_state) + 1
    for node in g['tree'].traverse():
        if node.is_root():
            continue
        child = node.numerical_label
        parent = node.up.numerical_label
        if state_tensor_anc[parent, :, :].sum()<g['float_tol']:
            continue
        if mode=='asis':
            sub_matrix = numpy.einsum("sa,sd,ad->sad", state_tensor_anc[parent,:,:], state_tensor[child,:,:], diag_zero)
            # s=site, a=ancestral, d=derived
            sub_tensor[child, :, 0, :, :] = sub_matrix
        elif mode=='syn':
            for s,aa in enumerate(g['amino_acid_orders']):
                ind = numpy.array(g['synonymous_indices'][aa])
                size = len(ind)
                diag_zero = numpy.diag([-1] * size) + 1
                parent_matrix = state_tensor_anc[parent, :, ind] # axis is swapped, shape=[state,site]
                child_matrix = state_tensor[child, :, ind] # axis is swapped, shape=[state,site]
                sub_matrix = numpy.einsum("as,ds,ad->sad", parent_matrix, child_matrix, diag_zero)
                sub_tensor[child, :, s, :size, :size] = sub_matrix
    if numpy.isnan(sub_tensor).any():
        sub_tensor = numpy.nan_to_num(sub_tensor, nan=0, copy=False)
    return sub_tensor

def apply_min_sub_pp(g, sub_tensor):
    if g['min_sub_pp']==0:
        return sub_tensor
    if (g['ml_anc']):
        print('--ml_anc is set. --min_sub_pp will not be applied.')
    else:
        if _is_sparse_sub_tensor(sub_tensor):
            threshold = g['min_sub_pp']
            new_blocks = dict()
            for key, mat in sub_tensor.blocks.items():
                new_mat = mat.copy()
                is_small = (new_mat.data < threshold)
                if is_small.any():
                    new_mat.data[is_small] = 0
                    new_mat.eliminate_zeros()
                if new_mat.nnz > 0:
                    new_blocks[key] = new_mat
            sub_tensor = substitution_sparse.SparseSubstitutionTensor(
                shape=sub_tensor.shape,
                dtype=sub_tensor.dtype,
                blocks=new_blocks,
            )
        else:
            sub_tensor[(sub_tensor<g['min_sub_pp'])] = 0
    return sub_tensor

def get_b(g, sub_tensor, attr, sitewise, min_sitewise_pp=0.5):
    column_names=['branch_name', 'branch_id', attr+'_sub']
    df = pandas.DataFrame(numpy.nan, index=range(0, g['num_node']), columns=column_names)
    df['branch_name'] = df['branch_name'].astype(str)
    if sitewise:
        df[attr + '_sitewise'] = ''
    branch_sub_counts = get_branch_sub_counts(sub_tensor) if _is_sparse_sub_tensor(sub_tensor) else None
    i=0
    for node in g['tree'].traverse():
        df.at[i,'branch_name'] = getattr(node, 'name')
        df.at[i,'branch_id'] = getattr(node, 'numerical_label')
        if _is_sparse_sub_tensor(sub_tensor):
            df.at[i,attr+'_sub'] = branch_sub_counts[node.numerical_label]
            branch_tensor = _get_sparse_branch_tensor(sub_tensor=sub_tensor, branch_id=node.numerical_label) if sitewise else None
        else:
            df.at[i,attr+'_sub'] = sub_tensor[node.numerical_label,:,:,:,:].sum()
            branch_tensor = sub_tensor[node.numerical_label, :, :, :, :] if sitewise else None
        if sitewise:
            sub_list = list()
            if attr=='N':
                state_order = g['amino_acid_orders']
            elif attr=='S':
                raise Exception('This function is not supported for synonymous substitutions.')
            for s in range(sub_tensor.shape[1]):
                max_value = branch_tensor[s, :, :, :].max()
                if max_value < min_sitewise_pp:
                    continue
                max_idx = numpy.where(branch_tensor[s, :, :, :]==max_value)
                ancestral_state = state_order[max_idx[1][0]]
                derived_state = state_order[max_idx[2][0]]
                sub_string = ancestral_state+str(s+1)+derived_state
                sub_list.append(sub_string)
            df.at[i, attr + '_sitewise'] = ','.join(sub_list)
        i+=1
    df = df.dropna(axis=0)
    df['branch_id'] = df['branch_id'].astype(int)
    df = df.sort_values(by='branch_id')
    df = table.set_substitution_dtype(df=df)
    return(df)

def get_s(sub_tensor, attr):
    column_names=['site',attr+'_sub']
    num_site = sub_tensor.shape[1]
    df = pandas.DataFrame(0, index=numpy.arange(0,num_site), columns=column_names)
    df['site'] = numpy.arange(0, num_site)
    df[attr+'_sub'] = get_site_sub_counts(sub_tensor)
    df['site'] = df['site'].astype(int)
    df = df.sort_values(by='site')
    df = table.set_substitution_dtype(df=df)
    return(df)

def get_cs(id_combinations, sub_tensor, attr):
    if _is_sparse_sub_tensor(sub_tensor):
        return get_cs_sparse(id_combinations=id_combinations, sub_tensor=sub_tensor, attr=attr)
    num_site = sub_tensor.shape[1]
    df = numpy.zeros([num_site, 5])
    df[:, 0] = numpy.arange(num_site)
    for i in numpy.arange(id_combinations.shape[0]):
        for sg in numpy.arange(sub_tensor.shape[2]): # Couldn't this sg included in the matrix calc using .sum()?
            df[:, 1] += numpy.nan_to_num(sub_tensor[id_combinations[i,:], :, sg, :, :].sum(axis=(2, 3)).prod(axis=0))  # any2any
            df[:, 2] += numpy.nan_to_num(sub_tensor[id_combinations[i,:], :, sg, :, :].sum(axis=3).prod(axis=0).sum(axis=1))  # spe2any
            df[:, 3] += numpy.nan_to_num(sub_tensor[id_combinations[i,:], :, sg, :, :].sum(axis=2).prod(axis=0).sum(axis=1))  # any2spe
            df[:, 4] += numpy.nan_to_num(sub_tensor[id_combinations[i,:], :, sg, :, :].prod(axis=0).sum(axis=(1, 2)))  # spe2spe
    cn = ['site',] + [ 'OC'+attr+subs for subs in ["any2any","spe2any","any2spe","spe2spe"] ]
    df = pandas.DataFrame(df, columns=cn)
    df = table.set_substitution_dtype(df=df)
    return (df)

def get_bs(S_tensor, N_tensor):
    num_site = S_tensor.shape[1]
    num_branch = S_tensor.shape[0]
    column_names=['branch_id','site','S_sub','N_sub']
    df = pandas.DataFrame(numpy.nan, index=numpy.arange(0,num_branch*num_site), columns=column_names)
    for i in numpy.arange(num_branch):
        ind = numpy.arange(i*num_site, (i+1)*num_site)
        df.loc[ind, 'site'] = numpy.arange(0, num_site)
        df.loc[ind, 'branch_id'] = i
        df.loc[ind, 'S_sub'] = get_branch_site_sub_counts(S_tensor, i)
        df.loc[ind, 'N_sub'] = get_branch_site_sub_counts(N_tensor, i)
    df = table.set_substitution_dtype(df=df)
    return(df)

def _can_use_cython_dense_cb(id_combinations, sub_tensor, mmap=False, df_mmap=None, float_type=numpy.float64):
    if _is_sparse_sub_tensor(sub_tensor):
        return False
    if id_combinations.shape[1] != 2:
        return False
    if (id_combinations.dtype.kind not in ['i', 'u']):
        return False
    if (sub_tensor.dtype != numpy.float64):
        return False
    if (float_type != numpy.float64):
        return False
    if mmap and ((df_mmap is None) or (df_mmap.dtype != numpy.float64)):
        return False
    return hasattr(substitution_cy, 'calc_combinatorial_sub_double_arity2')

def _can_use_cython_dense_cbs(id_combinations, sub_tensor, mmap=False, df_mmap=None):
    if _is_sparse_sub_tensor(sub_tensor):
        return False
    if id_combinations.shape[1] != 2:
        return False
    if (id_combinations.dtype.kind not in ['i', 'u']):
        return False
    if (sub_tensor.dtype != numpy.float64):
        return False
    if mmap and ((df_mmap is None) or (df_mmap.dtype != numpy.float64)):
        return False
    return hasattr(substitution_cy, 'calc_combinatorial_sub_by_site_double_arity2')

def _resolve_dense_cython_n_jobs(n_jobs, id_combinations, sub_tensor, g, task='cb'):
    if n_jobs <= 1:
        return 1
    min_combos_per_job = int(g.get('parallel_dense_cython_min_combos_per_job', 5000))
    min_ops_per_job = int(g.get('parallel_dense_cython_min_ops_per_job', 500000000))
    if min_combos_per_job < 1:
        min_combos_per_job = 1
    if min_ops_per_job < 1:
        min_ops_per_job = 1
    num_comb = int(id_combinations.shape[0])
    ops_per_comb = int(sub_tensor.shape[1]) * int(sub_tensor.shape[2]) * int(sub_tensor.shape[3]) * int(sub_tensor.shape[4])
    total_ops = num_comb * ops_per_comb
    max_jobs_by_combos = max(1, num_comb // min_combos_per_job)
    max_jobs_by_ops = max(1, total_ops // min_ops_per_job)
    resolved = max(1, min(int(n_jobs), max_jobs_by_combos, max_jobs_by_ops))
    txt = 'Dense Cython scheduler for {}: combos={}, estimated_ops={}, workers {} -> {}'
    print(txt.format(task, num_comb, total_ops, n_jobs, resolved), flush=True)
    return resolved

def sub_tensor2cb(id_combinations, sub_tensor, mmap=False, df_mmap=None, mmap_start=0, float_type=numpy.float64):
    if _can_use_cython_dense_cb(
        id_combinations=id_combinations,
        sub_tensor=sub_tensor,
        mmap=mmap,
        df_mmap=df_mmap,
        float_type=float_type,
    ):
        try:
            out = substitution_cy.calc_combinatorial_sub_double_arity2(
                id_combinations=numpy.asarray(id_combinations, dtype=numpy.int64),
                mmap_start=mmap_start,
                sub_tensor=sub_tensor,
                mmap=mmap,
                df_mmap=df_mmap,
            )
            if not mmap:
                return out
            return
        except Exception:
            pass
    arity = id_combinations.shape[1]
    if mmap:
        df = df_mmap
    else:
        if (sub_tensor.dtype == bool):
            data_type = numpy.int32
        else:
            data_type = float_type
        df = numpy.zeros([id_combinations.shape[0], arity + 4], dtype=data_type)
    start_time = time.time()
    start = mmap_start
    end = mmap_start + id_combinations.shape[0]
    df[start:end, :arity] = id_combinations[:, :]  # branch_ids
    for i,j in zip(numpy.arange(start, end),numpy.arange(id_combinations.shape[0])):
        sub_combo = sub_tensor[id_combinations[j, :], :, :, :, :]
        sum_any2any = sub_combo.sum(axis=(3, 4))
        sum_spe2any = sub_combo.sum(axis=4)
        sum_any2spe = sub_combo.sum(axis=3)
        prod_spe2spe = sub_combo.prod(axis=0)
        df[i, arity+0] += sum_any2any.prod(axis=0).sum() # any2any
        df[i, arity+1] += sum_spe2any.prod(axis=0).sum() # spe2any
        df[i, arity+2] += sum_any2spe.prod(axis=0).sum() # any2spe
        df[i, arity+3] += prod_spe2spe.sum() # spe2spe
        if j % 10000 == 0:
            mmap_end = mmap_start + id_combinations.shape[0]
            txt = 'cb: {:,}th in the id range {:,}-{:,}: {:,} sec'
            print(txt.format(j, mmap_start, mmap_end, int(time.time() - start_time)), flush=True)
    if not mmap:
        return (df)

def sub_tensor2cb_sparse(id_combinations, sub_tensor, mmap=False, df_mmap=None, mmap_start=0, float_type=numpy.float64):
    arity = id_combinations.shape[1]
    if mmap:
        df = df_mmap
    else:
        if (sub_tensor.dtype == bool):
            data_type = numpy.int32
        else:
            data_type = float_type
        df = numpy.zeros([id_combinations.shape[0], arity + 4], dtype=data_type)
    start_time = time.time()
    start = mmap_start
    end = mmap_start + id_combinations.shape[0]
    df[start:end, :arity] = id_combinations[:, :]  # branch_ids
    group_block_index = _get_sparse_group_block_index(sub_tensor)
    row_cache = dict()
    for i,j in zip(numpy.arange(start, end),numpy.arange(id_combinations.shape[0])):
        any2any, spe2any, any2spe, spe2spe = _get_sparse_site_vectors(
            sub_tensor=sub_tensor,
            branch_ids=id_combinations[j, :],
            data_type=df.dtype,
            group_block_index=group_block_index,
            row_cache=row_cache,
        )
        df[i, arity+0] += any2any.sum()
        df[i, arity+1] += spe2any.sum()
        df[i, arity+2] += any2spe.sum()
        df[i, arity+3] += spe2spe.sum()
        if j % 10000 == 0:
            mmap_end = mmap_start + id_combinations.shape[0]
            txt = 'cb(sparse): {:,}th in the id range {:,}-{:,}: {:,} sec'
            print(txt.format(j, mmap_start, mmap_end, int(time.time() - start_time)), flush=True)
    if not mmap:
        return (df)

def _write_cb_chunk_to_mmap(writer, ids, sub_tensor, mmap_out, axis, dtype, mmap_start, float_type):
    df_mmap = numpy.memmap(mmap_out, dtype=dtype, shape=axis, mode='r+')
    writer(ids, sub_tensor, True, df_mmap, mmap_start, float_type)
    df_mmap.flush()

def get_cb(id_combinations, sub_tensor, g, attr):
    sub_tensor = get_reducer_sub_tensor(sub_tensor=sub_tensor, g=g, label='cb_'+attr)
    arity = id_combinations.shape[1]
    cn = [ "branch_id_" + str(num+1) for num in range(0,arity) ]
    cn = cn + [ attr+subs for subs in ["any2any","spe2any","any2spe","spe2spe"] ]
    writer = sub_tensor2cb_sparse if _is_sparse_sub_tensor(sub_tensor) else sub_tensor2cb
    n_jobs = parallel.resolve_n_jobs(num_items=id_combinations.shape[0], threads=g['threads'])
    if (writer is sub_tensor2cb) and _can_use_cython_dense_cb(
        id_combinations=id_combinations,
        sub_tensor=sub_tensor,
        mmap=False,
        df_mmap=None,
        float_type=g['float_type'],
    ):
        n_jobs = _resolve_dense_cython_n_jobs(
            n_jobs=n_jobs,
            id_combinations=id_combinations,
            sub_tensor=sub_tensor,
            g=g,
            task='cb',
        )
    if n_jobs == 1:
        df = writer(id_combinations, sub_tensor, mmap=False, df_mmap=None, mmap_start=0, float_type=g['float_type'])
        df = pandas.DataFrame(df, columns=cn)
    else:
        chunk_factor = parallel.resolve_chunk_factor(g=g, task='reducer')
        id_chunks,mmap_starts = parallel.get_chunks(id_combinations, n_jobs, chunk_factor=chunk_factor)
        mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.cb.out.mmap')
        if os.path.exists(mmap_out): os.unlink(mmap_out)
        axis = (id_combinations.shape[0], arity+4)
        my_dtype = sub_tensor.dtype
        if 'bool' in str(my_dtype):
            my_dtype = numpy.int32
        df_mmap = numpy.memmap(mmap_out, dtype=my_dtype, shape=axis, mode='w+')
        backend = parallel.resolve_joblib_backend(g=g, task='reducer')
        tasks = [
            (writer, ids, sub_tensor, mmap_out, axis, my_dtype, ms, g['float_type'])
            for ids, ms in zip(id_chunks, mmap_starts)
        ]
        parallel.run_starmap(
            func=_write_cb_chunk_to_mmap,
            args_iterable=tasks,
            n_jobs=n_jobs,
            backend=backend,
        )
        df_mmap.flush()
        df = pandas.DataFrame(df_mmap, columns=cn)
        del df_mmap
        if os.path.exists(mmap_out): os.unlink(mmap_out)
    df = table.sort_branch_ids(df)
    df = df.dropna()
    if not attr.startswith('EC'):
        df = table.set_substitution_dtype(df=df)
    return df

def sub_tensor2cbs(id_combinations, sub_tensor, mmap=False, df_mmap=None, mmap_start=0):
    if _can_use_cython_dense_cbs(
        id_combinations=id_combinations,
        sub_tensor=sub_tensor,
        mmap=mmap,
        df_mmap=df_mmap,
    ):
        try:
            out = substitution_cy.calc_combinatorial_sub_by_site_double_arity2(
                id_combinations=numpy.asarray(id_combinations, dtype=numpy.int64),
                mmap_start=mmap_start,
                sub_tensor=sub_tensor,
                mmap=mmap,
                df_mmap=df_mmap,
            )
            if not mmap:
                return out
            return
        except Exception:
            pass
    arity = id_combinations.shape[1]
    num_site = sub_tensor.shape[1]
    sites = numpy.arange(num_site)
    if mmap:
        df = df_mmap
    else:
        shape = (int(id_combinations.shape[0]*num_site), arity+5)
        my_dtype = sub_tensor.dtype
        if 'bool' in str(my_dtype):
            my_dtype = numpy.int32
        df = numpy.zeros(shape=shape, dtype=my_dtype)
    node=0
    start_time = time.time()
    for i in numpy.arange(id_combinations.shape[0]):
        row_start = (node*num_site)+(mmap_start*num_site)
        row_end = ((node+1)*num_site)+(mmap_start*num_site)
        df[row_start:row_end,:arity] = id_combinations[node,:] # branch_ids
        df[row_start:row_end,arity] = sites # site
        ic = id_combinations[i,:]
        sub_combo = sub_tensor[ic, :, :, :, :]
        sum_any2any = sub_combo.sum(axis=(3, 4))
        sum_spe2any = sub_combo.sum(axis=4)
        sum_any2spe = sub_combo.sum(axis=3)
        prod_spe2spe = sub_combo.prod(axis=0)
        df[row_start:row_end,arity+1] += sum_any2any.prod(axis=0).sum(axis=1) #any2any
        df[row_start:row_end,arity+2] += sum_spe2any.prod(axis=0).sum(axis=(1,2)) #spe2any
        df[row_start:row_end,arity+3] += sum_any2spe.prod(axis=0).sum(axis=(1,2)) #any2spe
        df[row_start:row_end,arity+4] += prod_spe2spe.sum(axis=(1,2,3)) #spe2spe
        if (node%10000==0):
            mmap_start = mmap_start
            mmap_end = mmap_start+id_combinations.shape[0]
            txt = 'cbs: {:,}th in the id range {:,}-{:,}: {:,} sec'
            print(txt.format(node, mmap_start, mmap_end, int(time.time() - start_time)), flush=True)
        node += 1
    if not mmap:
        return df

def sub_tensor2cbs_sparse(id_combinations, sub_tensor, mmap=False, df_mmap=None, mmap_start=0):
    arity = id_combinations.shape[1]
    num_site = sub_tensor.shape[1]
    sites = numpy.arange(num_site)
    if mmap:
        df = df_mmap
    else:
        shape = (int(id_combinations.shape[0]*num_site), arity+5)
        my_dtype = sub_tensor.dtype
        if 'bool' in str(my_dtype):
            my_dtype = numpy.int32
        df = numpy.zeros(shape=shape, dtype=my_dtype)
    node = 0
    start_time = time.time()
    group_block_index = _get_sparse_group_block_index(sub_tensor)
    row_cache = dict()
    for i in numpy.arange(id_combinations.shape[0]):
        row_start = (node*num_site)+(mmap_start*num_site)
        row_end = ((node+1)*num_site)+(mmap_start*num_site)
        df[row_start:row_end,:arity] = id_combinations[node,:] # branch_ids
        df[row_start:row_end,arity] = sites # site
        any2any, spe2any, any2spe, spe2spe = _get_sparse_site_vectors(
            sub_tensor=sub_tensor,
            branch_ids=id_combinations[i, :],
            data_type=df.dtype,
            group_block_index=group_block_index,
            row_cache=row_cache,
        )
        df[row_start:row_end,arity+1] += any2any
        df[row_start:row_end,arity+2] += spe2any
        df[row_start:row_end,arity+3] += any2spe
        df[row_start:row_end,arity+4] += spe2spe
        if (node%10000==0):
            mmap_end = mmap_start+id_combinations.shape[0]
            txt = 'cbs(sparse): {:,}th in the id range {:,}-{:,}: {:,} sec'
            print(txt.format(node, mmap_start, mmap_end, int(time.time() - start_time)), flush=True)
        node += 1
    if not mmap:
        return df

def _write_cbs_chunk_to_mmap(writer, ids, sub_tensor, mmap_out, axis, dtype, mmap_start):
    df_mmap = numpy.memmap(mmap_out, dtype=dtype, shape=axis, mode='r+')
    writer(ids, sub_tensor, True, df_mmap, mmap_start)
    df_mmap.flush()

def get_cbs(id_combinations, sub_tensor, attr, g):
    sub_tensor = get_reducer_sub_tensor(sub_tensor=sub_tensor, g=g, label='cbs_'+attr)
    print("Calculating combinatorial substitutions: attr =", attr, flush=True)
    arity = id_combinations.shape[1]
    cn1 = [ "branch_id_" + str(num+1) for num in range(0,arity) ]
    cn2 = ["site",]
    cn3 = [ 'OC'+attr+subs for subs in ["any2any","spe2any","any2spe","spe2spe"] ]
    writer = sub_tensor2cbs_sparse if _is_sparse_sub_tensor(sub_tensor) else sub_tensor2cbs
    n_jobs = parallel.resolve_n_jobs(num_items=id_combinations.shape[0], threads=g['threads'])
    if (writer is sub_tensor2cbs) and _can_use_cython_dense_cbs(
        id_combinations=id_combinations,
        sub_tensor=sub_tensor,
        mmap=False,
        df_mmap=None,
    ):
        n_jobs = _resolve_dense_cython_n_jobs(
            n_jobs=n_jobs,
            id_combinations=id_combinations,
            sub_tensor=sub_tensor,
            g=g,
            task='cbs',
        )
    if n_jobs == 1:
        df = writer(id_combinations, sub_tensor)
        df = pandas.DataFrame(df, columns=cn1 + cn2 + cn3)
    else:
        chunk_factor = parallel.resolve_chunk_factor(g=g, task='reducer')
        id_chunks,mmap_starts = parallel.get_chunks(id_combinations, n_jobs, chunk_factor=chunk_factor)
        mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.cbs.out.mmap')
        if os.path.exists(mmap_out): os.remove(mmap_out)
        axis = (id_combinations.shape[0]*sub_tensor.shape[1], arity+5)
        my_dtype = sub_tensor.dtype
        if 'bool' in str(my_dtype):
            my_dtype = numpy.int32
        df_mmap = numpy.memmap(mmap_out, dtype=my_dtype, shape=axis, mode='w+')
        backend = parallel.resolve_joblib_backend(g=g, task='reducer')
        tasks = [
            (writer, ids, sub_tensor, mmap_out, axis, my_dtype, ms)
            for ids, ms in zip(id_chunks, mmap_starts)
        ]
        parallel.run_starmap(
            func=_write_cbs_chunk_to_mmap,
            args_iterable=tasks,
            n_jobs=n_jobs,
            backend=backend,
        )
        df_mmap.flush()
        df = pandas.DataFrame(df_mmap, columns=cn1 + cn2 + cn3)
        del df_mmap
        if os.path.exists(mmap_out): os.remove(mmap_out)
    df = df.dropna()
    df = table.sort_branch_ids(df)
    df = table.set_substitution_dtype(df=df)
    return(df)

def get_sub_sites(g, sS, sN, state_tensor):
    num_site = sS.shape[0]
    num_branch = len(list(g['tree'].traverse()))
    g['is_site_nonmissing'] = numpy.zeros(shape=[num_branch, num_site], dtype=bool)
    for node in g['tree'].traverse():
        nl = node.numerical_label
        g['is_site_nonmissing'][nl,:] = (state_tensor[nl,:,:].sum(axis=1)!=0)
    g['sub_sites'] = dict()
    g['sub_sites'][g['asrv']] = numpy.zeros(shape=[num_branch, num_site], dtype=g['float_type'])
    if (g['asrv']=='no'):
        sub_sites = numpy.ones(shape=[num_site,]) / num_site
    elif (g['asrv']=='pool'):
        sub_sites = sS['S_sub'].values + sN['N_sub'].values
    elif (g['asrv']=='file'):
        sub_sites = g['iqtree_rate_values']
    if (g['asrv']=='sn'):
        for SN,df in zip(['S','N'],[sS,sN]):
            g['sub_sites'][SN] = numpy.zeros(shape=[num_branch, num_site], dtype=g['float_type'])
            sub_sites = df[SN+'_sub'].values
            for node in g['tree'].traverse():
                nl = node.numerical_label
                adjusted_sub_sites = sub_sites * g['is_site_nonmissing'][nl,:]
                total_sub_sites = adjusted_sub_sites.sum()
                total_sub_sites = 1 if (total_sub_sites==0) else total_sub_sites
                adjusted_sub_sites = adjusted_sub_sites/total_sub_sites
                g['sub_sites'][SN][nl,:] = adjusted_sub_sites
    elif (g['asrv']!='each'): # if 'each', Defined later in get_each_sub_sites()
        for node in g['tree'].traverse():
            nl = node.numerical_label
            is_site_nonmissing = (state_tensor[nl,:,:].sum(axis=1)!=0)
            adjusted_sub_sites = sub_sites * is_site_nonmissing
            total_sub_sites = adjusted_sub_sites.sum()
            total_sub_sites = 1 if (total_sub_sites==0) else total_sub_sites
            adjusted_sub_sites = adjusted_sub_sites/total_sub_sites
            g['sub_sites'][g['asrv']][nl,:] = adjusted_sub_sites
    return g

def get_each_sub_sites(sub_sg, mode, sg, a, d, g): # sub_sites for each "sg" group
    sub_sites = numpy.zeros(shape=g['is_site_nonmissing'].shape, dtype=g['float_type'])
    if mode == 'spe2spe':
        nonadjusted_sub_sites = sub_sg[:, sg, a, d]
    elif mode == 'spe2any':
        nonadjusted_sub_sites = sub_sg[:, sg, a]
    elif mode == 'any2spe':
        nonadjusted_sub_sites = sub_sg[:, sg, d]
    elif mode == 'any2any':
        nonadjusted_sub_sites = sub_sg[:, sg]
    for node in g['tree'].traverse():
        nl = node.numerical_label
        sub_sites[nl,:] = nonadjusted_sub_sites * g['is_site_nonmissing'][nl,:]
        total_sub_sites = sub_sites[nl,:].sum()
        total_sub_sites = 1 if (total_sub_sites==0) else total_sub_sites
        sub_sites[nl,:] = sub_sites[nl,:] / total_sub_sites
    return sub_sites

def get_sub_branches(sub_bg, mode, sg, a, d):
    if mode == 'spe2spe':
        sub_branches = sub_bg[:, sg, a, d]
    elif mode == 'spe2any':
        sub_branches = sub_bg[:, sg, a]
    elif mode == 'any2spe':
        sub_branches = sub_bg[:, sg, d]
    elif mode == 'any2any':
        sub_branches = sub_bg[:, sg]
    return sub_branches

def get_substitutions_per_branch(cb, b, g):
    for a in numpy.arange(g['current_arity']):
        b_tmp = b.loc[:,['branch_id','S_sub','N_sub']]
        b_tmp.columns = [ c+'_'+str(a+1) for c in b_tmp.columns ]
        cb = pandas.merge(cb, b_tmp, on='branch_id_'+str(a+1), how='left')
        del b_tmp
    return(cb)

def add_dif_column(cb, col_dif, col_any, col_spe, tol):
    if ((col_any in cb.columns) & (col_spe in cb.columns)):
        cb.loc[:, col_dif] = cb[col_any] - cb[col_spe]
        is_negative = (cb[col_dif] < -tol)
        is_almost_zero = (~is_negative)&(cb[col_dif] < tol)
        cb.loc[is_negative, col_dif] = numpy.nan
        cb.loc[is_almost_zero, col_dif] = 0
    return cb

def add_dif_stats(cb, tol, prefix):
    for SN in ['S','N']:
        for anc in ['any','spe']:
            col_any = prefix+SN+anc+'2any'
            col_spe = prefix+SN+anc+'2spe'
            col_dif = prefix+SN+anc+'2dif'
            cb = add_dif_column(cb, col_dif, col_any, col_spe, tol)
        for des in ['any','spe','dif']:
            col_any = prefix+SN+'any2'+des
            col_spe = prefix+SN+'spe2'+des
            col_dif = prefix+SN+'dif2'+des
            cb = add_dif_column(cb, col_dif, col_any, col_spe, tol)
    return cb
