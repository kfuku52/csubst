import numpy as np
import pandas as pd
import scipy.sparse as sp

import os
import time
import warnings
import inspect
from collections import defaultdict

from csubst import table
from csubst import parallel
from csubst import substitution_cy
from csubst import substitution_sparse
try:
    from csubst import substitution_sparse_cy
except Exception:  # pragma: no cover - Cython extension is optional
    substitution_sparse_cy = None
from csubst import ete
from csubst import output_stat

_CB_BASE_SUBSTITUTIONS = ("any2any", "spe2any", "any2spe", "spe2spe")
_SUB_TENSOR_BACKENDS = ('auto', 'dense', 'sparse')
_CYTHON_FALLBACK_WARNED = set()


def _warn_cython_fallback(fastpath_name, exc):
    if fastpath_name in _CYTHON_FALLBACK_WARNED:
        return
    _CYTHON_FALLBACK_WARNED.add(fastpath_name)
    txt = 'Cython fast path "{}" failed ({}: {}). Falling back to Python implementation.'
    warnings.warn(txt.format(fastpath_name, type(exc).__name__, exc), RuntimeWarning, stacklevel=2)


def _resolve_cb_base_substitutions(selected_base_stats=None):
    if selected_base_stats is None:
        return list(_CB_BASE_SUBSTITUTIONS)
    if isinstance(selected_base_stats, str):
        selected_base_stats = [s for s in selected_base_stats.split(',')]
    tokens = [str(s).strip().lower() for s in selected_base_stats if str(s).strip() != ""]
    if len(tokens) == 0:
        raise ValueError("At least one base statistic should be specified.")
    invalid = sorted(set(tokens).difference(set(_CB_BASE_SUBSTITUTIONS)))
    if len(invalid):
        txt = "Unsupported base statistics: {}. Supported: {}."
        raise ValueError(txt.format(", ".join(invalid), ", ".join(_CB_BASE_SUBSTITUTIONS)))
    selected = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        selected.append(token)
    return [s for s in _CB_BASE_SUBSTITUTIONS if s in selected]


def _resolve_requested_sub_tensor_backend(g):
    requested = str(g.get('sub_tensor_backend', 'auto')).lower()
    if requested not in _SUB_TENSOR_BACKENDS:
        raise ValueError('Invalid sub_tensor_backend: {}'.format(requested))
    return requested


def _normalize_branch_ids(branch_ids):
    arr = np.asarray(branch_ids)
    if arr.size == 0:
        return np.array([], dtype=np.int64)
    return np.atleast_1d(arr).astype(np.int64, copy=False).reshape(-1)


def _get_selected_branch_set(g):
    selected_branch_ids = g.get('state_loaded_branch_ids', None)
    if selected_branch_ids is None:
        return None
    return set(int(v) for v in _normalize_branch_ids(selected_branch_ids))


def _resolve_output_dtype(source_dtype, default_dtype):
    if np.dtype(source_dtype) == np.dtype(bool):
        return np.int32
    return default_dtype


def _resolve_cb_stat_columns(selected, arity):
    stat_col = {stat: arity + i for i, stat in enumerate(selected)}
    return (
        stat_col.get('any2any'),
        stat_col.get('spe2any'),
        stat_col.get('any2spe'),
        stat_col.get('spe2spe'),
    )


def _get_combo_row_bounds(combo_index, num_site, mmap_start):
    row_start = (combo_index * num_site) + (mmap_start * num_site)
    row_end = row_start + num_site
    return row_start, row_end


def _remove_file_if_exists(path):
    if os.path.exists(path):
        os.unlink(path)


def _build_branch_id_columns(arity):
    return ["branch_id_" + str(num + 1) for num in range(0, arity)]


def _resolve_reducer_chunks(id_combinations, n_jobs, g):
    chunk_factor = parallel.resolve_chunk_factor(g=g, task='reducer')
    return parallel.get_chunks(id_combinations, n_jobs, chunk_factor=chunk_factor)


def _run_parallel_reducer_to_dataframe(write_func, args_iterable, n_jobs, mmap_out, axis, dtype, columns, g):
    _remove_file_if_exists(mmap_out)
    df_mmap = np.memmap(mmap_out, dtype=dtype, shape=axis, mode='w+')
    backend = parallel.resolve_parallel_backend(g=g, task='reducer')
    try:
        parallel.run_starmap(
            func=write_func,
            args_iterable=args_iterable,
            n_jobs=n_jobs,
            backend=backend,
        )
        df_mmap.flush()
        df = pd.DataFrame(df_mmap, columns=columns)
    finally:
        del df_mmap
        _remove_file_if_exists(mmap_out)
    return df


def _initialize_reducer_output_array(mmap, df_mmap, shape, source_dtype, default_dtype):
    if mmap:
        return df_mmap
    out_dtype = _resolve_output_dtype(source_dtype=source_dtype, default_dtype=default_dtype)
    return np.zeros(shape=shape, dtype=out_dtype)


def _get_combo_index_range(mmap_start, num_combinations):
    start = mmap_start
    end = mmap_start + num_combinations
    return start, end


def resolve_sub_tensor_backend(g):
    if 'resolved_sub_tensor_backend' in g.keys():
        return g['resolved_sub_tensor_backend']
    requested = _resolve_requested_sub_tensor_backend(g)
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
    arr = np.asarray(sub_tensor)
    if arr.size == 0:
        return 0.0
    if tol > 0:
        nnz = np.count_nonzero(np.abs(arr) > tol)
    else:
        nnz = np.count_nonzero(arr)
    return nnz / arr.size

def resolve_reducer_backend(g, sub_tensor=None, label=''):
    requested = _resolve_requested_sub_tensor_backend(g)
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
    out = np.zeros(
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
    selected_branch_set = _get_selected_branch_set(g)
    for node in g['tree'].traverse():
        if ete.is_root(node):
            continue
        child = ete.get_prop(node, "numerical_label")
        if (selected_branch_set is not None) and (child not in selected_branch_set):
            continue
        parent = ete.get_prop(node.up, "numerical_label")
        if state_tensor_anc[parent, :, :].sum() < g['float_tol']:
            continue
        parent_matrix = state_tensor_anc[parent, :, :]
        child_matrix = state_tensor[child, :, :]
        if mode == 'asis':
            for a in range(num_state):
                parent_state = parent_matrix[:, a]
                if not np.any(parent_state):
                    continue
                for d in range(num_state):
                    if a == d:
                        continue
                    vals = parent_state * child_matrix[:, d]
                    nz = np.where(vals != 0)[0]
                    if nz.shape[0] == 0:
                        continue
                    rows, cols, data = sparse_entries[(0, a, d)]
                    rows.extend([child] * nz.shape[0])
                    cols.extend(nz.tolist())
                    data.extend(vals[nz].tolist())
        elif mode == 'syn':
            for sg, aa in enumerate(g['amino_acid_orders']):
                ind = np.array(g['synonymous_indices'][aa], dtype=np.int64)
                size = ind.shape[0]
                if size <= 1:
                    continue
                # Keep indexing semantics consistent with dense path: [state, site].
                parent_sub = state_tensor_anc[parent, :, ind]
                child_sub = state_tensor[child, :, ind]
                for a in range(size):
                    parent_state = parent_sub[a, :]
                    if not np.any(parent_state):
                        continue
                    for d in range(size):
                        if a == d:
                            continue
                        vals = parent_state * child_sub[d, :]
                        nz = np.where(vals != 0)[0]
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
        mat = sp.coo_matrix(
            (np.asarray(data, dtype=dtype), (np.asarray(rows), np.asarray(cols))),
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
        out = np.zeros(shape=(sub_tensor.num_branch,), dtype=np.float64)
        for mat in sub_tensor.blocks.values():
            out += np.asarray(mat.sum(axis=1)).reshape(-1)
        return out
    return sub_tensor.sum(axis=(1, 2, 3, 4))


def get_site_sub_counts(sub_tensor):
    if _is_sparse_sub_tensor(sub_tensor):
        out = np.zeros(shape=(sub_tensor.num_site,), dtype=np.float64)
        for mat in sub_tensor.blocks.values():
            out += np.asarray(mat.sum(axis=0)).reshape(-1)
        return out
    return sub_tensor.sum(axis=(0, 2, 3, 4))


def get_branch_site_sub_counts(sub_tensor, branch_id):
    if _is_sparse_sub_tensor(sub_tensor):
        out = np.zeros(shape=(sub_tensor.num_site,), dtype=np.float64)
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
        gad = np.zeros(
            shape=(sub_tensor.num_group, sub_tensor.num_state_from, sub_tensor.num_state_to),
            dtype=np.float64,
        )
        for (sg, a, d), mat in sub_tensor.blocks.items():
            gad[sg, a, d] += float(mat.data.sum())
    else:
        gad = sub_tensor.sum(axis=(0, 1))
    ga = gad.sum(axis=2)
    gd = gad.sum(axis=1)
    return gad, ga, gd


def _get_sparse_branch_tensor(sub_tensor, branch_id):
    out = np.zeros(
        shape=(sub_tensor.num_site, sub_tensor.num_group, sub_tensor.num_state_from, sub_tensor.num_state_to),
        dtype=sub_tensor.dtype,
    )
    for (sg, a, d), mat in sub_tensor.blocks.items():
        row = mat.getrow(int(branch_id))
        if row.nnz == 0:
            continue
        out[row.indices, sg, a, d] = row.data
    return out


def _get_sparse_site_vectors(
    sub_tensor,
    branch_ids,
    data_type=np.float64,
    group_block_index=None,
    row_cache=None,
    selected_base_stats=None,
):
    selected = _resolve_cb_base_substitutions(selected_base_stats=selected_base_stats)
    is_any2any = ("any2any" in selected)
    is_spe2any = ("spe2any" in selected)
    is_any2spe = ("any2spe" in selected)
    is_spe2spe = ("spe2spe" in selected)
    num_site = sub_tensor.num_site
    any2any = np.zeros(shape=(num_site,), dtype=data_type)
    spe2any = np.zeros(shape=(num_site,), dtype=data_type)
    any2spe = np.zeros(shape=(num_site,), dtype=data_type)
    spe2spe = np.zeros(shape=(num_site,), dtype=data_type)
    for sg in range(sub_tensor.num_group):
        sub_sg = _get_sparse_combination_group_tensor(
            sub_tensor=sub_tensor,
            branch_ids=branch_ids,
            sg=int(sg),
            data_type=data_type,
            group_block_index=group_block_index,
            row_cache=row_cache,
        )
        if is_any2any:
            any2any += sub_sg.sum(axis=(2, 3)).prod(axis=0)
        if is_spe2any:
            spe2any += sub_sg.sum(axis=3).prod(axis=0).sum(axis=1)
        if is_any2spe:
            any2spe += sub_sg.sum(axis=2).prod(axis=0).sum(axis=1)
        if is_spe2spe:
            spe2spe += sub_sg.prod(axis=0).sum(axis=(1, 2))
    return any2any, spe2any, any2spe, spe2spe


def get_cs_sparse(id_combinations, sub_tensor, attr):
    num_site = sub_tensor.shape[1]
    df = np.zeros([num_site, 5], dtype=np.float64)
    df[:, 0] = np.arange(num_site)
    group_block_index = _get_sparse_group_block_index(sub_tensor)
    row_cache = dict()
    for i in np.arange(id_combinations.shape[0]):
        any2any, spe2any, any2spe, spe2spe = _get_sparse_site_vectors(
            sub_tensor=sub_tensor,
            branch_ids=id_combinations[i, :],
            data_type=np.float64,
            group_block_index=group_block_index,
            row_cache=row_cache,
        )
        df[:, 1] += any2any
        df[:, 2] += spe2any
        df[:, 3] += any2spe
        df[:, 4] += spe2spe
    cn = ['site',] + [ 'OC'+attr+subs for subs in ["any2any","spe2any","any2spe","spe2spe"] ]
    df = pd.DataFrame(df, columns=cn)
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
    sub_tensor = np.memmap(mmap_tensor, dtype=dtype, shape=axis, mode='w+')
    return sub_tensor

def get_substitution_tensor(state_tensor, state_tensor_anc=None, mode='', g=None, mmap_attr=''):
    if g is None:
        raise ValueError('g is required.')
    backend = resolve_sub_tensor_backend(g)
    if state_tensor_anc is None:
        state_tensor_anc = state_tensor
    selected_branch_set = _get_selected_branch_set(g)
    if backend == 'sparse':
        return _build_sparse_substitution_tensor(
            state_tensor=state_tensor,
            state_tensor_anc=state_tensor_anc,
            mode=mode,
            g=g,
        )
    sub_tensor = initialize_substitution_tensor(state_tensor, mode, g, mmap_attr)
    if (g['ml_anc']=='no'):
        sub_tensor[:,:,:,:,:] = np.nan
    if mode=='asis':
        num_state = state_tensor.shape[2]
        diag_zero = np.diag([-1] * num_state) + 1
    for node in g['tree'].traverse():
        if ete.is_root(node):
            continue
        child = ete.get_prop(node, "numerical_label")
        if (selected_branch_set is not None) and (child not in selected_branch_set):
            continue
        parent = ete.get_prop(node.up, "numerical_label")
        if state_tensor_anc[parent, :, :].sum()<g['float_tol']:
            continue
        if mode=='asis':
            sub_matrix = np.einsum("sa,sd,ad->sad", state_tensor_anc[parent,:,:], state_tensor[child,:,:], diag_zero)
            # s=site, a=ancestral, d=derived
            sub_tensor[child, :, 0, :, :] = sub_matrix
        elif mode=='syn':
            for s,aa in enumerate(g['amino_acid_orders']):
                ind = np.array(g['synonymous_indices'][aa])
                size = len(ind)
                diag_zero = np.diag([-1] * size) + 1
                parent_matrix = state_tensor_anc[parent, :, ind] # axis is swapped, shape=[state,site]
                child_matrix = state_tensor[child, :, ind] # axis is swapped, shape=[state,site]
                sub_matrix = np.einsum("as,ds,ad->sad", parent_matrix, child_matrix, diag_zero)
                sub_tensor[child, :, s, :size, :size] = sub_matrix
    if np.isnan(sub_tensor).any():
        sub_tensor = np.nan_to_num(sub_tensor, nan=0, copy=False)
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
    df = pd.DataFrame(np.nan, index=range(0, g['num_node']), columns=column_names)
    df['branch_name'] = df['branch_name'].astype(str)
    if sitewise:
        df[attr + '_sitewise'] = ''
    branch_sub_counts = get_branch_sub_counts(sub_tensor) if _is_sparse_sub_tensor(sub_tensor) else None
    i=0
    for node in g['tree'].traverse():
        df.at[i,'branch_name'] = getattr(node, 'name')
        df.at[i,'branch_id'] = ete.get_prop(node, "numerical_label")
        if _is_sparse_sub_tensor(sub_tensor):
            df.at[i,attr+'_sub'] = branch_sub_counts[ete.get_prop(node, "numerical_label")]
            branch_tensor = _get_sparse_branch_tensor(sub_tensor=sub_tensor, branch_id=ete.get_prop(node, "numerical_label")) if sitewise else None
        else:
            df.at[i,attr+'_sub'] = np.nansum(sub_tensor[ete.get_prop(node, "numerical_label"),:,:,:,:])
            branch_tensor = sub_tensor[ete.get_prop(node, "numerical_label"), :, :, :, :] if sitewise else None
        if sitewise:
            sub_list = list()
            if attr=='N':
                state_order = g['amino_acid_orders']
            elif attr=='S':
                raise ValueError('This function is not supported for synonymous substitutions.')
            for s in range(sub_tensor.shape[1]):
                site_values = branch_tensor[s, :, :, :]
                if not np.isfinite(site_values).any():
                    continue
                max_value = np.nanmax(site_values)
                if (not np.isfinite(max_value)) or (max_value < min_sitewise_pp):
                    continue
                max_idx = np.where(site_values == max_value)
                if (max_idx[1].shape[0] == 0) or (max_idx[2].shape[0] == 0):
                    continue
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
    df = pd.DataFrame(0, index=np.arange(0,num_site), columns=column_names)
    df['site'] = np.arange(0, num_site)
    df[attr+'_sub'] = get_site_sub_counts(sub_tensor)
    df['site'] = df['site'].astype(int)
    df = df.sort_values(by='site')
    df = table.set_substitution_dtype(df=df)
    return(df)

def get_cs(id_combinations, sub_tensor, attr):
    if _is_sparse_sub_tensor(sub_tensor):
        return get_cs_sparse(id_combinations=id_combinations, sub_tensor=sub_tensor, attr=attr)
    num_site = sub_tensor.shape[1]
    df = np.zeros([num_site, 5])
    df[:, 0] = np.arange(num_site)
    for i in np.arange(id_combinations.shape[0]):
        for sg in np.arange(sub_tensor.shape[2]): # Couldn't this sg included in the matrix calc using .sum()?
            df[:, 1] += np.nan_to_num(sub_tensor[id_combinations[i,:], :, sg, :, :].sum(axis=(2, 3)).prod(axis=0))  # any2any
            df[:, 2] += np.nan_to_num(sub_tensor[id_combinations[i,:], :, sg, :, :].sum(axis=3).prod(axis=0).sum(axis=1))  # spe2any
            df[:, 3] += np.nan_to_num(sub_tensor[id_combinations[i,:], :, sg, :, :].sum(axis=2).prod(axis=0).sum(axis=1))  # any2spe
            df[:, 4] += np.nan_to_num(sub_tensor[id_combinations[i,:], :, sg, :, :].prod(axis=0).sum(axis=(1, 2)))  # spe2spe
    cn = ['site',] + [ 'OC'+attr+subs for subs in ["any2any","spe2any","any2spe","spe2spe"] ]
    df = pd.DataFrame(df, columns=cn)
    df = table.set_substitution_dtype(df=df)
    return (df)

def get_bs(S_tensor, N_tensor):
    num_site = S_tensor.shape[1]
    num_branch = S_tensor.shape[0]
    column_names=['branch_id','site','S_sub','N_sub']
    df = pd.DataFrame(np.nan, index=np.arange(0,num_branch*num_site), columns=column_names)
    for i in np.arange(num_branch):
        ind = np.arange(i*num_site, (i+1)*num_site)
        df.loc[ind, 'site'] = np.arange(0, num_site)
        df.loc[ind, 'branch_id'] = i
        df.loc[ind, 'S_sub'] = get_branch_site_sub_counts(S_tensor, i)
        df.loc[ind, 'N_sub'] = get_branch_site_sub_counts(N_tensor, i)
    df = table.set_substitution_dtype(df=df)
    return(df)

def _is_cython_dense_arity2_compatible(id_combinations, sub_tensor, mmap=False, df_mmap=None):
    if _is_sparse_sub_tensor(sub_tensor):
        return False
    if id_combinations.shape[1] != 2:
        return False
    if id_combinations.dtype.kind not in ['i', 'u']:
        return False
    if sub_tensor.dtype != np.float64:
        return False
    if mmap and ((df_mmap is None) or (df_mmap.dtype != np.float64)):
        return False
    return True


def _can_use_cython_dense_cb(id_combinations, sub_tensor, mmap=False, df_mmap=None, float_type=np.float64):
    if not _is_cython_dense_arity2_compatible(
        id_combinations=id_combinations,
        sub_tensor=sub_tensor,
        mmap=mmap,
        df_mmap=df_mmap,
    ):
        return False
    if float_type != np.float64:
        return False
    return hasattr(substitution_cy, 'calc_combinatorial_sub_double_arity2')

def _can_use_cython_dense_cbs(id_combinations, sub_tensor, mmap=False, df_mmap=None):
    if not _is_cython_dense_arity2_compatible(
        id_combinations=id_combinations,
        sub_tensor=sub_tensor,
        mmap=mmap,
        df_mmap=df_mmap,
    ):
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


def _resolve_dense_reducer_batch_size(sub_tensor, arity, num_combinations, target_elements=8_000_000):
    if num_combinations <= 1:
        return 1
    combo_elements = (
        int(arity)
        * int(sub_tensor.shape[1])
        * int(sub_tensor.shape[2])
        * int(sub_tensor.shape[3])
        * int(sub_tensor.shape[4])
    )
    if combo_elements <= 0:
        return 1
    batch_size = int(target_elements // combo_elements)
    if batch_size < 1:
        batch_size = 1
    return min(int(num_combinations), batch_size)


def _resolve_reducer_progress_step(num_items, min_step=50_000, target_updates=20):
    if num_items <= 0:
        return min_step
    return max(int(num_items // max(target_updates, 1)), int(min_step))


def _extract_selected_stats_from_full_cb(full_df, selected, arity):
    if len(selected) == len(_CB_BASE_SUBSTITUTIONS):
        return full_df
    full_stat_cols = {
        "any2any": arity + 0,
        "spe2any": arity + 1,
        "any2spe": arity + 2,
        "spe2spe": arity + 3,
    }
    out = np.zeros(shape=(full_df.shape[0], arity + len(selected)), dtype=full_df.dtype)
    out[:, :arity] = full_df[:, :arity]
    for i, stat in enumerate(selected):
        out[:, arity + i] = full_df[:, full_stat_cols[stat]]
    return out


def _can_use_cython_sparse_cb_summary(id_combinations, sub_tensor, selected):
    if not _is_sparse_sub_tensor(sub_tensor):
        return False
    if not hasattr(substitution_cy, 'calc_combinatorial_sub_sparse_summary_double_arity2'):
        return False
    if id_combinations.shape[1] != 2:
        return False
    if id_combinations.dtype.kind not in ['i', 'u']:
        return False
    if ('spe2spe' in selected) and (not _sparse_summary_fastpath_supports_spe2spe()):
        return False
    return True


def _get_sparse_summary_fastpath_param_names():
    func = getattr(substitution_cy, 'calc_combinatorial_sub_sparse_summary_double_arity2', None)
    if func is None:
        return tuple()
    try:
        return tuple(inspect.signature(func).parameters.keys())
    except (TypeError, ValueError):
        return tuple()


def _sparse_summary_fastpath_supports_spe2spe():
    params = _get_sparse_summary_fastpath_param_names()
    return ('branch_group_pair_site_obj' in params) and ('calc_spe2spe' in params)


def _can_use_cython_sparse_summary_accumulator(rows, cols, vals):
    if substitution_sparse_cy is None:
        return False
    if not isinstance(rows, np.ndarray) or not isinstance(cols, np.ndarray) or not isinstance(vals, np.ndarray):
        return False
    if rows.dtype != np.int64:
        return False
    if cols.dtype != np.int64:
        return False
    if vals.dtype != np.float64:
        return False
    if rows.ndim != 1 or cols.ndim != 1 or vals.ndim != 1:
        return False
    if rows.shape[0] != cols.shape[0] or rows.shape[0] != vals.shape[0]:
        return False
    return hasattr(substitution_sparse_cy, 'accumulate_sparse_summary_block_double')


def _get_sparse_cb_summary_arrays(sub_tensor, selected):
    need_any2any = ('any2any' in selected)
    need_spe2any = ('spe2any' in selected)
    need_any2spe = ('any2spe' in selected)
    need_spe2spe = ('spe2spe' in selected)
    key = (need_any2any, need_spe2any, need_any2spe, need_spe2spe)
    cache = getattr(sub_tensor, '_cb_sparse_summary_cache', None)
    if cache is None:
        cache = dict()
    if key in cache:
        return cache[key]
    num_branch = int(sub_tensor.num_branch)
    num_site = int(sub_tensor.num_site)
    num_group = int(sub_tensor.num_group)
    num_state_from = int(sub_tensor.num_state_from)
    num_state_to = int(sub_tensor.num_state_to)
    num_state_pair = int(num_state_from * num_state_to)
    branch_group_site_total = None
    branch_group_from_site = None
    branch_group_to_site = None
    branch_group_pair_site = None
    total_flat = None
    from_flat = None
    to_flat = None
    pair_flat = None
    if need_any2any:
        branch_group_site_total = np.zeros(
            shape=(num_branch, num_group, num_site),
            dtype=np.float64,
        )
        total_flat = branch_group_site_total.reshape(-1)
    if need_spe2any:
        branch_group_from_site = np.zeros(
            shape=(num_branch, num_group, num_state_from, num_site),
            dtype=np.float64,
        )
        from_flat = branch_group_from_site.reshape(-1)
    if need_any2spe:
        branch_group_to_site = np.zeros(
            shape=(num_branch, num_group, num_state_to, num_site),
            dtype=np.float64,
        )
        to_flat = branch_group_to_site.reshape(-1)
    if need_spe2spe:
        branch_group_pair_site = np.zeros(
            shape=(num_branch, num_group, num_state_pair, num_site),
            dtype=np.float64,
        )
        pair_flat = branch_group_pair_site.reshape(-1)
    for (sg, a, d), mat in sub_tensor.blocks.items():
        coo = mat.tocoo(copy=False)
        if coo.nnz == 0:
            continue
        rows = np.asarray(coo.row, dtype=np.int64)
        cols = np.asarray(coo.col, dtype=np.int64)
        vals = np.asarray(coo.data, dtype=np.float64)
        if _can_use_cython_sparse_summary_accumulator(rows=rows, cols=cols, vals=vals):
            substitution_sparse_cy.accumulate_sparse_summary_block_double(
                rows=rows,
                cols=cols,
                vals=vals,
                sg=int(sg),
                a=int(a),
                d=int(d),
                num_group=num_group,
                num_site=num_site,
                num_state_from=num_state_from,
                num_state_to=num_state_to,
                total_flat_obj=total_flat,
                from_flat_obj=from_flat,
                to_flat_obj=to_flat,
                pair_flat_obj=pair_flat,
                need_any2any=need_any2any,
                need_spe2any=need_spe2any,
                need_any2spe=need_any2spe,
                need_spe2spe=need_spe2spe,
            )
        else:
            if need_any2any:
                ind = ((rows * num_group) + int(sg)) * num_site + cols
                np.add.at(total_flat, ind, vals)
            if need_spe2any:
                ind = (((rows * num_group) + int(sg)) * num_state_from + int(a)) * num_site + cols
                np.add.at(from_flat, ind, vals)
            if need_any2spe:
                ind = (((rows * num_group) + int(sg)) * num_state_to + int(d)) * num_site + cols
                np.add.at(to_flat, ind, vals)
            if need_spe2spe:
                pair_index = int(a) * num_state_to + int(d)
                ind = (((rows * num_group) + int(sg)) * num_state_pair + pair_index) * num_site + cols
                np.add.at(pair_flat, ind, vals)
    out = (branch_group_site_total, branch_group_from_site, branch_group_to_site, branch_group_pair_site)
    cache[key] = out
    setattr(sub_tensor, '_cb_sparse_summary_cache', cache)
    return out


def _clear_sparse_cb_summary_arrays(sub_tensor, selected):
    need_any2any = ('any2any' in selected)
    need_spe2any = ('spe2any' in selected)
    need_any2spe = ('any2spe' in selected)
    need_spe2spe = ('spe2spe' in selected)
    key = (need_any2any, need_spe2any, need_any2spe, need_spe2spe)
    cache = getattr(sub_tensor, '_cb_sparse_summary_cache', None)
    if isinstance(cache, dict) and (key in cache):
        cache.pop(key, None)


def _run_sparse_cb_summary_cython(
    id_combinations,
    selected,
    mmap,
    df_mmap,
    mmap_start,
    float_type,
    branch_group_site_total,
    branch_group_from_site,
    branch_group_to_site,
    branch_group_pair_site,
):
    arity = id_combinations.shape[1]
    kwargs = dict(
        id_combinations=np.asarray(id_combinations, dtype=np.int64),
        mmap_start=0,
        branch_group_site_total_obj=branch_group_site_total,
        branch_group_from_site_obj=branch_group_from_site,
        branch_group_to_site_obj=branch_group_to_site,
        mmap=False,
        df_mmap=None,
        calc_any2any=('any2any' in selected),
        calc_spe2any=('spe2any' in selected),
        calc_any2spe=('any2spe' in selected),
    )
    if _sparse_summary_fastpath_supports_spe2spe():
        kwargs['branch_group_pair_site_obj'] = branch_group_pair_site
        kwargs['calc_spe2spe'] = ('spe2spe' in selected)
    out_full = substitution_cy.calc_combinatorial_sub_sparse_summary_double_arity2(**kwargs)
    selected_df = _extract_selected_stats_from_full_cb(
        full_df=np.asarray(out_full, dtype=np.float64),
        selected=selected,
        arity=arity,
    )
    if mmap:
        row_start, row_end = _get_combo_index_range(mmap_start=mmap_start, num_combinations=id_combinations.shape[0])
        df_mmap[row_start:row_end, :] = selected_df.astype(df_mmap.dtype, copy=False)
        return None
    return selected_df.astype(float_type, copy=False)


def sub_tensor2cb(
    id_combinations,
    sub_tensor,
    mmap=False,
    df_mmap=None,
    mmap_start=0,
    float_type=np.float64,
    selected_base_stats=None,
):
    selected = _resolve_cb_base_substitutions(selected_base_stats=selected_base_stats)
    arity = id_combinations.shape[1]
    use_full_stats = (len(selected) == len(_CB_BASE_SUBSTITUTIONS))
    can_use_cython = _can_use_cython_dense_cb(
        id_combinations=id_combinations,
        sub_tensor=sub_tensor,
        mmap=mmap,
        df_mmap=df_mmap,
        float_type=float_type,
    )
    if can_use_cython and (use_full_stats or (not mmap)):
        try:
            out = substitution_cy.calc_combinatorial_sub_double_arity2(
                id_combinations=np.asarray(id_combinations, dtype=np.int64),
                mmap_start=mmap_start,
                sub_tensor=sub_tensor,
                mmap=mmap,
                df_mmap=df_mmap,
            )
            if not mmap:
                return _extract_selected_stats_from_full_cb(
                    full_df=out,
                    selected=selected,
                    arity=arity,
                )
            return
        except Exception as exc:
            _warn_cython_fallback('sub_tensor2cb', exc)
    shape = (id_combinations.shape[0], arity + len(selected))
    df = _initialize_reducer_output_array(
        mmap=mmap,
        df_mmap=df_mmap,
        shape=shape,
        source_dtype=sub_tensor.dtype,
        default_dtype=float_type,
    )
    col_any2any, col_spe2any, col_any2spe, col_spe2spe = _resolve_cb_stat_columns(
        selected=selected,
        arity=arity,
    )
    start_time = time.time()
    start, end = _get_combo_index_range(mmap_start=mmap_start, num_combinations=id_combinations.shape[0])
    df[start:end, :arity] = id_combinations[:, :]  # branch_ids
    progress_step = _resolve_reducer_progress_step(num_items=id_combinations.shape[0])
    batch_size = _resolve_dense_reducer_batch_size(
        sub_tensor=sub_tensor,
        arity=arity,
        num_combinations=id_combinations.shape[0],
    )
    for batch_start in range(0, id_combinations.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, id_combinations.shape[0])
        combo_batch = id_combinations[batch_start:batch_end, :]
        row_slice = slice(start + batch_start, start + batch_end)
        sub_batch = sub_tensor[combo_batch, :, :, :, :]
        if col_any2any is not None:
            df[row_slice, col_any2any] += sub_batch.sum(axis=(4, 5)).prod(axis=1).sum(axis=(1, 2))
        if col_spe2any is not None:
            df[row_slice, col_spe2any] += sub_batch.sum(axis=5).prod(axis=1).sum(axis=(1, 2, 3))
        if col_any2spe is not None:
            df[row_slice, col_any2spe] += sub_batch.sum(axis=4).prod(axis=1).sum(axis=(1, 2, 3))
        if col_spe2spe is not None:
            df[row_slice, col_spe2spe] += sub_batch.prod(axis=1).sum(axis=(1, 2, 3, 4))
        if (batch_start != 0) and (batch_start % progress_step == 0):
            mmap_end = mmap_start + id_combinations.shape[0]
            txt = 'cb: {:,}th in the id range {:,}-{:,}: {:,} sec'
            print(txt.format(batch_start, mmap_start, mmap_end, int(time.time() - start_time)), flush=True)
    if not mmap:
        return (df)

def sub_tensor2cb_sparse(
    id_combinations,
    sub_tensor,
    mmap=False,
    df_mmap=None,
    mmap_start=0,
    float_type=np.float64,
    selected_base_stats=None,
):
    selected = _resolve_cb_base_substitutions(selected_base_stats=selected_base_stats)
    arity = id_combinations.shape[1]
    shape = (id_combinations.shape[0], arity + len(selected))
    df = _initialize_reducer_output_array(
        mmap=mmap,
        df_mmap=df_mmap,
        shape=shape,
        source_dtype=sub_tensor.dtype,
        default_dtype=float_type,
    )
    if _can_use_cython_sparse_cb_summary(
        id_combinations=id_combinations,
        sub_tensor=sub_tensor,
        selected=selected,
    ):
        try:
            summary_arrays = _get_sparse_cb_summary_arrays(
                sub_tensor=sub_tensor,
                selected=selected,
            )
            branch_group_site_total = summary_arrays[0]
            branch_group_from_site = summary_arrays[1]
            branch_group_to_site = summary_arrays[2]
            branch_group_pair_site = summary_arrays[3] if len(summary_arrays) > 3 else None
            out = _run_sparse_cb_summary_cython(
                id_combinations=id_combinations,
                selected=selected,
                mmap=mmap,
                df_mmap=df_mmap,
                mmap_start=mmap_start,
                float_type=float_type,
                branch_group_site_total=branch_group_site_total,
                branch_group_from_site=branch_group_from_site,
                branch_group_to_site=branch_group_to_site,
                branch_group_pair_site=branch_group_pair_site,
            )
            _clear_sparse_cb_summary_arrays(sub_tensor=sub_tensor, selected=selected)
            if not mmap:
                return out
            return
        except Exception as exc:
            _clear_sparse_cb_summary_arrays(sub_tensor=sub_tensor, selected=selected)
            _warn_cython_fallback('sub_tensor2cb_sparse', exc)
    col_any2any, col_spe2any, col_any2spe, col_spe2spe = _resolve_cb_stat_columns(
        selected=selected,
        arity=arity,
    )
    start_time = time.time()
    start, end = _get_combo_index_range(mmap_start=mmap_start, num_combinations=id_combinations.shape[0])
    df[start:end, :arity] = id_combinations[:, :]  # branch_ids
    group_block_index = _get_sparse_group_block_index(sub_tensor)
    row_cache = dict()
    progress_step = _resolve_reducer_progress_step(num_items=id_combinations.shape[0])
    for j, branch_ids in enumerate(id_combinations):
        i = start + j
        any2any, spe2any, any2spe, spe2spe = _get_sparse_site_vectors(
            sub_tensor=sub_tensor,
            branch_ids=branch_ids,
            data_type=df.dtype,
            group_block_index=group_block_index,
            row_cache=row_cache,
            selected_base_stats=selected,
        )
        if col_any2any is not None:
            df[i, col_any2any] += any2any.sum()
        if col_spe2any is not None:
            df[i, col_spe2any] += spe2any.sum()
        if col_any2spe is not None:
            df[i, col_any2spe] += any2spe.sum()
        if col_spe2spe is not None:
            df[i, col_spe2spe] += spe2spe.sum()
        if (j != 0) and (j % progress_step == 0):
            mmap_end = mmap_start + id_combinations.shape[0]
            txt = 'cb(sparse): {:,}th in the id range {:,}-{:,}: {:,} sec'
            print(txt.format(j, mmap_start, mmap_end, int(time.time() - start_time)), flush=True)
    if not mmap:
        return (df)

def _write_cb_chunk_to_mmap(writer, ids, sub_tensor, mmap_out, axis, dtype, mmap_start, float_type, selected_base_stats):
    df_mmap = np.memmap(mmap_out, dtype=dtype, shape=axis, mode='r+')
    writer(ids, sub_tensor, True, df_mmap, mmap_start, float_type, selected_base_stats=selected_base_stats)
    df_mmap.flush()


def _resolve_cb_writer_and_columns(sub_tensor, attr, arity, selected):
    columns = _build_branch_id_columns(arity=arity) + [attr + subs for subs in selected]
    writer = sub_tensor2cb_sparse if _is_sparse_sub_tensor(sub_tensor) else sub_tensor2cb
    return writer, columns


def _resolve_cb_n_jobs(id_combinations, sub_tensor, g, writer, selected):
    n_jobs = parallel.resolve_n_jobs(num_items=id_combinations.shape[0], threads=g['threads'])
    if (writer is sub_tensor2cb) and _can_use_cython_dense_cb(
        id_combinations=id_combinations,
        sub_tensor=sub_tensor,
        mmap=False,
        df_mmap=None,
        float_type=g['float_type'],
    ):
        if len(selected) == len(_CB_BASE_SUBSTITUTIONS):
            n_jobs = _resolve_dense_cython_n_jobs(
                n_jobs=n_jobs,
                id_combinations=id_combinations,
                sub_tensor=sub_tensor,
                g=g,
                task='cb',
            )
        else:
            txt = 'Dense Cython cb reducer is bypassed for selective stats: {}'
            print(txt.format(','.join(selected)), flush=True)
    return n_jobs


def _run_cb_single_job(writer, id_combinations, sub_tensor, float_type, selected, columns):
    df = writer(
        id_combinations,
        sub_tensor,
        mmap=False,
        df_mmap=None,
        mmap_start=0,
        float_type=float_type,
        selected_base_stats=selected,
    )
    return pd.DataFrame(df, columns=columns)


def _run_cb_parallel_jobs(writer, id_combinations, sub_tensor, g, arity, selected, columns):
    n_jobs = _resolve_cb_n_jobs(
        id_combinations=id_combinations,
        sub_tensor=sub_tensor,
        g=g,
        writer=writer,
        selected=selected,
    )
    if n_jobs == 1:
        return _run_cb_single_job(
            writer=writer,
            id_combinations=id_combinations,
            sub_tensor=sub_tensor,
            float_type=g['float_type'],
            selected=selected,
            columns=columns,
        )
    id_chunks, mmap_starts = _resolve_reducer_chunks(id_combinations=id_combinations, n_jobs=n_jobs, g=g)
    mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.cb.out.mmap')
    axis = (id_combinations.shape[0], arity + len(selected))
    my_dtype = _resolve_output_dtype(source_dtype=sub_tensor.dtype, default_dtype=sub_tensor.dtype)
    tasks = [
        (writer, ids, sub_tensor, mmap_out, axis, my_dtype, ms, g['float_type'], selected)
        for ids, ms in zip(id_chunks, mmap_starts)
    ]
    return _run_parallel_reducer_to_dataframe(
        write_func=_write_cb_chunk_to_mmap,
        args_iterable=tasks,
        n_jobs=n_jobs,
        mmap_out=mmap_out,
        axis=axis,
        dtype=my_dtype,
        columns=columns,
        g=g,
    )


def get_cb(id_combinations, sub_tensor, g, attr, selected_base_stats=None):
    sub_tensor = get_reducer_sub_tensor(sub_tensor=sub_tensor, g=g, label='cb_'+attr)
    arity = id_combinations.shape[1]
    selected = _resolve_cb_base_substitutions(selected_base_stats=selected_base_stats)
    writer, columns = _resolve_cb_writer_and_columns(
        sub_tensor=sub_tensor,
        attr=attr,
        arity=arity,
        selected=selected,
    )
    df = _run_cb_parallel_jobs(
        writer=writer,
        id_combinations=id_combinations,
        sub_tensor=sub_tensor,
        g=g,
        arity=arity,
        selected=selected,
        columns=columns,
    )
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
                id_combinations=np.asarray(id_combinations, dtype=np.int64),
                mmap_start=mmap_start,
                sub_tensor=sub_tensor,
                mmap=mmap,
                df_mmap=df_mmap,
            )
            if not mmap:
                return out
            return
        except Exception as exc:
            _warn_cython_fallback('sub_tensor2cbs', exc)
    arity = id_combinations.shape[1]
    num_site = sub_tensor.shape[1]
    sites = np.arange(num_site)
    shape = (int(id_combinations.shape[0] * num_site), arity + 5)
    df = _initialize_reducer_output_array(
        mmap=mmap,
        df_mmap=df_mmap,
        shape=shape,
        source_dtype=sub_tensor.dtype,
        default_dtype=sub_tensor.dtype,
    )
    start_time = time.time()
    start, end = _get_combo_index_range(mmap_start=mmap_start, num_combinations=id_combinations.shape[0])
    progress_step = _resolve_reducer_progress_step(num_items=id_combinations.shape[0])
    batch_size = _resolve_dense_reducer_batch_size(
        sub_tensor=sub_tensor,
        arity=arity,
        num_combinations=id_combinations.shape[0],
    )
    for batch_start in range(0, id_combinations.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, id_combinations.shape[0])
        combo_batch = id_combinations[batch_start:batch_end, :]
        batch_combo_num = combo_batch.shape[0]
        combo_global_start = start + batch_start
        row_start = combo_global_start * num_site
        row_end = row_start + (batch_combo_num * num_site)
        df[row_start:row_end, :arity] = np.repeat(combo_batch, repeats=num_site, axis=0)  # branch_ids
        df[row_start:row_end, arity] = np.tile(sites, batch_combo_num)  # site
        sub_batch = sub_tensor[combo_batch, :, :, :, :]
        df[row_start:row_end, arity + 1] += sub_batch.sum(axis=(4, 5)).prod(axis=1).sum(axis=2).reshape(-1)  # any2any
        df[row_start:row_end, arity + 2] += sub_batch.sum(axis=5).prod(axis=1).sum(axis=(2, 3)).reshape(-1)  # spe2any
        df[row_start:row_end, arity + 3] += sub_batch.sum(axis=4).prod(axis=1).sum(axis=(2, 3)).reshape(-1)  # any2spe
        df[row_start:row_end, arity + 4] += sub_batch.prod(axis=1).sum(axis=(2, 3, 4)).reshape(-1)  # spe2spe
        if (batch_start != 0) and (batch_start % progress_step == 0):
            mmap_end = mmap_start+id_combinations.shape[0]
            txt = 'cbs: {:,}th in the id range {:,}-{:,}: {:,} sec'
            print(txt.format(batch_start, mmap_start, mmap_end, int(time.time() - start_time)), flush=True)
    if not mmap:
        return df

def sub_tensor2cbs_sparse(id_combinations, sub_tensor, mmap=False, df_mmap=None, mmap_start=0):
    arity = id_combinations.shape[1]
    num_site = sub_tensor.shape[1]
    sites = np.arange(num_site)
    shape = (int(id_combinations.shape[0] * num_site), arity + 5)
    df = _initialize_reducer_output_array(
        mmap=mmap,
        df_mmap=df_mmap,
        shape=shape,
        source_dtype=sub_tensor.dtype,
        default_dtype=sub_tensor.dtype,
    )
    start_time = time.time()
    group_block_index = _get_sparse_group_block_index(sub_tensor)
    row_cache = dict()
    start, end = _get_combo_index_range(mmap_start=mmap_start, num_combinations=id_combinations.shape[0])
    progress_step = _resolve_reducer_progress_step(num_items=id_combinations.shape[0])
    for combo_index, combo_branch_ids in enumerate(id_combinations):
        combo_global_index = start + combo_index
        row_start = combo_global_index * num_site
        row_end = row_start + num_site
        df[row_start:row_end, :arity] = combo_branch_ids  # branch_ids
        df[row_start:row_end, arity] = sites  # site
        any2any, spe2any, any2spe, spe2spe = _get_sparse_site_vectors(
            sub_tensor=sub_tensor,
            branch_ids=combo_branch_ids,
            data_type=df.dtype,
            group_block_index=group_block_index,
            row_cache=row_cache,
        )
        df[row_start:row_end,arity+1] += any2any
        df[row_start:row_end,arity+2] += spe2any
        df[row_start:row_end,arity+3] += any2spe
        df[row_start:row_end,arity+4] += spe2spe
        if (combo_index != 0) and (combo_index % progress_step == 0):
            mmap_end = mmap_start+id_combinations.shape[0]
            txt = 'cbs(sparse): {:,}th in the id range {:,}-{:,}: {:,} sec'
            print(txt.format(combo_index, mmap_start, mmap_end, int(time.time() - start_time)), flush=True)
    if not mmap:
        return df

def _write_cbs_chunk_to_mmap(writer, ids, sub_tensor, mmap_out, axis, dtype, mmap_start):
    df_mmap = np.memmap(mmap_out, dtype=dtype, shape=axis, mode='r+')
    writer(ids, sub_tensor, True, df_mmap, mmap_start)
    df_mmap.flush()


def _resolve_cbs_writer_and_columns(sub_tensor, attr, arity):
    branch_columns = _build_branch_id_columns(arity=arity)
    site_column = ['site']
    stat_columns = ['OC' + attr + subs for subs in ["any2any", "spe2any", "any2spe", "spe2spe"]]
    columns = branch_columns + site_column + stat_columns
    writer = sub_tensor2cbs_sparse if _is_sparse_sub_tensor(sub_tensor) else sub_tensor2cbs
    return writer, columns


def _resolve_cbs_n_jobs(id_combinations, sub_tensor, g, writer):
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
    return n_jobs


def _run_cbs_single_job(writer, id_combinations, sub_tensor, columns):
    df = writer(id_combinations, sub_tensor)
    return pd.DataFrame(df, columns=columns)


def _run_cbs_parallel_jobs(writer, id_combinations, sub_tensor, g, arity, columns):
    n_jobs = _resolve_cbs_n_jobs(
        id_combinations=id_combinations,
        sub_tensor=sub_tensor,
        g=g,
        writer=writer,
    )
    if n_jobs == 1:
        return _run_cbs_single_job(
            writer=writer,
            id_combinations=id_combinations,
            sub_tensor=sub_tensor,
            columns=columns,
        )
    id_chunks, mmap_starts = _resolve_reducer_chunks(id_combinations=id_combinations, n_jobs=n_jobs, g=g)
    mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.cbs.out.mmap')
    axis = (id_combinations.shape[0] * sub_tensor.shape[1], arity + 5)
    my_dtype = _resolve_output_dtype(source_dtype=sub_tensor.dtype, default_dtype=sub_tensor.dtype)
    tasks = [
        (writer, ids, sub_tensor, mmap_out, axis, my_dtype, ms)
        for ids, ms in zip(id_chunks, mmap_starts)
    ]
    return _run_parallel_reducer_to_dataframe(
        write_func=_write_cbs_chunk_to_mmap,
        args_iterable=tasks,
        n_jobs=n_jobs,
        mmap_out=mmap_out,
        axis=axis,
        dtype=my_dtype,
        columns=columns,
        g=g,
    )


def get_cbs(id_combinations, sub_tensor, attr, g):
    sub_tensor = get_reducer_sub_tensor(sub_tensor=sub_tensor, g=g, label='cbs_'+attr)
    print("Calculating combinatorial substitutions: attr =", attr, flush=True)
    arity = id_combinations.shape[1]
    writer, columns = _resolve_cbs_writer_and_columns(
        sub_tensor=sub_tensor,
        attr=attr,
        arity=arity,
    )
    df = _run_cbs_parallel_jobs(
        writer=writer,
        id_combinations=id_combinations,
        sub_tensor=sub_tensor,
        g=g,
        arity=arity,
        columns=columns,
    )
    df = df.dropna()
    df = table.sort_branch_ids(df)
    df = table.set_substitution_dtype(df=df)
    return(df)

def get_sub_sites(g, sS, sN, state_tensor):
    num_site = sS.shape[0]
    num_branch = len(list(g['tree'].traverse()))
    g['is_site_nonmissing'] = np.zeros(shape=[num_branch, num_site], dtype=bool)
    for node in g['tree'].traverse():
        nl = ete.get_prop(node, "numerical_label")
        g['is_site_nonmissing'][nl,:] = (state_tensor[nl,:,:].sum(axis=1)!=0)
    g['sub_sites'] = dict()
    g['sub_sites'][g['asrv']] = np.zeros(shape=[num_branch, num_site], dtype=g['float_type'])
    if (g['asrv']=='no'):
        sub_sites = np.ones(shape=[num_site,]) / num_site
    elif (g['asrv']=='pool'):
        sub_sites = sS['S_sub'].values + sN['N_sub'].values
    elif (g['asrv']=='file'):
        sub_sites = g['iqtree_rate_values']
    if (g['asrv']=='sn'):
        for SN,df in zip(['S','N'],[sS,sN]):
            g['sub_sites'][SN] = np.zeros(shape=[num_branch, num_site], dtype=g['float_type'])
            sub_sites = df[SN+'_sub'].values
            for node in g['tree'].traverse():
                nl = ete.get_prop(node, "numerical_label")
                adjusted_sub_sites = sub_sites * g['is_site_nonmissing'][nl,:]
                total_sub_sites = adjusted_sub_sites.sum()
                total_sub_sites = 1 if (total_sub_sites==0) else total_sub_sites
                adjusted_sub_sites = adjusted_sub_sites/total_sub_sites
                g['sub_sites'][SN][nl,:] = adjusted_sub_sites
    elif (g['asrv']!='each'): # if 'each', Defined later in get_each_sub_sites()
        for node in g['tree'].traverse():
            nl = ete.get_prop(node, "numerical_label")
            is_site_nonmissing = (state_tensor[nl,:,:].sum(axis=1)!=0)
            adjusted_sub_sites = sub_sites * is_site_nonmissing
            total_sub_sites = adjusted_sub_sites.sum()
            total_sub_sites = 1 if (total_sub_sites==0) else total_sub_sites
            adjusted_sub_sites = adjusted_sub_sites/total_sub_sites
            g['sub_sites'][g['asrv']][nl,:] = adjusted_sub_sites
    return g

def get_each_sub_sites(sub_sg, mode, sg, a, d, g): # sub_sites for each "sg" group
    sub_sites = np.zeros(shape=g['is_site_nonmissing'].shape, dtype=g['float_type'])
    if mode == 'spe2spe':
        nonadjusted_sub_sites = sub_sg[:, sg, a, d]
    elif mode == 'spe2any':
        nonadjusted_sub_sites = sub_sg[:, sg, a]
    elif mode == 'any2spe':
        nonadjusted_sub_sites = sub_sg[:, sg, d]
    elif mode == 'any2any':
        nonadjusted_sub_sites = sub_sg[:, sg]
    for node in g['tree'].traverse():
        nl = ete.get_prop(node, "numerical_label")
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
    for a in np.arange(g['current_arity']):
        b_tmp = b.loc[:,['branch_id','S_sub','N_sub']]
        b_tmp.columns = [ c+'_'+str(a+1) for c in b_tmp.columns ]
        cb = pd.merge(cb, b_tmp, on='branch_id_'+str(a+1), how='left')
        del b_tmp
    return(cb)

def add_dif_column(cb, col_dif, col_any, col_spe, tol):
    if ((col_any in cb.columns) & (col_spe in cb.columns)):
        cb.loc[:, col_dif] = cb[col_any] - cb[col_spe]
        is_negative = (cb[col_dif] < -tol)
        is_almost_zero = (~is_negative)&(cb[col_dif] < tol)
        cb.loc[is_negative, col_dif] = np.nan
        cb.loc[is_almost_zero, col_dif] = 0
    return cb

def add_dif_stats(cb, tol, prefix, output_stats=None):
    if output_stats is None:
        requested = set([
            'any2any','spe2any','any2spe','spe2spe',
            'any2dif','dif2any','dif2spe','spe2dif','dif2dif',
        ])
    else:
        requested = set(output_stat.get_required_dif_stats(output_stats))
    for SN in ['S','N']:
        for anc in ['any','spe']:
            stat = anc + '2dif'
            if stat in requested:
                col_any = prefix+SN+anc+'2any'
                col_spe = prefix+SN+anc+'2spe'
                col_dif = prefix+SN+anc+'2dif'
                cb = add_dif_column(cb, col_dif, col_any, col_spe, tol)
        for des in ['any','spe','dif']:
            stat = 'dif2' + des
            if stat in requested:
                col_any = prefix+SN+'any2'+des
                col_spe = prefix+SN+'spe2'+des
                col_dif = prefix+SN+'dif2'+des
                cb = add_dif_column(cb, col_dif, col_any, col_spe, tol)
    return cb
