#if __name__ == '__main__':
#    mp.set_start_method('spawn')
#    my_class = MyClass(1)
#    my_class.mp_simple_method()
#    my_class.wait()

import numpy
import scipy.stats as stats
from scipy.linalg import expm

import itertools
import os
import sys
import time

from csubst import parallel
from csubst import substitution
from csubst import substitution_sparse
from csubst import table
from csubst import ete

_UINT8_POPCOUNT = numpy.unpackbits(
    numpy.arange(256, dtype=numpy.uint8)[:, None],
    axis=1,
).sum(axis=1).astype(numpy.uint8)


def _get_cb_ids(cb):
    bid_columns = cb.columns[cb.columns.str.startswith('branch_id_')]
    cb_ids = cb.loc[:, bid_columns].values
    if not numpy.issubdtype(cb_ids.dtype, numpy.integer):
        cb_ids = cb_ids.astype(numpy.int64)
    return cb_ids


def _resolve_sub_sites(g, sub_sg, mode, sg, a, d, obs_col):
    if (g['asrv']=='each'):
        return substitution.get_each_sub_sites(sub_sg, mode, sg, a, d, g)
    if (g['asrv']=='sn'):
        if (obs_col.startswith('OCS')):
            return g['sub_sites']['S']
        if (obs_col.startswith('OCN')):
            return g['sub_sites']['N']
    return g['sub_sites'][g['asrv']]


def _calc_tmp_E_sum(cb_ids, sub_sites, sub_branches, float_type):
    if (cb_ids.shape[1] == 1):
        bids = cb_ids[:, 0]
        tmp_E = sub_sites[bids, :] * sub_branches[bids, None]
        return tmp_E.sum(axis=1)
    if (cb_ids.shape[1] == 2):
        bid1 = cb_ids[:, 0]
        bid2 = cb_ids[:, 1]
        tmp_E = sub_sites[bid1, :] * sub_branches[bid1, None]
        tmp_E *= sub_sites[bid2, :] * sub_branches[bid2, None]
        return tmp_E.sum(axis=1)
    tmp_E = numpy.ones(shape=(cb_ids.shape[0], sub_sites.shape[1]), dtype=float_type)
    for col in range(cb_ids.shape[1]):
        bids = cb_ids[:, col]
        tmp_E *= sub_sites[bids, :] * sub_branches[bids, None]
    return tmp_E.sum(axis=1)


def _weighted_sample_without_replacement_masks(p, size, niter):
    p = numpy.asarray(p, dtype=numpy.float64)
    if p.ndim != 1:
        raise ValueError('p should be a 1D array.')
    if size < 0:
        raise ValueError('size should be >= 0.')
    positive_sites = numpy.flatnonzero(p > 0)
    num_positive_sites = positive_sites.shape[0]
    masks = numpy.zeros(shape=(niter, p.shape[0]), dtype=bool)
    if (size == 0) or (num_positive_sites == 0):
        return masks
    if size > num_positive_sites:
        txt = 'Sample size ({}) exceeded number of positive-probability sites ({}) in quantile sampling.'
        raise ValueError(txt.format(size, num_positive_sites))
    if size == num_positive_sites:
        masks[:, positive_sites] = True
        return masks
    # Efraimidis-Spirakis weighted sampling without replacement (A-ES).
    positive_weights = p[positive_sites].astype(numpy.float32, copy=False)
    keys = numpy.random.random((niter, num_positive_sites)).astype(numpy.float32, copy=False)
    numpy.log(keys, out=keys)
    keys /= positive_weights
    kth = num_positive_sites - size
    sampled_local_indices = numpy.argpartition(keys, kth=kth, axis=1)[:, kth:]
    sampled_site_indices = positive_sites[sampled_local_indices]
    row_indices = numpy.arange(niter)[:, None]
    masks[row_indices, sampled_site_indices] = True
    return masks


def _get_permutations_fast(cb_ids, sub_branches, p, niter):
    cb_ids = numpy.asarray(cb_ids, dtype=numpy.int64)
    if cb_ids.ndim != 2:
        raise ValueError('cb_ids should be a 2D array.')
    if cb_ids.shape[0] == 0:
        return numpy.zeros((0, niter), dtype=numpy.int32)
    sub_branches = numpy.asarray(sub_branches, dtype=numpy.int64)
    if sub_branches.ndim != 1:
        raise ValueError('sub_branches should be a 1D array.')
    if sub_branches.shape[0] <= cb_ids.max():
        raise ValueError('cb_ids contain out-of-range branch IDs.')
    cb_branch_sizes = sub_branches[cb_ids]
    is_active_row = (cb_branch_sizes > 0).all(axis=1)
    if not is_active_row.any():
        return numpy.zeros((cb_ids.shape[0], niter), dtype=numpy.int32)
    active_row_indices = numpy.where(is_active_row)[0]
    active_cb_ids = cb_ids[active_row_indices, :]
    active_branch_ids, inverse_branch_ids = numpy.unique(active_cb_ids, return_inverse=True)
    remapped_active_cb_ids = inverse_branch_ids.reshape(active_cb_ids.shape)
    active_sub_branches = sub_branches[active_branch_ids]

    num_branch = active_sub_branches.shape[0]
    num_site = p.shape[0]
    num_packed_site = (num_site + 7) // 8
    packed_masks = numpy.zeros(shape=(num_branch, niter, num_packed_site), dtype=numpy.uint8)
    previous_branch_id_by_size = dict()
    for branch_id in range(num_branch):
        size = int(active_sub_branches[branch_id])
        if size == 0:
            continue
        if size in previous_branch_id_by_size:
            prev_branch_id = previous_branch_id_by_size[size]
            packed_masks[branch_id, :, :] = packed_masks[prev_branch_id, numpy.random.permutation(niter), :]
            continue
        previous_branch_id_by_size[size] = branch_id
        masks = _weighted_sample_without_replacement_masks(p=p, size=size, niter=niter)
        packed_masks[branch_id, :, :] = numpy.packbits(masks, axis=1)

    arity = remapped_active_cb_ids.shape[1]
    if arity == 1:
        active_out = _UINT8_POPCOUNT[packed_masks[remapped_active_cb_ids[:, 0], :, :]].sum(axis=2, dtype=numpy.int32)
    elif arity == 2:
        active_out = _UINT8_POPCOUNT[numpy.bitwise_and(
            packed_masks[remapped_active_cb_ids[:, 0], :, :],
            packed_masks[remapped_active_cb_ids[:, 1], :, :],
        )].sum(axis=2, dtype=numpy.int32)
    else:
        shared = packed_masks[remapped_active_cb_ids[:, 0], :, :].copy()
        for col in range(1, arity):
            shared = numpy.bitwise_and(shared, packed_masks[remapped_active_cb_ids[:, col], :, :])
        active_out = _UINT8_POPCOUNT[shared].sum(axis=2, dtype=numpy.int32)
    out = numpy.zeros((cb_ids.shape[0], niter), dtype=numpy.int32)
    out[active_row_indices, :] = active_out
    return out


def _resolve_quantile_parallel_plan(cb_rows, num_categories, quantile_niter, requested_n_jobs, requested_chunk_factor):
    n_jobs = int(requested_n_jobs)
    chunk_factor = int(requested_chunk_factor)
    if n_jobs <= 1:
        return 1, max(chunk_factor, 1)
    # For small branch-set workloads, thread startup and local-buffer reduction dominate.
    # Fall back to single-thread execution to avoid regressions under --threads > 1.
    workload = int(cb_rows) * int(num_categories) * int(quantile_niter)
    if (cb_rows <= 4) or (workload < 20_000_000):
        return 1, max(chunk_factor, 1)
    return n_jobs, max(chunk_factor, 4)


def calc_E_mean(mode, cb_ids, sub_sg, sub_bg, obs_col, list_igad, g):
    E_b = numpy.zeros(shape=(cb_ids.shape[0],), dtype=g['float_type'])
    for i,sg,a,d in list_igad:
        if (a==d):
            continue
        sub_sites = _resolve_sub_sites(g=g, sub_sg=sub_sg, mode=mode, sg=sg, a=a, d=d, obs_col=obs_col)
        sub_branches = substitution.get_sub_branches(sub_bg, mode, sg, a, d)
        E_b += _calc_tmp_E_sum(
            cb_ids=cb_ids,
            sub_sites=sub_sites,
            sub_branches=sub_branches,
            float_type=g['float_type'],
        )
    return E_b


def joblib_calc_E_mean(mode, cb_ids, sub_sg, sub_bg, dfEb, obs_col, num_gad_combinat, igad_chunk, g):
    iter_start = time.time()
    if (igad_chunk==[]):
        return None # This happens when the number of iteration is smaller than --threads
    i_start = igad_chunk[0][0]
    for i,sg,a,d in igad_chunk:
        if (a==d):
            continue
        sub_sites = _resolve_sub_sites(g=g, sub_sg=sub_sg, mode=mode, sg=sg, a=a, d=d, obs_col=obs_col)
        sub_branches = substitution.get_sub_branches(sub_bg, mode, sg, a, d)
        dfEb += _calc_tmp_E_sum(
            cb_ids=cb_ids,
            sub_sites=sub_sites,
            sub_branches=sub_branches,
            float_type=g['float_type'],
        )
    txt = 'E{}: {}-{}th of {} matrix_group/ancestral_state/derived_state combinations. Time elapsed: {:,} [sec]'
    print(txt.format(obs_col, i_start, i, num_gad_combinat, int(time.time()-iter_start)), flush=True)


def joblib_calc_quantile(mode, cb_ids, sub_sg, sub_bg, dfq, quantile_niter, obs_col, num_gad_combinat, igad_chunk, g):
    for i,sg,a,d in igad_chunk:
        if (a==d):
            continue
        sub_sites = _resolve_sub_sites(g=g, sub_sg=sub_sg, mode=mode, sg=sg, a=a, d=d, obs_col=obs_col)
        sub_branches = substitution.get_sub_branches(sub_bg, mode, sg, a, d)
        p = sub_sites[0]
        if p.sum()==0:
            continue
        pm_start = time.time()
        if 'float' in str(sub_branches.dtype):
            # TODO: warn this rounding (only once)
            sub_branches = sub_branches.round().astype(numpy.int64)
        dfq[:,:] += _get_permutations_fast(cb_ids, sub_branches, p, quantile_niter)
        txt = '{}: {}/{} matrix_group/ancestral_state/derived_state combinations. Time elapsed for {:,} permutation: {:,} [sec]'
        print(txt.format(obs_col, i+1, num_gad_combinat, quantile_niter, int(time.time()-pm_start)), flush=True)

def _calc_E_mean_chunk_to_mmap(mode, cb_ids, sub_sg, sub_bg, mmap_out, dtype, shape, obs_col, num_gad_combinat, igad_chunk, g):
    dfEb = numpy.memmap(filename=mmap_out, dtype=dtype, shape=shape, mode='r+')
    joblib_calc_E_mean(mode, cb_ids, sub_sg, sub_bg, dfEb, obs_col, num_gad_combinat, igad_chunk, g)
    dfEb.flush()

def _calc_quantile_chunk_local(mode, cb_ids, sub_sg, sub_bg, quantile_niter, obs_col, num_gad_combinat, igad_chunk, g):
    dfq_local = numpy.zeros(shape=(cb_ids.shape[0], quantile_niter), dtype=numpy.int32)
    joblib_calc_quantile(mode, cb_ids, sub_sg, sub_bg, dfq_local, quantile_niter, obs_col, num_gad_combinat, igad_chunk, g)
    return dfq_local

def calc_E_stat(cb, sub_tensor, mode, stat='mean', quantile_niter=1000, SN='', g={}):
    if isinstance(sub_tensor, substitution_sparse.SparseSubstitutionTensor):
        sub_bg, sub_sg = substitution_sparse.summarize_sparse_sub_tensor(sparse_tensor=sub_tensor, mode=mode)
    if mode=='spe2spe':
        if not isinstance(sub_tensor, substitution_sparse.SparseSubstitutionTensor):
            sub_bg = sub_tensor.sum(axis=1) # branch, matrix_group, ancestral_state, derived_state
            sub_sg = sub_tensor.sum(axis=0) # site, matrix_group, ancestral_state, derived_state
        list_gad = [ [g,a,d] for g,a,d in itertools.zip_longest(*g[SN+'_ind_nomissing_gad']) ]
    elif mode=='spe2any':
        if not isinstance(sub_tensor, substitution_sparse.SparseSubstitutionTensor):
            sub_bg = sub_tensor.sum(axis=(1, 4)) # branch, matrix_group, ancestral_state
            sub_sg = sub_tensor.sum(axis=(0, 4)) # site, matrix_group, ancestral_state
        list_gad = [ [g,a,'2any'] for g,a in itertools.zip_longest(*g[SN+'_ind_nomissing_ga']) ]
    elif mode=='any2spe':
        if not isinstance(sub_tensor, substitution_sparse.SparseSubstitutionTensor):
            sub_bg = sub_tensor.sum(axis=(1, 3)) # branch, matrix_group, derived_state
            sub_sg = sub_tensor.sum(axis=(0, 3)) # site, matrix_group, derived_state
        list_gad = [ [g,'any2',d] for g,d in itertools.zip_longest(*g[SN+'_ind_nomissing_gd']) ]
    elif mode=='any2any':
        if not isinstance(sub_tensor, substitution_sparse.SparseSubstitutionTensor):
            sub_bg = sub_tensor.sum(axis=(1, 3, 4)) # branch, matrix_group
            sub_sg = sub_tensor.sum(axis=(0, 3, 4)) # site, matrix_group
        list_gad = list(itertools.product(numpy.arange(sub_tensor.shape[2]), ['any2',], ['2any',]))
    num_gad_combinat = len(list_gad)
    txt = 'E{}{}: Total number of substitution categories after NaN removals: {}'
    print(txt.format(SN, mode, num_gad_combinat))
    list_igad = [ [i,]+list(items) for i,items in zip(range(num_gad_combinat), list_gad) ]
    obs_col = 'OC'+SN+mode
    cb_ids = _get_cb_ids(cb)
    requested_n_jobs = parallel.resolve_n_jobs(num_items=len(list_igad), threads=g['threads'])
    chunk_factor = parallel.resolve_chunk_factor(g=g, task='general')
    n_jobs = requested_n_jobs
    if stat == 'quantile':
        n_jobs, chunk_factor = _resolve_quantile_parallel_plan(
            cb_rows=cb.shape[0],
            num_categories=len(list_igad),
            quantile_niter=quantile_niter,
            requested_n_jobs=requested_n_jobs,
            requested_chunk_factor=chunk_factor,
        )
    igad_chunks,mmap_start_not_necessary_here = parallel.get_chunks(list_igad, n_jobs, chunk_factor=chunk_factor)
    if stat=='mean':
        if n_jobs == 1:
            E_b = calc_E_mean(mode, cb_ids, sub_sg, sub_bg, obs_col, list_igad, g)
        else:
            my_dtype = sub_tensor.dtype
            if 'bool' in str(my_dtype): my_dtype = g['float_type']
            mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.dfEb.mmap')
            if os.path.exists(mmap_out): os.unlink(mmap_out)
            axis = (cb.shape[0],)
            dfEb = numpy.memmap(filename=mmap_out, dtype=my_dtype, shape=axis, mode='w+')
            tasks = [
                (mode, cb_ids, sub_sg, sub_bg, mmap_out, my_dtype, axis, obs_col, num_gad_combinat, igad_chunk, g)
                for igad_chunk in igad_chunks
            ]
            parallel.run_starmap(
                func=_calc_E_mean_chunk_to_mmap,
                args_iterable=tasks,
                n_jobs=n_jobs,
                backend='threading',
            )
            dfEb.flush()
            E_b = dfEb
            del dfEb
            if os.path.exists(mmap_out): os.unlink(mmap_out)
    elif stat=='quantile':
        axis = (cb.shape[0], quantile_niter)
        dfq = numpy.zeros(shape=axis, dtype=numpy.int32)
        if n_jobs == 1:
            joblib_calc_quantile(mode, cb_ids, sub_sg, sub_bg, dfq, quantile_niter, obs_col, num_gad_combinat, list_igad, g)
        else:
            tasks = [
                (mode, cb_ids, sub_sg, sub_bg, quantile_niter, obs_col, num_gad_combinat, igad_chunk, g)
                for igad_chunk in igad_chunks
            ]
            chunk_dfs = parallel.run_starmap(
                func=_calc_quantile_chunk_local,
                args_iterable=tasks,
                n_jobs=n_jobs,
                backend='threading',
            )
            for dfq_chunk in chunk_dfs:
                dfq += dfq_chunk
        # num_gad_combinat: poisson approximation
        obs_values = cb.loc[:,obs_col].values.astype(g['float_type'], copy=False)
        gt_ranks = (dfq < obs_values[:, None]).sum(axis=1, dtype=numpy.int64)
        ge_ranks = (dfq <= obs_values[:, None]).sum(axis=1, dtype=numpy.int64)
        E_b = ((gt_ranks + ge_ranks) / 2) / quantile_niter
    return E_b

def subroot_E2nan(cb, tree):
    id_cols = cb.columns[cb.columns.str.startswith('branch_id_')]
    E_cols = cb.columns[cb.columns.str.startswith('E')]
    if (E_cols.shape[0]==0):
        return cb
    for node in tree.traverse():
        continue_flag = 1
        if ete.is_root(node):
            continue_flag = 0
        elif ete.is_root(node.up):
            continue_flag = 0
        if continue_flag:
            continue
        for id_col in id_cols:
            is_node = (cb.loc[:,id_col]==ete.get_prop(node, "numerical_label"))
            cb.loc[is_node,E_cols] = numpy.nan
    return cb

def get_E(cb, g, ON_tensor, OS_tensor):
    if (g['omegaC_method']=='modelfree'):
        ON_gad, ON_ga, ON_gd = substitution.get_group_state_totals(ON_tensor)
        OS_gad, OS_ga, OS_gd = substitution.get_group_state_totals(OS_tensor)
        g['N_ind_nomissing_gad'] = numpy.where(ON_gad!=0)
        g['N_ind_nomissing_ga'] = numpy.where(ON_ga!=0)
        g['N_ind_nomissing_gd'] = numpy.where(ON_gd!=0)
        g['S_ind_nomissing_gad'] = numpy.where(OS_gad!=0)
        g['S_ind_nomissing_ga'] = numpy.where(OS_ga!=0)
        g['S_ind_nomissing_gd'] = numpy.where(OS_gd!=0)
        for st in ['any2any','any2spe','spe2any','spe2spe']:
            cb['ECN'+st] = calc_E_stat(cb, ON_tensor, mode=st, stat='mean', SN='N', g=g)
            cb['ECS'+st] = calc_E_stat(cb, OS_tensor, mode=st, stat='mean', SN='S', g=g)
    if (g['omegaC_method']=='submodel'):
        id_cols = cb.columns[cb.columns.str.startswith('branch_id_')]
        state_pepE = get_exp_state(g=g, mode='pep')
        if (g['current_arity']==2):
            g['EN_tensor'] = substitution.get_substitution_tensor(state_pepE, g['state_pep'], mode='asis', g=g, mmap_attr='EN')
        txt = 'Number of total empirically expected nonsynonymous substitutions in the tree: {:,.2f}'
        print(txt.format(substitution.get_total_substitution(g['EN_tensor'])))
        print('Preparing the ECN table with {:,} process(es).'.format(g['threads']), flush=True)
        cbEN = substitution.get_cb(cb.loc[:,id_cols].values, g['EN_tensor'], g, 'ECN')
        cb = table.merge_tables(cb, cbEN)
        del state_pepE,cbEN
        state_cdnE = get_exp_state(g=g, mode='cdn')
        if (g['current_arity'] == 2):
            g['ES_tensor'] = substitution.get_substitution_tensor(state_cdnE, g['state_cdn'], mode='syn', g=g, mmap_attr='ES')
        txt = 'Number of total empirically expected synonymous substitutions in the tree: {:,.2f}'
        print(txt.format(substitution.get_total_substitution(g['ES_tensor'])))
        print('Preparing the ECS table with {:,} process(es).'.format(g['threads']), flush=True)
        cbES = substitution.get_cb(cb.loc[:,id_cols].values, g['ES_tensor'], g, 'ECS')
        cb = table.merge_tables(cb, cbES)
        del state_cdnE,cbES
    if g['calc_quantile']:
        for st in ['any2any','any2spe','spe2any','spe2spe']:
            cb['QCN'+st] = calc_E_stat(cb, ON_tensor, mode=st, stat='quantile', SN='N', g=g)
            cb['QCS'+st] = calc_E_stat(cb, OS_tensor, mode=st, stat='quantile', SN='S', g=g)
    cb = substitution.add_dif_stats(cb, g['float_tol'], prefix='EC')
    cb = subroot_E2nan(cb, tree=g['tree'])
    return cb

def get_exp_state(g, mode):
    if mode=='cdn':
        state = g['state_cdn'].astype(g['float_type'])
        inst = g['instantaneous_codon_rate_matrix']
    elif mode=='pep':
        state = g['state_pep'].astype(g['float_type'])
        inst = g['instantaneous_aa_rate_matrix']
    stateE = numpy.zeros_like(state, dtype=g['float_type'])
    for node in g['tree'].traverse():
        if ete.is_root(node):
            continue
        if mode=='cdn':
            branch_length = ete.get_prop(node, 'SNdist', 0)
        elif mode=='pep':
            branch_length = ete.get_prop(node, 'Ndist', 0)
        branch_length = max(branch_length, 0)
        if branch_length<g['float_tol']:
            continue # Skip if no substitution
        nl = ete.get_prop(node, "numerical_label")
        parent_nl = ete.get_prop(node.up, "numerical_label")
        if parent_nl>stateE.shape[0]:
            continue # Skip if parent is the root node
        inst_bl = inst * branch_length
        for site_rate in numpy.unique(g['iqtree_rate_values']):
            inst_bl_site = inst_bl * site_rate
            # Confirmed this implementation (with expm) correctly replicated the example in this instruction (Huelsenbeck, 2012)
            # https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
            transition_prob = expm(inst_bl_site)
            site_indices = numpy.where(g['iqtree_rate_values']==site_rate)[0]
            for s in site_indices:
                expected_transition_ad = numpy.einsum('a,ad->ad', state[parent_nl,s,:], transition_prob)
                if expected_transition_ad.sum()-1>g['float_tol']:
                    expected_transition_ad /= expected_transition_ad.sum()
                expected_derived_state = expected_transition_ad.sum(axis=0)
                stateE[nl,s,:] = expected_derived_state
    max_stateE = stateE.sum(axis=(2)).max()
    assert (max_stateE-1)<g['float_tol'], 'Total probability of expected states should not exceed 1. {}'.format(max_stateE)
    return stateE

def get_omega(cb, g):
    combinatorial_substitutions = ['any2any','any2spe','any2dif',
                                   'dif2any','dif2spe','dif2dif',
                                   'spe2any','spe2spe','spe2dif',
                                   ]
    for sub in combinatorial_substitutions:
        col_omega = 'omegaC'+sub
        col_N = 'OCN'+sub
        col_EN = 'ECN'+sub
        col_dNc = 'dNC'+sub
        col_S = 'OCS'+sub
        col_ES = 'ECS'+sub
        col_dSc = 'dSC'+sub
        if all([ col in cb.columns for col in [col_N,col_EN,col_S,col_ES] ]):
            cb.loc[:,col_dNc] = (cb.loc[:,col_N] / cb.loc[:,col_EN])
            is_N_zero = (cb.loc[:,col_N]<g['float_tol'])
            cb.loc[is_N_zero,col_dNc] = 0
            cb.loc[:,col_dSc] = (cb.loc[:,col_S] / cb.loc[:,col_ES])
            is_S_zero = (cb.loc[:,col_S]<g['float_tol'])
            cb.loc[is_S_zero,col_dSc] = 0
            cb.loc[:,col_omega] = cb.loc[:,col_dNc] / cb.loc[:,col_dSc]
            is_dN_zero = (cb.loc[:,col_dNc]<g['float_tol'])
            cb.loc[is_dN_zero,col_omega] = 0
    return cb

def get_CoD(cb, g):
    for NS in ['OCN','OCS']:
        cb.loc[:,NS+'CoD'] = cb[NS+'any2spe'] / cb[NS+'any2dif']
        is_Nzero = (cb[NS+'any2spe']<g['float_tol'])
        is_inf = numpy.isinf(cb.loc[:,NS+'CoD'])
        if (is_Nzero&is_inf).sum():
            cb.loc[(is_Nzero&is_inf), NS + 'CoD'] = numpy.nan
    return cb

def print_cb_stats(cb, prefix):
    arity = cb.columns.str.startswith('branch_id_').sum()
    hd = 'Arity = {:,}, {}:'.format(arity, prefix)
    combinatorial_substitutions = ['any2any','any2spe','any2dif',
                                   'dif2any','dif2spe','dif2dif',
                                   'spe2any','spe2spe','spe2dif',
                                   ]
    for sub in combinatorial_substitutions:
        col_omega = 'omegaC'+sub
        if not col_omega in cb.columns:
            continue
        median_value = cb.loc[:,col_omega].median()
        txt = '{} median {} (non-corrected for dNc vs dSc distribution ranges): {:.3f}'
        print(txt.format(hd, col_omega, median_value), flush=True)

def calc_omega(cb, OS_tensor, ON_tensor, g):
    cb = get_E(cb, g, ON_tensor, OS_tensor)
    cb = get_omega(cb, g)
    cb = get_CoD(cb, g)
    print_cb_stats(cb=cb, prefix='cb')
    return(cb, g)

def calibrate_dsc(cb, transformation='quantile'):
    prefix='cb'
    arity = cb.columns.str.startswith('branch_id_').sum()
    hd = 'Arity = {:,}, {}:'.format(arity, prefix)
    combinatorial_substitutions = ['any2any','any2spe','any2dif',
                                   'dif2any','dif2spe','dif2dif',
                                   'spe2any','spe2spe','spe2dif',
                                   ]
    for sub in combinatorial_substitutions:
        col_dNc = 'dNC'+sub
        col_dSc = 'dSC'+sub
        col_omega = 'omegaC'+sub
        col_noncalibrated_dSc = 'dSC'+sub+'_nocalib'
        col_noncalibrated_omega = 'omegaC'+sub+'_nocalib'
        if not all([col in cb.columns for col in [col_dNc, col_dSc, col_omega]]):
            continue
        dNc_values = cb.loc[:,col_dNc].replace([numpy.inf, -numpy.inf], numpy.nan)
        uncorrected_dSc_values = cb.loc[:,col_dSc].replace([numpy.inf, -numpy.inf], numpy.nan)
        is_na = (uncorrected_dSc_values.isnull() | dNc_values.isnull())
        if is_na.all():
            txt = 'dSc calibration could not be applied: {} (no finite dNc/dSc pairs)\n'
            sys.stderr.write(txt.format(sub))
            continue
        if (is_na.sum()>0):
            txt = 'dSc calibration could not be applied to {:,}/{:,} branch combinations for {}\n'
            sys.stderr.write(txt.format(is_na.sum(), cb.shape[0], sub))
        cb.columns = cb.columns.str.replace(col_dSc, col_noncalibrated_dSc, regex=False)
        cb.columns = cb.columns.str.replace(col_omega, col_noncalibrated_omega, regex=False)
        dNc_values_wo_na = dNc_values[~is_na]
        uncorrected_dSc_values_wo_na = uncorrected_dSc_values[~is_na]
        ranks = stats.rankdata(uncorrected_dSc_values_wo_na)
        quantiles = ranks / ranks.max()
        if (transformation=='gamma'):
            alpha,loc,beta = stats.gamma.fit(dNc_values_wo_na)
            cb.loc[~is_na,col_dSc] = stats.gamma.ppf(q=quantiles, a=alpha, loc=loc, scale=beta)
        elif (transformation=='quantile'):
            cb.loc[~is_na,col_dSc] = numpy.quantile(dNc_values_wo_na, quantiles)
        noncalibrated_dSc_values = cb.loc[:,col_noncalibrated_dSc].values
        is_nocalib_higher = (noncalibrated_dSc_values>cb.loc[:,col_dSc]).fillna(False)
        cb.loc[is_nocalib_higher,col_dSc] = noncalibrated_dSc_values[is_nocalib_higher]
        cb.loc[:,col_omega] = numpy.nan
        cb.loc[:,col_omega] = cb.loc[:,col_dNc] / cb.loc[:,col_dSc]
        median_value = cb.loc[:,col_omega].median()
        txt = '{} median {} ({:,}/{:,} branch combinations were corrected for dNc vs dSc distribution ranges): {:.3f}'
        print(txt.format(hd, col_omega, (~is_nocalib_higher).sum(), cb.shape[0], median_value), flush=True)
    return cb
