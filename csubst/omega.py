#if __name__ == '__main__':
#    mp.set_start_method('spawn')
#    my_class = MyClass(1)
#    my_class.mp_simple_method()
#    my_class.wait()

import joblib
import numpy
import scipy.stats as stats
from scipy.linalg import expm

import itertools
import os
import sys
import time

from csubst import parallel
from csubst import substitution
from csubst import table
from csubst import omega_cy

def calc_E_mean(mode, cb, sub_sg, sub_bg, obs_col, list_igad, g):
    E_b = numpy.zeros_like(cb.index, dtype=g['float_type'])
    bid_columns = cb.columns[cb.columns.str.startswith('branch_id_')]
    for i,sg,a,d in list_igad:
        if (a==d):
            continue
        if (g['asrv']=='each'):
            sub_sites = substitution.get_each_sub_sites(sub_sg, mode, sg, a, d, g)
        elif (g['asrv']=='sn'):
            if (obs_col.startswith('OCS')):
                sub_sites = g['sub_sites']['S']
            elif (obs_col.startswith('OCN')):
                sub_sites = g['sub_sites']['N']
        else:
            sub_sites = g['sub_sites'][g['asrv']]
        sub_branches = substitution.get_sub_branches(sub_bg, mode, sg, a, d)
        tmp_E = numpy.ones(shape=(E_b.shape[0], sub_sites.shape[1]), dtype=g['float_type'])
        for bid in numpy.unique(cb.loc[:,bid_columns].values):
            is_b = False
            for bc in bid_columns:
                is_b = (is_b)|(cb.loc[:,bc]==bid)
            tmp_E[is_b,:] *= sub_sites[bid,:] * sub_branches[bid]
        E_b += tmp_E.sum(axis=1)
    return E_b

def joblib_calc_E_mean(mode, cb, sub_sg, sub_bg, dfEb, obs_col, num_gad_combinat, igad_chunk, g):
    iter_start = time.time()
    bid_columns = cb.columns[cb.columns.str.startswith('branch_id_')]
    if (igad_chunk==[]):
        return None # This happens when the number of iteration is smaller than --threads
    i_start = igad_chunk[0][0]
    for i,sg,a,d in igad_chunk:
        if (a==d):
            continue
        if (g['asrv']=='each'):
            sub_sites = substitution.get_each_sub_sites(sub_sg, mode, sg, a, d, g)
        elif (g['asrv']=='sn'):
            if (obs_col.startswith('OCS')):
                sub_sites = g['sub_sites']['S']
            elif (obs_col.startswith('OCN')):
                sub_sites = g['sub_sites']['N']
        else:
            sub_sites = g['sub_sites'][g['asrv']]
        sub_branches = substitution.get_sub_branches(sub_bg, mode, sg, a, d)
        tmp_E = numpy.ones(shape=(dfEb.shape[0], sub_sites.shape[1]), dtype=g['float_type'])
        for bid in numpy.unique(cb.loc[:,bid_columns].values):
            is_b = False
            for bc in bid_columns:
                is_b = (is_b)|(cb.loc[:,bc]==bid)
            tmp_E[is_b,:] *= sub_sites[bid,:] * sub_branches[bid]
        dfEb += tmp_E.sum(axis=1)
    txt = 'E{}: {}-{}th of {} matrix_group/ancestral_state/derived_state combinations. Time elapsed: {:,} [sec]'
    print(txt.format(obs_col, i_start, i, num_gad_combinat, int(time.time()-iter_start)), flush=True)

def joblib_calc_quantile(mode, cb, sub_sg, sub_bg, dfq, quantile_niter, obs_col, num_gad_combinat, igad_chunk, g):
    bid_columns = cb.columns[cb.columns.str.startswith('branch_id_')]
    for i,sg,a,d in igad_chunk:
        if (a==d):
            continue
        if (g['asrv']=='each'):
            sub_sites = substitution.get_each_sub_sites(sub_sg, mode, sg, a, d, g)
        elif (g['asrv']=='sn'):
                if (obs_col.startswith('OCS')):
                    sub_sites = g['sub_sites']['S']
                elif (obs_col.startswith('OCN')):
                    sub_sites = g['sub_sites']['N']
        else:
            sub_sites = g['sub_sites'][g['asrv']]
        sub_branches = substitution.get_sub_branches(sub_bg, mode, sg, a, d)
        p = sub_sites[0]
        if p.sum()==0:
            continue
        pm_start = time.time()
        array_site = numpy.arange(p.shape[0])
        cb_ids = cb.loc[:,bid_columns].values
        if 'float' in str(sub_branches.dtype):
            # TODO: warn this rounding (only once)
            sub_branches = sub_branches.round().astype(numpy.int64)
        dfq[:,:] += omega_cy.get_permutations(cb_ids, array_site, sub_branches, p, quantile_niter)
        txt = '{}: {}/{} matrix_group/ancestral_state/derived_state combinations. Time elapsed for {:,} permutation: {:,} [sec]'
        print(txt.format(obs_col, i+1, num_gad_combinat, quantile_niter, int(time.time()-pm_start)), flush=True)

def calc_E_stat(cb, sub_tensor, mode, stat='mean', quantile_niter=1000, SN='', g={}):
    if mode=='spe2spe':
        sub_bg = sub_tensor.sum(axis=1) # branch, matrix_group, ancestral_state, derived_state
        sub_sg = sub_tensor.sum(axis=0) # site, matrix_group, ancestral_state, derived_state
        list_gad = [ [g,a,d] for g,a,d in itertools.zip_longest(*g[SN+'_ind_nomissing_gad']) ]
    elif mode=='spe2any':
        sub_bg = sub_tensor.sum(axis=(1, 4)) # branch, matrix_group, ancestral_state
        sub_sg = sub_tensor.sum(axis=(0, 4)) # site, matrix_group, ancestral_state
        list_gad = [ [g,a,'2any'] for g,a in itertools.zip_longest(*g[SN+'_ind_nomissing_ga']) ]
    elif mode=='any2spe':
        sub_bg = sub_tensor.sum(axis=(1, 3)) # branch, matrix_group, derived_state
        sub_sg = sub_tensor.sum(axis=(0, 3)) # site, matrix_group, derived_state
        list_gad = [ [g,'any2',d] for g,d in itertools.zip_longest(*g[SN+'_ind_nomissing_gd']) ]
    elif mode=='any2any':
        sub_bg = sub_tensor.sum(axis=(1, 3, 4)) # branch, matrix_group
        sub_sg = sub_tensor.sum(axis=(0, 3, 4)) # site, matrix_group
        list_gad = list(itertools.product(numpy.arange(sub_tensor.shape[2]), ['any2',], ['2any',]))
    num_gad_combinat = len(list_gad)
    txt = 'E{}{}: Total number of substitution categories after NaN removals: {}'
    print(txt.format(SN, mode, num_gad_combinat))
    list_igad = [ [i,]+list(items) for i,items in zip(range(num_gad_combinat), list_gad) ]
    obs_col = 'OC'+SN+mode
    if (g['threads']>1):
        igad_chunks,mmap_start_not_necessary_here = parallel.get_chunks(list_igad, g['threads'])
    if stat=='mean':
        if (g['threads']==1):
            E_b = calc_E_mean(mode, cb, sub_sg, sub_bg, obs_col, list_igad, g)
        else:
            my_dtype = sub_tensor.dtype
            if 'bool' in str(my_dtype): my_dtype = g['float_type']
            mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.dfEb.mmap')
            if os.path.exists(mmap_out): os.unlink(mmap_out)
            dfEb = numpy.memmap(filename=mmap_out, dtype=my_dtype, shape=(cb.shape[0]), mode='w+')
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1, user_api='blas'):
                joblib.Parallel(n_jobs=g['threads'], max_nbytes=None, backend='multiprocessing')(
                    joblib.delayed(joblib_calc_E_mean)
                    (mode, cb, sub_sg, sub_bg, dfEb, obs_col, num_gad_combinat, igad_chunk, g) for igad_chunk in igad_chunks
                )
            E_b = dfEb
            if os.path.exists(mmap_out): os.unlink(mmap_out)
    elif stat=='quantile':
        mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.dfq.mmap')
        if os.path.exists(mmap_out): os.unlink(mmap_out)
        my_dtype = sub_tensor.dtype
        if 'bool' in str(my_dtype): my_dtype = numpy.int32
        dfq = numpy.memmap(filename=mmap_out, dtype=my_dtype, shape=(cb.shape[0], quantile_niter), mode='w+')
        # TODO distinct thread/process
        joblib.Parallel(n_jobs=g['threads'], max_nbytes=None, backend='multiprocessing')(
            joblib.delayed(joblib_calc_quantile)
            (mode, cb, sub_sg, sub_bg, dfq, quantile_niter, obs_col, num_gad_combinat, igad_chunk, g) for igad_chunk in igad_chunks
        )
        if os.path.exists(mmap_out): os.unlink(mmap_out)
        E_b = numpy.zeros_like(cb.index, dtype=g['float_type'])
        for i in cb.index:
            # num_gad_combinat: poisson approximation
            obs_value = cb.loc[i,obs_col]
            gt_rank = (dfq[i,:]<obs_value).sum()
            ge_rank = (dfq[i,:]<=obs_value).sum()
            corrected_rank = (gt_rank+ge_rank)/2
            E_b[i] = corrected_rank / quantile_niter
    return E_b

def subroot_E2nan(cb, tree):
    id_cols = cb.columns[cb.columns.str.startswith('branch_id_')]
    E_cols = cb.columns[cb.columns.str.startswith('E')]
    if (E_cols.shape[0]==0):
        return cb
    for node in tree.traverse():
        continue_flag = 1
        if node.is_root():
            continue_flag = 0
        elif node.up.is_root():
            continue_flag = 0
        if continue_flag:
            continue
        for id_col in id_cols:
            is_node = (cb.loc[:,id_col]==node.numerical_label)
            cb.loc[is_node,E_cols] = numpy.nan
    return cb

def get_E(cb, g, N_tensor, S_tensor):
    if (g['omegaC_method']=='modelfree'):
        g['N_ind_nomissing_gad'] = numpy.where(N_tensor.sum(axis=(0,1))!=0)
        g['N_ind_nomissing_ga'] = numpy.where(N_tensor.sum(axis=(0,1,4))!=0)
        g['N_ind_nomissing_gd'] = numpy.where(N_tensor.sum(axis=(0,1,3))!=0)
        g['S_ind_nomissing_gad'] = numpy.where(S_tensor.sum(axis=(0,1))!=0)
        g['S_ind_nomissing_ga'] = numpy.where(S_tensor.sum(axis=(0,1,4))!=0)
        g['S_ind_nomissing_gd'] = numpy.where(S_tensor.sum(axis=(0,1,3))!=0)
        for st in ['any2any','any2spe','spe2any','spe2spe']:
            cb['ECN'+st] = calc_E_stat(cb, N_tensor, mode=st, stat='mean', SN='N', g=g)
            cb['ECS'+st] = calc_E_stat(cb, S_tensor, mode=st, stat='mean', SN='S', g=g)
    if (g['omegaC_method']=='submodel'):
        id_cols = cb.columns[cb.columns.str.startswith('branch_id_')]
        state_pepE = get_exp_state(g=g, mode='pep')
        if (g['current_arity']==2):
            g['ECN_tensor'] = substitution.get_substitution_tensor(state_pepE, g['state_pep'], mode='asis', g=g, mmap_attr='EN')
        txt = 'Number of total empirically expected nonsynonymous substitutions in the tree: {:,.2f}'
        print(txt.format(g['ECN_tensor'].sum()))
        print('Preparing the ECN table with {:,} process(es).'.format(g['threads']), flush=True)
        cbEN = substitution.get_cb(cb.loc[:,id_cols].values, g['ECN_tensor'], g, 'ECN')
        cb = table.merge_tables(cb, cbEN)
        del state_pepE,cbEN
        state_cdnE = get_exp_state(g=g, mode='cdn')
        if (g['current_arity'] == 2):
            g['ECS_tensor'] = substitution.get_substitution_tensor(state_cdnE, g['state_cdn'], mode='syn', g=g, mmap_attr='ES')
        txt = 'Number of total empirically expected synonymous substitutions in the tree: {:,.2f}'
        print(txt.format(g['ECS_tensor'].sum()))
        print('Preparing the ECS table with {:,} process(es).'.format(g['threads']), flush=True)
        cbES = substitution.get_cb(cb.loc[:,id_cols].values, g['ECS_tensor'], g, 'ECS')
        cb = table.merge_tables(cb, cbES)
        del state_cdnE,cbES
    if g['calc_quantile']:
        for st in ['any2any','any2spe','spe2any','spe2spe']:
            cb['QCN'+st] = calc_E_stat(cb, N_tensor, mode=st, stat='quantile', SN='N', g=g)
            cb['QCS'+st] = calc_E_stat(cb, S_tensor, mode=st, stat='quantile', SN='S', g=g)
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
        if node.is_root():
            continue
        if mode=='cdn':
            branch_length = node.SNdist
        elif mode=='pep':
            branch_length = node.Ndist
        branch_length = max(branch_length, 0)
        if branch_length<g['float_tol']:
            continue # Skip if no substitution
        nl = node.numerical_label
        parent_nl = node.up.numerical_label
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

def calc_omega(cb, S_tensor, N_tensor, g):
    cb = get_E(cb, g, N_tensor, S_tensor)
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
        dNc_values = cb.loc[:,col_dNc].replace([numpy.inf, -numpy.inf], numpy.nan)
        if (dNc_values.shape[0]==0):
            sys.stderr.write('dSc calibration could not be applied: {}\n'.format(sub))
            continue
        cb.columns = cb.columns.str.replace(col_dSc, col_noncalibrated_dSc)
        cb.columns = cb.columns.str.replace(col_omega, col_noncalibrated_omega)
        uncorrected_dSc_values = cb.loc[:,col_noncalibrated_dSc].replace([numpy.inf, -numpy.inf], numpy.nan)
        is_na = (uncorrected_dSc_values.isnull() | dNc_values.isnull())
        if (is_na.sum()>0):
            txt = 'dSc calibration could not be applied to {:,}/{:,} branch combinations for {}\n'
            sys.stderr.write(txt.format(is_na.sum(), cb.shape[0], sub))
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
