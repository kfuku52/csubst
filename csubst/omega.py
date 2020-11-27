import joblib
import numpy
from scipy.linalg import expm

import itertools
import os
import time

from csubst import omega_cy
from csubst import parallel
from csubst import substitution
from csubst import table

def calc_E_mean(mode, cb, sub_sad, sub_bad, obs_col, sg_a_d, g):
    E_b = numpy.zeros_like(cb.index, dtype=g['float_type'])
    bid_columns = cb.columns[cb.columns.str.startswith('branch_id_')]
    for i,sg,a,d in sg_a_d:
        # TODO: nan in a and d can be skipped in S
        if (a==d):
            continue
        if (g['asrv']=='each'):
            sub_sites = substitution.get_each_sub_sites(sub_sad, mode, sg, a, d, g)
        elif (g['asrv']=='sn'):
            if (obs_col.startswith('S')):
                sub_sites = g['sub_sites']['S']
            elif (obs_col.startswith('N')):
                sub_sites = g['sub_sites']['N']
        else:
            sub_sites = g['sub_sites'][g['asrv']]
        sub_branches = substitution.get_sub_branches(sub_bad, mode, sg, a, d)
        tmp_E = numpy.ones(shape=(E_b.shape[0], sub_sites.shape[1]), dtype=g['float_type'])
        for bid in numpy.unique(cb.loc[:,bid_columns].values):
            is_b = False
            for bc in bid_columns:
                is_b = (is_b)|(cb.loc[:,bc]==bid)
            tmp_E[is_b,:] *= sub_sites[bid,:] * sub_branches[bid]
        E_b += tmp_E.sum(axis=1)
    return E_b

def joblib_calc_quantile(mode, cb, sub_sad, sub_bad, dfq, quantile_niter, obs_col, num_sgad_combinat, sgad_chunk, g):
    bid_columns = cb.columns[cb.columns.str.startswith('branch_id_')]
    for i,sg,a,d in sgad_chunk:
        # TODO: nan in a and d can be skipped in S
        if (a==d):
            continue
        if (g['asrv']=='each'):
            sub_sites = substitution.get_each_sub_sites(sub_sad, mode, sg, a, d, g)
        elif (g['asrv']=='sn'):
                if (obs_col.startswith('S')):
                    sub_sites = g['sub_sites']['S']
                elif (obs_col.startswith('N')):
                    sub_sites = g['sub_sites']['N']
        else:
            sub_sites = g['sub_sites'][g['asrv']]
        sub_branches = substitution.get_sub_branches(sub_bad, mode, sg, a, d)
        p = sub_sites[0]
        if p.sum()==0:
            continue
        pm_start = time.time()
        array_site = numpy.arange(p.shape[0])
        cb_ids = cb.loc[:,bid_columns].values
        dfq[:,:] += omega_cy.get_permutations(cb_ids, array_site, sub_branches, p, quantile_niter)
        txt = '{}: {}/{} matrix_group/ancestral_state/derived_state combinations. Time elapsed for {:,} permutation: {:,} [sec]'
        print(txt.format(obs_col, i+1, num_sgad_combinat, quantile_niter, int(time.time()-pm_start)), flush=True)

def calc_E_stat(cb, sub_tensor, mode, stat='mean', quantile_niter=1000, SN='', g={}):
    sub_tensor = numpy.nan_to_num(sub_tensor)
    if mode=='spe2spe':
        sub_bad = sub_tensor.sum(axis=1)  # branch, matrix_group, ancestral_state, derived_state
        ancestral_states = numpy.arange(sub_bad.shape[2])
        derived_states = numpy.arange(sub_bad.shape[3])
        sub_sad = sub_tensor.sum(axis=0)
    elif mode=='spe2any':
        sub_bad = sub_tensor.sum(axis=(1, 4))  # branch, matrix_group, ancestral_state
        ancestral_states = numpy.arange(sub_bad.shape[2])
        derived_states = ['2any',] # dummy
        sub_sad = sub_tensor.sum(axis=(0, 4))
    elif mode=='any2spe':
        sub_bad = sub_tensor.sum(axis=(1, 3))  # branch, matrix_group, derived_state
        ancestral_states = ['any2',] # dummy
        derived_states = numpy.arange(sub_bad.shape[2])
        sub_sad = sub_tensor.sum(axis=(0, 3))
    elif mode=='any2any':
        sub_bad = sub_tensor.sum(axis=(1, 3, 4))  # branch, matrix_group
        ancestral_states = ['any2',] # dummy
        derived_states = ['2any',] # dummy
        sub_sad = sub_tensor.sum(axis=(0, 3, 4))
    sg_a_d = list(itertools.product(numpy.arange(sub_bad.shape[1]), ancestral_states, derived_states))
    num_sgad_combinat = len(sg_a_d)
    sg_a_d = [ [i,]+list(items) for i,items in zip(range(num_sgad_combinat), sg_a_d) ]
    obs_col = SN+mode
    if stat=='mean':
        # TODO, parallel computation
        E_b = calc_E_mean(mode, cb, sub_sad, sub_bad, obs_col, sg_a_d, g)
    elif stat=='quantile':
        mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.dfq.mmap')
        if os.path.exists(mmap_out): os.unlink(mmap_out)
        my_dtype = sub_tensor.dtype
        if 'bool' in str(my_dtype):
            my_dtype = numpy.int32
        dfq = numpy.memmap(filename=mmap_out, dtype=my_dtype, shape=(cb.shape[0], quantile_niter), mode='w+')
        sgad_chunks,mmap_start_not_necessary_here = parallel.get_chunks(sg_a_d, g['nslots'])
        joblib.Parallel(n_jobs=g['nslots'], max_nbytes=None, backend='multiprocessing')(
            joblib.delayed(joblib_calc_quantile)
            (mode, cb, sub_sad, sub_bad, dfq, quantile_niter, obs_col, num_sgad_combinat, sgad_chunk, g) for sgad_chunk in sgad_chunks
        )
        if os.path.exists(mmap_out): os.unlink(mmap_out)
        E_b = numpy.zeros_like(cb.index, dtype=g['float_type'])
        for i in cb.index:
            # TODO: poisson approximation
            obs_value = cb.loc[i,obs_col]
            gt_rank = (dfq[i,:]<obs_value).sum()
            ge_rank = (dfq[i,:]<=obs_value).sum()
            corrected_rank = (gt_rank+ge_rank)/2
            E_b[i] = corrected_rank / quantile_niter
    return E_b

def get_E(cb, g, N_tensor, S_tensor):
    if g['omega_method']=='pm':
        cb['ENany2any'] = calc_E_stat(cb, N_tensor, mode='any2any', stat='mean', SN='N', g=g)
        cb['ENspe2any'] = calc_E_stat(cb, N_tensor, mode='spe2any', stat='mean', SN='N', g=g)
        cb['ENany2spe'] = calc_E_stat(cb, N_tensor, mode='any2spe', stat='mean', SN='N', g=g)
        cb['ENspe2spe'] = calc_E_stat(cb, N_tensor, mode='spe2spe', stat='mean', SN='N', g=g)
        cb['ESany2any'] = calc_E_stat(cb, S_tensor, mode='any2any', stat='mean', SN='S', g=g)
        cb['ESspe2any'] = calc_E_stat(cb, S_tensor, mode='spe2any', stat='mean', SN='S', g=g)
        cb['ESany2spe'] = calc_E_stat(cb, S_tensor, mode='any2spe', stat='mean', SN='S', g=g)
        cb['ESspe2spe'] = calc_E_stat(cb, S_tensor, mode='spe2spe', stat='mean', SN='S', g=g)
        cb['ENany2dif'] = cb['ENany2any'] - cb['ENany2spe']
        cb['ESany2dif'] = cb['ESany2any'] - cb['ESany2spe']
    elif g['omega_method']=='rec':
        id_cols = cb.columns[cb.columns.str.startswith('branch_id_')]
        state_pepE = get_exp_state(g=g, mode='pep')
        EN_tensor = substitution.get_substitution_tensor(state_tensor=state_pepE, state_tensor_anc=g['state_pep'], mode='asis', g=g, mmap_attr='EN')
        print('Number of total empirically expected nonsynonymous substitutions in the tree: {:,.2f}'.format(EN_tensor.sum()))
        cbEN = substitution.get_cb(cb.loc[:,id_cols].values, EN_tensor, g, 'EN')
        os.remove( [f for f in os.listdir() if f.startswith('tmp.csubst.')&f.endswith('.EN.mmap') ][0])
        cb = table.merge_tables(cb, cbEN)
        del state_pepE,cbEN
        state_cdnE = get_exp_state(g=g, mode='cdn')
        ES_tensor = substitution.get_substitution_tensor(state_tensor=state_cdnE, state_tensor_anc=g['state_cdn'], mode='syn', g=g, mmap_attr='ES')
        print('Number of total empirically expected synonymous substitutions in the tree: {:,.2f}'.format(ES_tensor.sum()))
        cbES = substitution.get_cb(cb.loc[:,id_cols].values, ES_tensor, g, 'ES')
        os.remove( [f for f in os.listdir() if f.startswith('tmp.csubst.')&f.endswith('.ES.mmap') ][0])
        cb = table.merge_tables(cb, cbES)
        del state_cdnE,cbES
        cb['ENany2dif'] = cb['ENany2any'] - cb['ENany2spe']
        cb['ESany2dif'] = cb['ESany2any'] - cb['ESany2spe']
        E_cols = cb.columns[cb.columns.str.startswith('E')]
        for node in g['tree'].traverse():
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
    if g['calc_quantile']:
        cb['QNany2any'] = calc_E_stat(cb, N_tensor, mode='any2any', stat='quantile', SN='N', g=g)
        cb['QSany2any'] = calc_E_stat(cb, S_tensor, mode='any2any', stat='quantile', SN='S', g=g)
        #cb['QNspe2any'] = calc_E_stat(cb, N_tensor, mode='spe2any', stat='quantile', SN='N', g=g)
        #cb['QSspe2any'] = calc_E_stat(cb, S_tensor, mode='spe2any', stat='quantile', SN='S', g=g)
        cb['QNany2spe'] = calc_E_stat(cb, N_tensor, mode='any2spe', stat='quantile', SN='N', g=g)
        cb['QSany2spe'] = calc_E_stat(cb, S_tensor, mode='any2spe', stat='quantile', SN='S', g=g)
        #cb['QNspe2spe'] = calc_E_stat(cb, N_tensor, mode='spe2spe', stat='quantile', SN='N', g=g)
        #cb['QSspe2spe'] = calc_E_stat(cb, S_tensor, mode='spe2spe', stat='quantile', SN='S', g=g)
    return cb

def get_exp_state(g, mode, bl='asis'):
    if mode=='cdn':
        state = g['state_cdn'].astype(g['float_type'])
        inst = g['instantaneous_codon_rate_matrix']
        sub_col = 'S_sub'
    elif mode=='pep':
        state = g['state_pep'].astype(g['float_type'])
        inst = g['instantaneous_aa_rate_matrix']
        sub_col = 'N_sub'
    stateE = numpy.zeros_like(state, dtype=g['float_type'])
    for node in g['tree'].traverse():
        if node.is_root():
            continue
        if bl=='substitution':
            num_site = state.shape[1]
            branch_length = g['branch_table'].loc[node.numerical_label,sub_col] / num_site
        elif bl=='asis':
            if mode=='cdn':
                branch_length = node.SNdist
            elif mode=='pep':
                branch_length = node.Ndist
        branch_length = max(branch_length, 0)
        nl = node.numerical_label
        parent_nl = node.up.numerical_label
        if parent_nl>stateE.shape[0]:
            continue # Skip if parent is the root node
        inst_bl = inst * branch_length
        for site_rate in numpy.unique(g['iqtree_rate_values']):
            if bl=='substitution':
                inst_bl_site = numpy.copy(inst_bl)
            elif bl=='asis':
                inst_bl_site = inst_bl * site_rate # TODO is this valid when bl=='substitution'?
            transition_prob = expm(inst_bl_site)
            site_indices = numpy.where(g['iqtree_rate_values']==site_rate)[0]
            for s in site_indices:
                expected_transition_ad = numpy.einsum('a,ad->ad', state[parent_nl,s,:], transition_prob)
                if expected_transition_ad.sum()-1>g['float_tol']:
                    expected_transition_ad /= expected_transition_ad.sum()
                expected_derived_state = expected_transition_ad.sum(axis=0)
                stateE[nl,s,:] = expected_derived_state
                assert (expected_derived_state.sum()-1)<g['float_tol'], 'Derived state should be equal to 1. ({})'.format(expected_derived_state.sum())
    max_stateE = stateE.sum(axis=(2)).max()
    assert (max_stateE-1)<g['float_tol'], 'Total probability of expected states should not exceed 1. {}'.format(max_stateE)
    return stateE

def get_omega(cb):
    cb.loc[:,'omega_any2any'] = (cb.loc[:,'Nany2any'] / cb.loc[:,'ENany2any']) / (cb.loc[:,'Sany2any'] / cb.loc[:,'ESany2any'])
    cb.loc[:,'omega_any2spe'] = (cb.loc[:,'Nany2spe'] / cb.loc[:,'ENany2spe']) / (cb.loc[:,'Sany2spe'] / cb.loc[:,'ESany2spe'])
    dNdif = ((cb.loc[:,'Nany2any']-cb.loc[:,'Nany2spe'])/cb.loc[:,'ENany2dif'])
    dSdif = ((cb.loc[:,'Sany2any']-cb.loc[:,'Sany2spe'])/cb['ESany2dif'])
    cb.loc[:,'omega_any2dif'] = dNdif / dSdif
    return cb

def get_CoD(cb):
    cb['NCoD'] = cb['Nany2spe'] / (cb['Nany2any'] - cb['Nany2spe'])
    cb['SCoD'] = cb['Sany2spe'] / (cb['Sany2any'] - cb['Sany2spe'])
    cb['NCoDoSCoD'] = cb['NCoD'] / cb['SCoD']
    return cb

def print_cb_stats(cb, prefix):
    arity = cb.columns.str.startswith('branch_id_').sum()
    hd = 'arity='+str(arity)+', '+prefix+':'
    print(hd, 'median omega_any2any =', numpy.round(cb['omega_any2any'].median(), decimals=3), flush=True)
    print(hd, 'median omega_any2spe =', numpy.round(cb['omega_any2spe'].median(), decimals=3), flush=True)
    print(hd, 'median omega_any2dif  =', numpy.round(cb['omega_any2dif'].median(), decimals=3), flush=True)

def calc_omega(cb, S_tensor, N_tensor, g):
    cb = get_E(cb, g, N_tensor, S_tensor)
    cb = get_omega(cb)
    cb = get_CoD(cb)
    print_cb_stats(cb=cb, prefix='cb')
    return(cb, g)
