#import pyximport
#pyximport.install()
from csubst.substitution import *
from csubst.combination import *
from csubst.omega_cy import *

def get_Econv_unif_permutation(cb, sub_tensor):
    num_site = sub_tensor.shape[2]
    bid_columns = cb.columns[cb.columns.str.startswith('branch_id_')]
    sub_bad = sub_tensor.sum(axis=2)  # branch, matrix_group, ancestral_state, derived_state
    E_conv_b = 0
    for sg in numpy.arange(sub_bad.shape[1]):
        for a in numpy.arange(sub_bad.shape[2]):
            for d in numpy.arange(sub_bad.shape[3]):
                if a != d:
                    df_sub_ad = pandas.DataFrame({
                        'branch_id': numpy.arange(sub_bad.shape[0]),
                        'sub_per_site': sub_bad[:, sg, a, d] / num_site,
                    })
                    tmp_E_conv = 1
                    for bc in bid_columns:
                        df_tmp = pandas.merge(cb.loc[:, [bc, ]], df_sub_ad, left_on=bc, right_on='branch_id',
                                              how='left', sort=False)
                        tmp_E_conv *= df_tmp['sub_per_site']
                    E_conv_b += tmp_E_conv
    return E_conv_b

def calc_E_mean(mode, cb, sub_sad, sub_bad, asrv, sub_sites, sg_a_d):
    E_b = numpy.zeros_like(cb.index, dtype=numpy.float64)
    bid_columns = cb.columns[cb.columns.str.startswith('branch_id_')]
    for i,sg,a,d in sg_a_d:
        # TODO: nan in a and d can be skipped in S
        if (a==d):
            continue
        if (asrv=='each'):
            if mode == 'spe2spe':
                sub_sites = sub_sad[sg, :, a, d]
            elif mode == 'spe2any':
                sub_sites = sub_sad[sg, :, a]
            elif mode == 'any2spe':
                sub_sites = sub_sad[sg, :, d]
            elif mode == 'any2any':
                sub_sites = sub_sad[sg, :]
            sub_sites = get_relative_sub_sites(sub_sites)
        if mode == 'spe2spe':
            sub_branches = sub_bad[:, sg, a, d]
        elif mode == 'spe2any':
            sub_branches = sub_bad[:, sg, a]
        elif mode == 'any2spe':
            sub_branches = sub_bad[:, sg, d]
        elif mode == 'any2any':
            sub_branches = sub_bad[:, sg]
        df_sub_ad = pandas.DataFrame({
            'branch_id': numpy.arange(sub_bad.shape[0]),
            'sub_branches': sub_branches,
        })
        tmp_E = 1
        for bc in bid_columns:
            df_tmp = pandas.merge(cb.loc[:,[bc,]], df_sub_ad, left_on=bc, right_on='branch_id',
                                  how='left', sort=False)
            # This is meant to correct multiple substitutions, but it underestimates E.
            #tmp_E *= 1 - ((1-sub_sites) ** numpy.expand_dims(df_tmp['sub_branches'].values, axis=1))
            tmp_E *= (sub_sites * numpy.expand_dims(df_tmp['sub_branches'], axis=1))
        E_b += tmp_E.sum(axis=1)
    return E_b

def joblib_calc_quantile(mode, cb, sub_sad, sub_bad, dfq, asrv, sub_sites, quantile_niter, obs_col, num_sgad_combinat, sgad_chunk):
    bid_columns = cb.columns[cb.columns.str.startswith('branch_id_')]
    for i,sg,a,d in sgad_chunk:
        # TODO: nan in a and d can be skipped in S
        if (a==d):
            continue
        if (asrv=='each'):
            if mode == 'spe2spe':
                sub_sites = sub_sad[sg, :, a, d]
            elif mode == 'spe2any':
                sub_sites = sub_sad[sg, :, a]
            elif mode == 'any2spe':
                sub_sites = sub_sad[sg, :, d]
            elif mode == 'any2any':
                sub_sites = sub_sad[sg, :]
            sub_sites = get_relative_sub_sites(sub_sites)
        if mode == 'spe2spe':
            sub_branches = sub_bad[:, sg, a, d]
        elif mode == 'spe2any':
            sub_branches = sub_bad[:, sg, a]
        elif mode == 'any2spe':
            sub_branches = sub_bad[:, sg, d]
        elif mode == 'any2any':
            sub_branches = sub_bad[:, sg]
        p = sub_sites[0]
        if p.sum()==0:
            continue
        pm_start = time.time()
        array_site = numpy.arange(p.shape[0])
        cb_ids = cb.loc[:,bid_columns].values
        dfq[:,:] += get_permutations(cb_ids, array_site, sub_branches, p, quantile_niter)
        txt = '{}: {}/{} matrix_group/ancestral_state/derived_state combinations. Time elapsed for {:,} permutation: {:,} [sec]'
        print(txt.format(obs_col, i+1, num_sgad_combinat, quantile_niter, int(time.time()-pm_start)), flush=True)

def calc_E_stat(cb, sub_tensor, mode, stat='mean', quantile_niter=1000, SN='', g={}):
    sub_tensor = numpy.nan_to_num(sub_tensor)
    num_site = sub_tensor.shape[2]
    if (g['asrv']=='sn'):
        if (SN=='S'):
            sub_sites = g['sub_sites']['S']
        elif (SN=='N'):
            sub_sites = g['sub_sites']['N']
    else:
        sub_sites = g['sub_sites'][g['asrv']]
    if mode=='spe2spe':
        sub_bad = sub_tensor.sum(axis=2)  # branch, matrix_group, ancestral_state, derived_state
        ancestral_states = numpy.arange(sub_bad.shape[2])
        derived_states = numpy.arange(sub_bad.shape[3])
        sub_sad = sub_tensor.sum(axis=0)
    elif mode=='spe2any':
        sub_bad = sub_tensor.sum(axis=(2, 4))  # branch, matrix_group, ancestral_state
        ancestral_states = numpy.arange(sub_bad.shape[2])
        derived_states = ['2any',] # dummy
        sub_sad = sub_tensor.sum(axis=(0, 4))
    elif mode=='any2spe':
        sub_bad = sub_tensor.sum(axis=(2, 3))  # branch, matrix_group, derived_state
        ancestral_states = ['any2',] # dummy
        derived_states = numpy.arange(sub_bad.shape[2])
        sub_sad = sub_tensor.sum(axis=(0, 3))
    elif mode=='any2any':
        sub_bad = sub_tensor.sum(axis=(2, 3, 4))  # branch, matrix_group
        ancestral_states = ['any2',] # dummy
        derived_states = ['2any',] # dummy
        sub_sad = sub_tensor.sum(axis=(0, 3, 4))
    sg_a_d = list(itertools.product(numpy.arange(sub_bad.shape[1]), ancestral_states, derived_states))
    num_sgad_combinat = len(sg_a_d)
    sg_a_d = [ [i,]+list(items) for i,items in zip(range(num_sgad_combinat), sg_a_d) ]
    if stat=='mean':
        # TODO, parallel computation
        E_b = calc_E_mean(mode, cb, sub_sad, sub_bad, g['asrv'], sub_sites, sg_a_d)
    elif stat=='quantile':
        obs_col = SN+mode
        mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.dfq.mmap')
        if os.path.exists(mmap_out): os.unlink(mmap_out)
        dfq = numpy.memmap(filename=mmap_out, dtype=numpy.int32, shape=(cb.shape[0], quantile_niter), mode='w+')
        sgad_chunks,mmap_start_not_necessary_here = get_chunks(sg_a_d, g['nslots'])
        joblib.Parallel(n_jobs=g['nslots'], max_nbytes=None, backend='multiprocessing')(
            joblib.delayed(joblib_calc_quantile)
            (mode, cb, sub_sad, sub_bad, dfq, g['asrv'], sub_sites, quantile_niter, obs_col, num_sgad_combinat, sgad_chunk) for sgad_chunk in sgad_chunks
        )
        if os.path.exists(mmap_out): os.unlink(mmap_out)
        E_b = numpy.zeros_like(cb.index, dtype=numpy.float64)
        for i in cb.index:
            # TODO: poisson approximation
            obs_value = cb.loc[i,obs_col]
            gt_rank = (dfq[i,:]<obs_value).sum()
            ge_rank = (dfq[i,:]<=obs_value).sum()
            corrected_rank = (gt_rank+ge_rank)/2
            E_b[i] = corrected_rank / quantile_niter

    return E_b

def get_E(cb, g, N_tensor, S_tensor):
    if g['omega_method']=='rho':
        rhoNany2spe = g['df_cb_stats'].loc[(g['df_cb_stats']['arity'] == g['current_arity']), 'rhoNany2spe'].values
        rhoSany2spe = g['df_cb_stats'].loc[(g['df_cb_stats']['arity'] == g['current_arity']), 'rhoSany2spe'].values
        rhoNany2dif = 1 - rhoNany2spe
        rhoSany2dif = 1 - rhoSany2spe
        cb['ENany2any'] = calc_E_stat(cb, N_tensor, mode='any2any', stat='mean', SN='N', g=g)
        cb['ESany2any'] = calc_E_stat(cb, S_tensor, mode='any2any', stat='mean', SN='S', g=g)
        cb['ENany2spe'] = cb['ENany2any'] * rhoNany2spe
        cb['ENany2dif'] = cb['ENany2any'] * rhoNany2dif
        cb['ESany2spe'] = cb['ESany2any'] * rhoSany2spe
        cb['ESany2dif'] = cb['ESany2any'] * rhoSany2dif
    elif g['omega_method']=='permutation':
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
    if g['calc_distribution']:
        cb['QNany2any'] = calc_E_stat(cb, N_tensor, mode='any2any', stat='quantile', SN='N', g=g)
        cb['QSany2any'] = calc_E_stat(cb, S_tensor, mode='any2any', stat='quantile', SN='S', g=g)
        #cb['QNspe2any'] = calc_E_stat(cb, N_tensor, mode='spe2any', stat='quantile', SN='N', g=g)
        #cb['QSspe2any'] = calc_E_stat(cb, S_tensor, mode='spe2any', stat='quantile', SN='S', g=g)
        cb['QNany2spe'] = calc_E_stat(cb, N_tensor, mode='any2spe', stat='quantile', SN='N', g=g)
        cb['QSany2spe'] = calc_E_stat(cb, S_tensor, mode='any2spe', stat='quantile', SN='S', g=g)
        #cb['QNspe2spe'] = calc_E_stat(cb, N_tensor, mode='spe2spe', stat='quantile', SN='N', g=g)
        #cb['QSspe2spe'] = calc_E_stat(cb, S_tensor, mode='spe2spe', stat='quantile', SN='S', g=g)

    return cb

def get_omega(cb):
    cb['omega_pair'] = (cb['Nany2any'] / cb['ENany2any']) / (cb['Sany2any'] / cb['ESany2any'])
    cb['omega_conv'] = (cb['Nany2spe'] / cb['ENany2spe']) / (cb['Sany2spe'] / cb['ESany2spe'])
    cb['omega_div'] = ((cb['Nany2any']-cb['Nany2spe'])/cb['ENany2dif']) / ((cb['Sany2any']-cb['Sany2spe'])/cb['ESany2dif'])
    return cb

def get_CoD(cb):
    cb['NCoD'] = cb['Nany2spe'] / (cb['Nany2any'] - cb['Nany2spe'])
    cb['SCoD'] = cb['Sany2spe'] / (cb['Sany2any'] - cb['Sany2spe'])
    cb['NCoDoSCoD'] = cb['NCoD'] / cb['SCoD']
    return cb

def print_cb_stats(cb, prefix):
    arity = cb.columns.str.startswith('branch_id_').sum()
    hd = 'arity='+str(arity)+', '+prefix+':'
    print(hd, 'median omega_pair =', numpy.round(cb['omega_pair'].median(), decimals=3), flush=True)
    print(hd, 'median omega_conv =', numpy.round(cb['omega_conv'].median(), decimals=3), flush=True)
    print(hd, 'median omega_div  =', numpy.round(cb['omega_div'].median(), decimals=3), flush=True)

def get_rho(cb, b, g, N_tensor, S_tensor):
    if g['cb_stats'] is not None:
        print('Retrieving pre-calculated rho')
        rhoSany2spe = g['df_cb_stats'].loc[g['df_cb_stats']['arity'] == g['current_arity'], 'rhoSany2spe'].values
        rhoNany2spe = g['df_cb_stats'].loc[g['df_cb_stats']['arity'] == g['current_arity'], 'rhoNany2spe'].values
        flat_cb_stats = False
    elif g['current_arity'] == 2:
        print('Estimating rho from all combinations in cb table')
        cb_subsample = cb.copy()
        rhoNany2spe = cb_subsample['Nany2spe'].sum() / cb_subsample['Nany2any'].sum()
        rhoSany2spe = cb_subsample['Sany2spe'].sum() / cb_subsample['Sany2any'].sum()
        subsampling_type = 'all'
        flat_cb_stats = True
    else:
        print('Estimating rho from', str(g['num_subsample']), 'subsampled combinations')
        method = 'shotgun'  # 'shotgun' is approx. 2 times faster than 'rifle'
        if method == 'shotgun':
            id_combinations = node_combination_subsamples_shotgun(g=g, arity=g['current_arity'], rep=g['num_subsample'])
        elif method == 'rifle':
            id_combinations = node_combination_subsamples_rifle(g=g, arity=g['current_arity'], rep=g['num_subsample'])
        subsampling_type = 'subsample'
        if id_combinations.shape[0] == 0:
            print('Recalculating cb to estimate rho from all combinations.')
            id_combinations = get_node_combinations(g=g, arity=g['current_arity'], check_attr="name")
            subsampling_type = 'all'
        cbS = get_cb(id_combinations, S_tensor, g, attr='S')
        cbN = get_cb(id_combinations, N_tensor, g, attr='N')
        cb_subsample = merge_tables(cbS, cbN)
        del cbS, cbN
        rhoNany2spe = cb_subsample['Nany2spe'].sum() / cb_subsample['Nany2any'].sum()
        rhoSany2spe = cb_subsample['Sany2spe'].sum() / cb_subsample['Sany2any'].sum()
        flat_cb_stats = True
    if flat_cb_stats:
        g['df_cb_stats'] = g['df_cb_stats'].append({
            'arity': g['current_arity'],
            'rhoNany2spe': rhoNany2spe,
            'rhoSany2spe': rhoSany2spe,
            'method': subsampling_type,
            'num_processor': g['nslots'],
            }, ignore_index=True)
        if not 'S_sub_1' in cb.columns:
            cb_subsample = get_substitutions_per_branch(cb, b, g)
        cb_subsample = get_E(cb_subsample, g, N_tensor, S_tensor)
        cb_subsample = get_omega(cb_subsample)
        print_cb_stats(cb=cb_subsample, prefix='cb_subsample')
        column_names = ['num_combinat','Sany2any','Sany2spe','Nany2any','Nany2spe',
                        'ENany2any','ESany2any','ENany2spe','ESany2spe',]
        for col in column_names:
            if not col in g['df_cb_stats'].columns:
                g['df_cb_stats'][col] = numpy.nan
        cond = (g['df_cb_stats']['arity'] == g['current_arity'])
        g['df_cb_stats'].loc[cond,'num_combinat'] = cb_subsample.shape[0]
        g['df_cb_stats'].loc[cond,'Sany2any'] = cb_subsample['Sany2any'].sum()
        g['df_cb_stats'].loc[cond,'Sany2spe'] = cb_subsample['Sany2spe'].sum()
        g['df_cb_stats'].loc[cond,'Nany2any'] = cb_subsample['Nany2any'].sum()
        g['df_cb_stats'].loc[cond,'Nany2spe'] = cb_subsample['Nany2spe'].sum()
        g['df_cb_stats'].loc[cond,'ENany2any'] = cb_subsample['ENany2any'].sum()
        g['df_cb_stats'].loc[cond,'ESany2any'] = cb_subsample['ESany2any'].sum()
        g['df_cb_stats']['ENany2spe'] = g['df_cb_stats']['ENany2any'] * g['df_cb_stats']['rhoNany2spe']
        g['df_cb_stats']['ESany2spe'] = g['df_cb_stats']['ESany2any'] * g['df_cb_stats']['rhoSany2spe']
    if (g['cb_subsample']=='yes')&(g['df_cb_stats'].loc[(g['df_cb_stats']['arity']==g['current_arity']),'method'].values=='subsample'):
        file_name = "csubst_cb_subsample_" + str(g['current_arity']) + ".tsv"
        cb_subsample.to_csv(file_name, sep="\t", index=False, float_format='%.4f', chunksize=10000)
    return g

def get_substitutions_per_branch(cb, b, g):
    for a in numpy.arange(g['current_arity']):
        b_tmp = b.loc[:,['branch_id','S_sub','N_sub']]
        b_tmp.columns = [ c+'_'+str(a+1) for c in b_tmp.columns ]
        cb = pandas.merge(cb, b_tmp, on='branch_id_'+str(a+1), how='left')
        del b_tmp
    return(cb)

def calc_omega(cb, b, S_tensor, N_tensor, g):
    if (g['omega_method']=='rho')&(g['cb_subsample']=='yes'):
        g = get_rho(cb, b, g, N_tensor, S_tensor)
    cb = get_E(cb, g, N_tensor, S_tensor)
    cb = get_omega(cb)
    cb = get_CoD(cb)
    print_cb_stats(cb=cb, prefix='cb')
    return(cb, g)
