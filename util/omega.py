from util.substitution import *
from util.combination import *

def get_Econv_unif_permutation(cb, sub_tensor):
    num_site = sub_tensor.shape[2]
    bid_columns = cb.columns[cb.columns.str.startswith('branch_id_')]
    sub_bad = sub_tensor.sum(axis=2)  # branch, synonymous_group, ancestral_state, derived_state
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

def calc_E_permutation(cb, sub_tensor, mode, asrv): # TODO, parallel computation
    sub_tensor = numpy.nan_to_num(sub_tensor)
    num_site = sub_tensor.shape[2]
    bid_columns = cb.columns[cb.columns.str.startswith('branch_id_')]
    if mode=='spe2spe':
        sub_sad = sub_tensor.sum(axis=0)
        sub_bad = sub_tensor.sum(axis=2)  # branch, synonymous_group, ancestral_state, derived_state
        ancestral_states = numpy.arange(sub_bad.shape[2])
        derived_states = numpy.arange(sub_bad.shape[3])
    elif mode=='spe2any':
        sub_sad = sub_tensor.sum(axis=(0, 4))
        sub_bad = sub_tensor.sum(axis=(2, 4))  # branch, synonymous_group, ancestral_state
        ancestral_states = numpy.arange(sub_bad.shape[2])
        derived_states = ['2any',]
    elif mode=='any2spe':
        sub_sad = sub_tensor.sum(axis=(0, 3))
        sub_bad = sub_tensor.sum(axis=(2, 3))  # branch, synonymous_group, derived_state
        ancestral_states = ['any2',]
        derived_states = numpy.arange(sub_bad.shape[2])
    elif mode=='any2any':
        sub_sad = sub_tensor.sum(axis=(0, 3, 4))
        sub_bad = sub_tensor.sum(axis=(2, 3, 4))  # branch, synonymous_group
        ancestral_states = ['any2',]
        derived_states = ['2any',]
    E_b = 0
    for sg in numpy.arange(sub_bad.shape[1]):
        for a in ancestral_states: # nan can be skipped in S
            for d in derived_states: # nan can be skipped in S
                if a != d:
                    if asrv:
                        if mode == 'spe2spe':
                            sub_sites = sub_sad[sg, :, a, d]
                        elif mode == 'spe2any':
                            sub_sites = sub_sad[sg, :, a]
                        elif mode == 'any2spe':
                            sub_sites = sub_sad[sg, :, d]
                        elif mode == 'any2any':
                            sub_sites = sub_sad[sg, :]
                        sub_sites_sum = sub_sites.sum()
                        if sub_sites_sum==0:
                            sub_sites_sum = 1
                        sub_sites = numpy.reshape(sub_sites/sub_sites_sum, newshape=(1, num_site))
                    else:
                        sub_sites = numpy.ones(shape=(1, num_site)) / num_site
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
                        df_tmp = pandas.merge(cb.loc[:, [bc, ]], df_sub_ad, left_on=bc, right_on='branch_id',
                                              how='left', sort=False)
                        #tmp_E *= 1 - ((1-sub_sites) ** numpy.expand_dims(df_tmp['sub_branches'].values, axis=1)) # underestimate E
                        tmp_E *= (sub_sites * numpy.expand_dims(df_tmp['sub_branches'], axis=1))
                    E_b += tmp_E.sum(axis=1)
    return E_b

def get_E(cb, s, g, N_tensor, S_tensor):
    num_site = s.shape[0]
    if False: # can be deleted
        cb['ENany2any_unif'] = 1
        cb['ESany2any_unif'] = 1
        for a in numpy.arange(g['current_arity']):
            cb['ENany2any_unif'] *= (cb['N_sub_' + str(a + 1)] / num_site)
            cb['ESany2any_unif'] *= (cb['S_sub_' + str(a + 1)] / num_site)
        cb['ENany2any_unif'] = cb['ENany2any_unif'] * num_site
        cb['ESany2any_unif'] = cb['ESany2any_unif'] * num_site
        N_asrv = numpy.reshape((s['N_sub'] / s['N_sub'].sum()).values, newshape=(1, num_site))
        S_asrv = numpy.reshape((s['S_sub'] / s['S_sub'].sum()).values, newshape=(1, num_site))
        ENany2any_asrv = 1
        ESany2any_asrv = 1
        for a in numpy.arange(g['current_arity']):
            ENany2any_asrv *= (N_asrv * numpy.expand_dims(cb['N_sub_' + str(a + 1)], axis=1))
            ESany2any_asrv *= (S_asrv * numpy.expand_dims(cb['S_sub_' + str(a + 1)], axis=1))
        cb['ENany2any_asrv'] = ENany2any_asrv.sum(axis=1)
        cb['ESany2any_asrv'] = ESany2any_asrv.sum(axis=1)
        del ENany2any_asrv, ESany2any_asrv
    if g['omega_method']=='rho':
        rhoNany2spe = g['df_cb_stats'].loc[(g['df_cb_stats']['arity'] == g['current_arity']), 'rhoNany2spe'].values
        rhoSany2spe = g['df_cb_stats'].loc[(g['df_cb_stats']['arity'] == g['current_arity']), 'rhoSany2spe'].values
        rhoNany2dif = 1 - rhoNany2spe
        rhoSany2dif = 1 - rhoSany2spe
        cb['ENany2any_unif'] = calc_E_permutation(cb, N_tensor, mode='any2any', asrv=False)
        cb['ESany2any_unif'] = calc_E_permutation(cb, S_tensor, mode='any2any', asrv=False)
        cb['ENany2any_asrv'] = calc_E_permutation(cb, N_tensor, mode='any2any', asrv=True)
        cb['ESany2any_asrv'] = calc_E_permutation(cb, S_tensor, mode='any2any', asrv=True)
        cb['ENany2spe_unif'] = cb['ENany2any_unif'] * rhoNany2spe
        cb['ENany2dif_unif'] = cb['ENany2any_unif'] * rhoNany2dif
        cb['ESany2spe_unif'] = cb['ESany2any_unif'] * rhoSany2spe
        cb['ESany2dif_unif'] = cb['ESany2any_unif'] * rhoSany2dif
        cb['ENany2spe_asrv'] = cb['ENany2any_asrv'] * rhoNany2spe
        cb['ENany2dif_asrv'] = cb['ENany2any_asrv'] * rhoNany2dif
        cb['ESany2spe_asrv'] = cb['ESany2any_asrv'] * rhoSany2spe
        cb['ESany2dif_asrv'] = cb['ESany2any_asrv'] * rhoSany2dif
    elif g['omega_method']=='permutation':
        cb['ENany2any_unif'] = calc_E_permutation(cb, N_tensor, mode='any2any', asrv=False)
        cb['ENany2spe_unif'] = calc_E_permutation(cb, N_tensor, mode='any2spe', asrv=False)
        cb['ESany2any_unif'] = calc_E_permutation(cb, S_tensor, mode='any2any', asrv=False)
        cb['ESany2spe_unif'] = calc_E_permutation(cb, S_tensor, mode='any2spe', asrv=False)
        cb['ENany2dif_unif'] = cb['ENany2any_unif'] - cb['ENany2spe_unif']
        cb['ESany2dif_unif'] = cb['ESany2any_unif'] - cb['ESany2spe_unif']
        cb['ENany2any_asrv'] = calc_E_permutation(cb, N_tensor, mode='any2any', asrv=True)
        cb['ENspe2any_asrv'] = calc_E_permutation(cb, N_tensor, mode='spe2any', asrv=True)
        cb['ENany2spe_asrv'] = calc_E_permutation(cb, N_tensor, mode='any2spe', asrv=True)
        cb['ENspe2spe_asrv'] = calc_E_permutation(cb, N_tensor, mode='spe2spe', asrv=True)
        cb['ESany2any_asrv'] = calc_E_permutation(cb, S_tensor, mode='any2any', asrv=True)
        cb['ESspe2any_asrv'] = calc_E_permutation(cb, S_tensor, mode='spe2any', asrv=True)
        cb['ESany2spe_asrv'] = calc_E_permutation(cb, S_tensor, mode='any2spe', asrv=True)
        cb['ESspe2spe_asrv'] = calc_E_permutation(cb, S_tensor, mode='spe2spe', asrv=True)
        cb['ENany2dif_asrv'] = cb['ENany2any_asrv'] - cb['ENany2spe_asrv']
        cb['ESany2dif_asrv'] = cb['ESany2any_asrv'] - cb['ESany2spe_asrv']
    return cb

def get_omega(cb):
    cb['omega_pair_unif'] = (cb['Nany2any'] / cb['ENany2any_unif']) / (cb['Sany2any'] / cb['ESany2any_unif'])
    cb['omega_conv_unif'] = (cb['Nany2spe'] / cb['ENany2spe_unif']) / (cb['Sany2spe'] / cb['ESany2spe_unif'])
    cb['omega_div_unif'] = ((cb['Nany2any']-cb['Nany2spe']) / cb['ENany2dif_unif']) / ((cb['Sany2any']-cb['Sany2spe']) / cb['ESany2dif_unif'])
    cb['omega_pair_asrv'] = (cb['Nany2any'] / cb['ENany2any_asrv']) / (cb['Sany2any'] / cb['ESany2any_asrv'])
    cb['omega_conv_asrv'] = (cb['Nany2spe'] / cb['ENany2spe_asrv']) / (cb['Sany2spe'] / cb['ESany2spe_asrv'])
    cb['omega_div_asrv'] = ((cb['Nany2any']-cb['Nany2spe']) / cb['ENany2dif_asrv']) / ((cb['Sany2any']-cb['Sany2spe']) / cb['ESany2dif_asrv'])
    return cb

def get_CoD(cb):
    cb['NCoD'] = cb['Nany2spe'] / (cb['Nany2any'] - cb['Nany2spe'])
    cb['SCoD'] = cb['Sany2spe'] / (cb['Sany2any'] - cb['Sany2spe'])
    return cb

def get_cb_stats(cb, b, s, g, N_tensor, S_tensor):
    for a in numpy.arange(g['current_arity']):
        b_tmp = b.loc[:,['branch_id','S_sub','N_sub']]
        b_tmp.columns = [ c+'_'+str(a+1) for c in b_tmp.columns ]
        cb = pandas.merge(cb, b_tmp, on='branch_id_'+str(a+1), how='left')
        del b_tmp
    cb = get_E(cb, s, g, N_tensor, S_tensor)
    cb = get_omega(cb)
    cb = get_CoD(cb)
    return cb

def print_cb_stats(cb, prefix):
    arity = cb.columns.str.startswith('branch_id_').sum()
    hd = 'arity='+str(arity)+', '+prefix+':'
    print(hd, 'median omega_pair_unif =', numpy.round(cb['omega_pair_unif'].median(), decimals=3), flush=True)
    print(hd, 'median omega_conv_unif =', numpy.round(cb['omega_conv_unif'].median(), decimals=3), flush=True)
    print(hd, 'median omega_div_unif  =', numpy.round(cb['omega_div_unif'].median(), decimals=3), flush=True)
    print(hd, 'median omega_pair_asrv =', numpy.round(cb['omega_pair_asrv'].median(), decimals=3), flush=True)
    print(hd, 'median omega_conv_asrv =', numpy.round(cb['omega_conv_asrv'].median(), decimals=3), flush=True)
    print(hd, 'median omega_div_asrv  =', numpy.round(cb['omega_div_asrv'].median(), decimals=3), flush=True)

def get_rho(cb, b, s, g, N_tensor, S_tensor):
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
            id_combinations = prepare_node_combinations(g=g, arity=g['current_arity'], check_attr="name")
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
        cb_subsample = get_cb_stats(cb_subsample, b, s, g, N_tensor, S_tensor)
        print_cb_stats(cb=cb_subsample, prefix='cb_subsample')
        column_names = ['num_combinat','Sany2any','Sany2spe','Nany2any','Nany2spe','ENany2any_unif','ESany2any_unif',
                        'ENany2any_asrv','ESany2any_asrv','ENany2spe_unif','ESany2spe_unif','ENany2spe_asrv','ESany2spe_asrv',]
        for col in column_names:
            if not col in g['df_cb_stats'].columns:
                g['df_cb_stats'][col] = numpy.nan
        cond = (g['df_cb_stats']['arity'] == g['current_arity'])
        g['df_cb_stats'].loc[cond,'num_combinat'] = cb_subsample.shape[0]
        g['df_cb_stats'].loc[cond,'Sany2any'] = cb_subsample['Sany2any'].sum()
        g['df_cb_stats'].loc[cond,'Sany2spe'] = cb_subsample['Sany2spe'].sum()
        g['df_cb_stats'].loc[cond,'Nany2any'] = cb_subsample['Nany2any'].sum()
        g['df_cb_stats'].loc[cond,'Nany2spe'] = cb_subsample['Nany2spe'].sum()
        g['df_cb_stats'].loc[cond,'ENany2any_unif'] = cb_subsample['ENany2any_unif'].sum()
        g['df_cb_stats'].loc[cond,'ESany2any_unif'] = cb_subsample['ESany2any_unif'].sum()
        g['df_cb_stats'].loc[cond,'ENany2any_asrv'] = cb_subsample['ENany2any_asrv'].sum()
        g['df_cb_stats'].loc[cond,'ESany2any_asrv'] = cb_subsample['ESany2any_asrv'].sum()
        g['df_cb_stats']['ENany2spe_unif'] = g['df_cb_stats']['ENany2any_unif'] * g['df_cb_stats']['rhoNany2spe']
        g['df_cb_stats']['ESany2spe_unif'] = g['df_cb_stats']['ESany2any_unif'] * g['df_cb_stats']['rhoSany2spe']
        g['df_cb_stats']['ENany2spe_asrv'] = g['df_cb_stats']['ENany2any_asrv'] * g['df_cb_stats']['rhoNany2spe']
        g['df_cb_stats']['ESany2spe_asrv'] = g['df_cb_stats']['ESany2any_asrv'] * g['df_cb_stats']['rhoSany2spe']
    if (g['cb_subsample'])&(g['df_cb_stats'].loc[(g['df_cb_stats']['arity']==g['current_arity']),'method'].values=='subsample'):
        file_name = "csubst_cb_subsample_" + str(g['current_arity']) + ".tsv"
        cb_subsample.to_csv(file_name, sep="\t", index=False, float_format='%.4f', chunksize=10000)
    return g

def calc_omega(cb, b, s, S_tensor, N_tensor, g):
    if (g['omega_method']=='rho')|(g['cb_subsample']):
        g = get_rho(cb, b, s, g, N_tensor, S_tensor)
    cb = get_cb_stats(cb, b, s, g, N_tensor, S_tensor)
    print_cb_stats(cb=cb, prefix='cb')
    return(cb, g)
