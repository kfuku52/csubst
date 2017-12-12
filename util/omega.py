from util.substitution import *
from util.combination import *

def calc_omega(cb, b, s, S_tensor, N_tensor, g, rho_subsample):
    rho_stats = dict()
    num_site = s.shape[0]
    arity = cb.columns.str.startswith('branch_id_').sum()
    for a in numpy.arange(arity):
        b_tmp = b.loc[:,['branch_id','S_sub','N_sub']]
        b_tmp.columns = [ c+'_'+str(a+1) for c in b_tmp.columns ]
        cb = pandas.merge(cb, b_tmp, on='branch_id_'+str(a+1), how='left')
        del b_tmp
    del b
    flag = 0
    if g['cb_stats'] is None:
        flag = 1
    elif g['df_cb_stats']['arity'].max() < arity:
        flag = 1
    else:
        rho_stats = g['df_cb_stats'].loc[(g['df_cb_stats']['arity']==arity),:]
        rho_stats = rho_stats.iloc[0,:].to_dict()
    if flag:
        if rho_subsample=='all':
            print('Estimating rho from all combinations in cb table')
            cb_subsample = cb.copy()
            rhoNconv = cb_subsample['Nany2spe'].sum() / cb_subsample['Nany2any'].sum()
            rhoSconv = cb_subsample['Sany2spe'].sum() / cb_subsample['Sany2any'].sum()
            subsampling_type = 'all'
        elif isinstance(rho_subsample, dict):
            print('Estimating rho from rho at arity = 2')
            rhoSconv = rho_subsample['rhoSconv']
            rhoNconv = rho_subsample['rhoNconv']
        else:
            print('Estimating rho from', str(rho_subsample), 'subsampled combinations')
            method = 'shotgun' # 'shotgun' is approx. 2 times faster than 'rifle'
            if method=='shotgun':
                id_combinations = node_combination_subsamples_shotgun(g=g, arity=arity, rep=rho_subsample)
            elif method=='rifle':
                id_combinations = node_combination_subsamples_rifle(g=g, arity=arity, rep=rho_subsample)
            subsampling_type = 'subsample'
            if id_combinations.shape[0]==0:
                print('Recalculating cb to estimate rho from all combinations.')
                id_combinations = prepare_node_combinations(g=g, arity=arity, check_attr="name")
                subsampling_type = 'all'
            cbS = get_cb(id_combinations, S_tensor, g, attr='S')
            cbN = get_cb(id_combinations, N_tensor, g, attr='N')
            cb_subsample = merge_tables(cbS, cbN)
            del cbS, cbN
            rhoNconv = cb_subsample['Nany2spe'].sum() / cb_subsample['Nany2any'].sum()
            rhoSconv = cb_subsample['Sany2spe'].sum() / cb_subsample['Sany2any'].sum()
        rho_stats['arity'] = arity
        rho_stats['method'] = subsampling_type
        rho_stats['num_combinat'] = cb_subsample.shape[0]
        rho_stats['rhoSconv'] = rhoSconv
        rho_stats['rhoNconv'] = rhoNconv
        rho_stats['Sany2any'] = cb_subsample['Sany2any'].sum()
        rho_stats['Sany2spe'] = cb_subsample['Sany2spe'].sum()
        rho_stats['Nany2any'] = cb_subsample['Nany2any'].sum()
        rho_stats['Nany2spe'] = cb_subsample['Nany2spe'].sum()
        rho_stats['num_processor'] = g['nslots']
        del cb_subsample
    print('rhoSconv =', rho_stats['rhoSconv'],
          'total Sany2spe/Sany2any =', rho_stats['Sany2spe'], '/', rho_stats['Sany2any'])
    print('rhoNconv =', rho_stats['rhoNconv'],
          'total Nany2spe/Nany2any =', rho_stats['Nany2spe'], '/', rho_stats['Nany2any'])
    rhoNdiv = 1 - rho_stats['rhoNconv']
    rhoSdiv = 1 - rho_stats['rhoSconv']
    cb['EN_pair_unif'] = 1
    cb['ES_pair_unif'] = 1
    for a in numpy.arange(arity):
        cb['EN_pair_unif'] = cb['EN_pair_unif'] * (cb['N_sub_'+str(a+1)] / num_site)
        cb['ES_pair_unif'] = cb['ES_pair_unif'] * (cb['S_sub_'+str(a+1)] / num_site)
    cb['EN_pair_unif'] = cb['EN_pair_unif'] * num_site
    cb['ES_pair_unif'] = cb['ES_pair_unif'] * num_site
    cb['EN_conv_unif'] = cb['EN_pair_unif'] * rho_stats['rhoNconv']
    cb['EN_div_unif'] = cb['EN_pair_unif'] * rhoNdiv
    cb['ES_conv_unif'] = cb['ES_pair_unif'] * rho_stats['rhoSconv']
    cb['ES_div_unif'] = cb['ES_pair_unif'] * rhoSdiv
    N_asrv = numpy.reshape((s['N_sub'] / s['N_sub'].sum()).values, newshape=(1,s.shape[0]))
    S_asrv = numpy.reshape((s['S_sub'] / s['S_sub'].sum()).values, newshape=(1,s.shape[0]))
    EN_pair_asrv = 1
    ES_pair_asrv = 1
    for a in numpy.arange(arity):
        EN_pair_asrv = EN_pair_asrv * (1 - ((1 - N_asrv) ** numpy.expand_dims(cb['N_sub_'+str(a+1)], axis=1)))
        ES_pair_asrv = ES_pair_asrv * (1 - ((1 - S_asrv) ** numpy.expand_dims(cb['S_sub_'+str(a+1)], axis=1)))
    cb['EN_pair_asrv'] = EN_pair_asrv.sum(axis=1)
    cb['ES_pair_asrv'] = ES_pair_asrv.sum(axis=1)
    del EN_pair_asrv, ES_pair_asrv
    cb['EN_conv_asrv'] = cb['EN_pair_asrv'] * rho_stats['rhoNconv']
    cb['EN_div_asrv'] = cb['EN_pair_asrv'] * rhoNdiv
    cb['ES_conv_asrv'] = cb['ES_pair_asrv'] * rho_stats['rhoSconv']
    cb['ES_div_asrv'] = cb['ES_pair_asrv'] * rhoSdiv
    cb['omega_pair_unif'] = (cb['Nany2any'] / cb['EN_pair_unif']) / (cb['Sany2any'] / cb['ES_pair_unif'])
    cb['omega_conv_unif'] = (cb['Nany2spe'] / cb['EN_conv_unif']) / (cb['Sany2spe'] / cb['ES_conv_unif'])
    cb['omega_div_unif'] = ((cb['Nany2any']-cb['Nany2spe']) / cb['EN_div_unif']) / ((cb['Sany2any']-cb['Sany2spe']) / cb['ES_div_unif'])
    cb['omega_pair_asrvNS'] = (cb['Nany2any'] / cb['EN_pair_asrv']) / (cb['Sany2any'] / cb['ES_pair_asrv'])
    cb['omega_conv_asrvNS'] = (cb['Nany2spe'] / cb['EN_conv_asrv']) / (cb['Sany2spe'] / cb['ES_conv_asrv'])
    cb['omega_div_asrvNS'] = ((cb['Nany2any']-cb['Nany2spe']) / cb['EN_div_asrv']) / ((cb['Sany2any']-cb['Sany2spe']) / cb['ES_div_asrv'])
    cb['omega_pair_asrvN'] = (cb['Nany2any'] / cb['EN_pair_asrv']) / (cb['Sany2any'] / cb['ES_pair_unif'])
    cb['omega_conv_asrvN'] = (cb['Nany2spe'] / cb['EN_conv_asrv']) / (cb['Sany2spe'] / cb['ES_conv_unif'])
    cb['omega_div_asrvN'] = ((cb['Nany2any']-cb['Nany2spe']) / cb['EN_div_asrv']) / ((cb['Sany2any']-cb['Sany2spe']) / cb['ES_div_unif'])
    cb['NCoD'] = cb['Nany2spe'] / (cb['Nany2any'] - cb['Nany2spe'])
    cb['SCoD'] = cb['Sany2spe'] / (cb['Sany2any'] - cb['Sany2spe'])
    return(cb, rho_stats)
