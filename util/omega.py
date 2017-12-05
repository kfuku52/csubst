from util.substitution import *
from util.combination import *

def node_combination_subsamples_rifle(g, arity, rep):
    all_ids = [ n.numerical_label for n in g['tree'].traverse() ]
    sub_ids = g['sub_branches']
    all_dep_ids = dict()
    for node in g['tree'].traverse():
        ancestor_ids = [ node.numerical_label for node in node.iter_ancestors() if not node.is_root() ]
        descendant_ids = [ node.numerical_label for node in node.iter_descendants() ]
        dep_id = set([node.numerical_label,])
        dep_id = dep_id.union(set(ancestor_ids))
        dep_id = dep_id.union(set(descendant_ids))
        if g['exclude_sisters']:
            children = node.get_children()
            if len(children) > 1:
                dep_id = dep_id.union(set([node.numerical_label for node in children]))
        all_dep_ids[node.numerical_label] = dep_id
    num_fail = 0
    i = 0
    id_combinations = list()
    while (i <= rep)&(num_fail <= rep):
        if num_fail == rep:
            id_combinations = list()
            print('Node combination subsampling failed', str(rep), 'times. Exiting.')
            break
        selected_ids = set()
        nonselected_ids = sub_ids
        dep_ids = set()
        flag = 0
        for a in numpy.arange(arity):
            if len(nonselected_ids)==0:
                num_fail+=1
                break
            else:
                selected_id = numpy.random.choice(list(nonselected_ids), 1)[0]
                selected_ids = selected_ids.union(set([selected_id,]))
                dep_ids = dep_ids.union(all_dep_ids[selected_id])
                nonselected_ids = all_ids.difference(dep_ids)
                flag += 1
        if flag==arity:
            if selected_id in id_combinations:
                num_fail += 1
            else:
                id_combinations.append(selected_ids)
                i += 1
    id_combinations = numpy.array([ list(ic) for ic in id_combinations ])
    id_combinations = id_combinations[:rep, :]
    return id_combinations

def node_combination_subsamples_shotgun(g, arity, rep):
    all_ids = [ n.numerical_label for n in g['tree'].traverse() ]
    sub_ids = g['sub_branches']
    dep_ids = list()
    for leaf in g['tree'].iter_leaves():
        dep_id = [leaf.numerical_label,] + [ node.numerical_label for node in leaf.iter_ancestors() if not node.is_root() ]
        dep_id = numpy.sort(numpy.array(dep_id))
        dep_ids.append(dep_id)
    if g['exclude_sisters']:
        for node in g['tree'].traverse():
            children = node.get_children()
            if len(children) > 1:
                dep_id = numpy.sort(numpy.array([node.numerical_label for node in children]))
                dep_ids.append(dep_id)
    id_combinations = numpy.zeros(shape=(0,arity), dtype=numpy.int)
    id_combinations_dif = numpy.inf
    round = 1
    while (id_combinations.shape[0] < rep)&(id_combinations_dif > rep/50):
        ss_matrix = numpy.zeros(shape=(len(all_ids), rep), dtype=numpy.bool_, order='C')
        for i in numpy.arange(rep):
            ind = numpy.random.choice(a=sub_ids, size=arity, replace=False)
            ss_matrix[ind,i] = 1
        is_dependent_col = False
        for dep_id in dep_ids:
            is_dependent_col = (is_dependent_col)|(ss_matrix[dep_id,:].sum(axis=0)>1)
        ss_matrix = ss_matrix[:,~is_dependent_col]
        rows,cols = numpy.where(ss_matrix==1)
        unique_cols = numpy.unique(cols)
        tmp_id_combinations = numpy.zeros(shape=(unique_cols.shape[0], arity), dtype=numpy.int)
        for i in unique_cols:
            tmp_id_combinations[i,:] = rows[cols==i]
        previous_num = id_combinations.shape[0]
        id_combinations = numpy.concatenate((id_combinations, tmp_id_combinations), axis=0)
        id_combinations.sort(axis=1)
        id_combinations = pandas.DataFrame(id_combinations).drop_duplicates().values
        id_combinations_dif = id_combinations.shape[0] - previous_num
        print('round', round,'# id_combinations =', id_combinations.shape[0], 'subsampling rate =', id_combinations_dif/rep)
        round += 1
    if id_combinations.shape[0] < rep:
        print('Inefficient subsampling. Exiting node_combinations_subsamples()')
        id_combinations = numpy.array([])
    else:
        id_combinations = id_combinations[:rep,:]
    return id_combinations

def calc_combinat_branch_omega(cb, b, s, S_tensor, N_tensor, g, rho_subsample, arity):
    rho_stats = dict()
    num_site = s.shape[0]
    arity = cb.columns.str.startswith('branch_id_').sum()
    for a in numpy.arange(arity):
        b_tmp = b.loc[:,['branch_id','S_sub','N_sub']]
        b_tmp.columns = [ c+'_'+str(a+1) for c in b_tmp.columns ]
        cb = pandas.merge(cb, b_tmp, on='branch_id_'+str(a+1), how='left')
        del b_tmp
    del b
    if rho_subsample=='all':
        print('Estimating rho from all combinations in cb table')
        cb_subsample = cb.copy()
        rhoNconv = cb_subsample['Nany2spe'].sum() / cb_subsample['Nany2any'].sum()
        rhoSconv = cb_subsample['Sany2spe'].sum() / cb_subsample['Sany2any'].sum()
        print('rhoSconv =', rhoSconv, 'total Sany2spe/Sany2any =', cb_subsample['Sany2spe'].sum(), '/', cb_subsample['Sany2any'].sum())
        print('rhoNconv =', rhoNconv, 'total Nany2spe/Nany2any =', cb_subsample['Nany2spe'].sum(), '/', cb_subsample['Nany2any'].sum())
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
        print('rhoSconv =', rhoSconv, 'total Sany2spe/Sany2any =', cb_subsample['Sany2spe'].sum(), '/', cb_subsample['Sany2any'].sum())
        print('rhoNconv =', rhoNconv, 'total Nany2spe/Nany2any =', cb_subsample['Nany2spe'].sum(), '/', cb_subsample['Nany2any'].sum())
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
    rhoNdiv = 1 - rhoNconv
    rhoSdiv = 1 - rhoSconv
    cb['EN_pair_unif'] = 1
    cb['ES_pair_unif'] = 1
    for a in numpy.arange(arity):
        cb['EN_pair_unif'] = cb['EN_pair_unif'] * (cb['N_sub_'+str(a+1)] / num_site)
        cb['ES_pair_unif'] = cb['ES_pair_unif'] * (cb['S_sub_'+str(a+1)] / num_site)
    cb['EN_pair_unif'] = cb['EN_pair_unif'] * num_site
    cb['ES_pair_unif'] = cb['ES_pair_unif'] * num_site
    cb['EN_conv_unif'] = cb['EN_pair_unif'] * rhoNconv
    cb['EN_div_unif'] = cb['EN_pair_unif'] * rhoNdiv
    cb['ES_conv_unif'] = cb['ES_pair_unif'] * rhoSconv
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
    cb['EN_conv_asrv'] = cb['EN_pair_asrv'] * rhoNconv
    cb['EN_div_asrv'] = cb['EN_pair_asrv'] * rhoNdiv
    cb['ES_conv_asrv'] = cb['ES_pair_asrv'] * rhoSconv
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
    return(cb, rho_stats)
