import os
import joblib
import itertools
import numpy
import pandas
from csubst.parallel import *

# TODO: dep_id may be more efficient in bool matrix?

def node_union(index_combinations, target_nodes, df_mmap, mmap_start):
    arity = target_nodes.shape[1] + 1
    i = mmap_start
    for ic in index_combinations:
        node_union = numpy.union1d(target_nodes[ic[0], :], target_nodes[ic[1], :])
        if (node_union.shape[0] == arity):
            df_mmap[i, :] = node_union
            i += 1

def nc_matrix2id_combinations(nc_matrix, arity):
    rows,cols = numpy.where(nc_matrix==1)
    unique_cols = numpy.unique(cols)
    id_combinations = numpy.zeros(shape=(unique_cols.shape[0], arity), dtype=numpy.int)
    for i,col in enumerate(unique_cols):
        id_combinations[i,:] = rows[cols==col]
    id_combinations.sort(axis=1)
    return(id_combinations)

def get_node_combinations(g, target_nodes=None, arity=2, check_attr=None, verbose=True):
    tree = g['tree']
    all_nodes = [ node for node in tree.traverse() if not node.is_root() ]
    if verbose:
        print("Arity:", arity, flush=True)
        print("All nodes: {:,}".format(len(all_nodes)), flush=True)
    if (target_nodes is None):
        target_nodes = list()
        for node in all_nodes:
            if (check_attr is None)|(check_attr in dir(node)):
                target_nodes.append(node.numerical_label)
        target_nodes = numpy.array(target_nodes)
        node_combinations = list(itertools.combinations(target_nodes, arity))
        node_combinations = [set(nc) for nc in node_combinations]
        node_combinations = numpy.array([list(nc) for nc in node_combinations])
    elif isinstance(target_nodes, numpy.ndarray):
        if (target_nodes.shape.__len__()==1):
            target_nodes = numpy.expand_dims(target_nodes, axis=1)
        index_combinations = list(itertools.combinations(numpy.arange(target_nodes.shape[0]), 2))
        print('# combinations for unions =', len(index_combinations))
        axis = (len(index_combinations), arity)
        mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.node_combinations.mmap')
        if os.path.exists(mmap_out): os.unlink(mmap_out)
        df_mmap = numpy.memmap(mmap_out, dtype=numpy.int32, shape=axis, mode='w+')
        chunks,starts = get_chunks(index_combinations, g['nslots'])
        joblib.Parallel(n_jobs=g['nslots'], max_nbytes=None, backend='multiprocessing')(
            joblib.delayed(node_union)
            (ids, target_nodes, df_mmap, ms) for ids, ms in zip(chunks, starts)
        )
        is_valid_combination = (df_mmap.sum(axis=1)!=0)
        if (is_valid_combination.sum()>0):
            node_combinations = numpy.unique(df_mmap[is_valid_combination,:], axis=0)
        else:
            node_combinations = numpy.zeros(shape=[0,arity], dtype=numpy.int)
    if verbose:
        num_target_node = len(list(target_nodes.flat))
        print("all target nodes: {:,}".format(num_target_node), flush=True)
        print("all node combinations: {:,}".format(len(node_combinations)), flush=True)
    nc_matrix = numpy.zeros(shape=(len(all_nodes), node_combinations.shape[0]), dtype=numpy.bool)
    for i in numpy.arange(node_combinations.shape[0]):
        nc_matrix[node_combinations[i,:],i] = 1
    is_dependent_col = False
    for dep_id in g['dep_ids']:
        is_dependent_col = (is_dependent_col)|(nc_matrix[dep_id,:].sum(axis=0)>1)
    if verbose:
        print('removing {:,} dependent branch combinations.'.format(is_dependent_col.sum()), flush=True)
    nc_matrix = nc_matrix[:,~is_dependent_col]
    g['fg_dependent_id_combinations'] = None
    if (g['foreground'] is not None)&(len(g['fg_dep_ids']) > 0):
        is_fg_dependent_col = False
        for fg_dep_id in g['fg_dep_ids']:
            is_fg_dependent_col = (is_fg_dependent_col)|(nc_matrix[fg_dep_id, :].sum(axis=0) > 1)
        if (g['fg_force_exhaustive']):
            if verbose:
                txt = 'detected {:,} (out of {:,}) foreground branch combinations to be treated as non-foreground.'
                print(txt.format(is_fg_dependent_col.sum(), is_fg_dependent_col.shape[0]), flush=True)
            fg_dep_nc_matrix = numpy.copy(nc_matrix)
            fg_dep_nc_matrix[:,~is_fg_dependent_col] = False
            g['fg_dependent_id_combinations'] = nc_matrix2id_combinations(fg_dep_nc_matrix, arity)
        else:
            if verbose:
                txt = 'removing {:,} (out of {:,}) dependent foreground branch combinations.'
                print(txt.format(is_fg_dependent_col.sum(), is_fg_dependent_col.shape[0]), flush=True)
            nc_matrix = nc_matrix[:,~is_fg_dependent_col]
    id_combinations = nc_matrix2id_combinations(nc_matrix, arity)
    if verbose:
        print("independent node combinations: {:,}".format(id_combinations.shape[0]), flush=True)
    return(g,id_combinations)

def node_combination_subsamples_rifle(g, arity, rep):
    all_ids = [ n.numerical_label for n in g['tree'].traverse() ]
    sub_ids = g['sub_branches']
    all_dep_ids = g['dep_ids']
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
    id_combinations = numpy.zeros(shape=(0,arity), dtype=numpy.int)
    id_combinations_dif = numpy.inf
    round = 1
    while (id_combinations.shape[0] < rep)&(id_combinations_dif > rep/200):
        ss_matrix = numpy.zeros(shape=(len(all_ids), rep), dtype=numpy.bool_, order='C')
        for i in numpy.arange(rep):
            ind = numpy.random.choice(a=sub_ids, size=arity, replace=False)
            ss_matrix[ind,i] = 1
        is_dependent_col = False
        for dep_id in g['dep_ids']:
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

def calc_substitution_patterns(cb):
    for key in ['S_sub','N_sub']:
        cols = cb.columns[cb.columns.str.startswith(key)].tolist()
        sub_patterns = cb.loc[:,cols]
        sub_patterns2 = pandas.DataFrame(numpy.zeros(sub_patterns.shape), columns=cols)
        for i in numpy.arange(len(cols)):
            sub_patterns2.loc[:,cols[i]] = sub_patterns.apply(lambda x: numpy.sort(x)[i], axis=1)
        sub_patterns2.loc[:,'index2'] = numpy.arange(sub_patterns2.shape[0])
        sub_patterns3 = sub_patterns2.loc[:,cols].drop_duplicates()
        sp_min = int(sub_patterns3.sum(axis=1).min())
        sp_max = int(sub_patterns3.sum(axis=1).max())
        txt = 'Number of {} patterns among {:,} branch combinations={:,}, Min total subs={:,}, Max total subs={:,}'
        print(txt.format(key, cb.shape[0], sub_patterns3.shape[0], sp_min, sp_max), flush=True)
        sub_patterns3.loc[:,'sub_pattern_id'] = numpy.arange(sub_patterns3.shape[0])
        sub_patterns4 = pandas.merge(sub_patterns2, sub_patterns3, on=cols, sort=False)
        sub_patterns4 = sub_patterns4.sort_values(axis=0, by='index2', ascending=True).reset_index()
        cb.loc[:,key+'_pattern_id'] = sub_patterns4.loc[:,'sub_pattern_id']
    return cb
