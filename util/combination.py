import os
import joblib
import itertools
import numpy
import pandas
from util.parallel import *

def node_union(index_combinations, target_nodes, df_mmap, mmap_start):
    arity = target_nodes.shape[1] + 1
    i = mmap_start
    for ic in index_combinations:
        node_union = numpy.union1d(target_nodes[ic[0], :], target_nodes[ic[1], :])
        if (node_union.shape[0] == arity):
            df_mmap[i, :] = node_union
            i += 1

def prepare_node_combinations(g, target_nodes=None, arity=2, check_attr=None, verbose=True, foreground=False):
    tree = g['tree']
    all_nodes = [ node for node in tree.traverse() if not node.is_root() ]
    if verbose:
        print("arity:", arity, flush=True)
        print("all nodes:", len(all_nodes), flush=True)
    if (target_nodes is None):
        target_nodes = list()
        for node in all_nodes:
            if (check_attr is None)|(check_attr in dir(node)):
                target_nodes.append(node.numerical_label)
        node_combinations = list(itertools.combinations(target_nodes, arity))
        node_combinations = [set(nc) for nc in node_combinations]
        node_combinations = numpy.array([list(nc) for nc in node_combinations])
    if isinstance(target_nodes, list):
        node_combinations = list(itertools.combinations(target_nodes, arity))
        node_combinations = [set(nc) for nc in node_combinations]
        node_combinations = numpy.array([list(nc) for nc in node_combinations])
    elif 'shape' in dir(target_nodes):
        index_combinations = list(itertools.combinations(numpy.arange(target_nodes.shape[0]), 2))
        print('# combinations for unions =', len(index_combinations))
        axis = (len(index_combinations), target_nodes.shape[1]+1)
        mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.node_combinations.mmap')
        if os.path.exists(mmap_out): os.unlink(mmap_out)
        df_mmap = numpy.memmap(mmap_out, dtype=numpy.int64, shape=axis, mode='w+')
        chunks,starts = get_chunks(index_combinations, g['nslots'])
        joblib.Parallel(n_jobs=g['nslots'], max_nbytes=None, backend='multiprocessing')(
            joblib.delayed(node_union)
            (ids, target_nodes, df_mmap, ms) for ids, ms in zip(chunks, starts)
        )
        node_combinations = numpy.unique(df_mmap[df_mmap.sum(axis=1)!=0,:], axis=0)
    if verbose:
        print("target nodes:", len(target_nodes), flush=True)
        print("all node combinations: ", len(node_combinations), flush=True)
    nc_matrix = numpy.zeros(shape=(len(all_nodes), node_combinations.shape[0]), dtype=numpy.bool)
    for i in numpy.arange(node_combinations.shape[0]):
        nc_matrix[node_combinations[i,:],i] = 1
    is_dependent_col = False
    for dep_id in g['dep_ids']:
        is_dependent_col = (is_dependent_col)|(nc_matrix[dep_id,:].sum(axis=0)>1)
    if (foreground)&(g['foreground'] is not None):
        for fg_dep_id in g['fg_dep_ids']:
            is_dependent_col = (is_dependent_col) | (nc_matrix[fg_dep_id, :].sum(axis=0) > 1)
    nc_matrix = nc_matrix[:,~is_dependent_col]
    rows,cols = numpy.where(nc_matrix==1)
    unique_cols = numpy.unique(cols)
    id_combinations = numpy.zeros(shape=(unique_cols.shape[0], arity), dtype=numpy.int)
    for i in unique_cols:
        id_combinations[i,:] = rows[cols==i]
    id_combinations.sort(axis=1)
    if verbose:
        print("independent node combinations: ", id_combinations.shape[0], flush=True)
    return(id_combinations)

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
    while (id_combinations.shape[0] < rep)&(id_combinations_dif > rep/50):
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