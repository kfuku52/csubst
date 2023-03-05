import joblib
import numpy
import pandas

import itertools
import os
import time

from csubst import parallel
from csubst import combination_cy

def node_union(index_combinations, target_nodes, df_mmap, mmap_start):
    arity = target_nodes.shape[1] + 1
    i = mmap_start
    for ic in index_combinations:
        node_union = numpy.union1d(target_nodes[ic[0], :], target_nodes[ic[1], :])
        if (node_union.shape[0] == arity):
            df_mmap[i, :] = node_union
            i += 1

def nc_matrix2id_combinations(nc_matrix, arity, ncpu, verbose):
    start = time.time()
    rows, cols = numpy.where(numpy.equal(nc_matrix, 1))
    unique_cols = numpy.unique(cols)
    ind2  = numpy.arange(arity, dtype=numpy.int64)
    empty_id_combinations = numpy.zeros(shape=(unique_cols.shape[0], arity), dtype=numpy.int64)
    chunks, starts = parallel.get_chunks(empty_id_combinations, ncpu)
    out = joblib.Parallel(n_jobs=ncpu, max_nbytes=None, backend='multiprocessing')(
        joblib.delayed(combination_cy.generate_id_chunk)
        (chunk, start, arity, ind2, rows, cols, unique_cols) for chunk, start in zip(chunks, starts)
    )
    id_combinations = numpy.concatenate(out)
    if verbose:
        print('Time elapsed for generating branch combinations: {:,} sec'.format(int(time.time() - start)))
    return id_combinations

def get_node_combinations(g, target_nodes=None, arity=2, check_attr=None, verbose=True):
    tree = g['tree']
    all_nodes = [ node for node in tree.traverse() if not node.is_root() ]
    if verbose:
        print("All branches: {:,}".format(len(all_nodes)), flush=True)
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
        if verbose:
            print('Number of redundant branch combination unions: {:,}'.format(len(index_combinations)), flush=True)
        axis = (len(index_combinations), arity)
        mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.node_combinations.mmap')
        if os.path.exists(mmap_out): os.unlink(mmap_out)
        df_mmap = numpy.memmap(mmap_out, dtype=numpy.int32, shape=axis, mode='w+')
        chunks,starts = parallel.get_chunks(index_combinations, g['threads'])
        # TODO distinct thread/process
        joblib.Parallel(n_jobs=g['threads'], max_nbytes=None, backend='multiprocessing')(
            joblib.delayed(node_union)
            (ids, target_nodes, df_mmap, ms) for ids, ms in zip(chunks, starts)
        )
        is_valid_combination = (df_mmap.sum(axis=1)!=0)
        if (is_valid_combination.sum()>0):
            node_combinations = numpy.unique(df_mmap[is_valid_combination,:], axis=0)
        else:
            node_combinations = numpy.zeros(shape=[0,arity], dtype=numpy.int64)
    if verbose:
        num_target_node = numpy.unique(target_nodes.flatten()).shape[0]
        print("Number of target branches: {:,}".format(num_target_node), flush=True)
        print("Number of independent/non-independent branch combinations: {:,}".format(node_combinations.shape[0]), flush=True)
    nc_matrix = numpy.zeros(shape=(len(all_nodes), node_combinations.shape[0]), dtype=bool)
    for i in numpy.arange(node_combinations.shape[0]):
        nc_matrix[node_combinations[i,:],i] = 1
    is_dependent_col = False
    for dep_id in g['dep_ids']:
        is_dependent_col = (is_dependent_col)|(nc_matrix[dep_id,:].sum(axis=0)>1)
    if verbose:
        print('Removing {:,} non-independent branch combinations.'.format(is_dependent_col.sum()), flush=True)
    nc_matrix = nc_matrix[:,~is_dependent_col]
    g['fg_dependent_id_combinations'] = None
    if (g['foreground'] is not None)&(len(g['fg_dep_ids']) > 0):
        is_fg_dependent_col = False
        for fg_dep_id in g['fg_dep_ids']:
            is_fg_dependent_col |= (nc_matrix[fg_dep_id, :].sum(axis=0) > 1)
        if (g['exhaustive_until']>=arity):
            if verbose:
                txt = 'Detected {:,} (out of {:,}) foreground branch combinations to be treated as non-foreground '
                txt += '(e.g., parent-child pairs).'
                print(txt.format(is_fg_dependent_col.sum(), is_fg_dependent_col.shape[0]), flush=True)
            fg_dep_nc_matrix = numpy.copy(nc_matrix)
            fg_dep_nc_matrix[:,~is_fg_dependent_col] = False
            g['fg_dependent_id_combinations'] = nc_matrix2id_combinations(fg_dep_nc_matrix, arity, g['threads'], verbose)
        else:
            if verbose:
                txt = 'removing {:,} (out of {:,}) dependent foreground branch combinations.'
                print(txt.format(is_fg_dependent_col.sum(), is_fg_dependent_col.shape[0]), flush=True)
            nc_matrix = nc_matrix[:,~is_fg_dependent_col]
    id_combinations = nc_matrix2id_combinations(nc_matrix, arity, g['threads'], verbose)
    if verbose:
        print("Number of independent branch combinations: {:,}".format(id_combinations.shape[0]), flush=True)
    return g,id_combinations

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
    id_combinations = numpy.zeros(shape=(0,arity), dtype=numpy.int64)
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
        tmp_id_combinations = numpy.zeros(shape=(unique_cols.shape[0], arity), dtype=numpy.int64)
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
        txt = 'Number of {} patterns among {:,} branch combinations={:,}, Min total subs={:,.1f}, Max total subs={:,.1f}'
        print(txt.format(key, cb.shape[0], sub_patterns3.shape[0], sp_min, sp_max), flush=True)
        sub_patterns3.loc[:,'sub_pattern_id'] = numpy.arange(sub_patterns3.shape[0])
        sub_patterns4 = pandas.merge(sub_patterns2, sub_patterns3, on=cols, sort=False)
        sub_patterns4 = sub_patterns4.sort_values(axis=0, by='index2', ascending=True).reset_index()
        cb.loc[:,key+'_pattern_id'] = sub_patterns4.loc[:,'sub_pattern_id']
    return cb

def get_dep_ids(g):
    dep_ids = list()
    for leaf in g['tree'].iter_leaves():
        ancestor_nns = [ node.numerical_label for node in leaf.iter_ancestors() if not node.is_root() ]
        dep_id = [leaf.numerical_label,] + ancestor_nns
        dep_id = numpy.sort(numpy.array(dep_id))
        dep_ids.append(dep_id)
    if g['exclude_sister_pair']:
        for node in g['tree'].traverse():
            children = node.get_children()
            if len(children)>1:
                dep_id = numpy.sort(numpy.array([ node.numerical_label for node in children ]))
                dep_ids.append(dep_id)
    root_nn = g['tree'].numerical_label
    root_state_sum = g['state_cdn'][root_nn,:,:].sum()
    if (root_state_sum==0):
        print('Ancestral states were not estimated on the root node. Excluding sub-root nodes from the analysis.')
        subroot_nns = [ node.numerical_label for node in g['tree'].get_children() ]
        for subroot_nn in subroot_nns:
            for node in g['tree'].traverse():
                if node.is_root():
                    continue
                if subroot_nn==node.numerical_label:
                    continue
                ancestor_nns = [ anc.numerical_label for anc in node.iter_ancestors() ]
                if subroot_nn in ancestor_nns:
                    continue
                dep_ids.append(numpy.array([subroot_nn, node.numerical_label]))
    g['dep_ids'] = dep_ids
    if (g['foreground'] is not None)&(g['fg_exclude_wg']):
        fg_dep_ids = list()
        for i in numpy.arange(len(g['fg_leaf_name'])):
            tmp_fg_dep_ids = list()
            for node in g['tree'].traverse():
                is_all_leaf_lineage_fg = all([ ln in g['fg_leaf_name'][i] for ln in node.get_leaf_names() ])
                if not is_all_leaf_lineage_fg:
                    continue
                is_up_all_leaf_lineage_fg = all([ ln in g['fg_leaf_name'][i] for ln in node.up.get_leaf_names() ])
                if is_up_all_leaf_lineage_fg:
                    continue
                if node.is_leaf():
                    tmp_fg_dep_ids.append(node.numerical_label)
                else:
                    descendant_nn = [ n.numerical_label for n in node.get_descendants() ]
                    tmp_fg_dep_ids += [node.numerical_label,] + descendant_nn
            if len(tmp_fg_dep_ids)>1:
                fg_dep_ids.append(numpy.sort(numpy.array(tmp_fg_dep_ids)))
        if (g['mg_sister'])|(g['mg_parent']):
            fg_dep_ids.append(numpy.sort(numpy.array(g['mg_id'])))
        g['fg_dep_ids'] = fg_dep_ids
    else:
        g['fg_dep_ids'] = numpy.array([])
    return g