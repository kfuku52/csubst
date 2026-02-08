import numpy
import pandas

import itertools
import os
import sys
import time

from csubst import parallel
from csubst import combination_cy
from csubst import ete

def node_union(index_combinations, target_nodes, df_mmap, mmap_start):
    arity = target_nodes.shape[1] + 1
    i = mmap_start
    for ic in index_combinations:
        node_union = numpy.union1d(target_nodes[ic[0], :], target_nodes[ic[1], :])
        if (node_union.shape[0] == arity):
            df_mmap[i, :] = node_union
            i += 1

def nc_matrix2id_combinations(nc_matrix, arity, ncpu):
    rows, cols = numpy.where(numpy.equal(nc_matrix, 1))
    unique_cols = numpy.unique(cols)
    ind2  = numpy.arange(arity, dtype=numpy.int64)
    empty_id_combinations = numpy.zeros(shape=(unique_cols.shape[0], arity), dtype=numpy.int64)
    if unique_cols.shape[0] == 0:
        return empty_id_combinations
    n_jobs = parallel.resolve_n_jobs(num_items=unique_cols.shape[0], threads=ncpu)
    if n_jobs == 1:
        return combination_cy.generate_id_chunk(empty_id_combinations, 0, arity, ind2, rows, cols, unique_cols)
    chunks, starts = parallel.get_chunks(empty_id_combinations, n_jobs)
    tasks = [(chunk, start, arity, ind2, rows, cols, unique_cols) for chunk, start in zip(chunks, starts)]
    out = parallel.run_starmap(
        func=combination_cy.generate_id_chunk,
        args_iterable=tasks,
        n_jobs=n_jobs,
        backend='multiprocessing',
    )
    id_combinations = numpy.concatenate(out)
    return id_combinations

def get_node_combinations(g, target_id_dict=None, cb_passed=None, exhaustive=False, cb_all=False, arity=2,
                          check_attr=None, verbose=True):
    if sum([target_id_dict is not None, cb_passed is not None, exhaustive])!=1:
        raise Exception('Only one of target_id_dict, cb_passed, or exhaustive must be set.')
    g['fg_dependent_id_combinations'] = dict()
    tree = g['tree']
    all_nodes = [node for node in tree.traverse() if not ete.is_root(node)]
    if verbose:
        print("Number of all branches: {:,}".format(len(all_nodes)), flush=True)
    if exhaustive:
        target_nodes = list()
        for node in all_nodes:
            if (check_attr is None)|(check_attr in dir(node)):
                target_nodes.append(ete.get_prop(node, "numerical_label"))
        target_nodes = numpy.array(target_nodes)
        node_combinations = list(itertools.combinations(target_nodes, arity))
        node_combinations = [ set(nc) for nc in node_combinations ]
        node_combinations = numpy.array([ list(nc) for nc in node_combinations ])
    if target_id_dict is not None:
        trait_names = list(target_id_dict.keys())
        node_combination_dict = dict()
        is_all_trait_no_branch_combination = True
        for trait_name in trait_names:
            if (target_id_dict[trait_name].shape.__len__()==1):
                target_id_dict[trait_name] = numpy.expand_dims(target_id_dict[trait_name], axis=1)
            index_combinations = list(itertools.combinations(numpy.arange(target_id_dict[trait_name].shape[0]), 2))
            if len(index_combinations) > 0:
                is_all_trait_no_branch_combination = False
                if verbose:
                    txt = 'Number of branch combinations before independency check for {}: {:,}'
                    print(txt.format(trait_name, len(index_combinations)), flush=True)
            else:
                if verbose:
                    txt = 'There is no target branch combination for {} at K = {:,}.\n'
                    sys.stderr.write(txt.format(trait_name, arity))
                    continue
            axis = (len(index_combinations), arity)
            mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.node_combinations.mmap')
            if os.path.exists(mmap_out): os.unlink(mmap_out)
            df_mmap = numpy.memmap(mmap_out, dtype=numpy.int32, shape=axis, mode='w+')
            n_jobs = parallel.resolve_n_jobs(num_items=len(index_combinations), threads=g['threads'])
            if n_jobs == 1:
                node_union(index_combinations, target_id_dict[trait_name], df_mmap, 0)
            else:
                chunk_factor = parallel.resolve_chunk_factor(g=g, task='general')
                chunks, starts = parallel.get_chunks(index_combinations, n_jobs, chunk_factor=chunk_factor)
                # node_union writes into a shared memmap; threads reliably share that memory map.
                backend = 'threading'
                tasks = [(ids, target_id_dict[trait_name], df_mmap, ms) for ids, ms in zip(chunks, starts)]
                parallel.run_starmap(
                    func=node_union,
                    args_iterable=tasks,
                    n_jobs=n_jobs,
                    backend=backend,
                )
            is_valid_combination = (df_mmap.sum(axis=1)!=0)
            if (is_valid_combination.sum()>0):
                node_combination_dict[trait_name] = numpy.unique(df_mmap[is_valid_combination,:], axis=0)
            else:
                node_combination_dict[trait_name] = numpy.zeros(shape=[0,arity], dtype=numpy.int64)
        if is_all_trait_no_branch_combination:
            txt = 'There is no target branch combination for all traits at K = {:,}.\n'
            sys.stderr.write(txt.format(arity))
            id_combinations = numpy.zeros(shape=[0, arity], dtype=numpy.int64)
            return g, id_combinations
        node_combinations = numpy.unique(numpy.concatenate(list(node_combination_dict.values()), axis=0), axis=0)
    if cb_passed is not None:
        node_combinations_dict = dict()
        if cb_all:
            trait_names = ['all',]
        else:
            trait_names = g['fg_df'].columns[1:len(g['fg_df'].columns)].tolist()
        bid_cols = cb_passed.columns[cb_passed.columns.str.startswith('branch_id_')].tolist()
        is_all_trait_no_branch_combination = True
        for trait_name in trait_names:
            if cb_all:
                is_trait = numpy.ones(shape=(cb_passed.shape[0],), dtype=bool)
            else:
                is_trait = False
                is_trait |= (cb_passed.loc[:,'is_fg_'+trait_name]=='Y')
                is_trait |= (cb_passed.loc[:,'is_mf_'+trait_name]=='Y')
                is_trait |= (cb_passed.loc[:,'is_mg_'+trait_name]=='Y')
            bid_trait = cb_passed.loc[is_trait,bid_cols].values
            index_combinations = list(itertools.combinations(numpy.arange(bid_trait.shape[0]), 2))
            if len(index_combinations) > 0:
                is_all_trait_no_branch_combination = False
            else:
                txt = 'There is no target branch combination for {} at K = {:,}.\n'
                sys.stderr.write(txt.format(trait_name, arity))
                continue
            if verbose:
                txt = 'Number of redundant branch combination unions for {}: {:,}'
                print(txt.format(trait_name, len(index_combinations)), flush=True)
            axis = (len(index_combinations), arity)
            mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.node_combinations.mmap')
            if os.path.exists(mmap_out): os.unlink(mmap_out)
            df_mmap = numpy.memmap(mmap_out, dtype=numpy.int32, shape=axis, mode='w+')
            n_jobs = parallel.resolve_n_jobs(num_items=len(index_combinations), threads=g['threads'])
            if n_jobs == 1:
                node_union(index_combinations, bid_trait, df_mmap, 0)
            else:
                chunk_factor = parallel.resolve_chunk_factor(g=g, task='general')
                chunks, starts = parallel.get_chunks(index_combinations, n_jobs, chunk_factor=chunk_factor)
                # node_union writes into a shared memmap; threads reliably share that memory map.
                backend = 'threading'
                tasks = [(ids, bid_trait, df_mmap, ms) for ids, ms in zip(chunks, starts)]
                parallel.run_starmap(
                    func=node_union,
                    args_iterable=tasks,
                    n_jobs=n_jobs,
                    backend=backend,
                )
            is_valid_combination = (df_mmap.sum(axis=1)!=0)
            if (is_valid_combination.sum()>0):
                node_combinations_dict[trait_name] = numpy.unique(df_mmap[is_valid_combination,:], axis=0)
            else:
                node_combinations_dict[trait_name] = numpy.zeros(shape=[0,arity], dtype=numpy.int64)
        if is_all_trait_no_branch_combination:
            txt = 'There is no target branch combination for all traits at K = {:,}.\n'
            sys.stderr.write(txt.format(arity))
            id_combinations = numpy.zeros(shape=[0, arity], dtype=numpy.int64)
            return g, id_combinations
        node_combinations = numpy.unique(numpy.concatenate(list(node_combinations_dict.values()), axis=0), axis=0)
    if verbose:
        print("Number of all branch combinations before independency check: {:,}".format(node_combinations.shape[0]), flush=True)
    nc_matrix = numpy.zeros(shape=(len(all_nodes), node_combinations.shape[0]), dtype=bool)
    for i in numpy.arange(node_combinations.shape[0]):
        nc_matrix[node_combinations[i,:],i] = True
    is_dependent_col = False
    for dep_id in g['dep_ids']:
        is_dependent_col = (is_dependent_col)|(nc_matrix[dep_id,:].sum(axis=0)>1)
    if verbose:
        print('Number of non-independent branch combinations to be removed: {:,}'.format(is_dependent_col.sum()), flush=True)
    nc_matrix = nc_matrix[:,~is_dependent_col]
    id_combinations = numpy.zeros(shape=(0,arity), dtype=numpy.int64)
    start = time.time()
    trait_names = g['fg_df'].columns[1:len(g['fg_df'].columns)].tolist()
    for trait_name in trait_names:
        is_fg_dependent_col = numpy.zeros(shape=(nc_matrix.shape[1],), dtype=bool)
        for fg_dep_id in g['fg_dep_ids'][trait_name]:
            is_fg_dependent_col |= (nc_matrix[fg_dep_id, :].sum(axis=0) > 1)
        if (g['exhaustive_until']>=arity):
            if verbose:
                txt = 'Number of non-independent foreground branch combinations to be non-foreground-marked for {}: {:,} / {:,}'
                print(txt.format(trait_name, is_fg_dependent_col.sum(), is_fg_dependent_col.shape[0]), flush=True)
            fg_dep_nc_matrix = numpy.copy(nc_matrix)
            fg_dep_nc_matrix[:,~is_fg_dependent_col] = False
            g['fg_dependent_id_combinations'][trait_name] = nc_matrix2id_combinations(fg_dep_nc_matrix, arity, g['threads'])
            if trait_name == trait_names[0]:
                id_combinations = nc_matrix2id_combinations(nc_matrix, arity, g['threads'])
        else:
            if (verbose & is_fg_dependent_col.sum() > 0):
                txt = 'Removing {:,} (out of {:,}) non-independent foreground branch combinations for {}.'
                print(txt.format(is_fg_dependent_col.sum(), is_fg_dependent_col.shape[0], trait_name), flush=True)
            nc_matrix = nc_matrix[:,~is_fg_dependent_col]
            g['fg_dependent_id_combinations'][trait_name] = numpy.array([])
            trait_id_combinations = nc_matrix2id_combinations(nc_matrix, arity, g['threads'])
            id_combinations = numpy.unique(numpy.concatenate((id_combinations, trait_id_combinations), axis=0), axis=0)
    if verbose:
        print('Time elapsed for generating branch combinations: {:,} sec'.format(int(time.time() - start)))
        print("Number of independent branch combinations to be analyzed: {:,}".format(id_combinations.shape[0]), flush=True)
    return g,id_combinations

def node_combination_subsamples_rifle(g, arity, rep):
    all_ids = [ ete.get_prop(n, "numerical_label") for n in g['tree'].traverse() ]
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
    all_ids = [ ete.get_prop(n, "numerical_label") for n in g['tree'].traverse() ]
    sub_ids = g['sub_branches']
    id_combinations = numpy.zeros(shape=(0,arity), dtype=numpy.int64)
    id_combinations_dif = numpy.inf
    round = 1
    while (id_combinations.shape[0] < rep)&(id_combinations_dif > rep/200):
        ss_matrix = numpy.zeros(shape=(len(all_ids), rep), dtype=bool, order='C')
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

def get_global_dep_ids(g):
    global_dep_ids = list()
    for leaf in ete.iter_leaves(g['tree']):
        ancestor_nns = [ete.get_prop(node, "numerical_label") for node in ete.iter_ancestors(leaf) if not ete.is_root(node)]
        dep_id = [ete.get_prop(leaf, "numerical_label"), ] + ancestor_nns
        dep_id = numpy.sort(numpy.array(dep_id))
        global_dep_ids.append(dep_id)
        if g['exclude_sister_pair']:
            for node in g['tree'].traverse():
                children = ete.get_children(node)
                if len(children)>1:
                    dep_id = numpy.sort(numpy.array([ ete.get_prop(node, "numerical_label") for node in children ]))
                    global_dep_ids.append(dep_id)
    root_nn = ete.get_prop(g['tree'], "numerical_label")
    root_state_sum = g['state_cdn'][root_nn, :, :].sum()
    if (root_state_sum == 0):
        print('Ancestral states were not estimated on the root node. Excluding sub-root nodes from the analysis.')
        subroot_nns = [ete.get_prop(node, "numerical_label") for node in ete.get_children(g['tree'])]
        for subroot_nn in subroot_nns:
            for node in g['tree'].traverse():
                if ete.is_root(node):
                    continue
                if subroot_nn == ete.get_prop(node, "numerical_label"):
                    continue
                ancestor_nns = [ete.get_prop(anc, "numerical_label") for anc in ete.iter_ancestors(node)]
                if subroot_nn in ancestor_nns:
                    continue
                global_dep_ids.append(numpy.array([subroot_nn, ete.get_prop(node, "numerical_label")]))
    return global_dep_ids

def get_foreground_dep_ids(g):
    fg_dep_ids = dict()
    for trait_name in g['fg_df'].columns[1:len(g['fg_df'].columns)]:
        if (g['foreground'] is not None)&(g['fg_exclude_wg']):
            fg_dep_ids[trait_name] = list()
            for i in numpy.arange(len(g['fg_leaf_names'][trait_name])):
                fg_lineage_leaf_names = g['fg_leaf_names'][trait_name][i]
                tmp_fg_dep_ids = list()
                for node in g['tree'].traverse():
                    if ete.is_root(node):
                        continue
                    is_all_leaf_lineage_fg = all([ln in fg_lineage_leaf_names for ln in ete.get_leaf_names(node)])
                    if not is_all_leaf_lineage_fg:
                        continue
                    is_up_all_leaf_lineage_fg = all([ln in fg_lineage_leaf_names for ln in ete.get_leaf_names(node.up)])
                    if is_up_all_leaf_lineage_fg:
                        continue
                    if ete.is_leaf(node):
                        tmp_fg_dep_ids.append(ete.get_prop(node, "numerical_label"))
                    else:
                        descendant_nn = [ ete.get_prop(n, "numerical_label") for n in ete.get_descendants(node) ]
                        tmp_fg_dep_ids += [ete.get_prop(node, "numerical_label"),] + descendant_nn
                if len(tmp_fg_dep_ids)>1:
                    fg_dep_ids[trait_name].append(numpy.sort(numpy.array(tmp_fg_dep_ids)))
            if (g['mg_sister'])|(g['mg_parent']):
                fg_dep_ids[trait_name].append(numpy.sort(numpy.array(g['mg_ids'][trait_name])))
        else:
            fg_dep_ids[trait_name] = numpy.array([])
    return fg_dep_ids

def get_dep_ids(g):
    g['dep_ids'] = get_global_dep_ids(g)
    g['fg_dep_ids'] = get_foreground_dep_ids(g)
    return g
