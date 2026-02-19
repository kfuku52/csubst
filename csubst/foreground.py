import numpy
import pandas

import copy
import itertools
import re
import sys
import time
import warnings

from csubst import combination
from csubst import omega
from csubst import table
from csubst import param
from csubst import ete
from csubst import substitution
from csubst import tree
from csubst import output_stat


def _get_trait_names(g):
    return g['fg_df'].columns[1:len(g['fg_df'].columns)]


def _get_node_by_branch_id(g):
    node_by_id = dict()
    for node in g['tree'].traverse():
        branch_id = int(ete.get_prop(node, "numerical_label"))
        node_by_id[branch_id] = node
    return node_by_id


def _count_branch_memberships(cb, bid_cols, ids):
    if len(bid_cols) == 0:
        return numpy.zeros(shape=(cb.shape[0],), dtype=numpy.int64)
    id_list = list(ids)
    if len(id_list) == 0:
        return numpy.zeros(shape=(cb.shape[0],), dtype=numpy.int64)
    bid_matrix = cb.loc[:, bid_cols].to_numpy(copy=False)
    return numpy.isin(bid_matrix, id_list).sum(axis=1).astype(numpy.int64)


def _mark_dependent_foreground_rows(cb, bid_cols, trait_name, dependent_id_combinations):
    if dependent_id_combinations.shape[0] == 0:
        return cb
    col_is_fg = 'is_fg_' + trait_name
    for bids in dependent_id_combinations:
        conditions = [(cb[bid_col] == bid) for bid_col, bid in zip(bid_cols, bids)]
        is_dep = numpy.logical_and.reduce(conditions)
        cb.loc[is_dep, col_is_fg] = 'N'
    return cb


def _assign_trait_labels(cb, trait_name, arity):
    col_num_fg = 'branch_num_fg_' + trait_name
    col_num_mg = 'branch_num_mg_' + trait_name
    col_is_fg = 'is_fg_' + trait_name
    col_is_mg = 'is_mg_' + trait_name
    col_is_mf = 'is_mf_' + trait_name
    cb.loc[:, col_is_fg] = 'N'
    cb.loc[cb.loc[:, col_num_fg] == arity, col_is_fg] = 'Y'
    cb.loc[:, col_is_mg] = 'N'
    cb.loc[cb.loc[:, col_num_mg] == arity, col_is_mg] = 'Y'
    cb.loc[:, col_is_mf] = 'N'
    is_mf = (cb.loc[:, col_num_fg] > 0) & (cb.loc[:, col_num_mg] > 0)
    is_mf = is_mf & ((cb.loc[:, col_num_fg] + cb.loc[:, col_num_mg]) == arity)
    cb.loc[is_mf, col_is_mf] = 'Y'
    return cb


def _set_target_label_column(df, column_name, positive_index, positive='Y', negative='N'):
    df.loc[:, column_name] = negative
    if len(positive_index) > 0:
        df.loc[positive_index, column_name] = positive
    return df


def _calculate_fg_enrichment(num_enough, num_fg, num_fg_enough, num_all):
    if (num_enough == 0) or (num_fg == 0):
        return 0, 0
    percent_fg_enough = num_fg_enough / num_enough * 100
    enrichment_factor = (num_fg_enough / num_enough) / (num_fg / num_all)
    return percent_fg_enough, enrichment_factor


def combinations_count(n, r):
    # https://github.com/nkmk/python-snippets/blob/05a53ae96736577906a8805b38bce6cc210fe11f/notebook/combinations_count.py#L1-L14
    from operator import mul
    from functools import reduce
    r = min(r, n - r)
    numer = reduce(mul, range(n, n - r, -1), 1)
    denom = reduce(mul, range(1, r + 1), 1)
    return numer // denom

def get_df_clade_size(g, trait_name):
    branch_ids = sorted([
        int(ete.get_prop(n, "numerical_label"))
        for n in g['tree'].traverse()
        if not ete.is_root(n)
    ])
    cols = ['branch_id','size','is_fg_stem_'+trait_name]
    df_clade_size = pandas.DataFrame(index=branch_ids, columns=cols)
    df_clade_size.loc[:,'branch_id'] = branch_ids
    df_clade_size.loc[:,'is_fg_stem_'+trait_name] = False
    for node in g['tree'].traverse():
        if ete.is_root(node):
            continue
        bid = ete.get_prop(node, "numerical_label")
        df_clade_size.at[bid,'size'] = len(ete.get_leaf_names(node))
        is_fg = ete.get_prop(node, 'is_fg_' + trait_name, False)
        is_parent_fg = ete.get_prop(node.up, 'is_fg_' + trait_name, False)
        if (is_fg)&(~is_parent_fg):
            df_clade_size.at[bid,'is_fg_stem_'+trait_name] = True
    return df_clade_size


def _build_clade_permutation_bins(df_clade_size, trait_name, min_clade_bin_count, sample_original_foreground):
    size_array = df_clade_size.loc[:, 'size'].values.astype(numpy.int64)
    size_min = size_array.min()
    size_max = size_array.max()
    sizes = numpy.unique(size_array)[::-1]  # To start counting from rarer (larger) clades
    bins = numpy.array([size_max + 1], dtype=numpy.int64)
    count = 0
    counts = []
    for size in sizes:
        if sample_original_foreground:
            is_size = (size_array == size)
        else:
            is_size = ((size_array == size) & (~df_clade_size.loc[:, 'is_fg_stem_' + trait_name]))
        count += is_size.sum()
        if count >= min_clade_bin_count:
            bins = numpy.append(bins, size)
            counts.append(count)
            count = 0
    if len(bins) < 2:
        bins = numpy.array([size_min, size_max], dtype=numpy.int64)
    return bins, counts, size_array


def _randomize_foreground_flags(before_randomization, sample_original_foreground):
    if sample_original_foreground:
        return numpy.random.permutation(before_randomization)
    ind_fg = numpy.where(before_randomization == True)[0]
    ind_nonfg = numpy.where(before_randomization == False)[0]
    ind_rfg = numpy.random.choice(ind_nonfg, ind_fg.shape[0])
    after_randomization = numpy.zeros_like(before_randomization, dtype=bool)
    after_randomization[ind_rfg] = True
    return after_randomization


def _block_randomized_foreground_descendants(df_clade_size, is_bin, node_by_id):
    is_blocked = df_clade_size.loc[:, 'is_blocked'].values
    is_new_fg = is_bin & ~is_blocked & (df_clade_size.loc[:, 'is_fg_stem_randomized'] == True)
    new_fg_bids = df_clade_size.loc[is_new_fg, 'branch_id'].values
    for new_fg_bid in new_fg_bids:
        node = node_by_id.get(int(new_fg_bid), None)
        if node is None:
            continue
        des_bids = [ete.get_prop(d, "numerical_label") for d in node.traverse()]
        df_clade_size.loc[des_bids, 'is_blocked'] = True
    return df_clade_size


def foreground_clade_randomization(df_clade_size, g, trait_name, sample_original_foreground=False):
    bins, counts, size_array = _build_clade_permutation_bins(
        df_clade_size=df_clade_size,
        trait_name=trait_name,
        min_clade_bin_count=g['min_clade_bin_count'],
        sample_original_foreground=sample_original_foreground,
    )
    txt = 'Number of clade permutation bins = {:,} (bin limits = {}, counts = {})'
    print(txt.format(bins.shape[0]-1, ', '.join([ str(s) for s in bins ]), ', '.join([ str(s) for s in counts ])))
    bins = bins[::-1]
    df_clade_size.loc[:,'bin'] = numpy.digitize(size_array, bins, right=False)
    is_fg = (df_clade_size.loc[:,'is_fg_stem_'+trait_name]==True)
    fg_bins = df_clade_size.loc[is_fg,'bin']
    df_clade_size.loc[:,'is_fg_stem_randomized'] = df_clade_size.loc[:,'is_fg_stem_'+trait_name]
    df_clade_size.loc[:,'is_blocked'] = False
    node_by_id = _get_node_by_branch_id(g)
    for bin in fg_bins.unique():
        is_bin = (df_clade_size.loc[:,'bin']==bin)
        is_blocked = df_clade_size.loc[:,'is_blocked'].values
        before_randomization = df_clade_size.loc[is_bin&~is_blocked,'is_fg_stem_randomized'].values
        after_randomization = _randomize_foreground_flags(
            before_randomization=before_randomization,
            sample_original_foreground=sample_original_foreground,
        )
        df_clade_size.loc[is_bin&~is_blocked,'is_fg_stem_randomized'] = after_randomization
        df_clade_size = _block_randomized_foreground_descendants(
            df_clade_size=df_clade_size,
            is_bin=is_bin,
            node_by_id=node_by_id,
        )
    return df_clade_size

def get_new_foreground_ids(df_clade_size, g):
    is_new_fg = (df_clade_size.loc[:,'is_fg_stem_randomized']==True)
    fg_stem_bids = df_clade_size.loc[is_new_fg,'branch_id'].values
    new_fg_ids = list()
    if (g['fg_stem_only']):
        new_fg_ids = fg_stem_bids
    else:
        node_by_id = _get_node_by_branch_id(g)
        for fg_stem_bid in fg_stem_bids:
            node = node_by_id.get(int(fg_stem_bid), None)
            if node is None:
                continue
            new_lineage_fg_ids = [ete.get_prop(n, "numerical_label") for n in node.traverse()]
            new_fg_ids += new_lineage_fg_ids
    new_fg_ids = numpy.unique(numpy.array(new_fg_ids, dtype=numpy.int64))
    return new_fg_ids

def get_lineage_color_list():
    lineage_colors = [
        'red',
        'blue',
        'darkorange',
        'brown',
        'mediumseagreen',
        'purple',
        'hotpink',
        'olive',
        'darkturquoise',
        'green',
        'darkorchid',
    ]
    return lineage_colors

def get_fg_leaf_names(lineages, trait_name, g):
    fg_leaf_names = []
    all_leaf_names = ete.get_leaf_names(g['tree'])
    for i in numpy.arange(len(lineages)):
        fg_leaf_names.append([])
        is_lineage = (g['fg_df'].loc[:, trait_name] == lineages[i])
        lineage_regex_names = g['fg_df'].loc[is_lineage, :].loc[:, 'name'].unique().tolist()
        iter = itertools.product(all_leaf_names, lineage_regex_names)
        lineage_leaf_names = [ln for ln, lr in iter if re.match('^' + lr + '$', ln)]
        for lln in lineage_leaf_names:
            match_leaves = [ln for ln in all_leaf_names if lln == ln]
            if len(match_leaves) == 1:
                fg_leaf_names[i].append(match_leaves[0])
            else:
                print('The foreground leaf name cannot be identified:', lln, match_leaves)
    return fg_leaf_names

def annotate_lineage_foreground(lineages, trait_name, g):
    tree = g['tree']
    for i in numpy.arange(len(lineages)):
        fg_leaf_name_set = set(g['fg_leaf_names'][trait_name][i])
        for node in tree.traverse():
            node_leaf_name_set = set(ete.get_leaf_names(node))
            if len(node_leaf_name_set.difference(fg_leaf_name_set)) == 0:
                ete.add_features(node, **{'is_lineage_fg_'+trait_name+'_'+str(i+1): True})
            else:
                ete.add_features(node, **{'is_lineage_fg_'+trait_name+'_'+str(i+1): False})
    return tree

def _get_lineage_target_ids(lineage_index, trait_name, g):
    fg_leaf_name_set = set(g['fg_leaf_names'][trait_name][lineage_index])
    lineage_fg_ids = list()
    root_id = ete.get_prop(ete.get_tree_root(g['tree']), "numerical_label")
    lineage_flag_key = 'is_lineage_fg_' + trait_name + '_' + str(lineage_index + 1)
    for node in g['tree'].traverse():
        node_leaf_name_set = set(ete.get_leaf_names(node))
        if len(node_leaf_name_set.difference(fg_leaf_name_set)) != 0:
            continue
        if ete.is_root(node) or (not g['fg_stem_only']):
            lineage_fg_ids.append(ete.get_prop(node, "numerical_label"))
        else:
            is_lineage_fg = ete.get_prop(node, lineage_flag_key, False)
            is_parent_lineage_fg = ete.get_prop(node.up, lineage_flag_key, False)
            if (is_lineage_fg == True) & (is_parent_lineage_fg == False):
                lineage_fg_ids.append(ete.get_prop(node, "numerical_label"))
    dif = 1
    while dif:
        num_id = len(lineage_fg_ids)
        lineage_fg_id_set = set(lineage_fg_ids)
        for node in g['tree'].traverse():
            child_ids = [ete.get_prop(child, "numerical_label") for child in ete.get_children(node)]
            if all([id in lineage_fg_id_set for id in child_ids]) & (len(child_ids) != 0):
                node_id = ete.get_prop(node, "numerical_label")
                if node_id not in lineage_fg_id_set:
                    lineage_fg_ids.append(node_id)
        dif = len(lineage_fg_ids) - num_id
    lineage_fg_ids = numpy.unique(numpy.array(lineage_fg_ids, dtype=numpy.int64))
    lineage_fg_ids = lineage_fg_ids[lineage_fg_ids != root_id]
    lineage_fg_ids.sort()
    return lineage_fg_ids

def get_target_ids(lineages, trait_name, g):
    target_ids = numpy.zeros(shape=(0,), dtype=numpy.int64)
    for i in numpy.arange(len(lineages)):
        lineage_fg_ids = _get_lineage_target_ids(lineage_index=i, trait_name=trait_name, g=g)
        target_ids = numpy.concatenate([target_ids, lineage_fg_ids])
    target_ids = numpy.unique(target_ids.astype(numpy.int64))
    target_ids.sort()
    return target_ids

def annotate_foreground(lineages, trait_name, g):
    for node in g['tree'].traverse(): # Initialize
        ete.add_features(node, **{'is_fg_'+trait_name: False})
        ete.add_features(node, **{'foreground_lineage_id_' + trait_name: 0})
        ete.add_features(node, **{'color_'+trait_name: 'black'})
        ete.add_features(node, **{'labelcolor_' + trait_name: 'black'})
    lineage_colors = get_lineage_color_list()
    for i in numpy.arange(len(lineages)):
        fg_leaf_name_set = set(g['fg_leaf_names'][trait_name][i])
        lineage_color = lineage_colors[i % len(lineage_colors)]
        lineage_prop = 'is_lineage_fg_' + trait_name + '_' + str(i + 1)
        lineage_target_ids = set(_get_lineage_target_ids(lineage_index=i, trait_name=trait_name, g=g).tolist())
        for node in g['tree'].traverse():
            if g['fg_stem_only']:
                if ete.get_prop(node, "numerical_label") in lineage_target_ids:
                    ete.add_features(node, **{'is_fg_'+trait_name: True})
                    ete.add_features(node, **{'foreground_lineage_id_' + trait_name: int(i + 1)})
                    ete.add_features(node, **{'color_'+trait_name: lineage_color})
                    ete.add_features(node, **{'labelcolor_'+trait_name: lineage_color})
                if node.name in fg_leaf_name_set:
                    ete.add_features(node, **{'labelcolor_' + trait_name: lineage_color})
            else:
                is_lineage_fg = ete.get_prop(node, lineage_prop, False)
                if is_lineage_fg == True:
                    ete.add_features(node, **{'is_fg_' + trait_name: True})
                    ete.add_features(node, **{'foreground_lineage_id_' + trait_name: int(i + 1)})
                    ete.add_features(node, **{'color_' + trait_name: lineage_color})
                    ete.add_features(node, **{'labelcolor_' + trait_name: lineage_color})
    return g['tree']

def get_foreground_ids(g, write=True):
    g['fg_leaf_names'] = dict()
    g['target_ids'] = dict()
    g['fg_ids'] = dict()
    for trait_name in _get_trait_names(g):
        lineages = g['fg_df'].loc[:,trait_name].unique()
        lineages = lineages[lineages!=0]
        g['fg_leaf_names'][trait_name] = get_fg_leaf_names(lineages, trait_name, g)
        g['tree'] = annotate_lineage_foreground(lineages, trait_name, g)
        g['target_ids'][trait_name] = get_target_ids(lineages, trait_name, g)
        g['tree'] = annotate_foreground(lineages, trait_name, g)
        g['fg_ids'][trait_name] = copy.deepcopy(g['target_ids'][trait_name]) # marginal_ids may be added to target_id but fg_id won't be changed.
        if write:
            file_name = 'csubst_target_branch_'+trait_name+'.txt'
            file_name = file_name.replace('_PLACEHOLDER', '')
            with open(file_name, 'w') as f:
                for x in g['target_ids'][trait_name]:
                    f.write(str(x)+'\n')
    return g

def read_foreground_file(g):
    if g['fg_format'] == 1:
        fg_df = pandas.read_csv(g['foreground'], sep='\t', comment='#', skip_blank_lines=True, header=None)
        if fg_df.shape[1]!=2:
            txt = 'With --fg_format 1, --foreground file should be a tab-separated two-column table without header. '
            txt += 'First column = lineage IDs; Second column = Regex-compatible sequence names'
            raise Exception(txt)
        fg_df = fg_df.iloc[:,[1,0]]
        fg_df.columns = ['name','PLACEHOLDER']
    elif g['fg_format'] == 2:
        fg_df = pandas.read_csv(g['foreground'], sep='\t', comment='#', skip_blank_lines=True, header=0)
        if fg_df.shape[1]<2:
            txt = 'With --fg_format 2, --foreground file should be a tab-separated table with a header line and 2 or more columns. '
            txt += 'Header names should be "name", "TRAIT1", "TRAIT2", ..., where any trait names are allowed. '
            txt += 'First column = Regex-compatible sequence names; Second column and after = lineage IDs (0 = background)'
            raise Exception(txt)
        fg_df.columns = ['name'] + fg_df.columns[1:len(fg_df.columns)].tolist()
        txt = 'Trait names in --foreground file: {}'.format(', '.join(fg_df.columns[1:len(fg_df.columns)].tolist()))
        print(txt, flush=True)
    return fg_df

def dummy_foreground_annotation(tree, trait_name):
    for node in tree.traverse():
        ete.add_features(node, **{'is_lineage_fg_' + trait_name + '_1': False})
        ete.add_features(node, **{'is_fg_' + trait_name: False})
        ete.add_features(node, **{'foreground_lineage_id_' + trait_name: 0})
        ete.add_features(node, **{'color_' + trait_name: 'black'})
        ete.add_features(node, **{'labelcolor_' + trait_name: 'black'})
    return tree

def get_foreground_branch(g, simulate=False):
    if g['foreground'] is None:
        trait_name = 'PLACEHOLDER'
        g['tree'] = dummy_foreground_annotation(tree=g['tree'], trait_name=trait_name)
        g['fg_df'] = pandas.DataFrame(columns=['name', trait_name])
        g['fg_leaf_names'] = {trait_name: []}
        g['fg_ids'] = {trait_name: numpy.zeros(shape=(0,), dtype=numpy.int64)}
        g['target_ids'] = {trait_name: numpy.zeros(shape=(0,), dtype=numpy.int64)}
    else:
        g['fg_df'] = read_foreground_file(g)
        if simulate:
            if g['fg_format'] == 2:
                first_trait_name = g['fg_df'].columns[1]
                txt = 'With --fg_format 2, only the first trait column ({}) will be used for simulated molecular evolution.\n'
                sys.stderr.write(txt.format(first_trait_name))
            if g['fg_df'].shape[1] > 2:
                g['fg_df'] = g['fg_df'].iloc[:,[0,1]]
            g['fg_df'].columns = ['name','PLACEHOLDER']
        g = get_foreground_ids(g=g, write=True)
    return g

def print_num_possible_permuted_combinations(df_clade_size, trait_name, sample_original_foreground):
    import scipy
    num_possible_permutation_combination = 1
    is_fg_stem = df_clade_size.loc[:, 'is_fg_stem_'+trait_name].values
    for bin_no in df_clade_size.loc[:, 'bin'].unique():
        is_bin = (df_clade_size.loc[:, 'bin'] == bin_no)
        num_bin_fg = (is_bin & is_fg_stem).sum()
        if num_bin_fg == 0:
            continue
        if sample_original_foreground:
            num_bin_choice = is_bin.sum()
        else:
            num_bin_choice = (is_bin & ~is_fg_stem).sum()
        num_possible_permutation_combination_bin = scipy.special.comb(N=num_bin_choice, k=num_bin_fg, repetition=False)
        num_possible_permutation_combination *= num_possible_permutation_combination_bin
    txt = 'Number of possible clade-permuted combinations without considering branch dependency = {:,}'
    print(txt.format(int(num_possible_permutation_combination)))

def randomize_foreground_branch(g, trait_name, sample_original_foreground=False):
    g['r_target_ids'] = dict()
    g['r_fg_ids'] = dict()
    g['r_fg_leaf_names'] = dict()
    df_clade_size = get_df_clade_size(g, trait_name)
    r_df_clade_size = foreground_clade_randomization(df_clade_size, g, trait_name, sample_original_foreground)
    print_num_possible_permuted_combinations(r_df_clade_size, trait_name, sample_original_foreground)
    g['r_target_ids'][trait_name] = get_new_foreground_ids(r_df_clade_size, g)
    g['r_fg_ids'][trait_name] = copy.deepcopy(g['r_target_ids'][trait_name])
    new_fg_leaf_names = [ete.get_leaf_names(n) for n in g['tree'].traverse() if ete.get_prop(n, "numerical_label") in g['r_fg_ids'][trait_name]]
    new_fg_leaf_names = list(itertools.chain(*new_fg_leaf_names))
    new_fg_leaf_names = list(set(new_fg_leaf_names))
    g['r_fg_leaf_names'][trait_name] = new_fg_leaf_names
    return g

def get_marginal_branch(g):
    g['mg_ids'] = dict()
    for trait_name in _get_trait_names(g):
        g['mg_ids'][trait_name] = list()
        for node in g['tree'].traverse():
            if ete.is_root(node):
                continue
            is_fg = ete.get_prop(node, 'is_fg_' + trait_name, False)
            if (is_fg==False):
                continue
            if (g['mg_parent']):
                is_parent_fg = ete.get_prop(node.up, 'is_fg_' + trait_name, False)
                if (is_parent_fg==False):
                    g['mg_ids'][trait_name].append(ete.get_prop(node.up, "numerical_label"))
            if (g['mg_sister']):
                sisters = ete.get_sisters(node)
                for sister in sisters:
                    if (g['mg_sister_stem_only']):
                        is_sister_fg = ete.get_prop(sister, 'is_fg_' + trait_name, False)
                        if is_sister_fg==False:
                            g['mg_ids'][trait_name].append(ete.get_prop(sister, "numerical_label"))
                    else:
                        for sister_des in sister.traverse():
                            is_sister_des_fg = ete.get_prop(sister_des, 'is_fg_' + trait_name, False)
                            if is_sister_des_fg==False:
                                g['mg_ids'][trait_name].append(ete.get_prop(sister_des, "numerical_label"))
        concat_ids = list(set(g['mg_ids'][trait_name])-set(g['target_ids'][trait_name]))
        g['mg_ids'][trait_name] = numpy.array(concat_ids, dtype=numpy.int64)
        g['target_ids'][trait_name] = numpy.concatenate([g['target_ids'][trait_name], g['mg_ids'][trait_name]])
        for node in g['tree'].traverse():
            if ete.get_prop(node, "numerical_label") in g['mg_ids'][trait_name]:
                ete.add_features(node, **{'is_marginal_'+trait_name: True})
            else:
                ete.add_features(node, **{'is_marginal_'+trait_name: False})
        file_name = 'csubst_marginal_branch_' + trait_name + '.txt'
        file_name = file_name.replace('_PLACEHOLDER', '')
        if len(g['mg_ids'][trait_name]) > 0:
            with open(file_name, 'w') as f:
                for x in g['mg_ids'][trait_name]:
                    f.write(str(x)+'\n')
    return g

def get_foreground_branch_num(cb, g):
    start_time = time.time()
    bid_cols = cb.columns[cb.columns.str.startswith('branch_id_')]
    arity = len(bid_cols)
    trait_names = _get_trait_names(g)
    for trait_name in trait_names:
        col_num_fg = 'branch_num_fg_' + trait_name
        col_num_mg = 'branch_num_mg_' + trait_name
        col_num_fg_stem = 'branch_num_fg_stem_' + trait_name
        col_is_fg = 'is_fg_' + trait_name
        cb.loc[:, col_num_fg] = _count_branch_memberships(cb=cb, bid_cols=bid_cols, ids=g['fg_ids'][trait_name])
        cb.loc[:, col_num_mg] = _count_branch_memberships(cb=cb, bid_cols=bid_cols, ids=g['mg_ids'][trait_name])
        cb = _assign_trait_labels(cb=cb, trait_name=trait_name, arity=arity)
        cb = _mark_dependent_foreground_rows(
            cb=cb,
            bid_cols=bid_cols,
            trait_name=trait_name,
            dependent_id_combinations=g['fg_dependent_id_combinations'][trait_name],
        )
        df_clade_size = get_df_clade_size(g, trait_name)
        fg_stem_bids = df_clade_size.loc[df_clade_size.loc[:,'is_fg_stem_'+trait_name],'branch_id'].values
        cb.loc[:, col_num_fg_stem] = _count_branch_memberships(cb=cb, bid_cols=bid_cols, ids=fg_stem_bids)
        is_fg = (cb[col_is_fg] == 'Y')
        is_enough_stat = table.get_cutoff_stat_bool_array(cb=cb, cutoff_stat_str=g['cutoff_stat'])
        num_enough = is_enough_stat.sum()
        num_fg = is_fg.sum()
        num_fg_enough = (is_enough_stat&is_fg).sum()
        num_all = cb.shape[0]
        percent_fg_enough, enrichment_factor = _calculate_fg_enrichment(
            num_enough=num_enough,
            num_fg=num_fg,
            num_fg_enough=num_fg_enough,
            num_all=num_all,
        )
        txt = 'Arity (K) = {}: Foreground branch combinations with cutoff conditions {} for {} = {:.0f}% ({:,}/{:,}, ' \
              'total examined = {:,}, enrichment factor = {:.1f})'
        txt = txt.format(arity, g['cutoff_stat'], trait_name, percent_fg_enough, num_fg_enough, num_enough,
                         num_all, enrichment_factor)
        print(txt, flush=True)
        if not 'fg_enrichment_factor_'+trait_name in g['df_cb_stats'].columns:
            g['df_cb_stats']['fg_enrichment_factor_'+trait_name] = numpy.nan
        g['df_cb_stats'].at[0,'fg_enrichment_factor_'+trait_name] = enrichment_factor
    txt = 'Time elapsed for obtaining foreground branch numbers in the cb table: {:,} sec'
    print(txt.format(int(time.time() - start_time)))
    return cb, g

def annotate_b_foreground(b, g):
    trait_names = _get_trait_names(g)
    for trait_name in trait_names:
        col_fg = 'is_fg_' + trait_name
        col_mg = 'is_mg_' + trait_name
        if (g['foreground'] is None):
            b.loc[:, col_fg] = 'Y'
            b.loc[:, col_mg] = 'N'
        else:
            b = _set_target_label_column(
                df=b,
                column_name=col_fg,
                positive_index=g['fg_ids'][trait_name],
                positive='Y',
                negative='N',
            )
            b = _set_target_label_column(
                df=b,
                column_name=col_mg,
                positive_index=g['mg_ids'][trait_name],
                positive='Y',
                negative='N',
            )
    return b

def get_num_foreground_lineages(tree, trait_name):
    num_fl = 0
    prefix = 'is_lineage_fg_' + trait_name + '_'
    for node in tree.traverse():
        if hasattr(node, 'props'):
            keys = node.props.keys()
        elif hasattr(node, '__dict__'):
            keys = node.__dict__.keys()
        else:
            keys = []
        for key in keys:
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            if suffix.isdigit():
                num_fl = max(num_fl, int(suffix))
    return num_fl


def _try_randomized_foreground_combinations(g, trait_name, current_arity, sample_original_foreground):
    g = randomize_foreground_branch(g, trait_name, sample_original_foreground=sample_original_foreground)
    if g['r_target_ids'][trait_name].shape[0] < 2:
        return g, None
    g, rid_combinations = combination.get_node_combinations(
        g=g,
        target_id_dict=g['r_target_ids'],
        arity=current_arity,
        check_attr="name",
        verbose=False,
    )
    if rid_combinations.shape[0] == 0:
        return g, None
    return g, rid_combinations


def _raise_foreground_permutation_failure(num_trial, sample_original_foreground):
    if sample_original_foreground:
        txt = 'Foreground branch permutation failed {:,} times even when allowing sampling from original foreground clades.'
    else:
        txt = 'Foreground branch permutation failed {:,} times. There may not be enough numbers of "similar" clades.'
    raise Exception(txt.format(num_trial))


def set_random_foreground_branch(g, trait_name, num_trial=100, sample_original_foreground=False):
    # Refresh tree foreground annotations once before randomization attempts.
    g = get_foreground_branch(g)
    for i in numpy.arange(num_trial):
        g, rid_combinations = _try_randomized_foreground_combinations(
            g=g,
            trait_name=trait_name,
            current_arity=g['current_arity'],
            sample_original_foreground=sample_original_foreground,
        )
        if rid_combinations is None:
            continue
        print('Foreground branch permutation finished after {:,} trials.'.format(i + 1), flush=True)
        return g, rid_combinations
    _raise_foreground_permutation_failure(
        num_trial=num_trial,
        sample_original_foreground=sample_original_foreground,
    )


def _build_cb_target_masks(cb, trait_names, target_col_prefixes):
    is_targets = dict()
    for trait_name in trait_names:
        for target_col_prefix in target_col_prefixes:
            target_col = target_col_prefix + '_' + trait_name
            if target_col in cb.columns:
                suffix = target_col.replace('is_', '_')
                is_targets[suffix] = (cb[target_col] == 'Y')
            elif target_col_prefix == 'is_all':
                is_targets['_all'] = numpy.ones(shape=(cb.shape[0],), dtype=bool)
    return is_targets


def _resolve_cb_stats_for_median_and_total(cb):
    omega_cols = cb.columns[cb.columns.str.startswith('omegaC')].tolist()
    is_ON = cb.columns.str.startswith('OCNany') | cb.columns.str.startswith('OCNdif') | cb.columns.str.startswith('OCNspe')
    is_OS = cb.columns.str.startswith('OCSany') | cb.columns.str.startswith('OCSdif') | cb.columns.str.startswith('OCSspe')
    is_EN = cb.columns.str.startswith('ECNany') | cb.columns.str.startswith('ECNdif') | cb.columns.str.startswith('ECNspe')
    is_ES = cb.columns.str.startswith('ECSany') | cb.columns.str.startswith('ECSdif') | cb.columns.str.startswith('ECSspe')
    ON_cols = cb.columns[is_ON].tolist()
    OS_cols = cb.columns[is_OS].tolist()
    EN_cols = cb.columns[is_EN].tolist()
    ES_cols = cb.columns[is_ES].tolist()
    stats = dict()
    stats['median'] = ['dist_bl', 'dist_node_num'] + omega_cols
    stats['total'] = ON_cols + OS_cols + EN_cols + ES_cols
    return stats


def _append_cb_target_count_columns(df_cb_stats, is_targets, is_qualified):
    new_columns = {}
    for suffix, is_target in is_targets.items():
        new_columns['num' + suffix] = is_target.sum()
        new_columns['num_qualified' + suffix] = (is_target & is_qualified).sum()
    new_columns_df = pandas.DataFrame(new_columns, index=df_cb_stats.index)
    return pandas.concat([df_cb_stats, new_columns_df], axis=1)


def _ensure_df_cb_stats_column(df_cb_stats, col):
    if col in df_cb_stats.columns:
        return df_cb_stats
    newcol = pandas.DataFrame({col: numpy.zeros(shape=(df_cb_stats.shape[0]))})
    return pandas.concat([df_cb_stats, newcol], ignore_index=False, axis=1)


def _aggregate_cb_stats(df_cb_stats, cb, stats, is_targets):
    for stat in stats.keys():
        for suffix, is_target in is_targets.items():
            for ms in stats[stat]:
                col = stat + '_' + ms + suffix
                df_cb_stats = _ensure_df_cb_stats_column(df_cb_stats=df_cb_stats, col=col)
                if ms not in cb.columns:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if stat == 'median':
                        df_cb_stats.at[0, col] = cb.loc[is_target, ms].median()
                    elif stat == 'total':
                        df_cb_stats.at[0, col] = cb.loc[is_target, ms].sum()
    return df_cb_stats


def _print_total_oc_ec_summary(df_cb_stats):
    oc_total_cols = [
        col for col in df_cb_stats.columns
        if str(col).startswith('total_OC') and str(col).endswith('_all')
    ]
    for oc_col in sorted(oc_total_cols):
        key = str(oc_col).replace('total_OC', '').replace('_all', '')
        ec_col = 'total_EC' + key + '_all'
        if ec_col not in df_cb_stats.columns:
            continue
        total_oc = df_cb_stats.at[0, oc_col]
        total_ec = df_cb_stats.at[0, ec_col]
        if total_oc == 0:
            percent_value = numpy.nan
        else:
            percent_value = total_ec / total_oc * 100
        txt = 'Total OC{}/EC{} = {:,.1f}/{:,.1f} (expectation = {:,.1f}% of observation)'
        print(txt.format(key, key, total_oc, total_ec, percent_value))


def add_median_cb_stats(g, cb, current_arity, start, verbose=True):
    target_col_prefixes = ['is_fg', 'is_mg', 'is_mf', 'is_all']
    trait_names = _get_trait_names(g)
    is_targets = _build_cb_target_masks(
        cb=cb,
        trait_names=trait_names,
        target_col_prefixes=target_col_prefixes,
    )
    stats = _resolve_cb_stats_for_median_and_total(cb)
    is_qualified = table.get_cutoff_stat_bool_array(cb=cb, cutoff_stat_str=g['cutoff_stat'])
    g['df_cb_stats'] = _append_cb_target_count_columns(
        df_cb_stats=g['df_cb_stats'],
        is_targets=is_targets,
        is_qualified=is_qualified,
    )
    g['df_cb_stats'] = _aggregate_cb_stats(
        df_cb_stats=g['df_cb_stats'],
        cb=cb,
        stats=stats,
        is_targets=is_targets,
    )
    if verbose:
        _print_total_oc_ec_summary(df_cb_stats=g['df_cb_stats'])
    elapsed_time = int(time.time() - start)
    g['df_cb_stats'].at[0, 'elapsed_sec'] = elapsed_time
    if verbose:
        print(("Elapsed time for arity = {}: {:,.1f} sec\n".format(current_arity, elapsed_time)), flush=True)
    return g

def _recompute_missing_permutation_rows(g, missing_id_combinations, OS_tensor_reducer, ON_tensor_reducer):
    cbOS = substitution.get_cb(
        missing_id_combinations,
        OS_tensor_reducer,
        g,
        'OCS',
        selected_base_stats=g.get('output_base_stats'),
    )
    cbON = substitution.get_cb(
        missing_id_combinations,
        ON_tensor_reducer,
        g,
        'OCN',
        selected_base_stats=g.get('output_base_stats'),
    )
    cb_missing = table.merge_tables(cbOS, cbON)
    cb_missing = substitution.add_dif_stats(
        cb_missing,
        g['float_tol'],
        prefix='OC',
        output_stats=g.get('output_stats'),
    )
    cb_missing, g = omega.calc_omega(cb_missing, OS_tensor_reducer, ON_tensor_reducer, g)
    if g['calibrate_longtail'] and (g['exhaustive_until'] >= g['current_arity']):
        cb_missing = omega.calibrate_dsc(cb_missing, output_stats=g.get('output_stats'))
    if g['branch_dist']:
        cb_missing = tree.get_node_distance(
            tree=g['tree'],
            cb=cb_missing,
            ncpu=g['threads'],
            float_type=g['float_type'],
        )
    cb_missing = substitution.get_substitutions_per_branch(cb_missing, g['branch_table'], g)
    cb_missing = table.get_linear_regression(cb_missing)
    cb_missing = output_stat.drop_unrequested_stat_columns(cb_missing, g.get('output_stats'))
    cb_missing, g = get_foreground_branch_num(cb_missing, g)
    cb_missing = table.sort_cb(cb_missing)
    return cb_missing, g

def _get_permutation_cb_rows(rid_combinations, cb, cb_cache, g, OS_tensor_reducer=None, ON_tensor_reducer=None):
    bid_columns = ['branch_id_' + str(k + 1) for k in numpy.arange(rid_combinations.shape[1])]
    rid_combinations = pandas.DataFrame(rid_combinations, columns=bid_columns)
    rid_combinations = table.sort_branch_ids(rid_combinations)
    if cb_cache.shape[0] == 0:
        cb_pool = cb
    else:
        cb_pool = pandas.concat([cb, cb_cache], ignore_index=True)
    rcb = pandas.merge(rid_combinations, cb_pool, how='inner', on=bid_columns)
    dropped_before_recompute = rid_combinations.shape[0] - rcb.shape[0]
    if (dropped_before_recompute > 0) and (OS_tensor_reducer is not None) and (ON_tensor_reducer is not None):
        cb_id_rows = cb_pool.loc[:, bid_columns].drop_duplicates().reset_index(drop=True)
        missing = pandas.merge(
            rid_combinations,
            cb_id_rows,
            how='left',
            on=bid_columns,
            indicator=True,
        )
        missing = missing.loc[missing.loc[:, '_merge'] == 'left_only', bid_columns].drop_duplicates().reset_index(drop=True)
        if missing.shape[0] != 0:
            missing_id_combinations = missing.loc[:, bid_columns].to_numpy(copy=True, dtype=numpy.int64)
            cb_missing, g = _recompute_missing_permutation_rows(
                g=g,
                missing_id_combinations=missing_id_combinations,
                OS_tensor_reducer=OS_tensor_reducer,
                ON_tensor_reducer=ON_tensor_reducer,
            )
            if cb_missing.shape[0] != 0:
                cb_cache = pandas.concat([cb_cache, cb_missing], ignore_index=True)
                cb_pool = pandas.concat([cb, cb_cache], ignore_index=True)
                rcb = pandas.merge(rid_combinations, cb_pool, how='inner', on=bid_columns)
    dropped_after_recompute = rid_combinations.shape[0] - rcb.shape[0]
    return rcb, cb_cache, g, rid_combinations.shape[0], dropped_before_recompute, dropped_after_recompute

def _clade_permutation_mode_prefix(trait_name):
    return 'randomization_' + str(trait_name) + '_'


def _initialize_current_arity_cb_stats(g):
    g = param.initialize_df_cb_stats(g)
    is_arity = (g['df_cb_stats'].loc[:, 'arity'] == g['current_arity'])
    g['df_cb_stats'] = g['df_cb_stats'].loc[is_arity, :].reset_index(drop=True)
    return g


def _build_clade_permutation_mode(trait_name, iteration, randomized_bids, sample_original_foreground):
    mode = _clade_permutation_mode_prefix(trait_name) + 'iter' + str(iteration)
    if sample_original_foreground:
        mode += '_sampleorig'
    mode += '_bid' + ','.join(numpy.asarray(randomized_bids).astype(str))
    return mode


def _report_dropped_permutation_rows(dropped_after_recompute, requested_rows, kept_rows):
    if dropped_after_recompute == 0:
        return
    txt = '{:,} ({:,} - {:,}) permuted foreground branch combinations were dropped because they were not included in the cb table.'
    print(txt.format(dropped_after_recompute, requested_rows, kept_rows))


def _set_randomized_trait_flags(rcb, trait_name):
    rcb['is_fg_' + trait_name] = 'Y'
    rcb['is_mg_' + trait_name] = 'N'
    rcb['is_mf_' + trait_name] = 'N'
    return rcb


def _is_valid_clade_permutation_stat_row(g, trait_name, rid_combinations):
    omega_col = 'median_omegaCany2spe_fg_' + trait_name
    if omega_col not in g['df_cb_stats'].columns:
        txt = 'omegaCany2spe could not be obtained for trait "{}"; skipping this clade permutation trial.\n'
        sys.stderr.write(txt.format(trait_name))
        return False
    if pandas.isna(g['df_cb_stats'].loc[:, omega_col].values[0]):
        print('OmegaCany2spe could not be obtained for permuted foregrounds:')
        print(rid_combinations)
        print('')
        return False
    return True


def _resolve_observed_clade_permutation_omega(g, trait_name, obs_col):
    if obs_col not in g['df_cb_stats_observed'].columns:
        txt = 'Observed stats were unavailable for trait "{}"; skipping clade-permutation summary.\n'
        sys.stderr.write(txt.format(trait_name))
        return None, None
    is_arity_obs = (g['df_cb_stats_observed'].loc[:, 'arity'] == g['current_arity'])
    obs_values = g['df_cb_stats_observed'].loc[is_arity_obs, obs_col].dropna().values
    if obs_values.shape[0] == 0:
        txt = 'No observed omegaCany2spe value was found for trait "{}"; skipping clade-permutation summary.\n'
        sys.stderr.write(txt.format(trait_name))
        return None, None
    return obs_values[0], is_arity_obs


def _resolve_permutation_clade_permutation_omega(g, trait_name, obs_col):
    if 'mode' not in g['df_cb_stats_main'].columns:
        txt = 'Permutation rows were unavailable for trait "{}"; skipping clade-permutation summary.\n'
        sys.stderr.write(txt.format(trait_name))
        return None, None, None
    if obs_col not in g['df_cb_stats_main'].columns:
        txt = 'Permutation omegaCany2spe values were unavailable for trait "{}"; skipping clade-permutation summary.\n'
        sys.stderr.write(txt.format(trait_name))
        return None, None, None
    mode_prefix = _clade_permutation_mode_prefix(trait_name)
    is_arity_perm = (g['df_cb_stats_main'].loc[:, 'arity'] == g['current_arity'])
    is_stat_permutation = g['df_cb_stats_main'].loc[:, 'mode'].astype(str).str.startswith(mode_prefix)
    permutation_values = g['df_cb_stats_main'].loc[is_arity_perm & is_stat_permutation, obs_col].dropna().values
    if permutation_values.shape[0] == 0:
        txt = 'No valid clade permutations were available for trait "{}"; p value was not calculated.\n'
        sys.stderr.write(txt.format(trait_name))
        return None, None, None
    return permutation_values, is_arity_perm, is_stat_permutation


def _report_observed_clade_permutation_ocn(g, trait_name, obs_ocn_col, is_arity_obs):
    if obs_ocn_col not in g['df_cb_stats_observed'].columns:
        return
    obs_ocn_values = g['df_cb_stats_observed'].loc[is_arity_obs, obs_ocn_col].dropna().values
    if obs_ocn_values.shape[0] > 0:
        print('Trait {}: Observed total OCNany2spe in foreground lineages = {:,.3}'.format(trait_name, obs_ocn_values[0]))


def _report_permutation_clade_permutation_ocn(g, trait_name, obs_ocn_col, is_arity_perm, is_stat_permutation):
    if obs_ocn_col not in g['df_cb_stats_main'].columns:
        return
    permutation_ocns = g['df_cb_stats_main'].loc[is_arity_perm & is_stat_permutation, obs_ocn_col].dropna().values
    if permutation_ocns.shape[0] == 0:
        return
    txt = 'Trait {}: Total OCNany2spe in permutation lineages = {:,.3}; {:,.3} ± {:,.3} (median; mean ± SD excluding inf)'
    print(txt.format(trait_name, numpy.median(permutation_ocns), permutation_ocns.mean(), permutation_ocns.std()))


def _get_finite_mean_std(values):
    finite_values = values[numpy.isfinite(values)]
    if finite_values.shape[0] == 0:
        return numpy.nan, numpy.nan
    return finite_values.mean(), finite_values.std()


def _report_clade_permutation_stats(g, trait_name):
    obs_col = 'median_omegaCany2spe_fg_' + trait_name
    obs_ocn_col = 'total_OCNany2spe_fg_' + trait_name
    obs_value, is_arity_obs = _resolve_observed_clade_permutation_omega(
        g=g,
        trait_name=trait_name,
        obs_col=obs_col,
    )
    if obs_value is None:
        return
    permutation_values, is_arity_perm, is_stat_permutation = _resolve_permutation_clade_permutation_omega(
        g=g,
        trait_name=trait_name,
        obs_col=obs_col,
    )
    if permutation_values is None:
        return
    num_positive = (obs_value <= permutation_values).sum()
    num_all = permutation_values.shape[0]
    pvalue = num_positive / num_all
    _report_observed_clade_permutation_ocn(
        g=g,
        trait_name=trait_name,
        obs_ocn_col=obs_ocn_col,
        is_arity_obs=is_arity_obs,
    )
    _report_permutation_clade_permutation_ocn(
        g=g,
        trait_name=trait_name,
        obs_ocn_col=obs_ocn_col,
        is_arity_perm=is_arity_perm,
        is_stat_permutation=is_stat_permutation,
    )
    print('Trait {}: Observed median omegaCany2spe in foreground lineages = {:,.3}'.format(trait_name, obs_value))
    finite_mean, finite_std = _get_finite_mean_std(permutation_values)
    txt = 'Trait {}: Median omegaCany2spe in permutation lineages = {:,.3}; {:,.3} ± {:,.3} (median; mean ± SD excluding inf)'
    print(txt.format(trait_name, numpy.median(permutation_values), finite_mean, finite_std))
    txt = 'Trait {}: P value of foreground convergence (omegaCany2spe) by clade permutations = {} (observation <= permutation = {:,}/{:,})'
    print(txt.format(trait_name, pvalue, num_positive, num_all))


def _append_clade_permutation_failure_row(g, trait_name, trial_no, reason):
    g = _initialize_current_arity_cb_stats(g)
    mode_prefix = _clade_permutation_mode_prefix(trait_name)
    g['df_cb_stats'].loc[:, 'mode'] = mode_prefix + 'iter0_failed_trial' + str(trial_no)
    status_col = 'clade_permutation_status_' + trait_name
    g['df_cb_stats'].loc[:, status_col] = str(reason)
    g['df_cb_stats_main'] = pandas.concat([g['df_cb_stats_main'], g['df_cb_stats']], ignore_index=True)
    return g


def _attempt_randomized_combinations(g, trait_name, trial_no, sample_original_foreground):
    try:
        g, rid_combinations = set_random_foreground_branch(
            g,
            trait_name,
            sample_original_foreground=sample_original_foreground,
        )
        return g, rid_combinations, sample_original_foreground, False, False
    except Exception as exc:
        if not sample_original_foreground:
            txt = 'Clade permutation retry for trait "{}": allowing sampling from original foreground clades after trial {:,} failure ({})\n'
            sys.stderr.write(txt.format(trait_name, trial_no + 1, str(exc)))
            return g, None, True, True, False
        g = _append_clade_permutation_failure_row(g, trait_name, trial_no + 1, str(exc))
        txt = 'Clade permutation failed for trait "{}" at trial {:,}: {}\n'
        sys.stderr.write(txt.format(trait_name, trial_no + 1, str(exc)))
        return g, None, sample_original_foreground, False, True


def _resolve_clade_iteration_transition(g, trait_name, iteration, trial_no, max_trials):
    if iteration == g['fg_clade_permutation']:
        txt = 'Clade permutation successfully found {:,} new branch combinations for trait "{}" after {:,} trials.'
        print(txt.format(g['fg_clade_permutation'], trait_name, trial_no + 1))
        return False, True
    if trial_no == (max_trials - 1):
        txt = 'Clade permutation could not find enough number of new branch combinations for trait "{}" even after {:,} trials.\n'
        sys.stderr.write(txt.format(trait_name, max_trials))
    return True, False


def _resolve_clade_permutation_rows_and_mode(
    cb,
    cb_cache,
    g,
    trait_name,
    iteration,
    sample_original_foreground,
    rid_combinations,
    OS_tensor_reducer=None,
    ON_tensor_reducer=None,
):
    random_mode = _build_clade_permutation_mode(
        trait_name=trait_name,
        iteration=iteration,
        randomized_bids=g['r_fg_ids'][trait_name],
        sample_original_foreground=sample_original_foreground,
    )
    rcb, cb_cache, g, requested_rows, _dropped_before_recompute, dropped_after_recompute = _get_permutation_cb_rows(
        rid_combinations=rid_combinations,
        cb=cb,
        cb_cache=cb_cache,
        g=g,
        OS_tensor_reducer=OS_tensor_reducer,
        ON_tensor_reducer=ON_tensor_reducer,
    )
    _report_dropped_permutation_rows(
        dropped_after_recompute=dropped_after_recompute,
        requested_rows=requested_rows,
        kept_rows=rcb.shape[0],
    )
    return random_mode, rcb, cb_cache, g


def _finalize_clade_permutation_iteration(
    cb,
    cb_cache,
    g,
    trait_name,
    iteration,
    trial_no,
    max_trials,
    sample_original_foreground,
    rid_combinations,
    start,
    OS_tensor_reducer=None,
    ON_tensor_reducer=None,
):
    random_mode, rcb, cb_cache, g = _resolve_clade_permutation_rows_and_mode(
        cb=cb,
        cb_cache=cb_cache,
        g=g,
        trait_name=trait_name,
        iteration=iteration,
        sample_original_foreground=sample_original_foreground,
        rid_combinations=rid_combinations,
        OS_tensor_reducer=OS_tensor_reducer,
        ON_tensor_reducer=ON_tensor_reducer,
    )
    rcb = _set_randomized_trait_flags(rcb=rcb, trait_name=trait_name)
    g = add_median_cb_stats(g, rcb, g['current_arity'], start, verbose=False)
    g['df_cb_stats'].loc[:, 'mode'] = random_mode
    if not _is_valid_clade_permutation_stat_row(g=g, trait_name=trait_name, rid_combinations=rid_combinations):
        return g, cb_cache, False, False
    g['df_cb_stats_main'] = pandas.concat([g['df_cb_stats_main'], g['df_cb_stats']], ignore_index=True)
    print('')
    advance_iter, break_trait = _resolve_clade_iteration_transition(
        g=g,
        trait_name=trait_name,
        iteration=iteration,
        trial_no=trial_no,
        max_trials=max_trials,
    )
    return g, cb_cache, advance_iter, break_trait


def _run_clade_permutation_iteration(
    cb,
    cb_cache,
    g,
    trait_name,
    iteration,
    trial_no,
    max_trials,
    sample_original_foreground,
    OS_tensor_reducer=None,
    ON_tensor_reducer=None,
):
    start = time.time()
    txt = 'Starting foreground clade permutation round {:,} (of {:,}) for trait "{}"'
    print(txt.format(iteration, g['fg_clade_permutation'], trait_name), flush=True)
    g = _initialize_current_arity_cb_stats(g)
    g, rid_combinations, sample_original_foreground, should_continue, break_trait = _attempt_randomized_combinations(
        g=g,
        trait_name=trait_name,
        trial_no=trial_no,
        sample_original_foreground=sample_original_foreground,
    )
    if should_continue:
        return g, cb_cache, sample_original_foreground, False, False
    if break_trait:
        return g, cb_cache, sample_original_foreground, False, True

    g, cb_cache, advance_iter, break_trait = _finalize_clade_permutation_iteration(
        cb=cb,
        cb_cache=cb_cache,
        g=g,
        trait_name=trait_name,
        iteration=iteration,
        trial_no=trial_no,
        max_trials=max_trials,
        sample_original_foreground=sample_original_foreground,
        rid_combinations=rid_combinations,
        start=start,
        OS_tensor_reducer=OS_tensor_reducer,
        ON_tensor_reducer=ON_tensor_reducer,
    )
    return g, cb_cache, sample_original_foreground, advance_iter, break_trait


def _run_clade_permutation_for_trait(
    cb,
    cb_cache,
    g,
    trait_name,
    max_trials,
    OS_tensor_reducer=None,
    ON_tensor_reducer=None,
):
    iteration = 1
    sample_original_foreground = False
    for trial_no in numpy.arange(max_trials):
        g, cb_cache, sample_original_foreground, advance_iter, break_trait = _run_clade_permutation_iteration(
            cb=cb,
            cb_cache=cb_cache,
            g=g,
            trait_name=trait_name,
            iteration=iteration,
            trial_no=trial_no,
            max_trials=max_trials,
            sample_original_foreground=sample_original_foreground,
            OS_tensor_reducer=OS_tensor_reducer,
            ON_tensor_reducer=ON_tensor_reducer,
        )
        if advance_iter:
            iteration += 1
        if break_trait:
            break
    _report_clade_permutation_stats(g, trait_name)
    return g, cb_cache


def clade_permutation(cb, g, OS_tensor_reducer=None, ON_tensor_reducer=None):
    print('Starting foreground clade permutation.')
    trait_names = _get_trait_names(g)
    if len(trait_names) == 0:
        sys.stderr.write('No foreground traits were available for clade permutation.\n')
        return g
    g['df_cb_stats_observed'] = g['df_cb_stats'].copy()
    max_trials = g['fg_clade_permutation'] * 10
    cb_cache = pandas.DataFrame()
    for trait_name in trait_names:
        g, cb_cache = _run_clade_permutation_for_trait(
            cb=cb,
            cb_cache=cb_cache,
            g=g,
            trait_name=trait_name,
            max_trials=max_trials,
            OS_tensor_reducer=OS_tensor_reducer,
            ON_tensor_reducer=ON_tensor_reducer,
        )
    g['df_cb_stats'] = g['df_cb_stats_observed'].copy()
    return g
