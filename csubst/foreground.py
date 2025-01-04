import numpy
import pandas

import copy
import itertools
import re
import sys
import time
import warnings

from csubst import combination
from csubst import table
from csubst import param

def combinations_count(n, r):
    # https://github.com/nkmk/python-snippets/blob/05a53ae96736577906a8805b38bce6cc210fe11f/notebook/combinations_count.py#L1-L14
    from operator import mul
    from functools import reduce
    r = min(r, n - r)
    numer = reduce(mul, range(n, n - r, -1), 1)
    denom = reduce(mul, range(1, r + 1), 1)
    return numer // denom

def get_df_clade_size(g, trait_name):
    num_branch = max([ n.numerical_label for n in g['tree'].traverse() ])
    branch_ids = numpy.arange(num_branch)
    cols = ['branch_id','size','is_fg_stem_'+trait_name]
    df_clade_size = pandas.DataFrame(index=branch_ids, columns=cols)
    df_clade_size.loc[:,'branch_id'] = branch_ids
    df_clade_size.loc[:,'is_fg_stem_'+trait_name] = False
    for node in g['tree'].traverse():
        if node.is_root():
            continue
        bid = node.numerical_label
        df_clade_size.at[bid,'size'] = len(node.get_leaf_names())
        is_fg = getattr(node, 'is_fg_'+trait_name)
        is_parent_fg = getattr(node.up, 'is_fg_'+trait_name)
        if (is_fg)&(~is_parent_fg):
            df_clade_size.at[bid,'is_fg_stem_'+trait_name] = True
    return df_clade_size

def foreground_clade_randomization(df_clade_size, g, sample_original_foreground=False):
    size_array = df_clade_size.loc[:,'size'].values.astype(numpy.int64)
    size_min = size_array.min()
    size_max = size_array.max()
    sizes = numpy.unique(size_array)[::-1] # To start counting from rarer (larger) clades
    bins = numpy.array([size_max+1,], dtype=numpy.int64)
    count = 0
    counts = []
    for size in sizes:
        if sample_original_foreground:
            is_size = (size_array==size)
        else:
            is_size = ((size_array==size)&(~df_clade_size.loc[:,'is_fg_stem_'+trait_name]))
        count += is_size.sum()
        if (count >= g['min_clade_bin_count']):
            bins = numpy.append(bins, size)
            counts.append(count)
            count = 0
    if len(bins)<2:
        bins = numpy.array([size_min, size_max], dtype=numpy.int64)
    txt = 'Number of clade permutation bins = {:,} (bin limits = {}, counts = {})'
    print(txt.format(bins.shape[0]-1, ', '.join([ str(s) for s in bins ]), ', '.join([ str(s) for s in counts ])))
    bins = bins[::-1]
    df_clade_size.loc[:,'bin'] = numpy.digitize(size_array, bins, right=False)
    is_fg = (df_clade_size.loc[:,'is_fg_stem_'+trait_name]==True)
    fg_bins = df_clade_size.loc[is_fg,'bin']
    df_clade_size.loc[:,'is_fg_stem_randomized'] = df_clade_size.loc[:,'is_fg_stem_'+trait_name]
    df_clade_size.loc[:,'is_blocked'] = False
    for bin in fg_bins.unique():
        is_bin = (df_clade_size.loc[:,'bin']==bin)
        is_blocked = df_clade_size.loc[:,'is_blocked'].values
        #min_bin_size = df_clade_size.loc[is_bin,'size'].min()
        #max_bin_size = df_clade_size.loc[is_bin,'size'].max()
        #num_fg_bin = (fg_bins==bin).sum()
        #num_bin = is_bin.sum()
        #num_unblocked_bin = (is_bin&~is_blocked).sum()
        #cc_w = combinations_count(n=num_unblocked_bin, r=num_fg_bin)
        #cc_wo = combinations_count(n=num_bin, r=num_fg_bin)
        #txt = 'bin {}: foreground/all clades = {:,}/{:,}, ' \
        #      'min/max clade sizes = {:,}/{:,}, ' \
        #      'randomization complexity with/without considering branch dependency = {:,}/{:,}'
        #print(txt.format(bin, num_fg_bin, num_bin, min_bin_size, max_bin_size, cc_w, cc_wo), flush=True)
        before_randomization = df_clade_size.loc[is_bin&~is_blocked,'is_fg_stem_randomized'].values
        if sample_original_foreground:
            after_randomization = numpy.random.permutation(before_randomization)
        else:
            ind_fg = numpy.where(before_randomization==True)[0]
            ind_nonfg = numpy.where(before_randomization==False)[0]
            ind_rfg = numpy.random.choice(ind_nonfg, ind_fg.shape[0])
            after_randomization = numpy.zeros_like(before_randomization, dtype=bool)
            after_randomization[ind_rfg] = True
        df_clade_size.loc[is_bin&~is_blocked,'is_fg_stem_randomized'] = after_randomization
        is_new_fg = is_bin&~is_blocked&(df_clade_size.loc[:,'is_fg_stem_randomized']==True)
        new_fg_bids = df_clade_size.loc[is_new_fg,'branch_id'].values
        for new_fg_bid in new_fg_bids:
            for node in g['tree'].traverse():
                if (node.numerical_label==new_fg_bid):
                    des_bids = [ d.numerical_label for d in node.traverse() ]
                    df_clade_size.loc[des_bids,'is_blocked'] = True
                    break
    return df_clade_size

def get_new_foreground_ids(df_clade_size, g):
    is_new_fg = (df_clade_size.loc[:,'is_fg_stem_randomized']==True)
    fg_stem_bids = df_clade_size.loc[is_new_fg,'branch_id'].values
    new_fg_ids = list()
    if (g['fg_stem_only']):
        new_fg_ids = fg_stem_bids
    else:
        for fg_stem_bid in fg_stem_bids:
            for node in g['tree'].traverse():
                if (node.numerical_label==fg_stem_bid):
                    new_lineage_fg_ids = [ n.numerical_label for n in node.traverse() ]
                    new_fg_ids += new_lineage_fg_ids
                    break
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
    all_leaf_names = g['tree'].get_leaf_names()
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
            node_leaf_name_set = set(node.get_leaf_names())
            if len(node_leaf_name_set.difference(fg_leaf_name_set)) == 0:
                node.add_features(**{'is_lineage_fg_'+trait_name+'_'+str(i+1): True})
            else:
                node.add_features(**{'is_lineage_fg_'+trait_name+'_'+str(i+1): False})
    return tree

def get_target_ids(lineages, trait_name, g):
    target_ids = numpy.zeros(shape=(0,), dtype=numpy.int64)
    for i in numpy.arange(len(lineages)):
        fg_leaf_name_set = set(g['fg_leaf_names'][trait_name][i])
        lineage_fg_ids = list()
        for node in g['tree'].traverse():
            node_leaf_name_set = set(node.get_leaf_names())
            if len(node_leaf_name_set.difference(fg_leaf_name_set)) == 0:
                if node.is_root() | (not g['fg_stem_only']):
                    lineage_fg_ids.append(node.numerical_label)
                else:
                    is_lineage_fg = getattr(node, 'is_lineage_fg_' + trait_name + '_' + str(i + 1))
                    is_parent_lineage_fg = getattr(node.up, 'is_lineage_fg_' + trait_name + '_' + str(i + 1))
                    if (is_lineage_fg == True) & (is_parent_lineage_fg == False):
                        lineage_fg_ids.append(node.numerical_label)
        dif = 1
        while dif:
            num_id = len(lineage_fg_ids)
            for node in g['tree'].traverse():
                child_ids = [child.numerical_label for child in node.get_children()]
                if all([id in lineage_fg_ids for id in child_ids]) & (len(child_ids) != 0):
                    if node.numerical_label not in lineage_fg_ids:
                        lineage_fg_ids.append(node.numerical_label)
            dif = len(lineage_fg_ids) - num_id
        lineage_fg_ids = numpy.array(lineage_fg_ids, dtype=numpy.int64)
        target_ids = numpy.concatenate([target_ids, lineage_fg_ids])
    target_id = target_ids[target_ids != g['tree'].get_tree_root().numerical_label]
    target_id.sort()
    return target_ids

def annotate_foreground(lineages, trait_name, g):
    for node in g['tree'].traverse(): # Initialize
        node.add_features(**{'is_fg_'+trait_name: False})
        node.add_features(**{'color_'+trait_name: 'black'})
        node.add_features(**{'labelcolor_' + trait_name: 'black'})
    lineage_colors = get_lineage_color_list()
    for i in numpy.arange(len(lineages)):
        fg_leaf_name_set = set(g['fg_leaf_names'][trait_name][i])
        lineage_color = lineage_colors[i % len(lineage_colors)]
        for node in g['tree'].traverse():
            if g['fg_stem_only']:
                if node.numerical_label in g['target_ids'][trait_name]:
                    node.add_features(**{'is_fg_'+trait_name: True})
                    node.add_features(**{'color_'+trait_name: lineage_color})
                    node.add_features(**{'labelcolor_'+trait_name: lineage_color})
                if node.name in fg_leaf_name_set:
                    node.add_features(**{'labelcolor_' + trait_name: lineage_color})
            else:
                is_lineage_fg = getattr(node, 'is_lineage_fg_' + trait_name + '_' + str(i + 1))
                if is_lineage_fg == True:
                    node.add_features(**{'is_fg_' + trait_name: True})
                    node.add_features(**{'color_' + trait_name: lineage_color})
                    node.add_features(**{'labelcolor_' + trait_name: lineage_color})
    return g['tree']

def get_foreground_ids(g, write=True):
    g['fg_leaf_names'] = dict()
    g['target_ids'] = dict()
    g['fg_ids'] = dict()
    for trait_name in g['fg_df'].columns[1:len(g['fg_df'].columns)]:
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
        node.add_features(**{'is_lineage_fg_' + trait_name + '_1': False})
        node.add_features(**{'is_fg_' + trait_name: False})
        node.add_features(**{'color_' + trait_name: 'black'})
        node.add_features(**{'labelcolor_' + trait_name: 'black'})
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

def print_num_possible_permuted_combinations(df_clade_size, sample_original_foreground):
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
    r_df_clade_size = foreground_clade_randomization(df_clade_size, g, sample_original_foreground)
    print_num_possible_permuted_combinations(r_df_clade_size, sample_original_foreground)
    g['r_target_ids'][trait_name] = get_new_foreground_ids(r_df_clade_size, g)
    g['r_fg_ids'][trait_name] = copy.deepcopy(g['r_target_ids'][trait_name])
    new_fg_leaf_names = [ n.get_leaf_names() for n in g['tree'].traverse() if n.numerical_label in g['r_fg_ids'] ]
    new_fg_leaf_names = list(itertools.chain(*new_fg_leaf_names))
    new_fg_leaf_names = list(set(new_fg_leaf_names))
    g['r_fg_leaf_names'][trait_name] = new_fg_leaf_names
    return g

def get_marginal_branch(g):
    g['mg_ids'] = dict()
    for trait_name in g['fg_df'].columns[1:len(g['fg_df'].columns)]:
        g['mg_ids'][trait_name] = list()
        for node in g['tree'].traverse():
            if (node.is_root()):
                continue
            is_fg = getattr(node, 'is_fg_' + trait_name)
            if (is_fg==False):
                continue
            if (g['mg_parent']):
                is_parent_fg = getattr(node.up, 'is_fg_' + trait_name)
                if (is_parent_fg==False):
                    g['mg_ids'][trait_name].append(node.up.numerical_label)
            if (g['mg_sister']):
                sisters = node.get_sisters()
                for sister in sisters:
                    if (g['mg_sister_stem_only']):
                        is_sister_fg = getattr(sister, 'is_fg_' + trait_name)
                        if is_sister_fg==False:
                            g['mg_ids'][trait_name].append(sister.numerical_label)
                    else:
                        for sister_des in sister.traverse():
                            is_sister_des_fg = getattr(sister_des, 'is_fg_' + trait_name)
                            if is_sister_des_fg==False:
                                g['mg_ids'][trait_name].append(sister_des.numerical_label)
        concat_ids = list(set(g['mg_ids'][trait_name])-set(g['target_ids'][trait_name]))
        g['mg_ids'][trait_name] = numpy.array(concat_ids, dtype=numpy.int64)
        g['target_ids'][trait_name] = numpy.concatenate([g['target_ids'][trait_name], g['mg_ids'][trait_name]])
        for node in g['tree'].traverse():
            if node.numerical_label in g['mg_ids'][trait_name]:
                node.add_features(**{'is_marginal_'+trait_name: True})
            else:
                node.add_features(**{'is_marginal_'+trait_name: False})
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
    trait_names = g['fg_df'].columns[1:len(g['fg_df'].columns)]
    for trait_name in trait_names:
        for id_key,newcol in zip(['fg_ids','mg_ids'],['branch_num_fg_'+trait_name,'branch_num_mg_'+trait_name]):
            id_set = set(g[id_key][trait_name])
            cb.loc[:,newcol] = 0
            for bid_col in bid_cols:
                cb.loc[(cb[bid_col].isin(id_set)),newcol] += 1
        cb.loc[:,'is_fg_'+trait_name] = 'N'
        cb.loc[(cb.loc[:,'branch_num_fg_'+trait_name]==arity),'is_fg_'+trait_name] = 'Y'
        for i in numpy.arange(g['fg_dependent_id_combinations'][trait_name].shape[0]):
            bids = g['fg_dependent_id_combinations'][trait_name][i, :]
            conditions = [(cb[f'branch_id_{j + 1}'] == bid) for j, bid in enumerate(bids)]
            is_dep = numpy.logical_and.reduce(conditions)
            cb.loc[is_dep,'is_fg_'+trait_name] = 'N'
        cb.loc[:,'is_mg_'+trait_name] = 'N'
        cb.loc[(cb['branch_num_mg_'+trait_name]==arity),'is_mg_'+trait_name] = 'Y'
        cb.loc[:,'is_mf_'+trait_name] = 'N'
        is_mg = (cb['branch_num_fg_'+trait_name]>0) & (cb['branch_num_mg_'+trait_name]>0)
        is_mg = (is_mg) & ((cb['branch_num_fg_'+trait_name] + cb['branch_num_mg_'+trait_name])==arity)
        cb.loc[is_mg,'is_mf_'+trait_name] = 'Y'
        df_clade_size = get_df_clade_size(g, trait_name)
        fg_stem_bids = df_clade_size.loc[df_clade_size.loc[:,'is_fg_stem_'+trait_name],'branch_id'].values
        cb.loc[:,'branch_num_fg_stem_'+trait_name] = 0
        for bid_col in bid_cols:
            cb.loc[(cb[bid_col].isin(fg_stem_bids)),'branch_num_fg_stem_'+trait_name] += 1
        is_fg = (cb['is_fg_'+trait_name]=='Y')
        is_enough_stat = table.get_cutoff_stat_bool_array(cb=cb, cutoff_stat_str=g['cutoff_stat'])
        num_enough = is_enough_stat.sum()
        num_fg = is_fg.sum()
        num_fg_enough = (is_enough_stat&is_fg).sum()
        num_all = cb.shape[0]
        if (num_enough==0)|(num_fg==0):
            percent_fg_enough = 0
            enrichment_factor = 0
        else:
            percent_fg_enough = num_fg_enough / num_enough * 100
            enrichment_factor = (num_fg_enough/num_enough) / (num_fg/num_all)
        txt = 'Arity (K) = {}: Foreground branch combinations with cutoff conditions {} for {} = {:.0f}% ({:,}/{:,}, ' \
              'total examined = {:,}, enrichment factor = {:.1f})'
        txt = txt.format(arity, g['cutoff_stat'], trait_name, percent_fg_enough, num_fg_enough, num_enough,
                         num_all, enrichment_factor)
        print(txt, flush=True)
        is_arity = (g['df_cb_stats'].loc[:,'arity']==arity)
        if not 'fg_enrichment_factor_'+trait_name in g['df_cb_stats'].columns:
            g['df_cb_stats']['fg_enrichment_factor_'+trait_name] = numpy.nan
        g['df_cb_stats'].loc[is_arity,'fg_enrichment_factor_'+trait_name] = enrichment_factor
    txt = 'Time elapsed for obtaining foreground branch numbers in the cb table: {:,} sec'
    print(txt.format(int(time.time() - start_time)))
    return cb, g

def annotate_b_foreground(b, g):
    trait_names = g['fg_df'].columns[1:len(g['fg_df'].columns)]
    for trait_name in trait_names:
        if (g['foreground'] is None):
            b.loc[:, 'is_fg_' + trait_name] = 'Y'
            b.loc[:, 'is_mg_' + trait_name] = 'N'
        else:
            b.loc[:,'is_fg_'+trait_name] = 'N'
            b.loc[:, 'is_mg_' + trait_name] = 'N'
            if (len(g['fg_ids'][trait_name]) > 0):
                b.loc[g['fg_ids'][trait_name],'is_fg_'+trait_name] = 'Y'
            if len(g['mg_ids'][trait_name]) > 0:
                b.loc[g['mg_ids'][trait_name],'is_mg_'+trait_name] = 'Y'
    return b

def get_num_foreground_lineages(tree, trait_name):
    num_fl = 0
    for node in tree.traverse():
        for k in node.__dict__.keys():
            if k.startswith('is_lineage_fg_'+trait_name):
                num_fl = max(num_fl, int(k.replace('is_lineage_fg_'+trait_name+'_', '')))
        break
    return num_fl

def set_random_foreground_branch(g, trait_name, num_trial=100):
    for i in numpy.arange(num_trial):
        g = get_foreground_branch(g)
        g = randomize_foreground_branch(g, trait_name)
        if g['r_target_ids'][trait_name].shape[0]<2:
            continue
        g,rid_combinations = combination.get_node_combinations(g=g, target_id_dict=g['r_target_ids'],
                                                               arity=g['current_arity'],
                                                               check_attr="name", verbose=False)
        if rid_combinations.shape[0]==0:
            continue
        print('Foreground branch permutation finished after {:,} trials.'.format(i+1), flush=True)
        return g,rid_combinations
    txt = 'Foreground branch permutation failed {:,} times. There may not be enough numbers of "similar" clades.\n'
    raise Exception(txt.format(num_trial))

def add_median_cb_stats(g, cb, current_arity, start, verbose=True):
    is_arity = (g['df_cb_stats'].loc[:,'arity'] == current_arity)
    is_targets = dict()
    target_col_prefixes = ['is_fg','is_mg','is_mf','is_all']
    trait_names = g['fg_df'].columns[1:len(g['fg_df'].columns)]
    for trait_name in trait_names:
        for target_col_prefix in target_col_prefixes:
            target_col = target_col_prefix+'_'+trait_name
            if target_col in cb.columns:
                suffix = target_col.replace('is_','_')
                is_targets[suffix] = (cb[target_col]=='Y')
            elif target_col_prefix == 'is_all':
                is_targets['_all'] = numpy.ones(shape=(cb.shape[0],), dtype=bool)
    stats = dict()
    omega_cols = cb.columns[cb.columns.str.startswith('omegaC')].tolist()
    is_ON = cb.columns.str.startswith('OCNany') | cb.columns.str.startswith('OCNdif') | cb.columns.str.startswith('OCNspe')
    is_OS = cb.columns.str.startswith('OCSany') | cb.columns.str.startswith('OCSdif') | cb.columns.str.startswith('OCSspe')
    is_EN = cb.columns.str.startswith('ECNany') | cb.columns.str.startswith('ECNdif') | cb.columns.str.startswith('ECNspe')
    is_ES = cb.columns.str.startswith('ECSany') | cb.columns.str.startswith('ECSdif') | cb.columns.str.startswith('ECSspe')
    ON_cols = cb.columns[is_ON].tolist()
    OS_cols = cb.columns[is_OS].tolist()
    EN_cols = cb.columns[is_EN].tolist()
    ES_cols = cb.columns[is_ES].tolist()
    stats['median'] = ['dist_bl','dist_node_num',] + omega_cols
    stats['total'] = ON_cols + OS_cols + EN_cols + ES_cols
    is_qualified = table.get_cutoff_stat_bool_array(cb=cb, cutoff_stat_str=g['cutoff_stat'])
    new_columns = {}
    for suffix, is_target in is_targets.items():
        new_columns['num' + suffix] = is_target.sum()
        new_columns['num_qualified' + suffix] = (is_target & is_qualified).sum()
    new_columns_df = pandas.DataFrame(new_columns, index=g['df_cb_stats'].index)
    new_columns_df = new_columns_df.loc[is_arity]
    g['df_cb_stats'] = pandas.concat([g['df_cb_stats'], new_columns_df], axis=1)
    for stat in stats.keys():
        for suffix,is_target in is_targets.items():
            for ms in stats[stat]:
                col = stat+'_'+ms+suffix
                if not col in g['df_cb_stats'].columns:
                    newcol = pandas.DataFrame({col:numpy.zeros(shape=(g['df_cb_stats'].shape[0]))})
                    g['df_cb_stats'] = pandas.concat([g['df_cb_stats'], newcol], ignore_index=False, axis=1)
                if not ms in cb.columns:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if stat=='median':
                        g['df_cb_stats'].loc[is_arity,col] = cb.loc[is_target,ms].median()
                    elif stat=='total':
                        g['df_cb_stats'].loc[is_arity,col] = cb.loc[is_target,ms].sum()
    if verbose:
        for SN,anc,des in itertools.product(['S','N'], ['any','dif','spe'], ['any','dif','spe']):
            key = SN+anc+'2'+des
            totalON = g['df_cb_stats'].loc[is_arity, 'total_OC'+key+'_all'].values[0]
            totalEN = g['df_cb_stats'].loc[is_arity, 'total_EC'+key+'_all'].values[0]
            if totalON==0:
                percent_value = numpy.nan
            else:
                percent_value = totalEN / totalON * 100
            txt = 'Total OC{}/EC{} = {:,.1f}/{:,.1f} (expectation = {:,.1f}% of observation)'
            print(txt.format(key, key, totalON, totalEN, percent_value))
    elapsed_time = int(time.time() - start)
    g['df_cb_stats'].loc[is_arity, 'elapsed_sec'] = elapsed_time
    if verbose:
        print(("Elapsed time for arity = {}: {:,.1f} sec\n".format(current_arity, elapsed_time)), flush=True)
    return g

def clade_permutation(cb, g):
    print('Starting foreground clade permutation.')
    if g['fg_df'].shape[1] > 2:
        sys.stderr.write('Only the first trait column will be used for clade permutation.\n')
    g['df_cb_stats_observed'] = g['df_cb_stats'].copy()
    trait_name = g['fg_df'].columns[1]
    i = 1
    for trial_no in numpy.arange(g['fg_clade_permutation']*10):
        start = time.time()
        txt = 'Starting foreground clade permutation round {:,} (of {:,})'
        print(txt.format(i, g['fg_clade_permutation']), flush=True)
        g = param.initialize_df_cb_stats(g)
        g['df_cb_stats'] = g['df_cb_stats'].loc[(g['df_cb_stats'].loc[:,'arity']==g['current_arity']),:].reset_index(drop=True)
        g,rid_combinations = set_random_foreground_branch(g, trait_name)
        random_mode = 'randomization_iter'+str(i)+'_bid'+','.join(g['r_fg_ids'][trait_name].astype(str))
        bid_columns = [ 'branch_id_'+str(k+1) for k in numpy.arange(rid_combinations.shape[1]) ]
        rid_combinations = pandas.DataFrame(rid_combinations)
        rid_combinations.columns = bid_columns
        rcb = pandas.merge(rid_combinations, cb, how='inner', on=bid_columns)
        if (rid_combinations.shape[0] != rcb.shape[0]):
            txt = '{:,} ({:,} - {:,}) permuted foreground branch combinations were dropped because they were not included in the cb table.'
            print(txt.format(rid_combinations.shape[0]-rcb.shape[0], rid_combinations.shape[0], rcb.shape[0]))
        rcb['is_fg_'+trait_name] = 'Y'
        rcb['is_mg_'+trait_name] = 'N'
        rcb['is_mf_'+trait_name] = 'N'
        g = add_median_cb_stats(g, rcb, 2, start, verbose=False)
        g['df_cb_stats'].loc[:,'mode'] = random_mode
        if numpy.isnan(g['df_cb_stats'].loc[:,'median_omegaCany2spe_fg_'+trait_name].values[0]):
            print('OmegaCany2spe could not be obtained for permuted foregrounds:')
            print(rid_combinations)
            print('')
            continue
        g['df_cb_stats_main'] = pandas.concat([g['df_cb_stats_main'], g['df_cb_stats']], ignore_index=True)
        print('')
        if (i==g['fg_clade_permutation']):
            txt = 'Clade permutation successfully found {:,} new branch combinations after {:,} trials.'
            print(txt.format(g['fg_clade_permutation'], trial_no+1))
            break
        if (trial_no==(g['fg_clade_permutation']*10)):
            txt = 'Clade permutation could not find enough number of new branch combinations even after {:,} trials.\n'
            sys.stderr.write(txt.format(g['fg_clade_permutation']*10))
        i += 1
    is_arity = (g['df_cb_stats_main'].loc[:,'arity']==g['current_arity'])
    is_stat_fg = ~g['df_cb_stats_main'].loc[:,'mode'].str.startswith('randomization_')
    is_stat_permutation = g['df_cb_stats_main'].loc[:,'mode'].str.startswith('randomization_')
    obs_value = g['df_cb_stats_main'].loc[is_arity & is_stat_fg,'median_omegaCany2spe_fg_'+trait_name].values[0]
    permutation_values = g['df_cb_stats_main'].loc[is_arity & is_stat_permutation, 'median_omegaCany2spe_fg_'+trait_name].values
    num_positive = (obs_value<=permutation_values).sum()
    num_all = permutation_values.shape[0]
    pvalue = num_positive / num_all
    obs_ocn = g['df_cb_stats_main'].loc[is_arity & is_stat_fg,'total_OCNany2spe_fg_'+trait_name].values[0]
    print('Observed total OCNany2spe in foreground lineages = {:,.3}'.format(obs_ocn))
    permutation_ocns = g['df_cb_stats_main'].loc[is_arity & is_stat_permutation, 'total_OCNany2spe_fg_'+trait_name].values
    txt = 'Total OCNany2spe in permutation lineages = {:,.3}; {:,.3} ± {:,.3} (median; mean ± SD excluding inf)'
    print(txt.format(numpy.median(permutation_ocns), permutation_ocns.mean(), permutation_ocns.std()))
    print('Observed median omegaCany2spe in foreground lineages = {:,.3}'.format(obs_value))
    txt = 'Median omegaCany2spe in permutation lineages = {:,.3}; {:,.3} ± {:,.3} (median; mean ± SD excluding inf)'
    print(txt.format(numpy.median(permutation_values), permutation_values[numpy.isfinite(permutation_values)].mean(), permutation_values[numpy.isfinite(permutation_values)].std()))
    txt = 'P value of foreground convergence (omegaCany2spe) by clade permutations = {} (observation <= permutation = {:,}/{:,})'
    print(txt.format(pvalue, num_positive, num_all))
    g['df_cb_stats'] = g['df_cb_stats_observed'].copy()
    return g