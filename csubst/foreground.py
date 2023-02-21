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

def get_df_clade_size(g):
    num_branch = max([ n.numerical_label for n in g['tree'].traverse() ])
    branch_ids = numpy.arange(num_branch)
    cols = ['branch_id','size','is_foreground_stem']
    df_clade_size = pandas.DataFrame(index=branch_ids, columns=cols)
    df_clade_size.loc[:,'branch_id'] = branch_ids
    df_clade_size.loc[:,'is_foreground_stem'] = False
    for node in g['tree'].traverse():
        if node.is_root():
            continue
        bid = node.numerical_label
        df_clade_size.at[bid,'size'] = len(node.get_leaf_names())
        if (node.is_foreground==True)&(node.up.is_foreground==False):
            df_clade_size.at[bid,'is_foreground_stem'] = True
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
            is_size = ((size_array==size)&(~df_clade_size.loc[:,'is_foreground_stem']))
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
    is_fg = (df_clade_size.loc[:,'is_foreground_stem']==True)
    fg_bins = df_clade_size.loc[is_fg,'bin']
    df_clade_size.loc[:,'is_foreground_stem_randomized'] = df_clade_size.loc[:,'is_foreground_stem']
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
        before_randomization = df_clade_size.loc[is_bin&~is_blocked,'is_foreground_stem_randomized'].values
        if sample_original_foreground:
            after_randomization = numpy.random.permutation(before_randomization)
        else:
            ind_fg = numpy.where(before_randomization==True)[0]
            ind_nonfg = numpy.where(before_randomization==False)[0]
            ind_rfg = numpy.random.choice(ind_nonfg, ind_fg.shape[0])
            after_randomization = numpy.zeros_like(before_randomization, dtype=bool)
            after_randomization[ind_rfg] = True
        df_clade_size.loc[is_bin&~is_blocked,'is_foreground_stem_randomized'] = after_randomization
        is_new_fg = is_bin&~is_blocked&(df_clade_size.loc[:,'is_foreground_stem_randomized']==True)
        new_fg_bids = df_clade_size.loc[is_new_fg,'branch_id'].values
        for new_fg_bid in new_fg_bids:
            for node in g['tree'].traverse():
                if (node.numerical_label==new_fg_bid):
                    des_bids = [ d.numerical_label for d in node.traverse() ]
                    df_clade_size.loc[des_bids,'is_blocked'] = True
                    break
    return df_clade_size

def get_new_foreground_ids(df_clade_size, g):
    is_new_fg = (df_clade_size.loc[:,'is_foreground_stem_randomized']==True)
    fg_stem_bids = df_clade_size.loc[is_new_fg,'branch_id'].values
    new_fg_ids = list()
    if (g['fg_stem_only']):
        new_fg_ids = fg_stem_bids
    else:
        for fg_stem_bid in fg_stem_bids:
            for node in g['tree'].traverse():
                if (node.numerical_label==fg_stem_bid):
                    new_lineage_fg_ids = [ n.numerical_label for n in node.traverse() ]
                    new_fg_ids = new_fg_ids + new_lineage_fg_ids
    new_fg_ids = numpy.array(new_fg_ids, dtype=numpy.int64)
    return new_fg_ids

def annotate_foreground_branch(tree, fg_df, fg_stem_only):
    target_id = numpy.zeros(shape=(0,), dtype=numpy.int64)
    fg_leaf_name = []
    leaf_names = [ leaf.name for leaf in tree.get_leaves() ]
    if fg_df.shape[0]==0:
        lineages = []
    else:
        lineages = fg_df.iloc[:,0].unique()
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
    for i in numpy.arange(len(lineages)):
        fg_leaf_name.append([])
        is_lineage = (fg_df.iloc[:,0]==lineages[i])
        lineage_regex_names = fg_df.loc[is_lineage,:].iloc[:,1].unique().tolist()
        iter = itertools.product(leaf_names, lineage_regex_names)
        lineage_leaf_names = [ ln for ln,lr in iter if re.match('^'+lr+'$', ln) ]
        lineage_fg_id = list()
        for lln in lineage_leaf_names:
            match_leaves = [ ln for ln in leaf_names if lln==ln ]
            if len(match_leaves)==1:
                fg_leaf_name[i].append(match_leaves[0])
            else:
                print('The foreground leaf name cannot be identified:', lln, match_leaves)
        fg_leaf_name_set = set(fg_leaf_name[i])
        for node in tree.traverse():
            node.is_lineage_foreground = False # re-initializing
        for node in tree.traverse():
            node_leaf_name_set = set(node.get_leaf_names())
            if len(node_leaf_name_set.difference(fg_leaf_name_set))==0:
                node.is_lineage_foreground = True
                node.is_foreground = True
                node.foreground_lineage_id = i+1
                node.color = lineage_colors[i%len(lineage_colors)]
                node.labelcolor = lineage_colors[i%len(lineage_colors)]
            if node.name in fg_leaf_name_set:
                node.labelcolor = lineage_colors[i%len(lineage_colors)]
        for node in tree.traverse():
            node_leaf_name_set = set(node.get_leaf_names())
            if len(node_leaf_name_set.difference(fg_leaf_name_set))==0:
                if fg_stem_only:
                    if (node.is_lineage_foreground==True)&(node.up.is_lineage_foreground==False):
                        lineage_fg_id.append(node.numerical_label)
                else:
                    lineage_fg_id.append(node.numerical_label)
        dif = 1
        while dif:
            num_id = len(lineage_fg_id)
            for node in tree.traverse():
                child_ids = [ child.numerical_label for child in node.get_children() ]
                if all([ id in lineage_fg_id for id in child_ids ])&(len(child_ids)!=0):
                    if node.numerical_label not in lineage_fg_id:
                        lineage_fg_id.append(node.numerical_label)
            dif = len(lineage_fg_id) - num_id
        tmp = numpy.array(lineage_fg_id, dtype=numpy.int64)
        target_id = numpy.concatenate([target_id, tmp])
    if fg_stem_only:
        for node in tree.traverse():
            if node.numerical_label in target_id:
                node.is_foreground = True
            else:
                node.is_foreground = False
                node.color = 'black'
    with open('csubst_target_branch.txt', 'w') as f:
        for x in target_id:
            f.write(str(x)+'\n')
    return tree,fg_leaf_name,target_id

def initialize_foreground_annotation(tree):
    for node in tree.traverse():
        node.is_lineage_foreground = False
        node.is_foreground = False
        node.foreground_lineage_id = 0
        node.color = 'black'
        node.labelcolor = 'black'
    return tree

def get_foreground_branch(g):
    g['tree'] = initialize_foreground_annotation(tree=g['tree'])
    if g['foreground'] is None:
        g['fg_df'] = pandas.DataFrame()
        g['fg_id'] = list()
        g['fg_leaf_name'] = list()
        g['target_id'] = numpy.zeros(shape=(0,), dtype=numpy.int64)
    else:
        g['fg_df'] = pandas.read_csv(g['foreground'], sep='\t', comment='#', skip_blank_lines=True, header=None)
        if g['fg_df'].shape[1]!=2:
            txt = '--foreground file should have a tab-separated two-column table without header. ' \
                  'First column = lineage id; Second column = Sequence name'
            raise Exception(txt)
        # target_id is numerical_label for leaves in --foreground as well as their ancestors
        g['tree'],g['fg_leaf_name'],g['target_id'] = annotate_foreground_branch(g['tree'], g['fg_df'], g['fg_stem_only'])
        g['fg_id'] = copy.deepcopy(g['target_id']) # marginal_ids may be added to target_id but fg_id won't be changed.
    return g

def print_num_possible_permuted_combinations(df_clade_size, sample_original_foreground):
    import scipy
    num_possible_permutation_combination = 1
    is_fg_stem = df_clade_size.loc[:, 'is_foreground_stem'].values
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

def randomize_foreground_branch(g, sample_original_foreground=False):
    g['fg_id_original'] = copy.deepcopy(g['fg_id'])
    g['fg_leaf_name_original'] = copy.deepcopy(g['fg_leaf_name'])
    df_clade_size = get_df_clade_size(g)
    df_clade_size = foreground_clade_randomization(df_clade_size, g, sample_original_foreground)
    print_num_possible_permuted_combinations(df_clade_size, sample_original_foreground)
    g['target_id'] = get_new_foreground_ids(df_clade_size, g)
    g['fg_id'] = copy.deepcopy(g['target_id'])
    new_fg_leaf_names = [ n.get_leaf_names() for n in g['tree'].traverse() if n.numerical_label in g['fg_id'] ]
    new_fg_leaf_names = list(itertools.chain(*new_fg_leaf_names))
    new_fg_leaf_names = list(set(new_fg_leaf_names))
    g['fg_leaf_name'] = new_fg_leaf_names
    return g

def get_marginal_branch(g):
    marginal_ids = list()
    for node in g['tree'].traverse():
        if (node.is_root()):
            continue
        if (node.is_foreground==False):
            continue
        if (g['mg_parent']):
            if (node.up.is_foreground==False):
                marginal_ids.append(node.up.numerical_label)
        if (g['mg_sister']):
            sisters = node.get_sisters()
            for sister in sisters:
                if (g['mg_sister_stem_only']):
                    if sister.is_foreground==False:
                        marginal_ids.append(sister.numerical_label)
                else:
                    for sister_des in sister.traverse():
                        if sister_des.is_foreground==False:
                            marginal_ids.append(sister_des.numerical_label)
    g['mg_id'] = numpy.array(list(set(marginal_ids)-set(g['target_id'])), dtype=numpy.int64)
    g['target_id'] = numpy.concatenate([g['target_id'], g['mg_id']])
    for node in g['tree'].traverse():
        node.is_marginal = False # initializing
        if node.numerical_label in g['mg_id']:
            node.is_marginal = True
    with open('csubst_marginal_branch.txt', 'w') as f:
        for x in g['mg_id']:
            f.write(str(x)+'\n')
    return g

def get_foreground_branch_num(cb, g):
    start_time = time.time()
    bid_cols = cb.columns[cb.columns.str.startswith('branch_id_')]
    arity = len(bid_cols)
    if g['foreground'] is None:
        cb.loc[:,'branch_num_fg'] = arity
        cb.loc[:,'branch_num_mg'] = 0
    else:
        for id_key,newcol in zip(['fg_id','mg_id'],['branch_num_fg','branch_num_mg']):
            if not id_key in g.keys():
                continue
            id_set = set(g[id_key])
            cb.loc[:,newcol] = 0
            for i in cb.index:
                branch_id_set = set(cb.loc[i,bid_cols].values)
                cb.at[i,newcol] = arity - len(branch_id_set.difference(id_set))
    cb.loc[:,'is_fg'] = 'N'
    cb.loc[(cb.loc[:,'branch_num_fg']==arity),'is_fg'] = 'Y'
    if (g['fg_dependent_id_combinations'] is not None):
        for i in numpy.arange(g['fg_dependent_id_combinations'].shape[0]):
            is_dep = True
            for i,bids in enumerate(g['fg_dependent_id_combinations'][i,:]):
                is_dep = is_dep & (cb.loc[:,'branch_id_'+str(i+1)]==bids)
            cb.loc[is_dep,'is_fg'] = 'N'
    cb.loc[:,'is_mg'] = 'N'
    cb.loc[(cb.loc[:,'branch_num_mg']==arity),'is_mg'] = 'Y'
    cb.loc[:,'is_mf'] = 'N'
    is_mg = (cb.loc[:,'branch_num_fg']>0)& (cb.loc[:,'branch_num_mg']>0)
    is_mg = (is_mg)&((cb.loc[:,'branch_num_fg']+cb.loc[:,'branch_num_mg'])==arity)
    cb.loc[is_mg,'is_mf'] = 'Y'
    is_foreground = (cb.loc[:,'is_fg']=='Y')
    is_enough_stat = table.get_cutoff_stat_bool_array(cb=cb, cutoff_stat_str=g['cutoff_stat'])
    num_enough = is_enough_stat.sum()
    num_fg = is_foreground.sum()
    num_fg_enough = (is_enough_stat&is_foreground).sum()
    num_all = cb.shape[0]
    if (num_enough==0)|(num_fg==0):
        percent_fg_enough = 0
        enrichment_factor = 0
    else:
        percent_fg_enough = num_fg_enough / num_enough * 100
        enrichment_factor = (num_fg_enough/num_enough) / (num_fg/num_all)
    txt = 'arity={}, foreground branch combinations with cutoff conditions {} = {:.0f}% ({:,}/{:,}, ' \
          'total examined = {:,}, enrichment factor = {:.1f})'
    txt = txt.format(arity, g['cutoff_stat'], percent_fg_enough, num_fg_enough, num_enough,
                     num_all, enrichment_factor)
    print(txt, flush=True)
    is_arity = (g['df_cb_stats'].loc[:,'arity']==arity)
    g['df_cb_stats'].loc[is_arity,'fg_enrichment_factor'] = enrichment_factor
    print('Time elapsed for obtaining foreground branch numbers: {:,} sec'.format(int(time.time() - start_time)))
    return cb, g

def annotate_foreground(b, g):
    if len(g['fg_id']):
        b.loc[:,'is_fg'] = 'no'
        b.loc[g['fg_id'],'is_fg'] = 'yes'
    else:
        b.loc[:,'is_fg'] = 'yes'
    if len(g['mg_id']):
        b.loc[:,'is_mg'] = 'no'
        b.loc[g['mg_id'],'is_mg'] = 'yes'
    else:
        b.loc[:,'is_mg'] = 'no'
    return b

def get_num_foreground_lineages(tree):
    num_fl = 0
    for node in tree.traverse():
        num_fl = max(num_fl, node.foreground_lineage_id)
    return num_fl

def set_random_foreground_branch(g, num_trial=100):
    for i in numpy.arange(num_trial):
        g = get_foreground_branch(g)
        g = randomize_foreground_branch(g)
        if g['target_id'].shape[0]<2:
            continue
        g,rid_combinations = combination.get_node_combinations(g, target_nodes=g['target_id'], arity=g['current_arity'],
                                                               check_attr="name", verbose=False)
        if rid_combinations.shape[0]==0:
            continue
        txt = 'Foreground branch permutation finished after {:,} trials.'
        print(txt.format(i+1))
        return g,rid_combinations
    txt = 'Foreground branch permutation failed {:,} times. There may not be enough numbers of "similar" clades.\n'
    raise Exception(txt.format(num_trial))

def add_median_cb_stats(g, cb, current_arity, start, verbose=True):
    is_arity = (g['df_cb_stats'].loc[:,'arity'] == current_arity)
    suffices = list()
    is_targets = list()
    suffices.append('_all')
    is_targets.append(numpy.ones(shape=cb.shape[0], dtype=bool))
    target_cols = ['is_fg','is_mg','is_mf','dummy']
    suffix_candidates = ['_fg','_mg','_mf']
    if g['exhaustive_until']>=current_arity:
        suffix_candidates.append('_all')
    for target_col,sc in zip(target_cols,suffix_candidates):
        if target_col in cb.columns:
            suffices.append(sc)
            if sc=='_all':
                is_targets.append(True)
            else:
                is_targets.append(cb.loc[:,target_col]=='Y')
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
    for suffix, is_target in zip(suffices, is_targets):
        g['df_cb_stats'].loc[is_arity, 'num' + suffix] = is_target.sum()
        g['df_cb_stats'].loc[is_arity, 'num_qualified' + suffix] = (is_target&is_qualified).sum()
    for stat in stats.keys():
        for suffix,is_target in zip(suffices,is_targets):
            for ms in stats[stat]:
                col = stat+'_'+ms+suffix
                if not col in g['df_cb_stats'].columns:
                    newcol = pandas.DataFrame({col:numpy.zeros(shape=(g['df_cb_stats'].shape[0]))})
                    g['df_cb_stats'] = pandas.concat([g['df_cb_stats'], newcol], ignore_index=False, axis=1)
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
            txt = 'Total OC{}/EC{} = {:,.1f}/{:,.1f} (Expectation equals to {:,.1f}% of the observation.)'
            print(txt.format(key, key, totalON, totalEN, percent_value))
    elapsed_time = int(time.time() - start)
    g['df_cb_stats'].loc[is_arity, 'elapsed_sec'] = elapsed_time
    if verbose:
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)
    return g

def clade_permutation(cb, g):
    print('Starting foreground clade permutation. Note that --fg_clade_permutation examine the arity of 2.')
    i = 1
    for trial_no in numpy.arange(g['fg_clade_permutation']*10):
        start = time.time()
        txt = 'Starting foreground clade permutation round {:,} (of {:,})'
        print(txt.format(i, g['fg_clade_permutation']), flush=True)
        g = param.initialize_df_cb_stats(g)
        g['df_cb_stats'] = g['df_cb_stats'].loc[(g['df_cb_stats'].loc[:,'arity']==2),:].reset_index(drop=True)
        g,rid_combinations = set_random_foreground_branch(g)
        random_mode = 'randomization_iter'+str(i)+'_bid'+','.join(g['fg_id'].astype(str))
        bid_columns = [ 'branch_id_'+str(k+1) for k in numpy.arange(rid_combinations.shape[1]) ]
        rid_combinations = pandas.DataFrame(rid_combinations)
        rid_combinations.columns = bid_columns
        rcb = pandas.merge(rid_combinations, cb, how='inner', on=bid_columns)
        if (rid_combinations.shape[0] != rcb.shape[0]):
            txt = '{:,} ({:,} - {:,}) permuted foreground branch combinations were dropped because they were not included in the cb table.'
            print(txt.format(rid_combinations.shape[0]-rcb.shape[0], rid_combinations.shape[0], rcb.shape[0]))
        rcb.loc[:,'is_fg'] = 'Y'
        rcb.loc[:,'is_mg'] = 'N'
        rcb.loc[:,'is_mf'] = 'N'
        g = add_median_cb_stats(g, rcb, 2, start, verbose=False)
        g['df_cb_stats'].loc[:,'mode'] = random_mode
        if numpy.isnan(g['df_cb_stats'].loc[:,'median_omegaCany2spe_fg'].values[0]):
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

    is_arity2 = (g['df_cb_stats_main'].loc[:,'arity']==2)
    is_stat_fg = ~g['df_cb_stats_main'].loc[:,'mode'].str.startswith('randomization_')
    is_stat_permutation = g['df_cb_stats_main'].loc[:,'mode'].str.startswith('randomization_')
    obs_value = g['df_cb_stats_main'].loc[is_arity2 & is_stat_fg,'median_omegaCany2spe_fg'].values[0]
    permutation_values = g['df_cb_stats_main'].loc[is_arity2 & is_stat_permutation, 'median_omegaCany2spe_fg'].values
    num_positive = (obs_value<=permutation_values).sum()
    num_all = permutation_values.shape[0]
    pvalue = num_positive / num_all
    obs_ocn = g['df_cb_stats_main'].loc[is_arity2 & is_stat_fg,'total_OCNany2spe_fg'].values[0]
    print('Observed total OCNany2spe in foreground lineages = {:,.3}'.format(obs_ocn))
    permutation_ocns = g['df_cb_stats_main'].loc[is_arity2 & is_stat_permutation, 'total_OCNany2spe_fg'].values
    txt = 'Total OCNany2spe in permutation lineages = {:,.3}; {:,.3} ± {:,.3} (median; mean ± SD excluding inf)'
    print(txt.format(numpy.median(permutation_ocns), permutation_ocns.mean(), permutation_ocns.std()))
    print('Observed median omegaCany2spe in foreground lineages = {:,.3}'.format(obs_value))
    txt = 'Median omegaCany2spe in permutation lineages = {:,.3}; {:,.3} ± {:,.3} (median; mean ± SD excluding inf)'
    print(txt.format(numpy.median(permutation_values), permutation_values[numpy.isfinite(permutation_values)].mean(), permutation_values[numpy.isfinite(permutation_values)].std()))
    txt = 'P value of foreground convergence (omegaCany2spe) by clade permutations = {} (observation <= permutation = {:,}/{:,})'
    print(txt.format(pvalue, num_positive, num_all))
    return g