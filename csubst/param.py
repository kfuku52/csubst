import os
import numpy
import pandas
import itertools
import re
import copy

def get_global_parameters(args):
    g = dict()
    for attr in [a for a in dir(args) if not a.startswith('_')]:
        g[attr] = getattr(args, attr)
    if not g['infile_dir'].endswith('/'):
        g['infile_dir'] = g['infile_dir'] + '/'
    if (g['aln_file'] == '') & (g['infile_type'] in ['iqtree', ]):
        files = os.listdir(g['infile_dir'])
        aln_files = list()
        for ext in ['.fasta','.fa','.fas']:
            aln_files = aln_files + [ f for f in files if f.endswith(ext) ]
        if len(aln_files)==1:
            g['aln_file'] = g['infile_dir']+aln_files[0]
            print('Alignment file found:', g['aln_file'])
        else:
            print('Alignment file not found. Use --aln option.', aln_files)
    if (g['tre_file'] == '') & (g['infile_type'] in ['iqtree', ]):
        files = os.listdir(g['infile_dir'])
        tree_files = list()
        for ext in ['.r.nwk']:
            tree_files = tree_files + [ f for f in files if f.endswith(ext) ]
        if len(tree_files)==1:
            g['tre_file'] = g['infile_dir']+tree_files[0]
            print('Tree file found:', g['tre_file'])
        else:
            print('The rooted tree file not found. Use --tre option.', tree_files)
    return g

def get_dep_ids(g):
    dep_ids = list()
    for leaf in g['tree'].iter_leaves():
        dep_id = [leaf.numerical_label,] + [ node.numerical_label for node in leaf.iter_ancestors() if not node.is_root() ]
        dep_id = numpy.sort(numpy.array(dep_id))
        dep_ids.append(dep_id)
    if g['exclude_sisters']:
        for node in g['tree'].traverse():
            children = node.get_children()
            if len(children)>1:
                dep_id = numpy.sort(numpy.array([ node.numerical_label for node in children ]))
                dep_ids.append(dep_id)
    g['dep_ids'] = dep_ids
    if (g['foreground'] is not None)&(g['exclude_wg']):
        fg_dep_ids = list()
        for node in g['tree'].traverse():
            for i in numpy.arange(len(g['fg_leaf_name'])):
                if all([ ln in g['fg_leaf_name'][i] for ln in node.get_leaf_names() ])&(len(node.get_leaf_names())>1):
                    flag = 0
                    if node.is_root():
                        flag = 1
                    elif not all([ ln in g['fg_leaf_name'][i] for ln in node.up.get_leaf_names() ]):
                        flag = 1
                    if flag:
                        fg_dep_ids.append(numpy.array([node.numerical_label,] + [ n.numerical_label for n in node.get_descendants() ]))
        if (g['fg_sister'])|(g['fg_parent']):
            fg_dep_ids.append(numpy.array(g['marginal_id']))
        g['fg_dep_ids'] = fg_dep_ids
    else:
        g['fg_dep_ids'] = numpy.array([])
    return g

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

def foreground_clade_randomization(df_clade_size, g, min_bin_count=10):
    size_array = df_clade_size.loc[:,'size'].values.astype(numpy.int)
    size_min = size_array.min()
    size_max = size_array.max()
    sizes = numpy.unique(size_array)[::-1] # To start counting from rarer (larger) clades
    bins = numpy.array([size_max+1,], dtype=numpy.int)
    count = 0
    for size in sizes:
        is_size = (size_array==size)
        count += is_size.sum()
        if (count >= min_bin_count):
            bins = numpy.append(bins, size)
            count = 0
    if len(bins)<2:
        bins = numpy.array([size_min, size_max], dtype=numpy.int)
    bins = bins[::-1]
    df_clade_size.loc[:,'bin'] = numpy.digitize(size_array, bins, right=False)
    is_fg = (df_clade_size.loc[:,'is_foreground_stem']==True)
    fg_bins = df_clade_size.loc[is_fg,'bin']
    df_clade_size.loc[:,'is_foreground_stem_randomized'] = df_clade_size.loc[:,'is_foreground_stem']
    df_clade_size.loc[:,'is_blocked'] = False
    for bin in df_clade_size.loc[:,'bin'].unique()[::-1]:
        is_bin = (df_clade_size.loc[:,'bin']==bin)
        is_blocked = df_clade_size.loc[:,'is_blocked'].values
        min_bin_size = df_clade_size.loc[is_bin,'size'].min()
        max_bin_size = df_clade_size.loc[is_bin,'size'].max()
        num_fg_bin = (fg_bins==bin).sum()
        num_bin = is_bin.sum()
        num_unblocked_bin = (is_bin&~is_blocked).sum()
        cc_w = combinations_count(n=num_unblocked_bin, r=num_fg_bin)
        cc_wo = combinations_count(n=num_bin, r=num_fg_bin)
        txt = 'bin {}: foreground/all clades = {:,}/{:,}, ' \
              'min/max clade sizes = {:,}/{:,}, ' \
              'randomization complexity with/without considering branch independency = {:,}/{:,}'
        print(txt.format(bin, num_fg_bin, num_bin, min_bin_size, max_bin_size, cc_w, cc_wo), flush=True)
        before_randomization = df_clade_size.loc[is_bin&~is_blocked,'is_foreground_stem_randomized'].values
        after_randomization = numpy.random.permutation(before_randomization)
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
    new_fg_ids = numpy.array(new_fg_ids, dtype=numpy.int)
    return new_fg_ids

def get_foreground_branch(g):
    g['fg_df'] = pandas.read_csv(g['foreground'], sep='\t', comment='#', skip_blank_lines=True, header=None)
    g['fg_leaf_name'] = list()
    g['target_id'] = numpy.zeros(shape=(0,), dtype=numpy.int) # numerical_label for leaves in --foreground as well as their ancestors
    leaf_names = [ leaf.name for leaf in g['tree'].get_leaves() ]
    for node in g['tree'].traverse():
        node.is_foreground = False # initializing
    lineages = g['fg_df'].iloc[:,0].unique()
    for i in numpy.arange(len(lineages)):
        g['fg_leaf_name'].append([])
        is_lineage = (g['fg_df'].iloc[:,0]==lineages[i])
        lineage_regex_names = g['fg_df'].loc[is_lineage,:].iloc[:,1].unique().tolist()
        iter = itertools.product(leaf_names, lineage_regex_names)
        lineage_leaf_names = [ ln for ln,lr in iter if re.match(lr, ln) ]
        lineage_fg_id = list()
        for lln in lineage_leaf_names:
            match_leaves = [ ln for ln in leaf_names if lln==ln ]
            if len(match_leaves)==1:
                g['fg_leaf_name'][i].append(match_leaves[0])
            else:
                print('The foreground leaf name cannot be identified:', lln, match_leaves)
        fg_leaf_name_set = set(g['fg_leaf_name'][i])
        if (g['fg_stem_only']|g['fg_random']):
            for node in g['tree'].traverse():
                node.is_lineage_foreground = False # initializing
            for node in g['tree'].traverse():
                node_leaf_name_set = set(node.get_leaf_names())
                if len(node_leaf_name_set.difference(fg_leaf_name_set))==0:
                    node.is_lineage_foreground = True
                    node.is_foreground = True
        for node in g['tree'].traverse():
            node_leaf_name_set = set(node.get_leaf_names())
            if len(node_leaf_name_set.difference(fg_leaf_name_set))==0:
                if g['fg_stem_only']:
                    if (node.is_lineage_foreground==True)&(node.up.is_lineage_foreground==False):
                        lineage_fg_id.append(node.numerical_label)
                else:
                    lineage_fg_id.append(node.numerical_label)
        dif = 1
        while dif:
            num_id = len(lineage_fg_id)
            for node in g['tree'].traverse():
                child_ids = [ child.numerical_label for child in node.get_children() ]
                if all([ id in lineage_fg_id for id in child_ids ])&(len(child_ids)!=0):
                    if node.numerical_label not in lineage_fg_id:
                        lineage_fg_id.append(node.numerical_label)
            dif = len(lineage_fg_id) - num_id
        tmp = numpy.array(lineage_fg_id, dtype=numpy.int)
        g['target_id'] = numpy.concatenate([g['target_id'], tmp])
    with open('csubst_target_branch.txt', 'w') as f:
        for x in g['target_id']:
            f.write(str(x)+'\n')
    g['fg_id'] = copy.deepcopy(g['target_id']) # marginal_ids may be added to target_id but fg_id won't be changed.
    return g

def randomize_foreground_branch(g):
    g['fg_id_original'] = copy.deepcopy(g['fg_id'])
    g['fg_leaf_name_original'] = copy.deepcopy(g['fg_leaf_name'])
    df_clade_size = get_df_clade_size(g)
    df_clade_size = foreground_clade_randomization(df_clade_size, g, min_bin_count=10)
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
        if (node.numerical_label in g['target_id'])&(not node.is_root()):
            if (g['fg_parent']):
                marginal_ids.append(node.up.numerical_label)
            if (g['fg_sister']):
                sisters = node.get_sisters()
                if (g['fg_sister_stem_only']):
                    marginal_ids += [ sister.numerical_label for sister in sisters ]
                else:
                    for sister in sisters:
                        marginal_ids += [ sister_des.numerical_label for sister_des in sister.traverse() ]
    g['marginal_id'] = numpy.array(list(set(marginal_ids)-set(g['target_id'])), dtype=numpy.int)
    g['target_id'] = numpy.concatenate([g['target_id'], g['marginal_id']])
    with open('csubst_marginal_branch.txt', 'w') as f:
        for x in g['marginal_id']:
            f.write(str(x)+'\n')
    return g

def get_foreground_branch_num(cb, g):
    bid_cols = cb.columns[cb.columns.str.startswith('branch_id_')]
    arity = len(bid_cols)
    if g['foreground'] is None:
        cb.loc[:,'fg_branch_num'] = arity
    else:
        fg_id_set = set(g['fg_id'])
        cb.loc[:,'fg_branch_num'] = 0
        for i in cb.index:
            branch_id_set = set(cb.loc[i,bid_cols].values)
            cb.at[i,'fg_branch_num'] = arity - len(branch_id_set.difference(fg_id_set))
    is_enough_stat = (cb.loc[:,g['target_stat']]>=g['min_stat'])
    is_foreground = (cb.loc[:,'fg_branch_num']==arity)
    num_enough = is_enough_stat.sum()
    num_fg = is_foreground.sum()
    num_fg_enough = (is_enough_stat&is_foreground).sum()
    num_all = cb.shape[0]
    if (num_enough==0)|(num_fg==0):
        percent_fg_enough = 0
        conc_factor = 0
    else:
        percent_fg_enough = num_fg_enough / num_enough * 100
        conc_factor = (num_fg_enough/num_enough) / (num_fg/num_all)
    txt = 'arity={}, foreground branch combinations with {} >= {} = {:.0f}% ({:,}/{:,}, ' \
          'total examined = {:,}, concentration factor = {:.1f})'
    txt = txt.format(arity, g['target_stat'], g['min_stat'], percent_fg_enough, num_fg_enough, num_enough, num_all, conc_factor)
    print(txt, flush=True)
    is_arity = (g['df_cb_stats'].loc[:,'arity']==arity)
    g['df_cb_stats'].loc[is_arity,'num_examined'] = num_all
    g['df_cb_stats'].loc[is_arity,'num_fg'] = num_fg
    g['df_cb_stats'].loc[is_arity,'num_qualified'] = num_enough
    g['df_cb_stats'].loc[is_arity,'num_fg_qualified'] = num_fg_enough
    g['df_cb_stats'].loc[is_arity,'fg_conc_factor'] = conc_factor
    return cb, g

def annotate_foreground(b, g):
    if len(g['fg_id']):
        b.loc[:,'is_foreground'] = 'no'
        b.loc[g['fg_id'],'is_foreground'] = 'yes'
    else:
        b.loc[:,'is_foreground'] = 'yes'
    if len(g['marginal_id']):
        b.loc[:,'is_marginal'] = 'no'
        b.loc[g['marginal_id'],'is_marginal'] = 'yes'
    else:
        b.loc[:,'is_marginal'] = 'no'
    return b

def initialize_df_cb_stats(g):
    if g['cb_stats'] is None:
        ind = numpy.arange(0, g['max_arity'])
        cols = ['arity','elapsed_sec','fg_median_dist_bl','fg_median_dist_node_num',]
        cols = cols + ['num_examined','num_fg','num_qualified','num_fg_qualified','fg_conc_factor',]
        g['df_cb_stats'] = pandas.DataFrame(index=ind, columns=cols)
        g['df_cb_stats'].loc[:,'arity'] = ind + 1
        g['df_cb_stats'].loc[:,'target_stat'] = g['target_stat']
        g['df_cb_stats'].loc[:,'min_stat'] = g['min_stat']
    else:
        g['df_cb_stats'] = pandas.read_csv(g['cb_stats'], sep='\t', header=0)
    return(g)
