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

def get_foreground_branch(g):
    g['fg_df'] = pandas.read_csv(g['foreground'], sep='\t', comment='#', skip_blank_lines=True, header=None)
    g['fg_leaf_name'] = list()
    g['target_id'] = numpy.zeros(shape=(0,), dtype=numpy.int) # numerical_label for leaves in --foreground as well as their ancestors
    leaf_names = [ leaf.name for leaf in g['tree'].get_leaves() ]
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
        if g['fg_stem_only']:
            for node in g['tree'].traverse():
                node.is_foreground = False # initializing
            for node in g['tree'].traverse():
                node_leaf_name_set = set(node.get_leaf_names())
                if len(node_leaf_name_set.difference(fg_leaf_name_set))==0:
                    node.is_foreground = True
        for node in g['tree'].traverse():
            node_leaf_name_set = set(node.get_leaf_names())
            if len(node_leaf_name_set.difference(fg_leaf_name_set))==0:
                if g['fg_stem_only']:
                    if (node.is_foreground==True)&(node.up.is_foreground==False):
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
    num_fg_enough = (is_enough_stat&is_foreground).sum()
    num_all = cb.shape[0]
    percent_fg_enough = num_fg_enough / num_enough * 100
    conc_factor = (num_fg_enough/num_enough) / (num_enough/num_all)
    txt = 'arity={}, foreground branch combinations with {} >= {} = {:.0f}% ({:,}/{:,}, ' \
          'total examined = {:,}, concentration factor = {:.1f})'
    txt = txt.format(arity, g['target_stat'], g['min_stat'], percent_fg_enough, num_fg_enough, num_enough, num_all, conc_factor)
    print(txt, flush=True)
    return cb