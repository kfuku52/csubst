import os
import numpy
import pandas

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
    return g

def get_foreground_branch(g):
    g['fg_input'] = pandas.read_csv(g['foreground'], sep='\t', comment='#', skip_blank_lines=True, header=None)
    g['fg_input'] = g['fg_input'][0].tolist()
    g['fg_leaf_name'] = list()
    leaf_names = [ leaf.name for leaf in g['tree'].get_leaves() ]
    for fg in g['fg_input']:
        match_leaves = [ ln for ln in leaf_names if ln.startswith(fg) ]
        if len(match_leaves)==1:
            g['fg_leaf_name'].append(match_leaves[0])
        else:
            print('The foreground leaf name cannot be identified:', fg, match_leaves)
    g['fg_id'] = [ node.numerical_label for node in g['tree'].traverse() if node.name in g['fg_leaf_name'] ]
    dif = 1
    while dif:
        num_id = len(g['fg_id'])
        for node in g['tree'].traverse():
            child_ids = [ child.numerical_label for child in node.get_children() ]
            if all([ id in g['fg_id'] for id in child_ids ])&(len(child_ids)!=0):
                if node.numerical_label not in g['fg_id']:
                    g['fg_id'].append(node.numerical_label)
                    print(node.numerical_label)
        dif = len(g['fg_id']) - num_id
    return g

