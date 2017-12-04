import os
import sys
import itertools
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

def add_numerical_node_labels(tree):
    all_leaf_names = tree.get_leaf_names()
    all_leaf_names.sort()
    leaf_numerical_labels = dict()
    power = 0
    for i in range(0, len(all_leaf_names)):
        leaf_numerical_labels[all_leaf_names[i]] = 2**i
    numerical_labels = list()
    for node in tree.traverse():
        leaf_names = node.get_leaf_names()
        numerical_labels.append(sum([leaf_numerical_labels[leaf_name] for leaf_name in leaf_names]))
    argsort_labels = numpy.argsort(numerical_labels)
    short_labels = numpy.arange(len(argsort_labels))
    i=0
    for node in tree.traverse():
        node.numerical_label = short_labels[argsort_labels==i][0]
        i+=1
    return(tree)

def detach_node_posterior(tree):
    for node in tree.traverse():
        pp_attributes = [ attr for attr in dir(node) if attr.startswith('pp_') ]
        for attr in pp_attributes:
            delattr(node, attr)
    return (tree)

def sort_labels(df):
    swap_columns = df.columns[df.columns.str.startswith('branch_id')].tolist()
    if len(swap_columns)>1:
        swap_values = df.loc[:,swap_columns].values
        swap_values.sort(axis=1)
        df.loc[:,swap_columns] = swap_values
    if 'site' in df.columns:
        swap_columns.append('site')
    df = df.sort_values(by=swap_columns)
    for cn in swap_columns:
        df.loc[:,cn] = df[cn].astype(int)
    return df

def merge_tables(df1, df2):
    columns = []
    columns = columns + df1.columns[df1.columns.str.startswith('branch_name')].tolist()
    columns = columns + df1.columns[df1.columns.str.startswith('branch_id')].tolist()
    columns = columns + df1.columns[df1.columns.str.startswith('site')].tolist()
    df = pandas.merge(df1, df2, on=columns)
    df = sort_labels(df=df)
    return df

def get_num_site(tree):
    for node in tree.traverse():
        if 'pp_N' in dir(node):
            num_site = node.pp_N.shape[0]
            break
    return(num_site)

def prepare_node_combinations(g, target_nodes=None, arity=2, check_attr=None, verbose=True):
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
    else:
        target_nodes = [ set(tn) for tn in target_nodes ]
        node_combinations = list()
        for tn1 in target_nodes:
            for tn2 in target_nodes[1:]:
                node_union = tn1.union(tn2)
                if (len(node_union)==arity):
                    if node_union not in node_combinations:
                        node_combinations.append(node_union)
    if verbose:
        print("target nodes:", len(target_nodes), flush=True)
        print("all node combinations: ", len(node_combinations), flush=True)
    node_combinations = numpy.array([list(nc) for nc in node_combinations])
    nc_matrix = numpy.zeros(shape=(len(all_nodes), node_combinations.shape[0]), dtype=numpy.bool_, order='C')
    for i in numpy.arange(node_combinations.shape[0]):
        nc_matrix[node_combinations[i,:],i] = 1
    dep_ids = list()
    for leaf in tree.iter_leaves():
        dep_id = [leaf.numerical_label,] + [ node.numerical_label for node in leaf.iter_ancestors() if not node.is_root() ]
        dep_id = numpy.sort(numpy.array(dep_id))
        dep_ids.append(dep_id)
    if g['exclude_sisters']:
        for node in tree.traverse():
            #if 'get_children' in dir(node):
            children = node.get_children()
            if len(children)>1:
                dep_id = numpy.sort(numpy.array([ node.numerical_label for node in children ]))
                dep_ids.append(dep_id)
    is_dependent_col = False
    for dep_id in dep_ids:
        is_dependent_col = (is_dependent_col)|(nc_matrix[dep_id,:].sum(axis=0)>1)
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

def get_synonymous_groups(codon_table):
    amino_acids = [ c[0] for c in codon_table ]
    codons = [ c[1] for c in codon_table ]
    for aa in list(set(amino_acids)):
        sc

def nuc2cdn_state(state_nuc, g): # TODO, implement, exclude stop codon freq
    num_node = state_nuc.shape[0]
    num_nuc_site = state_nuc.shape[1]
    if num_nuc_site%3 != 0:
        raise Exception('The sequence length is not multiple of 3. num_site =', num_nuc_site)
    num_cdn_site = int(num_nuc_site/3)
    num_cdn_state = len(g['state_columns'])
    axis = [num_node, num_cdn_site, num_cdn_state]
    state_cdn = numpy.zeros(axis, dtype=state_nuc.dtype)
    for i in numpy.arange(len(g['state_columns'])):
        sites = numpy.arange(0, num_nuc_site, 3)
        state_cdn[:, :, i] = state_nuc[:, sites+0, g['state_columns'][i][0]]
        state_cdn[:, :, i] *= state_nuc[:, sites+1, g['state_columns'][i][1]]
        state_cdn[:, :, i] *= state_nuc[:, sites+2, g['state_columns'][i][2]]
    return state_cdn

def cdn2pep_state(state_cdn, g):
    num_node = state_cdn.shape[0]
    num_cdn_site = state_cdn.shape[1]
    num_pep_site = num_cdn_site
    num_pep_state = len(g['amino_acid_orders'])
    axis = [num_node, num_pep_site, num_pep_state]
    state_pep = numpy.zeros(axis, dtype=state_cdn.dtype)
    for i,aa in enumerate(g['amino_acid_orders']):
        state_pep[:, :, i] = state_cdn[:,:,g['synonymous_indices'][aa]].sum(axis=2)
    return state_pep

def get_substitution_tensor(state_tensor, mode, g):
    num_branch = state_tensor.shape[0]
    num_site = state_tensor.shape[1]
    if mode=='asis':
        num_syngroup = 1
        num_state = state_tensor.shape[2]
        diag_zero = numpy.diag([-1] * num_state) + 1
    elif mode=='syn':
        num_syngroup = len(g['amino_acid_orders'])
        num_state = g['max_synonymous_size']
    axis = [num_branch,num_syngroup,num_site,num_state,num_state] # axis = [branch,synonymous_group,site,state_from,state_to]
    sub_tensor = numpy.zeros(axis, dtype=state_tensor.dtype)
    if not g['ml_anc']:
        sub_tensor[:,:,:,:,:] = numpy.nan
    for node in g['tree'].traverse():
        if not node.is_root():
            child = node.numerical_label
            parent = node.up.numerical_label
            if mode=='asis':
                sub_matrix = numpy.einsum("sa,sd,ad->sad", state_tensor[parent, :, :], state_tensor[child, :, :], diag_zero) # s=site, a=ancestral, d=derived
                #sub_matrix = numpy.einsum("ij,jk,ik->jik", state_tensor[parent, :, :], state_tensor[child, :, :], diag_zero)
                sub_tensor[child, 0, :, :, :] = sub_matrix
            elif mode=='syn':
                for s,aa in enumerate(g['amino_acid_orders']):
                    ind = numpy.array(g['synonymous_indices'][aa])
                    size = len(ind)
                    diag_zero = numpy.diag([-1] * size) + 1
                    parent_matrix = state_tensor[parent, :, ind] # axis is swapped, shape=[state,site]
                    child_matrix = state_tensor[child, :, ind] # axis is swapped, shape=[state,site]
                    sub_matrix = numpy.einsum("as,ds,ad->sad", parent_matrix, child_matrix, diag_zero)
                    #sub_matrix = numpy.einsum("ij,jk,ik->jik", state_tensor[parent, :, ind], state_tensor[child, :, ind], diag_zero)
                    sub_tensor[child, s, :, :size, :size] = sub_matrix
    if g['min_sub_pp']!=0:
        sub_tensor = (numpy.nan_to_num(sub_tensor)>=g['min_pp'])
    print(mode, ': size of substitution tensor :', int(sys.getsizeof(sub_tensor) / (1024 * 1024)), 'MB', flush=True)
    return sub_tensor

