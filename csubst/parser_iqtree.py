import os
import re
import ete3
import pandas
from csubst.tree import *
from csubst.genetic_code import ambiguous_table

def read_treefile(g):
    g['rooted_tree'] = ete3.PhyloNode(g['rooted_tree_file'], format=1)
    assert len(g['rooted_tree'].get_children())==2, 'The input tree may be unrooted: {}'.format(g['rooted_tree_file'])
    g['rooted_tree'] = standardize_node_names(g['rooted_tree'])
    g['rooted_tree'] = add_numerical_node_labels(g['rooted_tree'])
    g['num_node'] = len(list(g['rooted_tree'].traverse()))
    g['iqtree_rate_values'] = read_rate(g)
    print('Using internal node names and branch lengths in --iqtree_treefile and the root position in --rooted_tree_file.')
    g['node_label_tree_file'] = g['iqtree_treefile']
    f = open(g['node_label_tree_file'])
    tree_string = f.readline()
    g['node_label_tree'] = ete3.PhyloNode(tree_string, format=1)
    f.close()
    g['node_label_tree'] = standardize_node_names(g['node_label_tree'])
    g['tree'] = transfer_root_and_dist(tree_to=g['node_label_tree'], tree_from=g['rooted_tree'], verbose=False)
    g['tree'] = add_numerical_node_labels(g['tree'])
    print('Total branch length of --rooted_tree_file:', sum([ n.dist for n in g['rooted_tree'].traverse() ]))
    print('Total branch length of --iqtree_treefile:', sum([ n.dist for n in g['node_label_tree'].traverse() ]))
    print('')
    return g

def read_state(g):
    print('Reading the state file:', g['iqtree_state'])
    state_table = pandas.read_csv(g['iqtree_state'], sep="\t", index_col=False, header=0, comment='#')
    g['num_input_site'] = state_table['Site'].unique().shape[0]
    g['num_input_state'] = state_table.shape[1] - 3
    g['input_state'] = state_table.columns[3:].str.replace('p_','').tolist()
    if g['num_input_state']==4:
        g['input_data_type'] = 'nuc'
    elif g['num_input_state']==20:
        g['input_data_type'] = 'pep'
    elif g['num_input_state'] > 20:
        g['input_data_type'] = 'cdn'
        g['codon_orders'] = state_table.columns[3:].str.replace('p_','').values
    if (g['input_data_type']=='cdn'):
        g['amino_acid_orders'] = sorted(list(set([ c[0] for c in g['codon_table'] if c[0]!='*' ])))
        matrix_groups = dict()
        for aa in list(set(g['amino_acid_orders'])):
            matrix_groups[aa] = [ c[1] for c in g['codon_table'] if c[0]==aa ]
        g['matrix_groups'] = matrix_groups
        synonymous_indices = dict()
        for aa in matrix_groups.keys():
            synonymous_indices[aa] = []
        for i,c in enumerate(g['codon_orders']):
            for aa in matrix_groups.keys():
                if c in matrix_groups[aa]:
                    synonymous_indices[aa].append(i)
                    break
        g['synonymous_indices'] = synonymous_indices
        g['max_synonymous_size'] = max([ len(si) for si in synonymous_indices.values() ])
    print('')
    return g

def read_rate(g):
    if (g['iqtree_rate']=='infer'):
        file_path = g['aln_file']+'.rate'
    else:
        file_path = g['iqtree_rate']
    err_txt = 'IQ-TREE\'s .rate file was not found in {}. Please specify the correct file PATH by --iqtree_rate.'
    assert os.path.exists(file_path), err_txt.format(file_path)
    print('IQ-TREE\'s .rate file was detected. Loading.')
    sub_sites = pandas.read_csv(file_path, sep='\t', header=0, comment='#')
    sub_sites = sub_sites.loc[:,'C_Rate'].values
    return sub_sites

def read_iqtree(g):
    if (g['iqtree_iqtree']=='infer'):
        file_path = g['aln_file']+'.iqtree'
    else:
        file_path = g['iqtree_iqtree']
    if not os.path.exists(file_path):
        print('File not found:', file_path)
        return g
    with open(file_path) as f:
        lines = f.readlines()
    for line in lines:
        model = re.match(r'Model of substitution: (.+)', line)
        if model is not None:
            g['substitution_model'] = model.group(1)
    with open(file_path) as f:
        txt = f.read()
    pi = pandas.DataFrame(index=g['codon_orders'], columns=['freq',])
    for m in re.finditer(r'  pi\(([A-Z]+)\) = ([0-9.]+)', txt, re.MULTILINE):
        pi.at[m.group(1),'freq'] = float(m.group(2))
    g['equilibrium_frequency'] = pi.loc[:,'freq'].values.astype(g['float_type'])
    g['equilibrium_frequency'] /= g['equilibrium_frequency'].sum()
    return g

def read_log(g):
    g['omega'] = None
    g['kappa'] = None
    g['reconstruction_codon_table'] = None
    if (g['iqtree_log']=='infer'):
        file_path = g['aln_file']+'.log'
    else:
        file_path = g['iqtree_log']
    if not os.path.exists(file_path):
        print('File not found:', file_path)
        return g
    with open(file_path) as f:
        lines = f.readlines()
    for line in lines:
        omega = re.match(r'Nonsynonymous/synonymous ratio \(omega\): ([0-9.]+)', line)
        if omega is not None:
            g['omega'] = float(omega.group(1))
        kappa = re.match(r'Transition/transversion ratio \(kappa\): ([0-9.]+)', line)
        if kappa is not None:
            g['kappa'] = float(kappa.group(1))
        rgc = re.match(r'Converting to codon sequences with genetic code ([0-9]+) \.\.\.', line)
        if rgc is not None:
            g['reconstruction_codon_table'] = int(rgc.group(1))
    return g

def get_state_index(state, input_state, ambiguous_table):
    #input_state = pandas.Series(input_state)
    if isinstance(state, str):
        states = [state,]
    else:
        print('state should be str instance.')
    state_set = set(list(state))
    key_set = set(ambiguous_table.keys())
    if (len(state_set.intersection(key_set))>0):
        for amb in ambiguous_table.keys():
            vals = ambiguous_table[amb]
            states = [ s.replace(amb, val) for s in states for val in vals ]
    state_index = [ int(numpy.where(input_state==s)[0]) for s in states ]
    return state_index

def mask_missing_sites(state_tensor, tree):
    for node in tree.traverse():
        if (node.is_root())|(node.is_leaf()):
            continue
        nl = node.numerical_label
        child0_leaf_nls = numpy.array([ l.numerical_label for l in node.get_children()[0].get_leaves() ], dtype=int)
        child1_leaf_nls = numpy.array([ l.numerical_label for l in node.get_children()[1].get_leaves() ], dtype=int)
        sister_leaf_nls = numpy.array([ l.numerical_label for l in node.get_sisters()[0].get_leaves() ], dtype=int)
        c0 = (state_tensor[child0_leaf_nls,:,:].sum(axis=(0,2))!=0) # is_child0_leaf_nonzero
        c1 = (state_tensor[child1_leaf_nls,:,:].sum(axis=(0,2))!=0) # is_child1_leaf_nonzero
        s = (state_tensor[sister_leaf_nls,:,:].sum(axis=(0,2))!=0) # is_sister_leaf_nonzero
        is_nonzero = (c0&c1)|(c0&s)|(c1&s)
        state_tensor[nl,:,:] = numpy.einsum('ij,i->ij', state_tensor[nl,:,:], is_nonzero)
    return state_tensor

def get_state_tensor(g):
    g['tree'].link_to_alignment(alignment=g['aln_file'], alg_format='fasta')
    num_node = len(list(g['tree'].traverse()))
    state_table = pandas.read_csv(g['iqtree_state'], sep="\t", index_col=False, header=0, comment='#')
    axis = [num_node, g['num_input_site'], g['num_input_state']]
    state_tensor = numpy.zeros(axis, dtype=g['float_type'])
    for node in g['tree'].traverse():
        if node.is_root():
            continue
        elif node.is_leaf():
            seq = node.sequence.upper()
            state_matrix = numpy.zeros([g['num_input_site'], g['num_input_state']], dtype=g['float_type'])
            if g['input_data_type']=='cdn':
                assert len(seq)%3==0, 'Sequence length is not multiple of 3. Node name = '+node.name
                for s in numpy.arange(int(len(seq)/3)):
                    codon = seq[(s*3):((s+1)*3)]
                    if (not '-' in codon)&(codon!='NNN'):
                        codon_index = get_state_index(state=codon, input_state=g['codon_orders'], ambiguous_table=ambiguous_table)
                        for ci in codon_index:
                            state_matrix[s,ci] = 1/len(codon_index)
            elif g['input_data_type']=='nuc':
                for s in numpy.arange(len(seq)):
                    if seq[s]!='-':
                        nuc_index = get_state_index(state=seq[s], input_state=g['input_state'], ambiguous_table=ambiguous_table)
                        for ni in nuc_index:
                            state_matrix[s, ni] = 1/len(nuc_index)
            state_tensor[node.numerical_label,:,:] = state_matrix
        else:
            state_matrix = state_table.loc[(state_table['Node']==node.name),:].iloc[:,3:]
            is_missing = (state_table.loc[:,'State']=='???') | (state_table.loc[:,'State']=='?')
            state_matrix.loc[is_missing,:] = 0
            if state_matrix.shape[0]==0:
                print('Node name not found in .state file:', node.name)
            else:
                state_tensor[node.numerical_label,:,:] = state_matrix
    state_tensor = mask_missing_sites(state_tensor, g['tree'])
    if (g['ml_anc']):
        print('Ancestral state frequency is converted to the ML-like binary states.')
        idxmax = numpy.argmax(state_tensor, axis=2)
        state_tensor2 = numpy.zeros(state_tensor.shape, dtype=numpy.bool_)
        for b in numpy.arange(state_tensor2.shape[0]):
            for s in numpy.arange(state_tensor2.shape[1]):
                if state_tensor[b,s,:].sum()!=0:
                    state_tensor2[b,s,idxmax[b,s]] = 1
        state_tensor = state_tensor2
        del state_tensor2
    return(state_tensor)

def get_input_information(g):
    g = read_treefile(g)
    g = read_state(g)
    g = read_iqtree(g)
    g = read_log(g)
    return g