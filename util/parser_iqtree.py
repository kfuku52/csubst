import os
import re
import ete3
from util.util import *

def rerooting_by_topology_matching(tree_from, tree_to):
    tree_from.ladderize()
    tree_to.ladderize()
    print('Before rerooting, Robinson-Foulds distance =', tree_from.robinson_foulds(tree_to, unrooted_trees=True)[0])
    print('Rerooting, Robinson-Foulds distance =')
    outgroup_labels = tree_from.get_descendants()[0].get_leaf_names()
    for node in tree_to.traverse():
        if (node.is_leaf())&(not node.name in outgroup_labels):
            non_outgroup_leaf = node
            break
    tree_to.set_outgroup(non_outgroup_leaf)
    outgroup_ancestor = tree_to.get_common_ancestor(outgroup_labels)
    tree_to.set_outgroup(outgroup_ancestor)
    tree_to = add_numerical_node_labels(tree_to)
    tree_to.ladderize()
    print('After rerooting, Robinson-Foulds distance =', tree_from.robinson_foulds(tree_to, unrooted_trees=False)[0])
    return tree_to

def transfer_internal_node_names(tree_from, tree_to):
    tree_from = add_numerical_node_labels(tree_from)
    tree_to = add_numerical_node_labels(tree_to)
    for t in tree_to.traverse():
        flag = 0
        for f in tree_from.traverse():
            if t.numerical_label==f.numerical_label:
                t.name = f.name
                flag = 1
                break
        if flag==0:
            print('Node name could not be transferred:', t.numerical_label, t.name)
    return tree_to

def get_input_information(g):
    files = os.listdir(g['infile_dir'])
    g['tree'] = ete3.PhyloNode(g['tre_file'], format=1)
    g['tree'] = add_numerical_node_labels(g['tree'])
    internal_node_name = True
    for node in g['tree'].traverse():
        if node.name=='':
            internal_node_name = False
    if not internal_node_name:
        g['node_label_tree_file'] = g['infile_dir']+[ f for f in files if f.endswith('.treefile') ][0]
        f = open(g['node_label_tree_file'])
        tree_string = f.readline()
        g['node_label_tree'] = ete3.PhyloNode(tree_string, format=1)
        f.close()
        for node in g['node_label_tree'].traverse():
            node.name = re.sub('\[.*', '', node.name)
            node.name = re.sub('/.*', '', node.name)
        g['node_label_tree'] = rerooting_by_topology_matching(tree_from=g['tree'], tree_to=g['node_label_tree'])
        g['tree'] = transfer_internal_node_names(tree_from=g['node_label_tree'], tree_to=g['tree'])
    g['num_node'] = len(list(g['tree'].traverse()))
    state_file = [ f for f in files if f.endswith('.state') ][0]
    state_table = pandas.read_csv(g['infile_dir'] + state_file, sep="\t", index_col=False, header=0, comment='#')
    g['num_input_site'] = state_table['Site'].unique().shape[0]
    g['num_input_state'] = state_table.shape[1] - 3
    g['input_state'] = state_table.columns[3:].str.replace('p_','').tolist()
    if g['num_input_state']==4:
        g['input_data_type'] = 'nuc'
    elif g['num_input_state']==20:
        g['input_data_type'] = 'pep'
    elif g['num_input_state'] > 20:
        g['input_data_type'] = 'cdn'
        g['codon_orders'] = state_table.columns[3:].str.replace('p_','').tolist()
    if g['nuc2cdn']:
        g['state_columns'] = list(itertools.product(numpy.arange(len(g['input_state'])), repeat=3))
        codon_orders = list(itertools.product(g['input_state'], repeat=3))
        codon_orders = [ c[0]+c[1]+c[2] for c in codon_orders]
        g['codon_orders'] = codon_orders
    if (g['nuc2cdn'])|(g['input_data_type']=='cdn'):
        g['amino_acid_orders'] = sorted(list(set([ c[0] for c in g['codon_table'] if c[0]!='*' ])))
        synonymous_groups = dict()
        for aa in list(set(g['amino_acid_orders'])):
            synonymous_groups[aa] = [ c[1] for c in g['codon_table'] if c[0]==aa ]
        g['synonymous_groups'] = synonymous_groups
        synonymous_indices = dict()
        for aa in synonymous_groups.keys():
            synonymous_indices[aa] = []
        for i,c in enumerate(g['codon_orders']):
            for aa in synonymous_groups.keys():
                if c in synonymous_groups[aa]:
                    synonymous_indices[aa].append(i)
                    break
        g['synonymous_indices'] = synonymous_indices
        g['max_synonymous_size'] = max([ len(si) for si in synonymous_indices.values() ])
    return g

def get_state_tensor(g):
    g['tree'].link_to_alignment(alignment=g['aln_file'], alg_format='fasta')
    num_node = len(list(g['tree'].traverse()))
    state_file = [ f for f in os.listdir(g['infile_dir']) if f.endswith('.state') ][0]
    state_table = pandas.read_csv(g['infile_dir'] + state_file, sep="\t", index_col=False, header=0, comment='#')
    axis = [num_node, g['num_input_site'], g['num_input_state']]
    state_tensor = numpy.zeros(axis, dtype=numpy.float64)
    for node in g['tree'].traverse():
        if node.is_leaf():
            seq = node.sequence
            if g['input_data_type']=='cdn':
                if len(seq)%3!=0:
                    print('Sequence length is not multiple of 3.', node.name)
                state_matrix = numpy.zeros([g['num_input_site'], g['num_input_state']], dtype=numpy.float64)
                for s in numpy.arange(int(len(seq)/3)):
                    codon = seq[(s*3):((s+1)*3)]
                    if not '-' in codon:
                        codon_index = g['codon_orders'].index(codon)
                        state_matrix[s,codon_index] = 1
            elif g['input_data_type']=='nuc':
                state_matrix = numpy.zeros([g['num_input_site'], g['num_input_state']], dtype=numpy.float64)
                for s in numpy.arange(len(seq)):
                    if seq[s]!='-':
                        nuc_index = g['input_state'].index(seq[s])
                        state_matrix[s, nuc_index] = 1
            state_tensor[node.numerical_label,:,:] = state_matrix
        else:
            state_matrix = state_table.loc[(state_table['Node']==node.name),:].iloc[:,3:]
            if state_matrix.shape[0]==0:
                print('Node name not found in .state file:', node.name)
            else:
                state_tensor[node.numerical_label,:,:] = state_matrix
    if g['ml_anc']:
        idxmax = numpy.argmax(state_tensor, axis=2)
        state_tensor = numpy.zeros(state_tensor.shape, dtype=numpy.bool)
        for b in numpy.arange(state_tensor.shape[0]):
            for s in numpy.arange(state_tensor.shape[1]):
                state_tensor[b,s,idxmax[b,s]] = 1
    return(state_tensor)


