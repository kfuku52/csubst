import os
import re
import ete3
import pandas
import itertools
from csubst.tree import *
from csubst.genetic_code import ambiguous_table

def get_input_information(g):
    files = os.listdir(g['infile_dir'])
    g['tree'] = ete3.PhyloNode(g['tre_file'], format=1)
    assert len(g['tree'].get_children())==2, 'The input tree may be unrooted: {}'.format(g['tre_file'])
    g['tree'] = add_numerical_node_labels(g['tree'])
    internal_node_name = True
    for node in g['tree'].traverse():
        if not node.is_root():
            if node.name=='':
                internal_node_name = False
    if not internal_node_name:
        tree_files = [ f for f in files if f.endswith('.treefile') ]
        assert (len(tree_files)!=0), 'No IQ-TREE *.treefile file detected.'
        if (len(tree_files)>1):
            print('Multiple IQ-TREE *.treefile files detected: '+','.join(tree_files))
            assumed_treefile_name = g['aln_file']+'.treefile'
            assert assumed_treefile_name in tree_files, 'IQ-TREE *.treefile file was NOT detected: '+assumed_treefile_name
            print('IQ-TREE *.treefile file was detected: '+assumed_treefile_name)
            g['node_label_tree_file'] = assumed_treefile_name
        else:
            g['node_label_tree_file'] = g['infile_dir']+tree_files[0]
        f = open(g['node_label_tree_file'])
        tree_string = f.readline()
        g['node_label_tree'] = ete3.PhyloNode(tree_string, format=1)
        f.close()
        for node in g['node_label_tree'].traverse():
            node.name = re.sub('\[.*', '', node.name)
            node.name = re.sub('/.*', '', node.name)
        g['node_label_tree'] = transfer_root(tree_to=g['node_label_tree'], tree_from=g['tree'], verbose=False)
        g['tree'] = transfer_internal_node_names(tree_to=g['tree'], tree_from=g['node_label_tree'])
    g['num_node'] = len(list(g['tree'].traverse()))
    state_files = [ f for f in os.listdir(g['infile_dir']) if f.endswith('.state') ]
    assert (len(state_files)!=0), 'No IQ-TREE *.state file detected.'
    if (len(state_files)>1):
        print('Multiple IQ-TREE *.state files were detected:', state_files, flush=True)
        assumed_state_file_name = g['aln_file']+'.state'
        assert assumed_state_file_name in state_files, 'IQ-TREE *.state file was NOT found:'+assumed_state_file_name
        print('IQ-TREE *.state file was found:', assumed_state_file_name)
        g['state_file'] = assumed_state_file_name
    else:
        g['state_file'] = state_files[0]
    print('Reading the state file:', g['state_file'])
    state_table = pandas.read_csv(g['infile_dir'] + g['state_file'], sep="\t", index_col=False, header=0, comment='#')
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
    if (g['calc_omega'])&(g['input_data_type']=='nuc'):
        g['state_columns'] = list(itertools.product(numpy.arange(len(g['input_state'])), repeat=3))
        codon_orders = list(itertools.product(g['input_state'], repeat=3))
        codon_orders = [ c[0]+c[1]+c[2] for c in codon_orders]
        g['codon_orders'] = codon_orders
    if (g['calc_omega'])|(g['input_data_type']=='cdn'):
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
    return g

def get_state_index(state, input_state, ambiguous_table):
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
    state_index = [ input_state.index(s) for s in states ]
    return state_index

def mask_missing_sites(state_tensor, tree):
    for node in tree.traverse():
        if (node.is_root())|(node.is_leaf()):
            continue
        nl = node.numerical_label
        leaf_nls = numpy.array([ l.numerical_label for l in node.get_leaves() ], dtype=int)
        leaf_tensor = state_tensor[leaf_nls,:,:].sum(axis=0)
        is_leaf_nonzero = (leaf_tensor!=0)
        state_tensor[nl,:,:] = state_tensor[nl,:,:] * is_leaf_nonzero
    return state_tensor

def get_state_tensor(g):
    g['tree'].link_to_alignment(alignment=g['aln_file'], alg_format='fasta')
    num_node = len(list(g['tree'].traverse()))
    path_state_table = g['infile_dir'] + g['state_file']
    state_table = pandas.read_csv(path_state_table, sep="\t", index_col=False, header=0, comment='#')
    axis = [num_node, g['num_input_site'], g['num_input_state']]
    state_tensor = numpy.zeros(axis, dtype=numpy.float64)
    for node in g['tree'].traverse():
        if node.is_root():
            continue
        elif node.is_leaf():
            seq = node.sequence.upper()
            state_matrix = numpy.zeros([g['num_input_site'], g['num_input_state']], dtype=numpy.float64)
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


