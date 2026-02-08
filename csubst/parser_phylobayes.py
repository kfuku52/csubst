import numpy
import pandas

import os
import itertools

from csubst import tree
from csubst import ete

def get_node_phylobayes_out(node, files):
    if ete.is_leaf(node):
        pp_file = [ file for file in files if file.find(node.name+"_"+node.name+".ancstatepostprob") > -1 ]
    else:
        pp_file = [ file for file in files if file.find(".ancstatepostprob") > -1 ]
        pp_file = [ file for file in pp_file if file.find("_sample_"+node.name+"_") > -1 ]
    return(pp_file)

def get_pp_nuc(phylobayes_dir, pp_file):
    pp_nuc = pandas.read_csv(phylobayes_dir+pp_file, sep="\t", index_col=False, header=0)
    pp_nuc = pp_nuc.iloc[:,2:].values
    return(pp_nuc)

def get_pp_cdn(pp_nuc, codon_table): # obsolete
    codon_columns = pp_nuc.columns.values.reshape((-1,1,1)) + pp_nuc.columns.values.reshape((1,-1,1)) + pp_nuc.columns.values.reshape((1,1,-1))
    codon_columns = codon_columns.ravel()
    aa_columns = []
    for codon_column in codon_columns:
        aa_columns.append([ codon[0] if codon[1]==codon_column else '*' for codon in codon_table ][0])
    num_codon = int(len(pp_nuc.index)/3)
    columns = pandas.MultiIndex.from_arrays([codon_columns, aa_columns], names=["codon", "aa"])
    pp_cdn = pandas.DataFrame(0, index=range(0, num_codon), columns=columns)
    for i in pp_cdn.index:
        pp_codon = pp_nuc.loc[i*3:i*3, : ].values.reshape((-1,1,1)) * pp_nuc.loc[i*3+1:i*3+1, : ].values.reshape((1,-1,1)) * pp_nuc.loc[i*3+2:i*3+2, : ].values.reshape((1,1,-1))
        pp_cdn.loc[i,:] = pp_codon.ravel()
    is_stop = pp_cdn.columns.get_level_values(1) == "*"
    pp_cdn = pp_cdn.loc[:,~is_stop]
    return(pp_cdn)

def get_pp_N(pp_cdn):
    aa_index = pp_cdn.columns.get_level_values(1)
    pp_N = pp_cdn.groupby(by=aa_index, axis=1).sum()
    return(pp_N)

def get_input_information(g):
    files = os.listdir(g['phylobayes_dir'])
    sample_labels = [file for file in files if "_sample.labels" in file][0]
    g['tree'] = ete.PhyloNode(g['phylobayes_dir'] + sample_labels, format=1)
    g['tree'] = tree.add_numerical_node_labels(g['tree'])
    g['num_node'] = len(list(g['tree'].traverse()))
    state_files = [ f for f in files if f.endswith('.ancstatepostprob') ]
    state_table = pandas.read_csv(g['phylobayes_dir'] + state_files[0], sep="\t", index_col=False, header=0)
    g['num_input_site'] = state_table.shape[0]
    g['num_input_state'] = state_table.shape[1] - 2
    g['input_state'] = state_table.columns[2:].tolist()
    g['num_ancstatepostprob_file'] = len(state_files)
    if g['num_input_state']==4:
        g['input_data_type'] = 'nuc'
    elif g['num_input_state']==20:
        g['input_data_type'] = 'pep'
    elif g['num_input_state'] > 20:
        g['input_data_type'] = 'cdn'
    if (g['input_data_type']=='nuc'):
        g['state_columns'] = list(itertools.product(numpy.arange(len(g['input_state'])), repeat=3))
        codon_orders = list(itertools.product(g['input_state'], repeat=3))
        codon_orders = [ c[0]+c[1]+c[2] for c in codon_orders]
        g['codon_orders'] = codon_orders
        amino_acids = sorted(list(set([ c[0] for c in g['codon_table'] if c[0]!='*' ])))
        g['amino_acid_orders'] = amino_acids
        matrix_groups = dict()
        for aa in list(set(amino_acids)):
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

def get_state_tensor(g, selected_branch_ids=None):
    num_node = len(list(g['tree'].traverse()))
    selected_set = None
    if selected_branch_ids is not None:
        selected_set = set([int(v) for v in numpy.asarray(selected_branch_ids).tolist()])
        root_nn = int(ete.get_prop(ete.get_tree_root(g['tree']), "numerical_label"))
        selected_set.add(root_nn)
    state_files = [ f for f in os.listdir(g['phylobayes_dir']) if f.endswith('.ancstatepostprob') ]
    print('The number of character states =', g['num_input_state'])
    axis = [num_node, g['num_input_site'], g['num_input_state']]
    if selected_set is None:
        state_tensor = numpy.zeros(tuple(axis), dtype=g['float_type'])
    else:
        mmap_tensor = os.path.join(os.getcwd(), 'tmp.csubst.state_tensor.mmap')
        if os.path.exists(mmap_tensor):
            os.unlink(mmap_tensor)
        txt = 'Generating memory map: dtype={}, axis={}, path={}'
        print(txt.format(g['float_type'], axis, mmap_tensor), flush=True)
        state_tensor = numpy.memmap(mmap_tensor, dtype=g['float_type'], shape=tuple(axis), mode='w+')
    for node in g['tree'].traverse():
        nl = int(ete.get_prop(node, "numerical_label"))
        if (selected_set is not None) and (nl not in selected_set):
            continue
        pp_file = get_node_phylobayes_out(node=node, files=state_files)
        if len(pp_file) == 1:
            pp_file = pp_file[0]
            state_tensor[nl,:,:] = get_pp_nuc(g['phylobayes_dir'], pp_file)
        elif (len(pp_file) > 1)&(not isinstance(pp_file, str)):
            raise Exception('Multiple .ancstatepostprob files for the node.',
                  'node.name =', node.name, 'ete.get_prop(node, "numerical_label") =', ete.get_prop(node, "numerical_label"),
                  'files =', pp_file)
        elif len(pp_file) == 0:
            print('Could not find .ancstatepostprob file for the node.',
                  'node.name =', node.name, 'ete.get_prop(node, "numerical_label") =', ete.get_prop(node, "numerical_label"),
                  'is_root =', ete.is_root(node), 'is_leaf =', ete.is_leaf(node))
    if (g['ml_anc']):
        if selected_set is None:
            idxmax = numpy.argmax(state_tensor, axis=2)
            state_tensor = numpy.zeros(state_tensor.shape, dtype=bool)
            for b in numpy.arange(state_tensor.shape[0]):
                for s in numpy.arange(state_tensor.shape[1]):
                    state_tensor[b,s,idxmax[b,s]] = 1
        else:
            mmap_tensor = os.path.join(os.getcwd(), 'tmp.csubst.state_tensor_ml.mmap')
            if os.path.exists(mmap_tensor):
                os.unlink(mmap_tensor)
            state_tensor2 = numpy.memmap(mmap_tensor, dtype=bool, shape=state_tensor.shape, mode='w+')
            for b in sorted(selected_set):
                idxmax = numpy.argmax(state_tensor[b,:,:], axis=1)
                for s in numpy.arange(state_tensor.shape[1]):
                    if state_tensor[b,s,:].sum()!=0:
                        state_tensor2[b,s,idxmax[s]] = True
            state_tensor = state_tensor2
    return(state_tensor)
