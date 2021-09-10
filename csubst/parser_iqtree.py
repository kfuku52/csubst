import numpy
import pandas

import os
import re
import subprocess
import sys
from collections import OrderedDict
from distutils.version import LooseVersion

from csubst import genetic_code
from csubst import sequence
from csubst import tree

def check_iqtree_dependency(g):
    test_iqtree = subprocess.run([g['iqtree_exe'], '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert (test_iqtree.returncode==0), "iqtree PATH cannot be found: "+g['iqtree_exe']
    version_iqtree = test_iqtree.stdout.decode('utf8')
    version_iqtree = version_iqtree.replace('\n','')
    version_iqtree = re.sub('.*version ', '', version_iqtree)
    version_iqtree = re.sub(' for.*', '', version_iqtree)
    is_satisfied_version = LooseVersion(version_iqtree) >= LooseVersion('2.0.0')
    assert is_satisfied_version, 'IQ-TREE version ({}) should be 2.0.0 or greater.'.format(version_iqtree)
    print("IQ-TREE's version: {}, PATH: {}".format(version_iqtree, g['iqtree_exe']), flush=True)
    return None

def check_intermediate_files(g):
    all_exist = True
    extensions = ['iqtree','log','rate','state','treefile']
    for ext in extensions:
        if (g['iqtree_'+ext]=='infer'):
            g['path_iqtree_'+ext] = g['alignment_file']+'.'+ext
        else:
            g['path_iqtree_'+ext] = g['iqtree_'+ext]
        if not os.path.exists(g['path_iqtree_'+ext]):
            print('Intermediate file is missing: {}'.format(g['path_iqtree_'+ext]))
            all_exist = False
    return g,all_exist

def run_iqtree_ancestral(g):
    file_tree = 'tmp.csubst.nwk'
    tree.write_tree(g['rooted_tree'], outfile=file_tree, add_numerical_label=False)
    command = [g['iqtree_exe'], '-s', g['alignment_file'], '-te', file_tree,
               '-m', g['iqtree_model'], '--seqtype', 'CODON'+str(g['genetic_code']),
               '--threads-max', str(g['threads']), '-T', 'AUTO', '--ancestral', '--rate', '--redo']
    run_iqtree = subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)
    assert (run_iqtree.returncode==0), "IQ-TREE did not finish safely: {}".format(run_iqtree.stdout.decode('utf8'))
    if os.path.exists(g['alignment_file']+'.ckp.gz'):
        os.remove(g['alignment_file']+'.ckp.gz')
    os.remove(file_tree)
    return None

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
        matrix_groups = OrderedDict()
        for aa in g['amino_acid_orders']:
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
    rate_sites = pandas.read_csv(g['path_iqtree_rate'], sep='\t', header=0, comment='#')
    rate_sites = rate_sites.loc[:,'C_Rate'].values
    if rate_sites.shape[0]==0:
        rate_sites = numpy.ones(g['num_input_site'])
    return rate_sites

def read_iqtree(g, eq=True):
    with open(g['path_iqtree_iqtree']) as f:
        lines = f.readlines()
    for line in lines:
        model = re.match(r'Model of substitution: (.+)', line)
        if model is not None:
            g['substitution_model'] = model.group(1)
    if eq:
        with open(g['path_iqtree_iqtree']) as f:
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
    with open(g['path_iqtree_log']) as f:
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
    g['tree'].link_to_alignment(alignment=g['alignment_file'], alg_format='fasta')
    num_codon_alignment = int(len(g['tree'].get_leaves()[0].sequence)/3)
    err_txt = 'The number of codon sites did not match between the alignment and ancestral states. ' \
              'Delete intermediate files and rerun.'
    assert num_codon_alignment==g['num_input_site'], err_txt
    num_node = len(list(g['tree'].traverse()))
    state_table = pandas.read_csv(g['path_iqtree_state'], sep="\t", index_col=False, header=0, comment='#')
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
                    codon_index = sequence.get_state_index(codon, g['codon_orders'], genetic_code.ambiguous_table)
                    for ci in codon_index:
                        state_matrix[s,ci] = 1/len(codon_index)
            elif g['input_data_type']=='nuc':
                for s in numpy.arange(len(seq)):
                    nuc_index = sequence.get_state_index(seq[s], g['input_state'], genetic_code.ambiguous_table)
                    for ni in nuc_index:
                        state_matrix[s, ni] = 1/len(nuc_index)
            state_tensor[node.numerical_label,:,:] = state_matrix
        else: # Internal nodes
            state_matrix = state_table.loc[(state_table['Node']==node.name),:].iloc[:,3:]
            is_missing = (state_table.loc[:,'State']=='???') | (state_table.loc[:,'State']=='?')
            state_matrix.loc[is_missing,:] = 0
            if state_matrix.shape[0]==0:
                print('Node name not found in .state file:', node.name)
            else:
                state_tensor[node.numerical_label,:,:] = state_matrix
    state_tensor = numpy.nan_to_num(state_tensor)
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
    g = read_state(g)
    g = read_iqtree(g)
    g = read_log(g)
    g['iqtree_rate_values'] = read_rate(g)
    return g