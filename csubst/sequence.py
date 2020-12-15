import numpy

import re

def calc_omega_state(state_nuc, g): # TODO, implement, exclude stop codon freq
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

def write_alignment(state, orders, outfile, mode, g):
    if mode=='codon':
        missing_state = '---'
    else:
        missing_state = '-'
    aln_out = ''
    for node in g['tree'].traverse():
        if node.is_root():
            continue
        nlabel = node.numerical_label
        aln_tmp = '>'+node.name+'|'+str(nlabel)+'\n'
        for i in numpy.arange(state.shape[1]):
            if state[nlabel,i,:].max()==0:
                aln_tmp += missing_state
            else:
                index = state[nlabel,i,:].argmax()
                aln_tmp += orders[index]
        aln_out += aln_tmp+'\n'
    with open(outfile, 'w') as f:
        print('Writing alignment:', outfile, flush=True)
        f.write(aln_out)

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
    state_index0 = [ numpy.where(input_state==s)[0] for s in states ]
    if state_index0[0].shape[0]==0:
        return None
    state_index = [ int(si) for si in state_index0 ]
    return state_index

def read_fasta(path):
    with open(path, mode='r') as f:
        txt = f.read()
    seqs = [ t for t in txt.split('>') if not t=='' ]
    seqs = [ s.split('\n', 1) for s in seqs ]
    seqs = [ s.replace('\n','') for seq in seqs for s in seq ]
    seq_dict = dict()
    for i in range(int(len(seqs)/2)):
        seq_dict[seqs[i*2]] = seqs[i*2+1]
    return seq_dict