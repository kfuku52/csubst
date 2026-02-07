import numpy

from csubst import ete

def calc_omega_state(state_nuc, g): # implement exclude stop codon freq
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

def translate_state(nlabel, mode, g):
    if mode=='codon':
        missing_state = '---'
        state = g['state_cdn']
        orders = g['codon_orders']
    elif mode=='aa':
        missing_state = '-'
        state = g['state_pep']
        orders = g['amino_acid_orders']
    seq_out = ''
    for i in numpy.arange(state.shape[1]):
        if state[nlabel,i,:].max()<g['float_tol']:
            seq_out += missing_state
        else:
            index = state[nlabel,i,:].argmax()
            seq_out += orders[index]
    return seq_out

def write_alignment(outfile, mode, g, leaf_only=False):
    aln_out = ''
    if leaf_only:
        nodes = ete.iter_leaves(g['tree'])
    else:
        nodes = g['tree'].traverse()
    for node in nodes:
        if ete.is_root(node):
            continue
        nlabel = node.numerical_label
        aln_tmp = '>'+node.name+'|'+str(nlabel)+'\n'
        aln_tmp += translate_state(nlabel, mode, g)
        aln_out += aln_tmp+'\n'
    with open(outfile, 'w') as f:
        print('Writing sequence alignment:', outfile, flush=True)
        f.write(aln_out)

def get_state_index(state, input_state, ambiguous_table):
    if ('-' in state)|(state=='NNN')|(state=='N'):
        return []
    states = [state,]
    state_set = set(list(state))
    key_set = set(ambiguous_table.keys())
    if (len(state_set.intersection(key_set))>0):
        for amb in [a for a in ambiguous_table.keys() if a in state_set]:
            vals = ambiguous_table[amb]
            states = [ s.replace(amb, val) for s in states for val in vals ]
    state_index0 = [ numpy.where(input_state==s)[0] for s in states ]
    state_index0 = [ s for s in state_index0 if s.shape[0]!=0 ]
    if len(state_index0)==0:
        return []
    state_index = [ int(idx) for si in state_index0 for idx in si ]
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

def calc_identity(seq1, seq2):
    assert len(seq1)==len(seq2), 'Sequence lengths should be identical.'
    num_same_site = 0
    for s1,s2 in zip(seq1,seq2):
        if s1==s2:
            num_same_site += 1
    identity_value = num_same_site / len(seq1)
    return identity_value
