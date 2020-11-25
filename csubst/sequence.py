import numpy

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
            index = numpy.where(state[nlabel,i,:]==1)[0]
            if len(index)==1:
                aln_tmp += orders[index[0]]
            elif len(index)==0:
                aln_tmp += missing_state
        aln_out += aln_tmp+'\n'
    with open(outfile, 'w') as f:
        print('Writing alignment:', outfile, flush=True)
        f.write(aln_out)