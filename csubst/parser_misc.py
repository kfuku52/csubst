import numpy
from csubst import sequence

def read_input(g):
    if (g['infile_type'] == 'phylobayes'):
        from csubst import parser_phylobayes
        g = parser_phylobayes.get_input_information(g)
    elif (g['infile_type'] == 'iqtree'):
        from csubst import parser_iqtree
        g = parser_iqtree.get_input_information(g)
        g = parser_iqtree.read_treefile(g)
        g = parser_iqtree.read_state(g)
        g = parser_iqtree.read_iqtree(g)
        g = parser_iqtree.read_log(g)
        if False: # TODO
            if (g['omega_method']=='mat'):
                from csubst.parser_misc import read_exchangeability_matrix
                file_path = '../substitution_matrix/ECMunrest.dat'
                smat = read_exchangeability_matrix(file=file_path)
                g['substitution_matrix'] = smat
    return g

def prep_state(g):
    state_nuc = None
    state_cdn = None
    state_pep = None
    if (g['infile_type'] == 'phylobayes'):
        from csubst import parser_phylobayes
        if g['input_data_type'] == 'nuc':
            state_nuc = parser_phylobayes.get_state_tensor(g)
            if (g['calc_omega']):
                state_cdn = calc_omega_state(state_nuc=state_nuc, g=g)
                state_pep = sequence.cdn2pep_state(state_cdn=state_cdn, g=g)
        elif g['input_data_type'] == 'cdn':
            state_cdn = parser_phylobayes.get_state_tensor(g)
            state_pep = sequence.cdn2pep_state(state_cdn=state_cdn, g=g)
    elif (g['infile_type'] == 'iqtree'):
        from csubst import parser_iqtree
        if g['input_data_type'] == 'nuc':
            state_nuc = parser_iqtree.get_state_tensor(g)
            if (g['calc_omega']):
                state_cdn = calc_omega_state(state_nuc=state_nuc, g=g)
                state_pep = sequence.cdn2pep_state(state_cdn=state_cdn, g=g)
        elif g['input_data_type'] == 'cdn':
            state_cdn = parser_iqtree.get_state_tensor(g)
            state_pep = sequence.cdn2pep_state(state_cdn=state_cdn, g=g)
    return g,state_nuc,state_cdn,state_pep


def read_exchangeability_matrix(file):
    import pkg_resources
    txt = pkg_resources.resource_string(__name__, file)
    txt = str(txt).replace('b\"','').replace('\\r','').split('\\n')
    txt_mat = txt[0:60]
    txt_mat = ''.join(txt_mat).split(' ')
    arr = numpy.array([ float(s) for s in txt_mat if s!='' ], dtype=float)
    assert (arr.shape[0]==1830), 'This is not a codon substitution matrix.'
    num_state = 61
    mat_exchangeability = numpy.zeros(shape=(num_state,num_state))
    ind = numpy.tril_indices_from(mat_exchangeability, k=-1)
    mat_exchangeability[ind] = arr
    mat_exchangeability += mat_exchangeability.T
    return mat_exchangeability

def get_exchangeability_codon_order():
    exchangeability_codon_order = [
        'TTT', 'TTC', 'TTA', 'TTG', 'TCT', 'TCC', 'TCA', 'TCG',
        'TAT', 'TAC', 'TGT', 'TGC', 'TGG', 'CTT', 'CTC', 'CTA',
        'CTG', 'CCT', 'CCC', 'CCA', 'CCG', 'CAT', 'CAC', 'CAA',
        'CAG', 'CGT', 'CGC', 'CGA', 'CGG', 'ATT', 'ATC', 'ATA',
        'ATG', 'ACT', 'ACC', 'ACA', 'ACG', 'AAT', 'AAC', 'AAA',
        'AAG', 'AGT', 'AGC', 'AGA', 'AGG', 'GTT', 'GTC', 'GTA',
        'GTG', 'GCT', 'GCC', 'GCA', 'GCG', 'GAT', 'GAC', 'GAA',
        'GAG', 'GGT', 'GGC', 'GGA', 'GGG',
    ]
    exchangeability_codon_order = numpy.array(exchangeability_codon_order)
    return exchangeability_codon_order

def get_exchangeability_eq_freq(file):
    import pkg_resources
    txt = pkg_resources.resource_string(__name__, file)
    txt = str(txt).replace('b\"','').replace('\\r','').split('\\n')
    freqs = txt[61].split(' ')
    freqs = numpy.array([ float(s) for s in freqs if s!='' ], dtype=float)
    assert freqs.shape[0]==61, 'Number of equilibrium frequencies ({}) should be 61.'.format(freqs.shape[0])
    return freqs