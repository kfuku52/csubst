import numpy
import itertools
import pkg_resources
from csubst import sequence

def read_input(g):
    if (g['infile_type'] == 'phylobayes'):
        from csubst import parser_phylobayes
        g = parser_phylobayes.get_input_information(g)
    elif (g['infile_type'] == 'iqtree'):
        from csubst import parser_iqtree
        g = parser_iqtree.get_input_information(g)
    if ('omega_method' in g.keys()):
        if (g['omega_method']=='mat'):
            matrix_file = 'substitution_matrix/ECMunrest.dat'
            g['exchangeability_matrix'] = read_exchangeability_matrix(file=matrix_file, g=g)
            g['exchangeability_eq_freq'] = read_exchangeability_eq_freq(file=matrix_file, g=g)
            g['instantaneous_codon_rate_matrix'] = get_instantaneous_rate_matrix(g=g)
            g['instantaneous_aa_rate_matrix'] = cdn2pep_matrix(inst_cdn=g['instantaneous_codon_rate_matrix'], g=g)
            g['rate_syn_tensor'] = get_rate_tensor(inst=g['instantaneous_codon_rate_matrix'], mode='syn', g=g)
            g['rate_aa_tensor'] = get_rate_tensor(inst=g['instantaneous_aa_rate_matrix'], mode='asis', g=g)
            sum_tensor_aa = g['rate_aa_tensor'].sum()
            sum_tensor_syn = g['rate_syn_tensor'].sum()
            sum_matrix_aa = g['instantaneous_aa_rate_matrix'][g['instantaneous_aa_rate_matrix']>0].sum()
            sum_matrix_cdn = g['instantaneous_codon_rate_matrix'][g['instantaneous_codon_rate_matrix']>0].sum()
            assert (sum_tensor_aa-sum_matrix_aa)<10**-9, 'Sum of rates did not match.'
            assert (sum_matrix_cdn-sum_tensor_syn-sum_tensor_aa)<10**-9, 'Sum of rates did not match.'
    return g

def get_rate_tensor(inst, mode, g):
    if mode=='asis':
        inst2 = numpy.copy(inst)
        numpy.fill_diagonal(inst2, 0)
        rate_tensor = numpy.expand_dims(inst2, axis=0)
    elif mode=='syn':
        num_syngroup = len(g['amino_acid_orders'])
        num_state = g['max_synonymous_size']
        axis = (num_syngroup,num_state,num_state)
        rate_tensor = numpy.zeros(axis, dtype=inst.dtype)
        for s,aa in enumerate(g['amino_acid_orders']):
            ind_cdn = numpy.array(g['synonymous_indices'][aa])
            ind_tensor = numpy.arange(len(ind_cdn))
            for it1,it2 in itertools.permutations(ind_tensor, 2):
                rate_tensor[s,it1,it2] = inst[ind_cdn[it1],ind_cdn[it2]]
    rate_tensor = rate_tensor.astype(numpy.float64)
    return rate_tensor

def cdn2pep_matrix(inst_cdn, g):
    num_pep_state = len(g['amino_acid_orders'])
    axis = [num_pep_state, num_pep_state]
    inst_pep = numpy.zeros(axis, dtype=inst_cdn.dtype)
    for i,aa1 in enumerate(g['amino_acid_orders']):
        for j,aa2 in enumerate(g['amino_acid_orders']):
            if aa1==aa2:
                continue
            val = 0
            aa1_indices = g['synonymous_indices'][aa1]
            aa2_indices = g['synonymous_indices'][aa2]
            for aa1_ind,aa2_ind in itertools.product(aa1_indices, aa2_indices):
                val += inst_cdn[aa1_ind,aa2_ind]
            inst_pep[i,j] = val
    inst_pep = fill_instantaneous_rate_matrix_diagonal(inst_pep)
    # Commented out because this shouldn't be readjusted.
    # Branch lengths are subst/codon.
    # If readjusted, we have to provide subst/aa to calculate expected nonsynonymous convergence.
    #eq_pep = get_equilibrium_frequency(g, mode='pep')
    #inst_pep = scale_instantaneous_rate_matrix(inst_pep, eq_pep)
    return inst_pep

def get_instantaneous_rate_matrix(g):
    ex = g['exchangeability_matrix']
    eq = get_equilibrium_frequency(g, mode='cdn')
    inst = ex.dot(numpy.diag(eq)).astype(numpy.float64)
    inst = fill_instantaneous_rate_matrix_diagonal(inst)
    inst = scale_instantaneous_rate_matrix(inst, eq)
    return inst

def get_equilibrium_frequency(g, mode):
    if 'equilibrium_frequency' in g.keys():
        print('Applying estimated codon frequencies to obtain the instantaneous rate matrix.')
        eq = g['equilibrium_frequency']
    else:
        print('Applying empirical codon frequencies to obtain the instantaneous rate matrix.')
        eq = g['exchangeability_eq_freq']
    if mode=='cdn':
        return eq
    elif mode=='pep':
        num_pep_state = len(g['amino_acid_orders'])
        eq_pep = numpy.zeros([num_pep_state,], dtype=eq.dtype)
        for i,aa in enumerate(g['amino_acid_orders']):
            aa_indices = g['synonymous_indices'][aa]
            eq_pep[i] = eq[aa_indices].sum()
        txt = 'Equilibrium amino acid frequency should sum to 1.'
        assert abs(eq_pep.sum()-1)<10**-9, txt
        return eq_pep

def fill_instantaneous_rate_matrix_diagonal(inst):
    for i in numpy.arange(inst.shape[0]):
        inst[i,i] = -inst[i,:].sum()
    return inst

def scale_instantaneous_rate_matrix(inst, eq):
    # scaling to satisfy Sum_i Sum_j!=i pi_i*q_ij, equals 1.
    q_ijxpi_i = numpy.einsum('ad,a->ad', inst, eq)
    scaling_factor = q_ijxpi_i.sum()-numpy.diag(q_ijxpi_i).sum()
    inst /= scaling_factor
    return inst

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

def read_exchangeability_matrix(file, g):
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
    ex_codon_order = get_exchangeability_codon_order()
    codon_order_index = get_codon_order_index(order_from=g['codon_orders'], order_to=ex_codon_order)
    mat_exchangeability = mat_exchangeability[codon_order_index,:][:,codon_order_index] # Index matches to g['codon_orders']
    return mat_exchangeability

def get_codon_order_index(order_from, order_to):
    assert len(order_from)==len(order_to), 'Codon order lengths should match. Emprical codon substitution models are currently supported only for the Standard codon table.'
    out = list()
    for fr in order_from:
        for i,to in enumerate(order_to):
            if fr==to:
                out.append(i)
                break
    out = numpy.array(out)
    return out

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

def read_exchangeability_eq_freq(file, g):
    txt = pkg_resources.resource_string(__name__, file)
    txt = str(txt).replace('b\"','').replace('\\r','').split('\\n')
    freqs = txt[61].split(' ')
    freqs = numpy.array([ float(s) for s in freqs if s!='' ], dtype=float)
    assert freqs.shape[0]==61, 'Number of equilibrium frequencies ({}) should be 61.'.format(freqs.shape[0])
    ex_codon_order = get_exchangeability_codon_order()
    codon_order_index = get_codon_order_index(order_from=g['codon_orders'], order_to=ex_codon_order)
    freqs = freqs[codon_order_index]
    return freqs
