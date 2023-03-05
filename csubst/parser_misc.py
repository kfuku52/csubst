import ete3
import numpy

import itertools
import pkg_resources
import re
import sys

from csubst import sequence
from csubst import parser_phylobayes
from csubst import parser_iqtree
from csubst import tree

def generate_intermediate_files(g, force_notree_run=False):
    if (g['infile_type'] == 'phylobayes'):
        raise Exception("PhyloBayes is not supported.")
    elif (g['infile_type'] == 'iqtree'):
        g,all_exist = parser_iqtree.check_intermediate_files(g)
        if (all_exist)&(not g['iqtree_redo']):
            print('IQ-TREE\'s intermediate files exist.')
            g = parser_iqtree.read_iqtree(g, eq=False)
            iqtree_model = g['substitution_model']
            g['substitution_model'] = None
            if (iqtree_model==g['iqtree_model']):
                txt = 'The model in the IQ-TREE\'s output ({}) matched --iqtree_model ({}). Skipping IQ-TREE.'
                print(txt.format(iqtree_model, g['iqtree_model']))
                return g
            else:
                txt = 'The model in the IQ-TREE\'s output ({}) did not match --iqtree_model ({}). Redoing IQ-TREE.'
                print(txt.format(iqtree_model, g['iqtree_model']))
        if (all_exist)&(g['iqtree_redo']):
            print('--iqtree_redo is set.')
        print('Starting IQ-TREE to estimate parameters and ancestral states.', flush=True)
        parser_iqtree.check_iqtree_dependency(g)
        parser_iqtree.run_iqtree_ancestral(g, force_notree_run=force_notree_run)
    return g

def read_input(g):
    if (g['infile_type'] == 'phylobayes'):
        g = parser_phylobayes.get_input_information(g)
    elif (g['infile_type'] == 'iqtree'):
        g = parser_iqtree.get_input_information(g)
    if ('omegaC_method' not in g.keys()):
        return g
    if (g['omegaC_method']!='submodel'):
        return g
    base_model = re.sub('\+G.*', '', g['substitution_model'])
    base_model = re.sub('\+R.*', '', base_model)
    txt = 'Instantaneous substitution rate matrix will be generated using the base model: {}'
    print(txt.format(base_model))
    txt = 'Transition matrix will be generated using the model in the ancestral state reconstruction: {}'
    print(txt.format(g['substitution_model']))
    if (g['substitution_model'].startswith('ECMK07')):
        matrix_file = 'substitution_matrix/ECMunrest.dat'
        g['exchangeability_matrix'] = read_exchangeability_matrix(matrix_file, g['codon_orders'])
        g['exchangeability_eq_freq'] = read_exchangeability_eq_freq(file=matrix_file, g=g)
        g['empirical_eq_freq'] = get_equilibrium_frequency(g, mode='cdn')
        g['instantaneous_codon_rate_matrix'] = exchangeability2Q(g['exchangeability_matrix'], g['empirical_eq_freq'], g['float_type'])
    elif (g['substitution_model'].startswith('ECMrest')):
        matrix_file = 'substitution_matrix/ECMrest.dat'
        g['exchangeability_matrix'] = read_exchangeability_matrix(matrix_file, g['codon_orders'])
        g['exchangeability_eq_freq'] = read_exchangeability_eq_freq(file=matrix_file, g=g)
        g['empirical_eq_freq'] = get_equilibrium_frequency(g, mode='cdn')
        g['instantaneous_codon_rate_matrix'] = exchangeability2Q(g['exchangeability_matrix'], g['empirical_eq_freq'], g['float_type'])
    elif (g['substitution_model'].startswith('GY')):
        txt = 'Estimated omega is not available in IQ-TREE\'s log file. Run IQ-TREE with a GY-based model.'
        assert (g['omega'] is not None), txt
        txt = 'Estimated kappa is not available in IQ-TREE\'s log file. Run IQ-TREE with a GY-based model.'
        assert (g['kappa'] is not None), txt
        g['instantaneous_codon_rate_matrix'] = get_mechanistic_instantaneous_rate_matrix(g=g)
    elif (g['substitution_model'].startswith('MG')):
        txt = 'Estimated omega is not available in IQ-TREE\'s log file. Run IQ-TREE with a GY-based model.'
        assert (g['omega'] is not None), txt
        g['instantaneous_codon_rate_matrix'] = get_mechanistic_instantaneous_rate_matrix(g=g)
    g['instantaneous_aa_rate_matrix'] = cdn2pep_matrix(inst_cdn=g['instantaneous_codon_rate_matrix'], g=g)
    g['rate_syn_tensor'] = get_rate_tensor(inst=g['instantaneous_codon_rate_matrix'], mode='syn', g=g)
    g['rate_aa_tensor'] = get_rate_tensor(inst=g['instantaneous_aa_rate_matrix'], mode='asis', g=g)
    sum_tensor_aa = g['rate_aa_tensor'].sum()
    sum_tensor_syn = g['rate_syn_tensor'].sum()
    sum_matrix_aa = g['instantaneous_aa_rate_matrix'][g['instantaneous_aa_rate_matrix']>0].sum()
    sum_matrix_cdn = g['instantaneous_codon_rate_matrix'][g['instantaneous_codon_rate_matrix']>0].sum()
    assert (sum_tensor_aa - sum_matrix_aa)<g['float_tol'], 'Sum of rates did not match.'
    txt = 'Sum of rates did not match. Check if --codon_table ({}) matches to that used in the ancestral state reconstruction ({}).'
    txt = txt.format(g['codon_table'], g['reconstruction_codon_table'])
    assert (sum_matrix_cdn - sum_tensor_syn - sum_tensor_aa)<g['float_tol'], txt
    numpy.savetxt('csubst_instantaneous_rate_matrix.tsv', g['instantaneous_codon_rate_matrix'], delimiter='\t')
    q_ij_x_pi_i = g['instantaneous_codon_rate_matrix'][0,1]*g['equilibrium_frequency'][0]
    q_ji_x_pi_j = g['instantaneous_codon_rate_matrix'][1,0]*g['equilibrium_frequency'][1]
    assert (q_ij_x_pi_i-q_ji_x_pi_j<g['float_tol']), 'Instantaneous codon rate matrix (Q) is not time-reversible.'
    return g

def get_mechanistic_instantaneous_rate_matrix(g):
    num_codon = len(g['codon_orders'])
    inst = numpy.ones(shape=(num_codon,num_codon))
    for i1,c1 in enumerate(g['codon_orders']):
        for i2,c2 in enumerate(g['codon_orders']):
            num_diff_codon_position = sum([ cp1!=cp2 for cp1,cp2 in zip(c1,c2) ])
            if (num_diff_codon_position!=1):
                inst[i1,i2] = 0 # prohibit double substitutions
    if g['omega'] is not None:
        inst *= g['omega'] # multiply omega for all elements
        for s,aa in enumerate(g['amino_acid_orders']):
            ind_cdn = numpy.array(g['synonymous_indices'][aa])
            for i1,i2 in itertools.permutations(ind_cdn, 2):
                inst[i1,i2] /= g['omega'] # restore rate of synonymous substitutions = 1, so nonsynonymous substitutions are left multiplied by omega
    if g['kappa'] is not None:
        for i1,c1 in enumerate(g['codon_orders']):
            for i2,c2 in enumerate(g['codon_orders']):
                num_diff_codon_position = sum([ cp1!=cp2 for cp1,cp2 in zip(c1,c2) ])
                if (num_diff_codon_position==1):
                    diff_nucs = [ cp1+cp2 for cp1,cp2 in zip(c1,c2) if cp1!=cp2 ][0]
                    if all([ (dn in ['A','G'])|(dn in ['C','T']) for dn in diff_nucs ]):
                        inst[i1,i2] *= g['kappa'] # multiply kappa to transition substitutions
    inst = inst.dot(numpy.diag(g['equilibrium_frequency'])).astype(g['float_type']) # pi_j * q_ij
    inst = scale_instantaneous_rate_matrix(inst, g['equilibrium_frequency'])
    inst = fill_instantaneous_rate_matrix_diagonal(inst)
    return inst

def fill_instantaneous_rate_matrix_diagonal(inst):
    for i in numpy.arange(inst.shape[0]):
        inst[i,i] = 0
        inst[i,i] = -inst[i,:].sum()
    return inst

def scale_instantaneous_rate_matrix(inst, eq):
    # scaling to satisfy Sum_i Sum_j!=i pi_i*q_ij, equals 1.
    # See Kosiol et al. 2007. https://academic.oup.com/mbe/article/24/7/1464/986344
    assert inst[0,0]==0, 'Diagonal elements should still be zeros.'
    q_ijxpi_i = numpy.einsum('ad,a->ad', inst, eq)
    scaling_factor = q_ijxpi_i.sum()
    inst /= scaling_factor
    return inst

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
    rate_tensor = rate_tensor.astype(g['float_type'])
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
    # Following lines were commented out because this shouldn't be readjusted.
    # Branch lengths are subst/codon.
    # If readjusted, we have to provide subst/aa to calculate expected nonsynonymous convergence.
    #eq_pep = get_equilibrium_frequency(g, mode='pep')
    #inst_pep = scale_instantaneous_rate_matrix(inst_pep, eq_pep)
    return inst_pep

def exchangeability2Q(ex, eq, float_type):
    inst = ex.dot(numpy.diag(eq)).astype(float_type) # pi_j * s_ij
    inst = scale_instantaneous_rate_matrix(inst, eq)
    inst = fill_instantaneous_rate_matrix_diagonal(inst)
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
        assert abs(eq_pep.sum()-1)<g['float_tol'], txt
        return eq_pep

def prep_state(g):
    state_nuc = None
    state_cdn = None
    state_pep = None
    if (g['infile_type'] == 'phylobayes'):
        from csubst import parser_phylobayes
        if g['input_data_type'] == 'nuc': # obsoleted
            state_nuc = parser_phylobayes.get_state_tensor(g)
            state_cdn = calc_omega_state(state_nuc=state_nuc, g=g)
            state_pep = sequence.cdn2pep_state(state_cdn=state_cdn, g=g)
        elif g['input_data_type'] == 'cdn':
            state_cdn = parser_phylobayes.get_state_tensor(g)
            state_pep = sequence.cdn2pep_state(state_cdn=state_cdn, g=g)
    elif (g['infile_type'] == 'iqtree'):
        from csubst import parser_iqtree
        if g['input_data_type'] == 'nuc': # obsoleted
            state_nuc = parser_iqtree.get_state_tensor(g)
            state_cdn = calc_omega_state(state_nuc=state_nuc, g=g)
            state_pep = sequence.cdn2pep_state(state_cdn=state_cdn, g=g)
        elif g['input_data_type'] == 'cdn':
            state_cdn = parser_iqtree.get_state_tensor(g)
            state_pep = sequence.cdn2pep_state(state_cdn=state_cdn, g=g)
    g['state_nuc'] = state_nuc
    g['state_cdn'] = state_cdn
    g['state_pep'] = state_pep
    return g

def read_exchangeability_matrix(file, codon_orders):
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
    codon_order_index = get_codon_order_index(order_from=codon_orders, order_to=ex_codon_order)
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

def annotate_tree(g, ignore_tree_inconsistency=False):
    g['node_label_tree_file'] = g['iqtree_treefile']
    f = open(g['node_label_tree_file'])
    tree_string = f.readline()
    g['node_label_tree'] = ete3.PhyloNode(tree_string, format=1)
    f.close()
    g['node_label_tree'] = tree.standardize_node_names(g['node_label_tree'])

    is_consistent_tree = tree.is_consistent_tree(tree1=g['node_label_tree'], tree2=g['rooted_tree'])
    if is_consistent_tree:
        g['tree'] = tree.transfer_root(tree_to=g['node_label_tree'], tree_from=g['rooted_tree'], verbose=False)
    else:
        sys.stderr.write('Input tree and iqtree\'s treefile did not have identical leaves.\n')
        if ignore_tree_inconsistency:
            sys.stderr.write('--rooted_tree will be used.\n')
            g['tree'] = g['rooted_tree']
        else:
            sys.stderr.write('Exiting.\n')
            sys.exit(1)
    g['tree'] = tree.add_numerical_node_labels(g['tree'])
    total_root_tree_len = sum([ n.dist for n in g['rooted_tree'].traverse() ])
    total_iqtree_len = sum([ n.dist for n in g['node_label_tree'].traverse() ])
    print('Total branch length of --rooted_tree_file: {:,.4f}'.format(total_root_tree_len))
    print('Total branch length of --iqtree_treefile: {:,.4f}'.format(total_iqtree_len))
    print('')
    return g