import numpy as np

import itertools
import pkgutil
import re
import sys

from csubst import sequence
from csubst import parser_phylobayes
from csubst import parser_iqtree
from csubst import recoding
from csubst import tree
from csubst import ete

def _read_package_text(file):
    txt = pkgutil.get_data('csubst', file)
    if txt is None:
        raise FileNotFoundError('Unable to read package data file: {}'.format(file))
    txt = txt.decode('utf-8')
    return txt

def generate_intermediate_files(g, force_notree_run=False):
    if (g['infile_type'] == 'phylobayes'):
        raise ValueError("PhyloBayes is not supported.")
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
        g = recoding.initialize_nonsyn_groups(g)
        if g['nonsyn_recode'] != 'none':
            txt = 'Applying nonsynonymous recoding scheme: {}'
            print(txt.format(g['nonsyn_recode']))
        return g
    if (g['omegaC_method']!='submodel'):
        g = recoding.initialize_nonsyn_groups(g)
        if g['nonsyn_recode'] != 'none':
            txt = 'Applying nonsynonymous recoding scheme: {}'
            print(txt.format(g['nonsyn_recode']))
        return g
    base_model = re.sub(r'\+G.*', '', g['substitution_model'])
    base_model = re.sub(r'\+R.*', '', base_model)
    txt = 'Instantaneous substitution rate matrix will be generated using the base model: {}'
    print(txt.format(base_model))
    txt = 'Transition matrix will be generated using the model in the ancestral state reconstruction: {}'
    print(txt.format(g['substitution_model']))
    if (g['substitution_model'].startswith('ECMK07')):
        matrix_file = 'substitution_matrix/ECMunrest.dat'
        g['exchangeability_matrix'] = read_exchangeability_matrix(matrix_file, g['codon_orders'])
        g['exchangeability_eq_freq'] = read_exchangeability_eq_freq(file=matrix_file, g=g)
        g['empirical_eq_freq'] = get_equilibrium_frequency(g, mode='cdn')
        g['instantaneous_codon_rate_matrix'] = exchangeability2Q(g['exchangeability_matrix'], g['empirical_eq_freq'])
    elif (g['substitution_model'].startswith('ECMrest')):
        matrix_file = 'substitution_matrix/ECMrest.dat'
        g['exchangeability_matrix'] = read_exchangeability_matrix(matrix_file, g['codon_orders'])
        g['exchangeability_eq_freq'] = read_exchangeability_eq_freq(file=matrix_file, g=g)
        g['empirical_eq_freq'] = get_equilibrium_frequency(g, mode='cdn')
        g['instantaneous_codon_rate_matrix'] = exchangeability2Q(g['exchangeability_matrix'], g['empirical_eq_freq'])
    elif (g['substitution_model'].startswith('GY')):
        txt = 'Estimated omega is not available in IQ-TREE\'s log file. Run IQ-TREE with a GY-based model.'
        if g['omega'] is None:
            raise AssertionError(txt)
        txt = 'Estimated kappa is not available in IQ-TREE\'s log file. Run IQ-TREE with a GY-based model.'
        if g['kappa'] is None:
            raise AssertionError(txt)
        g['instantaneous_codon_rate_matrix'] = get_mechanistic_instantaneous_rate_matrix(g=g)
    elif (g['substitution_model'].startswith('MG')):
        txt = 'Estimated omega is not available in IQ-TREE\'s log file. Run IQ-TREE with a GY-based model.'
        if g['omega'] is None:
            raise AssertionError(txt)
        g['instantaneous_codon_rate_matrix'] = get_mechanistic_instantaneous_rate_matrix(g=g)
    else:
        txt = 'Unsupported substitution model for --omegaC_method submodel: {}'
        raise ValueError(txt.format(g['substitution_model']))
    g = recoding.initialize_nonsyn_groups(g)
    if g['nonsyn_recode'] != 'none':
        txt = 'Applying nonsynonymous recoding scheme: {}'
        print(txt.format(g['nonsyn_recode']))
    g['instantaneous_aa_rate_matrix'] = cdn2pep_matrix(inst_cdn=g['instantaneous_codon_rate_matrix'], g=g)
    g['instantaneous_nsy_rate_matrix'] = cdn2nsy_matrix(inst_cdn=g['instantaneous_codon_rate_matrix'], g=g)
    g['rate_syn_tensor'] = get_rate_tensor(inst=g['instantaneous_codon_rate_matrix'], mode='syn', g=g)
    g['rate_aa_tensor'] = get_rate_tensor(inst=g['instantaneous_aa_rate_matrix'], mode='asis', g=g)
    g['rate_nsy_tensor'] = get_rate_tensor(inst=g['instantaneous_nsy_rate_matrix'], mode='asis', g=g)
    sum_tensor_aa = g['rate_aa_tensor'].sum()
    sum_tensor_nsy = g['rate_nsy_tensor'].sum()
    sum_tensor_syn = g['rate_syn_tensor'].sum()
    sum_matrix_aa = g['instantaneous_aa_rate_matrix'][g['instantaneous_aa_rate_matrix']>0].sum()
    sum_matrix_nsy = g['instantaneous_nsy_rate_matrix'][g['instantaneous_nsy_rate_matrix']>0].sum()
    sum_matrix_cdn = g['instantaneous_codon_rate_matrix'][g['instantaneous_codon_rate_matrix']>0].sum()
    if abs(sum_tensor_aa - sum_matrix_aa) >= g['float_tol']:
        raise AssertionError('Sum of rates did not match.')
    if (g['nonsyn_recode'] != 'none') and (abs(sum_tensor_nsy - sum_matrix_nsy) >= g['float_tol']):
        raise AssertionError('Sum of recoded nonsynonymous rates did not match.')
    txt = 'Sum of rates did not match. Check if --codon_table ({}) matches to that used in the ancestral state reconstruction ({}).'
    txt = txt.format(g['codon_table'], g['reconstruction_codon_table'])
    if abs(sum_matrix_cdn - sum_tensor_syn - sum_tensor_aa) >= g['float_tol']:
        raise AssertionError(txt)
    np.savetxt('csubst_instantaneous_rate_matrix.tsv', g['instantaneous_codon_rate_matrix'], delimiter='\t')
    q_ij_x_pi_i = g['instantaneous_codon_rate_matrix'][0,1]*g['equilibrium_frequency'][0]
    q_ji_x_pi_j = g['instantaneous_codon_rate_matrix'][1,0]*g['equilibrium_frequency'][1]
    if abs(q_ij_x_pi_i - q_ji_x_pi_j) >= g['float_tol']:
        raise AssertionError('Instantaneous codon rate matrix (Q) is not time-reversible.')
    return g

def get_mechanistic_instantaneous_rate_matrix(g):
    num_codon = len(g['codon_orders'])
    inst = np.ones(shape=(num_codon,num_codon))
    transition_pairs = {'AG', 'GA', 'CT', 'TC'}
    for i1,c1 in enumerate(g['codon_orders']):
        for i2,c2 in enumerate(g['codon_orders']):
            num_diff_codon_position = sum([ cp1!=cp2 for cp1,cp2 in zip(c1,c2) ])
            if (num_diff_codon_position!=1):
                inst[i1,i2] = 0 # prohibit double substitutions
    is_single_substitution = (inst != 0)
    if g['omega'] is not None:
        omega = float(g['omega'])
        if omega < 0:
            raise ValueError('omega should be >= 0.')
        if omega != 1.0:
            inst *= omega # nonsynonymous substitutions are scaled by omega
        # Keep synonymous substitutions at 1.0 (where codon transitions are allowed).
        for aa in g['amino_acid_orders']:
            ind_cdn = np.array(g['synonymous_indices'][aa])
            for i1,i2 in itertools.permutations(ind_cdn, 2):
                if is_single_substitution[i1,i2]:
                    inst[i1,i2] = 1.0
    if g['kappa'] is not None:
        for i1,c1 in enumerate(g['codon_orders']):
            for i2,c2 in enumerate(g['codon_orders']):
                num_diff_codon_position = sum([ cp1!=cp2 for cp1,cp2 in zip(c1,c2) ])
                if (num_diff_codon_position==1):
                    diff_nucs = [ cp1+cp2 for cp1,cp2 in zip(c1,c2) if cp1!=cp2 ][0]
                    if diff_nucs in transition_pairs:
                        inst[i1,i2] *= g['kappa'] # multiply kappa to transition substitutions
    inst = inst.dot(np.diag(g['equilibrium_frequency'])).astype(g['float_type']) # pi_j * q_ij
    inst = scale_instantaneous_rate_matrix(inst, g['equilibrium_frequency'])
    inst = fill_instantaneous_rate_matrix_diagonal(inst)
    return inst

def fill_instantaneous_rate_matrix_diagonal(inst):
    for i in np.arange(inst.shape[0]):
        inst[i,i] = 0
        inst[i,i] = -inst[i,:].sum()
    return inst

def scale_instantaneous_rate_matrix(inst, eq):
    # scaling to satisfy Sum_i Sum_j!=i pi_i*q_ij, equals 1.
    # See Kosiol et al. 2007. https://academic.oup.com/mbe/article/24/7/1464/986344
    diagonal = np.diag(inst)
    if not np.all(np.isclose(diagonal, 0)):
        raise AssertionError('Diagonal elements should still be zeros.')
    q_ijxpi_i = np.einsum('ad,a->ad', inst, eq)
    scaling_factor = q_ijxpi_i.sum()
    if scaling_factor <= 0:
        raise AssertionError('Instantaneous rate matrix scaling factor must be positive.')
    inst /= scaling_factor
    return inst

def get_rate_tensor(inst, mode, g):
    if mode=='asis':
        inst2 = np.copy(inst)
        np.fill_diagonal(inst2, 0)
        rate_tensor = np.expand_dims(inst2, axis=0)
    elif mode=='syn':
        num_syngroup = len(g['amino_acid_orders'])
        num_state = g['max_synonymous_size']
        axis = (num_syngroup,num_state,num_state)
        rate_tensor = np.zeros(axis, dtype=inst.dtype)
        for s,aa in enumerate(g['amino_acid_orders']):
            ind_cdn = np.array(g['synonymous_indices'][aa])
            ind_tensor = np.arange(len(ind_cdn))
            for it1,it2 in itertools.permutations(ind_tensor, 2):
                rate_tensor[s,it1,it2] = inst[ind_cdn[it1],ind_cdn[it2]]
    else:
        raise ValueError('Unsupported rate-tensor mode: {}'.format(mode))
    rate_tensor = rate_tensor.astype(g['float_type'])
    return rate_tensor

def _cdn2group_matrix(inst_cdn, group_orders, group_indices):
    group_orders = [str(order) for order in group_orders]
    num_group = len(group_orders)
    inst_group = np.zeros([num_group, num_group], dtype=inst_cdn.dtype)
    for i, group_from in enumerate(group_orders):
        idx_from = np.asarray(group_indices[group_from], dtype=np.int64).reshape(-1)
        if idx_from.shape[0] == 0:
            continue
        for j, group_to in enumerate(group_orders):
            if group_from == group_to:
                continue
            idx_to = np.asarray(group_indices[group_to], dtype=np.int64).reshape(-1)
            if idx_to.shape[0] == 0:
                continue
            inst_group[i, j] = inst_cdn[np.ix_(idx_from, idx_to)].sum()
    inst_group = fill_instantaneous_rate_matrix_diagonal(inst_group)
    return inst_group


def cdn2pep_matrix(inst_cdn, g):
    inst_pep = _cdn2group_matrix(
        inst_cdn=inst_cdn,
        group_orders=g['amino_acid_orders'],
        group_indices=g['synonymous_indices'],
    )
    # Following lines were commented out because this shouldn't be readjusted.
    # Branch lengths are subst/codon.
    # If readjusted, we have to provide subst/aa to calculate expected nonsynonymous convergence.
    #eq_pep = get_equilibrium_frequency(g, mode='pep')
    #inst_pep = scale_instantaneous_rate_matrix(inst_pep, eq_pep)
    return inst_pep


def cdn2nsy_matrix(inst_cdn, g):
    return _cdn2group_matrix(
        inst_cdn=inst_cdn,
        group_orders=g['nonsyn_state_orders'],
        group_indices=g['nonsynonymous_indices'],
    )

def exchangeability2Q(ex, eq):
    inst = ex.dot(np.diag(eq)).astype(np.float64) # pi_j * s_ij. float32 is not enough for Pyvolve
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
        eq_pep = np.zeros([num_pep_state,], dtype=eq.dtype)
        for i,aa in enumerate(g['amino_acid_orders']):
            aa_indices = g['synonymous_indices'][aa]
            eq_pep[i] = eq[aa_indices].sum()
        txt = 'Equilibrium amino acid frequency should sum to 1.'
        if abs(eq_pep.sum()-1) >= g['float_tol']:
            raise AssertionError(txt)
        return eq_pep
    else:
        raise ValueError('Unsupported equilibrium-frequency mode: {}'.format(mode))

def _can_use_selective_state_loading(g):
    if g.get('exhaustive_until', None) != 1:
        return False, None
    if g.get('foreground', None) is None:
        return False, None
    if not bool(g.get('cb', False)):
        return False, '--cb is disabled.'
    blocking_outputs = [name for name in ['b', 's', 'bs', 'cs', 'cbs'] if bool(g.get(name, False))]
    if len(blocking_outputs) > 0:
        txt = 'full-tree outputs are enabled ({})'
        return False, txt.format(', '.join(blocking_outputs))
    if bool(g.get('plot_state_aa', False)) or bool(g.get('plot_state_codon', False)):
        return False, 'state-tree plotting is enabled.'
    if int(g.get('fg_clade_permutation', 0)) > 0:
        return False, '--fg_clade_permutation is enabled.'
    return True, None


def _normalize_branch_ids(branch_ids):
    if branch_ids is None:
        return np.array([], dtype=np.int64)
    values = np.asarray(branch_ids, dtype=object)
    flat_values = np.atleast_1d(values).reshape(-1)
    if flat_values.size == 0:
        return np.array([], dtype=np.int64)
    normalized = []
    for value in flat_values.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('target_ids should be integer-like.')
        if isinstance(value, (int, np.integer)):
            normalized.append(int(value))
            continue
        if isinstance(value, (float, np.floating)):
            if (not np.isfinite(value)) or (not float(value).is_integer()):
                raise ValueError('target_ids should be integer-like.')
            normalized.append(int(value))
            continue
        value_txt = str(value).strip()
        if (value_txt == '') or (not bool(re.fullmatch(r'[+-]?[0-9]+(?:\.0+)?', value_txt))):
            raise ValueError('target_ids should be integer-like.')
        normalized.append(int(float(value_txt)))
    return np.array(normalized, dtype=np.int64)


def _get_required_state_branch_ids(g):
    root_nn = ete.get_prop(ete.get_tree_root(g['tree']), "numerical_label")
    required = set([int(root_nn)])
    node_by_id = dict()
    for node in g['tree'].traverse():
        node_by_id[int(ete.get_prop(node, "numerical_label"))] = node
    target_ids = set()
    for trait_name in g['target_ids'].keys():
        values = _normalize_branch_ids(g['target_ids'][trait_name])
        target_ids.update([int(v) for v in values.tolist()])
    for branch_id in target_ids:
        required.add(branch_id)
        node = node_by_id.get(branch_id, None)
        if (node is None) or ete.is_root(node):
            continue
        required.add(int(ete.get_prop(node.up, "numerical_label")))
    required_ids = np.array(sorted(required), dtype=np.int64)
    return required_ids


def resolve_state_loading(g):
    g['state_loaded_branch_ids'] = None
    g['is_state_selective_loading'] = False
    use_selective, reason = _can_use_selective_state_loading(g)
    if not use_selective:
        if (g.get('exhaustive_until', None) == 1) and (g.get('foreground', None) is not None) and (reason is not None):
            print('Selective state loading disabled: {}'.format(reason), flush=True)
        return g
    required_ids = _get_required_state_branch_ids(g)
    if required_ids.shape[0] == 0:
        print('Selective state loading disabled: no required branches were found.', flush=True)
        return g
    g['state_loaded_branch_ids'] = required_ids
    g['is_state_selective_loading'] = True
    txt = 'Selective state loading enabled: loading {:,} nodes out of {:,} total nodes.'
    print(txt.format(required_ids.shape[0], g['num_node']), flush=True)
    return g


def prep_state(g):
    state_nuc = None
    state_cdn = None
    state_pep = None
    state_nsy = None
    loaded_branch_ids = g.get('state_loaded_branch_ids', None)
    if (g['infile_type'] == 'phylobayes'):
        from csubst import parser_phylobayes
        if g['input_data_type'] == 'nuc': # obsoleted
            state_nuc = parser_phylobayes.get_state_tensor(g, selected_branch_ids=loaded_branch_ids)
            state_cdn = calc_omega_state(state_nuc=state_nuc, g=g)
            state_pep = sequence.cdn2pep_state(state_cdn=state_cdn, g=g, selected_branch_ids=loaded_branch_ids)
            state_nsy = sequence.cdn2nsy_state(state_cdn=state_cdn, g=g, selected_branch_ids=loaded_branch_ids)
        elif g['input_data_type'] == 'cdn':
            state_cdn = parser_phylobayes.get_state_tensor(g, selected_branch_ids=loaded_branch_ids)
            state_pep = sequence.cdn2pep_state(state_cdn=state_cdn, g=g, selected_branch_ids=loaded_branch_ids)
            state_nsy = sequence.cdn2nsy_state(state_cdn=state_cdn, g=g, selected_branch_ids=loaded_branch_ids)
    elif (g['infile_type'] == 'iqtree'):
        from csubst import parser_iqtree
        if g['input_data_type'] == 'nuc': # obsoleted
            state_nuc = parser_iqtree.get_state_tensor(g, selected_branch_ids=loaded_branch_ids)
            state_cdn = calc_omega_state(state_nuc=state_nuc, g=g)
            state_pep = sequence.cdn2pep_state(state_cdn=state_cdn, g=g, selected_branch_ids=loaded_branch_ids)
            state_nsy = sequence.cdn2nsy_state(state_cdn=state_cdn, g=g, selected_branch_ids=loaded_branch_ids)
        elif g['input_data_type'] == 'cdn':
            state_cdn = parser_iqtree.get_state_tensor(g, selected_branch_ids=loaded_branch_ids)
            state_pep = sequence.cdn2pep_state(state_cdn=state_cdn, g=g, selected_branch_ids=loaded_branch_ids)
            state_nsy = sequence.cdn2nsy_state(state_cdn=state_cdn, g=g, selected_branch_ids=loaded_branch_ids)
    g['state_nuc'] = state_nuc
    g['state_cdn'] = state_cdn
    g['state_pep'] = state_pep
    g['state_nsy'] = state_nsy
    return g

def read_exchangeability_matrix(file, codon_orders):
    txt = _read_package_text(file=file)
    txt = txt.replace('\r', '').split('\n')
    txt_mat = txt[0:60]
    txt_mat = ''.join(txt_mat).split()
    arr = np.array([ float(s) for s in txt_mat ], dtype=float)
    if arr.shape[0] != 1830:
        raise AssertionError('This is not a codon substitution matrix.')
    num_state = 61
    mat_exchangeability = np.zeros(shape=(num_state,num_state))
    ind = np.tril_indices_from(mat_exchangeability, k=-1)
    mat_exchangeability[ind] = arr
    mat_exchangeability += mat_exchangeability.T
    ex_codon_order = get_exchangeability_codon_order()
    codon_order_index = get_codon_order_index(order_from=codon_orders, order_to=ex_codon_order)
    mat_exchangeability = mat_exchangeability[codon_order_index,:][:,codon_order_index] # Index matches to g['codon_orders']
    return mat_exchangeability

def get_codon_order_index(order_from, order_to):
    if len(order_from) != len(order_to):
        txt = 'Codon order lengths should match. Emprical codon substitution models are currently supported only for the Standard codon table.'
        raise AssertionError(txt)
    source_codon_list = [str(fr) for fr in order_from]
    if len(source_codon_list) != len(set(source_codon_list)):
        duplicate_codons = sorted(list(set([c for c in source_codon_list if source_codon_list.count(c) > 1])))
        duplicate_txt = ','.join(duplicate_codons[:10])
        if len(duplicate_codons) > 10:
            duplicate_txt += ',...'
        raise ValueError('Duplicate codon found in source order: {}'.format(duplicate_txt))
    index_by_codon = dict()
    for i,to in enumerate(order_to):
        if to in index_by_codon:
            raise ValueError('Duplicate codon found in target order: {}'.format(to))
        index_by_codon[to] = i
    missing = [fr for fr in order_from if fr not in index_by_codon]
    if len(missing) > 0:
        missing_txt = ','.join(missing[:10])
        if len(missing) > 10:
            missing_txt += ',...'
        raise ValueError('Codon(s) from source order not found in target order: {}'.format(missing_txt))
    out = np.array([index_by_codon[fr] for fr in order_from], dtype=int)
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
    exchangeability_codon_order = np.array(exchangeability_codon_order)
    return exchangeability_codon_order

def read_exchangeability_eq_freq(file, g):
    txt = _read_package_text(file=file)
    txt = txt.replace('\r', '').split('\n')
    if len(txt) <= 61:
        txt = 'Exchangeability file format is invalid: expected equilibrium frequencies on line 62.'
        raise AssertionError(txt)
    freqs = txt[61].split()
    freqs = np.array([ float(s) for s in freqs ], dtype=float)
    if freqs.shape[0] != 61:
        raise AssertionError('Number of equilibrium frequencies ({}) should be 61.'.format(freqs.shape[0]))
    ex_codon_order = get_exchangeability_codon_order()
    codon_order_index = get_codon_order_index(order_from=g['codon_orders'], order_to=ex_codon_order)
    freqs = freqs[codon_order_index]
    return freqs

def annotate_tree(g, ignore_tree_inconsistency=False):
    g['node_label_tree_file'] = g['iqtree_treefile']
    with open(g['node_label_tree_file']) as f:
        node_label_tree_newick = f.read()
    g['node_label_tree'] = ete.PhyloNode(node_label_tree_newick, format=1)
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
            raise ValueError('Input tree and iqtree\'s treefile did not have identical leaves.')
    g['tree'] = tree.add_numerical_node_labels(g['tree'])
    total_root_tree_len = sum([(n.dist or 0.0) for n in g['rooted_tree'].traverse()])
    total_iqtree_len = sum([(n.dist or 0.0) for n in g['node_label_tree'].traverse()])
    print('Total branch length of --rooted_tree_file: {:,.4f}'.format(total_root_tree_len))
    print('Total branch length of --iqtree_treefile: {:,.4f}'.format(total_iqtree_len))
    print('')
    return g
