import numpy as np
import pandas as pd

import itertools
import os
import pkgutil
import re
import sys

from csubst import genetic_code
from csubst import sequence
from csubst import parser_phylobayes
from csubst import parser_iqtree
from csubst import recoding
from csubst import tree
from csubst import ete


def _initialize_and_report_nonsyn_recode(g):
    g = recoding.initialize_nonsyn_groups(g)
    write_pca = bool(g.get("plot_nonsyn_recode_pca", False))
    if g["nonsyn_recode"] != "no":
        txt = "Applying nonsynonymous recoding scheme: {}"
        print(txt.format(g["nonsyn_recode"]))
        recoding.write_nonsyn_recoding_table(g, output_path="csubst_nonsyn_recoding.tsv")
    if write_pca:
        recoding.write_nonsyn_recoding_pca_plot(g, output_path="csubst_nonsyn_recoding_pca.png")
    return g


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
        g = _initialize_and_report_nonsyn_recode(g)
        return g
    if (g['omegaC_method']!='submodel'):
        g = _initialize_and_report_nonsyn_recode(g)
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
    g = _initialize_and_report_nonsyn_recode(g)
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
    if (g['nonsyn_recode'] != 'no') and (abs(sum_tensor_nsy - sum_matrix_nsy) >= g['float_tol']):
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


def get_site_index_alignment(g, expected_num_site=None):
    site_index_alignment = g.get('site_index_alignment', None)
    if site_index_alignment is None:
        if expected_num_site is None:
            if g.get('state_cdn', None) is not None:
                expected_num_site = int(g['state_cdn'].shape[1])
            elif g.get('state_nuc', None) is not None:
                expected_num_site = int(g['state_nuc'].shape[1] // 3)
            elif g.get('num_input_site', None) is not None:
                expected_num_site = int(g['num_input_site'])
            else:
                expected_num_site = 0
        site_index_alignment = np.arange(int(expected_num_site), dtype=np.int64)
        g['site_index_alignment'] = site_index_alignment
    else:
        site_index_alignment = np.asarray(site_index_alignment, dtype=np.int64).reshape(-1)
        g['site_index_alignment'] = site_index_alignment
    if expected_num_site is not None:
        expected_num_site = int(expected_num_site)
        if site_index_alignment.shape[0] != expected_num_site:
            txt = 'site_index_alignment length ({}) did not match expected number of sites ({}).'
            raise ValueError(txt.format(site_index_alignment.shape[0], expected_num_site))
    return site_index_alignment


def map_internal_site_indices(site_indices, g, missing_value=-1, allow_invalid=False):
    arr = np.asarray(site_indices, dtype=np.int64)
    site_index_alignment = g.get('site_index_alignment', None)
    if site_index_alignment is None:
        return arr.copy()
    site_index_alignment = np.asarray(site_index_alignment, dtype=np.int64).reshape(-1)
    if site_index_alignment.size == 0:
        return arr.copy()
    if site_index_alignment.dtype != np.int64:
        site_index_alignment = site_index_alignment.astype(np.int64, copy=False)
    out = np.full(arr.shape, int(missing_value), dtype=np.int64)
    valid = (arr >= 0) & (arr < site_index_alignment.shape[0])
    out[valid] = site_index_alignment[arr[valid]]
    if (not allow_invalid) and (~valid).any():
        invalid_values = np.unique(arr[~valid]).tolist()
        invalid_txt = ','.join([str(int(v)) for v in invalid_values[:10]])
        if len(invalid_values) > 10:
            invalid_txt += ',...'
        txt = 'Internal site index out of range for mapping: {}'
        raise ValueError(txt.format(invalid_txt))
    return out


def write_site_index_map(g, output_path='csubst_site_index_map.tsv'):
    site_index_alignment = get_site_index_alignment(g=g)
    if site_index_alignment.shape[0] == 0:
        return None
    num_alignment_site = int(g.get('num_input_site', 0))
    if num_alignment_site <= 0:
        num_alignment_site = int(site_index_alignment.max()) + 1
    if site_index_alignment.max() >= num_alignment_site:
        txt = 'site_index_alignment included alignment index {} but num_input_site is {}.'
        raise ValueError(txt.format(int(site_index_alignment.max()), num_alignment_site))
    internal_site = np.full(shape=(num_alignment_site,), fill_value=-1, dtype=np.int64)
    internal_site[site_index_alignment] = np.arange(site_index_alignment.shape[0], dtype=np.int64)
    out = pd.DataFrame({
        'codon_site_alignment': np.arange(num_alignment_site, dtype=np.int64),
        'codon_site_alignment_1based': np.arange(1, num_alignment_site + 1, dtype=np.int64),
        'site': internal_site,
        'site_1based': np.where(internal_site >= 0, internal_site + 1, 0).astype(np.int64),
        'is_retained': np.where(internal_site >= 0, 'yes', 'no'),
    })
    out.to_csv(output_path, sep='\t', index=False)
    return os.path.abspath(output_path)


def get_site_retained_mask(g, num_alignment_site=None):
    site_index_alignment = get_site_index_alignment(g=g)
    if site_index_alignment.shape[0] == 0:
        if num_alignment_site is None:
            num_alignment_site = int(g.get('num_input_site', 0))
        return np.zeros(shape=(max(int(num_alignment_site), 0),), dtype=bool)
    if num_alignment_site is None:
        num_alignment_site = int(g.get('num_input_site', 0))
        if num_alignment_site <= 0:
            num_alignment_site = int(site_index_alignment.max()) + 1
    num_alignment_site = int(num_alignment_site)
    if num_alignment_site <= 0:
        return np.zeros(shape=(0,), dtype=bool)
    if int(site_index_alignment.max()) >= num_alignment_site:
        txt = 'site_index_alignment included alignment index {} but num_alignment_site is {}.'
        raise ValueError(txt.format(int(site_index_alignment.max()), num_alignment_site))
    retained = np.zeros(shape=(num_alignment_site,), dtype=bool)
    retained[site_index_alignment] = True
    return retained


def expand_site_axis_table_to_alignment(
    df,
    g,
    site_col='site',
    group_cols=None,
    site_is_one_based=False,
    retention_col='is_site_retained',
):
    if site_col not in df.columns:
        return df
    out = df.copy(deep=True)
    if group_cols is None:
        group_cols = list()
    for col in group_cols:
        if col not in out.columns:
            raise ValueError('Group column was not found in table: {}'.format(col))
    site_values = pd.to_numeric(out.loc[:, site_col], errors='coerce')
    out = out.loc[site_values.notna(), :].copy()
    out.loc[:, site_col] = site_values.loc[site_values.notna()].astype(np.int64).to_numpy(copy=False)
    key_cols = group_cols + [site_col]
    if out.duplicated(subset=key_cols).any():
        txt = 'Cannot expand table because duplicated key rows were found for columns: {}'
        raise ValueError(txt.format(','.join(key_cols)))
    num_alignment_site = int(g.get('num_input_site', 0))
    if num_alignment_site <= 0:
        site_index_alignment = get_site_index_alignment(g=g)
        if site_index_alignment.shape[0] == 0:
            return out
        num_alignment_site = int(site_index_alignment.max()) + 1
    if site_is_one_based:
        site_start = 1
    else:
        site_start = 0
    site_template = pd.DataFrame({
        site_col: np.arange(site_start, site_start + int(num_alignment_site), dtype=np.int64),
    })
    if len(group_cols) == 0:
        full = site_template
    else:
        groups = out.loc[:, group_cols].drop_duplicates().reset_index(drop=True)
        if groups.shape[0] == 0:
            return out
        groups = groups.copy(deep=True)
        site_template = site_template.copy(deep=True)
        groups.loc[:, '__tmpkey__'] = 1
        site_template.loc[:, '__tmpkey__'] = 1
        full = pd.merge(groups, site_template, on='__tmpkey__', how='inner')
        full = full.drop(columns=['__tmpkey__'])
    expanded = pd.merge(full, out, on=key_cols, how='left', sort=False)
    expanded = expanded.sort_values(by=key_cols).reset_index(drop=True)
    if retention_col is not None:
        retained_mask = get_site_retained_mask(g=g, num_alignment_site=num_alignment_site)
        site_values_full = expanded.loc[:, site_col].to_numpy(dtype=np.int64, copy=False)
        if site_is_one_based:
            site_values_zero = site_values_full - 1
        else:
            site_values_zero = site_values_full
        is_valid_site = (site_values_zero >= 0) & (site_values_zero < retained_mask.shape[0])
        retained_values = np.full(shape=(expanded.shape[0],), fill_value='no', dtype=object)
        retained_values[is_valid_site] = np.where(
            retained_mask[site_values_zero[is_valid_site]],
            'yes',
            'no',
        )
        expanded.loc[:, retention_col] = retained_values
    return expanded


def _get_tip_invariant_site_mask(g, site_index_alignment):
    leaf_seqs = list()
    for leaf in ete.iter_leaves(g['tree']):
        seq = ete.get_prop(leaf, 'sequence', '').upper()
        if seq == '':
            continue
        if (len(seq) % 3) != 0:
            raise AssertionError('Sequence length is not multiple of 3. Node name = {}'.format(leaf.name))
        num_codon_site = len(seq) // 3
        if num_codon_site != int(g['num_input_site']):
            msg = 'Codon site count did not match alignment size for leaf "{}". '
            msg += 'Expected {}, observed {}.'
            raise AssertionError(msg.format(leaf.name, g['num_input_site'], num_codon_site))
        leaf_seqs.append(seq)
    num_site = int(site_index_alignment.shape[0])
    is_tip_invariant = np.zeros(shape=(num_site,), dtype=bool)
    if len(leaf_seqs) == 0:
        return is_tip_invariant
    codon_orders = g['codon_orders']
    codon_state_lookup = sequence.build_state_index_lookup(codon_orders)
    for i, aln_site in enumerate(site_index_alignment.tolist()):
        observed_codon = -1
        num_nonmissing = 0
        is_invariant = True
        start = int(aln_site) * 3
        end = start + 3
        for seq in leaf_seqs:
            codon = seq[start:end]
            codon_id = codon_state_lookup.get(codon, None)
            if codon_id is not None:
                codon_index = [int(codon_id)]
            else:
                codon_index = sequence.get_state_index(
                    codon,
                    codon_orders,
                    genetic_code.ambiguous_table,
                    state_lookup=codon_state_lookup,
                )
            if len(codon_index) == 0:
                continue
            if len(codon_index) != 1:
                is_invariant = False
                break
            codon_id = int(codon_index[0])
            num_nonmissing += 1
            if observed_codon == -1:
                observed_codon = codon_id
            elif observed_codon != codon_id:
                is_invariant = False
                break
        # Drop also when only one unambiguous tip codon is observed:
        # such sites have no within-tip codon variation and are now treated as tip-invariant.
        is_tip_invariant[i] = bool(is_invariant and (num_nonmissing >= 1) and (observed_codon >= 0))
    return is_tip_invariant


def _get_drop_site_branch_pairs(g):
    selected_branch_ids = g.get('state_loaded_branch_ids', None)
    selected_branch_set = None
    if selected_branch_ids is not None:
        selected_branch_set = set(int(v) for v in _normalize_branch_ids(selected_branch_ids).tolist())
    branch_pairs = list()
    for node in g['tree'].traverse():
        if ete.is_root(node):
            continue
        child = int(ete.get_prop(node, "numerical_label"))
        if (selected_branch_set is not None) and (child not in selected_branch_set):
            continue
        parent = int(ete.get_prop(node.up, "numerical_label"))
        branch_pairs.append((child, parent))
    return branch_pairs


def _get_zero_substitution_mass_site_mask(g):
    state_cdn = g.get('state_cdn', None)
    state_nsy = g.get('state_nsy', None)
    if (state_cdn is None) or (state_nsy is None):
        raise ValueError('state_cdn and state_nsy are required for zero_sub_mass site filtering.')
    if state_cdn.shape[1] != state_nsy.shape[1]:
        txt = 'state_cdn and state_nsy have inconsistent site axes: {} vs {}.'
        raise ValueError(txt.format(state_cdn.shape[1], state_nsy.shape[1]))
    num_site = int(state_cdn.shape[1])
    if num_site == 0:
        return np.zeros(shape=(0,), dtype=bool)
    tol = float(g.get('float_tol', 0))
    is_nonzero_mass = np.zeros(shape=(num_site,), dtype=bool)
    syn_group_indices = [
        np.asarray(g['synonymous_indices'][aa], dtype=np.int64).reshape(-1)
        for aa in g['amino_acid_orders']
    ]
    syn_group_indices = [ind for ind in syn_group_indices if ind.shape[0] > 1]
    branch_pairs = _get_drop_site_branch_pairs(g)
    if len(branch_pairs) == 0:
        return np.zeros(shape=(num_site,), dtype=bool)
    for child, parent in branch_pairs:
        parent_nsy = state_nsy[parent, :, :]
        child_nsy = state_nsy[child, :, :]
        n_mass = (parent_nsy.sum(axis=1) * child_nsy.sum(axis=1)) - np.einsum('ij,ij->i', parent_nsy, child_nsy)
        is_nonzero_mass |= (n_mass > tol)
        parent_cdn = state_cdn[parent, :, :]
        child_cdn = state_cdn[child, :, :]
        s_mass = np.zeros(shape=(num_site,), dtype=state_cdn.dtype)
        for ind in syn_group_indices:
            parent_syn = parent_cdn[:, ind]
            child_syn = child_cdn[:, ind]
            s_mass += (
                (parent_syn.sum(axis=1) * child_syn.sum(axis=1))
                - np.einsum('ij,ij->i', parent_syn, child_syn)
            )
        is_nonzero_mass |= (s_mass > tol)
        if is_nonzero_mass.all():
            break
    return ~is_nonzero_mass


def _slice_site_axis_in_state_tensor(state_tensor, keep_mask):
    if state_tensor is None:
        return None
    if state_tensor.ndim < 2:
        return state_tensor
    if state_tensor.shape[1] == keep_mask.shape[0]:
        return state_tensor[:, keep_mask, ...]
    if state_tensor.shape[1] == (keep_mask.shape[0] * 3):
        nuc_keep_mask = np.repeat(keep_mask, 3)
        return state_tensor[:, nuc_keep_mask, ...]
    txt = 'State tensor site axis ({}) did not match keep_mask length ({}).'
    raise ValueError(txt.format(state_tensor.shape[1], keep_mask.shape[0]))


def drop_invariant_tip_sites(g):
    state_cdn = g.get('state_cdn', None)
    if state_cdn is None:
        return g
    site_index_alignment = get_site_index_alignment(g=g, expected_num_site=state_cdn.shape[1])
    mode = str(g.get('drop_invariant_tip_sites_mode', 'tip_invariant')).strip().lower()
    if mode == 'tip_invariant':
        is_drop_site = _get_tip_invariant_site_mask(g=g, site_index_alignment=site_index_alignment)
        mode_label = 'tip-invariant'
    elif mode == 'zero_sub_mass':
        is_drop_site = _get_zero_substitution_mass_site_mask(g=g)
        mode_label = 'zero-sub-mass'
    else:
        raise ValueError('Unsupported --drop_invariant_tip_sites_mode: {}'.format(mode))
    num_drop = int(is_drop_site.sum())
    if num_drop == 0:
        print('No codon sites were dropped (mode={}).'.format(mode), flush=True)
        g['drop_invariant_tip_sites_mode_applied'] = mode
        g['num_dropped_sites'] = 0
        g['dropped_site_alignment'] = np.array([], dtype=np.int64)
        g['num_dropped_tip_invariant_sites'] = 0
        g['dropped_tip_invariant_site_alignment'] = np.array([], dtype=np.int64)
        return g
    keep_mask = ~is_drop_site
    if keep_mask.sum() == 0:
        txt = 'All codon sites were classified as drop candidates and would be dropped (mode={}).'
        raise ValueError(txt.format(mode))
    g['state_nuc'] = _slice_site_axis_in_state_tensor(g.get('state_nuc', None), keep_mask)
    g['state_cdn'] = _slice_site_axis_in_state_tensor(g.get('state_cdn', None), keep_mask)
    g['state_pep'] = _slice_site_axis_in_state_tensor(g.get('state_pep', None), keep_mask)
    g['state_nsy'] = _slice_site_axis_in_state_tensor(g.get('state_nsy', None), keep_mask)
    g['site_index_alignment'] = site_index_alignment[keep_mask]
    dropped_alignment_sites = site_index_alignment[is_drop_site]
    g['drop_invariant_tip_sites_mode_applied'] = mode
    g['num_dropped_sites'] = num_drop
    g['dropped_site_alignment'] = dropped_alignment_sites
    g['num_dropped_tip_invariant_sites'] = num_drop
    g['dropped_tip_invariant_site_alignment'] = dropped_alignment_sites
    if 'iqtree_rate_values' in g and (g['iqtree_rate_values'] is not None):
        rate_values = np.asarray(g['iqtree_rate_values'])
        num_input_site = int(g.get('num_input_site', rate_values.shape[0]))
        if rate_values.shape[0] == num_input_site:
            g['iqtree_rate_values'] = rate_values[g['site_index_alignment']]
        elif rate_values.shape[0] == keep_mask.shape[0]:
            g['iqtree_rate_values'] = rate_values[keep_mask]
        else:
            txt = 'iqtree_rate_values length ({}) did not match num_input_site ({}) or current site axis ({}).'
            raise ValueError(txt.format(rate_values.shape[0], num_input_site, keep_mask.shape[0]))
    map_path = write_site_index_map(g=g, output_path='csubst_site_index_map.tsv')
    txt = 'Dropped {:,} codon site(s) by {} criterion; retained {:,} / {:,} sites.'
    print(txt.format(num_drop, mode_label, int(keep_mask.sum()), int(keep_mask.shape[0])), flush=True)
    if map_path is not None:
        print('Writing site index map: {}'.format(map_path), flush=True)
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
    if state_cdn is not None:
        get_site_index_alignment(g=g, expected_num_site=state_cdn.shape[1])
    if bool(g.get('drop_invariant_tip_sites', False)):
        g = drop_invariant_tip_sites(g)
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
