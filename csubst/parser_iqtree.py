import numpy as np
import pandas as pd

import os
import re
import subprocess
import sys
from collections import OrderedDict
from itertools import zip_longest

from csubst import genetic_code
from csubst import sequence
from csubst import tree
from csubst import ete
try:
    from csubst import parser_iqtree_cy
except Exception:  # pragma: no cover - Cython extension is optional
    parser_iqtree_cy = None


def _parse_version_tuple(version_text):
    parts = re.findall(r'[0-9]+', str(version_text))
    if len(parts) == 0:
        return (0,)
    return tuple(int(p) for p in parts)


def _encode_unambiguous_codon(codon):
    if (not isinstance(codon, str)) or (len(codon) != 3):
        return -1
    nuc2code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    try:
        return (nuc2code[codon[0]] * 16) + (nuc2code[codon[1]] * 4) + nuc2code[codon[2]]
    except KeyError:
        return -1


def _build_unambiguous_codon_lookup(codon_orders):
    lookup = np.full(64, -1, dtype=np.int64)
    for idx, codon in enumerate(np.asarray(codon_orders, dtype=object).tolist()):
        code = _encode_unambiguous_codon(str(codon).upper())
        if code >= 0:
            lookup[code] = int(idx)
    return lookup


def _can_use_cython_leaf_codon_fill(state_matrix, codon_lookup):
    if parser_iqtree_cy is None:
        return False
    if not isinstance(state_matrix, np.ndarray):
        return False
    if not isinstance(codon_lookup, np.ndarray):
        return False
    if state_matrix.dtype != np.float64:
        return False
    if codon_lookup.dtype != np.int64:
        return False
    if codon_lookup.shape != (64,):
        return False
    if state_matrix.ndim != 2:
        return False
    return hasattr(parser_iqtree_cy, 'fill_leaf_state_matrix_codon_unambiguous')


def _fill_leaf_state_matrix_codon(seq, state_matrix, g, codon_lookup):
    num_codon_site = state_matrix.shape[0]
    unresolved_sites = np.arange(num_codon_site, dtype=np.int64)
    if _can_use_cython_leaf_codon_fill(state_matrix=state_matrix, codon_lookup=codon_lookup):
        unresolved_mask = parser_iqtree_cy.fill_leaf_state_matrix_codon_unambiguous(
            seq.encode('ascii'),
            state_matrix,
            codon_lookup,
        )
        unresolved_sites = np.where(unresolved_mask != 0)[0]
    for s in unresolved_sites:
        codon = seq[(s * 3):((s + 1) * 3)]
        codon_index = sequence.get_state_index(codon, g['codon_orders'], genetic_code.ambiguous_table)
        if len(codon_index) == 0:
            continue
        weight = 1 / len(codon_index)
        for ci in codon_index:
            state_matrix[s, ci] = weight


def _is_version_at_least(version_text, minimum_text):
    version_tuple = _parse_version_tuple(version_text)
    minimum_tuple = _parse_version_tuple(minimum_text)
    for v, m in zip_longest(version_tuple, minimum_tuple, fillvalue=0):
        if v > m:
            return True
        if v < m:
            return False
    return True


def _parse_iqtree_version_text(txt):
    # IQ-TREE 2 and 3 both expose a semantic version string in startup banners.
    patterns = [
        r'IQ-TREE(?: multicore)? version\s+([0-9]+(?:\.[0-9]+)*)',
        r'IQ-TREE\s+([0-9]+(?:\.[0-9]+)*)',
    ]
    for pattern in patterns:
        m = re.search(pattern, txt)
        if m is not None:
            version = m.group(1)
            major = int(version.split('.')[0])
            return version, major
    return None, None

def _read_text(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return f.read()

def detect_iqtree_output_version(g):
    g['iqtree_output_version'] = None
    g['iqtree_output_version_major'] = None
    for key in ['path_iqtree_iqtree', 'path_iqtree_log']:
        path = g.get(key, None)
        if (path is None) or (not os.path.exists(path)):
            continue
        txt = _read_text(path)
        version, major = _parse_iqtree_version_text(txt)
        if version is not None:
            g['iqtree_output_version'] = version
            g['iqtree_output_version_major'] = major
            break
    return g

def _parse_substitution_model(iqtree_txt):
    model = None
    for line in iqtree_txt.splitlines():
        m = re.match(r'\s*Model of substitution:\s*(.+?)\s*$', line)
        if m is not None:
            model = m.group(1)
            break
    return model

def _parse_equilibrium_frequency(iqtree_txt, codon_orders, parser_name, float_type):
    # IQ-TREE 2 and 3 share similar labels but differ slightly in spacing/number formatting.
    pattern_iqtree2 = r'pi\(([A-Z]+)\)\s*=\s*([0-9.]+)(?![eE][+-]?[0-9]+)'
    pattern_iqtree3 = r'pi\(\s*([A-Z]+)\s*\)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    # Parse the robust IQ-TREE3-compatible pattern first and avoid overriding
    # already parsed codons with legacy partial matches.
    if parser_name == 'iqtree3':
        patterns = [pattern_iqtree3, pattern_iqtree2]
    else:
        patterns = [pattern_iqtree3, pattern_iqtree2]
    eq_map = dict()
    for pattern in patterns:
        for codon,freq in re.findall(pattern, iqtree_txt):
            codon = codon.upper().replace('U', 'T')
            if codon in eq_map:
                continue
            try:
                eq_map[codon] = float(freq)
            except ValueError:
                continue
    values = list()
    missing_codons = list()
    for codon in codon_orders:
        freq = eq_map.get(codon, np.nan)
        if not np.isfinite(freq):
            missing_codons.append(codon)
        values.append(freq)
    if len(missing_codons)>0:
        txt = 'Failed to parse equilibrium frequencies from {} output. '
        txt += 'Missing codon(s): {}'
        missing_txt = ','.join(missing_codons[0:10])
        if len(missing_codons)>10:
            missing_txt += ',...'
        raise AssertionError(txt.format(parser_name, missing_txt))
    equilibrium_frequency = np.array(values, dtype=float_type)
    total = equilibrium_frequency.sum()
    if not (total > 0):
        raise AssertionError('Failed to parse equilibrium frequencies: sum should be positive.')
    equilibrium_frequency /= total
    return equilibrium_frequency

def _parse_float_from_log_line(line, label):
    pattern = r'\s*{}\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'.format(re.escape(label))
    m = re.match(pattern, line)
    if m is not None:
        return float(m.group(1))
    return None

def _estimate_empirical_codon_frequency_from_alignment(alignment_file, codon_orders, float_type):
    seq_dict = sequence.read_fasta(alignment_file)
    codon_orders = np.array(codon_orders)
    counts = np.zeros(shape=(codon_orders.shape[0],), dtype=float_type)
    for seq in seq_dict.values():
        seq = seq.upper().replace('U', 'T')
        if (len(seq) % 3) != 0:
            raise AssertionError('Sequence length is not multiple of 3 in alignment file.')
        for s in np.arange(0, len(seq), 3):
            codon = seq[s:s+3]
            codon_idx = sequence.get_state_index(codon, codon_orders, genetic_code.ambiguous_table)
            if len(codon_idx)==0:
                continue
            counts[codon_idx] += 1 / len(codon_idx)
    total = counts.sum()
    if not (total > 0):
        raise AssertionError('Failed to estimate codon frequencies from alignment file.')
    counts /= total
    return counts

def check_iqtree_dependency(g):
    try:
        test_iqtree = subprocess.run([g['iqtree_exe'], '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise AssertionError("iqtree PATH cannot be found: " + g['iqtree_exe']) from exc
    if test_iqtree.returncode != 0:
        raise AssertionError("iqtree PATH cannot be found: " + g['iqtree_exe'])
    txt = test_iqtree.stdout.decode('utf8', errors='replace') + '\n'
    txt += test_iqtree.stderr.decode('utf8', errors='replace')
    version_iqtree,major_iqtree = _parse_iqtree_version_text(txt)
    if version_iqtree is None:
        version_iqtree = txt.split('\n')[0].strip()
    g['iqtree_version'] = version_iqtree
    g['iqtree_version_major'] = major_iqtree
    is_satisfied_version = _is_version_at_least(version_iqtree, '2.0.0')
    if not is_satisfied_version:
        raise AssertionError('IQ-TREE version ({}) should be 2.0.0 or greater.'.format(version_iqtree))
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

def run_iqtree_ancestral(g, force_notree_run=False):
    file_tree = 'tmp.csubst.nwk'
    tree.write_tree(g['rooted_tree'], outfile=file_tree, add_numerical_label=False)
    try:
        is_consistent = tree.is_consistent_tree_and_aln(g=g)
        if is_consistent:
            command = [g['iqtree_exe'], '-s', g['alignment_file'], '-te', file_tree,
                       '-m', g['iqtree_model'], '--seqtype', 'CODON'+str(g['genetic_code']),
                       '--threads-max', str(g['threads']), '-T', 'AUTO', '--ancestral', '--rate', '--redo']
        else:
            sys.stderr.write('--rooted_tree and --alignment_file are not consistent.\n')
            if force_notree_run:
                sys.stderr.write('Running iqtree without tree input to estimate simulation parameters.\n')
                command = [g['iqtree_exe'], '-s', g['alignment_file'],
                           '-m', g['iqtree_model'], '--seqtype', 'CODON'+str(g['genetic_code']),
                           '--threads-max', str(g['threads']), '-T', 'AUTO', '--ancestral', '--rate', '--redo']
            else:
                raise ValueError('--rooted_tree and --alignment_file are not consistent.')
        run_iqtree = subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)
        if run_iqtree.returncode != 0:
            msg = 'IQ-TREE did not finish safely (exit code {}).'
            raise AssertionError(msg.format(run_iqtree.returncode))
        if os.path.exists(g['alignment_file']+'.ckp.gz'):
            os.remove(g['alignment_file']+'.ckp.gz')
    finally:
        if os.path.exists(file_tree):
            os.remove(file_tree)
    return None

def read_state(g):
    print('Reading the state file:', g['iqtree_state'])
    state_table = pd.read_csv(g['iqtree_state'], sep="\t", index_col=False, header=0, comment='#')
    g['num_input_site'] = state_table['Site'].unique().shape[0]
    g['num_input_state'] = state_table.shape[1] - 3
    g['input_state'] = state_table.columns[3:].str.replace('p_','').tolist()
    if g['num_input_state']==4:
        raise NotImplementedError(
            'Nucleotide ancestral-state input is obsolete and no longer supported. '
            'Use codon input instead.'
        )
    elif g['num_input_state']==20:
        raise NotImplementedError(
            'Protein ancestral-state input is obsolete and no longer supported. '
            'Use codon input instead.'
        )
    elif g['num_input_state'] > 20:
        g['input_data_type'] = 'cdn'
        g['codon_orders'] = state_table.columns[3:].str.replace('p_','').values
        codon_orders_list = [str(c) for c in g['codon_orders'].tolist()]
        if len(codon_orders_list) != len(set(codon_orders_list)):
            duplicate_codons = sorted(list(set([c for c in codon_orders_list if codon_orders_list.count(c) > 1])))
            duplicate_txt = ','.join(duplicate_codons[:10])
            if len(duplicate_codons) > 10:
                duplicate_txt += ',...'
            raise ValueError('Duplicate codon state columns were found in .state file: {}'.format(duplicate_txt))
    else:
        msg = 'Unsupported number of input states: {}. Only codon input (>20 states) is supported.'
        raise AssertionError(msg.format(g['num_input_state']))
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
    rate_table = pd.read_csv(g['path_iqtree_rate'], sep='\t', header=0, comment='#')
    rate_table.columns = [str(col).strip() for col in rate_table.columns]
    if rate_table.shape[0] == 0:
        return np.ones(g['num_input_site'])
    column_by_lower = {str(col).lower(): str(col) for col in rate_table.columns}
    if 'c_rate' in column_by_lower:
        rate_col = column_by_lower['c_rate']
    elif 'rate' in column_by_lower:
        rate_col = column_by_lower['rate']
        txt = 'Using "Rate" column from IQ-TREE rate file because "C_Rate" was not found.'
        print(txt, flush=True)
    else:
        available_cols = ','.join([str(c) for c in rate_table.columns.tolist()])
        txt = 'Unable to find site-rate column ("C_Rate" or "Rate") in {}. Columns: {}'
        raise ValueError(txt.format(g['path_iqtree_rate'], available_cols))
    rate_sites = pd.to_numeric(rate_table.loc[:, rate_col], errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(rate_sites).all():
        txt = 'Non-finite site-rate value(s) were found in {} column of {}.'
        raise ValueError(txt.format(rate_col, g['path_iqtree_rate']))
    expected_sites = int(g['num_input_site'])
    if rate_sites.shape[0] != expected_sites:
        txt = 'The number of site-rate rows in {} ({}) did not match num_input_site ({}).'
        raise ValueError(txt.format(g['path_iqtree_rate'], rate_sites.shape[0], expected_sites))
    return rate_sites

def read_iqtree(g, eq=True):
    iqtree_txt = _read_text(g['path_iqtree_iqtree'])
    g = detect_iqtree_output_version(g)
    if g['iqtree_output_version_major'] is None:
        # Fall back to executable version if .iqtree file lacks a version banner.
        g['iqtree_output_version'] = g.get('iqtree_version', None)
        g['iqtree_output_version_major'] = g.get('iqtree_version_major', None)
    if (g['iqtree_output_version_major'] is not None) and (g['iqtree_output_version_major']>=3):
        parser_name = 'iqtree3'
    else:
        parser_name = 'iqtree2'
    g['iqtree_parser'] = parser_name
    g['substitution_model'] = _parse_substitution_model(iqtree_txt)
    if g['substitution_model'] is None:
        raise AssertionError('Failed to parse substitution model from IQ-TREE output.')
    if eq:
        try:
            g['equilibrium_frequency'] = _parse_equilibrium_frequency(
                iqtree_txt=iqtree_txt,
                codon_orders=g['codon_orders'],
                parser_name=parser_name,
                float_type=g['float_type'],
            )
        except AssertionError as e:
            # IQ-TREE 3 may omit codon pi(...) entries from .iqtree outputs for codon models.
            if parser_name == 'iqtree3':
                txt = 'Could not parse codon frequencies from IQ-TREE 3 output. '
                txt += 'Estimating empirical codon frequencies from alignment: {}'
                print(txt.format(g['alignment_file']), flush=True)
                g['equilibrium_frequency'] = _estimate_empirical_codon_frequency_from_alignment(
                    alignment_file=g['alignment_file'],
                    codon_orders=g['codon_orders'],
                    float_type=g['float_type'],
                )
            else:
                raise e
    return g

def read_log(g):
    g = detect_iqtree_output_version(g)
    g['omega'] = None
    g['kappa'] = None
    g['reconstruction_codon_table'] = None
    with open(g['path_iqtree_log']) as f:
        lines = f.readlines()
    for line in lines:
        omega = _parse_float_from_log_line(line, 'Nonsynonymous/synonymous ratio (omega)')
        if omega is not None:
            g['omega'] = omega
        kappa = _parse_float_from_log_line(line, 'Transition/transversion ratio (kappa)')
        if kappa is not None:
            g['kappa'] = kappa
        rgc = re.match(r'\s*Converting to codon sequences with genetic code\s+([0-9]+)\s*\.\.\.', line)
        if rgc is not None:
            g['reconstruction_codon_table'] = int(rgc.group(1))
    return g

def _initialize_state_tensor(axis, dtype, selective, mmap_name):
    axis = tuple(axis)
    if not selective:
        return np.zeros(axis, dtype=dtype)
    mmap_tensor = os.path.join(os.getcwd(), mmap_name)
    if os.path.exists(mmap_tensor):
        os.unlink(mmap_tensor)
    txt = 'Generating memory map: dtype={}, axis={}, path={}'
    print(txt.format(dtype, axis, mmap_tensor), flush=True)
    return np.memmap(mmap_tensor, dtype=dtype, shape=axis, mode='w+')


def _get_num_node_axis(tree_obj):
    node_ids = list()
    for node in tree_obj.traverse():
        nl = ete.get_prop(node, "numerical_label")
        if isinstance(nl, (bool, np.bool_)):
            raise ValueError('numerical_label should be an integer.')
        try:
            nl_int = int(nl)
        except (TypeError, ValueError) as exc:
            raise ValueError('numerical_label should be an integer.') from exc
        if nl_int < 0:
            raise ValueError('numerical_label should be non-negative.')
        node_ids.append(nl_int)
    if len(node_ids) == 0:
        return 0
    return max(node_ids) + 1


def _normalize_selected_branch_ids(selected_branch_ids):
    if selected_branch_ids is None:
        return np.array([], dtype=np.int64)
    arr = np.asarray(selected_branch_ids, dtype=object)
    arr = np.atleast_1d(arr).reshape(-1)
    if arr.size == 0:
        return np.array([], dtype=np.int64)
    normalized = []
    for value in arr.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('selected_branch_ids should be integer-like.')
        if isinstance(value, (int, np.integer)):
            normalized.append(int(value))
            continue
        if isinstance(value, (float, np.floating)):
            if (not np.isfinite(value)) or (not float(value).is_integer()):
                raise ValueError('selected_branch_ids should be integer-like.')
            normalized.append(int(value))
            continue
        value_txt = str(value).strip()
        if (value_txt == '') or (not bool(re.fullmatch(r'[+-]?[0-9]+(?:\.0+)?', value_txt))):
            raise ValueError('selected_branch_ids should be integer-like.')
        normalized.append(int(float(value_txt)))
    return np.array(normalized, dtype=np.int64)


def _get_selected_branch_context(tree, selected_branch_ids):
    if selected_branch_ids is None:
        return None, None, set()
    selected_set = set(int(v) for v in _normalize_selected_branch_ids(selected_branch_ids))
    selected_internal_ids = list()
    node_by_id = dict()
    for node in tree.traverse():
        node_by_id[int(ete.get_prop(node, "numerical_label"))] = node
    selected_set = selected_set.intersection(set(node_by_id.keys()))
    root_nn = int(ete.get_prop(ete.get_tree_root(tree), "numerical_label"))
    selected_set.add(root_nn)
    for branch_id in sorted(selected_set):
        node = node_by_id.get(branch_id, None)
        if node is None:
            continue
        if ete.is_root(node) or ete.is_leaf(node):
            continue
        selected_internal_ids.append(branch_id)
    required_leaf_ids = set()
    for branch_id in selected_internal_ids:
        node = node_by_id[int(branch_id)]
        children = ete.get_children(node)
        sisters = ete.get_sisters(node)
        for child in children:
            required_leaf_ids.update([int(ete.get_prop(leaf, "numerical_label")) for leaf in ete.get_leaves(child)])
        for sister in sisters:
            required_leaf_ids.update([int(ete.get_prop(leaf, "numerical_label")) for leaf in ete.get_leaves(sister)])
    return selected_set, selected_internal_ids, required_leaf_ids


def _get_leaf_nonmissing_sites(g, required_leaf_ids):
    num_node = _get_num_node_axis(g['tree'])
    leaf_nonmissing = np.zeros(shape=(num_node, g['num_input_site']), dtype=bool)
    if len(required_leaf_ids) == 0:
        return leaf_nonmissing
    for node in g['tree'].traverse():
        if not ete.is_leaf(node):
            continue
        nl = int(ete.get_prop(node, "numerical_label"))
        if nl not in required_leaf_ids:
            continue
        if (nl < 0) or (nl >= num_node):
            txt = 'Leaf branch ID {} is out of bounds for state tensor axis {}.'
            raise ValueError(txt.format(nl, num_node))
        seq = ete.get_prop(node, 'sequence', '').upper()
        if seq == '':
            continue
        if (len(seq) % 3) != 0:
            raise AssertionError('Sequence length is not multiple of 3. Node name = ' + node.name)
        num_codon_site = len(seq) // 3
        if num_codon_site != g['num_input_site']:
            msg = 'Codon site count did not match alignment size for leaf "{}". '
            msg += 'Expected {}, observed {}.'
            raise AssertionError(msg.format(node.name, g['num_input_site'], num_codon_site))
        for s in np.arange(num_codon_site):
            codon = seq[(s*3):((s+1)*3)]
            codon_index = sequence.get_state_index(codon, g['codon_orders'], genetic_code.ambiguous_table)
            leaf_nonmissing[nl, s] = (len(codon_index) > 0)
    return leaf_nonmissing


def mask_missing_sites(state_tensor, tree, selected_internal_ids=None, leaf_nonmissing_sites=None):
    selected_set = None if selected_internal_ids is None else set([int(v) for v in selected_internal_ids])
    for node in tree.traverse():
        if (ete.is_root(node)) | (ete.is_leaf(node)):
            continue
        nl = int(ete.get_prop(node, "numerical_label"))
        if (selected_set is not None) and (nl not in selected_set):
            continue
        children = ete.get_children(node)
        sisters = ete.get_sisters(node)
        groups = list(children) + list(sisters)
        if len(groups) < 2:
            continue
        group_nonzero = list()
        for grp in groups:
            leaf_nls = np.array([int(ete.get_prop(l, "numerical_label")) for l in ete.get_leaves(grp)], dtype=int)
            if leaf_nls.shape[0] == 0:
                continue
            if leaf_nonmissing_sites is None:
                grp_nonzero = (state_tensor[leaf_nls,:,:].sum(axis=(0,2)) != 0)
            else:
                grp_nonzero = (leaf_nonmissing_sites[leaf_nls,:].sum(axis=0) != 0)
            group_nonzero.append(grp_nonzero)
        if len(group_nonzero) < 2:
            continue
        group_nonzero = np.stack(group_nonzero, axis=0)
        is_nonzero = (group_nonzero.sum(axis=0) >= 2)
        state_tensor[nl,:,:] = np.einsum('ij,i->ij', state_tensor[nl,:,:], is_nonzero)
    return state_tensor


def _validate_state_table_node_site_rows(state_table, expected_num_site):
    if state_table.shape[0] == 0:
        return state_table, np.array([], dtype=np.int64)
    required_columns = ['Node', 'Site', 'State']
    missing_columns = [col for col in required_columns if col not in state_table.columns]
    if len(missing_columns) > 0:
        txt = '.state file is missing required column(s): {}.'
        raise ValueError(txt.format(','.join(missing_columns)))
    site_values = pd.to_numeric(state_table.loc[:, 'Site'], errors='coerce')
    site_values_arr = site_values.to_numpy(dtype=float)
    if not np.isfinite(site_values_arr).all():
        raise ValueError('Non-numeric Site value(s) were found in .state file.')
    rounded_site_values = np.round(site_values_arr)
    if not np.isclose(site_values_arr, rounded_site_values, rtol=0.0, atol=1e-12).all():
        raise ValueError('Non-integer Site value(s) were found in .state file.')
    state_table = state_table.copy()
    state_table.loc[:, 'Site'] = rounded_site_values.astype(np.int64)
    is_duplicate_node_site = state_table.loc[:, ['Node', 'Site']].duplicated(keep=False)
    if is_duplicate_node_site.any():
        duplicated = state_table.loc[is_duplicate_node_site, ['Node', 'Site']].drop_duplicates()
        duplicated_examples = duplicated.head(10).to_dict(orient='records')
        duplicated_txt = ', '.join(['{}:{}'.format(d['Node'], int(d['Site'])) for d in duplicated_examples])
        if duplicated.shape[0] > 10:
            duplicated_txt += ',...'
        txt = 'Duplicate Node/Site row(s) were found in .state file: {}'
        raise ValueError(txt.format(duplicated_txt))
    expected_num_site = int(expected_num_site)
    site_count_by_node = state_table.groupby('Node', sort=False)['Site'].nunique()
    invalid_nodes = site_count_by_node[site_count_by_node != expected_num_site]
    if invalid_nodes.shape[0] > 0:
        invalid_examples = invalid_nodes.head(10)
        invalid_txt = ', '.join(['{}:{}'.format(node, int(count)) for node, count in invalid_examples.items()])
        if invalid_nodes.shape[0] > 10:
            invalid_txt += ',...'
        txt = 'Unexpected number of unique Site rows in .state file. '
        txt += 'Expected {} sites per node; observed {}'
        raise ValueError(txt.format(expected_num_site, invalid_txt))
    site_labels = np.sort(state_table.loc[:, 'Site'].unique())
    if site_labels.shape[0] != expected_num_site:
        txt = 'Unexpected number of unique Site labels in .state file. '
        txt += 'Expected {}, observed {}.'
        raise ValueError(txt.format(expected_num_site, site_labels.shape[0]))
    expected_site_set = frozenset(site_labels.tolist())
    node_site_set = state_table.groupby('Node', sort=False)['Site'].apply(lambda s: frozenset(s.tolist()))
    mismatched_nodes = node_site_set[node_site_set != expected_site_set]
    if mismatched_nodes.shape[0] > 0:
        mismatch_examples = mismatched_nodes.head(10)
        mismatch_txt = ', '.join([str(node) for node in mismatch_examples.index.tolist()])
        if mismatched_nodes.shape[0] > 10:
            mismatch_txt += ',...'
        txt = 'Inconsistent Site label sets were found among nodes in .state file: {}'
        raise ValueError(txt.format(mismatch_txt))
    return state_table, site_labels


def get_state_tensor(g, selected_branch_ids=None):
    if g.get('input_data_type', None) != 'cdn':
        raise NotImplementedError(
            'Non-codon input is obsolete and no longer supported. '
            'Run ancestral reconstruction with a codon model.'
        )
    ete.link_to_alignment(g['tree'], alignment=g['alignment_file'], alg_format='fasta')
    first_leaf_seq = ete.get_prop(ete.get_leaves(g['tree'])[0], 'sequence', '')
    if first_leaf_seq == '':
        msg = 'Failed to map alignment to tree leaves. Check leaf labels in --alignment_file and --rooted_tree_file.'
        raise AssertionError(msg)
    if (len(first_leaf_seq) % 3) != 0:
        raise AssertionError('Sequence length is not multiple of 3 in alignment file.')
    num_alignment_site = int(len(first_leaf_seq)/3)
    err_txt = 'The number of codon sites did not match between the alignment and ancestral states. ' \
              'Delete intermediate files and rerun.'
    if num_alignment_site != g['num_input_site']:
        raise AssertionError(err_txt)
    num_node = _get_num_node_axis(g['tree'])
    selected_set, selected_internal_ids, required_leaf_ids = _get_selected_branch_context(
        tree=g['tree'],
        selected_branch_ids=selected_branch_ids,
    )
    state_table = pd.read_csv(g['path_iqtree_state'], sep="\t", index_col=False, header=0, comment='#')
    if selected_set is not None:
        target_internal_names = []
        for node in g['tree'].traverse():
            if ete.is_root(node) or ete.is_leaf(node):
                continue
            nl = int(ete.get_prop(node, "numerical_label"))
            if nl in selected_set:
                target_internal_names.append(node.name)
        if len(target_internal_names) == 0:
            state_table = state_table.iloc[0:0,:].copy()
        else:
            state_table = state_table.loc[state_table.loc[:, 'Node'].isin(target_internal_names), :].copy()
    state_table, site_labels = _validate_state_table_node_site_rows(
        state_table=state_table,
        expected_num_site=g['num_input_site'],
    )
    state_columns = state_table.columns[3:]
    if state_table.shape[0] > 0:
        is_missing = (state_table.loc[:,'State']=='???') | (state_table.loc[:,'State']=='?')
        state_table.loc[is_missing, state_columns] = 0
        state_table_by_node = dict()
        site_index_by_label = {int(site_label): i for i, site_label in enumerate(site_labels.tolist())}
        for node_name, tmp in state_table.groupby('Node', sort=False):
            state_matrix = np.zeros(shape=(g['num_input_site'], g['num_input_state']), dtype=g['float_type'])
            row_values = tmp.loc[:, state_columns].to_numpy(dtype=g['float_type'], copy=False)
            row_sites = tmp.loc[:, 'Site'].to_numpy(dtype=np.int64, copy=False)
            row_indices = np.array([site_index_by_label[int(site)] for site in row_sites], dtype=np.int64)
            state_matrix[row_indices, :] = row_values
            state_table_by_node[node_name] = state_matrix
    else:
        state_table_by_node = dict()
    axis = [num_node, g['num_input_site'], g['num_input_state']]
    state_tensor = _initialize_state_tensor(
        axis=axis,
        dtype=g['float_type'],
        selective=(selected_set is not None),
        mmap_name='tmp.csubst.state_tensor.mmap',
    )
    codon_lookup = None
    if g['input_data_type'] == 'cdn':
        codon_lookup = _build_unambiguous_codon_lookup(g['codon_orders'])
    for node in g['tree'].traverse():
        if ete.is_root(node):
            continue
        nl = int(ete.get_prop(node, "numerical_label"))
        if (nl < 0) or (nl >= num_node):
            txt = 'Branch ID {} is out of bounds for state tensor axis {}.'
            raise ValueError(txt.format(nl, num_node))
        if (selected_set is not None) and (nl not in selected_set):
            continue
        elif ete.is_leaf(node):
            seq = ete.get_prop(node, 'sequence', '').upper()
            if seq == '':
                msg = 'Leaf sequence not found for node "{}". Check tree/alignment labels.'
                raise AssertionError(msg.format(node.name))
            state_matrix = np.zeros([g['num_input_site'], g['num_input_state']], dtype=g['float_type'])
            if g['input_data_type']=='cdn':
                if (len(seq) % 3) != 0:
                    raise AssertionError('Sequence length is not multiple of 3. Node name = ' + node.name)
                num_codon_site = len(seq) // 3
                if num_codon_site != g['num_input_site']:
                    msg = 'Codon site count did not match alignment size for leaf "{}". '
                    msg += 'Expected {}, observed {}.'
                    raise AssertionError(msg.format(node.name, g['num_input_site'], num_codon_site))
                _fill_leaf_state_matrix_codon(
                    seq=seq,
                    state_matrix=state_matrix,
                    g=g,
                    codon_lookup=codon_lookup,
                )
            elif g['input_data_type']=='nuc':
                for s in np.arange(len(seq)):
                    nuc_index = sequence.get_state_index(seq[s], g['input_state'], genetic_code.ambiguous_table)
                    for ni in nuc_index:
                        state_matrix[s, ni] = 1/len(nuc_index)
            state_tensor[nl,:,:] = state_matrix
        else: # Internal nodes
            state_matrix = state_table_by_node.get(node.name, None)
            if state_matrix is None:
                print('Node name not found in .state file:', node.name)
            else:
                state_tensor[nl,:,:] = state_matrix
    state_tensor = np.nan_to_num(state_tensor, copy=False)
    if selected_set is None:
        state_tensor = mask_missing_sites(state_tensor, g['tree'])
    else:
        leaf_nonmissing_sites = _get_leaf_nonmissing_sites(g, required_leaf_ids=required_leaf_ids)
        state_tensor = mask_missing_sites(
            state_tensor=state_tensor,
            tree=g['tree'],
            selected_internal_ids=selected_internal_ids,
            leaf_nonmissing_sites=leaf_nonmissing_sites,
        )
    if (g['ml_anc']):
        print('Ancestral state frequency is converted to the ML-like binary states.')
        if selected_set is None:
            idxmax = np.argmax(state_tensor, axis=2)
            state_tensor2 = np.zeros(state_tensor.shape, dtype=bool)
            for b in np.arange(state_tensor2.shape[0]):
                for s in np.arange(state_tensor2.shape[1]):
                    if state_tensor[b,s,:].sum()!=0:
                        state_tensor2[b,s,idxmax[b,s]] = 1
            state_tensor = state_tensor2
            del state_tensor2
        else:
            state_tensor2 = _initialize_state_tensor(
                axis=state_tensor.shape,
                dtype=bool,
                selective=True,
                mmap_name='tmp.csubst.state_tensor_ml.mmap',
            )
            for b in sorted(selected_set):
                branch_state = state_tensor[b, :, :]
                if branch_state.sum() == 0:
                    continue
                idxmax = np.argmax(branch_state, axis=1)
                is_nonmissing = (branch_state.sum(axis=1) != 0)
                if is_nonmissing.any():
                    state_tensor2[b, is_nonmissing, idxmax[is_nonmissing]] = True
            state_tensor = state_tensor2
    return(state_tensor)

def get_input_information(g):
    g = read_state(g)
    g = read_iqtree(g)
    g = read_log(g)
    g['iqtree_rate_values'] = read_rate(g)
    return g
