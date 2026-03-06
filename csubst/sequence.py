import gzip
import os
import itertools
import re

import numpy as np

from csubst import ete
from csubst import runtime

_NSY_ALIGNMENT_SYMBOLS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz")
_GROUP_SUM_MATRIX_CACHE = dict()


def calc_omega_state(state_nuc, g):  # implement exclude stop codon freq
    num_node = state_nuc.shape[0]
    num_nuc_site = state_nuc.shape[1]
    if num_nuc_site % 3 != 0:
        raise ValueError('The sequence length is not multiple of 3. num_site = {}'.format(num_nuc_site))
    num_cdn_site = num_nuc_site // 3
    state_columns = g['state_columns']
    state_cdn = np.zeros((num_node, num_cdn_site, len(state_columns)), dtype=state_nuc.dtype)
    codon_sites = np.arange(0, num_nuc_site, 3)
    for i, (state0, state1, state2) in enumerate(state_columns):
        codon_prob = state_nuc[:, codon_sites + 0, state0]
        codon_prob *= state_nuc[:, codon_sites + 1, state1]
        codon_prob *= state_nuc[:, codon_sites + 2, state2]
        state_cdn[:, :, i] = codon_prob
    return state_cdn


def _initialize_state_array(axis, dtype, mmap_name=None):
    axis = tuple(axis)
    if mmap_name is None:
        return np.zeros(axis, dtype=dtype)
    mmap_path = runtime.temp_path(mmap_name)
    if os.path.exists(mmap_path):
        os.unlink(mmap_path)
    txt = 'Generating memory map: dtype={}, axis={}, path={}'
    print(txt.format(dtype, axis, mmap_path), flush=True)
    return np.memmap(mmap_path, dtype=dtype, shape=axis, mode='w+')


def _normalize_branch_ids(branch_ids):
    if branch_ids is None:
        return np.array([], dtype=np.int64)
    arr = np.asarray(branch_ids, dtype=object)
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


def _cdn2group_state(state_cdn, group_orders, group_indices, selected_branch_ids=None, mmap_name=None):
    num_node = state_cdn.shape[0]
    num_cdn_site = state_cdn.shape[1]
    num_pep_site = num_cdn_site
    num_pep_state = len(group_orders)
    axis = [num_node, num_pep_site, num_pep_state]
    selected_ids = None
    selected_state_cdn = state_cdn
    if selected_branch_ids is None:
        state_pep = _initialize_state_array(axis, dtype=state_cdn.dtype)
    else:
        state_pep = _initialize_state_array(
            axis=axis,
            dtype=state_cdn.dtype,
            mmap_name=mmap_name,
        )
        selected_ids_all = _normalize_branch_ids(selected_branch_ids)
        is_valid = (selected_ids_all >= 0) & (selected_ids_all < num_node)
        selected_ids = np.array(sorted(set(selected_ids_all[is_valid].tolist())), dtype=np.int64)
        selected_state_cdn = state_cdn[selected_ids, :, :]
    group_matrix = _get_group_sum_matrix(
        group_orders=group_orders,
        group_indices=group_indices,
        num_codon_state=state_cdn.shape[2],
        dtype=state_cdn.dtype,
    )
    target = np.tensordot(selected_state_cdn, group_matrix, axes=([2], [0]))
    if selected_ids is None:
        state_pep[:, :, :] = target
    else:
        state_pep[selected_ids, :, :] = target
    return state_pep


def _get_group_sum_matrix(group_orders, group_indices, num_codon_state, dtype):
    orders = tuple(str(order) for order in np.asarray(group_orders, dtype=object).reshape(-1).tolist())
    dtype = np.dtype(dtype)
    frozen_indices = tuple(
        tuple(int(i) for i in np.asarray(group_indices[state_name], dtype=np.int64).reshape(-1).tolist())
        for state_name in orders
    )
    cache_key = (orders, frozen_indices, int(num_codon_state), dtype.str)
    cached = _GROUP_SUM_MATRIX_CACHE.get(cache_key, None)
    if cached is not None:
        return cached
    matrix = np.zeros((int(num_codon_state), len(orders)), dtype=dtype)
    for col, indices in enumerate(frozen_indices):
        if len(indices) == 0:
            continue
        matrix[np.asarray(indices, dtype=np.int64), col] = 1
    _GROUP_SUM_MATRIX_CACHE[cache_key] = matrix
    return matrix


def cdn2pep_state(state_cdn, g, selected_branch_ids=None):
    return _cdn2group_state(
        state_cdn=state_cdn,
        group_orders=g['amino_acid_orders'],
        group_indices=g['synonymous_indices'],
        selected_branch_ids=selected_branch_ids,
        mmap_name='tmp.csubst.state_pep.mmap',
    )


def cdn2nsy_state(state_cdn, g, selected_branch_ids=None):
    return _cdn2group_state(
        state_cdn=state_cdn,
        group_orders=g['nonsyn_state_orders'],
        group_indices=g['nonsynonymous_indices'],
        selected_branch_ids=selected_branch_ids,
        mmap_name='tmp.csubst.state_nsy.mmap',
    )


def _build_nsy_alignment_symbols(orders):
    orders = [str(order) for order in np.asarray(orders, dtype=object).reshape(-1).tolist()]
    if len(orders) > len(_NSY_ALIGNMENT_SYMBOLS):
        txt = 'Too many recoded nonsynonymous states to encode as one-character alignment symbols: {}'
        raise ValueError(txt.format(len(orders)))
    unique_orders = set(orders)
    all_single_char = all([(len(order) == 1) for order in orders])
    if all_single_char and (len(unique_orders) == len(orders)) and ('-' not in unique_orders):
        # Keep canonical amino-acid symbols when no grouping is applied.
        return np.array(orders, dtype=object)
    # For grouped recoding schemes, use stable compact symbols by state order.
    return np.array(_NSY_ALIGNMENT_SYMBOLS[:len(orders)], dtype=object)


def _get_nsy_alignment_symbols(g):
    orders = tuple(np.asarray(g['nonsyn_state_orders'], dtype=object).reshape(-1).tolist())
    cache = g.get('_nsy_alignment_symbols_cache', None)
    if isinstance(cache, dict):
        if cache.get('orders', None) == orders:
            return cache['symbols']
    symbols = _build_nsy_alignment_symbols(orders=orders)
    g['_nsy_alignment_symbols_cache'] = {
        'orders': orders,
        'symbols': symbols,
    }
    return symbols


def translate_state(nlabel, mode, g):
    if mode == 'codon':
        missing_state = '---'
        state = g['state_cdn']
        orders = g['codon_orders']
    elif mode == 'aa':
        missing_state = '-'
        state = g['state_pep']
        orders = g['amino_acid_orders']
    elif mode == 'nsy':
        missing_state = '-'
        state = g['state_nsy']
        orders = _get_nsy_alignment_symbols(g)
    else:
        raise ValueError('Unsupported translate_state mode: {}'.format(mode))
    seq_out = list()
    for i in range(state.shape[1]):
        if state[nlabel, i, :].max() < g['float_tol']:
            seq_out.append(missing_state)
        else:
            index = state[nlabel, i, :].argmax()
            seq_out.append(orders[index])
    return ''.join(seq_out)


def _resolve_alignment_mode(mode, g):
    if mode == 'codon':
        missing_state = '---'
        state = g['state_cdn']
        orders = g['codon_orders']
    elif mode == 'aa':
        missing_state = '-'
        state = g['state_pep']
        orders = g['amino_acid_orders']
    elif mode == 'nsy':
        missing_state = '-'
        state = g['state_nsy']
        orders = _get_nsy_alignment_symbols(g)
    else:
        raise ValueError('Unsupported translate_state mode: {}'.format(mode))
    return missing_state, state, np.asarray(orders, dtype=object)


def write_alignment(outfile, mode, g, leaf_only=False, branch_ids=None):
    aln_out = list()
    branch_id_set = None
    if branch_ids is not None:
        branch_id_set = set(int(bid) for bid in _normalize_branch_ids(branch_ids))
    missing_state, state, orders = _resolve_alignment_mode(mode=mode, g=g)
    records = list()
    if leaf_only:
        nodes = ete.iter_leaves(g['tree'])
    else:
        nodes = g['tree'].traverse()
    for node in nodes:
        if ete.is_root(node):
            continue
        nlabel = ete.get_prop(node, "numerical_label")
        if (branch_id_set is not None) and (nlabel not in branch_id_set):
            continue
        node_name = '' if (node.name is None) else str(node.name)
        records.append((node_name, int(nlabel)))
    if len(records) != 0:
        target_ids = np.asarray([r[1] for r in records], dtype=np.int64)
        state_target = state[target_ids, :, :]
        site_max = state_target.max(axis=2)
        site_argmax = state_target.argmax(axis=2)
        is_missing = (site_max < g['float_tol'])
        for row_index, (node_name, _nlabel) in enumerate(records):
            symbols = orders[site_argmax[row_index, :]]
            if np.any(is_missing[row_index, :]):
                symbols = symbols.copy()
                symbols[is_missing[row_index, :]] = missing_state
            aln_out.append('>' + node_name)
            aln_out.append(''.join(symbols.tolist()))
    with open(outfile, 'w') as f:
        print('Writing sequence alignment:', outfile, flush=True)
        if len(aln_out) != 0:
            f.write('\n'.join(aln_out) + '\n')


def build_state_index_lookup(input_state):
    input_states = np.asarray(input_state, dtype=object).reshape(-1)
    lookup = dict()
    for i, raw_state in enumerate(input_states.tolist()):
        state_txt = str(raw_state).upper()
        if state_txt in lookup:
            continue
        lookup[state_txt] = int(i)
    return lookup


def get_state_index(state, input_state, ambiguous_table, state_lookup=None):
    state = str(state).upper()
    if ('-' in state) or (state == 'NNN') or (state == 'N'):
        return []
    if state_lookup is None:
        state_lookup = build_state_index_lookup(input_state=input_state)
    state_options = []
    has_ambiguous_char = False
    for state_char in state:
        if state_char in ambiguous_table:
            has_ambiguous_char = True
            state_options.append([str(c).upper() for c in ambiguous_table[state_char]])
        else:
            state_options.append([state_char])
    if not has_ambiguous_char:
        index = state_lookup.get(state, None)
        if index is None:
            return []
        return [int(index)]
    state_index = list()
    seen = set()
    for chars in itertools.product(*state_options):
        resolved_state = ''.join(chars)
        if resolved_state in seen:
            continue
        seen.add(resolved_state)
        index = state_lookup.get(resolved_state, None)
        if index is None:
            continue
        state_index.append(int(index))
    return state_index


def read_fasta(path):
    seq_dict = dict()
    seq_name = None
    seq_parts = list()
    seen_names = set()
    path_txt = str(path)
    open_fn = gzip.open if path_txt.lower().endswith('.gz') else open
    with open_fn(path_txt, mode='rt', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip('\n')
            if line.startswith('>'):
                if seq_name is not None:
                    seq_dict[seq_name] = ''.join(seq_parts)
                seq_name = line[1:].strip()
                if seq_name == '':
                    txt = 'Invalid FASTA header in {} at line {}: sequence name is empty.'
                    raise ValueError(txt.format(path, line_no))
                if seq_name in seen_names:
                    txt = 'Duplicate FASTA header "{}" found in {} at line {}.'
                    raise ValueError(txt.format(seq_name, path, line_no))
                seen_names.add(seq_name)
                seq_parts = list()
                continue
            if seq_name is None:
                if line.strip() == '':
                    continue
                txt = 'Invalid FASTA format in {} at line {}: sequence line appeared before header.'
                raise ValueError(txt.format(path, line_no))
            if line.strip() == '':
                continue
            seq_parts.append(line.strip())
    if seq_name is not None:
        seq_dict[seq_name] = ''.join(seq_parts)
    return seq_dict


def calc_identity(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError('Sequence lengths should be identical.')
    if len(seq1) == 0:
        raise ValueError('Sequences should be non-empty.')
    num_same_site = sum(1 for s1, s2 in zip(seq1, seq2) if s1 == s2)
    return num_same_site / len(seq1)
