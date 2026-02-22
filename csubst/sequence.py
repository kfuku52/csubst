import os
import itertools
import re

import numpy as np

from csubst import ete


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
    mmap_path = os.path.join(os.getcwd(), mmap_name)
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


def cdn2pep_state(state_cdn, g, selected_branch_ids=None):
    num_node = state_cdn.shape[0]
    num_cdn_site = state_cdn.shape[1]
    num_pep_site = num_cdn_site
    num_pep_state = len(g['amino_acid_orders'])
    axis = [num_node, num_pep_site, num_pep_state]
    selected_ids = None
    selected_state_cdn = state_cdn
    if selected_branch_ids is None:
        state_pep = _initialize_state_array(axis, dtype=state_cdn.dtype)
    else:
        state_pep = _initialize_state_array(
            axis=axis,
            dtype=state_cdn.dtype,
            mmap_name='tmp.csubst.state_pep.mmap',
        )
        selected_ids_all = _normalize_branch_ids(selected_branch_ids)
        is_valid = (selected_ids_all >= 0) & (selected_ids_all < num_node)
        selected_ids = np.array(sorted(set(selected_ids_all[is_valid].tolist())), dtype=np.int64)
        selected_state_cdn = state_cdn[selected_ids, :, :]
    for i, aa in enumerate(g['amino_acid_orders']):
        target = selected_state_cdn[:, :, g['synonymous_indices'][aa]].sum(axis=2)
        if selected_ids is None:
            state_pep[:, :, i] = target
        else:
            state_pep[selected_ids, :, i] = target
    return state_pep


def translate_state(nlabel, mode, g):
    if mode == 'codon':
        missing_state = '---'
        state = g['state_cdn']
        orders = g['codon_orders']
    elif mode == 'aa':
        missing_state = '-'
        state = g['state_pep']
        orders = g['amino_acid_orders']
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


def write_alignment(outfile, mode, g, leaf_only=False, branch_ids=None):
    aln_out = list()
    branch_id_set = None
    if branch_ids is not None:
        branch_id_set = set(int(bid) for bid in _normalize_branch_ids(branch_ids))
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
        aln_out.append('>' + node.name + '|' + str(nlabel))
        aln_out.append(translate_state(nlabel, mode, g))
    with open(outfile, 'w') as f:
        print('Writing sequence alignment:', outfile, flush=True)
        if len(aln_out) != 0:
            f.write('\n'.join(aln_out) + '\n')


def get_state_index(state, input_state, ambiguous_table):
    state = str(state).upper()
    if ('-' in state) or (state == 'NNN') or (state == 'N'):
        return []
    state_options = []
    for state_char in state:
        if state_char in ambiguous_table:
            state_options.append([str(c).upper() for c in ambiguous_table[state_char]])
        else:
            state_options.append([state_char])
    states = [''.join(chars) for chars in itertools.product(*state_options)]
    states = list(dict.fromkeys(states))
    state_index0 = [np.where(input_state == s)[0] for s in states]
    state_index0 = [s for s in state_index0 if s.shape[0] != 0]
    if len(state_index0) == 0:
        return []
    state_index = [int(idx) for si in state_index0 for idx in si]
    return state_index


def read_fasta(path):
    seq_dict = dict()
    seq_name = None
    seq_parts = list()
    seen_names = set()
    with open(path, mode='r') as f:
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
