"""Sparse projection helpers for codon-model expected substitutions.

This module is deliberately independent of the high-level omega workflow.  It
owns projection packing and incremental CSR construction so the large analysis
module does not also have to implement sparse storage mechanics.
"""

import warnings

import numpy as np
from scipy import sparse as sp

from csubst._extensions import load_optional_extension


omega_cy = load_optional_extension('omega_cy')
_WARNED_CYTHON_FALLBACKS: set[str] = set()


def _warn_cython_fallback(operation, exc):
    if operation in _WARNED_CYTHON_FALLBACKS:
        return
    _WARNED_CYTHON_FALLBACKS.add(operation)
    warnings.warn(
        'Cython {} fast path failed; using NumPy fallback: {}'.format(operation, exc),
        RuntimeWarning,
        stacklevel=3,
    )


def feature_count(stat, num_group, num_state):
    if stat == 'any2any':
        return int(num_group)
    if stat in ('spe2any', 'any2spe'):
        return int(num_group) * int(num_state)
    if stat == 'spe2spe':
        return int(num_group) * int(num_state) * int(num_state)
    raise ValueError('Unsupported expected projection statistic: {}'.format(stat))


def build_state_indices(sub_mode, num_group, num_state, syn_indices_list):
    if sub_mode == 'asis':
        return np.arange(num_state, dtype=np.int64).reshape(1, num_state)
    state_indices = np.full((num_group, num_state), -1, dtype=np.int64)
    for group_index, indices in enumerate(syn_indices_list):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.shape[0] > num_state:
            raise ValueError('A synonymous state group is larger than max_synonymous_size.')
        state_indices[group_index, :indices.shape[0]] = indices
    return state_indices


def projection_values(
    parent_state,
    expected_state,
    sub_mode,
    num_group,
    num_state,
    syn_indices_list,
    selected,
):
    num_site = parent_state.shape[0]
    out = {
        stat: np.zeros(
            (num_site, feature_count(stat, num_group, num_state)),
            dtype=np.float64,
        )
        for stat in selected
    }
    total_by_site = np.zeros(num_site, dtype=np.float64)
    groups = [np.arange(num_state, dtype=np.int64)] if sub_mode == 'asis' else syn_indices_list
    for group_index, state_indices in enumerate(groups):
        state_indices = np.asarray(state_indices, dtype=np.int64)
        size = int(state_indices.shape[0])
        if size <= 1:
            continue
        parent = np.asarray(parent_state[:, state_indices], dtype=np.float64)
        child = np.asarray(expected_state[:, state_indices], dtype=np.float64)
        parent_sum = parent.sum(axis=1)
        child_sum = child.sum(axis=1)
        diagonal = (parent * child).sum(axis=1)
        total_group = parent_sum * child_sum - diagonal
        total_by_site += total_group
        if 'any2any' in out:
            out['any2any'][:, group_index] = total_group
        if 'spe2any' in out:
            base = group_index * num_state
            out['spe2any'][:, base:base + size] = parent * (child_sum[:, None] - child)
        if 'any2spe' in out:
            base = group_index * num_state
            out['any2spe'][:, base:base + size] = child * (parent_sum[:, None] - parent)
        if 'spe2spe' in out:
            pair = parent[:, :, None] * child[:, None, :]
            diagonal_ids = np.arange(size)
            pair[:, diagonal_ids, diagonal_ids] = 0.0
            base = group_index * num_state * num_state
            for state_index in range(size):
                row_start = base + state_index * num_state
                out['spe2spe'][:, row_start:row_start + size] = pair[:, state_index, :]
    return out, total_by_site


def branch_projection_payloads(
    parent_state,
    expected_state,
    sub_mode,
    num_group,
    num_state,
    syn_indices_list,
    state_indices,
    selected,
):
    """Return packed CSR row payloads without materializing dense projections."""
    cython_fn = None if omega_cy is None else getattr(
        omega_cy,
        'build_expected_projection_rows_double',
        None,
    )
    if cython_fn is not None:
        try:
            rows = cython_fn(
                np.ascontiguousarray(parent_state, dtype=np.float64),
                np.ascontiguousarray(expected_state, dtype=np.float64),
                np.ascontiguousarray(state_indices, dtype=np.int64),
                int(num_state),
                'any2any' in selected,
                'spe2any' in selected,
                'any2spe' in selected,
                'spe2spe' in selected,
            )
            arrays = {
                'any2any': (rows[0], rows[1]),
                'spe2any': (rows[2], rows[3]),
                'any2spe': (rows[4], rows[5]),
                'spe2spe': (rows[6], rows[7]),
            }
            return {
                stat: arrays[stat]
                for stat in selected
                if arrays[stat][0].shape[0] > 0
            }, float(rows[8])
        except Exception as exc:
            _warn_cython_fallback('expected_projection_rows', exc)
    branch_values, total_by_site = projection_values(
        parent_state=parent_state,
        expected_state=expected_state,
        sub_mode=sub_mode,
        num_group=num_group,
        num_state=num_state,
        syn_indices_list=syn_indices_list,
        selected=selected,
    )
    payloads = {}
    for stat, values in branch_values.items():
        flat = values.T.reshape(-1)
        nonzero = np.flatnonzero(flat != 0)
        if nonzero.shape[0] > 0:
            payloads[stat] = (
                nonzero.astype(np.int32, copy=False),
                flat[nonzero].astype(np.float64, copy=False),
            )
    return payloads, float(total_by_site.sum())


class CSRRowBuilder:
    """Append monotonically ordered CSR rows into reusable growable buffers."""

    def __init__(self, num_row, num_column, initial_capacity=1024):
        self.num_row = int(num_row)
        self.num_column = int(num_column)
        self.indptr = np.zeros(self.num_row + 1, dtype=np.int64)
        capacity = max(1, int(initial_capacity))
        self.indices = np.empty(capacity, dtype=np.int32)
        self.data = np.empty(capacity, dtype=np.float64)
        self.nnz = 0
        self.next_row = 0

    def _reserve(self, required):
        if required <= self.indices.shape[0]:
            return
        capacity = max(required, self.indices.shape[0] + self.indices.shape[0] // 2, 1024)
        self.indices.resize(capacity, refcheck=False)
        self.data.resize(capacity, refcheck=False)

    def append(self, row_id, row_indices, row_data):
        row_id = int(row_id)
        if row_id < self.next_row:
            raise ValueError('CSR rows must be appended in nondecreasing order.')
        if (row_id < 0) or (row_id >= self.num_row):
            raise ValueError('CSR row {} is outside [0, {}).'.format(row_id, self.num_row))
        while self.next_row < row_id:
            self.indptr[self.next_row + 1] = self.nnz
            self.next_row += 1
        row_indices = np.asarray(row_indices, dtype=np.int32).reshape(-1)
        row_data = np.asarray(row_data, dtype=np.float64).reshape(-1)
        if row_indices.shape[0] != row_data.shape[0]:
            raise ValueError('CSR row index/data lengths do not match.')
        stop = self.nnz + row_indices.shape[0]
        self._reserve(stop)
        self.indices[self.nnz:stop] = row_indices
        self.data[self.nnz:stop] = row_data
        self.nnz = stop
        self.next_row = row_id + 1
        self.indptr[self.next_row] = self.nnz

    def finalize(self):
        while self.next_row < self.num_row:
            self.indptr[self.next_row + 1] = self.nnz
            self.next_row += 1
        self.indices.resize(self.nnz, refcheck=False)
        self.data.resize(self.nnz, refcheck=False)
        matrix = sp.csr_matrix(
            (self.data, self.indices, self.indptr),
            shape=(self.num_row, self.num_column),
            dtype=np.float64,
            copy=False,
        )
        matrix.sort_indices()
        return matrix
