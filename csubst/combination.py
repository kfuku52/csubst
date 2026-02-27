import numpy as np
import pandas as pd

import itertools
import math
import re
import sys
import time
from collections import defaultdict

from csubst import parallel
from csubst import ete
try:
    from csubst import combination_cy
except Exception:  # pragma: no cover - Cython extension is optional
    combination_cy = None


_ARITY3_DENSE_EDGE_LOOKUP_MAX_NODES = 6000


def _unique_rows_int64(values, hash_threshold=2048):
    values = np.asarray(values, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError('values should be a 2D array.')
    if values.shape[0] == 0:
        return np.zeros(shape=(0, values.shape[1]), dtype=np.int64)
    if values.shape[0] < int(hash_threshold):
        return np.unique(values, axis=0)
    unique_values = pd.DataFrame(values, copy=False).drop_duplicates(ignore_index=False).to_numpy(dtype=np.int64, copy=False)
    if unique_values.shape[0] <= 1:
        return unique_values.reshape(-1, values.shape[1])
    order = np.lexsort(unique_values.T[::-1])
    return unique_values[order, :]


def _unique_sorted_int64_1d(values, bitmap_ratio=6):
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if values.shape[0] == 0:
        return np.array([], dtype=np.int64)
    min_value = int(values.min())
    max_value = int(values.max())
    span = max_value - min_value + 1
    if (min_value >= 0) and (span > 0) and (span <= int(bitmap_ratio) * int(values.shape[0])):
        bitmap = np.zeros(shape=(span,), dtype=np.uint8)
        bitmap[values - min_value] = 1
        return np.flatnonzero(bitmap).astype(np.int64, copy=False) + np.int64(min_value)
    return np.unique(values)


def _generate_all_k_combinations_from_sorted_nodes(unique_nodes, k):
    unique_nodes = np.asarray(unique_nodes, dtype=np.int64).reshape(-1)
    k = int(k)
    if k <= 0:
        return np.zeros(shape=(0, 0), dtype=np.int64)
    if unique_nodes.shape[0] < k:
        return np.zeros(shape=(0, k), dtype=np.int64)
    if combination_cy is not None:
        cython_fn = getattr(combination_cy, 'generate_all_k_combinations_from_sorted_nodes_int64', None)
        if cython_fn is not None:
            try:
                return np.asarray(cython_fn(unique_nodes, int(k)), dtype=np.int64)
            except Exception:
                pass
    if k == 3:
        return _generate_all_triples_from_sorted_nodes(unique_nodes)
    total = int(math.comb(int(unique_nodes.shape[0]), int(k)))
    if total == 0:
        return np.zeros(shape=(0, k), dtype=np.int64)
    flat = np.fromiter(
        (value for comb in itertools.combinations(unique_nodes.tolist(), k) for value in comb),
        dtype=np.int64,
        count=total * k,
    )
    return flat.reshape(total, k)


def _is_complete_subset_family(sorted_nodes):
    sorted_nodes = np.asarray(sorted_nodes, dtype=np.int64)
    if sorted_nodes.ndim != 2:
        return False, np.array([], dtype=np.int64)
    width = int(sorted_nodes.shape[1])
    if width <= 0 or sorted_nodes.shape[0] == 0:
        return False, np.array([], dtype=np.int64)
    unique_nodes = _unique_sorted_int64_1d(sorted_nodes.reshape(-1))
    if unique_nodes.shape[0] < width:
        return False, unique_nodes
    expected_rows = int(math.comb(int(unique_nodes.shape[0]), width))
    return expected_rows == int(sorted_nodes.shape[0]), unique_nodes


def _normalize_node_ids(node_ids):
    if node_ids is None:
        return np.array([], dtype=np.int64)
    values = np.asarray(node_ids)
    if values.ndim == 0:
        scalar = values.item()
        if isinstance(scalar, (list, tuple, set, np.ndarray)):
            values = np.asarray(list(scalar))
    values = np.atleast_1d(values).reshape(-1)
    if values.size == 0:
        return np.array([], dtype=np.int64)
    kind = values.dtype.kind
    if kind == 'b':
        raise ValueError('Dependency node IDs should be integer-like.')
    if kind in ('i', 'u'):
        return values.astype(np.int64, copy=False)
    if kind == 'f':
        is_finite = np.all(np.isfinite(values))
        is_integer_like = np.all(np.equal(values, np.floor(values)))
        if (not is_finite) or (not is_integer_like):
            raise ValueError('Dependency node IDs should be integer-like.')
        return values.astype(np.int64, copy=False)
    normalized = []
    for value in values.tolist():
        if isinstance(value, bool):
            raise ValueError('Dependency node IDs should be integer-like.')
        if isinstance(value, (int, np.integer)):
            normalized.append(int(value))
            continue
        if isinstance(value, (float, np.floating)):
            if (not np.isfinite(value)) or (not float(value).is_integer()):
                raise ValueError('Dependency node IDs should be integer-like.')
            normalized.append(int(value))
            continue
        value_txt = str(value).strip()
        if (value_txt == '') or (not bool(re.fullmatch(r'[+-]?[0-9]+(?:\.0+)?', value_txt))):
            raise ValueError('Dependency node IDs should be integer-like.')
        normalized.append(int(float(value_txt)))
    return np.array(normalized, dtype=np.int64)


def _build_node_id_to_row(node_ids):
    normalized_ids = _normalize_node_ids(node_ids)
    unique_ids = np.unique(normalized_ids)
    if unique_ids.shape[0] != normalized_ids.shape[0]:
        raise ValueError('Node IDs should be unique.')
    return normalized_ids, {int(node_id): i for i, node_id in enumerate(normalized_ids.tolist())}


def _map_node_ids_to_rows(node_ids, id_to_row, context):
    mapped = []
    missing = []
    for node_id in _normalize_node_ids(node_ids).tolist():
        row = id_to_row.get(int(node_id), None)
        if row is None:
            missing.append(int(node_id))
            continue
        mapped.append(int(row))
    if len(missing) > 0:
        missing_txt = ', '.join([str(v) for v in sorted(set(missing))])
        txt = '{} contain node IDs that are not present in the tree: {}'
        raise ValueError(txt.format(context, missing_txt))
    return np.array(mapped, dtype=np.int64)


def _map_row_combinations_to_node_ids(row_combinations, node_ids, row_ids_are_node_ids=False):
    row_combinations = np.asarray(row_combinations, dtype=np.int64)
    if row_combinations.size == 0:
        arity = row_combinations.shape[1] if row_combinations.ndim == 2 else 0
        return np.zeros(shape=(0, arity), dtype=np.int64)
    if row_ids_are_node_ids:
        return row_combinations
    return node_ids[row_combinations]


def _pairwise_node_combinations(values):
    values = _normalize_node_ids(values)
    if values.shape[0] < 2:
        return np.zeros(shape=(0, 2), dtype=np.int64)
    left, right = np.triu_indices(values.shape[0], k=1)
    out = np.empty(shape=(left.shape[0], 2), dtype=np.int64)
    left_values = values[left]
    right_values = values[right]
    out[:, 0] = np.minimum(left_values, right_values)
    out[:, 1] = np.maximum(left_values, right_values)
    if out.shape[0] == 0:
        return out
    out = out[(out[:, 0] != out[:, 1]), :]
    if out.shape[0] == 0:
        return np.zeros(shape=(0, 2), dtype=np.int64)
    return _unique_rows_int64(out)


def _generate_valid_unions_by_pair_scan(target_nodes, arity):
    target_nodes = np.asarray(target_nodes, dtype=np.int64)
    if target_nodes.shape[0] < 2:
        return np.zeros(shape=(0, arity), dtype=np.int64)
    out = list()
    for i, j in itertools.combinations(np.arange(target_nodes.shape[0]), 2):
        node_union = np.union1d(target_nodes[i, :], target_nodes[j, :])
        if node_union.shape[0] == arity:
            out.append(node_union)
    if len(out) == 0:
        return np.zeros(shape=(0, arity), dtype=np.int64)
    return _unique_rows_int64(np.asarray(out, dtype=np.int64))


def _generate_all_triples_from_sorted_nodes(unique_nodes):
    unique_nodes = np.asarray(unique_nodes, dtype=np.int64).reshape(-1)
    num_nodes = int(unique_nodes.shape[0])
    if num_nodes < 3:
        return np.zeros(shape=(0, 3), dtype=np.int64)
    if combination_cy is not None:
        cython_fn = getattr(combination_cy, 'generate_all_triples_from_sorted_nodes_int64', None)
        if cython_fn is not None:
            try:
                return cython_fn(unique_nodes)
            except Exception:
                pass
    total = (num_nodes * (num_nodes - 1) * (num_nodes - 2)) // 6
    out = np.empty(shape=(total, 3), dtype=np.int64)
    write_pos = 0
    triu_cache = dict()
    for i in np.arange(num_nodes - 2):
        rem = num_nodes - int(i) - 1
        triu = triu_cache.get(rem, None)
        if triu is None:
            triu = np.triu_indices(rem, k=1)
            triu_cache[rem] = triu
        left_idx, right_idx = triu
        chunk_size = left_idx.shape[0]
        if chunk_size == 0:
            continue
        next_pos = write_pos + chunk_size
        out[write_pos:next_pos, 0] = unique_nodes[i]
        out[write_pos:next_pos, 1] = unique_nodes[int(i) + 1 + left_idx]
        out[write_pos:next_pos, 2] = unique_nodes[int(i) + 1 + right_idx]
        write_pos = next_pos
    return out


def _decode_arity3_encoded_to_nodes(unique_encoded, unique_nodes, num_nodes):
    unique_encoded = np.asarray(unique_encoded, dtype=np.int64).reshape(-1)
    if unique_encoded.shape[0] == 0:
        return np.zeros(shape=(0, 3), dtype=np.int64)
    unique_nodes = np.asarray(unique_nodes, dtype=np.int64).reshape(-1)
    num_nodes = np.int64(num_nodes)
    base2 = num_nodes * num_nodes
    i0 = unique_encoded // base2
    rem = unique_encoded % base2
    i1 = rem // num_nodes
    i2 = rem % num_nodes
    out = np.empty(shape=(unique_encoded.shape[0], 3), dtype=np.int64)
    out[:, 0] = unique_nodes[i0]
    out[:, 1] = unique_nodes[i1]
    out[:, 2] = unique_nodes[i2]
    return out


def _generate_union_candidates_arity3_from_pairs(pair_nodes, pair_nodes_are_sorted_unique=False):
    pair_nodes = np.asarray(pair_nodes, dtype=np.int64)
    if pair_nodes.ndim != 2 or pair_nodes.shape[1] != 2:
        raise ValueError('pair_nodes should be a 2D array with exactly two columns.')
    if pair_nodes.shape[0] < 2:
        return np.zeros(shape=(0, 3), dtype=np.int64)
    if pair_nodes_are_sorted_unique:
        if np.any(pair_nodes[:, 0] == pair_nodes[:, 1]):
            return _generate_valid_unions_by_pair_scan(pair_nodes, arity=3)
    else:
        pair_nodes = np.sort(pair_nodes, axis=1)
        # Keep exact set-union semantics for degenerate rows with duplicates.
        if np.any(pair_nodes[:, 0] == pair_nodes[:, 1]):
            return _generate_valid_unions_by_pair_scan(pair_nodes, arity=3)
        pair_nodes = _unique_rows_int64(pair_nodes)
        if pair_nodes.shape[0] < 2:
            return np.zeros(shape=(0, 3), dtype=np.int64)

    unique_nodes = _unique_sorted_int64_1d(pair_nodes.reshape(-1))
    num_nodes = np.int64(unique_nodes.shape[0])
    expected_pairs = (num_nodes * (num_nodes - 1)) // 2
    if pair_nodes.shape[0] == expected_pairs:
        return _generate_all_triples_from_sorted_nodes(unique_nodes)
    remapped_pairs = np.searchsorted(unique_nodes, pair_nodes).astype(np.int64, copy=False)
    if int(num_nodes) <= int(_ARITY3_DENSE_EDGE_LOOKUP_MAX_NODES):
        if combination_cy is not None:
            cython_fn = getattr(combination_cy, 'generate_union_encoded_arity3_dense_int64', None)
            if cython_fn is not None:
                try:
                    cython_encoded = np.asarray(cython_fn(remapped_pairs, int(num_nodes)), dtype=np.int64)
                    if cython_encoded.shape[0] == 0:
                        return np.zeros(shape=(0, 3), dtype=np.int64)
                    cython_encoded.sort()
                    return _decode_arity3_encoded_to_nodes(
                        unique_encoded=cython_encoded,
                        unique_nodes=unique_nodes,
                        num_nodes=num_nodes,
                    )
                except Exception:
                    pass
    left = remapped_pairs[:, 0]
    right = remapped_pairs[:, 1]
    dense_edge_lookup = None
    sorted_edge_keys = None
    if int(num_nodes) <= int(_ARITY3_DENSE_EDGE_LOOKUP_MAX_NODES):
        dense_edge_lookup = np.zeros(shape=(int(num_nodes), int(num_nodes)), dtype=np.uint8)
        dense_edge_lookup[left, right] = 1
        dense_edge_lookup[right, left] = 1
    else:
        edge_keys = (left * num_nodes) + right
        sorted_edge_keys = np.sort(edge_keys)
    centers = np.concatenate((left, right), axis=0)
    neighbors = np.concatenate((right, left), axis=0)
    order = np.argsort(centers, kind='mergesort')
    centers = centers[order]
    neighbors = neighbors[order]

    starts = np.r_[0, np.flatnonzero(centers[1:] != centers[:-1]) + 1]
    ends = np.r_[starts[1:], centers.shape[0]]
    encoded_chunks = list()
    base2 = num_nodes * num_nodes
    triu_cache = dict()
    for start, end in zip(starts.tolist(), ends.tolist()):
        degree = end - start
        if degree < 2:
            continue
        local_neighbors = neighbors[start:end]
        triu = triu_cache.get(degree, None)
        if triu is None:
            triu = np.triu_indices(degree, k=1)
            triu_cache[degree] = triu
        left_idx, right_idx = triu
        if left_idx.shape[0] == 0:
            continue
        center_val = np.int64(centers[start])
        left_vals = local_neighbors[left_idx]
        right_vals = local_neighbors[right_idx]
        uv_min = np.minimum(left_vals, right_vals)
        uv_max = np.maximum(left_vals, right_vals)
        # Triangle-aware canonicalization:
        # If uv edge exists, keep only the smallest center among the triangle.
        if dense_edge_lookup is not None:
            uv_exists = (dense_edge_lookup[uv_min, uv_max] != 0)
        else:
            uv_edge_keys = (uv_min * num_nodes) + uv_max
            uv_pos = np.searchsorted(sorted_edge_keys, uv_edge_keys)
            uv_exists = np.zeros(shape=uv_edge_keys.shape, dtype=bool)
            in_bounds = (uv_pos < sorted_edge_keys.shape[0])
            uv_exists[in_bounds] = (
                sorted_edge_keys[uv_pos[in_bounds]] == uv_edge_keys[in_bounds]
            )
        keep = (~uv_exists) | (center_val < uv_min)
        if not np.any(keep):
            continue
        uv_min = uv_min[keep]
        uv_max = uv_max[keep]
        low = np.minimum(uv_min, center_val)
        high = np.maximum(uv_max, center_val)
        mid = uv_min + uv_max + center_val - low - high
        encoded = ((low * num_nodes) + mid) * num_nodes + high
        encoded_chunks.append(encoded)
    if len(encoded_chunks) == 0:
        return np.zeros(shape=(0, 3), dtype=np.int64)

    unique_encoded = np.concatenate(encoded_chunks, axis=0)
    unique_encoded.sort()
    return _decode_arity3_encoded_to_nodes(
        unique_encoded=unique_encoded,
        unique_nodes=unique_nodes,
        num_nodes=num_nodes,
    )


def _generate_union_candidates_arity4_from_triples(triple_nodes, triple_nodes_are_sorted_unique=False):
    triple_nodes = np.asarray(triple_nodes, dtype=np.int64)
    if triple_nodes.ndim != 2 or triple_nodes.shape[1] != 3:
        raise ValueError('triple_nodes should be a 2D array with exactly three columns.')
    if triple_nodes.shape[0] < 2:
        return np.zeros(shape=(0, 4), dtype=np.int64)
    if triple_nodes_are_sorted_unique:
        if np.any(np.diff(triple_nodes, axis=1) == 0):
            return _generate_valid_unions_by_pair_scan(triple_nodes, arity=4)
    else:
        triple_nodes = np.sort(triple_nodes, axis=1)
        if np.any(np.diff(triple_nodes, axis=1) == 0):
            return _generate_valid_unions_by_pair_scan(triple_nodes, arity=4)
        triple_nodes = _unique_rows_int64(triple_nodes)
        if triple_nodes.shape[0] < 2:
            return np.zeros(shape=(0, 4), dtype=np.int64)
    if combination_cy is not None:
        cython_fn = getattr(combination_cy, 'generate_union_candidates_arity4_from_triples_int64', None)
        if cython_fn is not None:
            try:
                cy_out = np.asarray(cython_fn(triple_nodes), dtype=np.int64)
                if cy_out.shape[0] == 0:
                    return np.zeros(shape=(0, 4), dtype=np.int64)
                return _unique_rows_int64(cy_out)
            except Exception:
                pass
    return _generate_union_candidates_by_shared_subset_python_dict(
        sorted_nodes=triple_nodes,
        arity=4,
    )


def _generate_union_candidates_by_shared_subset_cython(sorted_nodes, arity):
    if combination_cy is not None:
        cython_fn = getattr(combination_cy, 'generate_union_candidates_shared_subset_int64', None)
        if cython_fn is not None:
            try:
                cy_out = np.asarray(cython_fn(sorted_nodes), dtype=np.int64)
                if cy_out.ndim != 2 or cy_out.shape[1] != int(arity):
                    raise ValueError('Unexpected Cython output shape for shared-subset unions.')
                if cy_out.shape[0] == 0:
                    return np.zeros(shape=(0, arity), dtype=np.int64)
                return _unique_rows_int64(cy_out)
            except Exception:
                pass
    return _generate_union_candidates_by_shared_subset_python_dict(
        sorted_nodes=sorted_nodes,
        arity=arity,
    )


def _generate_union_candidates_by_shared_subset_python_dict(sorted_nodes, arity):
    width = int(sorted_nodes.shape[1])
    key_to_values = defaultdict(set)
    if width == 3:
        for row in sorted_nodes:
            a, b, c = row.tolist()
            key_to_values[(b, c)].add(a)
            key_to_values[(a, c)].add(b)
            key_to_values[(a, b)].add(c)
    elif width == 4:
        for row in sorted_nodes:
            a, b, c, d = row.tolist()
            key_to_values[(b, c, d)].add(a)
            key_to_values[(a, c, d)].add(b)
            key_to_values[(a, b, d)].add(c)
            key_to_values[(a, b, c)].add(d)
    else:
        for row in sorted_nodes:
            row_tuple = tuple(row.tolist())
            for drop in range(width):
                key = row_tuple[:drop] + row_tuple[(drop + 1):]
                key_to_values[key].add(row_tuple[drop])

    generated = list()
    key_len = width - 1
    for key, values in key_to_values.items():
        if len(values) < 2:
            continue
        unique_values = np.asarray(sorted(values), dtype=np.int64)
        left, right = np.triu_indices(unique_values.shape[0], k=1)
        if left.shape[0] == 0:
            continue
        chunk = np.empty(shape=(left.shape[0], arity), dtype=np.int64)
        if key_len > 0:
            chunk[:, :key_len] = np.asarray(key, dtype=np.int64)
        chunk[:, key_len] = unique_values[left]
        chunk[:, key_len + 1] = unique_values[right]
        chunk.sort(axis=1)
        generated.append(chunk)
    if len(generated) == 0:
        return np.zeros(shape=(0, arity), dtype=np.int64)
    return _unique_rows_int64(np.concatenate(generated, axis=0))


def _generate_union_candidates_by_shared_subset_grouped(sorted_nodes, arity):
    sorted_nodes = np.asarray(sorted_nodes, dtype=np.int64)
    width = int(sorted_nodes.shape[1])
    key_len = width - 1
    key_blocks = list()
    val_blocks = list()
    for drop in range(width):
        if drop == 0:
            key_blocks.append(sorted_nodes[:, 1:])
        elif drop == (width - 1):
            key_blocks.append(sorted_nodes[:, :(-1)])
        else:
            key_blocks.append(np.concatenate((sorted_nodes[:, :drop], sorted_nodes[:, (drop + 1):]), axis=1))
        val_blocks.append(sorted_nodes[:, drop])
    all_keys = np.concatenate(key_blocks, axis=0)
    all_vals = np.concatenate(val_blocks, axis=0)

    if key_len == 1:
        order = np.argsort(all_keys[:, 0], kind='mergesort')
        sorted_keys = all_keys[order]
        sorted_vals = all_vals[order]
        starts = np.r_[0, np.flatnonzero(sorted_keys[1:, 0] != sorted_keys[:-1, 0]) + 1]
    else:
        order = np.lexsort(all_keys.T[::-1])
        sorted_keys = all_keys[order, :]
        sorted_vals = all_vals[order]
        starts = np.r_[0, np.flatnonzero(np.any(sorted_keys[1:, :] != sorted_keys[:-1, :], axis=1)) + 1]

    generated = list()
    triu_cache = dict()
    ends = np.r_[starts[1:], sorted_keys.shape[0]]
    for start, end in zip(starts.tolist(), ends.tolist()):
        unique_vals = np.unique(sorted_vals[start:end])
        group_size = int(unique_vals.shape[0])
        if group_size < 2:
            continue
        triu = triu_cache.get(group_size, None)
        if triu is None:
            triu = np.triu_indices(group_size, k=1)
            triu_cache[group_size] = triu
        left, right = triu
        if left.shape[0] == 0:
            continue
        chunk = np.empty(shape=(left.shape[0], arity), dtype=np.int64)
        chunk[:, :key_len] = sorted_keys[start, :]
        chunk[:, key_len] = unique_vals[left]
        chunk[:, key_len + 1] = unique_vals[right]
        chunk.sort(axis=1)
        generated.append(chunk)
    if len(generated) == 0:
        return np.zeros(shape=(0, arity), dtype=np.int64)
    return _unique_rows_int64(np.concatenate(generated, axis=0))


def _generate_union_candidates_by_shared_subset(target_nodes, arity):
    target_nodes = np.asarray(target_nodes, dtype=np.int64)
    if target_nodes.ndim != 2:
        raise ValueError('target_nodes should be a 2D array.')
    if target_nodes.shape[0] < 2:
        return np.zeros(shape=(0, arity), dtype=np.int64)
    width = int(target_nodes.shape[1])
    if width + 1 != int(arity):
        raise ValueError('target_nodes width should be arity - 1.')
    sorted_nodes = np.sort(target_nodes, axis=1)
    if width > 1:
        # Keep exact set-union semantics for degenerate rows with duplicates.
        if np.any(np.diff(sorted_nodes, axis=1) == 0):
            return _generate_valid_unions_by_pair_scan(sorted_nodes, arity=arity)
    sorted_nodes = _unique_rows_int64(sorted_nodes)
    if sorted_nodes.shape[0] < 2:
        return np.zeros(shape=(0, arity), dtype=np.int64)
    if width >= 3:
        is_complete_family, family_nodes = _is_complete_subset_family(sorted_nodes)
        if is_complete_family:
            return _generate_all_k_combinations_from_sorted_nodes(family_nodes, arity)
    if width == 1:
        return _pairwise_node_combinations(sorted_nodes[:, 0])
    if width == 2:
        return _generate_union_candidates_arity3_from_pairs(
            pair_nodes=sorted_nodes,
            pair_nodes_are_sorted_unique=True,
        )
    if width == 3:
        return _generate_union_candidates_arity4_from_triples(
            triple_nodes=sorted_nodes,
            triple_nodes_are_sorted_unique=True,
        )
    return _generate_union_candidates_by_shared_subset_cython(
        sorted_nodes=sorted_nodes,
        arity=arity,
    )


def _get_dependency_row_indices(g, all_node_ids, node_id_to_row, trait_names):
    cache = g.get('_combination_dependency_row_cache', None)
    if cache is not None:
        cached_ids = cache.get('all_node_ids', None)
        cached_trait_names = cache.get('trait_names', None)
        if (
            (cached_ids is not None)
            and np.array_equal(cached_ids, all_node_ids)
            and (cached_trait_names == tuple(trait_names))
            and (cache.get('dep_ids_obj', None) is g.get('dep_ids'))
            and (cache.get('fg_dep_ids_obj', None) is g.get('fg_dep_ids'))
        ):
            return cache['dep_indices'], cache['fg_dep_indices']
    dep_indices = []
    for dep_id in g['dep_ids']:
        dep_indices.append(
            _map_node_ids_to_rows(
                node_ids=dep_id,
                id_to_row=node_id_to_row,
                context='Dependency groups',
            )
        )
    fg_dep_indices = dict()
    for trait_name in trait_names:
        trait_indices = []
        for fg_dep_id in g['fg_dep_ids'][trait_name]:
            trait_indices.append(
                _map_node_ids_to_rows(
                    node_ids=fg_dep_id,
                    id_to_row=node_id_to_row,
                    context='Foreground dependency groups',
                )
            )
        fg_dep_indices[trait_name] = trait_indices
    g['_combination_dependency_row_cache'] = {
        'all_node_ids': np.array(all_node_ids, dtype=np.int64, copy=True),
        'trait_names': tuple(trait_names),
        'dep_ids_obj': g.get('dep_ids'),
        'fg_dep_ids_obj': g.get('fg_dep_ids'),
        'dep_indices': dep_indices,
        'fg_dep_indices': fg_dep_indices,
    }
    return dep_indices, fg_dep_indices


def node_union(index_combinations, target_nodes, df_mmap, mmap_start):
    arity = target_nodes.shape[1] + 1
    i = mmap_start
    for ic in index_combinations:
        node_union = np.union1d(target_nodes[ic[0], :], target_nodes[ic[1], :])
        if (node_union.shape[0] == arity):
            df_mmap[i, :] = node_union
            i += 1

def _resolve_combination_step_n_jobs(g, num_items, min_items_key, min_items_per_job_key, default_min_items, default_min_items_per_job):
    return parallel.resolve_adaptive_n_jobs(
        num_items=num_items,
        threads=int(g.get('threads', 1)),
        min_items_for_parallel=int(g.get(min_items_key, default_min_items)),
        min_items_per_job=int(g.get(min_items_per_job_key, default_min_items_per_job)),
    )


def _generate_unions_from_trait_rows(trait_rows, arity):
    return _generate_union_candidates_by_shared_subset(
        target_nodes=trait_rows,
        arity=arity,
    )


def _generate_trait_unions_parallel(g, trait_row_items, arity):
    if len(trait_row_items) == 0:
        return dict()
    if len(trait_row_items) == 1:
        trait_name, trait_rows = trait_row_items[0]
        return {trait_name: _generate_unions_from_trait_rows(trait_rows=trait_rows, arity=arity)}
    if combination_cy is None:
        n_jobs = 1
    else:
        n_jobs = _resolve_combination_step_n_jobs(
            g=g,
            num_items=len(trait_row_items),
            min_items_key='parallel_min_items_trait_unions',
            min_items_per_job_key='parallel_min_items_per_job_trait_unions',
            default_min_items=2,
            default_min_items_per_job=1,
        )
    args_iterable = [(trait_rows, arity) for _, trait_rows in trait_row_items]
    if n_jobs == 1:
        unions = [
            _generate_unions_from_trait_rows(trait_rows=trait_rows, arity=arity_local)
            for trait_rows, arity_local in args_iterable
        ]
    else:
        unions = parallel.run_starmap(
            func=_generate_unions_from_trait_rows,
            args_iterable=args_iterable,
            n_jobs=n_jobs,
            backend='threading',
        )
    return {trait_name: union for (trait_name, _), union in zip(trait_row_items, unions)}


def _map_node_combinations_to_rows_vectorized(node_combinations, sorted_node_ids, sorted_row_ids):
    node_combinations = np.asarray(node_combinations, dtype=np.int64)
    if node_combinations.size == 0:
        return np.zeros(shape=node_combinations.shape, dtype=np.int64)
    flat_ids = node_combinations.reshape(-1)
    positions = np.searchsorted(sorted_node_ids, flat_ids)
    is_in_bounds = (positions >= 0) & (positions < sorted_node_ids.shape[0])
    if not np.all(is_in_bounds):
        raise ValueError('Node combinations contain node IDs that are not present in the tree.')
    is_exact_match = (sorted_node_ids[positions] == flat_ids)
    if not np.all(is_exact_match):
        raise ValueError('Node combinations contain node IDs that are not present in the tree.')
    mapped_rows = sorted_row_ids[positions]
    return mapped_rows.reshape(node_combinations.shape)


def _map_node_combination_chunk_to_matrix_indices(node_chunk, start_col, sorted_node_ids, sorted_row_ids):
    if node_chunk.shape[0] == 0:
        return np.zeros(shape=(0,), dtype=np.int64), np.zeros(shape=(0,), dtype=np.int64)
    mapped_rows = _map_node_combinations_to_rows_vectorized(
        node_combinations=node_chunk,
        sorted_node_ids=sorted_node_ids,
        sorted_row_ids=sorted_row_ids,
    )
    arity = int(node_chunk.shape[1])
    col_ids = np.repeat(np.arange(start_col, start_col + node_chunk.shape[0], dtype=np.int64), arity)
    row_ids = mapped_rows.reshape(-1)
    return row_ids, col_ids


def _populate_nc_matrix(nc_matrix, node_combinations, all_node_ids, g):
    num_combinations = int(node_combinations.shape[0])
    if num_combinations == 0:
        return None
    all_node_ids = np.asarray(all_node_ids, dtype=np.int64).reshape(-1)
    sorted_order = np.argsort(all_node_ids)
    sorted_node_ids = all_node_ids[sorted_order]
    sorted_row_ids = sorted_order.astype(np.int64, copy=False)
    n_jobs = _resolve_combination_step_n_jobs(
        g=g,
        num_items=num_combinations,
        min_items_key='parallel_min_items_nc_matrix',
        min_items_per_job_key='parallel_min_items_per_job_nc_matrix',
        default_min_items=100000,
        default_min_items_per_job=25000,
    )
    if n_jobs == 1:
        row_ids, col_ids = _map_node_combination_chunk_to_matrix_indices(
            node_chunk=node_combinations,
            start_col=0,
            sorted_node_ids=sorted_node_ids,
            sorted_row_ids=sorted_row_ids,
        )
        nc_matrix[row_ids, col_ids] = True
        return None
    chunk_factor = parallel.resolve_chunk_factor(g=g, task='general')
    chunks, starts = parallel.get_chunks(node_combinations, n_jobs, chunk_factor=chunk_factor)
    tasks = [(chunk, start, sorted_node_ids, sorted_row_ids) for chunk, start in zip(chunks, starts)]
    out = parallel.run_starmap(
        func=_map_node_combination_chunk_to_matrix_indices,
        args_iterable=tasks,
        n_jobs=n_jobs,
        backend='threading',
    )
    for row_ids, col_ids in out:
        if row_ids.shape[0] == 0:
            continue
        nc_matrix[row_ids, col_ids] = True
    return None


def _mark_dependent_row_combinations(row_combinations, dep_row_groups):
    row_combinations = np.asarray(row_combinations, dtype=np.int64)
    if row_combinations.shape[0] == 0:
        return np.zeros(shape=(0,), dtype=bool)
    is_dependent = np.zeros(shape=(row_combinations.shape[0],), dtype=bool)
    for dep_rows in dep_row_groups:
        dep_rows = np.asarray(dep_rows, dtype=np.int64).reshape(-1)
        if dep_rows.shape[0] < 2:
            continue
        if dep_rows.shape[0] == 2:
            dep_a = int(dep_rows[0])
            dep_b = int(dep_rows[1])
            has_a = np.any(row_combinations == dep_a, axis=1)
            has_b = np.any(row_combinations == dep_b, axis=1)
            is_dependent |= (has_a & has_b)
            continue
        is_dependent |= (np.isin(row_combinations, dep_rows).sum(axis=1) > 1)
    return is_dependent


def nc_matrix2id_combinations(
    nc_matrix,
    arity,
    ncpu,
    min_items_for_parallel=0,
    min_items_per_job=1,
):
    # Parallel reconstruction through Python callbacks is slower than direct vectorized decoding.
    col_sums = np.asarray(nc_matrix.sum(axis=0)).reshape(-1)
    valid_cols = np.flatnonzero(col_sums == arity)
    if valid_cols.shape[0] == 0:
        return np.zeros(shape=(0, arity), dtype=np.int64)
    valid_view = np.asarray(nc_matrix[:, valid_cols], dtype=bool).T
    flat_row_major = np.flatnonzero(valid_view)
    expected = int(valid_cols.shape[0]) * int(arity)
    if flat_row_major.shape[0] != expected:
        raise ValueError('nc_matrix contains columns with invalid arity during combination decoding.')
    row_ids = flat_row_major % int(nc_matrix.shape[0])
    return row_ids.reshape(valid_cols.shape[0], arity).astype(np.int64, copy=False)

def get_node_combinations(g, target_id_dict=None, cb_passed=None, exhaustive=False, cb_all=False, arity=2,
                          check_attr=None, verbose=True):
    if sum([target_id_dict is not None, cb_passed is not None, exhaustive])!=1:
        raise ValueError('Only one of target_id_dict, cb_passed, or exhaustive must be set.')
    g['fg_dependent_id_combinations'] = dict()
    tree = g['tree']
    all_nodes = [node for node in tree.traverse() if not ete.is_root(node)]
    all_node_ids, node_id_to_row = _build_node_id_to_row([ete.get_prop(node, "numerical_label") for node in all_nodes])
    all_node_ids = np.asarray(all_node_ids, dtype=np.int64).reshape(-1)
    row_ids_are_node_ids = np.array_equal(
        all_node_ids,
        np.arange(all_node_ids.shape[0], dtype=np.int64),
    )
    sorted_order = np.argsort(all_node_ids)
    sorted_node_ids = all_node_ids[sorted_order]
    sorted_row_ids = sorted_order.astype(np.int64, copy=False)
    if verbose:
        print("Number of all branches: {:,}".format(len(all_nodes)), flush=True)
    row_combinations = None
    if exhaustive:
        target_nodes = list()
        for node in all_nodes:
            if (check_attr is None)|(check_attr in dir(node)):
                target_nodes.append(ete.get_prop(node, "numerical_label"))
        target_nodes = np.array(target_nodes)
        node_combinations = list(itertools.combinations(target_nodes, arity))
        node_combinations = [ set(nc) for nc in node_combinations ]
        node_combinations = np.array([ list(nc) for nc in node_combinations ])
        row_combinations = _map_node_combinations_to_rows_vectorized(
            node_combinations=node_combinations,
            sorted_node_ids=sorted_node_ids,
            sorted_row_ids=sorted_row_ids,
        )
    if target_id_dict is not None:
        trait_names = list(target_id_dict.keys())
        node_combination_dict = dict()
        trait_union_items = list()
        is_all_trait_no_branch_combination = True
        for trait_name in trait_names:
            trait_target_nodes = np.asarray(target_id_dict[trait_name])
            if trait_target_nodes.ndim <= 1:
                trait_target_nodes = np.expand_dims(_normalize_node_ids(trait_target_nodes), axis=1)
            elif trait_target_nodes.ndim == 2:
                original_shape = trait_target_nodes.shape
                trait_target_nodes = _normalize_node_ids(trait_target_nodes.reshape(-1)).reshape(original_shape)
            else:
                raise ValueError('target_id_dict values should be 1D or 2D arrays.')
            trait_target_rows = _map_node_combinations_to_rows_vectorized(
                node_combinations=trait_target_nodes,
                sorted_node_ids=sorted_node_ids,
                sorted_row_ids=sorted_row_ids,
            )
            pair_count = trait_target_nodes.shape[0] * (trait_target_nodes.shape[0] - 1) // 2
            if (arity == 2) and (trait_target_nodes.shape[1] == 1):
                if pair_count > 0:
                    is_all_trait_no_branch_combination = False
                    pairwise = _pairwise_node_combinations(trait_target_rows[:, 0])
                    if verbose:
                        txt = 'Number of branch combinations before independency check for {}: {:,}'
                        print(txt.format(trait_name, pairwise.shape[0]), flush=True)
                    node_combination_dict[trait_name] = pairwise
                else:
                    if verbose:
                        txt = 'There is no target branch combination for {} at K = {:,}.\n'
                        sys.stderr.write(txt.format(trait_name, arity))
                    node_combination_dict[trait_name] = np.zeros(shape=[0, arity], dtype=np.int64)
                continue
            if pair_count > 0:
                is_all_trait_no_branch_combination = False
                if verbose:
                    txt = 'Number of branch combinations before independency check for {}: {:,}'
                    print(txt.format(trait_name, pair_count), flush=True)
            else:
                if verbose:
                    txt = 'There is no target branch combination for {} at K = {:,}.\n'
                    sys.stderr.write(txt.format(trait_name, arity))
                continue
            trait_union_items.append((trait_name, trait_target_rows))
        node_combination_dict.update(
            _generate_trait_unions_parallel(
                g=g,
                trait_row_items=trait_union_items,
                arity=arity,
            )
        )
        if is_all_trait_no_branch_combination:
            txt = 'There is no target branch combination for all traits at K = {:,}.\n'
            sys.stderr.write(txt.format(arity))
            id_combinations = np.zeros(shape=[0, arity], dtype=np.int64)
            return g, id_combinations
        if len(node_combination_dict) == 1:
            row_combinations = list(node_combination_dict.values())[0]
        else:
            row_combinations = _unique_rows_int64(np.concatenate(list(node_combination_dict.values()), axis=0))
    if cb_passed is not None:
        node_combinations_dict = dict()
        if cb_all:
            trait_names = ['all',]
        else:
            trait_names = g['fg_df'].columns[1:len(g['fg_df'].columns)].tolist()
        bid_cols = cb_passed.columns[cb_passed.columns.str.startswith('branch_id_')].tolist()
        bid_matrix = cb_passed.loc[:, bid_cols].to_numpy(copy=False)
        if cb_all:
            cb_all_mask = np.ones(shape=(cb_passed.shape[0],), dtype=bool)
            trait_mask_dict = None
        else:
            cb_all_mask = None
            if len(trait_names) == 1:
                trait_name = trait_names[0]
                is_fg = (cb_passed.loc[:, 'is_fg_' + trait_name].to_numpy(copy=False) == 'Y')
                is_mf = (cb_passed.loc[:, 'is_mf_' + trait_name].to_numpy(copy=False) == 'Y')
                is_mg = (cb_passed.loc[:, 'is_mg_' + trait_name].to_numpy(copy=False) == 'Y')
                trait_mask_dict = {trait_name: (is_fg | is_mf | is_mg)}
            else:
                trait_mask_dict = dict()
                for trait_name in trait_names:
                    is_fg = (cb_passed.loc[:, 'is_fg_' + trait_name].to_numpy(copy=False) == 'Y')
                    is_mf = (cb_passed.loc[:, 'is_mf_' + trait_name].to_numpy(copy=False) == 'Y')
                    is_mg = (cb_passed.loc[:, 'is_mg_' + trait_name].to_numpy(copy=False) == 'Y')
                    trait_mask_dict[trait_name] = (is_fg | is_mf | is_mg)
        trait_union_items = list()
        is_all_trait_no_branch_combination = True
        for trait_name in trait_names:
            if cb_all:
                is_trait = cb_all_mask
            else:
                is_trait = trait_mask_dict[trait_name]
            bid_trait = bid_matrix[is_trait, :]
            if bid_trait.size > 0:
                bid_trait = _normalize_node_ids(bid_trait.reshape(-1)).reshape(bid_trait.shape)
            else:
                bid_trait = np.zeros(shape=(0, len(bid_cols)), dtype=np.int64)
            bid_trait_rows = _map_node_combinations_to_rows_vectorized(
                node_combinations=bid_trait,
                sorted_node_ids=sorted_node_ids,
                sorted_row_ids=sorted_row_ids,
            )
            pair_count = bid_trait.shape[0] * (bid_trait.shape[0] - 1) // 2
            if (arity == 2) and (bid_trait.shape[1] == 1):
                if pair_count > 0:
                    is_all_trait_no_branch_combination = False
                    pairwise = _pairwise_node_combinations(bid_trait_rows[:, 0])
                    if verbose:
                        txt = 'Number of redundant branch combination unions for {}: {:,}'
                        print(txt.format(trait_name, pairwise.shape[0]), flush=True)
                    node_combinations_dict[trait_name] = pairwise
                else:
                    txt = 'There is no target branch combination for {} at K = {:,}.\n'
                    sys.stderr.write(txt.format(trait_name, arity))
                continue
            if pair_count > 0:
                is_all_trait_no_branch_combination = False
            else:
                txt = 'There is no target branch combination for {} at K = {:,}.\n'
                sys.stderr.write(txt.format(trait_name, arity))
                continue
            if verbose:
                txt = 'Number of redundant branch combination unions for {}: {:,}'
                print(txt.format(trait_name, pair_count), flush=True)
            trait_union_items.append((trait_name, bid_trait_rows))
        node_combinations_dict.update(
            _generate_trait_unions_parallel(
                g=g,
                trait_row_items=trait_union_items,
                arity=arity,
            )
        )
        if is_all_trait_no_branch_combination:
            txt = 'There is no target branch combination for all traits at K = {:,}.\n'
            sys.stderr.write(txt.format(arity))
            id_combinations = np.zeros(shape=[0, arity], dtype=np.int64)
            return g, id_combinations
        if len(node_combinations_dict) == 1:
            row_combinations = list(node_combinations_dict.values())[0]
        else:
            row_combinations = _unique_rows_int64(np.concatenate(list(node_combinations_dict.values()), axis=0))
    if row_combinations is None:
        row_combinations = np.zeros(shape=(0, arity), dtype=np.int64)
    if verbose:
        print("Number of all branch combinations before independency check: {:,}".format(row_combinations.shape[0]), flush=True)
    trait_names = g['fg_df'].columns[1:len(g['fg_df'].columns)].tolist()
    dep_indices, fg_dep_indices = _get_dependency_row_indices(
        g=g,
        all_node_ids=all_node_ids,
        node_id_to_row=node_id_to_row,
        trait_names=trait_names,
    )
    is_dependent_col = _mark_dependent_row_combinations(
        row_combinations=row_combinations,
        dep_row_groups=dep_indices,
    )
    if verbose:
        print('Number of non-independent branch combinations to be removed: {:,}'.format(is_dependent_col.sum()), flush=True)
    independent_row_combinations = row_combinations[~is_dependent_col, :]
    id_combinations = np.zeros(shape=(0,arity), dtype=np.int64)
    start = time.time()
    for trait_name in trait_names:
        is_fg_dependent_col = _mark_dependent_row_combinations(
            row_combinations=independent_row_combinations,
            dep_row_groups=fg_dep_indices[trait_name],
        )
        if (g['exhaustive_until']>=arity):
            if verbose:
                txt = 'Number of non-independent foreground branch combinations to be non-foreground-marked for {}: {:,} / {:,}'
                print(txt.format(trait_name, is_fg_dependent_col.sum(), is_fg_dependent_col.shape[0]), flush=True)
            g['fg_dependent_id_combinations'][trait_name] = _map_row_combinations_to_node_ids(
                row_combinations=independent_row_combinations[is_fg_dependent_col, :],
                node_ids=all_node_ids,
                row_ids_are_node_ids=row_ids_are_node_ids,
            )
            if trait_name == trait_names[0]:
                id_combinations = _map_row_combinations_to_node_ids(
                    row_combinations=independent_row_combinations,
                    node_ids=all_node_ids,
                    row_ids_are_node_ids=row_ids_are_node_ids,
                )
        else:
            if verbose and (is_fg_dependent_col.sum() > 0):
                txt = 'Removing {:,} (out of {:,}) non-independent foreground branch combinations for {}.'
                print(txt.format(is_fg_dependent_col.sum(), is_fg_dependent_col.shape[0], trait_name), flush=True)
            g['fg_dependent_id_combinations'][trait_name] = np.array([])
            trait_id_combinations = _map_row_combinations_to_node_ids(
                row_combinations=independent_row_combinations[~is_fg_dependent_col, :],
                node_ids=all_node_ids,
                row_ids_are_node_ids=row_ids_are_node_ids,
            )
            if id_combinations.shape[0] == 0:
                id_combinations = trait_id_combinations
            else:
                id_combinations = _unique_rows_int64(np.concatenate((id_combinations, trait_id_combinations), axis=0))
    if verbose:
        print('Time elapsed for generating branch combinations: {:,} sec'.format(int(time.time() - start)))
        print("Number of independent branch combinations to be analyzed: {:,}".format(id_combinations.shape[0]), flush=True)
    return g,id_combinations

def node_combination_subsamples_rifle(g, arity, rep):
    sub_ids = set(_normalize_node_ids(g['sub_branches']).tolist())
    if (rep <= 0) or (len(sub_ids) < arity):
        return np.zeros(shape=(0, arity), dtype=np.int64)

    dep_lookup = dict()
    for dep_group in g['dep_ids']:
        dep_group_set = set([int(x) for x in _normalize_node_ids(dep_group).tolist()])
        if len(dep_group_set) == 0:
            continue
        for node_id in dep_group_set:
            dep_lookup.setdefault(node_id, set()).update(dep_group_set)

    max_consecutive_fail = max(int(rep) * 10, 100)
    consecutive_fail = 0
    selected_combinations = list()
    seen_combinations = set()
    while (len(selected_combinations) < rep) and (consecutive_fail < max_consecutive_fail):
        selected_ids = set()
        unavailable_ids = set()
        for _ in np.arange(arity):
            available_ids = sub_ids.difference(unavailable_ids)
            if len(available_ids) == 0:
                consecutive_fail += 1
                break
            selected_id = int(np.random.choice(list(available_ids), 1)[0])
            selected_ids.add(selected_id)
            unavailable_ids |= dep_lookup.get(selected_id, {selected_id})
        if len(selected_ids) != arity:
            continue
        selected_tuple = tuple(sorted(selected_ids))
        if selected_tuple in seen_combinations:
            consecutive_fail += 1
            continue
        seen_combinations.add(selected_tuple)
        selected_combinations.append(selected_tuple)
        consecutive_fail = 0

    if (len(selected_combinations) == 0) and (rep > 0):
        print('Node combination subsampling failed', str(rep), 'times. Exiting.')
    return np.asarray(selected_combinations, dtype=np.int64)

def node_combination_subsamples_shotgun(g, arity, rep):
    all_ids, node_id_to_row = _build_node_id_to_row([ete.get_prop(n, "numerical_label") for n in g['tree'].traverse()])
    sub_ids = _normalize_node_ids(g['sub_branches'])
    sub_rows = _map_node_ids_to_rows(
        node_ids=sub_ids,
        id_to_row=node_id_to_row,
        context='sub_branches',
    )
    if (rep <= 0) or (len(sub_ids) < arity):
        return np.zeros(shape=(0, arity), dtype=np.int64)
    id_combinations = np.zeros(shape=(0,arity), dtype=np.int64)
    id_combinations_dif = np.inf
    round = 1
    while (id_combinations.shape[0] < rep)&(id_combinations_dif > rep/200):
        ss_matrix = np.zeros(shape=(len(all_ids), rep), dtype=bool, order='C')
        for i in np.arange(rep):
            ind = np.random.choice(a=sub_rows, size=arity, replace=False)
            ss_matrix[ind,i] = 1
        is_dependent_col = np.zeros(shape=(ss_matrix.shape[1],), dtype=bool)
        for dep_id in g['dep_ids']:
            dep_indices = _map_node_ids_to_rows(
                node_ids=dep_id,
                id_to_row=node_id_to_row,
                context='Dependency groups',
            )
            if dep_indices.shape[0] == 0:
                continue
            is_dependent_col |= (ss_matrix[dep_indices, :].sum(axis=0) > 1)
        ss_matrix = ss_matrix[:,~is_dependent_col]
        rows,cols = np.where(ss_matrix==1)
        unique_cols = np.unique(cols)
        tmp_id_combinations = np.zeros(shape=(unique_cols.shape[0], arity), dtype=np.int64)
        for j, col in enumerate(unique_cols.tolist()):
            tmp_id_combinations[j,:] = all_ids[rows[cols==col]]
        previous_num = id_combinations.shape[0]
        id_combinations = np.concatenate((id_combinations, tmp_id_combinations), axis=0)
        id_combinations.sort(axis=1)
        id_combinations = pd.DataFrame(id_combinations).drop_duplicates().values
        id_combinations_dif = id_combinations.shape[0] - previous_num
        print('round', round,'# id_combinations =', id_combinations.shape[0], 'subsampling rate =', id_combinations_dif/rep)
        round += 1
    if id_combinations.shape[0] < rep:
        print('Inefficient subsampling. Exiting node_combinations_subsamples()')
        id_combinations = np.zeros(shape=(0, arity), dtype=np.int64)
    else:
        id_combinations = id_combinations[:rep,:]
    return id_combinations

def calc_substitution_patterns(cb):
    for key in ['S_sub','N_sub']:
        cols = cb.columns[cb.columns.str.startswith(key)].tolist()
        sub_patterns = cb.loc[:,cols]
        sub_patterns2 = pd.DataFrame(np.zeros(sub_patterns.shape), columns=cols)
        for i in np.arange(len(cols)):
            sub_patterns2.loc[:,cols[i]] = sub_patterns.apply(lambda x: np.sort(x)[i], axis=1)
        sub_patterns2.loc[:,'index2'] = np.arange(sub_patterns2.shape[0])
        sub_patterns3 = sub_patterns2.loc[:,cols].drop_duplicates()
        sp_min = int(sub_patterns3.sum(axis=1).min())
        sp_max = int(sub_patterns3.sum(axis=1).max())
        txt = 'Number of {} patterns among {:,} branch combinations={:,}, Min total subs={:,.1f}, Max total subs={:,.1f}'
        print(txt.format(key, cb.shape[0], sub_patterns3.shape[0], sp_min, sp_max), flush=True)
        sub_patterns3.loc[:,'sub_pattern_id'] = np.arange(sub_patterns3.shape[0])
        sub_patterns4 = pd.merge(sub_patterns2, sub_patterns3, on=cols, sort=False)
        sub_patterns4 = sub_patterns4.sort_values(axis=0, by='index2', ascending=True).reset_index()
        cb.loc[:,key+'_pattern_id'] = sub_patterns4.loc[:,'sub_pattern_id']
    return cb

def get_global_dep_ids(g):
    global_dep_ids = list()
    for leaf in ete.iter_leaves(g['tree']):
        ancestor_nns = [ete.get_prop(node, "numerical_label") for node in ete.iter_ancestors(leaf) if not ete.is_root(node)]
        dep_id = [ete.get_prop(leaf, "numerical_label"), ] + ancestor_nns
        dep_id = np.sort(np.array(dep_id))
        global_dep_ids.append(dep_id)
        if g['exclude_sister_pair']:
            for node in g['tree'].traverse():
                children = ete.get_children(node)
                if len(children)>1:
                    dep_id = np.sort(np.array([ ete.get_prop(node, "numerical_label") for node in children ]))
                    global_dep_ids.append(dep_id)
    root_nn = ete.get_prop(g['tree'], "numerical_label")
    root_state_sum = g['state_cdn'][root_nn, :, :].sum()
    if (root_state_sum == 0):
        print('Ancestral states were not estimated on the root node. Excluding sub-root nodes from the analysis.')
        subroot_nns = [ete.get_prop(node, "numerical_label") for node in ete.get_children(g['tree'])]
        for subroot_nn in subroot_nns:
            for node in g['tree'].traverse():
                if ete.is_root(node):
                    continue
                if subroot_nn == ete.get_prop(node, "numerical_label"):
                    continue
                ancestor_nns = [ete.get_prop(anc, "numerical_label") for anc in ete.iter_ancestors(node)]
                if subroot_nn in ancestor_nns:
                    continue
                global_dep_ids.append(np.array([subroot_nn, ete.get_prop(node, "numerical_label")]))
    return global_dep_ids

def get_foreground_dep_ids(g):
    fg_dep_ids = dict()
    for trait_name in g['fg_df'].columns[1:len(g['fg_df'].columns)]:
        if (g['foreground'] is not None)&(g['fg_exclude_wg']):
            fg_dep_ids[trait_name] = list()
            for i in np.arange(len(g['fg_leaf_names'][trait_name])):
                fg_lineage_leaf_names = g['fg_leaf_names'][trait_name][i]
                tmp_fg_dep_ids = list()
                for node in g['tree'].traverse():
                    if ete.is_root(node):
                        continue
                    is_all_leaf_lineage_fg = all([ln in fg_lineage_leaf_names for ln in ete.get_leaf_names(node)])
                    if not is_all_leaf_lineage_fg:
                        continue
                    is_up_all_leaf_lineage_fg = all([ln in fg_lineage_leaf_names for ln in ete.get_leaf_names(node.up)])
                    if is_up_all_leaf_lineage_fg:
                        continue
                    if ete.is_leaf(node):
                        tmp_fg_dep_ids.append(ete.get_prop(node, "numerical_label"))
                    else:
                        descendant_nn = [ ete.get_prop(n, "numerical_label") for n in ete.get_descendants(node) ]
                        tmp_fg_dep_ids += [ete.get_prop(node, "numerical_label"),] + descendant_nn
                if len(tmp_fg_dep_ids)>1:
                    fg_dep_ids[trait_name].append(np.sort(np.array(tmp_fg_dep_ids)))
            if (g['mg_sister'])|(g['mg_parent']):
                fg_dep_ids[trait_name].append(np.sort(np.array(g['mg_ids'][trait_name])))
        else:
            fg_dep_ids[trait_name] = np.array([])
    return fg_dep_ids

def get_dep_ids(g):
    g['dep_ids'] = get_global_dep_ids(g)
    g['fg_dep_ids'] = get_foreground_dep_ids(g)
    return g
