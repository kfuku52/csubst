import numpy as np
import pytest
import itertools

from csubst import combination


def test_pairwise_node_combinations_ignores_duplicate_ids():
    out = combination._pairwise_node_combinations([1, 1, 2])
    expected = np.array([[1, 2]], dtype=np.int64)
    np.testing.assert_array_equal(out, expected)


def test_unique_rows_int64_matches_numpy_unique_large():
    rng = np.random.default_rng(11)
    rows = rng.integers(0, 100, size=(8000, 5), dtype=np.int64)
    rows = np.sort(rows, axis=1)
    rows = np.concatenate((rows, rows[:2000]), axis=0)
    observed = combination._unique_rows_int64(rows, hash_threshold=100)
    expected = np.unique(rows, axis=0)
    np.testing.assert_array_equal(observed, expected)


def test_generate_all_k_combinations_from_sorted_nodes_matches_itertools():
    node_ids = np.array([2, 4, 7, 11, 13], dtype=np.int64)
    out = combination._generate_all_k_combinations_from_sorted_nodes(node_ids, k=4)
    expected = np.array(list(itertools.combinations(node_ids.tolist(), 4)), dtype=np.int64)
    np.testing.assert_array_equal(out, expected)


def test_generate_all_k_combinations_from_sorted_nodes_falls_back_when_cython_raises(monkeypatch):
    class _BrokenCy:
        @staticmethod
        def generate_all_k_combinations_from_sorted_nodes_int64(_unique_nodes, _k):
            raise RuntimeError("boom")

    monkeypatch.setattr(combination, "combination_cy", _BrokenCy)
    node_ids = np.array([2, 4, 7, 11, 13], dtype=np.int64)
    with pytest.warns(RuntimeWarning, match="generate_all_k_combinations"):
        out = combination._generate_all_k_combinations_from_sorted_nodes(node_ids, k=4)
    expected = np.array(list(itertools.combinations(node_ids.tolist(), 4)), dtype=np.int64)
    np.testing.assert_array_equal(out, expected)


def test_generate_union_candidates_by_shared_subset_matches_pair_scan():
    target_nodes = np.array(
        [
            [5, 1],
            [5, 3],
            [1, 3],
            [2, 4],
            [4, 5],
            [2, 5],
        ],
        dtype=np.int64,
    )
    out = combination._generate_union_candidates_by_shared_subset(target_nodes=target_nodes, arity=3)
    expected = combination._generate_valid_unions_by_pair_scan(target_nodes=target_nodes, arity=3)
    np.testing.assert_array_equal(out, expected)


def test_generate_union_candidates_arity3_from_pairs_matches_pair_scan_random():
    rng = np.random.default_rng(0)
    for _ in range(3):
        target_nodes = rng.integers(0, 30, size=(60, 2), dtype=np.int64)
        out = combination._generate_union_candidates_arity3_from_pairs(pair_nodes=target_nodes)
        expected = combination._generate_valid_unions_by_pair_scan(target_nodes=target_nodes, arity=3)
        np.testing.assert_array_equal(out, expected)


def test_generate_all_triples_from_sorted_nodes_matches_itertools():
    node_ids = np.array([2, 4, 7, 11], dtype=np.int64)
    out = combination._generate_all_triples_from_sorted_nodes(node_ids)
    expected = np.array(list(itertools.combinations(node_ids.tolist(), 3)), dtype=np.int64)
    np.testing.assert_array_equal(out, expected)


def test_decode_arity3_encoded_to_nodes_roundtrip():
    unique_nodes = np.array([10, 20, 40, 80], dtype=np.int64)
    num_nodes = np.int64(unique_nodes.shape[0])
    triples_idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    encoded = ((triples_idx[:, 0] * num_nodes) + triples_idx[:, 1]) * num_nodes + triples_idx[:, 2]
    out = combination._decode_arity3_encoded_to_nodes(
        unique_encoded=encoded,
        unique_nodes=unique_nodes,
        num_nodes=num_nodes,
    )
    expected = np.array([[10, 20, 40], [10, 40, 80]], dtype=np.int64)
    np.testing.assert_array_equal(out, expected)


def test_generate_union_candidates_arity3_from_pairs_falls_back_when_dense_cython_raises(monkeypatch):
    class _BrokenCy:
        @staticmethod
        def generate_union_encoded_arity3_dense_int64(_pairs, _num_nodes):
            raise RuntimeError("boom")

    monkeypatch.setattr(combination, "combination_cy", _BrokenCy)
    pair_nodes = np.array(
        [
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 5],
            [3, 6],
            [4, 7],
            [5, 6],
        ],
        dtype=np.int64,
    )
    with pytest.warns(RuntimeWarning, match="generate_union_arity3"):
        out = combination._generate_union_candidates_arity3_from_pairs(pair_nodes=pair_nodes)
    expected = combination._generate_valid_unions_by_pair_scan(target_nodes=pair_nodes, arity=3)
    np.testing.assert_array_equal(out, expected)


def test_generate_union_candidates_general_grouped_matches_dict_width4_random():
    rng = np.random.default_rng(2)
    for _ in range(3):
        target_nodes = rng.integers(0, 50, size=(70, 4), dtype=np.int64)
        sorted_nodes = np.sort(target_nodes, axis=1)
        sorted_nodes = np.unique(sorted_nodes, axis=0)
        observed = combination._generate_union_candidates_by_shared_subset_grouped(
            sorted_nodes=sorted_nodes,
            arity=5,
        )
        expected = combination._generate_union_candidates_by_shared_subset_python_dict(
            sorted_nodes=sorted_nodes,
            arity=5,
        )
        np.testing.assert_array_equal(observed, expected)


def test_generate_union_candidates_by_shared_subset_arity5_matches_pair_scan_random():
    rng = np.random.default_rng(5)
    for _ in range(3):
        target_nodes = rng.integers(0, 50, size=(70, 4), dtype=np.int64)
        out = combination._generate_union_candidates_by_shared_subset(target_nodes=target_nodes, arity=5)
        expected = combination._generate_valid_unions_by_pair_scan(target_nodes=target_nodes, arity=5)
        np.testing.assert_array_equal(out, expected)


def test_generate_union_candidates_by_shared_subset_arity5_falls_back_when_cython_raises(monkeypatch):
    class _BrokenCy:
        @staticmethod
        def generate_union_candidates_shared_subset_int64(_sorted_nodes):
            raise RuntimeError("boom")

    monkeypatch.setattr(combination, "combination_cy", _BrokenCy)
    target_nodes = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 5],
            [1, 2, 4, 5],
            [1, 3, 4, 5],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [2, 4, 5, 6],
        ],
        dtype=np.int64,
    )
    with pytest.warns(RuntimeWarning, match="generate_union_shared_subset"):
        out = combination._generate_union_candidates_by_shared_subset(target_nodes=target_nodes, arity=5)
    expected = combination._generate_valid_unions_by_pair_scan(target_nodes=target_nodes, arity=5)
    np.testing.assert_array_equal(out, expected)
