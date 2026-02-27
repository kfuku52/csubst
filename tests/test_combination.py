import numpy as np
import pandas as pd
import pytest
import itertools

from csubst import combination
from csubst import tree
from csubst import ete


def test_normalize_node_ids_rejects_non_integer_like_values():
    with pytest.raises(ValueError, match="integer-like"):
        combination._normalize_node_ids([1.5])
    with pytest.raises(ValueError, match="integer-like"):
        combination._normalize_node_ids(["2.5"])
    with pytest.raises(ValueError, match="integer-like"):
        combination._normalize_node_ids([True])


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


def test_unique_sorted_int64_1d_matches_numpy_unique():
    rng = np.random.default_rng(12)
    values = rng.integers(0, 200, size=(5000,), dtype=np.int64)
    observed = combination._unique_sorted_int64_1d(values)
    expected = np.unique(values)
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
    for _ in range(20):
        target_nodes = rng.integers(0, 30, size=(60, 2), dtype=np.int64)
        out = combination._generate_union_candidates_arity3_from_pairs(pair_nodes=target_nodes)
        expected = combination._generate_valid_unions_by_pair_scan(target_nodes=target_nodes, arity=3)
        np.testing.assert_array_equal(out, expected)


def test_generate_union_candidates_arity3_from_pairs_complete_graph_matches_pair_scan():
    node_ids = np.array([3, 5, 8, 13, 21, 34], dtype=np.int64)
    pair_nodes = np.array(list(itertools.combinations(node_ids.tolist(), 2)), dtype=np.int64)
    out = combination._generate_union_candidates_arity3_from_pairs(pair_nodes=pair_nodes)
    expected = combination._generate_valid_unions_by_pair_scan(target_nodes=pair_nodes, arity=3)
    np.testing.assert_array_equal(out, expected)


def test_generate_all_triples_from_sorted_nodes_matches_itertools():
    node_ids = np.array([2, 4, 7, 11], dtype=np.int64)
    out = combination._generate_all_triples_from_sorted_nodes(node_ids)
    expected = np.array(list(itertools.combinations(node_ids.tolist(), 3)), dtype=np.int64)
    np.testing.assert_array_equal(out, expected)


def test_generate_all_triples_from_sorted_nodes_falls_back_when_cython_raises(monkeypatch):
    class _BrokenCy:
        @staticmethod
        def generate_all_triples_from_sorted_nodes_int64(_unique_nodes):
            raise RuntimeError("boom")

    monkeypatch.setattr(combination, "combination_cy", _BrokenCy)
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
    out = combination._generate_union_candidates_arity3_from_pairs(pair_nodes=pair_nodes)
    expected = combination._generate_valid_unions_by_pair_scan(target_nodes=pair_nodes, arity=3)
    np.testing.assert_array_equal(out, expected)


def test_generate_union_candidates_arity4_from_triples_matches_pair_scan_random():
    rng = np.random.default_rng(4)
    for _ in range(10):
        triple_nodes = rng.integers(0, 40, size=(60, 3), dtype=np.int64)
        out = combination._generate_union_candidates_arity4_from_triples(triple_nodes=triple_nodes)
        expected = combination._generate_valid_unions_by_pair_scan(target_nodes=triple_nodes, arity=4)
        np.testing.assert_array_equal(out, expected)


def test_generate_union_candidates_arity4_from_triples_falls_back_when_cython_raises(monkeypatch):
    class _BrokenCy:
        @staticmethod
        def generate_union_candidates_arity4_from_triples_int64(_triples):
            raise RuntimeError("boom")

    monkeypatch.setattr(combination, "combination_cy", _BrokenCy)
    triple_nodes = np.array(
        [
            [1, 2, 3],
            [1, 2, 4],
            [1, 3, 4],
            [2, 3, 4],
            [4, 5, 6],
            [4, 5, 7],
        ],
        dtype=np.int64,
    )
    out = combination._generate_union_candidates_arity4_from_triples(triple_nodes=triple_nodes)
    expected = combination._generate_valid_unions_by_pair_scan(target_nodes=triple_nodes, arity=4)
    np.testing.assert_array_equal(out, expected)


def test_generate_union_candidates_general_grouped_matches_dict_width3_random():
    rng = np.random.default_rng(1)
    for _ in range(10):
        target_nodes = rng.integers(0, 40, size=(60, 3), dtype=np.int64)
        sorted_nodes = np.sort(target_nodes, axis=1)
        sorted_nodes = np.unique(sorted_nodes, axis=0)
        observed = combination._generate_union_candidates_by_shared_subset_grouped(
            sorted_nodes=sorted_nodes,
            arity=4,
        )
        expected = combination._generate_union_candidates_by_shared_subset_python_dict(
            sorted_nodes=sorted_nodes,
            arity=4,
        )
        np.testing.assert_array_equal(observed, expected)


def test_generate_union_candidates_general_grouped_matches_dict_width4_random():
    rng = np.random.default_rng(2)
    for _ in range(8):
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


def test_generate_union_candidates_by_shared_subset_arity4_matches_pair_scan_random():
    rng = np.random.default_rng(3)
    for _ in range(10):
        target_nodes = rng.integers(0, 35, size=(50, 3), dtype=np.int64)
        out = combination._generate_union_candidates_by_shared_subset(target_nodes=target_nodes, arity=4)
        expected = combination._generate_valid_unions_by_pair_scan(target_nodes=target_nodes, arity=4)
        np.testing.assert_array_equal(out, expected)


def test_generate_union_candidates_by_shared_subset_arity5_matches_pair_scan_random():
    rng = np.random.default_rng(5)
    for _ in range(8):
        target_nodes = rng.integers(0, 50, size=(70, 4), dtype=np.int64)
        out = combination._generate_union_candidates_by_shared_subset(target_nodes=target_nodes, arity=5)
        expected = combination._generate_valid_unions_by_pair_scan(target_nodes=target_nodes, arity=5)
        np.testing.assert_array_equal(out, expected)


def test_generate_union_candidates_by_shared_subset_complete_family_arity5_matches_itertools():
    node_ids = np.array([3, 5, 8, 13, 21, 34, 55], dtype=np.int64)
    target_nodes = np.array(list(itertools.combinations(node_ids.tolist(), 4)), dtype=np.int64)
    out = combination._generate_union_candidates_by_shared_subset(target_nodes=target_nodes, arity=5)
    expected = np.array(list(itertools.combinations(node_ids.tolist(), 5)), dtype=np.int64)
    np.testing.assert_array_equal(out, expected)


def test_generate_union_candidates_by_shared_subset_complete_family_skips_broken_cython(monkeypatch):
    class _BrokenCy:
        @staticmethod
        def generate_union_candidates_shared_subset_int64(_sorted_nodes):
            raise RuntimeError("boom")

    monkeypatch.setattr(combination, "combination_cy", _BrokenCy)
    node_ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    target_nodes = np.array(list(itertools.combinations(node_ids.tolist(), 4)), dtype=np.int64)
    out = combination._generate_union_candidates_by_shared_subset(target_nodes=target_nodes, arity=5)
    expected = np.array(list(itertools.combinations(node_ids.tolist(), 5)), dtype=np.int64)
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
    out = combination._generate_union_candidates_by_shared_subset(target_nodes=target_nodes, arity=5)
    expected = combination._generate_valid_unions_by_pair_scan(target_nodes=target_nodes, arity=5)
    np.testing.assert_array_equal(out, expected)


def test_map_row_combinations_to_node_ids_identity_fast_path_returns_row_ids():
    rows = np.array([[0, 2], [1, 3]], dtype=np.int64)
    node_ids = np.array([10, 20, 30, 40], dtype=np.int64)
    out = combination._map_row_combinations_to_node_ids(
        row_combinations=rows,
        node_ids=node_ids,
        row_ids_are_node_ids=True,
    )
    np.testing.assert_array_equal(out, rows)


def test_mark_dependent_row_combinations_flags_pairs_only_when_group_has_two_or_more():
    row_combinations = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )
    dep_groups = [
        np.array([0, 1], dtype=np.int64),
        np.array([2, 3], dtype=np.int64),
        np.array([9], dtype=np.int64),
    ]
    observed = combination._mark_dependent_row_combinations(row_combinations=row_combinations, dep_row_groups=dep_groups)
    expected = np.array([True, True, True, True], dtype=bool)
    np.testing.assert_array_equal(observed, expected)


def test_get_node_combinations_target_dict_verbose_false():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    non_root_ids = [ete.get_prop(n, "numerical_label") for n in tr.traverse() if not ete.is_root(n)]
    leaf_ids = [ete.get_prop(n, "numerical_label") for n in ete.iter_leaves(tr)]
    g = {
        "tree": tr,
        "dep_ids": [np.array([bid], dtype=np.int64) for bid in non_root_ids],
        "fg_dep_ids": {"traitA": []},
        "fg_df": pd.DataFrame({"name": ["A", "B"], "traitA": [1, 1]}),
        "threads": 1,
        "exhaustive_until": 1,
    }
    target_id_dict = {"traitA": np.array(leaf_ids, dtype=np.int64)}

    out_g, id_combinations = combination.get_node_combinations(
        g=g,
        target_id_dict=target_id_dict,
        arity=2,
        check_attr="name",
        verbose=False,
    )

    assert "fg_dependent_id_combinations" in out_g
    assert id_combinations.shape == (1, 2)
    assert set(id_combinations[0].tolist()) == set(leaf_ids)


def test_get_node_combinations_target_dict_rejects_non_integer_2d_target_ids():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1,C:1)R;", format=1))
    non_root_ids = [ete.get_prop(n, "numerical_label") for n in tr.traverse() if not ete.is_root(n)]
    g = {
        "tree": tr,
        "dep_ids": [np.array([bid], dtype=np.int64) for bid in non_root_ids],
        "fg_dep_ids": {"traitA": []},
        "fg_df": pd.DataFrame({"name": ["A", "B", "C"], "traitA": [1, 1, 1]}),
        "threads": 1,
        "exhaustive_until": 1,
    }
    target_id_dict = {
        "traitA": np.array(
            [
                [non_root_ids[0]],
                [float(non_root_ids[1]) + 0.5],
            ],
            dtype=float,
        )
    }
    with pytest.raises(ValueError, match="integer-like"):
        combination.get_node_combinations(
            g=g,
            target_id_dict=target_id_dict,
            arity=2,
            check_attr="name",
            verbose=False,
        )


def test_get_node_combinations_cb_passed_rejects_non_integer_branch_ids():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1,C:1,D:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    g = {
        "tree": tr,
        "dep_ids": [np.array([labels["A"]], dtype=np.int64)],
        "fg_dep_ids": {"traitA": []},
        "fg_df": pd.DataFrame({"name": ["A", "B", "C", "D"], "traitA": [1, 1, 1, 1]}),
        "threads": 1,
        "exhaustive_until": 2,
    }
    cb_passed = pd.DataFrame(
        {
            "branch_id_1": [labels["A"], labels["A"]],
            "branch_id_2": [labels["B"], float(labels["C"]) + 0.5],
            "is_fg_traitA": ["Y", "Y"],
            "is_mf_traitA": ["N", "N"],
            "is_mg_traitA": ["N", "N"],
        }
    )
    with pytest.raises(ValueError, match="integer-like"):
        combination.get_node_combinations(
            g=g,
            cb_passed=cb_passed,
            cb_all=False,
            arity=2,
            check_attr="name",
            verbose=False,
        )


def test_get_node_combinations_requires_exactly_one_selector():
    with pytest.raises(ValueError, match="Only one of target_id_dict, cb_passed, or exhaustive"):
        combination.get_node_combinations(g={}, target_id_dict=None, cb_passed=None, exhaustive=False, verbose=False)


def test_get_node_combinations_target_dict_accepts_scalar_dep_ids():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    non_root_ids = [ete.get_prop(n, "numerical_label") for n in tr.traverse() if not ete.is_root(n)]
    leaf_ids = [ete.get_prop(n, "numerical_label") for n in ete.iter_leaves(tr)]
    g = {
        "tree": tr,
        "dep_ids": [np.int64(bid) for bid in non_root_ids],
        "fg_dep_ids": {"traitA": []},
        "fg_df": pd.DataFrame({"name": ["A", "B"], "traitA": [1, 1]}),
        "threads": 1,
        "exhaustive_until": 1,
    }
    target_id_dict = {"traitA": np.array(leaf_ids, dtype=np.int64)}
    _, id_combinations = combination.get_node_combinations(
        g=g,
        target_id_dict=target_id_dict,
        arity=2,
        check_attr="name",
        verbose=False,
    )
    assert id_combinations.shape == (1, 2)
    assert set(id_combinations[0].tolist()) == set(leaf_ids)


def test_get_node_combinations_target_dict_accepts_scalar_target_id():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    non_root_ids = [ete.get_prop(n, "numerical_label") for n in tr.traverse() if not ete.is_root(n)]
    g = {
        "tree": tr,
        "dep_ids": [np.array([bid], dtype=np.int64) for bid in non_root_ids],
        "fg_dep_ids": {"traitA": []},
        "fg_df": pd.DataFrame({"name": ["A", "B"], "traitA": [1, 1]}),
        "threads": 1,
        "exhaustive_until": 1,
    }
    target_id_dict = {"traitA": np.int64(non_root_ids[0])}
    _, id_combinations = combination.get_node_combinations(
        g=g,
        target_id_dict=target_id_dict,
        arity=2,
        check_attr="name",
        verbose=False,
    )
    assert id_combinations.shape == (0, 2)
    assert id_combinations.dtype == np.int64


def test_get_node_combinations_target_dict_pairwise_fast_path_skips_node_union(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1,C:1)R;", format=1))
    non_root_ids = [ete.get_prop(n, "numerical_label") for n in tr.traverse() if not ete.is_root(n)]
    leaf_ids = [ete.get_prop(n, "numerical_label") for n in ete.iter_leaves(tr)]
    g = {
        "tree": tr,
        "dep_ids": [np.array([bid], dtype=np.int64) for bid in non_root_ids],
        "fg_dep_ids": {"traitA": []},
        "fg_df": pd.DataFrame({"name": ["A", "B", "C"], "traitA": [1, 1, 1]}),
        "threads": 2,
        "parallel_backend": "multiprocessing",
        "parallel_chunk_factor": 1,
        "parallel_min_items_node_union": 0,
        "parallel_min_items_per_job_node_union": 1,
        "exhaustive_until": 1,
    }
    target_id_dict = {"traitA": np.array(leaf_ids, dtype=np.int64)}
    calls = []

    def fake_run_starmap(func, args_iterable, n_jobs, backend="multiprocessing", chunksize=None):
        calls.append((func.__name__, backend))
        return [func(*args) for args in args_iterable]

    monkeypatch.setattr(combination.parallel, "run_starmap", fake_run_starmap)

    _, id_combinations = combination.get_node_combinations(
        g=g,
        target_id_dict=target_id_dict,
        arity=2,
        check_attr="name",
        verbose=False,
    )

    assert id_combinations.shape[0] > 0
    node_union_calls = [backend for func_name, backend in calls if func_name == "node_union"]
    assert node_union_calls == []


def test_get_node_combinations_handles_noncontiguous_branch_ids():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,(B:1,C:1)X:1)R;", format=1))
    reassigned = {"A": 11, "B": 29, "C": 41, "X": 73, "R": 5}
    for node in tr.traverse():
        ete.set_prop(node, "numerical_label", reassigned[node.name])
    non_root_ids = [
        int(ete.get_prop(node, "numerical_label"))
        for node in tr.traverse()
        if not ete.is_root(node)
    ]
    g = {
        "tree": tr,
        "dep_ids": [np.array([bid], dtype=np.int64) for bid in non_root_ids],
        "fg_dep_ids": {"traitA": []},
        "fg_df": pd.DataFrame({"name": ["A", "B", "C"], "traitA": [1, 1, 1]}),
        "threads": 1,
        "exhaustive_until": 1,
    }
    target_id_dict = {"traitA": np.array([11, 29, 41], dtype=np.int64)}
    _, id_combinations = combination.get_node_combinations(
        g=g,
        target_id_dict=target_id_dict,
        arity=2,
        check_attr="name",
        verbose=False,
    )
    observed = {tuple(sorted(row.tolist())) for row in id_combinations}
    expected = {(11, 29), (11, 41), (29, 41)}
    assert observed == expected


def test_get_node_combinations_cb_passed_avoids_node_union_path(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1,C:1,D:1)R;", format=1))
    non_root_ids = [ete.get_prop(n, "numerical_label") for n in tr.traverse() if not ete.is_root(n)]
    leaf_ids = [ete.get_prop(n, "numerical_label") for n in ete.iter_leaves(tr)]
    a, b, c, d = sorted(leaf_ids)
    g = {
        "tree": tr,
        "dep_ids": [np.array([bid], dtype=np.int64) for bid in non_root_ids],
        "fg_dep_ids": {"traitA": []},
        "fg_df": pd.DataFrame({"name": ["A", "B", "C", "D"], "traitA": [1, 1, 1, 1]}),
        "threads": 2,
        "parallel_backend": "multiprocessing",
        "parallel_chunk_factor": 1,
        "parallel_min_items_node_union": 0,
        "parallel_min_items_per_job_node_union": 1,
        "exhaustive_until": 3,
    }
    cb_passed = pd.DataFrame(
        {
            "branch_id_1": [a, a, a],
            "branch_id_2": [b, c, d],
            "is_fg_traitA": ["Y", "Y", "Y"],
            "is_mf_traitA": ["N", "N", "N"],
            "is_mg_traitA": ["N", "N", "N"],
        }
    )
    calls = []

    def fake_run_starmap(func, args_iterable, n_jobs, backend="multiprocessing", chunksize=None):
        calls.append((func.__name__, backend))
        return [func(*args) for args in args_iterable]

    monkeypatch.setattr(combination.parallel, "run_starmap", fake_run_starmap)

    _, id_combinations = combination.get_node_combinations(
        g=g,
        cb_passed=cb_passed,
        cb_all=False,
        arity=3,
        check_attr="name",
        verbose=False,
    )

    assert id_combinations.shape[0] > 0
    node_union_calls = [backend for func_name, backend in calls if func_name == "node_union"]
    assert node_union_calls == []


def test_get_node_combinations_exhaustive_parallel_nc_matrix_matches_single_thread():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1,C:1,D:1,E:1,F:1)R;", format=1))
    non_root_ids = [ete.get_prop(n, "numerical_label") for n in tr.traverse() if not ete.is_root(n)]
    g_single = {
        "tree": tr,
        "dep_ids": [np.array([bid], dtype=np.int64) for bid in non_root_ids],
        "fg_dep_ids": {"traitA": []},
        "fg_df": pd.DataFrame({"name": ["A", "B", "C", "D", "E", "F"], "traitA": [1, 1, 1, 1, 1, 1]}),
        "threads": 1,
        "exhaustive_until": 2,
    }
    g_parallel = dict(
        g_single,
        threads=2,
        parallel_min_items_nc_matrix=0,
        parallel_min_items_per_job_nc_matrix=1,
    )
    _, ids_single = combination.get_node_combinations(
        g=g_single,
        exhaustive=True,
        arity=2,
        check_attr="name",
        verbose=False,
    )
    _, ids_parallel = combination.get_node_combinations(
        g=g_parallel,
        exhaustive=True,
        arity=2,
        check_attr="name",
        verbose=False,
    )
    set_single = {tuple(sorted(row.tolist())) for row in ids_single}
    set_parallel = {tuple(sorted(row.tolist())) for row in ids_parallel}
    assert set_parallel == set_single


def test_node_combination_subsamples_rifle_handles_dep_ids_list_without_crashing():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1,C:1,D:1)R;", format=1))
    sub_ids = [ete.get_prop(n, "numerical_label") for n in ete.iter_leaves(tr)]
    g = {
        "tree": tr,
        "sub_branches": sub_ids,
        "dep_ids": [np.array([bid], dtype=np.int64) for bid in sub_ids],
    }
    np.random.seed(0)

    id_combinations = combination.node_combination_subsamples_rifle(g=g, arity=2, rep=2)

    assert id_combinations.shape == (2, 2)
    assert id_combinations.dtype == np.int64
    assert set(id_combinations.ravel().tolist()).issubset(set(sub_ids))
    assert len({tuple(sorted(row.tolist())) for row in id_combinations}) == 2


def test_node_combination_subsamples_rifle_accepts_scalar_dep_ids():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1,C:1,D:1)R;", format=1))
    sub_ids = [ete.get_prop(n, "numerical_label") for n in ete.iter_leaves(tr)]
    g = {
        "tree": tr,
        "sub_branches": sub_ids,
        "dep_ids": [np.int64(bid) for bid in sub_ids],
    }
    np.random.seed(0)
    id_combinations = combination.node_combination_subsamples_rifle(g=g, arity=2, rep=2)
    assert id_combinations.shape == (2, 2)
    assert id_combinations.dtype == np.int64
    assert set(id_combinations.ravel().tolist()).issubset(set(sub_ids))


def test_node_combination_subsamples_rifle_rejects_non_integer_sub_branches():
    g = {
        "sub_branches": [1.5, 2, 3],
        "dep_ids": [np.array([1], dtype=np.int64), np.array([2], dtype=np.int64), np.array([3], dtype=np.int64)],
    }
    with pytest.raises(ValueError, match="integer-like"):
        combination.node_combination_subsamples_rifle(g=g, arity=2, rep=2)


def test_node_combination_subsamples_rifle_tolerates_initial_duplicate_trials(monkeypatch):
    g = {
        "sub_branches": [1, 2, 3],
        "dep_ids": [np.array([1], dtype=np.int64), np.array([2], dtype=np.int64), np.array([3], dtype=np.int64)],
    }
    picks = iter([1, 2, 1, 2, 1, 2, 1, 3])

    def _fake_choice(a, size, replace=False):
        picked = int(next(picks))
        if picked not in set([int(v) for v in list(a)]):
            raise AssertionError("picked unavailable branch id")
        return np.array([picked], dtype=np.int64)

    monkeypatch.setattr(combination.np.random, "choice", _fake_choice)
    out = combination.node_combination_subsamples_rifle(g=g, arity=2, rep=2)
    assert out.shape == (2, 2)
    assert {tuple(sorted(row.tolist())) for row in out} == {(1, 2), (1, 3)}


def test_node_combination_subsamples_shotgun_returns_2d_empty_when_no_independent_combo():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    sub_ids = [ete.get_prop(n, "numerical_label") for n in ete.iter_leaves(tr)]
    g = {
        "tree": tr,
        "sub_branches": sub_ids,
        "dep_ids": [np.array(sub_ids, dtype=np.int64)],
    }

    id_combinations = combination.node_combination_subsamples_shotgun(g=g, arity=2, rep=5)

    assert id_combinations.shape == (0, 2)
    assert id_combinations.dtype == np.int64


def test_node_combination_subsamples_shotgun_handles_noncontiguous_branch_ids():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,(B:1,C:1)X:1)R;", format=1))
    reassigned = {"A": 11, "B": 29, "C": 41, "X": 73, "R": 5}
    for node in tr.traverse():
        ete.set_prop(node, "numerical_label", reassigned[node.name])
    sub_ids = [11, 29, 41, 73]
    g = {
        "tree": tr,
        "sub_branches": sub_ids,
        "dep_ids": [np.array([bid], dtype=np.int64) for bid in sub_ids],
    }
    np.random.seed(0)
    id_combinations = combination.node_combination_subsamples_shotgun(g=g, arity=2, rep=5)
    assert id_combinations.shape == (5, 2)
    assert id_combinations.dtype == np.int64
    assert set(id_combinations.ravel().tolist()).issubset(set(sub_ids))
    assert len({tuple(sorted(row.tolist())) for row in id_combinations}) == 5


def test_get_node_combinations_cb_passed_multi_trait_does_not_overprune_between_traits():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1,C:1,D:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    a,b,c,d = labels["A"], labels["B"], labels["C"], labels["D"]
    g = {
        "tree": tr,
        "dep_ids": [np.array([bid], dtype=np.int64) for bid in [a, b, c, d]],
        "fg_dep_ids": {
            "traitA": [np.array([b, c], dtype=np.int64)],
            "traitB": [np.array([b, d], dtype=np.int64)],
        },
        "fg_df": pd.DataFrame(
            {
                "name": ["A", "B", "C", "D"],
                "traitA": [1, 1, 1, 1],
                "traitB": [1, 1, 1, 1],
            }
        ),
        "threads": 1,
        "exhaustive_until": 2,
    }
    cb_passed = pd.DataFrame(
        {
            "branch_id_1": [a, a, a],
            "branch_id_2": [b, c, d],
            "is_fg_traitA": ["Y", "Y", "N"],
            "is_mf_traitA": ["N", "N", "N"],
            "is_mg_traitA": ["N", "N", "N"],
            "is_fg_traitB": ["Y", "N", "Y"],
            "is_mf_traitB": ["N", "N", "N"],
            "is_mg_traitB": ["N", "N", "N"],
        }
    )

    _, id_combinations = combination.get_node_combinations(
        g=g,
        cb_passed=cb_passed,
        cb_all=False,
        arity=3,
        check_attr="name",
        verbose=False,
    )

    observed = {tuple(sorted(row.tolist())) for row in id_combinations}
    expected = {tuple(sorted([a, b, c])), tuple(sorted([a, b, d]))}
    assert observed == expected


def test_get_node_combinations_logs_foreground_removal_when_count_is_even(capsys):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1,C:1,D:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    a,b,c,d = labels["A"], labels["B"], labels["C"], labels["D"]
    g = {
        "tree": tr,
        "dep_ids": [np.array([bid], dtype=np.int64) for bid in [a, b, c, d]],
        "fg_dep_ids": {"traitA": [np.array([b, c], dtype=np.int64)]},
        "fg_df": pd.DataFrame({"name": ["A", "B", "C", "D"], "traitA": [1, 1, 1, 1]}),
        "threads": 1,
        "exhaustive_until": 2,
    }
    cb_passed = pd.DataFrame(
        {
            "branch_id_1": [a, a, a, b, b, c],
            "branch_id_2": [b, c, d, c, d, d],
            "is_fg_traitA": ["Y", "Y", "Y", "Y", "Y", "Y"],
            "is_mf_traitA": ["N", "N", "N", "N", "N", "N"],
            "is_mg_traitA": ["N", "N", "N", "N", "N", "N"],
        }
    )

    _ = combination.get_node_combinations(
        g=g,
        cb_passed=cb_passed,
        cb_all=False,
        arity=3,
        check_attr="name",
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "Removing 2 (out of 4) non-independent foreground branch combinations for traitA." in captured.out


def test_nc_matrix2id_combinations_ignores_columns_with_wrong_arity():
    # First and third columns do not have exactly arity=2 active rows and should be ignored.
    nc_matrix = np.array(
        [
            [0, 1, 1],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    )
    out = combination.nc_matrix2id_combinations(nc_matrix=nc_matrix, arity=2, ncpu=1)
    expected = np.array([[0, 1]], dtype=np.int64)
    np.testing.assert_array_equal(out, expected)
