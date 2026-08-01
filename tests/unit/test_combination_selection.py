import numpy as np
import pandas as pd
import pytest

from csubst import combination
from csubst import tree
from csubst import ete


def test_get_global_dep_ids_does_not_duplicate_sister_groups_per_leaf():
    tr = tree.add_numerical_node_labels(
        ete.PhyloNode("((A:1,B:1)X:1,(C:1,D:1)Y:1)R;", format=1)
    )
    root_id = int(ete.get_prop(tr, "numerical_label"))
    g = {
        "tree": tr,
        "exclude_sister_pair": True,
        "state_cdn": np.ones(shape=(root_id + 1, 1, 1), dtype=float),
    }
    dep_groups = combination.get_global_dep_ids(g)
    normalized = [
        tuple(np.asarray(group, dtype=np.int64).reshape(-1).tolist())
        for group in dep_groups
    ]
    assert len(normalized) == len(set(normalized))
    assert len(normalized) == 7


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


def test_get_node_combinations_exhaustive_parallel_nc_matrix_matches_single_thread(monkeypatch):
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
    g_parallel = dict(g_single, threads=2)
    _, ids_single = combination.get_node_combinations(
        g=g_single,
        exhaustive=True,
        arity=2,
        check_attr="name",
        verbose=False,
    )
    monkeypatch.setattr(
        combination.parallel,
        "resolve_task_n_jobs",
        lambda num_items, threads, task: min(int(threads), max(1, int(num_items))),
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
