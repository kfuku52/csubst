import numpy as np
import pandas as pd

from csubst import combination
from csubst import tree
from csubst import ete


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


def test_get_node_combinations_target_dict_uses_threading_for_union(monkeypatch):
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
    assert node_union_calls
    assert all(backend == "threading" for backend in node_union_calls)


def test_get_node_combinations_cb_passed_uses_threading_for_union(monkeypatch):
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
    assert node_union_calls
    assert all(backend == "threading" for backend in node_union_calls)


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
