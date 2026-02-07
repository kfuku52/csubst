import numpy
import pandas

from csubst import combination
from csubst import tree
from csubst import ete


def test_get_node_combinations_target_dict_verbose_false():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    non_root_ids = [ete.get_prop(n, "numerical_label") for n in tr.traverse() if not ete.is_root(n)]
    leaf_ids = [ete.get_prop(n, "numerical_label") for n in ete.iter_leaves(tr)]
    g = {
        "tree": tr,
        "dep_ids": [numpy.array([bid], dtype=numpy.int64) for bid in non_root_ids],
        "fg_dep_ids": {"traitA": []},
        "fg_df": pandas.DataFrame({"name": ["A", "B"], "traitA": [1, 1]}),
        "threads": 1,
        "exhaustive_until": 1,
    }
    target_id_dict = {"traitA": numpy.array(leaf_ids, dtype=numpy.int64)}

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
        "dep_ids": [numpy.array([bid], dtype=numpy.int64) for bid in non_root_ids],
        "fg_dep_ids": {"traitA": []},
        "fg_df": pandas.DataFrame({"name": ["A", "B", "C"], "traitA": [1, 1, 1]}),
        "threads": 2,
        "parallel_backend": "multiprocessing",
        "parallel_chunk_factor": 1,
        "exhaustive_until": 1,
    }
    target_id_dict = {"traitA": numpy.array(leaf_ids, dtype=numpy.int64)}
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
        "dep_ids": [numpy.array([bid], dtype=numpy.int64) for bid in non_root_ids],
        "fg_dep_ids": {"traitA": []},
        "fg_df": pandas.DataFrame({"name": ["A", "B", "C", "D"], "traitA": [1, 1, 1, 1]}),
        "threads": 2,
        "parallel_backend": "multiprocessing",
        "parallel_chunk_factor": 1,
        "exhaustive_until": 3,
    }
    cb_passed = pandas.DataFrame(
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
