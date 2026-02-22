import os
import numpy as np
import pandas as pd
import pytest

from csubst import main_analyze
from csubst import tree
from csubst import ete


def test_cb_search_respects_max_combination_limit(monkeypatch):
    captured = {}

    def fake_get_node_combinations(g, target_id_dict=None, cb_passed=None, exhaustive=False, cb_all=False, arity=2,
                                   check_attr=None, verbose=True):
        if arity == 2:
            return g, np.array([[0, 1]], dtype=np.int64)
        captured["cb_passed"] = cb_passed.copy(deep=True)
        return g, np.zeros((0, arity), dtype=np.int64)

    def fake_get_cb(id_combinations, sub_tensor, g, attr, selected_base_stats=None):
        return pd.DataFrame({"_unused": [1.0]})

    def fake_merge_tables(cbOS, cbON):
        return pd.DataFrame(
            {
                "branch_id_1": [10, 11, 12, 13],
                "branch_id_2": [20, 21, 22, 23],
                "score": [9.0, 8.0, 7.0, 6.0],
                "is_fg_traitA": ["Y", "Y", "Y", "Y"],
                "is_mf_traitA": ["N", "N", "N", "N"],
                "is_mg_traitA": ["N", "N", "N", "N"],
            }
        )

    monkeypatch.setattr(main_analyze.combination, "get_node_combinations", fake_get_node_combinations)
    monkeypatch.setattr(main_analyze.substitution, "get_reducer_sub_tensor", lambda sub_tensor, g, label: sub_tensor)
    monkeypatch.setattr(main_analyze.substitution, "get_cb", fake_get_cb)
    monkeypatch.setattr(main_analyze.table, "merge_tables", fake_merge_tables)
    monkeypatch.setattr(
        main_analyze.substitution,
        "add_dif_stats",
        lambda cb, tol, prefix, output_stats=None: cb,
    )
    monkeypatch.setattr(main_analyze.omega, "calc_omega", lambda cb, OS, ON, g: (cb, g))
    monkeypatch.setattr(main_analyze.substitution, "get_substitutions_per_branch", lambda cb, b, g: cb)
    monkeypatch.setattr(main_analyze.table, "get_linear_regression", lambda cb: cb)
    monkeypatch.setattr(main_analyze.foreground, "get_foreground_branch_num", lambda cb, g: (cb, g))
    monkeypatch.setattr(main_analyze.table, "sort_cb", lambda cb: cb)
    monkeypatch.setattr(main_analyze.foreground, "add_median_cb_stats", lambda g, cb, current_arity, start: g)

    g = {
        "max_arity": 3,
        "exhaustive_until": 2,
        "foreground": None,
        "cutoff_stat": "score,0",
        "max_combination": 2,
        "threads": 1,
        "float_tol": 1e-12,
        "calibrate_longtail": False,
        "branch_dist": False,
        "float_format": "%.6f",
        "fg_clade_permutation": 0,
        "df_cb_stats_main": pd.DataFrame(),
    }

    main_analyze.cb_search(g=g, b=None, OS_tensor=None, ON_tensor=None, id_combinations=None, write_cb=False)

    cb_passed = captured["cb_passed"]
    assert cb_passed.shape[0] == 2
    assert cb_passed["score"].tolist() == [9.0, 8.0]


def test_cb_search_rejects_invalid_max_arity():
    g = {
        "max_arity": 1,
        "max_combination": 1,
    }
    with pytest.raises(ValueError, match="--max_arity should be >= 2"):
        main_analyze.cb_search(g=g, b=None, OS_tensor=None, ON_tensor=None, id_combinations=None, write_cb=False)


def test_cb_search_rejects_invalid_max_combination():
    g = {
        "max_arity": 2,
        "max_combination": 0,
    }
    with pytest.raises(ValueError, match="--max_combination should be >= 1"):
        main_analyze.cb_search(g=g, b=None, OS_tensor=None, ON_tensor=None, id_combinations=None, write_cb=False)


def test_annotate_branch_length_column_uses_branch_id_mapping_not_index():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:2)X:3,C:4)R;", format=1))
    by_name = {n.name: n for n in tr.traverse()}
    rows = [
        {"branch_id": int(ete.get_prop(by_name["R"], "numerical_label")), "dummy": 1.0},
        {"branch_id": int(ete.get_prop(by_name["A"], "numerical_label")), "dummy": 2.0},
        {"branch_id": int(ete.get_prop(by_name["X"], "numerical_label")), "dummy": 3.0},
    ]
    # Intentionally scramble index labels so index!=branch_id.
    b = pd.DataFrame(rows, index=[100, 101, 102])
    out = main_analyze._annotate_branch_length_column(b=b.copy(), tree_obj=tr)
    for row in out.itertuples(index=False):
        node = next(n for n in tr.traverse() if int(ete.get_prop(n, "numerical_label")) == int(row.branch_id))
        expected = 0.0 if (node.dist is None) else float(node.dist)
        assert row.branch_length == pytest.approx(expected, abs=1e-12)


def test_plot_state_tree_in_directory_restores_cwd_after_plot_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original_cwd = os.getcwd()

    def fake_plot_state_tree(state, orders, mode, g):
        raise RuntimeError("plot failed")

    monkeypatch.setattr(main_analyze.tree, "plot_state_tree", fake_plot_state_tree)
    with pytest.raises(RuntimeError, match="plot failed"):
        main_analyze._plot_state_tree_in_directory(
            output_dir="csubst_plot_state_aa",
            state=np.zeros((1, 0, 1), dtype=float),
            orders=np.array([], dtype=str),
            mode="aa",
            g={},
        )
    assert os.getcwd() == original_cwd
    assert (tmp_path / "csubst_plot_state_aa").exists()
