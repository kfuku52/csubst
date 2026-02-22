import numpy as np
import pandas as pd
import pytest

from csubst import main_analyze


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
