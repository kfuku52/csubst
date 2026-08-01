from contextlib import contextmanager

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import warnings

from csubst import foreground


def test_clade_permutation_uses_observed_stats_when_main_table_has_only_permutations(monkeypatch):
    observed = pd.DataFrame(
        {
            "arity": [2],
            "mode": ["foreground"],
            "median_omegaCany2spe_fg_traitA": [0.5],
            "total_OCNany2spe_fg_traitA": [3.0],
        }
    )
    g = {
        "fg_df": pd.DataFrame({"name": ["tip1"], "traitA": [1]}),
        "df_cb_stats": observed.copy(deep=True),
        "df_cb_stats_main": pd.DataFrame(),
        "fg_clade_permutation": 1,
        "current_arity": 2,
        "cutoff_stat": "dist_bl>0",
    }
    cb = pd.DataFrame({"branch_id_1": [1], "branch_id_2": [2]})

    def fake_initialize_df_cb_stats(local_g):
        local_g["df_cb_stats"] = pd.DataFrame({"arity": [local_g["current_arity"]], "mode": [""]})
        return local_g

    def fake_set_random_foreground_branch(local_g, trait_name):
        local_g["r_fg_ids"] = {trait_name: np.array([10, 11], dtype=np.int64)}
        return local_g, np.array([[1, 2]], dtype=np.int64)

    def fake_add_median_cb_stats(local_g, rcb, current_arity, start, verbose=False):
        local_g["df_cb_stats"].loc[:, "arity"] = current_arity
        local_g["df_cb_stats"].loc[:, "median_omegaCany2spe_fg_traitA"] = 0.9
        local_g["df_cb_stats"].loc[:, "total_OCNany2spe_fg_traitA"] = 2.0
        return local_g

    monkeypatch.setattr(foreground.param, "initialize_df_cb_stats", fake_initialize_df_cb_stats)
    monkeypatch.setattr(foreground, "set_random_foreground_branch", fake_set_random_foreground_branch)
    monkeypatch.setattr(foreground, "add_median_cb_stats", fake_add_median_cb_stats)

    out = foreground.clade_permutation(cb, g)

    pdt.assert_frame_equal(out["df_cb_stats"], observed)
    assert out["df_cb_stats_main"].shape[0] == 1
    assert out["df_cb_stats_main"].loc[0, "mode"].startswith("randomization_traitA_")


def test_clade_permutation_iterates_all_traits_in_fg_format2(monkeypatch):
    observed = pd.DataFrame(
        {
            "arity": [2],
            "mode": ["foreground"],
            "median_omegaCany2spe_fg_traitA": [0.4],
            "total_OCNany2spe_fg_traitA": [2.5],
            "median_omegaCany2spe_fg_traitB": [0.8],
            "total_OCNany2spe_fg_traitB": [4.0],
        }
    )
    g = {
        "fg_df": pd.DataFrame({"name": ["tip1"], "traitA": [1], "traitB": [0]}),
        "df_cb_stats": observed.copy(deep=True),
        "df_cb_stats_main": pd.DataFrame(),
        "fg_clade_permutation": 1,
        "current_arity": 2,
        "cutoff_stat": "dist_bl>0",
    }
    cb = pd.DataFrame({"branch_id_1": [1], "branch_id_2": [2]})
    called_traits = []

    def fake_initialize_df_cb_stats(local_g):
        local_g["df_cb_stats"] = pd.DataFrame({"arity": [local_g["current_arity"]], "mode": [""]})
        return local_g

    def fake_set_random_foreground_branch(local_g, trait_name, num_trial=100, sample_original_foreground=False):
        called_traits.append(trait_name)
        local_g["r_fg_ids"] = {trait_name: np.array([21, 22], dtype=np.int64)}
        return local_g, np.array([[1, 2]], dtype=np.int64)

    def fake_add_median_cb_stats(local_g, rcb, current_arity, start, verbose=False):
        local_g["df_cb_stats"].loc[:, "arity"] = current_arity
        trait_cols = [c for c in rcb.columns if c.startswith("is_fg_") and (rcb[c] == "Y").all()]
        assert len(trait_cols) == 1
        focal_trait = trait_cols[0].replace("is_fg_", "")
        for trait_name in ["traitA", "traitB"]:
            local_g["df_cb_stats"].loc[:, "median_omegaCany2spe_fg_" + trait_name] = np.nan
            local_g["df_cb_stats"].loc[:, "total_OCNany2spe_fg_" + trait_name] = np.nan
        local_g["df_cb_stats"].loc[:, "median_omegaCany2spe_fg_" + focal_trait] = 1.0
        local_g["df_cb_stats"].loc[:, "total_OCNany2spe_fg_" + focal_trait] = 3.0
        return local_g

    monkeypatch.setattr(foreground.param, "initialize_df_cb_stats", fake_initialize_df_cb_stats)
    monkeypatch.setattr(foreground, "set_random_foreground_branch", fake_set_random_foreground_branch)
    monkeypatch.setattr(foreground, "add_median_cb_stats", fake_add_median_cb_stats)

    out = foreground.clade_permutation(cb, g)

    assert called_traits == ["traitA", "traitB"]
    assert out["df_cb_stats_main"].shape[0] == 2
    modes = out["df_cb_stats_main"].loc[:, "mode"].tolist()
    assert any(mode.startswith("randomization_traitA_") for mode in modes)
    assert any(mode.startswith("randomization_traitB_") for mode in modes)


def test_clade_permutation_continues_when_randomization_fails(monkeypatch):
    observed = pd.DataFrame(
        {
            "arity": [2],
            "mode": ["foreground"],
            "median_omegaCany2spe_fg_traitA": [0.5],
            "total_OCNany2spe_fg_traitA": [2.0],
        }
    )
    g = {
        "fg_df": pd.DataFrame({"name": ["tip1"], "traitA": [1]}),
        "df_cb_stats": observed.copy(deep=True),
        "df_cb_stats_main": pd.DataFrame(),
        "fg_clade_permutation": 1,
        "current_arity": 2,
        "cutoff_stat": "dist_bl>0",
    }
    cb = pd.DataFrame({"branch_id_1": [1], "branch_id_2": [2]})

    def fake_initialize_df_cb_stats(local_g):
        local_g["df_cb_stats"] = pd.DataFrame({"arity": [local_g["current_arity"]], "mode": [""]})
        return local_g

    def fake_set_random_foreground_branch(local_g, trait_name, num_trial=100, sample_original_foreground=False):
        raise Exception("no permutation candidates")

    monkeypatch.setattr(foreground.param, "initialize_df_cb_stats", fake_initialize_df_cb_stats)
    monkeypatch.setattr(foreground, "set_random_foreground_branch", fake_set_random_foreground_branch)

    out = foreground.clade_permutation(cb, g)

    pdt.assert_frame_equal(out["df_cb_stats"], observed)
    assert out["df_cb_stats_main"].shape[0] == 1
    mode_value = out["df_cb_stats_main"].loc[0, "mode"]
    assert mode_value.startswith("randomization_traitA_iter0_failed_trial")
    assert out["df_cb_stats_main"].loc[0, "clade_permutation_status_traitA"] == "no permutation candidates"


def test_clade_permutation_retries_with_sample_original_foreground(monkeypatch):
    observed = pd.DataFrame(
        {
            "arity": [2],
            "mode": ["foreground"],
            "median_omegaCany2spe_fg_traitA": [0.4],
            "total_OCNany2spe_fg_traitA": [2.5],
        }
    )
    g = {
        "fg_df": pd.DataFrame({"name": ["tip1"], "traitA": [1]}),
        "df_cb_stats": observed.copy(deep=True),
        "df_cb_stats_main": pd.DataFrame(),
        "fg_clade_permutation": 1,
        "current_arity": 2,
        "cutoff_stat": "dist_bl>0",
    }
    cb = pd.DataFrame({"branch_id_1": [1], "branch_id_2": [2]})
    sampled_flags = []

    def fake_initialize_df_cb_stats(local_g):
        local_g["df_cb_stats"] = pd.DataFrame({"arity": [local_g["current_arity"]], "mode": [""]})
        return local_g

    def fake_set_random_foreground_branch(local_g, trait_name, num_trial=100, sample_original_foreground=False):
        sampled_flags.append(sample_original_foreground)
        if not sample_original_foreground:
            raise Exception("strict mode failed")
        local_g["r_fg_ids"] = {trait_name: np.array([1, 2], dtype=np.int64)}
        return local_g, np.array([[1, 2]], dtype=np.int64)

    def fake_add_median_cb_stats(local_g, rcb, current_arity, start, verbose=False):
        local_g["df_cb_stats"].loc[:, "arity"] = current_arity
        local_g["df_cb_stats"].loc[:, "median_omegaCany2spe_fg_traitA"] = 0.9
        local_g["df_cb_stats"].loc[:, "total_OCNany2spe_fg_traitA"] = 3.0
        return local_g

    monkeypatch.setattr(foreground.param, "initialize_df_cb_stats", fake_initialize_df_cb_stats)
    monkeypatch.setattr(foreground, "set_random_foreground_branch", fake_set_random_foreground_branch)
    monkeypatch.setattr(foreground, "add_median_cb_stats", fake_add_median_cb_stats)

    out = foreground.clade_permutation(cb, g)

    assert sampled_flags == [False, True]
    assert out["df_cb_stats_main"].shape[0] == 1
    assert "_sampleorig_" in out["df_cb_stats_main"].loc[0, "mode"]


def test_clade_permutation_parallelizes_candidates_over_resolved_backend(monkeypatch):
    observed = pd.DataFrame(
        {
            "arity": [2],
            "mode": ["foreground"],
            "median_omegaCany2spe_fg_traitA": [0.4],
            "total_OCNany2spe_fg_traitA": [2.5],
        }
    )
    g = {
        "fg_df": pd.DataFrame({"name": ["tip1"], "traitA": [1]}),
        "df_cb_stats": observed.copy(deep=True),
        "df_cb_stats_main": pd.DataFrame(),
        "fg_clade_permutation": 2,
        "current_arity": 2,
        "cutoff_stat": "dummy,0",
        "threads": 2,
    }
    cb = pd.DataFrame({"branch_id_1": [1], "branch_id_2": [2], "dummy": [1.0]})
    random_calls = []
    parallel_calls = []

    def fake_set_random_foreground_branch(local_g, trait_name, num_trial=100, sample_original_foreground=False):
        random_calls.append(sample_original_foreground)
        offset = len(random_calls) * 10
        local_g["r_fg_ids"] = {trait_name: np.array([offset + 1, offset + 2], dtype=np.int64)}
        return local_g, np.array([[1, 2]], dtype=np.int64)

    def fake_add_median_cb_stats(local_g, rcb, current_arity, start, verbose=False):
        assert (rcb.loc[:, "is_fg_traitA"] == "Y").all()
        local_g["df_cb_stats"].loc[:, "arity"] = current_arity
        local_g["df_cb_stats"].loc[:, "median_omegaCany2spe_fg_traitA"] = 0.9
        local_g["df_cb_stats"].loc[:, "total_OCNany2spe_fg_traitA"] = 3.0
        return local_g

    @contextmanager
    def fake_persistent_starmap_runner(n_jobs, backend="multiprocessing"):
        def run(func, args_iterable):
            args = list(args_iterable)
            parallel_calls.append((len(args), n_jobs, backend, None))
            return [func(*arg) for arg in args]
        yield run

    monkeypatch.setattr(foreground, "set_random_foreground_branch", fake_set_random_foreground_branch)
    monkeypatch.setattr(foreground, "add_median_cb_stats", fake_add_median_cb_stats)
    monkeypatch.setattr(
        foreground.parallel,
        "persistent_starmap_runner",
        fake_persistent_starmap_runner,
    )

    out = foreground.clade_permutation(cb=cb, g=g)

    assert random_calls == [False, False]
    assert parallel_calls == [(2, 2, "multiprocessing", None)]
    assert out["df_cb_stats_main"].shape[0] == 2
    assert out["df_cb_stats_main"].loc[:, "clade_permutation_backend"].tolist() == ["multiprocessing", "multiprocessing"]
    assert out["df_cb_stats_main"].loc[:, "clade_permutation_n_jobs"].tolist() == [2, 2]
    assert out["df_cb_stats_main"].loc[:, "mode"].tolist() == [
        "randomization_traitA_iter1_bid11,12",
        "randomization_traitA_iter2_bid21,22",
    ]


def test_clade_permutation_recomputes_missing_randomized_combinations(monkeypatch, capsys):
    observed = pd.DataFrame(
        {
            "arity": [2],
            "mode": ["foreground"],
            "median_omegaCany2spe_fg_traitA": [0.5],
            "total_OCNany2spe_fg_traitA": [2.0],
        }
    )
    g = {
        "fg_df": pd.DataFrame({"name": ["tip1"], "traitA": [1]}),
        "df_cb_stats": observed.copy(deep=True),
        "df_cb_stats_main": pd.DataFrame(),
        "fg_clade_permutation": 1,
        "current_arity": 2,
        "cutoff_stat": "dist_bl>0",
    }
    cb = pd.DataFrame({"branch_id_1": [1], "branch_id_2": [2], "dummy": [0.0]})
    recompute_calls = []

    def fake_initialize_df_cb_stats(local_g):
        local_g["df_cb_stats"] = pd.DataFrame({"arity": [local_g["current_arity"]], "mode": [""]})
        return local_g

    def fake_set_random_foreground_branch(local_g, trait_name, num_trial=100, sample_original_foreground=False):
        local_g["r_fg_ids"] = {trait_name: np.array([1, 2], dtype=np.int64)}
        # include one combination that is missing from cb to trigger recomputation
        return local_g, np.array([[1, 2], [2, 3]], dtype=np.int64)

    def fake_recompute_missing(g, missing_id_combinations, OS_tensor_reducer, ON_tensor_reducer):
        recompute_calls.append(missing_id_combinations.tolist())
        cb_missing = pd.DataFrame({"branch_id_1": [2], "branch_id_2": [3], "dummy": [1.0]})
        return cb_missing, g

    def fake_add_median_cb_stats(local_g, rcb, current_arity, start, verbose=False):
        branch_pairs = {(int(r.branch_id_1), int(r.branch_id_2)) for r in rcb.itertuples(index=False)}
        assert branch_pairs == {(1, 2), (2, 3)}
        local_g["df_cb_stats"].loc[:, "arity"] = current_arity
        local_g["df_cb_stats"].loc[:, "median_omegaCany2spe_fg_traitA"] = 0.9
        local_g["df_cb_stats"].loc[:, "total_OCNany2spe_fg_traitA"] = 3.0
        return local_g

    monkeypatch.setattr(foreground.param, "initialize_df_cb_stats", fake_initialize_df_cb_stats)
    monkeypatch.setattr(foreground, "set_random_foreground_branch", fake_set_random_foreground_branch)
    monkeypatch.setattr(foreground, "_recompute_missing_permutation_rows", fake_recompute_missing)
    monkeypatch.setattr(foreground, "add_median_cb_stats", fake_add_median_cb_stats)

    out = foreground.clade_permutation(
        cb=cb,
        g=g,
        OS_tensor_reducer=object(),
        ON_tensor_reducer=object(),
    )

    assert recompute_calls == [[[2, 3]]]
    assert out["df_cb_stats_main"].shape[0] == 1
    captured = capsys.readouterr()
    assert "permuted foreground branch combinations were dropped" not in captured.out


def test_recompute_missing_rows_passes_float_tol_and_preserves_infinite_omega(monkeypatch):
    missing_id_combinations = np.array([[1, 2], [2, 3]], dtype=np.int64)
    captured = {}

    def fake_get_cb(id_combinations, sub_tensor, g, attr, selected_base_stats=None):
        return pd.DataFrame(
            {
                "branch_id_1": id_combinations[:, 0],
                "branch_id_2": id_combinations[:, 1],
            }
        )

    def fake_merge_tables(cbOS, cbON):
        return cbOS.copy(deep=True)

    def fake_calc_omega(cb_missing, OS_tensor_reducer, ON_tensor_reducer, g):
        out = cb_missing.copy(deep=True)
        out["dNCany2spe"] = np.array([1.0, np.inf], dtype=np.float64)
        out["dSCany2spe"] = np.array([1.0, 1.0], dtype=np.float64)
        out["omegaCany2spe"] = np.array([1.0, np.inf], dtype=np.float64)
        return out, g

    original_calibrate_dsc = foreground.omega.calibrate_dsc

    def capture_calibrate_dsc(cb_missing, output_stats=None, float_tol=1e-12):
        captured["float_tol"] = float(float_tol)
        captured["output_stats"] = output_stats
        return original_calibrate_dsc(
            cb=cb_missing,
            output_stats=output_stats,
            float_tol=float_tol,
        )

    monkeypatch.setattr(foreground.substitution, "get_cb", fake_get_cb)
    monkeypatch.setattr(foreground.table, "merge_tables", fake_merge_tables)
    monkeypatch.setattr(
        foreground.substitution,
        "add_dif_stats",
        lambda cb, tol, prefix, output_stats=None: cb,
    )
    monkeypatch.setattr(foreground.omega, "calc_omega", fake_calc_omega)
    monkeypatch.setattr(foreground.omega, "calibrate_dsc", capture_calibrate_dsc)
    monkeypatch.setattr(foreground.substitution, "get_substitutions_per_branch", lambda cb, b, g: cb)
    monkeypatch.setattr(foreground.table, "get_linear_regression", lambda cb: cb)
    monkeypatch.setattr(foreground.output_stat, "drop_unrequested_stat_columns", lambda cb, output_stats=None: cb)
    monkeypatch.setattr(foreground, "get_foreground_branch_num", lambda cb, g: (cb, g))
    monkeypatch.setattr(foreground.table, "sort_cb", lambda cb: cb)

    g = {
        "float_tol": 1e-7,
        "output_stats": ["any2spe"],
        "output_base_stats": ["any2spe"],
        "calibrate_longtail": True,
        "exhaustive_until": 2,
        "current_arity": 2,
        "branch_dist": False,
        "branch_table": None,
        "threads": 1,
        "float_type": np.float64,
    }
    out, _ = foreground._recompute_missing_permutation_rows(
        g=g,
        missing_id_combinations=missing_id_combinations,
        OS_tensor_reducer=object(),
        ON_tensor_reducer=object(),
    )

    assert captured["float_tol"] == pytest.approx(1e-7)
    assert captured["output_stats"] == ["any2spe"]
    assert "omegaCany2spe_nocalib" in out.columns
    is_inf = np.isinf(out.loc[:, "omegaCany2spe"].to_numpy(dtype=np.float64))
    assert int(is_inf.sum()) == 1
    assert np.isinf(out.loc[is_inf, "omegaCany2spe_nocalib"].to_numpy(dtype=np.float64)).all()


def test_clade_permutation_reports_dropped_without_recomputation(monkeypatch, capsys):
    observed = pd.DataFrame(
        {
            "arity": [2],
            "mode": ["foreground"],
            "median_omegaCany2spe_fg_traitA": [0.5],
            "total_OCNany2spe_fg_traitA": [2.0],
        }
    )
    g = {
        "fg_df": pd.DataFrame({"name": ["tip1"], "traitA": [1]}),
        "df_cb_stats": observed.copy(deep=True),
        "df_cb_stats_main": pd.DataFrame(),
        "fg_clade_permutation": 1,
        "current_arity": 2,
        "cutoff_stat": "dist_bl>0",
    }
    cb = pd.DataFrame({"branch_id_1": [1], "branch_id_2": [2], "dummy": [0.0]})

    def fake_initialize_df_cb_stats(local_g):
        local_g["df_cb_stats"] = pd.DataFrame({"arity": [local_g["current_arity"]], "mode": [""]})
        return local_g

    def fake_set_random_foreground_branch(local_g, trait_name, num_trial=100, sample_original_foreground=False):
        local_g["r_fg_ids"] = {trait_name: np.array([1, 2], dtype=np.int64)}
        return local_g, np.array([[1, 2], [2, 3]], dtype=np.int64)

    def fake_add_median_cb_stats(local_g, rcb, current_arity, start, verbose=False):
        # without recomputation only one row survives from the merge
        assert rcb.shape[0] == 1
        local_g["df_cb_stats"].loc[:, "arity"] = current_arity
        local_g["df_cb_stats"].loc[:, "median_omegaCany2spe_fg_traitA"] = 0.9
        local_g["df_cb_stats"].loc[:, "total_OCNany2spe_fg_traitA"] = 3.0
        return local_g

    monkeypatch.setattr(foreground.param, "initialize_df_cb_stats", fake_initialize_df_cb_stats)
    monkeypatch.setattr(foreground, "set_random_foreground_branch", fake_set_random_foreground_branch)
    monkeypatch.setattr(foreground, "add_median_cb_stats", fake_add_median_cb_stats)

    out = foreground.clade_permutation(cb=cb, g=g)

    assert out["df_cb_stats_main"].shape[0] == 1
    captured = capsys.readouterr()
    assert "permuted foreground branch combinations were dropped" in captured.out


def test_report_permutation_clade_permutation_ocn_excludes_inf_from_mean_std(capsys):
    g = {
        "df_cb_stats_main": pd.DataFrame(
            {
                "arity": [2, 2, 2],
                "mode": [
                    "randomization_traitA_iter1",
                    "randomization_traitA_iter2",
                    "randomization_traitA_iter3",
                ],
                "total_OCNany2spe_fg_traitA": [1.0, np.inf, 3.0],
            }
        )
    }
    is_arity_perm = g["df_cb_stats_main"].loc[:, "arity"] == 2
    is_stat_permutation = g["df_cb_stats_main"].loc[:, "mode"].astype(str).str.startswith("randomization_traitA_")
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        foreground._report_permutation_clade_permutation_ocn(
            g=g,
            trait_name="traitA",
            obs_ocn_col="total_OCNany2spe_fg_traitA",
            is_arity_perm=is_arity_perm,
            is_stat_permutation=is_stat_permutation,
        )
    assert not any("invalid value encountered" in str(w.message) for w in captured_warnings)
    out = capsys.readouterr().out
    assert "Trait traitA: Total OCNany2spe in permutation lineages = 3.0; 2.0 ± 1.0" in out


def test_is_valid_clade_permutation_stat_row_handles_empty_stats_dataframe(capsys):
    g = {
        "df_cb_stats": pd.DataFrame(columns=["median_omegaCany2spe_fg_traitA"]),
    }
    ok = foreground._is_valid_clade_permutation_stat_row(
        g=g,
        trait_name="traitA",
        rid_combinations=np.array([[1, 2]], dtype=np.int64),
    )
    assert ok is False
