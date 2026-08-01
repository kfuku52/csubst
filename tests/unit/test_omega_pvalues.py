import numpy as np
import pandas as pd
import pytest

from csubst import omega


def test_get_cb_ids_rejects_non_integer_like_branch_ids():
    cb = pd.DataFrame({"branch_id_1": [1.5], "branch_id_2": [2]})
    with pytest.raises(ValueError, match="integer-like"):
        omega._get_cb_ids(cb)


def test_get_cb_ids_accepts_integer_like_strings():
    cb = pd.DataFrame({"branch_id_1": ["1.0", "2"], "branch_id_2": ["3", "4.0"]})
    out = omega._get_cb_ids(cb)
    np.testing.assert_array_equal(out, np.array([[1, 3], [2, 4]], dtype=np.int64))
    assert out.dtype == np.int64


def test_get_cb_ids_rejects_negative_branch_ids():
    cb = pd.DataFrame({"branch_id_1": [-1], "branch_id_2": [2]})
    with pytest.raises(ValueError, match="non-negative"):
        omega._get_cb_ids(cb)


def test_get_cb_ids_requires_branch_id_columns():
    cb = pd.DataFrame({"other": [1], "value": [2]})
    with pytest.raises(ValueError, match="at least one branch_id_"):
        omega._get_cb_ids(cb)


def test_resolve_hypergeom_parallel_plan_falls_back_for_small_workload():
    n_jobs, chunk_factor = omega._resolve_hypergeom_parallel_plan(
        cb_rows=1,
        num_categories=200,
        niter=1000,
        requested_n_jobs=4,
        requested_chunk_factor=1,
    )
    assert n_jobs == 1
    assert chunk_factor == 1


def test_resolve_hypergeom_parallel_plan_keeps_parallel_for_large_workload():
    n_jobs, chunk_factor = omega._resolve_hypergeom_parallel_plan(
        cb_rows=1000,
        num_categories=300,
        niter=1000,
        requested_n_jobs=4,
        requested_chunk_factor=1,
    )
    assert n_jobs == 4
    assert chunk_factor == 4


def test_resolve_omega_pvalue_niter_schedule_auto_defaults():
    schedule = omega._resolve_omega_pvalue_niter_schedule(g={})
    assert schedule == [100, 1000]


def test_resolve_omega_pvalue_niter_schedule_accepts_auto_string():
    schedule = omega._resolve_omega_pvalue_niter_schedule(g={"omega_pvalue_niter_schedule": "auto"})
    assert schedule == [100, 1000]


def test_resolve_omega_pvalue_niter_schedule_uses_custom_schedule():
    schedule = omega._resolve_omega_pvalue_niter_schedule(g={"omega_pvalue_niter_schedule": [200, 600]})
    assert schedule == [200, 600]


def test_needs_omega_pvalue_upper_tail_edge_refinement():
    refine = omega._needs_omega_pvalue_upper_tail_edge_refinement(
        obs_omega=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        exp_S=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        ge_ranks=np.array([0, 2, 3, 0], dtype=np.int64),
        valid_niter=np.array([100, 100, 100, 0], dtype=np.int64),
        edge_bins=2,
    )
    np.testing.assert_array_equal(refine, np.array([True, True, False, True], dtype=bool))


def test_calc_e_stat_rejects_quantile_stat():
    cb = pd.DataFrame({"branch_id_1": [0], "OCNany2any": [1.0]})
    sub_tensor = np.zeros((2, 3, 1, 2, 2), dtype=np.float64)
    g = {"float_type": np.float64, "threads": 1, "asrv": "each"}
    with pytest.raises(ValueError, match="Unsupported E-stat summary statistic"):
        omega.calc_E_stat(
            cb=cb,
            sub_tensor=sub_tensor,
            mode="any2any",
            stat="quantile",
            SN="N",
            g=g,
        )


def test_get_cod_skips_when_required_columns_missing():
    cb = pd.DataFrame(
        {
            "OCNany2spe": [1.0, 2.0],
            "OCSany2spe": [2.0, 3.0],
        }
    )
    out = omega.get_CoD(cb.copy(), g={"float_tol": 1e-12})
    assert "OCNCoD" not in out.columns
    assert "OCSCoD" not in out.columns


def test_get_cod_maps_zero_over_zero_to_zero_and_keeps_positive_over_zero_infinite():
    cb = pd.DataFrame(
        {
            "OCNany2spe": [0.0, 2.0],
            "OCNany2dif": [0.0, 0.0],
            "OCSany2spe": [0.0, 1.0],
            "OCSany2dif": [0.0, 2.0],
        }
    )
    out = omega.get_CoD(cb.copy(), g={"float_tol": 1e-12})
    np.testing.assert_allclose(out.loc[:, "OCNCoD"].to_numpy(dtype=np.float64), np.array([0.0, np.inf]))
    np.testing.assert_allclose(out.loc[:, "OCSCoD"].to_numpy(dtype=np.float64), np.array([0.0, 0.5]))


def test_calc_dif_count_matrix_marks_impossible_negative_counts_as_undefined():
    any_count = np.array([[4.0, 1.0], [0.0, 2.0]], dtype=np.float64)
    spe_count = np.array([[2.0, 2.0], [1.0, 2.0]], dtype=np.float64)
    out = omega._calc_dif_count_matrix(any_count=any_count, spe_count=spe_count, tol=1e-9)
    expected = np.array([[2.0, np.nan], [np.nan, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_calc_omega_empirical_upper_tail_pvalues_uses_upper_tail_mid_p():
    obs_omega = np.array([2.0, 1.0], dtype=np.float64)
    exp_N = np.array([1.0, 1.0], dtype=np.float64)
    exp_S = np.array([1.0, 1.0], dtype=np.float64)
    perm_count_N = np.array([[2.0, 1.0, 3.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    perm_count_S = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    out = omega._calc_omega_empirical_upper_tail_pvalues(
        obs_omega=obs_omega,
        exp_N=exp_N,
        exp_S=exp_S,
        perm_count_N=perm_count_N,
        perm_count_S=perm_count_S,
        float_tol=1e-12,
    )
    np.testing.assert_allclose(out, np.array([0.75, 0.75], dtype=np.float64))


def test_calc_omega_empirical_upper_tail_pvalues_from_perm_matches_wrapper():
    obs_omega = np.array([2.0, 1.0], dtype=np.float64)
    exp_N = np.array([1.0, 1.0], dtype=np.float64)
    exp_S = np.array([1.0, 1.0], dtype=np.float64)
    perm_count_N = np.array([[2.0, 1.0, 3.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    perm_count_S = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    wrapper_out = omega._calc_omega_empirical_upper_tail_pvalues(
        obs_omega=obs_omega,
        exp_N=exp_N,
        exp_S=exp_S,
        perm_count_N=perm_count_N,
        perm_count_S=perm_count_S,
        float_tol=1e-12,
    )
    perm_omega = omega._calc_permutation_omega_matrix(
        exp_N=exp_N,
        exp_S=exp_S,
        perm_count_N=perm_count_N,
        perm_count_S=perm_count_S,
        float_tol=1e-12,
    )
    from_perm_out = omega._calc_omega_empirical_upper_tail_pvalues_from_perm(
        obs_omega=obs_omega,
        exp_S=exp_S,
        perm_omega=perm_omega,
    )
    np.testing.assert_allclose(wrapper_out, from_perm_out)


def test_calc_omega_empirical_upper_tail_pvalues_supports_dsc_calibrated_null():
    obs_omega = np.array([1.0, 1.0], dtype=np.float64)
    exp_N = np.array([1.0, 1.0], dtype=np.float64)
    exp_S = np.array([1.0, 1.0], dtype=np.float64)
    perm_count_N = np.array([[2.0, 4.0, 6.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    perm_count_S = np.array([[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    wrapper_out = omega._calc_omega_empirical_upper_tail_pvalues(
        obs_omega=obs_omega,
        exp_N=exp_N,
        exp_S=exp_S,
        perm_count_N=perm_count_N,
        perm_count_S=perm_count_S,
        float_tol=1e-12,
        calibrate_dsc_transformation="quantile",
    )
    perm_omega = omega._calc_permutation_omega_matrix(
        exp_N=exp_N,
        exp_S=exp_S,
        perm_count_N=perm_count_N,
        perm_count_S=perm_count_S,
        float_tol=1e-12,
        calibrate_dsc_transformation="quantile",
    )
    perm_omega_raw = omega._calc_permutation_omega_matrix(
        exp_N=exp_N,
        exp_S=exp_S,
        perm_count_N=perm_count_N,
        perm_count_S=perm_count_S,
        float_tol=1e-12,
    )
    from_perm_out = omega._calc_omega_empirical_upper_tail_pvalues_from_perm(
        obs_omega=obs_omega,
        exp_S=exp_S,
        perm_omega=perm_omega,
    )
    assert not np.allclose(perm_omega, perm_omega_raw)
    np.testing.assert_allclose(wrapper_out, from_perm_out)


def test_calc_bh_fdr_qvalues_handles_nan_and_monotonicity():
    pvalues = np.array([0.01, 0.04, 0.03, np.nan], dtype=np.float64)
    out = omega._calc_bh_fdr_qvalues(pvalues)
    expected = np.array([0.03, 0.04, 0.04, np.nan], dtype=np.float64)
    np.testing.assert_allclose(out, expected, equal_nan=True)
