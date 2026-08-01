import numpy as np
import pandas as pd
import pytest

from csubst import omega


def test_add_omega_empirical_pvalues_supports_dif_stats(monkeypatch):
    cb = pd.DataFrame(
        {
            "branch_id_1": [0, 1],
            "branch_id_2": [2, 3],
            "omegaCany2dif": [2.0, 0.0],
            "ECNany2dif": [1.0, 1.0],
            "ECSany2dif": [1.0, 1.0],
        }
    )
    mode_counts = {
        ("N", "any2any"): np.array([[4.0, 2.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64),
        ("N", "any2spe"): np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]], dtype=np.float64),
        ("S", "any2any"): np.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]], dtype=np.float64),
        ("S", "any2spe"): np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64),
    }
    calls = []

    def fake_get_mode_permutation_count_matrix(cb_ids, sub_tensor, mode, SN, niter, g, obs_count=None):
        calls.append((SN, mode, int(niter), cb_ids.shape))
        return mode_counts[(SN, mode)]

    monkeypatch.setattr(omega, "_get_mode_permutation_count_matrix", fake_get_mode_permutation_count_matrix)
    out = omega.add_omega_empirical_pvalues(
        cb=cb.copy(),
        ON_tensor=None,
        OS_tensor=None,
        g={
            "calc_omega_pvalue": True,
            "expectation_method": "urn",
            "omega_pvalue_niter_schedule": [3],
            "output_stats": ["any2dif"],
            "float_tol": 1e-12,
        },
    )
    assert calls == [
        ("N", "any2any", 3, (2, 2)),
        ("S", "any2any", 3, (2, 2)),
        ("N", "any2spe", 3, (2, 2)),
        ("S", "any2spe", 3, (2, 2)),
    ]
    np.testing.assert_allclose(out.loc[:, "pomegaCany2dif"].to_numpy(dtype=np.float64), np.array([0.5, 1.0]))
    np.testing.assert_allclose(out.loc[:, "qomegaCany2dif"].to_numpy(dtype=np.float64), np.array([1.0, 1.0]))


def test_add_omega_empirical_pvalues_hypergeom_refines_only_upper_edge_rows(monkeypatch):
    cb = pd.DataFrame(
        {
            "branch_id_1": [0, 1, 2],
            "branch_id_2": [3, 4, 5],
            "omegaCany2spe": [3.0, 1.0, 2.0],
            "ECNany2spe": [1.0, 1.0, 1.0],
            "ECSany2spe": [1.0, 1.0, 1.0],
        }
    )
    calls = list()

    def fake_get_mode_permutation_count_matrix(cb_ids, sub_tensor, mode, SN, niter, g, obs_count=None):
        calls.append((SN, mode, int(niter), cb_ids.shape[0]))
        if mode != "any2spe":
            raise AssertionError("unexpected mode")
        if SN == "S":
            return np.ones((cb_ids.shape[0], int(niter)), dtype=np.float64)
        if (int(niter) == 100) and (cb_ids.shape[0] == 3):
            out = np.zeros((3, 100), dtype=np.float64)
            out[1, :] = 2.0
            out[2, :95] = 1.0
            out[2, 95:] = 3.0
            return out
        if (int(niter) == 900) and (cb_ids.shape[0] == 1):
            return np.zeros((1, 900), dtype=np.float64)
        raise AssertionError("unexpected staged request")

    monkeypatch.setattr(omega, "_get_mode_permutation_count_matrix", fake_get_mode_permutation_count_matrix)
    out = omega.add_omega_empirical_pvalues(
        cb=cb.copy(),
        ON_tensor=None,
        OS_tensor=None,
        g={
            "calc_omega_pvalue": True,
            "expectation_method": "urn",
            "omega_pvalue_null_model": "hypergeom",
            "omega_pvalue_niter_schedule": [100, 1000],
            "omega_pvalue_refine_upper_edge_bins": 2,
            "output_stats": ["any2spe"],
            "float_tol": 1e-12,
        },
    )
    assert calls == [
        ("N", "any2spe", 100, 3),
        ("S", "any2spe", 100, 3),
        ("N", "any2spe", 900, 1),
        ("S", "any2spe", 900, 1),
    ]
    np.testing.assert_allclose(
        out.loc[:, "pomegaCany2spe"].to_numpy(dtype=np.float64),
        np.array([(0.0 + 1.0) / (1000.0 + 1.0), 1.0, (5.0 + 1.0) / (100.0 + 1.0)], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    assert float(out.loc[0, "pomegaCany2spe"]) < float(out.loc[2, "pomegaCany2spe"])


def test_add_omega_empirical_pvalues_uses_dsc_calibrated_null_when_columns_present(monkeypatch):
    cb = pd.DataFrame(
        {
            "branch_id_1": [0, 1],
            "branch_id_2": [2, 3],
            "omegaCany2spe": [1.0, 1.0],
            "omegaCany2spe_nocalib": [2.0, 1.0],
            "dSCany2spe": [2.0, 2.0],
            "dSCany2spe_nocalib": [1.0, 1.0],
            "ECNany2spe": [1.0, 1.0],
            "ECSany2spe": [1.0, 1.0],
        }
    )
    mode_counts = {
        ("N", "any2spe"): np.array([[2.0, 4.0, 6.0], [1.0, 1.0, 1.0]], dtype=np.float64),
        ("S", "any2spe"): np.array([[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64),
    }

    def fake_get_mode_permutation_count_matrix(cb_ids, sub_tensor, mode, SN, niter, g, obs_count=None):
        return mode_counts[(SN, mode)]

    monkeypatch.setattr(omega, "_get_mode_permutation_count_matrix", fake_get_mode_permutation_count_matrix)
    out = omega.add_omega_empirical_pvalues(
        cb=cb.copy(),
        ON_tensor=None,
        OS_tensor=None,
        g={
            "calc_omega_pvalue": True,
            "expectation_method": "urn",
            "omega_pvalue_niter_schedule": [3],
            "output_stats": ["any2spe"],
            "float_tol": 1e-12,
            "calibrate_longtail_transformation": "quantile",
        },
    )
    np.testing.assert_allclose(
        out.loc[:, "pomegaCany2spe"].to_numpy(dtype=np.float64),
        np.array([1.0, 0.25], dtype=np.float64),
    )
    np.testing.assert_allclose(
        out.loc[:, "qomegaCany2spe"].to_numpy(dtype=np.float64),
        np.array([1.0, 0.5], dtype=np.float64),
    )


def test_calibrate_dsc_renames_empirical_pq_columns_to_nocalib():
    cb = pd.DataFrame(
        {
            "branch_id_1": [0, 1],
            "branch_id_2": [2, 3],
            "dNCany2spe": [2.0, 1.0],
            "dSCany2spe": [1.0, 2.0],
            "omegaCany2spe": [2.0, 0.5],
            "pomegaCany2spe": [0.05, 0.20],
            "qomegaCany2spe": [0.10, 0.20],
        }
    )
    out = omega.calibrate_dsc(cb.copy(), output_stats=["any2spe"])
    assert "pomegaCany2spe" not in out.columns
    assert "qomegaCany2spe" not in out.columns
    assert "pomegaCany2spe_nocalib" in out.columns
    assert "qomegaCany2spe_nocalib" in out.columns
    np.testing.assert_allclose(
        out.loc[:, "pomegaCany2spe_nocalib"].to_numpy(dtype=np.float64),
        np.array([0.05, 0.20], dtype=np.float64),
    )
    np.testing.assert_allclose(
        out.loc[:, "qomegaCany2spe_nocalib"].to_numpy(dtype=np.float64),
        np.array([0.10, 0.20], dtype=np.float64),
    )


def test_calibrate_dsc_sets_zero_for_zero_over_zero():
    cb = pd.DataFrame(
        {
            "branch_id_1": [0, 1],
            "branch_id_2": [2, 3],
            "dNCany2spe": [0.0, 0.0],
            "dSCany2spe": [0.0, 1.0],
            "omegaCany2spe": [np.nan, np.nan],
        }
    )
    out = omega.calibrate_dsc(
        cb.copy(),
        output_stats=["any2spe"],
        float_tol=1e-12,
    )
    np.testing.assert_allclose(
        out.loc[:, "omegaCany2spe"].to_numpy(dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        atol=0.0,
    )
    assert np.isfinite(out.loc[:, "omegaCany2spe"].to_numpy(dtype=np.float64)).all()


def test_calibrate_dsc_excludes_infinite_rows_from_fit_and_preserves_original_omega():
    cb = pd.DataFrame(
        {
            "branch_id_1": [0, 1, 2],
            "branch_id_2": [3, 4, 5],
            "dNCany2spe": [1.0, np.inf, 0.0],
            "dSCany2spe": [1.0, 1.0, 0.0],
            "omegaCany2spe": [1.0, np.inf, np.nan],
        }
    )
    out = omega.calibrate_dsc(
        cb.copy(),
        output_stats=["any2spe"],
        float_tol=1e-12,
    )
    assert np.isinf(out.loc[1, "omegaCany2spe_nocalib"])
    assert np.isinf(out.loc[1, "omegaCany2spe"])
    assert out.loc[1, "dSCany2spe"] == pytest.approx(1.0)
