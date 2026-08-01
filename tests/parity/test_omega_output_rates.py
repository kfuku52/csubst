import numpy as np
import pandas as pd
import pytest

from csubst import omega


def test_calibrate_dsc_skips_substitution_class_without_finite_pairs():
    combinatorial_substitutions = [
        "any2any",
        "any2spe",
        "any2dif",
        "dif2any",
        "dif2spe",
        "dif2dif",
        "spe2any",
        "spe2spe",
        "spe2dif",
    ]
    row = {"branch_id_1": 0}
    for sub in combinatorial_substitutions:
        row["dNC" + sub] = np.nan
        row["dSC" + sub] = np.nan
        row["omegaC" + sub] = np.nan
    cb = pd.DataFrame([row])

    out = omega.calibrate_dsc(cb=cb.copy())

    for sub in combinatorial_substitutions:
        assert "dSC" + sub in out.columns
        assert "omegaC" + sub in out.columns
        assert "dSC" + sub + "_nocalib" not in out.columns
        assert "omegaC" + sub + "_nocalib" not in out.columns


def test_get_omega_applies_dirichlet_pseudocount_to_any2spe_columns():
    cb = pd.DataFrame(
        {
            "OCNany2spe": [0.0],
            "ECNany2spe": [0.0],
            "OCSany2spe": [1.0],
            "ECSany2spe": [0.0],
        }
    )
    g = {
        "float_tol": 1e-12,
        "output_stat": "any2spe",
        "pseudocount_alpha": 0.5,
        "pseudocount_mode": "symmetric",
        "pseudocount_target": "both",
        "pseudocount_report": False,
    }
    out = omega.get_omega(cb=cb.copy(), g=g)
    assert np.isfinite(out.loc[0, "omegaCany2spe"])
    assert out.loc[0, "omegaCany2spe"] == pytest.approx(1.0 / 3.0)
    assert out.loc[0, "omegaCany2spe_raw"] == pytest.approx(0.0)
    assert "omegaCany2spe_smoothed" in out.columns
    assert "logomegaCany2spe_smoothed" in out.columns


def test_get_omega_keeps_legacy_behavior_when_pseudocount_disabled():
    cb = pd.DataFrame(
        {
            "OCNany2spe": [2.0],
            "ECNany2spe": [4.0],
            "OCSany2spe": [1.0],
            "ECSany2spe": [2.0],
        }
    )
    g = {
        "float_tol": 1e-12,
        "output_stat": "any2spe",
        "pseudocount_alpha": 0.0,
        "pseudocount_mode": "none",
        "pseudocount_target": "both",
        "pseudocount_report": False,
    }
    out = omega.get_omega(cb=cb.copy(), g=g)
    assert out.loc[0, "dNCany2spe"] == pytest.approx(0.5)
    assert out.loc[0, "dSCany2spe"] == pytest.approx(0.5)
    assert out.loc[0, "omegaCany2spe"] == pytest.approx(1.0)
    assert "omegaCany2spe_raw" not in out.columns


def test_get_omega_accepts_auto_alpha_and_estimates_finite_smoothed_rates():
    cb = pd.DataFrame(
        {
            "OCNany2spe": [0.0, 1.0, 0.0, 3.0],
            "ECNany2spe": [0.0, 1.0, 2.0, 0.0],
            "OCSany2spe": [0.0, 0.0, 1.0, 2.0],
            "ECSany2spe": [0.0, 1.0, 1.0, 0.0],
        }
    )
    g = {
        "float_tol": 1e-12,
        "output_stat": "any2spe",
        "pseudocount_alpha": "auto",
        "pseudocount_mode": "symmetric",
        "pseudocount_target": "both",
        "pseudocount_report": False,
    }
    out = omega.get_omega(cb=cb.copy(), g=g)
    assert np.isfinite(out["dNCany2spe"].to_numpy()).all()
    assert np.isfinite(out["dSCany2spe"].to_numpy()).all()
    assert np.isfinite(out["omegaCany2spe"].to_numpy()).all()


def test_get_omega_accepts_prevalidated_auto_alpha_configuration():
    cb = pd.DataFrame(
        {
            "OCNany2spe": [0.0, 1.0, 0.0, 3.0],
            "ECNany2spe": [0.0, 1.0, 2.0, 0.0],
            "OCSany2spe": [0.0, 0.0, 1.0, 2.0],
            "ECSany2spe": [0.0, 1.0, 1.0, 0.0],
        }
    )
    g = {
        "float_tol": 1e-12,
        "output_stat": "any2spe",
        "pseudocount_alpha": 0.0,
        "pseudocount_alpha_auto": True,
        "pseudocount_mode": "symmetric",
        "pseudocount_target": "both",
        "pseudocount_report": False,
    }
    out = omega.get_omega(cb=cb.copy(), g=g)
    assert "omegaCany2spe_raw" in out.columns
    assert np.isfinite(out["omegaCany2spe"].to_numpy()).all()
    assert not np.allclose(
        out["omegaCany2spe"].to_numpy(dtype=np.float64),
        out["omegaCany2spe_raw"].to_numpy(dtype=np.float64),
        equal_nan=True,
    )


def test_get_omega_empirical_mode_returns_finite_values():
    cb = pd.DataFrame(
        {
            "OCNany2any": [10.0],
            "ECNany2any": [8.0],
            "OCSany2any": [8.0],
            "ECSany2any": [7.0],
            "OCNany2spe": [0.0],
            "ECNany2spe": [3.0],
            "OCSany2spe": [0.0],
            "ECSany2spe": [2.0],
        }
    )
    g = {
        "float_tol": 1e-12,
        "output_stat": "any2any,any2spe",
        "pseudocount_alpha": 0.5,
        "pseudocount_mode": "empirical",
        "pseudocount_target": "both",
        "pseudocount_report": False,
    }
    out = omega.get_omega(cb=cb.copy(), g=g)
    assert np.isfinite(out.loc[0, "omegaCany2any"])
    assert np.isfinite(out.loc[0, "omegaCany2spe"])
