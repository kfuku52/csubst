import numpy as np
import pytest

from csubst import output_stat
from csubst import pseudocount


def test_validate_args_defaults_disable_smoothing():
    out = pseudocount.validate_args({})
    assert out["pseudocount_alpha"] == pytest.approx(0.0)
    assert out["pseudocount_alpha_auto"] is False
    assert out["pseudocount_mode"] == "none"
    assert out["pseudocount_target"] == "both"
    assert out["pseudocount_enabled"] is False
    assert out["pseudocount_add_output_columns"] is False


def test_validate_args_accepts_auto_alpha_token():
    out = pseudocount.validate_args(
        {
            "pseudocount_alpha": "auto",
            "pseudocount_mode": "symmetric",
            "pseudocount_target": "both",
        }
    )
    assert out["pseudocount_alpha_auto"] is True
    assert out["pseudocount_alpha"] == pytest.approx(0.0)
    assert out["pseudocount_enabled"] is True


def test_validate_args_preserves_auto_flag_for_prevalidated_mapping():
    out = pseudocount.validate_args(
        {
            "pseudocount_alpha": 0.0,
            "pseudocount_alpha_auto": True,
            "pseudocount_mode": "symmetric",
            "pseudocount_target": "both",
        }
    )
    assert out["pseudocount_alpha"] == pytest.approx(0.0)
    assert out["pseudocount_alpha_auto"] is True
    assert out["pseudocount_enabled"] is True


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"pseudocount_alpha": -0.1}, "pseudocount_alpha"),
        ({"pseudocount_alpha": np.nan}, "pseudocount_alpha"),
        ({"pseudocount_alpha": "abc"}, "pseudocount_alpha"),
        ({"pseudocount_mode": "weird"}, "pseudocount_mode"),
        ({"pseudocount_target": "all"}, "pseudocount_target"),
    ],
)
def test_validate_args_rejects_invalid_values(kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        pseudocount.validate_args(kwargs)


def test_validate_args_rejects_removed_strength_parameter():
    with pytest.raises(ValueError, match="pseudocount_strength"):
        pseudocount.validate_args({"pseudocount_strength": 2.0})


def test_smooth_ratio_alpha_zero_matches_raw_ratio():
    O = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    E = np.array([1.0, 2.0, 4.0, 6.0], dtype=np.float64)
    out = pseudocount.smooth_ratio(O=O, E=E, alpha_obs=0.0, alpha_exp=0.0)
    np.testing.assert_allclose(out, O / E, atol=1e-12)


def test_smooth_ratio_alpha_zero_maps_zero_over_zero_to_zero_only():
    O = np.array([0.0, 1.0], dtype=np.float64)
    E = np.array([0.0, 0.0], dtype=np.float64)
    out = pseudocount.smooth_ratio(O=O, E=E, alpha_obs=0.0, alpha_exp=0.0)
    assert out[0] == pytest.approx(0.0)
    assert np.isinf(out[1])


def test_smooth_ratio_zero_zero_is_finite_with_positive_alpha():
    out = pseudocount.smooth_ratio(O=0.0, E=0.0, alpha_obs=0.5, alpha_exp=0.5)
    assert np.isfinite(out)
    assert out == pytest.approx(1.0)


def test_smooth_ratio_zero_positive_matches_closed_form_for_target_both():
    out = pseudocount.smooth_ratio(O=0.0, E=2.0, alpha_obs=0.5, alpha_exp=0.5)
    assert out == pytest.approx(0.5 / 2.5)


def test_empirical_alpha_vector_and_smoothed_probabilities_are_normalized():
    p_global = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    alpha_k = pseudocount.compute_alpha_vector(
        mode="empirical",
        alpha=2.0,
        p_global=p_global,
        K=3,
    )
    assert alpha_k.sum() == pytest.approx(2.0)
    n_k = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    p_hat = pseudocount.smooth_probs(n_k=n_k, alpha_k=alpha_k)
    assert p_hat.sum() == pytest.approx(1.0)
    assert (p_hat >= 0).all()


def test_smoothed_ratios_are_finite_and_monotonic_toward_one():
    rng = np.random.default_rng(7)
    O = rng.poisson(lam=0.3, size=512).astype(np.float64)
    E = rng.poisson(lam=0.3, size=512).astype(np.float64)
    r_small = pseudocount.smooth_ratio(O=O, E=E, alpha_obs=0.1, alpha_exp=0.1)
    r_large = pseudocount.smooth_ratio(O=O, E=E, alpha_obs=1.0, alpha_exp=1.0)
    assert np.isfinite(r_small).all()
    assert np.isfinite(r_large).all()
    assert (np.abs(r_large - 1.0) <= (np.abs(r_small - 1.0) + 1e-12)).all()


def test_estimate_alpha_empirical_bayes_returns_positive_finite_alpha():
    obs = np.array([0, 0, 1, 0, 2, 0, 5, 0], dtype=np.float64)
    exp = np.array([0, 1, 1, 2, 2, 5, 5, 10], dtype=np.float64)
    alpha, diag = pseudocount.estimate_alpha_empirical_bayes(obs_list=[obs], exp_list=[exp], float_tol=1e-12)
    assert np.isfinite(alpha)
    assert alpha > 0
    assert diag["num_pairs"] == obs.shape[0]
    assert np.isfinite(diag["fit_loglikelihood"])


def test_empirical_stat_alphas_respect_overlap_identities_when_stats_are_added():
    masses = {
        "any2any": 100.0,
        "any2dif": 70.0,
        "any2spe": 30.0,
        "dif2spe": 10.0,
        "spe2spe": 20.0,
    }
    stats_base = ["any2any", "any2dif", "any2spe"]
    stats_extended = ["any2any", "any2dif", "any2spe", "dif2spe"]
    alpha_base, _ = pseudocount.compute_empirical_stat_alphas(
        stat_masses=masses,
        stats=stats_base,
        alpha=1.0,
    )
    alpha_extended, _ = pseudocount.compute_empirical_stat_alphas(
        stat_masses=masses,
        stats=stats_extended,
        alpha=1.0,
    )
    np.testing.assert_allclose(alpha_base, alpha_extended[:3], atol=1e-12)


def test_empirical_stat_alphas_preserve_additive_identities():
    masses = {
        "spe2spe": 12.0,
        "spe2dif": 18.0,
        "dif2spe": 30.0,
        "dif2dif": 40.0,
        "any2spe": 42.0,
        "any2dif": 58.0,
        "spe2any": 30.0,
        "dif2any": 70.0,
        "any2any": 100.0,
    }
    stats = [
        "any2any",
        "any2dif",
        "any2spe",
        "spe2any",
        "spe2spe",
        "spe2dif",
        "dif2any",
        "dif2spe",
        "dif2dif",
    ]
    alpha_stats, _ = pseudocount.compute_empirical_stat_alphas(
        stat_masses=masses,
        stats=stats,
        alpha=1.0,
    )
    alpha_map = {name: value for name, value in zip(stats, alpha_stats)}
    assert alpha_map["any2any"] == pytest.approx(1.0)
    assert alpha_map["any2any"] == pytest.approx(alpha_map["any2dif"] + alpha_map["any2spe"])
    assert alpha_map["any2any"] == pytest.approx(alpha_map["spe2any"] + alpha_map["dif2any"])
    assert alpha_map["spe2any"] == pytest.approx(alpha_map["spe2spe"] + alpha_map["spe2dif"])
    assert alpha_map["dif2any"] == pytest.approx(alpha_map["dif2spe"] + alpha_map["dif2dif"])
    assert alpha_map["any2spe"] == pytest.approx(alpha_map["spe2spe"] + alpha_map["dif2spe"])
    assert alpha_map["any2dif"] == pytest.approx(alpha_map["spe2dif"] + alpha_map["dif2dif"])


def test_output_stat_atomic_weight_map_covers_all_output_stats():
    assert set(output_stat.STAT_TO_ATOMIC_WEIGHTS.keys()) == set(output_stat.ALL_OUTPUT_STATS)
    for stat_name in output_stat.ALL_OUTPUT_STATS:
        weights = output_stat.STAT_TO_ATOMIC_WEIGHTS[stat_name]
        assert len(weights) == len(output_stat.ATOMIC_OUTPUT_STATS)
