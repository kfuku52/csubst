import numpy as np
import pandas as pd
import pytest

from csubst import omega
from csubst import tree
from csubst import ete


def test_weighted_sample_without_replacement_masks_excludes_zero_probability_sites():
    p = np.array([0.7, 0.3, 0.0, 0.0], dtype=np.float64)
    masks = omega._weighted_sample_without_replacement_masks(p=p, size=2, niter=128)

    assert masks.shape == (128, 4)
    assert masks.dtype == bool
    assert masks[:, 2].sum() == 0
    assert masks[:, 3].sum() == 0
    np.testing.assert_array_equal(masks.sum(axis=1), np.full((128,), 2))


def test_weighted_sample_without_replacement_masks_raises_when_size_exceeds_positive_sites():
    p = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    with pytest.raises(ValueError):
        omega._weighted_sample_without_replacement_masks(p=p, size=2, niter=8)


def test_get_permutations_fast_returns_expected_shape_and_bounds():
    cb_ids = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
    sub_branches = np.array([2, 3, 5], dtype=np.int64)
    p = np.array([0.2, 0.1, 0.25, 0.15, 0.1, 0.1, 0.05, 0.05], dtype=np.float64)

    out = omega._get_permutations_fast(
        cb_ids=cb_ids,
        sub_branches=sub_branches,
        p=p / p.sum(),
        niter=256,
    )

    assert out.shape == (3, 256)
    assert out.dtype == np.int32
    assert out.min() >= 0
    assert out[0, :].max() <= min(sub_branches[0], sub_branches[1])
    assert out[1, :].max() <= min(sub_branches[1], sub_branches[2])
    assert out[2, :].max() <= min(sub_branches[0], sub_branches[2])


def test_get_permutations_fast_caps_oversized_branch_counts_to_positive_sites():
    cb_ids = np.array([[0, 1]], dtype=np.int64)
    sub_branches = np.array([5, 2], dtype=np.int64)
    p = np.array([0.0, 0.6, 0.4], dtype=np.float64)

    out = omega._get_permutations_fast(
        cb_ids=cb_ids,
        sub_branches=sub_branches,
        p=p,
        niter=128,
    )

    assert out.shape == (1, 128)
    assert out.dtype == np.int32
    assert out.min() >= 0
    assert out.max() <= 2


def test_get_permutations_fast_uses_branch_specific_probabilities_when_given_matrix():
    cb_ids = np.array([[1, 2], [2, 3]], dtype=np.int64)
    sub_branches = np.array([0, 2, 2, 2], dtype=np.int64)
    p_by_branch = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],  # branch 0 has no informative site weights
            [0.4, 0.3, 0.2, 0.1],
            [0.4, 0.3, 0.2, 0.1],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float64,
    )

    np.random.seed(0)
    out_branch_specific = omega._get_permutations_fast(
        cb_ids=cb_ids,
        sub_branches=sub_branches,
        p=p_by_branch,
        niter=64,
    )
    np.random.seed(0)
    out_branch0_only = omega._get_permutations_fast(
        cb_ids=cb_ids,
        sub_branches=sub_branches,
        p=p_by_branch[0, :],
        niter=64,
    )

    assert out_branch_specific.shape == (2, 64)
    assert out_branch_specific.dtype == np.int32
    assert out_branch_specific.max() <= 2
    assert out_branch_specific.sum() > 0
    assert out_branch0_only.sum() == 0


def test_get_permutations_fast_rejects_branch_probability_row_mismatch():
    cb_ids = np.array([[0, 1]], dtype=np.int64)
    sub_branches = np.array([1, 2], dtype=np.int64)
    p = np.ones((3, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="number of rows"):
        omega._get_permutations_fast(cb_ids=cb_ids, sub_branches=sub_branches, p=p, niter=16)


def test_get_permutations_fast_rejects_negative_branch_ids():
    cb_ids = np.array([[0, -1]], dtype=np.int64)
    sub_branches = np.array([2, 3], dtype=np.int64)
    p = np.array([0.5, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="non-negative"):
        omega._get_permutations_fast(cb_ids=cb_ids, sub_branches=sub_branches, p=p, niter=8)


def test_prepare_permutation_branch_sizes_supports_stochastic_mode():
    sub_branches = np.array([0.2, 1.6, 2.0], dtype=np.float64)
    np.random.seed(0)
    out = omega._prepare_permutation_branch_sizes(
        sub_branches=sub_branches,
        niter=64,
        g={"omega_pvalue_rounding": "stochastic"},
    )
    assert out.shape == (3, 64)
    assert out.dtype == np.int64
    assert set(np.unique(out[0, :]).tolist()).issubset({0, 1})
    assert set(np.unique(out[1, :]).tolist()).issubset({1, 2})
    assert np.all(out[2, :] == 2)


def test_calc_wallenius_inclusion_probabilities_matches_sampling():
    p = np.array([0.5, 0.25, 0.15, 0.10, 0.0], dtype=np.float64)
    draw_size = 3
    expected = omega._calc_wallenius_inclusion_probabilities(
        site_weights=p,
        draw_size=draw_size,
        float_type=np.float64,
    )
    np.random.seed(2)
    masks = omega._weighted_sample_without_replacement_masks(p=p, size=draw_size, niter=16000)
    empirical = masks.mean(axis=0)
    np.testing.assert_allclose(expected, empirical, atol=0.03)
    np.testing.assert_allclose(expected.sum(), float(draw_size), atol=1e-8)


def test_calc_wallenius_expected_overlap_matches_permutation_mean_for_skewed_weights():
    cb_ids = np.array([[0, 1]], dtype=np.int64)
    sub_branches = np.array([80.0, 80.0], dtype=np.float64)
    p = np.zeros(100, dtype=np.float64)
    p[0] = 0.9
    p[1:] = 0.1 / 99.0

    expected = omega._calc_wallenius_expected_overlap(
        cb_ids=cb_ids,
        sub_sites=p,
        sub_branches=sub_branches,
        g={"omega_pvalue_rounding": "round"},
        float_type=np.float64,
    )

    np.random.seed(3)
    sizes = omega._prepare_permutation_branch_sizes(
        sub_branches=sub_branches,
        niter=1,
        g={"omega_pvalue_rounding": "round"},
    )
    perm = omega._get_permutations_fast(
        cb_ids=cb_ids,
        sub_branches=sizes,
        p=p,
        niter=20000,
    )
    perm_mean = float(perm.mean())
    assert expected.shape == (1,)
    assert expected[0] <= 80.0 + 1e-9
    np.testing.assert_allclose(expected[0], perm_mean, atol=0.6)


def test_calc_wallenius_expected_overlap_supports_stochastic_rounding():
    cb_ids = np.array([[0, 1]], dtype=np.int64)
    sub_branches = np.array([3.4, 2.6], dtype=np.float64)
    p = np.array([0.35, 0.30, 0.20, 0.10, 0.05], dtype=np.float64)
    g = {"omega_pvalue_rounding": "stochastic"}

    expected = omega._calc_wallenius_expected_overlap(
        cb_ids=cb_ids,
        sub_sites=p,
        sub_branches=sub_branches,
        g=g,
        float_type=np.float64,
    )
    np.random.seed(5)
    size_by_iter = omega._prepare_permutation_branch_sizes(
        sub_branches=sub_branches,
        niter=20000,
        g=g,
    )
    perm = omega._get_permutations_fast(
        cb_ids=cb_ids,
        sub_branches=size_by_iter,
        p=p,
        niter=20000,
    )
    np.testing.assert_allclose(expected[0], float(perm.mean()), atol=0.08)


def test_get_permutations_fast_accepts_per_iteration_branch_sizes():
    cb_ids = np.array([[0, 1], [1, 2]], dtype=np.int64)
    sub_branches = np.array(
        [
            [1, 0, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.int64,
    )
    p = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    np.random.seed(3)
    out = omega._get_permutations_fast(cb_ids=cb_ids, sub_branches=sub_branches, p=p, niter=4)
    assert out.shape == (2, 4)
    assert out.dtype == np.int32
    assert out[0, 1] == 0  # branch 0 size=0 in iteration 2
    assert out[1, 2] == 0  # branch 2 size=0 in iteration 3
    assert out[1, 3] == 0  # branch 2 size=0 in iteration 4


def test_get_permutations_fast_rejects_branch_size_matrix_niter_mismatch():
    cb_ids = np.array([[0, 1]], dtype=np.int64)
    sub_branches = np.array([[1, 1], [1, 1]], dtype=np.int64)
    p = np.array([0.5, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="number of columns"):
        omega._get_permutations_fast(cb_ids=cb_ids, sub_branches=sub_branches, p=p, niter=3)


def test_weighted_sample_without_replacement_packed_matches_mask_packbits():
    p = np.array([0.4, 0.25, 0.2, 0.15, 0.0], dtype=np.float64)
    np.random.seed(11)
    packed = omega._weighted_sample_without_replacement_packed(p=p, size=2, niter=64)
    np.random.seed(11)
    masks = omega._weighted_sample_without_replacement_masks(p=p, size=2, niter=64)
    expected = np.packbits(masks, axis=1)
    np.testing.assert_array_equal(packed, expected)


def test_weighted_sample_without_replacement_packed_handles_full_positive_sites():
    p = np.array([0.0, 0.3, 0.2, 0.5], dtype=np.float64)
    packed = omega._weighted_sample_without_replacement_packed(p=p, size=3, niter=8)
    masks = np.unpackbits(packed, axis=1)[:, :p.shape[0]].astype(bool)
    expected = np.array([False, True, True, True], dtype=bool)
    for row in masks:
        np.testing.assert_array_equal(row, expected)


def test_pack_sampled_site_indices_to_uint8_can_use_cython(monkeypatch):
    class DummyOmegaCy:
        def __init__(self):
            self.called = 0

        def pack_sampled_site_indices_uint8(self, sampled_site_indices, num_site):
            self.called += 1
            out = np.zeros((sampled_site_indices.shape[0], (num_site + 7) // 8), dtype=np.uint8)
            for i in range(sampled_site_indices.shape[0]):
                for site in sampled_site_indices[i, :]:
                    out[i, int(site) >> 3] |= np.uint8(1 << (7 - (int(site) & 7)))
            return out

    sampled = np.array([[0, 3, 4], [1, 2, 7]], dtype=np.int64)
    dummy = DummyOmegaCy()
    monkeypatch.setattr(omega, "omega_cy", dummy)
    out_cy = omega._pack_sampled_site_indices_to_uint8(sampled_site_indices=sampled, num_site=8)
    assert dummy.called == 1

    monkeypatch.setattr(omega, "omega_cy", None)
    out_np = omega._pack_sampled_site_indices_to_uint8(sampled_site_indices=sampled, num_site=8)
    np.testing.assert_array_equal(out_cy, out_np)


def test_get_permutations_fast_can_use_cython_packed_shared_counts(monkeypatch):
    class DummyOmegaCy:
        def __init__(self):
            self.called = 0

        def calc_shared_counts_packed_uint8(self, packed_masks, remapped_cb_ids):
            self.called += 1
            arity = remapped_cb_ids.shape[1]
            if arity == 1:
                out = omega._UINT8_POPCOUNT[packed_masks[remapped_cb_ids[:, 0], :, :]].sum(axis=2, dtype=np.int32)
            elif arity == 2:
                out = omega._UINT8_POPCOUNT[np.bitwise_and(
                    packed_masks[remapped_cb_ids[:, 0], :, :],
                    packed_masks[remapped_cb_ids[:, 1], :, :],
                )].sum(axis=2, dtype=np.int32)
            else:
                shared = packed_masks[remapped_cb_ids[:, 0], :, :].copy()
                for col in range(1, arity):
                    shared = np.bitwise_and(shared, packed_masks[remapped_cb_ids[:, col], :, :])
                out = omega._UINT8_POPCOUNT[shared].sum(axis=2, dtype=np.int32)
            return out.astype(np.int32, copy=False)

    cb_ids = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
    sub_branches = np.array([2, 3, 5], dtype=np.int64)
    p = np.array([0.2, 0.1, 0.25, 0.15, 0.1, 0.1, 0.05, 0.05], dtype=np.float64)

    dummy = DummyOmegaCy()
    monkeypatch.setattr(omega, "omega_cy", dummy)
    np.random.seed(7)
    out_cy = omega._get_permutations_fast(
        cb_ids=cb_ids,
        sub_branches=sub_branches,
        p=p / p.sum(),
        niter=128,
    )
    assert dummy.called > 0

    monkeypatch.setattr(omega, "omega_cy", None)
    np.random.seed(7)
    out_np = omega._get_permutations_fast(
        cb_ids=cb_ids,
        sub_branches=sub_branches,
        p=p / p.sum(),
        niter=128,
    )
    np.testing.assert_array_equal(out_cy, out_np)


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


def test_calc_dif_count_matrix_marks_negative_counts_as_nan():
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


def test_calc_poisson_count_matrix_matches_expected_means():
    cb_ids = np.array([[0, 1], [1, 2]], dtype=np.int64)
    sub_bg = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    sub_sg = np.zeros((2, 1), dtype=np.float64)
    static_sub_sites = np.array(
        [
            [0.6, 0.4],
            [0.5, 0.5],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )
    list_igad = [[0, 0, "any2", "2any"]]
    expected_mean = omega._calc_wallenius_expected_overlap(
        cb_ids=cb_ids,
        sub_sites=static_sub_sites,
        sub_branches=sub_bg[:, 0],
        g={"omega_pvalue_rounding": "round"},
        float_type=np.float64,
    )
    np.random.seed(7)
    out = omega._calc_poisson_count_matrix(
        mode="any2any",
        cb_ids=cb_ids,
        sub_sg=sub_sg,
        sub_bg=sub_bg,
        niter=4000,
        obs_col="OCNany2any",
        num_gad_combinat=1,
        list_igad=list_igad,
        g={"float_tol": 1e-12},
        static_sub_sites=static_sub_sites,
    )
    assert out.shape == (2, 4000)
    assert out.dtype == np.float64
    assert np.all(out >= 0)
    np.testing.assert_allclose(out, np.round(out), atol=0.0)
    np.testing.assert_allclose(out.mean(axis=1), expected_mean, atol=0.12)


def test_calc_poisson_count_matrix_uses_wallenius_mean_for_skewed_weights():
    cb_ids = np.array([[0, 1]], dtype=np.int64)
    sub_bg = np.array([[80.0], [80.0]], dtype=np.float64)
    sub_sg = np.zeros((100, 1), dtype=np.float64)
    static_sub_sites = np.zeros((2, 100), dtype=np.float64)
    static_sub_sites[:, 0] = 0.9
    static_sub_sites[:, 1:] = 0.1 / 99.0
    list_igad = [[0, 0, "any2", "2any"]]
    expected_mean = omega._calc_wallenius_expected_overlap(
        cb_ids=cb_ids,
        sub_sites=static_sub_sites,
        sub_branches=sub_bg[:, 0],
        g={"omega_pvalue_rounding": "round"},
        float_type=np.float64,
    )
    legacy_mean = omega._calc_tmp_E_sum(
        cb_ids=cb_ids,
        sub_sites=static_sub_sites,
        sub_branches=sub_bg[:, 0],
        float_type=np.float64,
    )
    np.random.seed(17)
    out = omega._calc_poisson_count_matrix(
        mode="any2any",
        cb_ids=cb_ids,
        sub_sg=sub_sg,
        sub_bg=sub_bg,
        niter=8000,
        obs_col="OCNany2any",
        num_gad_combinat=1,
        list_igad=list_igad,
        g={"float_tol": 1e-12},
        static_sub_sites=static_sub_sites,
    )
    assert out.shape == (1, 8000)
    np.testing.assert_allclose(out.mean(axis=1), expected_mean, atol=0.35)
    assert abs(float(out.mean()) - float(legacy_mean[0])) > 1000.0


def test_calc_poisson_full_count_matrix_matches_expected_means():
    cb_ids = np.array([[0, 1], [1, 2]], dtype=np.int64)
    sub_tensor = np.zeros((3, 2, 1, 2, 2), dtype=np.float64)
    sub_tensor[:, :, 0, 0, 1] = np.array(
        [
            [0.4, 0.6],
            [0.3, 0.7],
            [0.8, 0.2],
        ],
        dtype=np.float64,
    )
    list_igad = [[0, 0, "any2", "2any"]]
    site_mass = sub_tensor[:, :, 0, :, :].sum(axis=(2, 3))
    branch_totals = site_mass.sum(axis=1, dtype=np.float64)
    site_probs = np.zeros_like(site_mass, dtype=np.float64)
    nz = (branch_totals > 0)
    site_probs[nz, :] = site_mass[nz, :] / branch_totals[nz, None]
    expected_mean = omega._calc_wallenius_expected_overlap(
        cb_ids=cb_ids,
        sub_sites=site_probs,
        sub_branches=branch_totals,
        g={"omega_pvalue_rounding": "round"},
        float_type=np.float64,
    )
    np.random.seed(11)
    out = omega._calc_poisson_full_count_matrix(
        mode="any2any",
        cb_ids=cb_ids,
        sub_tensor=sub_tensor,
        niter=4000,
        obs_col="OCNany2any",
        num_gad_combinat=1,
        list_igad=list_igad,
        g={"omega_pvalue_rounding": "round"},
    )
    assert out.shape == (2, 4000)
    assert out.dtype == np.float64
    assert np.all(out >= 0)
    np.testing.assert_allclose(out, np.round(out), atol=0.0)
    np.testing.assert_allclose(out.mean(axis=1), expected_mean, atol=0.06)


def test_calc_poisson_full_count_matrix_uses_wallenius_mean_for_skewed_weights():
    cb_ids = np.array([[0, 1]], dtype=np.int64)
    sub_tensor = np.zeros((2, 100, 1, 1, 1), dtype=np.float64)
    p = np.zeros(100, dtype=np.float64)
    p[0] = 0.9
    p[1:] = 0.1 / 99.0
    sub_tensor[:, :, 0, 0, 0] = 80.0 * np.broadcast_to(p.reshape(1, -1), (2, p.shape[0]))
    list_igad = [[0, 0, "any2", "2any"]]
    site_mass = sub_tensor[:, :, 0, :, :].sum(axis=(2, 3))
    branch_totals = site_mass.sum(axis=1, dtype=np.float64)
    site_probs = np.zeros_like(site_mass, dtype=np.float64)
    nz = (branch_totals > 0)
    site_probs[nz, :] = site_mass[nz, :] / branch_totals[nz, None]
    expected_mean = omega._calc_wallenius_expected_overlap(
        cb_ids=cb_ids,
        sub_sites=site_probs,
        sub_branches=branch_totals,
        g={"omega_pvalue_rounding": "round"},
        float_type=np.float64,
    )
    legacy_mean = omega._calc_tmp_E_sum(
        cb_ids=cb_ids,
        sub_sites=site_mass,
        sub_branches=np.ones(shape=(site_mass.shape[0],), dtype=np.float64),
        float_type=np.float64,
    )
    np.random.seed(19)
    out = omega._calc_poisson_full_count_matrix(
        mode="any2any",
        cb_ids=cb_ids,
        sub_tensor=sub_tensor,
        niter=8000,
        obs_col="OCNany2any",
        num_gad_combinat=1,
        list_igad=list_igad,
        g={"omega_pvalue_rounding": "round"},
    )
    assert out.shape == (1, 8000)
    np.testing.assert_allclose(out.mean(axis=1), expected_mean, atol=0.35)
    assert abs(float(out.mean()) - float(legacy_mean[0])) > 1000.0


def test_get_mode_permutation_count_matrix_uses_poisson_model(monkeypatch):
    cb_ids = np.array([[0, 1]], dtype=np.int64)
    expected = np.full((1, 5), 2.0, dtype=np.float64)

    def fake_prepare(sub_tensor, mode, SN, g):
        return np.zeros((2, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64), [[0, 0, "any2", "2any"]], "OCNany2any", 1

    def fake_static(g, sub_sg, mode, obs_col):
        return np.ones((2, 3), dtype=np.float64) / 3.0

    def fake_poisson(mode, cb_ids, sub_sg, sub_bg, niter, obs_col, num_gad_combinat, list_igad, g, static_sub_sites):
        return expected.copy()

    def fake_quantile(*args, **kwargs):
        raise AssertionError("hypergeom path should not be used for poisson null model")

    monkeypatch.setattr(omega, "_prepare_substitution_permutation_components", fake_prepare)
    monkeypatch.setattr(omega, "_get_static_sub_sites_if_available", fake_static)
    monkeypatch.setattr(omega, "_calc_poisson_count_matrix", fake_poisson)
    monkeypatch.setattr(omega, "_calc_hypergeom_count_matrix", fake_quantile)

    out = omega._get_mode_permutation_count_matrix(
        cb_ids=cb_ids,
        sub_tensor=None,
        mode="any2any",
        SN="N",
        niter=5,
        g={"omega_pvalue_null_model": "poisson"},
    )
    np.testing.assert_allclose(out, expected)


def test_get_mode_permutation_count_matrix_uses_poisson_full_model(monkeypatch):
    cb_ids = np.array([[0, 1]], dtype=np.int64)
    expected = np.full((1, 5), 3.0, dtype=np.float64)

    def fake_prepare(sub_tensor, mode, SN, g):
        return np.zeros((2, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64), [[0, 0, "any2", "2any"]], "OCNany2any", 1

    def fake_static(g, sub_sg, mode, obs_col):
        return np.ones((2, 3), dtype=np.float64) / 3.0

    def fake_poisson(*args, **kwargs):
        raise AssertionError("factorized poisson path should not be used for poisson_full null model")

    def fake_poisson_full(mode, cb_ids, sub_tensor, niter, obs_col, num_gad_combinat, list_igad, g):
        return expected.copy()

    def fake_quantile(*args, **kwargs):
        raise AssertionError("hypergeom path should not be used for poisson_full null model")

    monkeypatch.setattr(omega, "_prepare_substitution_permutation_components", fake_prepare)
    monkeypatch.setattr(omega, "_get_static_sub_sites_if_available", fake_static)
    monkeypatch.setattr(omega, "_calc_poisson_count_matrix", fake_poisson)
    monkeypatch.setattr(omega, "_calc_poisson_full_count_matrix", fake_poisson_full)
    monkeypatch.setattr(omega, "_calc_hypergeom_count_matrix", fake_quantile)

    out = omega._get_mode_permutation_count_matrix(
        cb_ids=cb_ids,
        sub_tensor=np.zeros((2, 3, 1, 1, 1), dtype=np.float64),
        mode="any2any",
        SN="N",
        niter=5,
        g={"omega_pvalue_null_model": "poisson_full"},
    )
    np.testing.assert_allclose(out, expected)


def test_get_mode_permutation_count_matrix_uses_nbinom_model(monkeypatch):
    cb_ids = np.array([[0, 1]], dtype=np.int64)
    expected = np.full((1, 5), 4.0, dtype=np.float64)

    def fake_prepare(sub_tensor, mode, SN, g):
        return np.zeros((2, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64), [[0, 0, "any2", "2any"]], "OCNany2any", 1

    def fake_static(g, sub_sg, mode, obs_col):
        return np.ones((2, 3), dtype=np.float64) / 3.0

    def fake_nbinom(mode, cb_ids, sub_sg, sub_bg, niter, obs_col, num_gad_combinat, list_igad, g, static_sub_sites, obs_count):
        assert obs_count is not None
        return expected.copy()

    def fake_poisson(*args, **kwargs):
        raise AssertionError("poisson path should not be used for nbinom null model")

    def fake_quantile(*args, **kwargs):
        raise AssertionError("hypergeom path should not be used for nbinom null model")

    monkeypatch.setattr(omega, "_prepare_substitution_permutation_components", fake_prepare)
    monkeypatch.setattr(omega, "_get_static_sub_sites_if_available", fake_static)
    monkeypatch.setattr(omega, "_calc_nbinom_count_matrix", fake_nbinom)
    monkeypatch.setattr(omega, "_calc_poisson_count_matrix", fake_poisson)
    monkeypatch.setattr(omega, "_calc_hypergeom_count_matrix", fake_quantile)

    out = omega._get_mode_permutation_count_matrix(
        cb_ids=cb_ids,
        sub_tensor=None,
        mode="any2any",
        SN="N",
        niter=5,
        g={"omega_pvalue_null_model": "nbinom"},
        obs_count=np.array([2.0], dtype=np.float64),
    )
    np.testing.assert_allclose(out, expected)


def test_calc_nbinom_count_matrix_supports_fixed_overdispersion():
    cb_ids = np.array([[0, 1], [1, 2]], dtype=np.int64)
    sub_bg = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    sub_sg = np.zeros((2, 1), dtype=np.float64)
    static_sub_sites = np.array(
        [
            [0.6, 0.4],
            [0.5, 0.5],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )
    list_igad = [[0, 0, "any2", "2any"]]
    expected_mean = omega._calc_wallenius_expected_overlap(
        cb_ids=cb_ids,
        sub_sites=static_sub_sites,
        sub_branches=sub_bg[:, 0],
        g={"omega_pvalue_rounding": "round"},
        float_type=np.float64,
    )
    np.random.seed(13)
    out = omega._calc_nbinom_count_matrix(
        mode="any2any",
        cb_ids=cb_ids,
        sub_sg=sub_sg,
        sub_bg=sub_bg,
        niter=4000,
        obs_col="OCNany2any",
        num_gad_combinat=1,
        list_igad=list_igad,
        g={"float_tol": 1e-12, "omega_pvalue_nbinom_alpha": 0.8},
        static_sub_sites=static_sub_sites,
        obs_count=np.array([2.0, 3.0], dtype=np.float64),
    )
    assert out.shape == (2, 4000)
    assert out.dtype == np.float64
    assert np.all(out >= 0)
    np.testing.assert_allclose(out.mean(axis=1), expected_mean, atol=0.18)
    # With alpha>0, variance should exceed mean for overdispersed rows.
    assert np.all(out.var(axis=1) > out.mean(axis=1))


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
            "omegaC_method": "modelfree",
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
            "omegaC_method": "modelfree",
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
            "omegaC_method": "modelfree",
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


def test_prepare_epistasis_supports_s_channel_and_applies_only_to_ocs():
    ON_tensor = np.zeros(shape=(2, 3, 1, 1, 1), dtype=np.float64)
    OS_tensor = np.zeros(shape=(2, 3, 1, 1, 1), dtype=np.float64)
    OS_tensor[0, :, 0, 0, 0] = np.array([2.0, 1.0, 0.0], dtype=np.float64)
    OS_tensor[1, :, 0, 0, 0] = np.array([0.0, 1.0, 3.0], dtype=np.float64)
    g = {
        "float_tol": 1e-12,
        "is_site_nonmissing": np.ones(shape=(2, 3), dtype=bool),
        "_asrv_branch_ids": np.array([0, 1], dtype=np.int64),
        "epistasis_apply_to": "S",
        "epistasis_beta_auto": False,
        "epistasis_beta_value": 1.0,
        "epistasis_clip_auto": False,
        "epistasis_clip_value": 3.0,
        "asrv_dirichlet_alpha": 1.0,
        "epistasis_site_degree_internal": np.array([-1.0, 0.0, 1.0], dtype=np.float64),
    }
    g = omega.prepare_epistasis(g=g, ON_tensor=ON_tensor, OS_tensor=OS_tensor)
    assert g["epistasis_enabled"] is True
    assert "S" in g["_epistasis_state"]
    assert "N" not in g["_epistasis_state"]
    sub_sites = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.2, 0.3, 0.5],
        ],
        dtype=np.float64,
    )
    out_s = omega._apply_epistasis_to_sub_sites(sub_sites=sub_sites, obs_col="OCSany2any", g=g)
    out_n = omega._apply_epistasis_to_sub_sites(sub_sites=sub_sites, obs_col="OCNany2any", g=g)
    assert not np.allclose(out_s, sub_sites)
    np.testing.assert_allclose(out_n, sub_sites)


def test_prepare_epistasis_prints_negative_control_summary_for_s_channel(capsys):
    ON_tensor = np.zeros(shape=(2, 3, 1, 1, 1), dtype=np.float64)
    OS_tensor = np.zeros(shape=(2, 3, 1, 1, 1), dtype=np.float64)
    OS_tensor[0, :, 0, 0, 0] = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    OS_tensor[1, :, 0, 0, 0] = np.array([0.0, 2.0, 1.0], dtype=np.float64)
    g = {
        "float_tol": 1e-12,
        "is_site_nonmissing": np.ones(shape=(2, 3), dtype=bool),
        "_asrv_branch_ids": np.array([0, 1], dtype=np.int64),
        "epistasis_apply_to": "S",
        "epistasis_beta_auto": False,
        "epistasis_beta_value": 1.0,
        "epistasis_clip_auto": False,
        "epistasis_clip_value": 3.0,
        "asrv_dirichlet_alpha": 1.0,
        "epistasis_site_metric": "proximity",
        "epistasis_site_metric_resolved": "proximity",
        "epistasis_site_degree_internal": np.array([-1.0, 0.0, 1.0], dtype=np.float64),
    }
    omega.prepare_epistasis(g=g, ON_tensor=ON_tensor, OS_tensor=OS_tensor)
    captured = capsys.readouterr()
    assert "Epistasis summary: apply_to=S, site_metric=proximity" in captured.out
    assert "Epistasis negative-control summary: apply_to=S" in captured.out


def test_apply_epistasis_beta_to_probs_supports_multifeature_site_profiles():
    base_probs = np.full((2, 3), 1.0 / 3.0, dtype=np.float64)
    mask = np.ones_like(base_probs, dtype=bool)
    branch_context = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    site_features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=np.float64,
    )
    out = omega._apply_epistasis_beta_to_probs(
        base_probs=base_probs,
        branch_context=branch_context,
        degree_z=site_features,
        beta=1.5,
        clip_value=5.0,
        is_site_nonmissing=mask,
        float_tol=1e-12,
    )
    assert out.shape == base_probs.shape
    assert out[0, 0] > out[0, 1]
    assert out[1, 1] > out[1, 0]


def test_prepare_epistasis_branch_depth_partition_records_branchwise_parameters():
    ON_tensor = np.zeros(shape=(4, 3, 1, 1, 1), dtype=np.float64)
    OS_tensor = np.zeros(shape=(4, 3, 1, 1, 1), dtype=np.float64)
    ON_tensor[:, :, 0, 0, 0] = np.array(
        [
            [3.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    g = {
        "float_tol": 1e-12,
        "is_site_nonmissing": np.ones(shape=(4, 3), dtype=bool),
        "_asrv_branch_ids": np.array([0, 1, 2, 3], dtype=np.int64),
        "epistasis_apply_to": "N",
        "epistasis_beta_auto": False,
        "epistasis_beta_value": 0.8,
        "epistasis_clip_auto": False,
        "epistasis_clip_value": 3.0,
        "epistasis_beta_partition": "branch_depth",
        "epistasis_branch_depth_bins": 2,
        "epistasis_joint_auto": False,
        "asrv_dirichlet_alpha": 1.0,
        "epistasis_site_degree_internal": np.array([-1.0, 0.0, 1.0], dtype=np.float64),
    }
    out = omega.prepare_epistasis(g=g, ON_tensor=ON_tensor, OS_tensor=OS_tensor)
    state = out["_epistasis_state"]["N"]
    assert state["beta_diag"]["partition"] == "branch_depth"
    assert len(state["beta_diag"]["bins"]) == 2
    assert state["beta_by_branch"].shape == (4,)
    assert state["clip_by_branch"].shape == (4,)
    np.testing.assert_allclose(state["beta_by_branch"], np.full((4,), 0.8, dtype=np.float64))


def test_prepare_epistasis_joint_auto_selects_alpha_and_clip_from_grid():
    ON_tensor = np.zeros(shape=(4, 3, 1, 1, 1), dtype=np.float64)
    OS_tensor = np.zeros(shape=(4, 3, 1, 1, 1), dtype=np.float64)
    ON_tensor[:, :, 0, 0, 0] = np.array(
        [
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    g = {
        "float_tol": 1e-12,
        "is_site_nonmissing": np.ones(shape=(4, 3), dtype=bool),
        "_asrv_branch_ids": np.array([0, 1, 2, 3], dtype=np.int64),
        "epistasis_apply_to": "N",
        "epistasis_beta_auto": True,
        "epistasis_beta_value": np.nan,
        "epistasis_clip_auto": True,
        "epistasis_clip_value": np.nan,
        "epistasis_joint_auto": True,
        "epistasis_joint_alpha_grid": [0.0, 0.5, 1.0],
        "epistasis_joint_clip_grid": [1.5, 2.0, 3.0],
        "epistasis_beta_partition": "global",
        "asrv_dirichlet_alpha": 1.0,
        "epistasis_site_degree_internal": np.array([-1.0, 0.0, 1.0], dtype=np.float64),
    }
    out = omega.prepare_epistasis(g=g, ON_tensor=ON_tensor, OS_tensor=OS_tensor)
    state = out["_epistasis_state"]["N"]
    assert state["alpha"] in {0.0, 0.5, 1.0}
    assert state["clip"] in {1.5, 2.0, 3.0}


def test_get_epistasis_branch_depth_by_id_uses_cumulative_branch_lengths():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:2)X:3,C:4)R;", format=1))
    by_name = {n.name: n for n in tr.traverse()}
    max_id = max(int(ete.get_prop(n, "numerical_label")) for n in tr.traverse())
    out = omega._get_epistasis_branch_depth_by_id(g={"tree": tr}, num_branch=max_id + 1)
    id_A = int(ete.get_prop(by_name["A"], "numerical_label"))
    id_B = int(ete.get_prop(by_name["B"], "numerical_label"))
    id_X = int(ete.get_prop(by_name["X"], "numerical_label"))
    id_C = int(ete.get_prop(by_name["C"], "numerical_label"))
    assert out[id_X] == pytest.approx(3.0, abs=1e-12)
    assert out[id_A] == pytest.approx(4.0, abs=1e-12)
    assert out[id_B] == pytest.approx(5.0, abs=1e-12)
    assert out[id_C] == pytest.approx(4.0, abs=1e-12)


def test_fit_epistasis_beta_cv_returns_zero_when_active_branches_are_insufficient():
    counts = np.array([[2.0, 1.0, 0.0]], dtype=np.float64)
    mask = np.ones(shape=counts.shape, dtype=bool)
    base_probs = np.array([[2.0 / 3.0, 1.0 / 3.0, 0.0]], dtype=np.float64)
    branch_context = np.array([0.1], dtype=np.float64)
    degree_z = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    beta, diag = omega._fit_epistasis_beta_cv(
        counts=counts,
        base_probs=base_probs,
        branch_context=branch_context,
        degree_z=degree_z,
        is_site_nonmissing=mask,
        clip_value=3.0,
        float_tol=1e-12,
    )
    assert beta == pytest.approx(0.0)
    assert diag["active_branch_count"] == 1


def test_fit_epistasis_beta_cv_detects_positive_signal():
    degree_z = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    branch_context = np.array([-1.2, -0.8, -0.4, 0.4, 0.8, 1.2], dtype=np.float64)
    mask = np.ones(shape=(branch_context.shape[0], degree_z.shape[0]), dtype=bool)
    base_probs = np.full(shape=mask.shape, fill_value=(1.0 / 3.0), dtype=np.float64)
    probs = omega._apply_epistasis_beta_to_probs(
        base_probs=base_probs,
        branch_context=branch_context,
        degree_z=degree_z,
        beta=1.2,
        clip_value=5.0,
        is_site_nonmissing=mask,
        float_tol=1e-12,
    )
    counts = np.round(probs * 120.0).astype(np.float64)
    beta, diag = omega._fit_epistasis_beta_cv(
        counts=counts,
        base_probs=base_probs,
        branch_context=branch_context,
        degree_z=degree_z,
        is_site_nonmissing=mask,
        clip_value=5.0,
        float_tol=1e-12,
        dirichlet_alpha=1.0,
    )
    assert beta > 0.0
    assert diag["selection_rule"] == "argmax_mean_cv_loglik"
    assert diag["best_beta"] == pytest.approx(beta)


def test_auto_epistasis_clip_stays_within_bounds():
    branch_context = np.array([0.0, 1.0], dtype=np.float64)
    degree_z = np.array([-2.0, 0.0, 2.0], dtype=np.float64)
    mask = np.ones(shape=(2, 3), dtype=bool)
    clip = omega._auto_epistasis_clip(
        beta=10.0,
        branch_context=branch_context,
        degree_z=degree_z,
        is_site_nonmissing=mask,
    )
    assert clip >= 1.5
    assert clip <= 5.0
