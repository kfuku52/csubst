import numpy as np
import pytest

from csubst import omega


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
    masks = omega._weighted_sample_without_replacement_masks(
        p=p,
        size=draw_size,
        niter=16000,
        rng=np.random.default_rng(2),
    )
    empirical = masks.mean(axis=0)
    np.testing.assert_allclose(expected, empirical, atol=0.03)
    np.testing.assert_allclose(expected.sum(), float(draw_size), atol=1e-8)


def test_weighted_urn_draw_one_matches_normalized_weights_exactly():
    weights = np.array([0.9, 0.1], dtype=np.float64)
    wallenius = omega._calc_wallenius_inclusion_probabilities(weights, 1)
    fisher = omega._calc_fisher_inclusion_probabilities(weights, 1)
    np.testing.assert_allclose(wallenius, [0.9, 0.1], atol=1e-12)
    np.testing.assert_allclose(fisher, [0.9, 0.1], atol=1e-12)


def test_calc_fisher_inclusion_probabilities_returns_valid_probabilities():
    p = np.array([0.8, 0.1, 0.05, 0.05, 0.0], dtype=np.float64)
    draw_size = 2
    fisher = omega._calc_fisher_inclusion_probabilities(
        site_weights=p,
        draw_size=draw_size,
        float_type=np.float64,
    )
    wallenius = omega._calc_wallenius_inclusion_probabilities(
        site_weights=p,
        draw_size=draw_size,
        float_type=np.float64,
    )
    assert fisher.shape == p.shape
    assert np.all(fisher >= 0)
    assert np.all(fisher <= 1)
    np.testing.assert_allclose(fisher.sum(), float(draw_size), atol=1e-8)
    assert not np.allclose(fisher, wallenius)
    masks = omega._weighted_sample_without_replacement_masks(
        p=p,
        size=draw_size,
        niter=20000,
        rng=np.random.default_rng(9),
        sampling_model="fisher",
    )
    np.testing.assert_allclose(masks.mean(axis=0), fisher, atol=0.025)


def test_calc_urn_expected_overlap_factorized_approx_matches_legacy_tmp_E_sum():
    cb_ids = np.array([[0, 1], [1, 2]], dtype=np.int64)
    sub_sites = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.2, 0.7, 0.1],
            [0.4, 0.3, 0.3],
        ],
        dtype=np.float64,
    )
    sub_branches = np.array([1.5, 2.0, 0.7], dtype=np.float64)
    expected = omega._calc_tmp_E_sum(
        cb_ids=cb_ids,
        sub_sites=sub_sites,
        sub_branches=sub_branches,
        float_type=np.float64,
    )
    observed = omega._calc_urn_expected_overlap(
        cb_ids=cb_ids,
        sub_sites=sub_sites,
        sub_branches=sub_branches,
        g={"urn_model": "factorized_approx"},
        float_type=np.float64,
    )
    np.testing.assert_allclose(observed, expected, atol=1e-12)


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
    rng = np.random.default_rng(5)
    size_by_iter = omega._prepare_permutation_branch_sizes(
        sub_branches=sub_branches,
        niter=20000,
        g=g,
        rng=rng,
    )
    perm = omega._get_permutations_fast(
        cb_ids=cb_ids,
        sub_branches=size_by_iter,
        p=p,
        niter=20000,
        rng=rng,
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
    packed = omega._weighted_sample_without_replacement_packed(
        p=p, size=2, niter=64, rng=np.random.default_rng(11)
    )
    masks = omega._weighted_sample_without_replacement_masks(
        p=p, size=2, niter=64, rng=np.random.default_rng(11)
    )
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
    out_cy = omega._get_permutations_fast(
        cb_ids=cb_ids,
        sub_branches=sub_branches,
        p=p / p.sum(),
        niter=128,
        rng=np.random.default_rng(7),
    )
    assert dummy.called > 0

    monkeypatch.setattr(omega, "omega_cy", None)
    out_np = omega._get_permutations_fast(
        cb_ids=cb_ids,
        sub_branches=sub_branches,
        p=p / p.sum(),
        niter=128,
        rng=np.random.default_rng(7),
    )
    np.testing.assert_array_equal(out_cy, out_np)
