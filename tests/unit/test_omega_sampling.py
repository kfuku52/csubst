from fractions import Fraction
import itertools

import numpy as np
import pytest

from csubst import omega


@pytest.mark.parametrize('weights', [[0., 1., 2., 4.], [1., 1., 1., 1.],
                                    [1e-100, .1, 1., 1e100], [1e300, 2e-300, 1e-300]])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_fisher_inclusion_matches_independent_weighted_subsets(weights, dtype):
    values = np.array(weights)
    values.flags.writeable = False
    original = values.tobytes()
    positive = np.flatnonzero(values > 0)
    for draw in range(len(positive) + 2):
        size = min(draw, len(positive))
        total = Fraction(0)
        included = [Fraction(0) for _ in values]
        for selected in itertools.combinations(positive, size):
            mass = Fraction(1)
            for site in selected:
                mass *= Fraction(float(values[site]))
            total += mass
            for site in selected:
                included[site] += mass
        expected = np.array([float(mass / total) for mass in included])
        actual = omega._calc_fisher_inclusion_probabilities(values, draw, dtype)
        assert actual.dtype == dtype
        np.testing.assert_allclose(actual, expected, rtol=2e-7 if dtype == np.float32 else 3e-13,
                                   atol=1e-14)
    assert values.tobytes() == original
    assert not values.flags.writeable


def test_fisher_dense_draw_retains_tiny_inclusion_probability():
    # Selecting three sites normally omits the fourth. Its small probability
    # of inclusion is still representable even when 1 - P(omission) is zero.
    actual = omega._calc_fisher_inclusion_probabilities([1., 2., 4., 1e-40], 3)
    np.testing.assert_allclose(actual, [1., 1., 1., 1.75e-40], atol=0., rtol=2e-14)


@pytest.mark.parametrize('arity', [1, 2, 3])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_urn_overlap_preserves_arithmetic_across_row_blocks(monkeypatch, arity, dtype):
    weights = np.arange(42, dtype=np.float64).reshape(6, 7)[:, ::2] / 100
    ids = np.array([[0, 1, 2], [3, 4, 0], [1, 0, 5], [3, 1, 2], [2, 4, 1]])[:, :arity]
    monkeypatch.setattr(omega, '_URN_OVERLAP_WORK_BYTES', 4 * weights.shape[1] * np.dtype(dtype).itemsize)
    reference = weights.astype(dtype)[ids].prod(axis=1, dtype=dtype).sum(axis=1, dtype=dtype)
    actual = omega._calc_cb_site_overlap(ids, weights, dtype)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize('rounding,expected_calls', [('round', 3), ('stochastic', 4)])
def test_urn_overlap_reuses_identical_profiles_and_draw_sizes(rounding, expected_calls):
    weights = np.array([[.4, .3, .2, .1], [.4, .3, .2, .1], [.4, .3, .2, .1], [.1, .2, .3, .4]])
    branch_sizes = np.array([1.2, 1.2, 2., 1.2])
    ids = np.array([[0, 1], [0, 2], [2, 3]])
    g = {'omega_pvalue_rounding': rounding}
    calls = []

    def inclusion(site_weights, draw_size, float_type):
        calls.append((site_weights[0], draw_size))
        return omega._calc_fisher_inclusion_probabilities(site_weights, draw_size, float_type)

    actual = omega._calc_weighted_urn_expected_overlap(ids, weights, branch_sizes, g, np.float64, inclusion)
    assert len(calls) == expected_calls
    assert len(calls) == len(set(calls))
    reference = []
    for row, size in zip(weights, branch_sizes):
        lo = int(size)
        first = omega._calc_fisher_inclusion_probabilities(row, lo)
        if rounding == 'stochastic':
            frac = size - lo
            second = omega._calc_fisher_inclusion_probabilities(row, lo + 1)
            first = (1. - frac) * first + frac * second
        reference.append(first)
    expected = omega._calc_cb_site_overlap(ids, reference, np.float64)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('weights,size,expected', [
    ([1e-100, 2e-100, 1e-100], 1, [.25, .5, .25]),
    ([1e300, 2e-300, 1e-300], 2, [1., 2/3, 1/3]),
])
def test_wallenius_sampling_preserves_extreme_positive_odds(weights, size, expected):
    niter = 20_000
    mask = omega._weighted_sample_without_replacement_masks(weights, size, niter, np.random.default_rng(71))
    packed = omega._weighted_sample_without_replacement_packed(weights, size, niter, np.random.default_rng(71))
    np.testing.assert_array_equal(packed, np.packbits(mask, axis=1))
    np.testing.assert_array_equal(mask.sum(axis=1), size)
    np.testing.assert_allclose(mask.mean(axis=0), expected, atol=.014, rtol=0.)


@pytest.mark.native
def test_native_packed_sampling_matches_boolean_masks_and_python_counts(monkeypatch):
    if omega.omega_cy is None:
        pytest.skip('Cython extension is unavailable')
    sampled = np.array([[0, 3, 8], [1, 2, 9]], dtype=np.int64)
    packed = omega._pack_sampled_site_indices_to_uint8(sampled, num_site=10)
    np.testing.assert_array_equal(packed, [[144, 128], [96, 64]])

    p = np.array([0.2, 0.1, 0.25, 0.15, 0.1, 0.1, 0.05, 0.05, 0.0, 0.0])
    ids = np.array([[2, 0, 1], [1, 1, 0]], dtype=np.int64)
    sizes = np.array([2, 3, 5], dtype=np.int64)
    native = omega._get_permutations_fast(
        ids, sizes, p, 128, rng=np.random.default_rng(7),
    )
    monkeypatch.setattr(omega, 'omega_cy', None)
    fallback = omega._get_permutations_fast(
        ids, sizes, p, 128, rng=np.random.default_rng(7),
    )
    np.testing.assert_array_equal(native, fallback)


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


@pytest.mark.parametrize('shared_profile', [True, False])
def test_equal_size_branches_draw_independent_uniform_subsets(shared_profile):
    rng = np.random.default_rng(208)
    profiles = np.full(5, .2)
    if not shared_profile:
        profiles = np.tile(profiles, (2, 1))
    ids = np.array([[0, 1], [1, 0]])
    counts = np.concatenate([
        omega._get_permutations_fast(ids, [2, 2], profiles, 4, rng=rng)
        for _ in range(256)
    ], axis=1)
    np.testing.assert_array_equal(counts[0], counts[1])
    # Conditional on either branch, one of C(5, 2) independent subsets
    # has full overlap. Reordering one shared sample overstates that event.
    assert np.mean(counts[0] == 2) == pytest.approx(.1, abs=.04)


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


def test_fisher_samples_positive_weights_below_the_normalized_float_range():
    masks = omega._weighted_sample_without_replacement_masks(
        p=np.array([1e300, 2e-300, 0., 1e-300]),
        size=2,
        niter=12000,
        rng=np.random.default_rng(80),
        sampling_model='fisher',
    )
    assert masks[:, 0].all() and not masks[:, 2].any()
    np.testing.assert_array_equal(masks.sum(axis=1), 2)
    np.testing.assert_allclose(masks.mean(axis=0), [1., 2/3, 0., 1/3], atol=.02)


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
