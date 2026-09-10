import numpy as np
import pytest

from csubst import omega


def test_twenty_one_sites_two_draws_matches_independent_ordered_pairs():
    w = np.array([20.] + [1.] * 20)
    exact = np.zeros(21)
    for first in range(21):
        for second in range(21):
            if first == second:
                continue
            probability = w[first] / w.sum() * w[second] / (w.sum() - w[first])
            exact[first] += probability
            exact[second] += probability
    for scale in (1e-200, 1., 1e200):
        actual = omega._calc_wallenius_inclusion_probabilities(w * scale, 2)
        np.testing.assert_allclose(actual, exact, atol=1e-14)
    assert exact[0] == pytest.approx(.7564102564102564)


def test_two_draws_extreme_dominant_weight_retains_second_event():
    actual = omega._calc_wallenius_inclusion_probabilities([1e200, 1., 1.], 2)
    np.testing.assert_allclose(actual, [1., .5, .5], atol=1e-14)


def test_uniform_large_urn_is_exact():
    actual = omega._calc_wallenius_inclusion_probabilities(np.ones(1000), 333, policy='exact')
    np.testing.assert_array_equal(actual, np.full(1000, .333))


def test_one_draw_does_not_overflow_sum_of_finite_weights():
    actual = omega._calc_wallenius_inclusion_probabilities([1e308, 1e308], 1)
    np.testing.assert_array_equal(actual, [.5, .5])


def test_explicit_exact_policy_rejects_large_approximation():
    with pytest.raises(ValueError, match='enumeration budget'):
        omega._calc_wallenius_inclusion_probabilities(np.arange(1., 101.), 50, policy='exact')


def test_approximation_is_scale_invariant_and_recorded():
    weights = np.arange(1., 101.)
    expected = omega._calc_wallenius_inclusion_probabilities(weights, 50)
    actual = omega._calc_wallenius_inclusion_probabilities(weights * 1e-200, 50)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    g = {'urn_wallenius_expectation': 'auto'}
    omega._calc_wallenius_expected_overlap(np.array([[0, 1]]), np.tile(weights, (2, 1)), np.array([50., 50.]), g, np.float64)
    assert g['_urn_expectation_methods'] == {'approximate_mean': True}
