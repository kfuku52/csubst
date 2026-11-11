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


@pytest.mark.parametrize('weights', [[1e200, 2., 1.], [1e300, 2e-300, 1e-300]])
def test_two_draws_extreme_dominant_weight_retains_second_event(weights):
    actual = omega._calc_wallenius_inclusion_probabilities(weights, 2)
    np.testing.assert_allclose(actual, [1., 2/3, 1/3], atol=1e-14)


def test_uniform_large_urn_is_exact():
    actual = omega._calc_wallenius_inclusion_probabilities(np.ones(1000), 333, policy='exact')
    np.testing.assert_array_equal(actual, np.full(1000, .333))


def test_sequential_enumeration_rescales_the_remaining_sites():
    weights = [1e300, 5e-300, 1e-300, 1e-300]
    actual = omega._calc_wallenius_inclusion_probabilities(weights, 3, policy='exact')
    np.testing.assert_allclose(actual, [1., 20/21, 11/21, 11/21], rtol=1e-14, atol=0.)


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


def test_mean_approximation_brackets_very_small_weights():
    weights = [1.] * 5 + [1e-100] * 40 + [2e-100] * 60
    actual = omega._calc_wallenius_inclusion_probabilities(weights, 25)
    # The heavy sites are saturated. The mean equation for the other sites is
    # 40(1 - u) + 60(1 - u**2) = 20, where u is survival at the lighter weight.
    u = (-40 + np.sqrt(40**2 + 4*60*80)) / (2*60)
    np.testing.assert_allclose(actual, [1.] * 5 + [1-u] * 40 + [1-u*u] * 60,
                               rtol=3e-14, atol=0.)
    assert actual.sum() == pytest.approx(25., abs=1e-13, rel=0.)


def test_mean_approximation_balances_masses_below_roundoff_of_total():
    weights = [1.] * 5 + [1e-30] * 30 + [2e-30] * 40
    actual = omega._calc_wallenius_inclusion_probabilities(weights, 5)
    np.testing.assert_array_equal(actual[:5], 1.)
    time = -np.log1p(-actual[5]) / weights[5]
    assert actual[5:].sum() == pytest.approx(5 * np.exp(-time), rel=5e-12, abs=0.)
