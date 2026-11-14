import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import beta

from csubst import omega, urn_expectation


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


def test_two_weight_groups_match_an_independent_last_draw_probability():
    odds = .1
    weights = np.array([1., odds, 0., odds] + [1.] * 34)
    # If 36 of 37 sites are drawn, a particular light site is omitted when
    # its exponential clock exceeds every other clock. Integrating its
    # density against the other CDFs gives odds * (B(odds,36) - B(2*odds,36)).
    light = 1. - odds * (beta(odds, 36) - beta(2*odds, 36))
    heavy = (36 - 2 * light) / 35
    expected = np.array([heavy, light, 0., light] + [heavy] * 34)
    actual = omega._calc_wallenius_inclusion_probabilities(weights, 36, policy='exact')
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0.)

    g = {'urn_wallenius_expectation': 'exact'}
    overlap = omega._calc_wallenius_expected_overlap(
        np.array([[0, 1]]), weights, np.array([36., 36.]), g, np.float64,
    )
    np.testing.assert_allclose(overlap, [expected @ expected])
    assert g['_urn_expectation_methods'] == {'exact_two_weight_groups': True}


@pytest.mark.parametrize('odds,counts', [([.1, .6, 1.], [1, 2, 34]), ([.1, .6, 1., 2.], [1, 2, 33, 1])])
def test_multiple_weight_groups_match_independent_exponential_order_statistics(odds, counts):
    odds = np.array(odds)
    weights = np.repeat(odds, counts)
    expected = []
    for group, weight in enumerate(odds):
        others = np.array(counts)
        others[group] -= 1

        def last_density(t):
            return weight * np.exp(-weight * t) * np.prod((-np.expm1(-odds * t))**others)

        omitted, _ = quad(last_density, 0., np.inf, epsabs=1e-13)
        expected.append(1. - omitted)
    actual = omega._calc_wallenius_inclusion_probabilities(weights, len(weights) - 1, policy='exact')
    np.testing.assert_allclose(actual, np.repeat(expected, counts), rtol=1e-12, atol=2e-14)
    assert urn_expectation.wallenius_method(weights, len(weights) - 1) == 'exact_multiple_weight_groups'


def test_approximation_is_scale_invariant_and_recorded():
    weights = np.arange(1., 101.)
    expected = omega._calc_wallenius_inclusion_probabilities(weights, 50)
    actual = omega._calc_wallenius_inclusion_probabilities(weights * 1e-200, 50)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    g = {'urn_wallenius_expectation': 'auto'}
    omega._calc_wallenius_expected_overlap(np.array([[0, 1]]), np.tile(weights, (2, 1)), np.array([50., 50.]), g, np.float64)
    assert g['_urn_expectation_methods'] == {'approximate_mean': True}


@pytest.mark.parametrize('draw_size', [3, 6])
def test_quadrature_matches_exact_sequential_draws_on_both_tails(draw_size):
    weights = np.array([.03, .05, .13, .25, .6, 2., 7.])
    exact = urn_expectation.enumerated_wallenius_inclusion(weights, draw_size)
    for scale in (1.e-180, 1., 1.e180):
        actual = urn_expectation.wallenius_quadrature_inclusion(weights * scale, draw_size)
        np.testing.assert_allclose(actual, exact, rtol=1e-9, atol=2e-12)


def test_quadrature_corrects_overlap_for_distinct_skewed_weights():
    weights = np.r_[40., 15., 7., np.geomspace(.01, 1., 437)]
    g = {'urn_wallenius_expectation': 'quadrature'}
    exact = omega._calc_wallenius_expected_overlap([[0, 1]], weights, [5., 5.], g, np.float64)
    approximate = omega._calc_wallenius_expected_overlap([[0, 1]], weights, [5., 5.], {}, np.float64)
    # Independent exponential-clock sampling estimates 0.93828 (SE 0.00125).
    assert exact[0] == pytest.approx(.93828, abs=.004)
    assert approximate[0] < exact[0] - .01
    assert g['_urn_expectation_methods'] == {'quadrature': True}


def test_quadrature_matches_independent_last_arrival_integrals():
    weights = np.geomspace(.1, 3., 25)
    expected = []
    for i, weight in enumerate(weights):
        other = np.delete(weights, i)

        def last_density(time):
            return weight * np.exp(-weight * time) * np.prod(-np.expm1(-other * time))

        omitted, _ = quad(last_density, 0., np.inf, epsabs=1e-13)
        expected.append(1. - omitted)
    actual = omega._calc_wallenius_inclusion_probabilities(weights, len(weights) - 1, policy='quadrature')
    np.testing.assert_allclose(actual, expected, atol=5e-11, rtol=0.)


def test_mean_approximation_brackets_very_small_weights():
    weights = [1.] * 5 + [1e-100] * 40 + [2e-100] * 60
    actual = urn_expectation.wallenius_mean_inclusion(np.asarray(weights), 25)
    # The heavy sites are saturated. The mean equation for the other sites is
    # 40(1 - u) + 60(1 - u**2) = 20, where u is survival at the lighter weight.
    u = (-40 + np.sqrt(40**2 + 4*60*80)) / (2*60)
    np.testing.assert_allclose(actual, [1.] * 5 + [1-u] * 40 + [1-u*u] * 60,
                               rtol=3e-14, atol=0.)
    assert actual.sum() == pytest.approx(25., abs=1e-13, rel=0.)


def test_mean_approximation_balances_masses_below_roundoff_of_total():
    weights = [1.] * 5 + [1e-30] * 30 + [2e-30] * 40
    actual = urn_expectation.wallenius_mean_inclusion(np.asarray(weights), 5)
    np.testing.assert_array_equal(actual[:5], 1.)
    time = -np.log1p(-actual[5]) / weights[5]
    assert actual[5:].sum() == pytest.approx(5 * np.exp(-time), rel=5e-12, abs=0.)
