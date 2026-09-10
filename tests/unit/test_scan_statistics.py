import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, chi2, poisson

from csubst import scan_statistics as stats


def test_sparse_poisson_diagnostic_is_not_an_exact_test():
    score, p = stats.rate_score(2, 1, 0, 1)
    assert p == pytest.approx(.04794548357123278)
    assert score == pytest.approx(-math.log10(p))
    assert binom.sf(1, 2, .5) == .25
    # Total=2, equal exposures: enumerating the conditional null gives a 25%
    # rejection probability for the asymptotic diagnostic at nominal 5%.
    rejection = sum(binom.pmf(x, 2, .5) for x in range(3) if stats.rate_score(x, 1, 2-x, 1)[1] <= .05)
    assert rejection == pytest.approx(.25)


@pytest.mark.parametrize('exposures', [(1., 1.), (.1, 10.), (5., .25)])
def test_score_matches_independent_likelihood_calculation(exposures):
    a, b = exposures
    for x, y in [(2., 0.), (.7, .2), (20., 10.), (0., 0.)]:
        score, p = stats.rate_score(x, a, y, b)
        if x/a <= y/b:
            assert (score, p) == (0., 1.)
            continue
        common = (x+y)/(a+b)
        alt = poisson.logpmf(x, x) + poisson.logpmf(y, y) if x.is_integer() and y.is_integer() else None
        if alt is not None:
            null = poisson.logpmf(x, a*common) + poisson.logpmf(y, b*common)
            expected = .5*chi2.sf(2*(alt-null), 1)
        else:
            terms = [v*math.log((v/exposure)/common) if v else 0 for v,exposure in [(x,a),(y,b)]]
            expected = .5*chi2.sf(2*sum(terms), 1)
        assert p == pytest.approx(expected)
        assert score == pytest.approx(-math.log10(expected))


def test_large_counts_keep_distinct_finite_scores_after_p_underflow():
    score1, p1 = stats.rate_score(10000, 1, 0, 1)
    score2, p2 = stats.rate_score(20000, 1, 0, 1)
    assert p1 == p2 == 0.
    assert np.isfinite([score1, score2]).all() and score2 > score1


def test_extreme_exposure_ratios_do_not_make_valid_scores_undefined():
    score, p = stats.rate_score(2, 1e20, 0, 1)
    assert np.isfinite(score) and p == pytest.approx(.5)
    assert stats.rate_score(1, 1e300, 1, 1e-300) == (0., 1.)
    score, p = stats.rate_score(2, 1e-300, 0, 1e300)
    # With y=0, the likelihood-ratio deviance is -2*x*log(p_target).
    from scipy.special import log_ndtr
    expected = -log_ndtr(-math.sqrt(4 * 600 * math.log(10))) / math.log(10)
    assert score == pytest.approx(expected)
    assert p == 0.


@pytest.mark.parametrize('values', [(-1,1,0,1), (1,0,0,1), (1,1,0,0), (np.nan,1,0,1), (1,np.inf,0,1)])
def test_invalid_rate_inputs_remain_undefined(values):
    assert np.isnan(stats.rate_score(*values)).all()


def test_empirical_reference_preserves_ties_and_empty_draws():
    assert stats.empirical_pvalue(3, [3, 3-1e-14, 3+1e-14], 3) == 1.
    assert stats.empirical_pvalue(3, [-np.inf, 2, 4], 3) == .5
    assert stats.empirical_pvalue(0, [-np.inf]*20, 20) == 1.
    assert stats.empirical_pvalue(-np.inf, [-np.inf]*20, 20) == 1.
    assert np.isnan(stats.empirical_pvalue(np.nan, [1, 2]))
    assert np.isnan(stats.empirical_pvalue(np.inf, [1, 2]))
    with pytest.raises(ValueError, match='undefined'):
        stats.empirical_pvalue(3, [np.nan, 2])
    with pytest.raises(ValueError, match='Missing'):
        stats.empirical_pvalue(3, [2], 2)
    with pytest.raises(ValueError, match='undefined'):
        stats.empirical_pvalue(3, [2, np.inf], 2)


def test_empty_family_is_not_an_undefined_score():
    assert stats.maximum_score(pd.DataFrame({'score_rate_enrichment': []})) == -np.inf
    with pytest.raises(ValueError, match='Undefined'):
        stats.maximum_score(pd.DataFrame({'score_rate_enrichment': [2., np.nan]}))


def test_zero_exposure_is_a_declared_no_test_but_bad_data_still_fail():
    row = dict(score_rate_enrichment=np.nan, target_event_count=2., other_event_count=.01,
               target_exposure_branch_length=0., other_exposure_branch_length=0.)
    assert stats.calibration_score(row) == -np.inf
    assert stats.maximum_score(pd.DataFrame([row])) == -np.inf
    with pytest.raises(ValueError, match='Undefined'):
        stats.calibration_score(dict(row, target_event_count=np.nan))
    with pytest.raises(ValueError, match='Undefined'):
        stats.calibration_score(dict(row, target_event_count=-1.))
    with pytest.raises(ValueError, match='Undefined'):
        stats.calibration_score(dict(row, target_exposure_branch_length=1., other_exposure_branch_length=1.))


def test_enumerated_reference_includes_candidate_discovery():
    # All eight equally probable outcomes of three independent binary features;
    # discovery retains only positive features. Max-score reference includes
    # the all-zero (empty-discovery) dataset and its probability mass.
    from itertools import product
    reference = []
    for bits in product((0, 1), repeat=3):
        frame = pd.DataFrame({'score_rate_enrichment': [i+1 for i,bit in enumerate(bits) if bit]})
        reference.append(stats.maximum_score(frame))
    assert reference.count(-np.inf) == 1
    assert stats.empirical_pvalue(3., reference) == 5/9
    # Searching any number of columns uses the maximum from the SAME draw.
    assert max(reference) == 3.
