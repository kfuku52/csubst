"""Exploratory rate scores and explicit scan calibration contracts.

Posterior event mass is not an independent integer Poisson observation. The
asymptotic tail below is a diagnostic, not a calibrated significance test.
"""

import math

import numpy as np
from scipy.special import log_ndtr


SCORE_METHOD = 'negative_log10_one_sided_poisson_asymptotic_tail'
BH_SCOPE = 'selected_candidates_within_trait_match'
MAXIMUM_SCOPE = 'all_testable_selected_candidates_all_traits_matches_in_this_run'


def rate_score(x_target, l_target, x_other, l_other):
    """Return (score, asymptotic diagnostic P), without rounding event mass.

    Non-enrichment has score zero / diagnostic P one, preserving the former
    ordering. log_ndtr avoids tying extreme enrichment through P underflow.
    Invalid counts/exposures are undefined, including absent control exposure.
    """
    x, a, y, b = map(float, (x_target, l_target, x_other, l_other))
    if not np.isfinite([x, a, y, b]).all() or min(x, y) < 0 or min(a, b) <= 0:
        return np.nan, np.nan
    if x == 0 or (y > 0 and math.log(x) - math.log(a) <= math.log(y) - math.log(b)):
        return 0., 1.
    # Calculate each allocation probability independently in log space. Forming
    # p then 1-p loses the control exposure when p rounds to one; direct count
    # or exposure ratios can also overflow even though the score is finite.
    log_p = -float(np.logaddexp(0., math.log(b) - math.log(a)))
    log_q = -float(np.logaddexp(0., math.log(a) - math.log(b)))
    log_fraction_x = 0. if y == 0 else -float(np.logaddexp(0., math.log(y) - math.log(x)))
    terms = x * (log_fraction_x - log_p)
    if y > 0:
        log_fraction_y = -float(np.logaddexp(0., math.log(x) - math.log(y)))
        terms += y * (log_fraction_y - log_q)
    deviance = max(0., 2 * terms)
    log_tail = float(log_ndtr(-math.sqrt(deviance)))
    return -log_tail / math.log(10), math.exp(log_tail)


def maximum_score(frame):
    """Max over testable candidates, with explicit zero-exposure no-tests."""
    values = np.array([calibration_score(row) for _, row in frame.iterrows()])
    return float(values.max()) if values.size else -np.inf


def calibration_score(row):
    score = float(row['score_rate_enrichment'])
    if np.isfinite(score) and score >= 0:
        return score
    # This is a declared testability rule, not silent omission of undefined
    # arithmetic: current instantaneous-Q exposure can be zero for observed
    # multi-step endpoint changes. That contrast has no identifiable rate.
    required = ('target_event_count', 'other_event_count',
                'target_exposure_branch_length', 'other_exposure_branch_length')
    if np.isnan(score) and all(key in row for key in required):
        counts_exposure = np.array([row[key] for key in required], dtype=float)
        if (np.isfinite(counts_exposure).all() and (counts_exposure >= 0).all()
                and (counts_exposure[2:] == 0).any()):
            return -np.inf
    raise ValueError('Undefined rate score in selected candidates; null replicate cannot be omitted.')


def empirical_pvalue(observed, reference, expected_count=None, exact=False):
    """Inclusive upper tail; +1 for Monte Carlo, no offset for enumeration.

    -inf denotes a successful replicate with no candidate. Approximate ties
    are counted conservatively, including TSV serialization roundoff.
    """
    values = np.asarray(reference, dtype=float)
    if values.ndim != 1 or not values.size or np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError('A nonempty null reference without undefined replicates is required.')
    if expected_count is not None and values.size != expected_count:
        raise ValueError('Missing null replicates; success-conditioned calibration is not supported.')
    if np.isnan(observed) or np.isposinf(observed):
        return np.nan
    if observed <= 0:
        return 1.0  # No foreground enrichment, even when null draws have no candidate.
    ties = np.isclose(values, observed, rtol=1e-12, atol=1e-12)
    exceedances = np.count_nonzero((values >= observed) | ties)
    offset = 0 if exact else 1
    return float((offset + exceedances) / (offset + values.size))
