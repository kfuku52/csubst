"""Exploratory rate scores and explicit scan calibration contracts.

Posterior event mass is not an independent integer Poisson observation. The
asymptotic tail below is a diagnostic, not a calibrated significance test.
"""

import math
import sys

import numpy as np
from scipy.special import log_ndtr


SCORE_METHOD = 'negative_log10_one_sided_poisson_asymptotic_tail'
BH_SCOPE = 'selected_candidates_within_trait_match'
MAXIMUM_SCOPE = 'all_testable_selected_candidates_all_traits_matches_in_this_run'


def _softplus(value):
    return max(value, 0.) + math.log1p(math.exp(-abs(value)))


def _local_poisson_deviance(count, residual):
    # Series for -log1p(-r) - r, where r is the residual divided by
    # the observed count. At |r| <= 0.0011, omitted terms are below roundoff.
    high_terms = .2 + residual * (1/6 + residual/7)
    series = .5 + residual * (1/3 + residual * (.25 + residual * high_terms))
    return (count * residual) * residual * series


def _relative_pooled_residuals(x, a, y, b):
    # Express (x*b - y*a) / (a+b) in exact integer arithmetic before
    # dividing by each event count. Near equal rates lose this residual
    # if the products are rounded separately.
    nx, dx = x.as_integer_ratio()
    na, da = a.as_integer_ratio()
    ny, dy = y.as_integer_ratio()
    nb, db = b.as_integer_ratio()
    residual = nx * nb * dy * da - ny * na * dx * db
    if residual <= 0:
        return None
    exposure = na * db + nb * da
    return residual / (nx * dy * exposure), -residual / (ny * dx * exposure)


def _small_exposure_deviance(x, a, y, b):
    # The control expectation is x*b/a up to a relative correction of
    # order b/a. Separate exponents so b/a need not be representable.
    x_mantissa, x_exponent = math.frexp(x)
    a_mantissa, a_exponent = math.frexp(a)
    b_mantissa, b_exponent = math.frexp(b)
    mean = math.ldexp(x_mantissa * b_mantissa / a_mantissa,
                      x_exponent + b_exponent - a_exponent)
    if y == 0:
        return mean
    if y >= mean:
        return 0.
    deficit = mean - y
    if y >= mean / 2:
        log_ratio = math.log1p(-deficit / mean)
    else:
        log_ratio = math.log(y) - math.log(mean)
    return deficit + y * log_ratio


def rate_score(x_target, l_target, x_other, l_other):
    """Return (score, asymptotic diagnostic P), without rounding event mass.

    Non-enrichment has score zero / diagnostic P one, preserving the former
    ordering. log_ndtr avoids tying extreme enrichment through P underflow.
    Invalid counts/exposures are undefined, including absent control exposure.
    """
    x, a, y, b = map(float, (x_target, l_target, x_other, l_other))
    if (not (math.isfinite(x) and math.isfinite(a) and math.isfinite(y) and math.isfinite(b))
            or min(x, y) < 0 or min(a, b) <= 0):
        return np.nan, np.nan
    if x == 0:
        return 0., 1.
    log_exposure_odds = math.log(a) - math.log(b)
    if y == 0:
        target_log_ratio = _softplus(-log_exposure_odds)
        other_log_ratio = 0.
        terms = x * target_log_ratio
    else:
        log_count_odds = math.log(x) - math.log(y)
        gap = log_count_odds - log_exposure_odds
        if abs(gap) <= .001:
            residuals = _relative_pooled_residuals(x, a, y, b)
            if residuals is None:
                return 0., 1.
            terms = (_local_poisson_deviance(x, residuals[0])
                     + _local_poisson_deviance(y, residuals[1]))
        elif gap < 0:
            return 0., 1.
        else:
            # Compute each allocation probability in log space to retain the
            # smaller group even when the larger probability rounds to one.
            target_log_ratio = _softplus(-log_exposure_odds) - _softplus(-log_count_odds)
            other_log_ratio = _softplus(log_exposure_odds) - _softplus(log_count_odds)
            terms = x * target_log_ratio + y * other_log_ratio
    if log_exposure_odds > 700 and (y == 0 or gap > .001):
        terms = _small_exposure_deviance(x, a, y, b)
    if terms > sys.float_info.max / 2:
        # At this scale the normal log tail differs from -terms by less
        # than one float64 ulp. Compute the base-10 score without doubling
        # the log likelihood ratio or overflowing its individual terms.
        if math.isinf(terms):
            scale = max(x, y)
            scaled_terms = (x / scale) * target_log_ratio + (y / scale) * other_log_ratio
            return scale * (scaled_terms / math.log(10)), 0.
        return terms / math.log(10), 0.
    deviance = max(0., 2 * terms)
    log_tail = float(log_ndtr(-math.sqrt(deviance)))
    return -log_tail / math.log(10), math.exp(log_tail)


def maximum_score(frame):
    """Max over testable candidates, with explicit zero-exposure no-tests."""
    if len(frame) == 0:
        return -np.inf
    values = frame['score_rate_enrichment'].to_numpy(dtype=float, copy=True)
    invalid = ~(np.isfinite(values) & (values >= 0))
    for index in np.flatnonzero(invalid):
        values[index] = calibration_score(frame.iloc[index])
    return float(values.max())


def calibration_score(row):
    score = float(row['score_rate_enrichment'])
    if math.isfinite(score) and score >= 0:
        return score
    # This is a declared testability rule, not silent omission of undefined
    # arithmetic: current instantaneous-Q exposure can be zero for observed
    # multi-step endpoint changes. That contrast has no identifiable rate.
    required = ('target_event_count', 'other_event_count',
                'target_exposure' if 'target_exposure' in row else 'target_exposure_branch_length',
                'other_exposure' if 'other_exposure' in row else 'other_exposure_branch_length')
    if math.isnan(score) and all(key in row for key in required):
        counts_exposure = np.array([row[key] for key in required], dtype=float)
        if (np.isfinite(counts_exposure).all() and (counts_exposure >= 0).all()
                and (counts_exposure[2:] == 0).any()):
            return -np.inf
    raise ValueError('Undefined rate score in selected candidates; null replicate cannot be omitted.')


def _empirical_upper_tail(observed, reference):
    # The inclusive tail also counts roundoff-sized gaps below observation.
    return (observed - reference) <= (1e-12 + 1e-12 * observed)


def _validated_reference(reference, expected_count):
    values = np.asarray(reference, dtype=float)
    if values.ndim != 1 or not values.size or np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError('A nonempty null reference without undefined replicates is required.')
    if expected_count is not None and values.size != expected_count:
        raise ValueError('Missing null replicates; success-conditioned calibration is not supported.')
    return values


def empirical_pvalue(observed, reference, expected_count=None, exact=False):
    """Inclusive upper tail; +1 for Monte Carlo, no offset for enumeration.

    -inf denotes a successful replicate with no candidate. Approximate ties
    are counted conservatively, including TSV serialization roundoff.
    """
    values = _validated_reference(reference, expected_count)
    if np.isnan(observed) or np.isposinf(observed):
        return np.nan
    if observed <= 0:
        return 1.0  # No foreground enrichment, even when null draws have no candidate.
    exceedances = np.count_nonzero(_empirical_upper_tail(observed, values))
    offset = 0 if exact else 1
    return float((offset + exceedances) / (offset + values.size))



def empirical_pvalues(observed, reference, expected_count=None, exact=False):
    """Inclusive upper tails for multiple scores against one null reference."""
    values = _validated_reference(reference, expected_count)
    observed = np.asarray(observed, dtype=float)
    pvalues = np.full(observed.shape, np.nan)
    pvalues[observed <= 0] = 1.
    testable = np.isfinite(observed) & (observed > 0)
    scores = observed[testable]
    offset = 0 if exact else 1
    if len(scores) <= 4:
        counts = [np.count_nonzero(_empirical_upper_tail(score, values)) for score in scores]
    else:
        ordered = np.empty(values.size + 1)
        ordered[:-1] = values
        ordered[:-1].sort()
        ordered[-1] = np.inf
        lower = np.zeros(len(scores), dtype=np.intp)
        upper = np.full(len(scores), values.size, dtype=np.intp)
        # Use the exact same comparison at every search step. A cutoff
        # obtained by subtracting the tie tolerance can round across a tie.
        with np.errstate(over='ignore'):
            while (lower < upper).any():
                middle = (lower + upper) // 2
                in_tail = _empirical_upper_tail(scores, ordered[middle])
                upper = np.where(in_tail, middle, upper)
                lower = np.where(in_tail, lower, middle + 1)
        counts = values.size - lower
    pvalues[testable] = (offset + np.asarray(counts)) / (offset + values.size)
    return pvalues
