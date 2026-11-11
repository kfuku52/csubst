"""Numerical helpers for weighted-urn probabilities."""

import numpy as np


ENUMERATION_MAX_SITES = 20


def normalized_log_weights(positive_weights):
    """Normalize positive weights in log space when their ratios are subnormal."""
    largest = positive_weights.max()
    ratios = positive_weights / largest
    normal = ratios >= np.finfo(np.float64).tiny
    np.log(ratios, out=ratios, where=normal)
    ratios[~normal] = np.log(positive_weights[~normal]) - np.log(largest)
    return ratios


def fisher_suffix(log_weights, size):
    """Log subset weights for each suffix and draw size."""
    num_positive = len(log_weights)
    suffix = np.full((num_positive + 1, size + 1), -np.inf, dtype=np.float64)
    suffix[:, 0] = 0.
    for index in range(num_positive - 1, -1, -1):
        last = min(size, num_positive - index)
        next_row = suffix[index + 1]
        np.logaddexp(next_row[1:last + 1], log_weights[index] + next_row[:last],
                     out=suffix[index, 1:last + 1])
    return suffix


def wallenius_method(positive_weights, draw_size):
    weights = np.asarray(positive_weights, dtype=np.float64)
    n = weights.size
    if draw_size == 0 or n == 0 or draw_size >= n:
        return 'exact_boundary'
    if np.all(weights == weights[0]):
        return 'exact_uniform'
    if draw_size <= 2:
        return 'exact_small_draw'
    # Retain the original enumeration domain for larger draw sizes. Extending
    # it by final subset count alone can multiply work across many categories.
    # One/two-draw formulas above extend exactness without subset enumeration.
    return 'exact_enumeration' if n <= ENUMERATION_MAX_SITES else 'approximate_mean'


def two_draw_inclusion(weights):
    """P(i first) + sum_{j != i} P(j first, i second), in O(n).

    Prefix/suffix sums retain the small remaining mass when a single weight
    dominates. Subtracting the largest weight from the total can erase it.
    """
    remaining = np.array(weights, dtype=np.longdouble, copy=True)
    largest = int(remaining.argmax())
    w = remaining / remaining[largest]
    prefix = np.concatenate((np.zeros(1, dtype=w.dtype), np.cumsum(w)[:-1]))
    suffix = np.concatenate((np.cumsum(w[::-1])[-2::-1], np.zeros(1, dtype=w.dtype)))
    ratio = np.zeros_like(w)
    others = np.arange(w.size) != largest
    np.divide(w, prefix + suffix, out=ratio, where=others)
    before = np.concatenate((np.zeros(1, dtype=w.dtype), np.cumsum(ratio)[:-1]))
    after = np.concatenate((np.cumsum(ratio[::-1])[-2::-1], np.zeros(1, dtype=w.dtype)))
    first = w / w.sum()
    inclusion = first * (1 + before + after)
    # Handle the largest site as the first draw in its own scale. The
    # remaining weights may all vanish when scaled by that site's weight.
    remaining[largest] = 0
    remaining /= remaining.max()
    inclusion += first[largest] * (remaining / remaining.sum())
    return np.asarray(np.clip(inclusion, 0, 1), dtype=np.float64)


def enumerated_wallenius_inclusion(positive_weights, draw_size):
    """Accumulate all sequential draw probabilities by selected subset."""
    scaled = positive_weights / positive_weights.max()
    rescale = bool((scaled < np.finfo(float).tiny).any())
    weights = positive_weights if rescale else scaled
    sites = [(1 << i, float(weight)) for i, weight in enumerate(weights)]
    distribution = {0: 1.}
    for _ in range(draw_size):
        next_distribution: dict[int, float] = {}
        for selected, probability in distribution.items():
            remaining = [(bit, weight) for bit, weight in sites if not (selected & bit)]
            if rescale:
                # Once the large sites are removed, tiny weights may have
                # substantial conditional probability among those still left.
                largest = max(weight for _, weight in remaining)
                remaining = [(bit, weight / largest) for bit, weight in remaining]
            total = sum(weight for _, weight in remaining)
            for bit, weight in remaining:
                updated = selected | bit
                next_distribution[updated] = next_distribution.get(updated, 0.) + probability * (weight / total)
        distribution = next_distribution
    inclusion = np.zeros(len(sites), dtype=np.float64)
    for selected, probability in distribution.items():
        for i, (bit, _) in enumerate(sites):
            if selected & bit:
                inclusion[i] += probability
    return inclusion


def wallenius_mean_inclusion(positive_weights, draw_size):
    """Solve the approximate Wallenius mean equation in log time."""
    from scipy.optimize import brentq

    log_weights = normalized_log_weights(positive_weights)
    split = log_weights.size - draw_size
    order = np.argpartition(log_weights, split)
    log_weights -= log_weights[order[split]]
    top = log_weights[order[split:]]
    rest = log_weights[order[:split]]
    # Once the clock exceeds 746, its survival probability rounds to zero.
    saturation = np.log(746.)

    def clock(log_time, weights):
        return np.exp(np.minimum(log_time + weights, saturation))

    def residual(log_time):
        # Balance the top sites' deficits against the other sites' inclusion
        # masses so probabilities rounded to one cannot flatten the equation.
        included = -np.expm1(-clock(log_time, rest))
        excluded = np.exp(-clock(log_time, top))
        return included.sum() - excluded.sum()

    # All probabilities lie on the same side of draw_size / n at either
    # bound. Pad the log-time interval to allow for rounding at the endpoints.
    uniform_time = np.log(-np.log1p(-draw_size / log_weights.size))
    lo = uniform_time - log_weights.max() - 1.
    hi = uniform_time - log_weights.min() + 1.
    root = brentq(residual, lo, hi, xtol=np.finfo(float).eps, maxiter=200)
    return -np.expm1(-clock(root, log_weights))
