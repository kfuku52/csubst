"""Numerical helpers for weighted-urn probabilities."""

import numpy as np


ENUMERATION_MAX_SITES = 20


def fisher_log_weights(positive_weights):
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
    w = np.asarray(weights, dtype=np.longdouble)
    w = w / w.max()
    prefix = np.concatenate((np.zeros(1, dtype=w.dtype), np.cumsum(w)[:-1]))
    suffix = np.concatenate((np.cumsum(w[::-1])[-2::-1], np.zeros(1, dtype=w.dtype)))
    ratio = w / (prefix + suffix)
    before = np.concatenate((np.zeros(1, dtype=w.dtype), np.cumsum(ratio)[:-1]))
    after = np.concatenate((np.cumsum(ratio[::-1])[-2::-1], np.zeros(1, dtype=w.dtype)))
    return np.asarray(np.clip((w / w.sum()) * (1 + before + after), 0, 1), dtype=np.float64)
