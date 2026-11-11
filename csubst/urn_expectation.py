"""Numerical helpers for weighted-urn probabilities."""

import numpy as np


ENUMERATION_MAX_SITES = 20
TWO_GROUP_MAX_DRAWS = 256
TWO_GROUP_MAX_STATE_UPDATES = 8_192


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


def wallenius_sample_blocks(positive_sites, positive_weights, size, niter, rng):
    """Yield row blocks of Efraimidis-Spirakis weighted site draws."""
    nsites = len(positive_sites)
    kth = nsites - size
    max_float = np.finfo(np.float32).max
    # A positive Generator.random() draw has -log(u) < 37. Use log
    # waiting times when direct float32 keys could overflow or lose weights.
    wide = positive_weights.min() < 64. / max_float or positive_weights.max() > max_float
    weights = normalized_log_weights(positive_weights) if wide else positive_weights.astype(np.float32)
    block_size = max(1, (32 * 1024**2) // (16 * nsites))
    for start in range(0, niter, block_size):
        stop = min(start + block_size, niter)
        uniforms = rng.random((stop - start, nsites))
        if wide:
            keys = uniforms
            with np.errstate(divide='ignore'):
                np.log(keys, out=keys)
                np.negative(keys, out=keys)
                np.log(keys, out=keys)
            np.negative(keys, out=keys)
            keys += weights
        else:
            keys = uniforms.astype(np.float32)
            with np.errstate(divide='ignore'):
                np.log(keys, out=keys)
            keys /= weights
        del uniforms
        order = np.argpartition(keys, kth=kth, axis=1)
        del keys
        sites = positive_sites[order[:, kth:]]
        del order
        yield start, sites


def wallenius_method(positive_weights, draw_size):
    weights = np.asarray(positive_weights, dtype=np.float64)
    n = weights.size
    if draw_size == 0 or n == 0 or draw_size >= n:
        return 'exact_boundary'
    first = weights == weights[0]
    first_size = int(np.count_nonzero(first))
    if first_size == n:
        return 'exact_uniform'
    if draw_size <= 2:
        return 'exact_small_draw'
    group_states = min(first_size, n - first_size, draw_size) + 1
    if draw_size <= TWO_GROUP_MAX_DRAWS and draw_size * group_states <= TWO_GROUP_MAX_STATE_UPDATES:
        rest = weights[~first]
        if np.all(rest == rest[0]):
            return 'exact_two_weight_groups'
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


def two_group_wallenius_inclusion(positive_weights, draw_size):
    """Exact recurrence on the number of sites selected from one weight group."""
    first = positive_weights == positive_weights[0]
    first_size = int(np.count_nonzero(first))
    second_size = positive_weights.size - first_size
    if first_size > second_size:
        first = ~first
        first_size, second_size = second_size, first_size
    weights = np.array([positive_weights[first][0], positive_weights[~first][0]])
    log_weights = normalized_log_weights(weights)
    log_masses = []
    for count, log_weight in zip((first_size, second_size), log_weights):
        entries = np.empty(count + 1)
        entries[0] = -np.inf
        entries[1:] = log_weight + np.log(np.arange(1, count + 1))
        log_masses.append(entries)
    first_mass, second_mass = log_masses
    selected_counts = np.arange(min(first_size, draw_size) + 1)
    # The spare cell receives zero when no further first-group draw is possible.
    probability = np.zeros(selected_counts.size + 1)
    updated = np.zeros_like(probability)
    probability[0] = 1.
    for drawn in range(draw_size):
        lo = max(0, drawn - second_size)
        hi = min(drawn, first_size) + 1
        selected = selected_counts[lo:hi]
        log_odds = first_mass[first_size - selected] - second_mass[second_size - drawn + selected]
        from_state = probability[lo:hi]
        updated.fill(0.)
        # Evaluate both choices directly to retain a small probability when the
        # other group is almost certain to supply the next draw.
        updated[lo:hi] = from_state * np.exp(-np.logaddexp(0., log_odds))
        updated[lo + 1:hi + 1] += from_state * np.exp(-np.logaddexp(0., -log_odds))
        probability, updated = updated, probability
    first_inclusion = (probability[:-1] @ selected_counts) / first_size
    second_inclusion = (probability[:-1] @ (draw_size - selected_counts)) / second_size
    return np.clip(np.where(first, first_inclusion, second_inclusion), 0., 1.)


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
