"""Numerical helpers for weighted-urn probabilities."""

from functools import lru_cache
from math import isqrt, prod

import numpy as np


ENUMERATION_MAX_SITES = 20
GROUPED_MAX_DRAWS = 256
GROUPED_MAX_STATE_UPDATES = 8_192
_FISHER_SUFFIX_BYTES = 64 * 1024**2


def normalized_log_weights(positive_weights):
    """Normalize positive weights in log space when their ratios are subnormal."""
    largest = positive_weights.max()
    ratios = positive_weights / largest
    normal = ratios >= np.finfo(np.float64).tiny
    np.log(ratios, out=ratios, where=normal)
    ratios[~normal] = np.log(positive_weights[~normal]) - np.log(largest)
    return ratios


def fisher_suffix(log_weights, size, *, terminal=None, following=0):
    """Log subset weights for each suffix, with an optional following block."""
    num_positive = len(log_weights)
    suffix = np.full((num_positive + 1, size + 1), -np.inf, dtype=np.float64)
    suffix[:, 0] = 0.
    if terminal is not None:
        suffix[-1] = terminal
    for index in range(num_positive - 1, -1, -1):
        last = min(size, num_positive - index + following)
        next_row = suffix[index + 1]
        np.logaddexp(next_row[1:last + 1], log_weights[index] + next_row[:last],
                     out=suffix[index, 1:last + 1])
    return suffix


def fisher_suffix_blocks(log_weights, size):
    """Yield suffix tables, reconstructing large tables from boundary rows."""
    n = len(log_weights)
    if (n + 1) * (size + 1) * 8 <= _FISHER_SUFFIX_BYTES:
        yield 0, fisher_suffix(log_weights, size)
        return

    # A square-root block size balances the saved boundary rows and the
    # recomputed rows in one block. Both passes use the same recurrence.
    block_size = max(1, isqrt(n))
    starts = range(0, n, block_size)
    terminal = np.full(size + 1, -np.inf)
    terminal[0] = 0.
    boundaries = {n: terminal}
    for start in reversed(starts[1:]):
        stop = min(start + block_size, n)
        boundary = fisher_suffix(log_weights[start:stop], size, terminal=boundaries[stop], following=n - stop)
        boundaries[start] = boundary[0].copy()
        del boundary
    for start in starts:
        stop = min(start + block_size, n)
        suffix = fisher_suffix(log_weights[start:stop], size, terminal=boundaries.pop(stop), following=n - stop)
        yield start, suffix


def fisher_suffix_rows(log_weights, size):
    for _, suffix in fisher_suffix_blocks(log_weights, size):
        yield from zip(suffix[:-1], suffix[1:])


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


def _tracked_group_shape(sorted_counts, draw_size):
    # The number drawn from the largest group is implied by the other counts.
    return tuple(int(min(count, draw_size)) + 1 for count in sorted_counts[:-1])


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
    if draw_size <= GROUPED_MAX_DRAWS and draw_size * group_states <= GROUPED_MAX_STATE_UPDATES:
        rest = weights[~first]
        if np.all(rest == rest[0]):
            return 'exact_two_weight_groups'
    if draw_size <= GROUPED_MAX_DRAWS:
        _, counts = np.unique(weights, return_counts=True)
        if 3 <= len(counts) <= 4:
            states = prod(_tracked_group_shape(np.sort(counts), draw_size))
            if draw_size * states <= GROUPED_MAX_STATE_UPDATES:
                return 'exact_multiple_weight_groups'
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


def grouped_wallenius_inclusion(positive_weights, draw_size):
    """Exact sequential draw recurrence on counts of selected weight groups."""
    unique, inverse, counts = np.unique(positive_weights, return_inverse=True, return_counts=True)
    order = np.argsort(counts, kind='stable')
    group_sizes = counts[order]
    weights = unique[order, None]
    tiny = np.finfo(float).tiny
    wide = weights.min() / weights.max() < tiny
    shape = _tracked_group_shape(group_sizes, draw_size)
    occupancy = np.indices(shape).reshape(len(shape), -1)
    total_tracked = occupancy.sum(axis=0)
    remaining = group_sizes[:-1, None] - occupancy
    strides = [prod(shape[axis+1:]) for axis in range(len(shape))]
    probability = np.zeros(prod(shape))
    updated = np.zeros_like(probability)
    probability[0] = 1.
    for drawn in range(draw_size):
        active = np.flatnonzero(probability > 0)
        last_remaining = group_sizes[-1] - drawn + total_tracked[active]
        available = np.vstack((remaining[:, active], last_remaining))
        present = available > 0
        # Rescale each state after large-weight groups are depleted. Ratios
        # between the remaining small weights then keep their precision.
        scale = np.where(present, weights, 0.).max(axis=0)
        ratios = np.divide(weights, scale, out=np.zeros(available.shape), where=present)
        masses = ratios * available
        if wide:
            groups, states = np.nonzero(present & (ratios < tiny))
            # The count can make a group mass representable even if its
            # individual weight ratio rounded to zero.
            masses[groups, states] = np.exp(
                (np.log(weights[groups, 0]) - np.log(scale[states])) + np.log(available[groups, states])
            )
        next_prob = (masses / masses.sum(axis=0)) * probability[active]
        updated.fill(0.)
        updated[active] = next_prob[-1]
        for axis, stride in enumerate(strides):
            can_pick = present[axis]
            updated[active[can_pick] + stride] += next_prob[axis, can_pick]
        probability, updated = updated, probability
    means = np.concatenate((occupancy @ probability, [(draw_size - total_tracked) @ probability]))
    inclusion = np.empty_like(unique)
    inclusion[order] = means / group_sizes
    return np.clip(inclusion[inverse], 0., 1.)


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


@lru_cache(maxsize=4)
def _quadrature_rule(order):
    return np.polynomial.legendre.leggauss(order)


def _wallenius_clock_density(log_time, log_weights, draw_size):
    """Density of each log arrival time with rank at most draw_size."""
    num_sites = len(log_weights)
    log_clock = log_time[:, None] + log_weights
    saturation = np.log(746.)
    clock = np.exp(np.minimum(log_clock, saturation))
    survival = np.exp(-clock)
    arrived = -np.expm1(-clock)
    # At a fixed time all other arrivals are independent Bernoulli trials.
    # Count the shorter tail, keeping the probability of reaching its bound
    # in a separate, absorbing bin.
    upper_tail = num_sites - draw_size < draw_size
    limit = num_sites - draw_size if upper_tail else draw_size
    success, failure = (survival, arrived) if upper_tail else (arrived, survival)
    prefix = np.zeros((num_sites, len(log_time), limit + 1))
    prefix[0, :, 0] = 1.
    for site in range(num_sites - 1):
        previous = prefix[site]
        current = prefix[site + 1]
        p, q = success[:, site, None], failure[:, site, None]
        current[:, :-1] = previous[:, :-1] * q
        current[:, 1:-1] += previous[:, :-2] * p
        current[:, -1] = previous[:, -1] + previous[:, -2] * p[:, 0]
    suffix = prefix[0].copy()
    probability = np.empty_like(clock)
    for site in range(num_sites - 1, -1, -1):
        if upper_tail:
            tail = np.cumsum(suffix[:, :0:-1], axis=1)
            probability[:, site] = prefix[site, :, -1] + (prefix[site, :, :-1] * tail).sum(axis=1)
        else:
            head = np.cumsum(suffix[:, :-1], axis=1)[:, ::-1]
            probability[:, site] = (prefix[site, :, :-1] * head).sum(axis=1)
        p, q = success[:, site, None], failure[:, site, None]
        suffix[:, -1] += suffix[:, -2] * p[:, 0]
        suffix[:, 1:-1] = suffix[:, 1:-1] * q + suffix[:, :-2] * p
        suffix[:, 0] *= q[:, 0]
    log_density = np.where(log_clock < saturation, log_clock - clock, -np.inf)
    return np.exp(log_density) * probability


def wallenius_quadrature_inclusion(positive_weights, draw_size):
    """Integrate the exponential arrival density against a Poisson-binomial CDF."""
    log_weights = normalized_log_weights(positive_weights)
    limit = min(draw_size, len(log_weights) - draw_size)
    if 8 * len(log_weights) * (limit + 1) > 64 * 1024**2:
        raise ValueError('Wallenius quadrature requires more than 64 MiB for one integration point.')
    # For every site the omitted clock mass at either end is at most 1e-14.
    tail = 1e-14
    lower = np.log(tail)
    upper = np.log(-np.log(tail)) - log_weights.min()
    boundaries = np.linspace(lower, upper, int(np.ceil((upper - lower) / 4)) + 1)
    pending = [(lo, hi, 0) for lo, hi in zip(boundaries[:-1], boundaries[1:])]
    total = np.zeros_like(log_weights)
    # Bound the number of time points in the prefix workspace. A single time
    # point still requires n * (limit + 1) doubles.
    batch_size = max(1, (32 * 1024**2) // (8 * len(log_weights) * (limit + 1)))

    def integrate(lo, hi, order):
        nodes, weights = _quadrature_rule(order)
        half = (hi - lo) / 2
        value = np.zeros_like(log_weights)
        for start in range(0, order, batch_size):
            sl = slice(start, start + batch_size)
            points = lo + half * (1. + nodes[sl])
            density = _wallenius_clock_density(points, log_weights, draw_size)
            value += weights[sl] @ density
        return half * value

    while pending:
        lo, hi, depth = pending.pop()
        coarse = integrate(lo, hi, 8)
        fine = integrate(lo, hi, 16)
        if np.all(np.abs(coarse - fine) <= 1e-11 * (hi - lo) / (upper - lower) + 1e-9 * fine):
            total += fine
        elif depth < 12:
            middle = (lo + hi) / 2
            pending.extend(((lo, middle, depth + 1), (middle, hi, depth + 1)))
        else:
            raise ValueError('Wallenius quadrature did not converge.')
    if (total.max() > 1. + 1e-9
            or not np.isclose(total.sum(), draw_size, rtol=1e-8, atol=1e-10)):
        raise ValueError('Wallenius quadrature did not recover the branch draw size.')
    return np.minimum(total, 1.)
