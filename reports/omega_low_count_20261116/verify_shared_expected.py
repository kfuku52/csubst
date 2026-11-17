"""Independently enumerate the predictive EC variants on a small tree."""

from itertools import product

import numpy as np
from scipy.linalg import expm

from csubst.endpoint import EndpointModel

from shared_expected import predicted_counts


def main():
    parents = np.array([-1, 0, 0, 1, 1, 2, 2])
    lengths = [0., .3, .8, .5, .2, .4, .1]
    pi = np.array([.2, .3, .5])
    q = np.tile(pi, (3, 1)) - np.eye(3)
    rates, weights = [0., .8, 2.], [.2, .35, .45]
    amino = np.array([0, 0, 1])
    tips = {3: np.eye(3)[[0, 1, 0]], 4: np.eye(3)[[1, 2, 2]], 5: np.eye(3)[[0, 1, 2]],
            6: np.array([[1., 1., 0.], [0., 0., 0.], [0., 1., 0.]])}
    missing = {6: np.array([False, True, False])}
    eligible = np.ones((len(parents), 3), dtype=bool)
    eligible[6, 1] = False
    pairs = np.array([(3, 4), (3, 5), (3, 6), (1, 5)])
    assignments = np.array(list(product(range(len(pi)), repeat=len(parents))))
    transitions = np.array([[expm(q * length * rate) for length in lengths] for rate in rates])
    mass = np.empty((len(rates), len(assignments), 3))
    for c, weight in enumerate(weights):
        prior = weight * pi[assignments[:, 0]]
        for child in range(1, len(parents)):
            prior *= transitions[c, child, assignments[:, parents[child]], assignments[:, child]]
        mass[c] = prior[:, None]
        for leaf, values in tips.items():
            emissions = np.where(missing[leaf][:, None], 1., values) if leaf in missing else values
            mass[c] *= emissions[:, assignments[:, leaf]].T
    pp = mass / mass.sum(axis=(0, 1))
    rate_pp = pp.sum(axis=1)
    inv = np.divide(1., rate_pp, out=np.zeros_like(rate_pp), where=rate_pp > 0.)
    expected = np.zeros((3, len(pairs), 2))
    for i, (a, b) in enumerate(pairs):
        mask = eligible[a] & eligible[b]
        for channel, names in enumerate((amino, np.arange(len(pi)))):
            for d in np.unique(names):
                projected = []
                for child in (a, b):
                    previous = assignments[:, parents[child]]
                    transition = transitions[:, child, previous, :]
                    changed = amino[previous, None] != amino
                    if channel == 1:
                        changed = ~changed & (previous[:, None] != np.arange(len(pi)))
                    projected.append((transition * (changed & (names == d))).sum(axis=-1))
                x, y = projected
                x_rate = (pp * x[..., None]).sum(axis=1)
                y_rate = (pp * y[..., None]).sum(axis=1)
                expected[0, i, channel] += (x_rate.sum(axis=0) * y_rate.sum(axis=0))[mask].sum()
                expected[1, i, channel] += (x_rate * y_rate * inv)[:, mask].sum()
                expected[2, i, channel] += (pp * (x * y)[..., None])[..., mask].sum()
    model = EndpointModel(parents, lengths, q, pi, rates, weights)
    for size in (1, 2, 4):
        totals = predicted_counts(model, tips, amino, pairs, size, tip_missing=missing, eligible=eligible)
        np.testing.assert_allclose(totals, expected, rtol=0., atol=2e-14)
    # With one duplicated called pattern, grouping sites must preserve all three sums.
    one = {leaf: np.eye(3)[[index]] for leaf, index in zip(tips, (0, 1, 0, 2))}
    four = {leaf: np.repeat(value, 4, axis=0) for leaf, value in one.items()}
    np.testing.assert_allclose(predicted_counts(model, four, amino, pairs, 2),
                               4 * predicted_counts(model, one, amino, pairs, 2), rtol=0., atol=2e-14)
    print('Separate, shared-rate, and joint expected counts match exhaustive enumeration.')


if __name__ == '__main__':
    main()
