"""Exhaustively verify the sister-branch diagnostics on a small tree."""

import itertools

import numpy as np
from scipy.linalg import expm

from csubst.endpoint import EndpointModel

from sister_n_diagnostics import log_no_event, scores

def main():
    parents = np.array([-1, 0, 0, 1, 1])
    lengths = [0., .3, .8, .2, .4]
    pi = np.array([.2, .3, .5])
    Q = np.tile(pi, (3, 1)) - np.eye(3)
    rates, weights = [.3, 2.], [.2, .8]
    aa = np.array([0, 0, 1])
    pairs = np.array([[1, 2], [3, 4]])
    parent_truth = np.array([[0, 1, 0], [1, 0, 0]])
    tips = {2: np.eye(3)[[0, 2, 2]], 3: np.eye(3)[[0, 0, 0]], 4: np.eye(3)[[0, 0, 1]]}
    missing = {leaf: np.zeros(3, dtype=bool) for leaf in tips}
    missing[4][1] = True
    both = np.array([[True, True, True], [True, False, True]])
    m = EndpointModel(parents, lengths, Q, pi, rates, weights)
    states = np.array(list(itertools.product(range(3), repeat=len(parents))))
    mass = np.zeros((2, len(states), 3))
    for c, (r, w) in enumerate(zip(rates, weights)):
        prior = w * pi[states[:, 0]]
        for child in range(1, len(parents)):
            transition = expm(Q * lengths[child] * r)
            prior *= transition[states[:, parents[child]], states[:, child]]
        mass[c] = prior[:, None]
        for leaf, values in tips.items():
            emissions = np.where(missing[leaf][:, None], 1., values)
            mass[c] *= emissions[:, states[:, leaf]].T
    pp = mass / mass.sum(axis=(0, 1))
    p_no_event = np.ones(pp.shape[1:], dtype=bool)
    expected = np.zeros((len(pairs), 3, 6))
    for i, (a,b) in enumerate(pairs):
        p = aa[states[:, parents[a]]]
        a, b = aa[states[:, a]], aa[states[:, b]]
        event = (a == b) & (p != a)
        p_no_event &= ~(event[:, None] & both[i])
        rate = pp.sum(axis=1)
        for d in np.unique(aa):
            x = pp[:, (a == d) & (a != p)].sum(axis=1)
            y = pp[:, (b == d) & (b != p)].sum(axis=1)
            expected[i, :, 0] += x.sum(axis=0) * y.sum(axis=0)
            expected[i, :, 1] += (x * y / rate).sum(axis=0)
        expected[i, :, 2] = pp[:, event].sum(axis=(0, 1))
        for s in range(3):
            known = p == parent_truth[i, s]
            expected[i, s, 5] = pp[:, known, s].sum()
            expected[i, s, 3] = pp[:, known & event, s].sum() / expected[i, s, 5]
            expected[i, s, 4] = pp[:, ~known & event, s].sum()
    actual = scores(m, tips, aa, pairs, parent_truth, missing, both)
    expected[~both] = 0.
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
    np.testing.assert_allclose(np.exp(log_no_event(m, tips, aa, pairs, both, missing)), (pp * p_no_event).sum(axis=(0, 1)).prod(),
                               atol=2e-14, rtol=0)
    print('Sister scores and probability of no events match independent enumeration.')


if __name__ == '__main__':
    main()
