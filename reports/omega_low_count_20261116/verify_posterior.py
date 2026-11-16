"""Compare the research posterior replay with exhaustive finite-state sums."""

from itertools import product

import numpy as np
from scipy.linalg import expm

from csubst.endpoint import EndpointModel
from csubst.endpoint_pair import _edge_kernel
from endpoint_posterior import ConditionalEdges, count_summary, joint_distribution, pair_probabilities
from joint_pair_benchmark_kernel import exact_counts
from pair_counts_compact import compact_counts


def verify_branch_pairs():
    parents = [-1, 0, 0, 1, 1, 3, 2, 2, 0]
    lengths = [0., .3, .12, .44, .09, .7, .11, .2, .15]
    pi = np.array([.2, .5, .3])
    q = np.tile(pi, (len(pi), 1)) - np.eye(len(pi))
    rates, weights = [0., .45, 2.1], [.15, .35, .5]
    aas = np.array([0, 0, 1])
    pairs = [(4, 5), (6, 7), (5, 7), (3, 6), (5, 8), (7, 8)]
    tips = {
        4: np.array([[1., 0, 0], [1., 1, 1], [0., 1, 0]]),
        5: np.array([[0., 1, 0], [0., 0, 1], [0., 1, 0]]),
        6: np.array([[0., 0, 1], [1., 0, 1], [0., 1, 0]]),
        7: np.array([[1., 0, 0], [1., 0, 0], [0., 1, 0]]),
        8: np.array([[0., 1, 0], [0., 0, 1], [0., 1, 0]]),
    }
    states = np.array(list(product(range(len(pi)), repeat=len(parents))))
    site_count = next(iter(tips.values())).shape[0]
    posterior = np.empty((len(rates), len(states), site_count))
    for c, (rate, weight) in enumerate(zip(rates, weights)):
        prior = weight * pi[states[:, 0]]
        for child in range(1, len(parents)):
            transition = expm(q * (lengths[child] * rate))
            prior *= transition[states[:, parents[child]], states[:, child]]
        evidence = np.broadcast_to(prior[:, None], posterior[c].shape).copy()
        for tip, likelihood in tips.items():
            evidence *= likelihood[:, states[:, tip]].T
        posterior[c] = evidence
    posterior /= posterior.sum(axis=(0, 1))
    rate_posterior = posterior.sum(axis=1)
    expected = np.zeros((3, len(pairs), site_count, 2))
    for i, (a, b) in enumerate(pairs):
        from_a, from_b = states[:, parents[a]], states[:, parents[b]]
        to_a, to_b = states[:, a], states[:, b]
        for j, symbol in enumerate((aas, np.arange(len(pi)))):
            if j == 0:
                event_a, event_b = aas[from_a] != aas[to_a], aas[from_b] != aas[to_b]
            else:
                event_a = (from_a != to_a) & (aas[from_a] == aas[to_a])
                event_b = (from_b != to_b) & (aas[from_b] == aas[to_b])
            for destination in np.unique(symbol):
                a_match = event_a & (symbol[to_a] == destination)
                b_match = event_b & (symbol[to_b] == destination)
                a_category = posterior[:, a_match].sum(axis=1)
                b_category = posterior[:, b_match].sum(axis=1)
                expected[0, i, :, j] += a_category.sum(axis=0) * b_category.sum(axis=0)
                joint_category = np.divide(a_category * b_category, rate_posterior, where=rate_posterior > 0,
                                           out=np.zeros_like(a_category))
                expected[1, i, :, j] += joint_category.sum(axis=0)
                expected[2, i, :, j] += posterior[:, a_match & b_match].sum(axis=(0, 1))
    model = EndpointModel(parents, lengths, q, pi, rates, weights)
    for size in (1, 2, 32):
        actual = pair_probabilities(model, tips, aas, pairs, size)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
        for release in (False, True):
            totals = exact_counts(model, tips, aas, pairs, size, release=release)
            np.testing.assert_allclose(totals, expected[2].sum(axis=1), rtol=0, atol=2e-14)
        for compose in (False, True):
            totals = compact_counts(model, tips, aas, pairs, size, compose=compose)
            np.testing.assert_allclose(totals, expected[2].sum(axis=1), rtol=0, atol=2e-14)
    repeated = {node: values[[0, 0, 2, 2, 0]] for node, values in tips.items()}
    for compose in (False, True):
        totals = compact_counts(model, repeated, aas, pairs, 1, compose=compose)
        np.testing.assert_allclose(totals, 3 * expected[2][:, 0] + 2 * expected[2][:, 2], rtol=0, atol=2e-14)


def verify_count_posterior():
    probs = np.array([[.1, .3], [0., 0.], [.2, .6], [.6, .3]])
    m = len(probs)
    expected = np.zeros((m+1, m+1))
    for outcomes in product(range(3), repeat=m):
        p = 1.
        for j, outcome in enumerate(outcomes):
            n, s = probs[j]
            p *= (1 - n - s, n, s)[outcome]
        expected[outcomes.count(1), outcomes.count(2)] += p
    actual = joint_distribution(probs)
    shape = tuple(slice(0, length) for length in actual.shape)
    np.testing.assert_allclose(actual, expected[shape], rtol=0, atol=2e-14)
    for cases, truth, undefined, infinite, zero in (
        ([[0., 0.]], (0, 0), 1., 0., 0.),
        ([[1., 0.]], (1, 0), 0., 1., 0.),
        ([[0., 1.]], (0, 1), 0., 0., 1.),
    ):
        result = count_summary(np.asarray(cases), truth, (.5, .5))
        assert (result['p_undefined'], result['p_infinite'], result['p_zero']) == (undefined, infinite, zero)
        assert result['n_covered'] and result['s_covered']
        assert result['ratio_covered'] is None if undefined else result['ratio_covered']


def verify_composed_paths():
    parents = [-1, 0, 0, 1, 1, 3, 3, 2, 2, 5]
    pi = np.array([.2, .5, .3])
    q = np.tile(pi, (len(pi), 1)) - np.eye(len(pi))
    model = EndpointModel(parents, [0., .5, .3, .4, .2, .7, .8, .2, .1, .6], q, pi,
                          [.3, 1., 2.], [.3, .4, .3])
    tips = {node: np.eye(3)[(np.arange(7) + node) % 3] for node in (4, 6, 7, 8, 9)}
    amino = np.array([0, 0, 1])
    # Branch 5 needs its feature at the root for (5, 2). The (5, 4)
    # pair can use a downward query at node 1, so the feature skips node 1.
    pairs = [(5, 2), (5, 4), (5, 7), (6, 7)]
    expected = pair_probabilities(model, tips, amino, pairs, block_size=2)[2].sum(axis=1)
    for block_size in (1, 2, 4):
        actual = compact_counts(model, tips, amino, pairs, block_size, compose=True)
        np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-14)


def verify_edge_kernel():
    transitions = np.array([[[1., 1e-310], [.7, .3]], [[1., 0.], [0., 1.]]])
    likelihoods = np.array([[[0., 1.], [.6, .4]], [[0., 1.], [1., 0.]]])
    reducer = ConditionalEdges(len(transitions))
    expected = np.stack([reducer(None, row, transition, None, None)[c]
                         for c, (transition, row) in enumerate(zip(transitions, likelihoods))])
    actual = _edge_kernel(transitions, likelihoods)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-14)


if __name__ == '__main__':
    verify_branch_pairs()
    verify_count_posterior()
    verify_composed_paths()
    verify_edge_kernel()
    print('Branch posteriors and count probabilities match exhaustive enumeration.')
