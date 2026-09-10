"""Independent state enumeration, not replication of the pruning algorithm."""
import itertools

import numpy as np
import pytest
from scipy.linalg import expm

from csubst.endpoint import EndpointModel


def enumerate_model(parents, lengths, q, pi, tips, rates, weights):
    n, k = len(parents), len(pi)
    size = next(iter(tips.values())).shape[0]
    nodes = np.zeros((n, size, k))
    edges = {i: np.zeros((size, k, k)) for i, p in enumerate(parents) if p >= 0}
    prediction = {i: np.zeros((size, k, k)) for i in edges}
    norm = np.zeros(size)
    root = list(parents).index(-1)
    for rate, weight in zip(rates, weights):
        transitions = {i: expm(q * lengths[i] * rate) for i in edges}
        for assignment in itertools.product(range(k), repeat=n):
            mass = np.full(size, weight * pi[assignment[root]])
            for child in edges:
                mass *= transitions[child][assignment[parents[child]], assignment[child]]
            for leaf, likelihood in tips.items():
                mass *= likelihood[:, assignment[leaf]]
            norm += mass
            for node, state in enumerate(assignment):
                nodes[node, :, state] += mass
            for child in edges:
                a, d = assignment[parents[child]], assignment[child]
                edges[child][:, a, d] += mass
                prediction[child][:, a, :] += mass[:, None] * transitions[child][a]
    nodes /= norm[None, :, None]
    for child in edges:
        edges[child] /= norm[:, None, None]
        prediction[child] /= norm[:, None, None]
    return nodes, edges, prediction


@pytest.mark.parametrize('block_size', [1, 2, 64])
@pytest.mark.parametrize('rates,weights', [([1.], [1.]), ([0., .2, 2.], [.2, .4, .4])])
def test_joint_and_prediction_match_exhaustive_assignments(block_size, rates, weights):
    parents = [-1, 0, 1, 1, 0]
    lengths = [0, .03, .4, .2, .7]
    pi = np.array([.2, .3, .5])
    q = np.tile(pi, (3, 1)) - np.eye(3)
    tips = {2: np.array([[1., 0, 0], [0, 1, 0], [1, 1, 1]]),
            3: np.array([[0., 1, 0], [0, 1, 0], [1, 1, 1]]),
            4: np.array([[1., 1, 0], [.1, .5, .8], [1, 1, 1]])}
    nodes, edges, predictions = enumerate_model(parents, lengths, q, pi, tips, rates, weights)
    model = EndpointModel(parents, lengths, q, pi, rates, weights)
    for record in model.iter_blocks(tips, block_size=block_size, predictive=True):
        sl = slice(record.start, record.stop)
        np.testing.assert_allclose(record.node, nodes[record.child, sl], atol=2e-14, rtol=0)
        if record.parent >= 0:
            np.testing.assert_allclose(record.joint, edges[record.child][sl], atol=2e-14, rtol=0)
            np.testing.assert_allclose(record.predictive, predictions[record.child][sl], atol=2e-14, rtol=0)
            np.testing.assert_allclose(record.joint.sum(2), nodes[record.parent, sl], atol=2e-14, rtol=0)
            np.testing.assert_allclose(record.joint.sum(1), record.node, atol=2e-14, rtol=0)


@pytest.mark.parametrize('length', [0., 1e-10, 1e-5, .1, 10.])
def test_ambiguous_zero_and_short_branch(length):
    model = EndpointModel([-1, 0, 1, 1, 0], [0, length, 1, 1, 1],
                          [[-1, 1], [1, -1]], [.5, .5])
    tips = {2: np.array([[1., 0]]), 3: np.array([[0., 1]]), 4: np.ones((1, 2))}
    records = {x.child: x for x in model.iter_blocks(tips, predictive=True)}
    edge = records[1]
    expected = -np.expm1(-2 * length) / 2
    assert edge.joint[0, 0, 1] + edge.joint[0, 1, 0] == pytest.approx(expected, abs=1e-15)
    assert edge.predictive[0, 0, 1] + edge.predictive[0, 1, 0] == pytest.approx(expected, abs=1e-15)
    # Same marginals give a spurious .5 under the previous estimator.
    np.testing.assert_allclose(records[0].node, [[.5, .5]], atol=1e-14)
    np.testing.assert_allclose(edge.node, [[.5, .5]], atol=1e-14)


def test_zero_likelihood_fails_instead_of_fabricating_posterior():
    model = EndpointModel([-1, 0, 0], [0, 0, 0], [[-1, 1], [1, -1]], [.5, .5])
    with pytest.raises(ValueError, match='Zero likelihood'):
        list(model.iter_blocks({1: np.array([[1., 0]]), 2: np.array([[0., 1]])}))


def test_state_permutation_and_selected_edges():
    tips = {1: np.array([[.1, .9]]), 2: np.array([[1., 0.]])}
    q = np.array([[-.3, .3], [.7, -.7]])
    m = EndpointModel([-1, 0, 0], [0, .4, .3], q, [.7, .3])
    reference = list(m.iter_blocks(tips))
    m2 = EndpointModel([-1, 0, 0], [0, .4, .3], q[::-1, ::-1], [.3, .7])
    reordered = list(m2.iter_blocks({i: t[:, ::-1] for i, t in tips.items()}, branch_ids=[1]))
    for a, b in zip(reference, reordered):
        np.testing.assert_allclose(a.node, b.node[:, ::-1])
    np.testing.assert_allclose(reference[1].joint, reordered[1].joint[:, ::-1, ::-1])
    assert reordered[2].joint is None


def test_inactive_structural_states_and_missing_tips():
    q = np.array([[-1., 1, 0], [1, -1, 0], [0, 0, 0]])
    m = EndpointModel([-1, 0, 0], [0, .1, .2], q, [.5, .5, 0])
    records = list(m.iter_blocks({1: np.ones((2, 3)), 2: np.ones((2, 3))}))
    for record in records:
        np.testing.assert_allclose(record.node, [[.5, .5, 0]] * 2)


@pytest.mark.parametrize('parents,lengths', [([-1, 2, 1], [0, 1, 1]), ([-1, -1], [0, 0]),
                                           ([-1, 0], [0, -1]), ([-1, 0], [1, 1])])
def test_invalid_tree_rejected(parents, lengths):
    with pytest.raises(ValueError):
        EndpointModel(parents, lengths, [[-1, 1], [1, -1]], [.5, .5])
