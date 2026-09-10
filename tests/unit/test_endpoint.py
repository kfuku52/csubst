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


@pytest.mark.parametrize('length', [0, 1e-10, 1e-5, .1, 1, 10, 100])
def test_shared_transition_powers_match_expm_for_rare_events(length):
    # A chain forces transitions across many jumps, exercising tiny entries
    # that can lose relative accuracy under spectral reconstruction.
    k = 8
    q = np.zeros((k, k))
    for i in range(k - 1):
        q[i, i + 1] = q[i + 1, i] = .7
    np.fill_diagonal(q, -q.sum(axis=1))
    model = EndpointModel([-1, 0], [0, length], q, np.ones(k) / k)
    expected = expm(q * length)
    if 0 < length < .01:
        # Pade is not an entrywise-relative reference for these tiny multi-hop
        # probabilities. Independently sum exp(Qt) in decimal arithmetic.
        from decimal import Decimal, localcontext
        with localcontext() as ctx:
            ctx.prec = 110
            operator = [[Decimal(str(q[i, j])) * Decimal(str(length)) for j in range(k)] for i in range(k)]
            term = [[Decimal(int(i == j)) for j in range(k)] for i in range(k)]
            total = [row.copy() for row in term]
            for n in range(1, 50):
                term = [[sum(term[i][z] * operator[z][j] for z in range(k)) / n
                         for j in range(k)] for i in range(k)]
                total = [[total[i][j] + term[i][j] for j in range(k)] for i in range(k)]
                if max(abs(x) for row in term for x in row) < Decimal('1e-100'):
                    break
            expected = np.array(total, dtype=float)
    np.testing.assert_allclose(model.transition(1, 0), expected, atol=1e-300, rtol=3e-14)
    assert model._uniform_powers.nbytes <= 8 * 1024 * 1024


def test_shared_transition_nonreversible_and_zero_generator():
    q = np.array([[-1., 1, 0], [0, -1., 1], [1, 0, -1.]])
    model = EndpointModel([-1, 0, 0], [0, .3, .8], q, np.ones(3) / 3)
    for branch in [1, 2]:
        np.testing.assert_allclose(model.transition(branch, 0), expm(q * model.lengths[branch]), atol=1e-15)
    zero = EndpointModel([-1, 0], [0, 10], np.zeros((3, 3)), np.ones(3) / 3)
    np.testing.assert_array_equal(zero.transition(1, 0), np.eye(3))
