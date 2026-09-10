import itertools

import numpy as np
import pytest
from scipy.linalg import expm

from csubst.structural_observation import fixed_gtr_posteriors, observation_likelihoods


PARENTS = np.array([-1, 0, 1, 1, 0])
LENGTHS = np.array([0, .2, .3, .4, .5])
PI = np.array([.6, .4])
Q = np.array([[-1 / 1.2, 1 / 1.2], [1 / .8, -1 / .8]])


def enumerate_states(parents, lengths, q, pi, tips):
    """Independent finite sum over every node-state assignment."""
    root = int(np.flatnonzero(parents == -1)[0])
    edges = {i: expm(q * lengths[i]) for i in range(len(parents)) if i != root}
    sites = next(iter(tips.values())).shape[0]
    mass = np.zeros(sites)
    nodes = np.zeros((len(parents), sites, len(pi)))
    joints = {i: np.zeros((sites, len(pi), len(pi))) for i in edges}
    for states in itertools.product(range(len(pi)), repeat=len(parents)):
        weight = np.full(sites, pi[states[root]])
        for child, transition in edges.items():
            weight *= transition[states[parents[child]], states[child]]
        for node, likelihood in tips.items():
            weight *= likelihood[:, states[node]]
        mass += weight
        for node, state in enumerate(states):
            nodes[node, :, state] += weight
        for child in edges:
            joints[child][:, states[parents[child]], states[child]] += weight
    return mass, nodes / mass[None, :, None], {i: j / mass[:, None, None] for i, j in joints.items()}


@pytest.mark.parametrize("confusion", [np.eye(2), np.array([[.9, .1], [.3, .7]])])
def test_fixed_gtr_matches_full_enumeration_with_missing_sites(confusion):
    tips = {2: observation_likelihoods(np.array([0, 1, -1]), confusion),
            3: observation_likelihoods(np.array([1, 1, -1]), confusion),
            4: observation_likelihoods(np.array([0, -1, -1]), confusion)}
    result = fixed_gtr_posteriors(PARENTS, LENGTHS, Q, PI, tips)
    mass, nodes, joints = enumerate_states(PARENTS, LENGTHS, Q, PI, tips)
    np.testing.assert_allclose(result["site_log_likelihood"], np.log(mass), atol=1e-14)
    np.testing.assert_allclose(result["node_posterior"], nodes, atol=1e-14)
    for child, joint in joints.items():
        np.testing.assert_allclose(result["edge_joint"][child], joint, atol=1e-14)
        np.testing.assert_allclose(joint.sum(axis=2), nodes[PARENTS[child]], atol=1e-14)
        np.testing.assert_allclose(joint.sum(axis=1), nodes[child], atol=1e-14)
    np.testing.assert_allclose(nodes[:, 2], np.broadcast_to(PI, (5, 2)), atol=1e-14)


def test_observation_likelihood_orientation_is_not_posterior():
    matrix = np.array([[.9, .1], [.3, .7]])
    np.testing.assert_array_equal(observation_likelihoods(np.array([0, 1, -1]), matrix),
                                  [[.9, .3], [.1, .7], [1, 1]])


def test_reversible_reroot_preserves_all_node_marginals():
    tips = {i: np.array([[.9, .1], [.3, .7]]) for i in (2, 3, 4)}
    before = fixed_gtr_posteriors(PARENTS, LENGTHS, Q, PI, tips)
    after = fixed_gtr_posteriors(np.array([1, -1, 1, 1, 0]), [.2, 0, .3, .4, .5], Q, PI, tips)
    np.testing.assert_allclose(before["node_posterior"], after["node_posterior"], atol=1e-14)
    np.testing.assert_allclose(before["site_log_likelihood"], after["site_log_likelihood"], atol=1e-14)
    np.testing.assert_allclose(before["edge_joint"][1], after["edge_joint"][0].transpose(0, 2, 1), atol=1e-14)


def test_state_relabeling_and_twenty_state_axes():
    k = 20
    pi = np.full(k, 1 / k)
    q = (np.ones((k, k)) - k * np.eye(k)) / (k - 1)
    tips = {i: np.eye(k)[[i, (i + 1) % k]] for i in (2, 3, 4)}
    before = fixed_gtr_posteriors(PARENTS, LENGTHS, q, pi, tips)
    permutation = np.random.default_rng(1).permutation(k)
    after = fixed_gtr_posteriors(PARENTS, LENGTHS, q[np.ix_(permutation, permutation)], pi[permutation],
                                 {i: value[:, permutation] for i, value in tips.items()})
    np.testing.assert_allclose(after["node_posterior"], before["node_posterior"][:, :, permutation], atol=1e-14)
    np.testing.assert_allclose(after["site_log_likelihood"], before["site_log_likelihood"], atol=1e-14)


def test_edge_joint_is_not_product_of_marginals():
    tips = {i: np.ones((1, 2)) for i in (2, 3, 4)}
    result = fixed_gtr_posteriors(PARENTS, LENGTHS, Q, PI, tips)
    assert not np.allclose(result["edge_joint"][1][0], PI[:, None] * PI[None, :])


def test_nonuniform_four_state_gtr_matches_enumeration():
    pi = np.array([.1, .2, .3, .4])
    rates = np.array([[0, 1, 2, .4], [1, 0, .7, 1.3], [2, .7, 0, .2], [.4, 1.3, .2, 0]])
    q = rates * pi
    np.fill_diagonal(q, -q.sum(axis=1))
    q /= -pi @ q.diagonal()
    tips = {2: np.array([[.7, .1, .1, .1]]), 3: np.array([[.2, .5, .1, .2]]),
            4: np.array([[.1, .2, .3, .4]])}
    mass, nodes, joints = enumerate_states(PARENTS, LENGTHS, q, pi, tips)
    result = fixed_gtr_posteriors(PARENTS, LENGTHS, q, pi, tips)
    np.testing.assert_allclose(result["site_log_likelihood"], np.log(mass), atol=1e-14)
    np.testing.assert_allclose(result["node_posterior"], nodes, atol=1e-14)
    for child, joint in joints.items():
        np.testing.assert_allclose(result["edge_joint"][child], joint, atol=1e-14)


def test_log_recursion_handles_underflow_and_zero_edges():
    n = 1200
    parents = np.array([-1] + [0] * n)
    lengths = np.array([0] + [.1] * n)
    tips = {i: np.array([[.01, .02]]) for i in range(1, n + 1)}
    result = fixed_gtr_posteriors(parents, lengths, Q, PI, tips)
    assert result["site_log_likelihood"][0] < -4000
    np.testing.assert_allclose(result["node_posterior"].sum(axis=2), 1, atol=1e-9)
    simple = fixed_gtr_posteriors([-1, 0], [0, 0], Q, PI, {1: np.ones((1, 2))})
    np.testing.assert_allclose(simple["edge_joint"][1][0], np.diag(PI), atol=1e-14)


@pytest.mark.parametrize("parents,lengths", [([-1, 2, 1], [0, .1, .1]),
                                           ([-1, 0], [0, -.1]), ([-1, 0], [.1, .2])])
def test_invalid_trees_rejected(parents, lengths):
    with pytest.raises(ValueError):
        fixed_gtr_posteriors(parents, lengths, Q, PI, {1: np.ones((1, 2))})


def test_impossible_observations_and_nonreversible_q_rejected():
    with pytest.raises(ValueError, match="zero likelihood"):
        fixed_gtr_posteriors([-1, 0], [0, .1], Q, PI, {1: np.zeros((1, 2))})
    bad = np.array([[-1, 1, 0], [0, -1, 1], [1, 0, -1]])
    with pytest.raises(ValueError, match="reversible"):
        fixed_gtr_posteriors([-1, 0], [0, .1], bad, np.ones(3) / 3, {1: np.ones((1, 3))})
