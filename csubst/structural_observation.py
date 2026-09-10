"""Reference inference under a fixed reversible Q and observation error model.

This research API does not fit Q, modify production ASR, or compute omegaC.
Edge joints describe endpoint states, not CTMC jump counts or whole histories.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.special import logsumexp


def observation_likelihoods(observed, confusion):
    """Return site-by-true-state P(observation | state); -1 denotes missing."""
    values = np.asarray(observed)
    matrix = np.asarray(confusion, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2
            or not np.isfinite(matrix).all() or np.any(matrix < 0)
            or not np.allclose(matrix.sum(axis=1), 1, atol=1e-12, rtol=0)):
        raise ValueError("Confusion must be finite, nonnegative and row-normalized (true by observed).")
    if values.ndim != 1 or values.dtype.kind not in "iu" or np.any((values < -1) | (values >= len(matrix))):
        raise ValueError("Observations must be integer state indices, with -1 for missing.")
    result: NDArray[np.float64] = np.ones((len(values), len(matrix)), dtype=float)
    known = values >= 0
    result[known] = matrix[:, values[known]].T
    return result


def fixed_gtr_posteriors(parents, branch_lengths, q, pi, tip_likelihoods):
    """Log-space pruning/outside recursion for all nodes and parent-child joints.

    parents is a node-indexed integer vector with one -1 root; branch_lengths[i]
    belongs to the edge entering node i (root length must be zero). pi is positive,
    stationary, and satisfies detailed balance with a unit-mean-rate Q. Every
    leaf must supply an L-by-K likelihood array. Missing observations are rows
    of ones. A wholly missing site returns the model prior, with log evidence 0.
    """
    parents = np.asarray(parents)
    lengths = np.asarray(branch_lengths, dtype=float)
    q, pi = np.asarray(q, dtype=float), np.asarray(pi, dtype=float)
    if (parents.ndim != 1 or not parents.size or parents.dtype.kind not in "iu"
            or np.any((parents < -1) | (parents >= len(parents))) or np.sum(parents == -1) != 1):
        raise ValueError("parents must contain valid node indices and exactly one -1 root.")
    root = int(np.flatnonzero(parents == -1)[0])
    if (lengths.shape != parents.shape or not np.isfinite(lengths).all()
            or np.any(lengths < 0) or lengths[root] != 0):
        raise ValueError("Branch lengths must be finite/nonnegative with zero at the root.")
    k = len(pi) if pi.ndim == 1 else 0
    if (k < 2 or q.shape != (k, k) or not np.isfinite(q).all() or not np.isfinite(pi).all()
            or np.any(pi <= 0) or not np.isclose(pi.sum(), 1, atol=1e-12, rtol=0)):
        raise ValueError("Q and positive normalized pi must have compatible finite state axes.")
    off = q.copy()
    np.fill_diagonal(off, 0)
    flux = pi[:, None] * q
    if (np.any(off < 0) or np.any(q.diagonal() > 0)
            or not np.allclose(q.sum(axis=1), 0, atol=1e-12, rtol=0)
            or not np.allclose(flux, flux.T, atol=1e-12, rtol=1e-10)
            or not np.isclose(-pi @ q.diagonal(), 1, atol=1e-10, rtol=1e-10)):
        raise ValueError("Q must be a reversible generator with unit stationary mean rate.")
    children: list[list[int]] = [[] for _ in parents]
    for child, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(child)
    order, seen = [], set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in seen:
            raise ValueError("Tree contains a cycle.")
        seen.add(node)
        order.append(node)
        stack.extend(children[node])
    if len(seen) != len(parents):
        raise ValueError("Tree is disconnected or cyclic.")
    leaves = {i for i, ch in enumerate(children) if not ch}
    if set(tip_likelihoods) != leaves:
        raise ValueError("Supply likelihoods for exactly the tree's leaves.")
    likelihoods = {i: np.asarray(value, dtype=float) for i, value in tip_likelihoods.items()}
    first = next(iter(likelihoods.values()))
    if first.ndim != 2 or first.shape[1] != k:
        raise ValueError("Tip likelihoods must have site-by-state shape.")
    for value in likelihoods.values():
        if value.shape != first.shape or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("Tip likelihoods must have equal finite nonnegative site-by-state arrays.")
    shape = first.shape
    log_transition = {}
    for node in order[1:]:
        transition = expm(q * lengths[node])
        if transition.min() < -1e-12 or not np.allclose(transition.sum(axis=1), 1, atol=1e-10):
            raise ValueError("Invalid transition matrix from Q.")
        with np.errstate(divide="ignore"):
            log_transition[node] = np.log(np.maximum(transition, 0))
    inside: dict[int, NDArray[np.float64]] = {}
    messages: dict[int, NDArray[np.float64]] = {}
    with np.errstate(divide="ignore"):
        for node in reversed(order):
            value = np.log(likelihoods[node]) if node in leaves else np.zeros(shape)
            for child in children[node]:
                messages[child] = logsumexp(log_transition[child][None, :, :] + inside[child][:, None, :], axis=2)
                value = value + messages[child]
            inside[node] = value
    evidence = logsumexp(inside[root] + np.log(pi), axis=1)
    if not np.isfinite(evidence).all():
        raise ValueError("At least one site has zero likelihood under the specified model.")
    outside = {root: np.broadcast_to(np.log(pi), shape)}
    posterior = np.empty((len(parents), *shape))
    joints = {}
    for node in order:
        posterior[node] = np.exp(outside[node] + inside[node] - evidence[:, None])
        for child in children[node]:
            context = outside[node].copy()
            for sibling in children[node]:
                if sibling != child:
                    context += messages[sibling]
            edge = context[:, :, None] + log_transition[child][None, :, :]
            joints[child] = np.exp(edge + inside[child][:, None, :] - evidence[:, None, None])
            outside[child] = logsumexp(edge, axis=1)
    return {"node_posterior": posterior, "edge_joint": joints,
            "site_log_likelihood": evidence, "root": root,
            "scope": "fixed_gtr_endpoint_inference"}
