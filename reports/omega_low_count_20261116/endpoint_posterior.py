"""Research replay for the joint posterior of any2spe events on two branches."""

from functools import lru_cache
from itertools import groupby

import numpy as np

from csubst.endpoint import _conditional_transition


def pair_paths(parents, pair):
    paths = []
    for child in pair:
        path = [int(child)]
        while parents[path[-1]] >= 0:
            path.append(int(parents[path[-1]]))
        paths.append(path)
    left, right = paths
    ancestor = next(node for node in left if node in right)
    paths = [tuple(path[:path.index(ancestor)]) for path in paths]
    if not all(paths):
        raise ValueError('Focal branches must lie on separate LCA child paths.')
    return ancestor, *paths


class ConditionalEdges:
    def __init__(self, categories):
        self.categories = categories
        self.index = 0

    def __call__(self, left, right, transition, time, weight):
        category = self.index % self.categories
        self.index += 1
        kernel = transition[None, :, :] * right[:, None, :]
        mass = kernel.sum(axis=-1, keepdims=True)
        np.divide(kernel, mass, out=kernel, where=mass > 0)
        rare = ((mass < np.finfo(float).tiny ** .5) & (mass > 0)).any(axis=(1, 2))
        for site in np.flatnonzero(rare):
            with np.errstate(divide='ignore'):
                kernel[site] = _conditional_transition(transition, np.log(right[site]))
        return {category: kernel}


def pair_probabilities(model, tips, amino_acids, pairs, block_size=16):
    """Return factorized, shared-rate, and jointly conditioned probabilities."""
    plans = [pair_paths(model.parents, pair) for pair in pairs]
    selected = {edge for _, *paths in plans for path in paths for edge in path}
    focal = {path[0] for _, *paths in plans for path in paths}
    node_ids = {node for node, *_ in plans} | {int(model.parents[branch]) for branch in focal}
    counts = len(model.rates)
    transform = ConditionalEdges(counts)
    length = len(next(iter(tips.values())))
    probs = np.empty((3, len(pairs), length, 2))
    amino_count = int(max(amino_acids)) + 1
    membership = np.eye(amino_count)[amino_acids]
    nonsyn = amino_acids[:, None] != amino_acids
    syn = ~nonsyn & ~np.eye(len(amino_acids), dtype=bool)
    channels = (slice(None, amino_count), slice(amino_count, None))

    records = model.iter_blocks(
        tips, block_size=block_size, branch_ids=selected, joint=False, transform=transform,
        transform_rate_context=True, category_nodes=True, rate_posterior=True,
    )
    for _, block in groupby(records, key=lambda record: record.start):
        first = next(block)
        posteriors, kernels = {first.child: first.category_node}, {}
        for record in block:
            if record.child in node_ids:
                posteriors[record.child] = record.category_node
            if record.child in selected:
                kernels[record.child] = np.stack([record.reduced[k] for k in range(counts)])
        direct = {
            child: np.concatenate(((kernels[child] * nonsyn) @ membership, kernels[child] * syn), axis=-1)
            for child in focal
        }
        weights = first.rate_posterior
        # Conditional event features at the focal parent, integrated over
        # its states. Keeping the joint category weights separate avoids two
        # independent copies of a rate which is shared within each site.
        marginals = {
            child: np.einsum('csk,cskd->csd', posteriors[model.parents[child]], values)
            for child, values in direct.items()
        }
        per_branch = {child: table.sum(axis=0) for child, table in marginals.items()}
        conditional_marginals = {
            child: np.divide(table, weights[:, :, None], out=np.zeros_like(table), where=weights[:, :, None] > 0)
            for child, table in marginals.items()
        }

        @lru_cache(maxsize=None)
        def to_common_ancestor(path):
            if len(path) == 1:
                return direct[path[0]]
            return kernels[path[-1]] @ to_common_ancestor(path[:-1])

        sl = slice(first.start, first.stop)
        for i, (ancestor, left, right) in enumerate(plans):
            x, y = to_common_ancestor(left), to_common_ancestor(right)
            a, b = left[0], right[0]
            for j, channel in enumerate(channels):
                probs[0, i, sl, j] = (per_branch[a][:, channel] * per_branch[b][:, channel]).sum(axis=1)
                probs[1, i, sl, j] = (marginals[a][..., channel] * conditional_marginals[b][..., channel]).sum(axis=(0, 2))
                probs[2, i, sl, j] = np.einsum('csk,cskd,cskd->s', posteriors[ancestor], x[..., channel], y[..., channel])
        to_common_ancestor.cache_clear()

    total = probs.sum(axis=-1, keepdims=True)
    if not np.isfinite(probs).all() or (probs < 0).any() or (total > 1 + 1e-10).any():
        raise ValueError('Invalid joint pair event probabilities.')
    np.divide(probs, total, out=probs, where=total > 1)
    return probs


def count_distribution(probabilities):
    """Exact marginal Poisson-binomial count PMF."""
    pmf = np.array([1.])
    for p in probabilities:
        if p > 0:
            next_pmf = np.zeros(len(pmf) + 1)
            next_pmf[:-1] = pmf * (1 - p)
            next_pmf[1:] += pmf * p
            pmf = next_pmf
    return pmf


def joint_distribution(probabilities):
    """Bivariate count PMF with a certified bound on the truncated tail."""
    limit = [np.searchsorted(np.cumsum(count_distribution(p)), 1 - 5e-13) for p in probabilities.T]
    joint = np.zeros((limit[0] + 1, limit[1] + 1))
    joint[0, 0] = 1
    for n, s in probabilities:
        if n + s > 0:
            next_joint = joint * max(0, 1 - n - s)
            next_joint[1:, :] += joint[:-1, :] * n
            next_joint[:, 1:] += joint[:, :-1] * s
            joint = next_joint
    if abs(1 - joint.sum()) > 2e-12:
        raise ValueError('Joint count posterior lost too much probability mass.')
    return joint / joint.sum()


def count_summary(probabilities, truth, ranks):
    joint = joint_distribution(probabilities)
    summary = {
        'p_undefined': joint[0, 0], 'p_infinite': joint[1:, 0].sum(),
        'p_zero': joint[0, 1:].sum(), 'p_finite_positive': joint[1:, 1:].sum(),
    }
    for channel, true, u, pmf in zip(('n', 's'), truth, ranks, (joint.sum(axis=1), joint.sum(axis=0))):
        cdf = np.cumsum(pmf)
        lower, upper = np.searchsorted(cdf, [.025, .975])
        rank = pmf[:true].sum() + (pmf[true] * u if true < len(pmf) else 0)
        summary.update({
            f'{channel}_mean': pmf @ np.arange(len(pmf)),
            f'{channel}_lower': lower, f'{channel}_upper': upper,
            f'{channel}_covered': lower <= true <= upper,
            f'{channel}_pit': rank,
        })
    # Equal tails on the realized ratio when defined. Keep the infinite
    # atom and report the undefined atom separately, without imputation.
    n_values, s_values = np.indices(joint.shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = (n_values / s_values).ravel()[1:]
    weights = joint.ravel()[1:]
    defined_mass = weights.sum()
    if defined_mass <= 1e-12:
        summary.update(ratio_lower=np.nan, ratio_upper=np.nan, ratio_covered=None)
    else:
        order = np.argsort(ratios, kind='stable')
        cdf = np.cumsum(weights[order]) / defined_mass
        lower, upper = ratios[order[np.searchsorted(cdf, [.025, .975])]]
        t_n, t_s = truth
        observed = t_n / t_s if t_s else (np.inf if t_n else np.nan)
        covered = lower <= observed <= upper if not np.isnan(observed) else None
        summary.update(ratio_lower=lower, ratio_upper=upper, ratio_covered=covered)
    return summary
