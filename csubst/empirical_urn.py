"""Cross-fitted destination profiles for the pairwise codon urn."""

from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import minimize_scalar

from csubst import ete, omega


FOLD_COUNT = 4
ROUTE_PRIOR_COUNT = 0.5
SUPPORTED_STATS = frozenset(('any2any', 'any2spe'))


def _row_probabilities(mass, uniform):
    total = mass.sum(axis=1, keepdims=True)
    return np.divide(mass, total, out=uniform.copy(), where=total > 0.)


def fit_site_rates(counts, eligible):
    """Choose a mixture with the uniform profile by leave-one-branch-out scoring."""
    site_counts = counts.sum(axis=0)
    support = eligible.sum(axis=0)
    per_site = np.divide(site_counts, support, out=np.zeros_like(site_counts), where=support > 0)
    if len(counts) < 2 or not site_counts.any():
        return np.ones(counts.shape[1]), 1.
    # Correct for missing branch-sites in both the training profile and the
    # validation scores. With complete data all denominators are constant.
    other_support = support - eligible
    other_mass = np.maximum(site_counts - counts, 0.)
    reference = np.divide(other_mass, other_support, out=np.zeros_like(counts), where=other_support > 0)
    reference *= eligible
    uniform = eligible / eligible.sum(axis=1, keepdims=True)
    reference = _row_probabilities(reference, uniform)
    used = counts > 0.
    validation_counts = counts[used]
    baseline, empirical = uniform[used], reference[used]

    def loss(weight):
        p = (1. - weight) * empirical + weight * baseline
        if (p <= 0.).any():
            return np.inf
        return -float(validation_counts @ np.log(p))

    result = minimize_scalar(loss, bounds=(0., 1.), method='bounded', options={'xatol': 1e-7})
    if not result.success:
        raise ValueError('Could not fit empirical urn site weights.')
    weight = min((0., result.x, 1.), key=loss)
    # Extrapolate at sites with no observed training branches using the
    # mean over sites supported by training data.
    per_site[support == 0] = per_site[support > 0].mean()
    return (1. - weight) * (per_site / per_site.mean()) + weight, weight


def _event_totals(tensor, routes):
    """Reduce one sparse branch at a time, without a dense event tensor."""
    nb, ns, ng, na, nd = tensor.shape
    site = np.zeros((nb, ns))
    shape = (na, nd) if routes else (ng, nd)
    totals = np.zeros((nb, *shape))
    for b, s, group, source, dest, weight in tensor.iter_branch_coordinates():
        site[b] = np.bincount(s, weights=weight, minlength=ns)
        first = source if routes else group
        totals[b] = np.bincount(first * nd + dest, weights=weight, minlength=tensor.num_state_to * shape[0]).reshape(shape)
    return site, totals


def _route_rates(count, exposure):
    other = np.maximum(exposure.sum() - exposure, 0.)
    pooled = np.divide(count.sum(axis=0), other, out=np.zeros_like(exposure), where=other > 0.)
    rates = ((count + ROUTE_PRIOR_COUNT) * pooled
             / (exposure[:, None] * pooled + ROUTE_PRIOR_COUNT))
    np.fill_diagonal(rates, 0.)
    return rates


def _pool_n_totals(count, exposure, routes, eval_profile, eval_effort):
    training_flux = (exposure[:, None] * routes).sum()
    scale = count.sum() / training_flux if training_flux > 0. else 0.
    return eval_profile.sum(axis=1) * (scale * eval_effort)[:, None]


def _add_group_overlaps(result, pairs, profile, totals, g):
    use_any = 'any2any' in result
    use_spe = 'any2spe' in result
    all_dest = np.zeros(profile.shape[:2]) if use_any else None
    for weights, draws in zip(np.moveaxis(profile, -1, 0), totals.T):
        if not draws.any():
            continue
        included = omega.wallenius_inclusion_profiles(weights, draws, g)
        if use_any:
            all_dest += included
        if use_spe:
            result['any2spe'] += omega._calc_cb_site_overlap(pairs, included, np.float64)
    if use_any:
        result['any2any'] += omega._calc_cb_site_overlap(pairs, all_dest, np.float64)


def _paired_counts(g, pairs, ids, train, parents, aa_parent, groups, eligible, ns_counts, route_totals, syn_totals, stats):
    state = g['state_cdn']
    rate, uniform = fit_site_rates(ns_counts[train], eligible[train])
    effort = np.divide(ns_counts.sum(axis=1), eligible.sum(axis=1), out=np.zeros(len(eligible)),
                       where=eligible.any(axis=1))
    exposure = np.einsum('b,bs,bsa->a', effort[train], rate * eligible[train], aa_parent[train], optimize=True)
    count = route_totals[train].sum(axis=0)
    routes = _route_rates(count, exposure)
    n_rate = rate * eligible[ids]
    n_weights = (aa_parent[ids] @ routes) * n_rate[:, :, None]
    n_total = _pool_n_totals(count, exposure, routes, n_weights, effort[ids])

    counts = {kind: {stat: np.zeros(len(pairs)) for stat in stats} for kind in ('N', 'S')}
    _add_group_overlaps(counts['N'], pairs, n_weights, n_total, g)
    # Keep the S estimator used in the ECN study, including its fixed prior.
    support = eligible[train].sum(axis=0)
    s_rate = np.divide(ns_counts[train].sum(axis=0) * len(train), support,
                       out=np.zeros(state.shape[1]), where=support > 0) + 0.5
    s_rate = s_rate / s_rate.mean() * eligible[ids]
    for i, group in enumerate(groups):
        if len(group) < 2:
            continue
        parent = state[parents[ids, None, None], np.arange(state.shape[1])[None, :, None], group]
        s_weights = (parent @ (1. - np.eye(len(group)))) * s_rate[:, :, None]
        _add_group_overlaps(counts['S'], pairs, s_weights, syn_totals[ids, i, :len(group)], g)
    return counts, uniform


def expected_counts(g, pairs, nonsyn, syn, stats):
    """Estimate EC using all eligible branches outside the pair's two folds."""
    stats = tuple(stats)
    shape = len(pairs)
    out = {'EC' + kind + stat: np.full(shape, np.nan) for kind in ('N', 'S') for stat in stats}
    if not shape:
        return out
    if pairs.shape[1] != 2 or not set(stats).issubset(SUPPORTED_STATS):
        raise ValueError('Empirical urn expects pairs and destination-based statistics.')
    eligible = np.asarray(nonsyn.eligible)
    nodes = tuple(g['tree'].traverse())
    parents = np.zeros(len(nodes), dtype=int)
    in_scope = np.zeros(len(nodes), dtype=bool)
    for node in nodes:
        if not ete.is_root(node):
            i = int(ete.get_prop(node, 'numerical_label'))
            parents[i] = int(ete.get_prop(node.up, 'numerical_label'))
            in_scope[i] = not ete.is_root(node.up)
    valid = in_scope[pairs].all(axis=1)
    training_ids = np.flatnonzero(in_scope & eligible.any(axis=1))
    g['_asrv_training_ids'] = training_ids
    if not valid.any():
        return out

    n_sites, route_totals = _event_totals(nonsyn, routes=True)
    s_sites, syn_totals = _event_totals(syn, routes=False)
    ns_counts = n_sites + s_sites
    groups = [np.asarray(g['synonymous_indices'][a], dtype=int) for a in g['amino_acid_orders']]
    state = g['state_cdn']
    aa_membership = np.zeros((state.shape[2], len(groups)))
    for i, group in enumerate(groups):
        aa_membership[group, i] = 1.
    aa_parent = (state @ aa_membership)[parents]

    folds = np.sort(pairs % FOLD_COUNT, axis=1)
    diagnostics = []
    for a, b in combinations_with_replacement(range(FOLD_COUNT), 2):
        select = valid & (folds[:, 0] == a) & (folds[:, 1] == b)
        if not select.any():
            continue
        train = training_ids[~np.isin(training_ids % FOLD_COUNT, (a, b))]
        if not train.size:
            diagnostics.append(dict(held_out_folds=[a, b], training_count=0, uniform_weight=None))
            continue
        ids, inverse = np.unique(pairs[select], return_inverse=True)
        counts, uniform = _paired_counts(g, inverse.reshape(-1, 2), ids, train, parents, aa_parent, groups,
                                         eligible, ns_counts, route_totals, syn_totals, stats)
        for kind, values in counts.items():
            for stat, expected in values.items():
                out['EC' + kind + stat][select] = expected
        diagnostics.append(dict(held_out_folds=[a, b], training_count=len(train), uniform_weight=float(uniform)))
    g['_empirical_urn_folds'] = diagnostics
    return out
