"""Destination profiles for all-branch and cross-fitted codon urns."""

from dataclasses import dataclass

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
    if getattr(tensor, 'event_totals', None) is not None:
        return tensor.branch_site, tensor.event_totals
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


def _allocate_branch_totals(profile, observed, channel):
    """Distribute each branch's observed mass over all its possible destinations."""
    weights = profile.reshape(len(observed), -1)
    norm = weights.sum(axis=1)
    if ((observed > 0.) & (norm == 0.)).any():
        raise ValueError(f'Cannot allocate positive branch {channel} mass to compatible destinations.')
    scale = np.divide(observed, norm, out=np.zeros_like(observed), where=norm > 0.)
    return (weights * scale[:, None]).reshape(profile.shape)


def _overlaps(combos, inclusions):
    # Dense fold pairs share most of their branch rows. Reuse the dot
    # products without gathering a pair-by-site array for each destination.
    n = len(inclusions)
    if combos.shape[1] == 2 and n <= 512 and len(combos) * 8 >= n * n:
        left, right = combos.T
        return (inclusions @ inclusions.T)[left, right]
    return omega._calc_cb_site_overlap(combos, inclusions, np.float64)


def _add_group_overlaps(result, combos, profile, totals, g):
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
            result['any2spe'] += _overlaps(combos, included)
    if use_any:
        result['any2any'] += _overlaps(combos, all_dest)


@dataclass
class _BranchData:
    parents: np.ndarray
    in_scope: np.ndarray
    groups: list
    eligible: np.ndarray
    aa_parent: np.ndarray
    effort: np.ndarray
    ns_counts: np.ndarray
    route_totals: np.ndarray
    syn_totals: np.ndarray


def _branch_data(g, nonsyn, syn):
    eligible = np.asarray(nonsyn.eligible)
    nodes = tuple(g['tree'].traverse())
    parents = np.zeros(len(nodes), dtype=int)
    in_scope = np.zeros(len(nodes), dtype=bool)
    for node in nodes:
        if not ete.is_root(node):
            i = int(ete.get_prop(node, 'numerical_label'))
            parents[i] = int(ete.get_prop(node.up, 'numerical_label'))
            in_scope[i] = not ete.is_root(node.up)
    n_sites, route_totals = _event_totals(nonsyn, routes=True)
    s_sites, syn_totals = _event_totals(syn, routes=False)
    ns_counts = n_sites + s_sites
    effort = np.divide(ns_counts.sum(axis=1), eligible.sum(axis=1), out=np.zeros(len(eligible)),
                       where=eligible.any(axis=1))
    groups = [np.asarray(g['synonymous_indices'][a], dtype=int) for a in g['amino_acid_orders']]
    state = g['state_cdn']
    aa_membership = np.zeros((state.shape[2], len(groups)))
    for i, group in enumerate(groups):
        aa_membership[group, i] = 1.
    return _BranchData(parents, in_scope, groups, eligible, (state @ aa_membership)[parents], effort,
                       ns_counts, route_totals, syn_totals)


@dataclass
class _Rates:
    n_rate: np.ndarray
    s_rate: np.ndarray
    routes: np.ndarray
    count: np.ndarray
    exposure: np.ndarray
    uniform: float
    s_draw_totals: np.ndarray | None = None


def _fit(data, train):
    mass = data.ns_counts[train]
    eligible = data.eligible[train]
    n_rate, uniform = fit_site_rates(mass, eligible)
    exposure = np.einsum('b,bs,bsa->a', data.effort[train], n_rate * eligible, data.aa_parent[train], optimize=True)
    count = data.route_totals[train].sum(axis=0)
    routes = _route_rates(count, exposure)
    # Keep the S estimator used in the ECN study, including its fixed prior.
    support = eligible.sum(axis=0)
    s_rate = np.divide(mass.sum(axis=0) * len(train), support,
                       out=np.zeros_like(n_rate), where=support > 0) + 0.5
    return _Rates(n_rate, s_rate / s_rate.mean(), routes, count, exposure, float(uniform))


def _pool_s_destinations(state, data, train, site_rate):
    """Allocate each branch's observed S total using pooled counts per opportunity."""
    observed = data.syn_totals[train]
    total = observed.sum(axis=(1, 2))
    draws = np.zeros_like(data.syn_totals)
    if not total.any():
        return draws
    # First reduce sites to codon exposures, without materializing a copy of
    # the branch-by-site-by-codon state tensor for all training branches.
    codon_exposure = np.empty((len(train), state.shape[2]))
    for i, b in enumerate(train):
        codon_exposure[i] = (site_rate * data.eligible[b]) @ state[data.parents[b]]
    opportunity = np.zeros_like(observed)
    for i, group in enumerate(data.groups):
        if len(group) > 1:
            opportunity[:, i, :len(group)] = codon_exposure[:, group] @ (1. - np.eye(len(group)))
    effort = total / data.eligible[train].sum(axis=1)
    exposure = np.einsum('b,bgd->gd', effort, opportunity)
    mass = observed.sum(axis=0)
    if ((mass > 0.) & (exposure == 0.)).any():
        raise ValueError('Positive synonymous destination mass has no eligible parental opportunity.')
    rate = np.divide(mass, exposure, out=np.zeros_like(mass), where=exposure > 0.)
    draws[train] = _allocate_branch_totals(opportunity * rate, total, 'S')
    return draws


def fit_all_branches(g, nonsyn, syn):
    data = _branch_data(g, nonsyn, syn)
    train = np.flatnonzero(data.in_scope & data.eligible.any(axis=1))
    g['_asrv_training_ids'] = train
    rates = _fit(data, train) if train.size else None
    if rates is not None:
        rates.s_draw_totals = _pool_s_destinations(g['state_cdn'], data, train, rates.s_rate)
    g['_empirical_urn_uniform_weight'] = rates.uniform if rates is not None else None
    return data, rates


def _score(g, combos, select, data, rates, stats, out):
    ids, inverse = np.unique(combos[select], return_inverse=True)
    local = inverse.reshape(-1, combos.shape[1])
    n_counts = {stat: np.zeros(len(local)) for stat in stats}
    s_counts = {stat: np.zeros(len(local)) for stat in stats}
    n_rate = rates.n_rate * data.eligible[ids]
    n_weights = (data.aa_parent[ids] @ rates.routes) * n_rate[:, :, None]
    if g.get('asrv') == 'empirical_all':
        observed_n = data.route_totals[ids].sum(axis=(1, 2))
        n_total = _allocate_branch_totals(n_weights.sum(axis=1), observed_n, 'N')
    else:
        n_total = _pool_n_totals(rates.count, rates.exposure, rates.routes, n_weights, data.effort[ids])
    _add_group_overlaps(n_counts, local, n_weights, n_total, g)
    state = g['state_cdn']
    s_rate = rates.s_rate * data.eligible[ids]
    s_draws = data.syn_totals if rates.s_draw_totals is None else rates.s_draw_totals
    for i, group in enumerate(data.groups):
        if len(group) < 2:
            continue
        parent = state[data.parents[ids, None, None], np.arange(state.shape[1])[None, :, None], group]
        s_weights = (parent @ (1. - np.eye(len(group)))) * s_rate[:, :, None]
        _add_group_overlaps(s_counts, local, s_weights, s_draws[ids, i, :len(group)], g)
    for kind, values in (('N', n_counts), ('S', s_counts)):
        for stat, expected in values.items():
            out['EC' + kind + stat][select] = expected


def expected_counts(g, pairs, nonsyn, syn, stats):
    """Estimate EC with one pooled fit, or the original held-out pair folds."""
    combos = pairs  # Keep the original pairs= keyword for pairwise callers.
    stats = tuple(stats)
    pooled = g.get('asrv') == 'empirical_all'
    if combos.ndim != 2 or combos.shape[1] < 2 or (not pooled and combos.shape[1] != 2):
        raise ValueError('Empirical urn expects branch combinations (pairs only for cross-fitting).')
    if not set(stats).issubset(SUPPORTED_STATS):
        raise ValueError('Empirical urn expects destination-based statistics.')
    out = {'EC' + kind + stat: np.full(len(combos), np.nan) for kind in ('N', 'S') for stat in stats}
    if not len(combos):
        return out
    if pooled:
        prepared = g.get('_empirical_urn_fit')
        data, rates = prepared if prepared is not None else fit_all_branches(g, nonsyn, syn)
        select = data.in_scope[combos].all(axis=1)
        if rates is not None and select.any():
            _score(g, combos, select, data, rates, stats, out)
        return out

    data = _branch_data(g, nonsyn, syn)
    valid = data.in_scope[combos].all(axis=1)
    training_ids = np.flatnonzero(data.in_scope & data.eligible.any(axis=1))
    g['_asrv_training_ids'] = training_ids
    folds = np.sort(combos % FOLD_COUNT, axis=1)
    diagnostics = []
    for a, b in combinations_with_replacement(range(FOLD_COUNT), 2):
        select = valid & (folds[:, 0] == a) & (folds[:, 1] == b)
        if not select.any():
            continue
        train = training_ids[~np.isin(training_ids % FOLD_COUNT, (a, b))]
        uniform = None
        if train.size:
            rates = _fit(data, train)
            _score(g, combos, select, data, rates, stats, out)
            uniform = rates.uniform
        diagnostics.append(dict(held_out_folds=[a, b], training_count=len(train), uniform_weight=uniform))
    g['_empirical_urn_folds'] = diagnostics
    return out
