"""Learn parent-to-destination urn weights from background endpoint events on PEPC."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd

from csubst import endpoint_io

from asr_bootstrap import omega_score
from pilot import EXPECTED, KEY, NULL, OBS, TRUTH
from urn_accuracy import load_search
from urn_parent_opportunity import load_fitted_events, metrics, opportunity_features, urn_expected


def parent_evidence(model, tips, amino_index, background):
    site_count = len(next(iter(tips.values())))
    membership = np.eye(amino_index.max() + 1)[amino_index]
    parent = np.zeros((len(model.parents), site_count, len(amino_index)))
    parent_rate = np.zeros_like(parent)
    routes = np.zeros((len(model.parents), membership.shape[1], membership.shape[1]))
    transform = endpoint_io._event_transform({}, ['N'], {'N': membership})
    records = model.iter_blocks(tips, branch_ids=background, joint=False, category_nodes=True, transform=transform)
    for record in records:
        sites = slice(record.start, record.stop)
        i = record.child
        parent[i, sites] = record.node
        parent_rate[i, sites] = np.einsum('r,rsc->sc', model.rates, record.category_node)
        if record.reduced:
            routes[i] += record.reduced['N', 'events'][:, 0].sum(axis=0)
    parent = parent[model.parents]
    parent_rate = parent_rate[model.parents]
    parent[model.root] = 0.
    parent_rate[model.root] = 0.
    return membership, parent, parent_rate, routes


def learned_rates(counts, exposure, shape):
    """Estimate off-diagonal endpoint rate using exposure and a destination-pooled prior."""
    pooled = counts.sum(axis=0) / (exposure.sum() - exposure)
    if (exposure <= 0).any() or (pooled <= 0).any() or not np.isfinite(pooled).all():
        raise ValueError('The training branches do not support every amino acid.')
    rate = (counts + shape) / (exposure[:, None] + shape / pooled)
    np.fill_diagonal(rate, 0.)
    return rate


def estimate_group(pairs, train, channels, row_totals, membership, parent, parent_rate, routes, lengths, extra,
                   only_empirical_routes=False):
    ids, remapped = np.unique(pairs, return_inverse=True)
    pair_ids = remapped.reshape(pairs.shape)
    totals_n, totals_s = row_totals
    train_counts = routes[train].sum(axis=0)

    # Empirical relative site rate and branch effort in endpoint-count units.
    ns_site = channels[0][train].sum(axis=2) + channels[1][train].sum(axis=2)
    site = ns_site.sum(axis=0) + .5
    site /= site.mean()
    effort = ns_site.mean(axis=1)
    empirical_parent = parent[ids] * site[None, :, None]
    exposure_emp = (np.einsum('b,s,bsc->c', effort, site, parent[train]) @ membership)
    rates_emp = learned_rates(train_counts, exposure_emp, .5)
    compatibility_emp, syn_emp = opportunity_features(empirical_parent, membership.argmax(axis=1))
    syn_expected_emp = urn_expected(pair_ids, syn_emp, totals_s[ids])
    parent_aa_emp = empirical_parent @ membership
    n_emp = urn_expected(pair_ids, parent_aa_emp @ rates_emp, totals_n[ids])
    if only_empirical_routes:
        return {'empirical_routes': (n_emp, syn_expected_emp)}
    n_compat = urn_expected(pair_ids, compatibility_emp, totals_n[ids])

    # Retain fitted rate-category exposure for an otherwise matched control.
    exposure_fit = np.einsum('b,bsc->c', lengths[train], parent_rate[train]) @ membership
    rates_fit = learned_rates(train_counts, exposure_fit, .5)
    n_fit = urn_expected(pair_ids, (parent_rate[ids] @ membership) @ rates_fit, totals_n[ids])
    result = {'empirical_routes': (n_emp, syn_expected_emp), 'empirical_compat': (n_compat, syn_expected_emp),
              'fitted_rate_routes': (n_fit, None)}
    if extra:
        # On the same empirical exposure, check sensitivity to pooling rare routes.
        unpooled = learned_rates(train_counts, exposure_emp, 0.)
        n_unpooled = urn_expected(pair_ids, parent_aa_emp @ unpooled, totals_n[ids])
        result['empirical_unpooled'] = (n_unpooled, syn_expected_emp)
    return result


def validate_rates():
    counts = np.array([[0., 8., 0.], [1., 0., 3.], [2., 0., 0.]])
    exposure = np.array([2., 4., 8.])
    unpooled = learned_rates(counts, exposure, 0.)
    np.testing.assert_array_equal(unpooled, counts / exposure[:, None])
    pooled = learned_rates(counts, exposure, .5)
    np.testing.assert_array_equal(np.diag(pooled), 0.)
    assert pooled[0, 2] > 0. and 0. < pooled[0, 1] < unpooled[0, 1]


def crossfit_groups(pairs, background):
    """Exclude the folds of both evaluated incoming branches from training."""
    folds = np.sort(pairs % 4, axis=1)
    for i, j in combinations_with_replacement(range(4), 2):
        keep = (folds[:, 0] == i) & (folds[:, 1] == j)
        if keep.any():
            train = background[~np.isin(background % 4, (i, j))]
            assert not np.intersect1d(train, pairs[keep]).size
            yield keep, train


def analyze(task):
    rep, archives = task
    scenario, seed = rep.name.rsplit('-', 1)
    if archives is None:
        codon, meta = load_search(rep, 'default')
        report, = (rep / 'iqtree-default').glob('*.iqtree')
        tree_path = rep / 'sim/sim_true_asr.treefile'
    else:
        source = archives / rep.name
        codon, meta = load_search(source, 'codon_model')
        report, tree_path = source / 'fit.iqtree', source / 'tree.nwk'
    frame, _, model, tips, aa, observed, predicted = load_fitted_events(rep, codon, meta, report, tree_path)
    pairs = frame[KEY].to_numpy()
    foreground_ids = np.loadtxt(rep / 'sim/sim_foreground_branch.txt', dtype=int, ndmin=1)
    for num in (1, 2):
        np.testing.assert_array_equal(frame[f'foreground_{num}'].eq(1), np.isin(frame[f'branch_id_{num}'], foreground_ids))
    active = np.unique(pairs)
    background = active[~np.isin(active, foreground_ids)]
    membership, parent, parent_rate, routes = parent_evidence(model, tips, aa, background)
    totals = [x.sum(axis=1) for x in observed]
    np.testing.assert_allclose(routes[background].sum(axis=1), totals[0][background], rtol=1e-10, atol=1e-12)
    compatible, syn = opportunity_features(parent_rate, aa)
    fitted_s = urn_expected(pairs, syn, totals[1])
    expected_n_compatible = urn_expected(pairs, compatible, totals[0])
    expected_n_q = urn_expected(pairs, predicted[0], totals[0])
    cases = {'codon_model': frame[EXPECTED].to_numpy(),
             'fitted_compat': np.column_stack((expected_n_compatible, fitted_s)),
             'q_profile_control': np.column_stack((expected_n_q, fitted_s))}
    inputs = observed, totals, membership, parent, parent_rate, routes, model.lengths
    values = estimate_group(pairs, background, *inputs, extra=True)
    for label, (n, s) in values.items():
        cases[label + '_background'] = np.column_stack((n, fitted_s if s is None else s))

    for keep, train in crossfit_groups(pairs, background):
        for label, (n, s) in estimate_group(pairs[keep], train, *inputs, extra=False).items():
            key = label + '_crossfit'
            if key not in cases:
                cases[key] = np.full((len(pairs), 2), np.nan)
            cases[key][keep] = np.column_stack((n, fitted_s[keep] if s is None else s))
    assert all(np.isfinite(x).all() for x in cases.values())

    a, b = pairs.T
    foreground = frame['foreground_1'].eq(1) & frame['foreground_2'].eq(1)
    scopes = {'all': np.ones(len(frame), dtype=bool), 'sisters': model.parents[a] == model.parents[b],
              'different_parents': model.parents[a] != model.parents[b], 'two_foreground': foreground,
              'other_pairs': ~foreground, 'true_n_positive': frame[TRUTH[0]].gt(0),
              'expected_s_ge_0_5': frame[NULL[1]].ge(.5)}
    obs = frame[OBS].to_numpy()
    finite = np.logical_and.reduce([np.isfinite(omega_score(obs, x)) for x in cases.values()])
    print(f'{rep.name}: {len(frame)} pairs with {len(background)} background training branches.', flush=True)
    return [dict(panel='fresh_fits' if archives is None else 'archived', scenario=scenario,
                 seed=int(seed), subset=tag, method=label, **metrics(frame, values, subset, finite))
            for tag, subset in scopes.items() if subset.any() for label, values in cases.items()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('panel', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--archived-fits', type=Path)
    p.add_argument('--jobs', type=int, default=5)
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--scenarios', nargs='+', choices=['baseline', 'rate_mixture', 'biased'],
                   default=['baseline', 'rate_mixture', 'biased'])
    args = p.parse_args()
    validate_rates()
    defaults = (101, 202, 303) if args.archived_fits else (411, 522, 633)
    tasks = [(args.panel / 'PEPC' / f'{scenario}-{seed}', args.archived_fits)
             for scenario in args.scenarios for seed in (args.seeds or defaults)]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = [row for batch in pool.map(analyze, tasks) for row in batch]
    pd.DataFrame(rows).to_csv(args.output, sep='\t', index=False, float_format='%.12g')


if __name__ == '__main__':
    main()
