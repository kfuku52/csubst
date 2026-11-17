"""Test parent-state × site-rate urn weights on fixed PEPC endpoint fits."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from csubst import endpoint, endpoint_io, ete, omega, tree

from asr_bootstrap import edge_features, omega_score, pair_counts, projection
from pilot import EXPECTED, KEY, NULL, OBS, TRUTH
from replay_posterior import fitted_generator, load_tips
from urn_accuracy import GENERATING, load_search


def rate_weighted_states(model, tips, tip_missing=None):
    size = next(iter(tips.values())).shape[0]
    result = np.zeros((len(model.parents), size, len(model.pi)))
    records = model.iter_blocks(tips, joint=False, category_nodes=True, tip_missing=tip_missing)
    for row in records:
        result[row.child, row.start:row.stop] = np.einsum('r,rsc->sc', model.rates, row.category_node)
    return result


def normalize(values, scale):
    mass = values.sum(axis=1)
    with np.errstate(divide='ignore'):
        scale += np.log(mass)
    values /= np.where(mass > 0., mass, 1.)[:, None]


def outside_parent_profiles(model, tips):
    """Compute E[rate × one_hot(parent)] with the focal child subtree omitted."""
    size = next(iter(tips.values())).shape[0]
    shape = (len(model.parents), size, len(model.pi))
    estimate = np.zeros(shape)
    accumulated = np.full(shape[:2], -np.inf)
    for category, (rate, prior) in enumerate(zip(model.rates, model.weights)):
        transitions = {child: model.transition(child, category) for child in model.order[1:]}
        info = np.ones(shape)
        up = np.empty(shape)
        log_up = np.zeros(shape[:2])
        for node in reversed(model.order):
            if node in tips:
                info[node] = tips[node]
            for child in model.children[node]:
                info[node] *= up[child]
                log_up[node] += log_up[child]
                normalize(info[node], log_up[node])
            if node != model.root:
                up[node] = info[node] @ transitions[node].T
        # Each cavity is the outside likelihood at the parent times the
        # pruning messages of the focal child's siblings, with absolute scale.
        outside = np.zeros(shape)
        outside[model.root] = model.pi
        log_out = np.zeros(shape[:2])
        for parent in model.order:
            children = model.children[parent]
            for child in children:
                cavity = outside[parent].copy()
                scale = log_out[parent].copy()
                for sibling in children:
                    if sibling != child:
                        cavity *= up[sibling]
                        scale += log_up[sibling]
                        normalize(cavity, scale)
                outside[child] = cavity @ transitions[child]
                log_out[child] = scale
                new_log = scale + np.log(prior)
                total = np.logaddexp(accumulated[child], new_log)
                supported = np.isfinite(total)
                estimate[child, supported] = (
                    estimate[child, supported] * np.exp(accumulated[child, supported] - total[supported])[:, None]
                    + rate * cavity[supported] * np.exp(new_log[supported] - total[supported])[:, None]
                )
                accumulated[child] = total
    if not np.isfinite(accumulated[model.order[1:]]).all():
        raise ValueError('No fitted rate category supports a subtree-excluded alignment.')
    return estimate


def verify_exclusion(model, tips, outside):
    def subtree_leaves(branch):
        if branch in tips:
            return [branch]
        return [tip for child in model.children[branch] for tip in subtree_leaves(child)]

    largest = max(model.order[1:], key=lambda branch: len(subtree_leaves(branch)))
    terminal = max(tips, key=lambda branch: model.lengths[branch])
    internal = max((i for i in model.order[1:] if i not in tips), key=lambda branch: model.lengths[branch])
    for branch in {largest, terminal, internal}:
        short_tips = {i: values[::31] for i, values in tips.items()}
        missing = {i: np.ones(len(short_tips[i]), dtype=bool) for i in subtree_leaves(branch)}
        check = rate_weighted_states(model, short_tips, tip_missing=missing)
        np.testing.assert_allclose(outside[branch, ::31], check[model.parents[branch]], rtol=2e-9, atol=1e-11)


def opportunity_features(parent, aa):
    distinct = aa[:, None] != np.arange(aa.max() + 1)
    synonymous = (aa[:, None] == aa) & ~np.eye(len(aa), dtype=bool)
    return parent @ distinct.astype(float), parent @ synonymous.astype(float)


def urn_expected(pairs, profile, totals):
    expected = np.zeros(len(pairs))
    for weights, masses in zip(profile.transpose(2, 0, 1), totals.T):
        if not masses.any():
            continue
        if (masses > np.count_nonzero(weights, axis=1) + 1e-10).any():
            raise ValueError('A destination has insufficient eligible sites for its inferred branch total.')
        expected += omega._calc_wallenius_expected_overlap(pairs, weights, masses.copy(), {}, np.float64)
    return expected


def metrics(frame, expected, subset, common_omega=None):
    obs, null, simulated, realized = [frame.loc[subset, cols].to_numpy()
                                      for cols in (OBS, NULL, GENERATING, TRUTH)]
    values = expected[subset]
    ratio = omega_score(obs, values)
    with np.errstate(divide='ignore', invalid='ignore'):
        target = (simulated[:, 0] / null[:, 0]) / (simulated[:, 1] / null[:, 1])
    finite = np.isfinite(ratio) & np.isfinite(target)
    if common_omega is not None:
        finite &= common_omega[subset]
    error = np.log1p(ratio[finite]) - np.log1p(target[finite])
    result = dict(n_pairs=len(obs), finite_omega=int(finite.sum()), omega_sse=np.square(error).sum())
    for label, counts in [('observed', obs), ('q_null', null), ('q_sim', simulated), ('realized', realized),
                          ('expected', values), ('sse', np.square(values - null))]:
        result.update({label + '_' + kind: total for kind, total in zip('ns', counts.sum(axis=0))})
    return result


def load_fitted_events(rep, codon, meta, report, tree_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode(str(tree_path), format=1))
    parents, _ = endpoint_io._tree_arrays({'tree': tr}, structural=False)
    q, pi, codons, aa = fitted_generator(rep, report, meta)
    model = endpoint.EndpointModel(parents, meta['branch_lengths'], q, pi, meta['rates'], meta['weights'])
    tips = load_tips(rep, codons, tr)
    transforms, indices, _ = projection(codons)
    projected, predicted = edge_features(model, tips, transforms, indices)
    observed = np.column_stack([pair_counts(v, codon[KEY].to_numpy()) for v in projected])
    np.testing.assert_allclose(observed, codon[OBS], rtol=1e-9, atol=6e-12)
    keep = codon[EXPECTED].notna().all(axis=1)
    frame = codon.loc[keep].merge(pd.read_csv(rep / 'sim/sim_expected_omega.tsv', sep='\t'),
                                 on=KEY, how='left', validate='one_to_one')
    assert frame[NULL + GENERATING].notna().all().all()
    pairs = frame[KEY].to_numpy()
    names = {int(ete.get_prop(node, 'numerical_label')): node.name for node in tr.traverse()}
    for i in (1, 2):
        assert frame[f'branch_id_{i}'].map(names).eq(frame[f'branch_name_{i}']).all()
    np.testing.assert_array_equal(pairs, codon.loc[keep, KEY])
    np.testing.assert_allclose(np.column_stack([pair_counts(x, pairs) for x in predicted]), frame[EXPECTED],
                               rtol=1e-9, atol=6e-12)
    return frame, keep, model, tips, aa, projected, predicted


def analyze(case):
    rep, runs, extra = case
    scenario, seed = rep.name.rsplit('-', 1)
    source = runs / rep.name
    codon, meta = load_search(source, 'codon_model')
    urn, urn_model = load_search(source, 'urn-each')
    hybrid, hybrid_model = load_search(extra / rep.name, 'urn-file_each')
    for used in (urn_model, hybrid_model):
        for key in ('rates', 'weights', 'branch_lengths'):
            assert used[key] == meta[key]
        assert used['parameter_source']['generator_sha256'] == meta['parameter_source']['generator_sha256']
    np.testing.assert_array_equal(codon[KEY], urn[KEY])
    np.testing.assert_array_equal(codon[KEY], hybrid[KEY])
    np.testing.assert_allclose(urn[OBS], codon[OBS], rtol=1e-9, atol=6e-12)
    np.testing.assert_allclose(hybrid[OBS], codon[OBS], rtol=1e-9, atol=6e-12)
    frame, keep, model, tips, aa, projected, predicted = load_fitted_events(
        rep, codon, meta, source / 'fit.iqtree', source / 'tree.nwk',
    )
    pairs = frame[KEY].to_numpy()
    a, b = pairs.T
    parents = model.parents
    total = [channel.sum(axis=1) for channel in projected]
    default = urn.loc[keep, EXPECTED].to_numpy()
    methods = {'codon_model': frame[EXPECTED].to_numpy(), 'urn_each': default,
               'urn_file_each': hybrid.loc[keep, EXPECTED].to_numpy()}
    full = rate_weighted_states(model, tips)[parents]
    full[model.root] = 0.
    excluded = outside_parent_profiles(model, tips)
    verify_exclusion(model, tips, excluded)
    for tag, parent in [('full', full), ('outside', excluded)]:
        profiles = opportunity_features(parent, aa)
        n, s = [urn_expected(pairs, w, m) for w, m in zip(profiles, total)]
        methods['n_' + tag] = np.column_stack([n, default[:, 1]])
        methods['s_' + tag] = np.column_stack([default[:, 0], s])
        methods['both_' + tag] = np.column_stack([n, s])
        if tag == 'full':
            # Change only one N component at a time while retaining the S correction.
            model_total_n = urn_expected(pairs, profiles[0], predicted[0].sum(axis=1))
            model_profile_n = urn_expected(pairs, predicted[0], total[0])
            model_profile_s = urn_expected(pairs, predicted[1], total[1])
            methods['model_totals_n'] = np.column_stack([model_total_n, s])
            methods['model_profile_n'] = np.column_stack([model_profile_n, s])
            methods['model_profile_both'] = np.column_stack([model_profile_n, model_profile_s])
    foreground = frame['foreground_1'].eq(1) & frame['foreground_2'].eq(1)
    scopes = {'all': np.ones(len(frame), dtype=bool), 'sisters': parents[a] == parents[b],
              'different_parents': parents[a] != parents[b], 'two_foreground': foreground,
              'other_pairs': ~foreground, 'true_n_positive': frame[TRUTH[0]].gt(0),
              'expected_s_ge_0_5': frame[NULL[1]].ge(.5)}
    print(f'{rep.name}: {len(frame)} pairs and explicit clade-exclusion checks passed.', flush=True)
    return [dict(scenario=scenario, seed=int(seed), subset=tag, method=method, **metrics(frame, result, subset))
            for tag, subset in scopes.items() if subset.any() for method, result in methods.items()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('panel', type=Path)
    p.add_argument('original_runs', type=Path)
    p.add_argument('additional_runs', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--jobs', type=int, default=3)
    p.add_argument('--scenarios', nargs='+', choices=['baseline', 'biased', 'rate_mixture'],
                   default=['baseline', 'biased', 'rate_mixture'])
    p.add_argument('--seeds', nargs='+', type=int, default=[101, 202, 303])
    options = p.parse_args()
    jobs = [(options.panel / 'PEPC' / f'{scenario}-{seed}', options.original_runs, options.additional_runs)
            for scenario in options.scenarios for seed in options.seeds]
    with ProcessPoolExecutor(max_workers=options.jobs) as pool:
        rows = [row for case in pool.map(analyze, jobs) for row in case]
    pd.DataFrame(rows).to_csv(options.output, sep='\t', index=False, float_format='%.12g')


if __name__ == '__main__':
    main()
