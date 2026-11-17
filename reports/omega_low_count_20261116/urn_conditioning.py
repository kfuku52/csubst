"""Compare N draw totals and fixed-size subset laws on PEPC simulations."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from csubst import omega

from asr_bootstrap import omega_score
from pilot import EXPECTED, KEY, NULL, OBS, TRUTH
from urn_accuracy import load_search
from urn_parent_opportunity import (load_fitted_events, metrics, opportunity_features,
                                    rate_weighted_states, urn_expected)


def conditioned_counts(pairs, probabilities, observed, model):
    if not (np.isfinite(probabilities).all() and (probabilities >= 0.).all() and (probabilities < 1.).all()):
        raise ValueError('Fitted N probabilities must be in [0, 1) for the odds comparison.')
    totals = {'observed': observed, 'mixed': .5 * (observed + model), 'model': model}
    rules = {'wallenius': (probabilities, omega._calc_wallenius_inclusion_probabilities),
             'fisher_p': (probabilities, omega._calc_fisher_inclusion_probabilities),
             'fisher_odds': (probabilities / (1. - probabilities), omega._calc_fisher_inclusion_probabilities)}
    expected = {}
    for rule, (profiles, calc_inclusion) in rules.items():
        out = {label: np.zeros(len(pairs)) for label in totals}
        for d, weights in enumerate(profiles.transpose(2, 0, 1)):
            capacities = np.count_nonzero(weights, axis=1)

            @lru_cache(maxsize=None)
            def inclusion(branch_id, draw_size):
                return calc_inclusion(weights[branch_id], draw_size)

            for label, masses in totals.items():
                counts = masses[:, d].copy()
                if (counts > capacities + 1e-10).any():
                    raise ValueError('A destination has insufficient eligible sites for its draw total.')
                if counts.any():
                    out[label] += omega._calc_weighted_urn_expected_overlap(
                        pairs, weights, counts, {}, np.float64, calc_inclusion, profile_inclusion=inclusion,
                    )
            inclusion.cache_clear()
        expected.update({rule + '_' + label: values for label, values in out.items()})
    return expected


def verify_fisher_odds():
    """Compare the exact conditional Bernoulli law with subset enumeration."""
    p = np.array([.01, .2, .5, .9])
    odds = p / (1 - p)
    marginal = np.zeros_like(p)
    for size in range(len(p) + 1):
        mass = 0.
        included = np.zeros_like(p)
        for indices in combinations(range(len(p)), size):
            subset = np.isin(np.arange(len(p)), indices)
            probability = np.prod(np.where(subset, p, 1 - p))
            mass += probability
            included[subset] += probability
        estimated = omega._calc_fisher_inclusion_probabilities(odds, size)
        np.testing.assert_allclose(estimated, included / mass, atol=1e-14)
        marginal += mass * estimated
    np.testing.assert_allclose(marginal, p, atol=1e-14)


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
    parents = rate_weighted_states(model, tips)[model.parents]
    parents[model.root] = 0.
    _, s_profile = opportunity_features(parents, aa)
    s = urn_expected(pairs, s_profile, observed[1].sum(axis=1))
    cases = {'codon_model': frame[EXPECTED].to_numpy(),
             'unconditioned_n': np.column_stack([frame[EXPECTED[0]], s])}
    for label, ec_n in conditioned_counts(pairs, predicted[0], observed[0].sum(axis=1), predicted[0].sum(axis=1)).items():
        cases[label] = np.column_stack([ec_n, s])
    a, b = pairs.T
    foreground = frame['foreground_1'].eq(1) & frame['foreground_2'].eq(1)
    scopes = {'all': np.ones(len(frame), dtype=bool), 'sisters': model.parents[a] == model.parents[b],
              'different_parents': model.parents[a] != model.parents[b], 'two_foreground': foreground,
              'other_pairs': ~foreground, 'true_n_positive': frame[TRUTH[0]].gt(0),
              'expected_s_ge_0_5': frame[NULL[1]].ge(.5)}
    obs = frame[OBS].to_numpy()
    finite = np.logical_and.reduce([np.isfinite(omega_score(obs, values)) for values in cases.values()])
    result = [dict(panel='fresh_fits' if archives is None else 'archived', scenario=scenario,
                   seed=int(seed), subset=tag, method=rule, **metrics(frame, values, subset, finite))
              for tag, subset in scopes.items() if subset.any() for rule, values in cases.items()]
    print(f'{rep.name}: {len(frame)} eligible pairs evaluated.', flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('panel', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--archived-fits', type=Path)
    p.add_argument('--jobs', type=int, default=5)
    p.add_argument('--scenarios', nargs='+', choices=['baseline', 'rate_mixture', 'biased'],
                   default=['baseline', 'rate_mixture', 'biased'])
    p.add_argument('--seeds', nargs='+', type=int)
    opts = p.parse_args()
    verify_fisher_odds()
    defaults = [101, 202, 303] if opts.archived_fits is not None else [411, 522, 633]
    tasks = [(opts.panel / 'PEPC' / f'{scenario}-{seed}', opts.archived_fits)
             for scenario in opts.scenarios for seed in (opts.seeds or defaults)]
    with ProcessPoolExecutor(max_workers=opts.jobs) as pool:
        rows = [row for result in pool.map(analyze, tasks) for row in result]
    pd.DataFrame(rows).to_csv(opts.output, sep='\t', index=False, float_format='%.12g')


if __name__ == '__main__':
    main()
