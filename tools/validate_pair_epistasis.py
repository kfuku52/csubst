#!/usr/bin/env python3
"""Oracle recovery of convergence expectations under a two-site Potts CTMC.

Not a production omega P-value method, inferred-landscape validation, or proof
of performance on real proteins. The independent Gillespie simulator uses local
fitness differences, while the exact oracle enumerates full sequence energies.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csubst import pair_epistasis as pair  # noqa: E402


def simulate_endpoints(model, initial, duration, draws, rng):
    """Vectorized Gillespie, independently coded without the oracle generator."""
    k = model.alphabet_size
    targets = np.zeros((k * k, 2 * (k - 1)), dtype=int)
    rates = np.zeros_like(targets, dtype=float)
    for a in range(k):
        for b in range(k):
            row = a * k + b
            column = 0
            for new in range(k):
                if new != a:
                    delta = model.fields[0, new] - model.fields[0, a] + model.coupling[new, b] - model.coupling[a, b]
                    targets[row, column] = new * k + b
                    rates[row, column] = model.mutation_rates[0] / (k - 1) * np.exp(delta / 2.)
                    column += 1
            for new in range(k):
                if new != b:
                    delta = model.fields[1, new] - model.fields[1, b] + model.coupling[a, new] - model.coupling[a, b]
                    targets[row, column] = a * k + new
                    rates[row, column] = model.mutation_rates[1] / (k - 1) * np.exp(delta / 2.)
                    column += 1
    states = np.full(draws, int(initial[0] * k + initial[1]), dtype=int)
    remaining = np.full(draws, duration, dtype=float)
    active = np.arange(draws)
    totals = rates.sum(axis=1)
    cumulative = np.cumsum(rates / totals[:, None], axis=1)
    cumulative[:, -1] = 1.
    while len(active):
        remaining[active] -= rng.exponential(1. / totals[states[active]])
        active = active[remaining[active] > 0]
        if not len(active):
            break
        event = np.sum(rng.random(len(active))[:, None] > cumulative[states[active]], axis=1)
        states[active] = targets[states[active], event]
    return np.column_stack((states // k, states % k))


def sample_categories(model, parents, lengths, draws, rng):
    endpoints = np.stack([simulate_endpoints(model, parent, length, draws, rng)
                          for parent, length in zip(parents, lengths)], axis=1)
    return pair.endpoint_categories(parents, endpoints)


def rejection_summary(reference, actual, observations, level=.05):
    reject = pair.upper_tail(reference) <= level
    successes = int(reject[observations].sum())
    interval = binomtest(successes, len(observations)).proportion_ci()
    return {'exact_rejection_probability': float(actual[reject].sum()),
            'mc_rejection_fraction': successes / len(observations),
            'mc_interval95': [float(interval.low), float(interval.high)]}


def run(replicates=5000, num_pairs=20, seed=20260911):
    scenarios = []
    fields = np.array([[0., .15, -.1], [-.1, .1, .2]])
    # Fixed grid established before examining outcomes; all cells are retained.
    for strength in (0., .8, 1.6):
        for duration in (.1, .5, 1.5):
            for name, parents in [('same_stable', [[0, 0], [0, 0]]),
                                  ('same_mismatched', [[0, 1], [0, 1]]),
                                  ('different_parents', [[0, 0], [1, 0]])]:
                scenarios.append((strength, duration, name, np.array(parents)))
    results = []
    for index, (strength, duration, name, parents) in enumerate(scenarios):
        model = pair.PairLandscape(fields, strength * np.eye(3), np.array([.6, 1.]))
        independent = pair.matched_independent_landscape(model)
        lengths = np.array([duration, duration * 1.3])
        true_pmfs = pair.exact_category_pmfs(model, parents, lengths)
        independent_pmfs = pair.exact_category_pmfs(independent, parents, lengths)
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, 0]))
        draws = replicates * num_pairs
        counts = sample_categories(model, parents, lengths, draws, rng)
        categories = {}
        for i, category in enumerate(pair.CATEGORIES):
            exact = float(true_pmfs[category] @ np.arange(3))
            baseline = float(independent_pmfs[category] @ np.arange(3))
            observed = float(counts[:, i].mean())
            se = float(counts[:, i].std(ddof=1) / np.sqrt(draws))
            categories[category] = {'oracle_expectation': exact, 'independent_expectation': baseline,
                                    'mc_mean': observed, 'mc_standard_error': se,
                                    'oracle_standardized_error': (observed - exact) / se if se else 0.,
                                    'independent_bias': baseline - exact}
        statistic = pair.CATEGORIES.index('any2spe')
        totals = counts[:, statistic].reshape(replicates, num_pairs).sum(axis=1)
        truth = pair.repeated_pair_pmf(true_pmfs['any2spe'], num_pairs)
        baseline = pair.repeated_pair_pmf(independent_pmfs['any2spe'], num_pairs)
        # Both branches experience a common field shift toward state 2 at site 0.
        # Its observed outcomes NEVER enter construction of the null landscape.
        alternative_fields = fields.copy()
        alternative_fields[0, 2] += 2.
        alternative = pair.PairLandscape(alternative_fields, model.coupling, model.mutation_rates)
        alt_pmf = pair.repeated_pair_pmf(pair.exact_category_pmfs(alternative, parents, lengths)['any2spe'], num_pairs)
        alt_rng = np.random.default_rng(np.random.SeedSequence([seed, index, 1]))
        alt_counts = sample_categories(alternative, parents, lengths, draws, alt_rng)
        alt_totals = alt_counts[:, statistic].reshape(replicates, num_pairs).sum(axis=1)
        result = {'coupling': strength, 'duration': duration, 'background': name,
                  'parents': parents.tolist(), 'branch_lengths': lengths.tolist(), 'categories': categories,
                  'oracle_null': rejection_summary(truth, truth, totals),
                  'independent_null': rejection_summary(baseline, truth, totals),
                  'oracle_alternative': rejection_summary(truth, alt_pmf, alt_totals),
                  'independent_alternative': rejection_summary(baseline, alt_pmf, alt_totals)}
        results.append(result)
        print('Completed {}/{}: J={} t={} {}'.format(index + 1, len(scenarios), strength, duration, name), flush=True)
    max_error = max(abs(c['oracle_standardized_error']) for r in results for c in r['categories'].values())
    return {'schema_version': 1, 'seed': seed, 'replicates_per_cell': replicates, 'independent_pairs_per_replicate': num_pairs,
            'scope': 'known_landscape_known_parents_endpoint_convergence_oracle', 'production_omega_calibrated': False,
            'observation': 'net endpoint substitutions; hidden recurrent events are not counted',
            'null': 'constant pair landscape across branches; epistasis allowed, no shared adaptive field shift',
            'alternative': 'same landscape plus field +2 toward state 2 at site 0 on both branches',
            'baseline': 'independent sites matched to true stationary marginals and mean per-site substitution rates',
            'max_absolute_oracle_mc_standardized_error': max_error, 'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates', type=int, default=5000)
    parser.add_argument('--pairs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260911)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    if args.replicates < 2 or args.pairs < 1:
        parser.error('At least two replicates and one pair are required.')
    result = run(args.replicates, args.pairs, args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (args.outdir / 'summary.tsv').open('w') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(['coupling', 'duration', 'background', 'oracle_E', 'independent_E', 'mc_E', 'mc_SE',
                         'oracle_FPR', 'independent_FPR', 'oracle_power', 'independent_raw_power'])
        for row in result['results']:
            c = row['categories']['any2spe']
            writer.writerow([row['coupling'], row['duration'], row['background'], c['oracle_expectation'],
                             c['independent_expectation'], c['mc_mean'], c['mc_standard_error'],
                             *[row[key]['exact_rejection_probability'] for key in
                               ('oracle_null', 'independent_null', 'oracle_alternative', 'independent_alternative')]])


if __name__ == '__main__':
    main()
