#!/usr/bin/env python3
"""Independent history-level validation of exploratory structure weighting.

Gillespie evolution uses single-site mutations and a pairwise sequence fitness,
not the predictor's exp(beta * context * feature) model. Binary and 61-sense-
codon alphabets are supported. Exact pre-branch states give oracle context;
this does not validate context reconstructed from all observed tips or omega P
values. The IID experiment supplies independent random context.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csubst import epistasis, ete, genetic_code, tree  # noqa: E402


class Landscape:
    def __init__(self, fields, couplings, site_rates, alphabet='binary'):
        self.fields = np.asarray(fields, float)
        self.couplings = np.asarray(couplings, float)
        self.site_rates = np.asarray(site_rates, float)
        n = len(self.fields)
        if (self.couplings.shape != (n, n) or self.site_rates.shape != (n,) or
                not np.allclose(self.couplings, self.couplings.T) or np.any(np.diag(self.couplings)) or
                not all(np.isfinite(a).all() for a in (self.fields, self.couplings, self.site_rates)) or
                np.any(self.site_rates <= 0)):
            raise ValueError('Invalid fitness landscape.')
        self.alphabet = alphabet
        if alphabet == 'binary':
            self.symbols, self.amino_acids = ['0', '1'], ['0', '1']
            self.traits = np.array([-1., 1.])
            self.neighbors = [np.array([1]), np.array([0])]
        elif alphabet == 'codon':
            table = sorted((c, a) for a, c in genetic_code.get_codon_table(1) if a != '*')
            self.symbols, self.amino_acids = map(list, zip(*table))
            # A deliberately simple AA phenotype, fixed before any evolution.
            aa_values = dict(zip(sorted(set(self.amino_acids)), np.linspace(-1., 1., 20)))
            self.traits = np.array([aa_values[a] for a in self.amino_acids])
            self.neighbors = [np.array([j for j, y in enumerate(self.symbols)
                                        if sum(a != b for a, b in zip(x, y)) == 1]) for x in self.symbols]
        else:
            raise ValueError('Unknown alphabet.')

    def fitness(self, sequence):
        x = self.traits[np.asarray(sequence)]
        return float(self.fields @ x + .5 * x @ self.couplings @ x)

    def transitions(self, sequence):
        x = self.traits[sequence]
        background = self.fields + self.couplings @ x
        sites, targets, rates, synonymous = [], [], [], []
        for i, old in enumerate(sequence):
            for new in self.neighbors[old]:
                delta = (self.traits[new] - x[i]) * background[i]
                # Symmetric single-nucleotide proposals, scaled per site.
                rate = self.site_rates[i] * np.exp(.5 * delta)
                sites.append(i)
                targets.append(new)
                rates.append(rate)
                synonymous.append(self.amino_acids[old] == self.amino_acids[new])
        return np.array(sites), np.array(targets), np.array(rates), np.array(synonymous)

    def evolve(self, initial, duration, rng):
        if not np.isfinite(duration) or duration < 0:
            raise ValueError('Branch duration must be finite and nonnegative.')
        sequence = np.array(initial, dtype=int, copy=True)
        counts = np.zeros((2, len(sequence)))  # N, S; includes recurrent events.
        elapsed = 0.
        while True:
            sites, targets, rates, synonymous = self.transitions(sequence)
            total = rates.sum()
            elapsed += rng.exponential(1. / total)
            if elapsed >= duration:
                break
            event = rng.choice(len(rates), p=rates / total)
            counts[int(synonymous[event]), sites[event]] += 1
            sequence[sites[event]] = targets[event]
        return sequence, counts

    def exact_generator(self):
        if self.alphabet != 'binary' or len(self.fields) > 8:
            raise ValueError('Exact enumeration is restricted to small binary landscapes.')
        states = np.array(list(itertools.product([0, 1], repeat=len(self.fields))))
        indices = {tuple(s): i for i, s in enumerate(states)}
        q = np.zeros((len(states), len(states)))
        for i, s in enumerate(states):
            sites, targets, rates, _ = self.transitions(s)
            for site, target, rate in zip(sites, targets, rates):
                new = s.copy()
                new[site] = target
                q[i, indices[tuple(new)]] = rate
            q[i, i] = -q[i].sum()
        fitness = np.array([self.fitness(s) for s in states])
        pi = np.exp(fitness - fitness.max())
        return states, q, pi / pi.sum()


def example_tree():
    clades = ['((T{0}:0.3,T{1}:0.3):0.3,(T{2}:0.3,T{3}:0.3):0.3):0.3'.format(*range(i, i + 4))
              for i in range(0, 16, 4)]
    return tree.add_numerical_node_labels(ete.PhyloNode('(' + ','.join(clades) + ');'))


def simulate_history(model, tr, rng, foreground_shift=0.):
    num_branch = len(list(tr.traverse()))
    states = {}
    counts = np.zeros((2, num_branch, len(model.fields)))
    context = np.zeros((num_branch, 1))
    for node in tr.traverse('preorder'):
        i = int(ete.get_prop(node, 'numerical_label'))
        if ete.is_root(node):
            # Explicit uniform root, shared by all clades; not a claim of stationarity.
            states[i] = rng.integers(len(model.symbols), size=len(model.fields))
            continue
        parent = states[int(ete.get_prop(node.up, 'numerical_label'))]
        context[i, 0] = model.traits[parent].mean()  # available before the branch evolves
        current = model
        taxa = set(ete.get_leaf_names(node))
        if foreground_shift and taxa.issubset({'T0', 'T1', 'T2', 'T3', 'T8', 'T9', 'T10', 'T11'}):
            current = Landscape(model.fields + foreground_shift, model.couplings, model.site_rates, model.alphabet)
        states[i], counts[:, i] = current.evolve(parent, float(node.dist), rng)
    return counts, context, states


def evaluate(counts, context, features, mask, groups, config):
    partitions = np.zeros(len(counts), dtype=int)
    fitted = epistasis.crossfit(counts, context, features, mask, groups, partitions, config)
    folds = [f for f in fitted['beta_diag']['outer_folds'] if f['scored']]
    events = sum(f['events'] for f in folds)
    gain = sum(f['outer_log_score'] - f['outer_null_log_score'] for f in folds)
    return {'events': events, 'outer_gain_per_event': gain / events if events else 0.,
            'mean_beta': float(fitted['beta_by_branch'][mask.any(axis=1)].mean()),
            'positive_beta_fraction': float((fitted['beta_by_branch'][mask.any(axis=1)] > 0).mean())}


def run(replicates, seed, scenarios):
    results = []
    config = {'epistasis_beta_auto': True, 'asrv_dirichlet_alpha': 1., 'epistasis_clip_value': 3.}
    for index, scenario in enumerate(scenarios):
        for replicate in range(replicates):
            # Stable scenario streams, unaffected by which other scenarios run.
            scenario_id = ['iid', 'binary_null', 'binary_epistatic', 'codon_null', 'codon_epistatic',
                           'codon_convergence', 'codon_epistatic_convergence'].index(scenario)
            rng = np.random.default_rng(np.random.SeedSequence([seed, scenario_id, replicate]))
            if scenario == 'iid':
                counts = np.eye(20)[rng.integers(20, size=500)]
                features = np.linspace(-1., 1., 20)[:, None]
                context = features[rng.integers(20, size=500)]
                groups = np.repeat(np.arange(5), 100)
                mask = np.ones_like(counts, bool)
                result = evaluate(counts, context, features, mask, groups, config)
            else:
                alphabet = 'codon' if scenario.startswith('codon') else 'binary'
                size = 6
                adjacency = np.zeros((size, size))
                for i, j in [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5)]:
                    adjacency[i, j] = adjacency[j, i] = 1.
                interaction = .7 if 'epistatic' in scenario else 0.
                model = Landscape(np.linspace(-.3, .3, size), interaction * adjacency,
                                  np.linspace(.15, .45, size), alphabet)
                tr = example_tree()
                counts, context, _ = simulate_history(model, tr, rng, 1. if 'convergence' in scenario else 0.)
                features = adjacency.sum(axis=1)[:, None]
                features = (features - features.mean()) / features.std()
                _, groups, _ = epistasis.tree_layout(tr, counts.shape[1], 4)
                mask = np.ones_like(counts[0], bool)
                mask[int(ete.get_prop(tr, 'numerical_label'))] = False
                result = evaluate(counts[0], context, features, mask, groups, config)
                result['synonymous_events'] = float(counts[1].sum())
            results.append(dict(result, scenario=scenario, replicate=replicate))
        print('Completed {}: {} replicates'.format(scenario, replicates), flush=True)
    summary = {}
    for scenario in scenarios:
        rows = [r for r in results if r['scenario'] == scenario]
        gains = np.array([r['outer_gain_per_event'] for r in rows])
        se = float(gains.std(ddof=1) / np.sqrt(len(gains))) if len(gains) > 1 else None
        summary[scenario] = {'replicates': len(rows), 'mean_outer_gain_nats_per_event': float(gains.mean()),
                             'monte_carlo_standard_error': se,
                             'mean_positive_beta_fraction': float(np.mean([r['positive_beta_fraction'] for r in rows]))}
    return {'seed': seed, 'scope': 'history_level_prediction_only', 'omega_pvalue_calibrated': False,
            'context': 'independent random for IID; exact pre-branch parent state for CTMC (oracle)',
            'root_distribution': 'uniform; no stationary-root assumption',
            'interpretation': 'Positive beta is not an epistasis test; gain is on outer clades.',
            'summary': summary, 'replicates': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--scenarios', default='iid,binary_null,binary_epistatic,codon_null,codon_epistatic')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error('--replicates must be positive')
    payload = run(args.replicates, args.seed, args.scenarios.split(','))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
