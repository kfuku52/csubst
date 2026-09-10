#!/usr/bin/env python3
"""Independent train/validation simulations, ASR + discovery, and jump-count error.

A small two-state process permits complete jump histories as ground truth.
The installed scan candidate-selection and rate-statistic code is used unchanged.
This is a conditional-model calibration experiment, not codon-model misspecification validation.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from csubst import ete, scan_ctmc, substitution_scan, tree  # noqa: E402


def simulate(g, rng, alternative=False):
    q = g['instantaneous_codon_rate_matrix']
    state = np.zeros_like(g['state_cdn'])
    truth = np.zeros((state.shape[0], state.shape[1], 1, 2, 2))
    draws = {}
    for node in g['tree'].traverse('preorder'):
        i = int(ete.get_prop(node, 'numerical_label'))
        if ete.is_root(node):
            draws[i] = rng.integers(2, size=state.shape[1])
        else:
            draws[i] = draws[int(ete.get_prop(node.up, 'numerical_label'))].copy()
            rate = q.copy()
            if alternative and node.name.startswith('A'):
                rate[0, 1] *= g['validation_alt_factor']
                rate[0, 0] = -rate[0, 1]
            for site, start in enumerate(draws[i]):
                elapsed, current = 0., start
                while True:
                    elapsed += rng.exponential(1 / -rate[current, current])
                    if elapsed >= node.dist:
                        break
                    truth[i, site, 0, current, 1-current] += 1
                    current = 1-current
                draws[i][site] = current
        if not ete.get_children(node):
            state[i, np.arange(state.shape[1]), draws[i]] = 1
    return state, truth


def evaluate(g, emissions):
    posterior, joint = scan_ctmc.infer(g['tree'], emissions, g['instantaneous_codon_rate_matrix'], [.5, .5], [0, 1], 'joint')
    _, bridge = scan_ctmc.infer(g['tree'], emissions, g['instantaneous_codon_rate_matrix'], [.5, .5], [0, 1], 'bridge')
    marginal = np.zeros_like(joint)
    for node in g['tree'].traverse():
        if ete.is_root(node):
            continue
        i = int(ete.get_prop(node, 'numerical_label'))
        parent = int(ete.get_prop(node.up, 'numerical_label'))
        marginal[i, :, 0] = posterior[parent, :, :, None] * posterior[i, :, None, :]
    marginal[:, :, 0, 0, 0] = marginal[:, :, 0, 1, 1] = 0
    for node in g['tree'].traverse():
        i = int(ete.get_prop(node, 'numerical_label'))
        ete.set_prop(node, 'Ndist', float(marginal[i].sum() / emissions.shape[1]))
    scores = {}
    for mode, tensor in [('legacy_default', marginal), ('marginal', marginal), ('joint', joint), ('bridge', bridge)]:
        context = dict(g, state_cdn=posterior, state_nsy=posterior, state_pep=posterior,
                       scan_observation='marginal' if mode == 'legacy_default' else mode,
                       scan_rate_length='n_rescaled' if mode == 'legacy_default' else 'raw',
                       scan_rate_exposure='q_weighted' if mode in ('marginal', 'legacy_default') else 'endpoint')
        frame, _ = substitution_scan._scan_substitutions_core(context, tensor, tensor)
        p = frame['p_rate_enrichment_asymptotic'].to_numpy(float)
        if not np.isfinite(p).all():
            raise ValueError('Undefined statistic')
        scores[mode] = float(p.min()) if len(p) else 1.
    return scores, dict(legacy_default=marginal, marginal=marginal, joint=joint, bridge=bridge)


def interval(k, n):
    return [0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1. if k == n else float(beta.ppf(.975, k+1, n-k))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train', type=int, default=1999)
    parser.add_argument('--validate', type=int, default=2000)
    parser.add_argument('--sites', type=int, default=8)
    parser.add_argument('--pairs', type=int, default=8)
    parser.add_argument('--tip-length', type=float, default=.12)
    parser.add_argument('--length-step', type=float, default=.02)
    parser.add_argument('--stem-length', type=float, default=.12)
    parser.add_argument('--min-support', type=int, default=3)
    parser.add_argument('--alt-factor', type=float, default=30)
    args = parser.parse_args()
    tr = tree.add_numerical_node_labels(ete.PhyloNode('(' + ','.join(
        f'(A{i}:{args.tip_length+args.length_step*i},B{i}:{args.tip_length+args.length_step*i})X{i}:{args.stem_length}' for i in range(args.pairs)) + ')R;', format=1))
    labels = {node.name: int(ete.get_prop(node, 'numerical_label')) for node in tr.traverse()}
    foreground = {f'A{i}' for i in range(args.pairs)}
    for node in tr.traverse():
        ete.add_features(node, is_fg_trait=node.name in foreground, SNdist=float(node.dist or 0), Ndist=float(node.dist or 0))
        for i in range(args.pairs):
            ete.add_features(node, **{f'is_lineage_fg_trait_{i+1}': node.name == f'A{i}'})
    state = np.zeros((len(labels), args.sites, 2))
    state[:, :, 0] = 1
    g = dict(tree=tr, fg_df=pd.DataFrame({'name': [f'A{i}' for i in range(args.pairs)], 'trait': list(range(1,args.pairs+1))}),
             fg_leaf_names={'trait': [[f'A{i}'] for i in range(args.pairs)]},
             fg_ids={'trait': np.array([labels[f'A{i}'] for i in range(args.pairs)])},
             fg_stem_only=True, scan_sister_stem_only=True,
             state_cdn=state, state_nsy=state, state_pep=state,
             nonsyn_state_orders=np.array(['A', 'K']), amino_acid_orders=np.array(['A', 'K']),
             instantaneous_codon_rate_matrix=np.array([[-1., 1.], [1., -1.]]),
             equilibrium_frequency=np.array([.5, .5]), substitution_model='GY', nonsynonymous_indices={'A': [0], 'K': [1]},
             iqtree_rate_values=np.ones(args.sites), scan_rate_exposure='endpoint', scan_observation='joint',
             float_tol=1e-12, nonsyn_recode='no', scan_match='any2spe', scan_min_event_pp=.5,
             scan_min_support=str(args.min_support), scan_rate_length='raw', scan_rate_event_mode='posterior_sum',
             scan_other_scope='all', scan_pvalue_calibration='none', scan_n_permutations=0,
             scan_permutation_seed=1, min_clade_bin_count=1, validation_alt_factor=args.alt_factor)
    modes = ['legacy_default', 'marginal', 'joint', 'bridge']
    scores = {phase: {mode: [] for mode in modes} for phase in ['train', 'null', 'alternative']}
    errors = {mode: [] for mode in modes}
    start = time.perf_counter()
    for phase, count, seed in [('train', args.train, 89001), ('null', args.validate, 89002), ('alternative', args.validate, 89003)]:
        rng = np.random.default_rng(seed)
        for i in range(count):
            emissions, truth = simulate(g, rng, phase == 'alternative')
            with contextlib.redirect_stdout(io.StringIO()):
                values, tensors = evaluate(g, emissions)
            for mode in modes:
                scores[phase][mode].append(values[mode])
                if phase == 'null':
                    errors[mode].append(float(((tensors[mode] - truth)**2).sum() / (3*args.pairs*args.sites*2)))
            if (i+1) % 25 == 0:
                print(phase, i+1, flush=True)
    result = dict(config=vars(args) | {'out': str(args.out)}, seconds=time.perf_counter()-start, modes={})
    for mode in modes:
        reference = np.asarray(scores['train'][mode])
        report = {'jump_count_rmse': float(np.sqrt(np.mean(errors[mode])))}
        for phase in ['null', 'alternative']:
            values = np.asarray(scores[phase][mode])
            empirical = (1 + (reference[:, None] <= values).sum(axis=0)) / (args.train + 1)
            for kind, pv in [('nominal', values), ('bootstrap', empirical)]:
                hits = int((pv <= .05).sum())
                report[phase + '_' + kind] = dict(rejections=hits, n=args.validate, rate=hits/args.validate,
                                                  interval_95=interval(hits, args.validate))
        result['modes'][mode] = report
    result['minimum_p_scores'] = scores
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['modes'], indent=2))


if __name__ == '__main__':
    main()
