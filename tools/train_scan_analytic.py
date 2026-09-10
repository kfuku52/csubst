#!/usr/bin/env python3
"""Train once on CTMC histories, freeze, then evaluate paired untouched scans."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CSUBST_DISABLE_EXTENSIONS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from csubst import scan_analytic, ete
from scan_evolution_benchmark import simulate, scan_frame

SCENARIOS = ['sparse', 'unequal', 'uncertain']


def basis():
    nodes, weights = np.polynomial.legendre.leggauss(8)
    multipliers = np.exp((nodes + 1) / 2 * np.log(1000))
    groups = [scan_analytic.default_profile()['atoms']]
    for rho in (1., .5):
        groups.append([dict(multiplier=float(m), participation=rho, weight=float(w / 2))
                       for m, w in zip(multipliers, weights)])
    for rho in (.5, 1.):
        groups.append([dict(multiplier=None, participation=rho, weight=1.)])
    keys = list(dict.fromkeys((a['multiplier'], a['participation']) for group in groups for a in group))
    matrix = np.zeros((len(groups), len(keys)))
    for i, group in enumerate(groups):
        for atom in group:
            matrix[i, keys.index((atom['multiplier'], atom['participation']))] += atom['weight']
    atoms = [dict(multiplier=m, participation=r, weight=1 / len(keys)) for m, r in keys]
    return dict(version=1, atoms=atoms), matrix


def component_logs(logs, matrix):
    return np.array([logsumexp(logs[1:], b=weights) - logs[0] for weights in matrix])


def evaluate_frame(frame, data, engine, matrix, learned):
    output = []
    labels = {n.name: int(ete.get_prop(n, 'numerical_label')) for n in data['tree'].traverse()}
    branches = {'trait1': data['fg_ids'], 'trait2': [labels['b'], labels['f']]}
    evidence = []
    cache = {}
    for _, row in frame.iterrows():
        site = int(row['site'])
        key = (site, row['trait'], str(row['from_state_ids']), str(row['to_state_ids']))
        if key not in cache:
            tips = {leaf: obs[site] for leaf, obs in data['observations'].items()}
            logs = engine.log_likelihoods(tips, branches[row['trait']],
                [int(v) for v in key[2].split(',') if v], [int(v) for v in key[3].split(',') if v])
            cache[key] = component_logs(logs, matrix)
        evidence.append(cache[key])
    evidence = np.asarray(evidence).reshape((-1, len(matrix)))
    pvalues = dict(poisson=frame['p_rate_enrichment_asymptotic'].to_numpy(dtype=float),
                   fixed=np.exp(np.minimum(0, -evidence[:, 0])),
                   learned=np.exp(np.minimum(0, -logsumexp(evidence, b=learned, axis=1))))
    signal = data['signal_sites']
    for method, p in pvalues.items():
        q = np.ones(len(frame))
        for indices in frame.groupby(['trait', 'scan_match'], sort=False).indices.values():
            q[indices] = scan_analytic.adjusted_pvalues(p[indices], len(indices))
        hits = frame.iloc[np.flatnonzero(q <= .05)]
        false = int((hits['site'].astype(int) >= signal).sum())
        correct = hits[(hits['trait'] == 'trait1') & (hits['to_state_ids'].astype(str) == '1')]
        detected = set(correct['site'].astype(int)) & set(range(signal))
        end_truth = set(np.flatnonzero(data['recurrent_endpoints'][:signal]))
        jump_truth = set(np.flatnonzero(data['recurrent_jumps'][:signal]))
        output.append(dict(method=method, candidates=len(frame), discoveries=len(hits), false=false,
                           false_any=int(false > 0), fdp=false / max(len(hits), 1),
                           signal=signal, detected=len(detected),
                           endpoints=len(end_truth), endpoint_detected=len(detected & end_truth),
                           jumps=len(jump_truth), jump_detected=len(detected & jump_truth)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-replicates', type=int, default=100)
    parser.add_argument('--test-replicates', type=int, default=100)
    parser.add_argument('--resume-frozen', action='store_true', help='Reuse saved weights after an interrupted evaluation.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'profile.json').exists() and not args.resume_frozen:
        raise ValueError('Use a fresh output directory: frozen profiles must not be overwritten.')
    start = time.perf_counter()
    profile, matrix = basis()
    config = dict(training_replicates=args.train_replicates, test_replicates=args.test_replicates,
                  seed=9102026, role_train=0, role_holdout=1, training_sites=8,
                  test_sites=12, test_signal_sites=3, scenarios=SCENARIOS, foreground_counts=[2, 4],
                  test_regimes=['null', 'moderate', 'heterogeneous'], filters=[False, True],
                  basis_names=['fixed_four', 'log_uniform_full', 'log_uniform_half', 'infinite_half', 'infinite_full'],
                  basis_matrix=matrix.tolist(), atom_profile=profile,
                  objective='mean log predictive likelihood on all training signal sites, no candidate selection',
                  correction='BH within identical selected trait x match rows for all methods; diagnostic only',
                  truth='null sites for FDP; process changed sites, recurrent endpoints and jumps for sensitivity')
    if args.resume_frozen:
        saved = json.loads((args.output / 'preregistered.json').read_text())
        if saved != config:
            raise ValueError('Resume configuration differs from preregistration.')
        payload = (args.output / 'profile.json').read_text()
        frozen = scan_analytic.validate_profile(json.loads(payload))
        learned = np.array(frozen['provenance']['component_weights'])
    else:
        (args.output / 'preregistered.json').write_text(json.dumps(config, indent=2) + '\n')
        training = []
        for si, scenario in enumerate(SCENARIOS):
            for fg in (2, 4):
                for rep in range(args.train_replicates):
                    data = simulate(scenario, fg, 8, 8, 'training', [9102026, 0, si, fg, rep])
                    engine = scan_analytic.EndpointEnrichment(data['model'], np.array([0, 0, 1, 1]), profile)
                    for site in range(8):
                        logs = engine.log_likelihoods({leaf: obs[site] for leaf, obs in data['observations'].items()},
                                                     data['fg_ids'], [0], [1])
                        training.append(component_logs(logs, matrix))
                print('training', scenario, fg, flush=True)
        training = np.array(training)
        scaled = np.exp(training - training.max(axis=1, keepdims=True))
        def objective(w):
            mixture = scaled @ w
            return -np.log(mixture).mean(), -(scaled / mixture[:, None]).mean(axis=0)
        fit = minimize(objective, np.full(5, .2), jac=True, method='SLSQP', bounds=[(1e-8, 1)] * 5,
                       constraints=[dict(type='eq', fun=lambda w: w.sum() - 1, jac=lambda w: np.ones(5))],
                       options=dict(ftol=1e-12, maxiter=1000))
        if not fit.success:
            raise RuntimeError(fit.message)
        learned = fit.x / fit.x.sum()
        frozen = dict(version=1, provenance=dict(training_config='preregistered.json',
            training_sites=len(training), component_weights=learned.tolist(), objective=float(fit.fun),
            optimizer_message=str(fit.message), holdout_used=False), atoms=[])
        for atom, weight in zip(profile['atoms'], learned @ matrix):
            frozen['atoms'].append(dict(atom, weight=float(weight)))
        scan_analytic.validate_profile(frozen)
        payload = json.dumps(frozen, indent=2) + '\n'
        (args.output / 'profile.json').write_text(payload)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    print('FROZEN', digest, learned.tolist(), flush=True)
    results = []
    for si, scenario in enumerate(SCENARIOS):
        for fg in (2, 4):
            for ri, regime in enumerate(config['test_regimes']):
                for rep in range(args.test_replicates):
                    data = simulate(scenario, fg, 12, 0 if regime == 'null' else 3, regime,
                                    [9102026, 1, si, fg, ri, rep])
                    engine = scan_analytic.EndpointEnrichment(data['model'], np.array([0, 0, 1, 1]), profile)
                    # scan_frame rescaling mutates its tree; each filter needs a fresh topology.
                    for filtered in (False, True):
                        scan_data = simulate(scenario, fg, 12, data['signal_sites'], regime, data['seed_key'])
                        frame = scan_frame(scan_data, filtered).reset_index(drop=True)
                        for record in evaluate_frame(frame, data, engine, matrix, learned):
                            results.append(dict(scenario=scenario, foregrounds=fg, regime=regime,
                                                filtered=filtered, replicate=rep, **record))
                print('holdout', scenario, fg, regime, flush=True)
                pd.DataFrame(results).to_csv(args.output / 'paired_results.csv', index=False)
    if hashlib.sha256((args.output / 'profile.json').read_bytes()).hexdigest() != digest:
        raise RuntimeError('Frozen profile changed during evaluation.')
    (args.output / 'run.json').write_text(json.dumps(dict(profile_sha256=digest,
        seconds=time.perf_counter() - start, python=sys.version), indent=2) + '\n')


if __name__ == '__main__':
    main()
