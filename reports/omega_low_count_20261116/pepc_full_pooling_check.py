"""Replay empirical_all search and compare each seed to independent full-pooling scores."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def check_scores(out, rep, scope, k, reference):
    ids = [f'branch_id_{j+1}' for j in range(k)]
    def number(value):
        return np.nan if value == 'NA' else float(value)

    counts = ('ECN', 'ECS', 'OCN', 'OCS', 'omegaC')
    cb = pd.read_table(out / f'csubst_cb_{k}.tsv', converters={x + 'any2spe': number for x in counts})
    provenance = json.loads((out / 'csubst_urn_provenance.json').read_text())
    assert provenance['asrv'] == 'empirical_all'
    assert provenance['empirical_urn']['n_draw_totals'] == 'observed_branch_n_with_pooled_route_rates'
    assert provenance['empirical_urn']['s_draw_totals'] == 'observed_branch_s_with_pooled_destination_rates'
    cb = cb[cb[ids].isin(provenance['training_branch_ids']).all(axis=1)]
    cb.index = cb[ids].astype(str).agg(','.join, axis=1)
    if scope == 'all':
        q = pd.read_table(rep / 'sim/sim_expected_omega.tsv')
        q.index = q[ids].astype(str).agg(','.join, axis=1)
    else:
        q = pd.read_table(rep / 'truth/replay/sim_expected_omega.tsv').set_index('branches')
    assert cb.index.is_unique and q.index.is_unique
    q = q.loc[cb.index]
    assert len(q) == reference.combos
    for channel in ('N', 'S'):
        estimate = cb[f'EC{channel}any2spe'].to_numpy()
        truth = q[f'mu{channel}any2spe_null'].to_numpy()
        assert np.isfinite(estimate).all()
        # cb is printed with 17 decimal places. Summation can incur one
        # half-unit in its final decimal place per combination.
        np.testing.assert_allclose(
            [estimate.sum(), np.abs(estimate - truth).sum()],
            [reference['ec_' + channel.lower()], reference['abs_' + channel.lower()]],
            rtol=2e-10, atol=len(q)*5e-18,
        )
    supported = (cb.OCNany2spe >= 1e-9) & (cb.OCSany2spe >= 1e-9)
    assert supported.sum() == reference.supported
    target = (q.muNany2spe_sim / q.muNany2spe_null) / (q.muSany2spe_sim / q.muSany2spe_null)
    delta = np.log1p(cb.omegaCany2spe[supported].to_numpy()) - np.log1p(target[supported].to_numpy())
    np.testing.assert_allclose(delta @ delta, reference.raw_sse, rtol=2e-8, atol=1e-14)


def replay(panel, destination, case, seed, refs):
    rep = panel / case
    inputs = destination / case / 'inputs'
    inputs.mkdir(parents=True)
    fit, = (rep / 'ECMK07_GY0K+F' / 'iqtree').glob('*.iqtree')
    fa, tree = rep / 'sim/sim.fa', rep / 'sim/sim_true_asr.treefile'
    for p in (fa, tree, fit.with_suffix('.ckp.gz'), panel / 'PEPC/inputs/foreground.txt'):
        shutil.copyfile(p, inputs / p.name)
    args = [sys.executable, '-m', 'csubst', 'search', '--alignment_file', str(inputs/fa.name),
            '--rooted_tree_file', str(inputs/tree.name), '--expectation_method', 'urn', '--output_stat', 'any2spe',
            '--exclude_sister_pair', 'no', '--b', 'no', '--threads', '1', '--blas_threads', '1',
            '--float_digit', '17', '--random_seed', str(seed), '--outdir', 'out']
    for ext in ('iqtree', 'state', 'log', 'rate', 'treefile'):
        local = inputs / fit.with_suffix('.' + ext).name
        shutil.copyfile(fit.with_suffix('.' + ext), local)
        args.extend(['--iqtree_' + ext, str(local)])
    for scope, options in [
        ('all', ['--max_arity', '2', '--exhaustive_until', '2']),
        ('stem', ['--max_arity', '6', '--exhaustive_until', '1', '--foreground', str(inputs/'foreground.txt'),
                  '--fg_exclude_wg', 'no', '--cutoff_stat', 'OCNany2spe,0']),
    ]:
        cwd = destination / case / scope
        cwd.mkdir()
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
        with (cwd / 'search.log').open('w') as log:
            subprocess.run(args + options, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        for k, ref in refs.loc[scope].iterrows():
            check_scores(cwd / 'out', rep, scope, k, ref)
    print('Verified', case, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    parser.add_argument('fresh_output', type=Path)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    scores = pd.read_table(HERE / 'pepc_pair_overlap_cv_results.tsv')
    groups = scores.query("panel == 'new' and method == 'both_full'").groupby(['scenario', 'seed'])
    with ThreadPoolExecutor(args.jobs) as pool:
        jobs = [pool.submit(replay, args.panel.resolve(), args.fresh_output.resolve(), f'{scenario}-{seed}',
                            seed, refs.set_index(['scope', 'k'])) for (scenario, seed), refs in groups]
        for job in jobs:
            job.result()
