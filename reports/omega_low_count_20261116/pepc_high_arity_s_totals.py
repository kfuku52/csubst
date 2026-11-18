"""Factorial pilot: S destination totals and site rates on the existing PEPC stem panel."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import os
from pathlib import Path

import numpy as np
import pandas as pd

from csubst import cli, empirical_urn, endpoint_io, omega, param, parser_misc, runtime
from pepc_high_arity_accuracy import EXT, FITS, SCENARIOS, SEEDS, STEMS


def score(task):
    panel, earlier, work, scenario, seed = task
    rep = f'{scenario}-{seed}'
    wd = work / rep
    wd.mkdir()
    os.chdir(wd)
    sim = panel / rep / 'sim'
    fit, = (panel / rep / FITS[1] / 'iqtree').glob('*.iqtree')
    args = ['search', '--alignment_file', str(sim/'sim.fa'), '--rooted_tree_file', str(sim/'sim_true_asr.treefile'),
            '--expectation_method', 'urn', '--asrv', 'empirical_all', '--output_stat', 'any2spe',
            '--max_arity', '6', '--b', 'no']
    for ext in EXT:
        args.extend(['--iqtree_' + ext, str(fit.with_suffix('.' + ext))])
    g = param.get_global_parameters(cli._build_parser().parse_args(args))
    g = runtime.ensure_output_layout(g, create_dir=True)
    g = parser_misc.prepare_input_context(g, include_foreground=True, include_marginal=True,
                                          resolve_state_subset=True, prepare_state=True)
    endpoint_io.prepare(g)
    data, rates = empirical_urn.fit_all_branches(g, g['_endpoint_tensors']['N'], g['_endpoint_tensors']['S'])
    train, ids = g['_asrv_training_ids'], np.array(STEMS)
    state = g['state_cdn']
    syn = data.syn_totals
    s_total = syn.sum(axis=(1, 2))
    effort = np.divide(s_total, data.eligible.sum(axis=1), out=np.zeros_like(s_total), where=data.eligible.any(axis=1))
    true_rate = pd.read_table(sim/'sim_true_asr.rate')['C_Rate'].to_numpy()
    truth = pd.read_table(earlier/rep/'replay/sim_expected_omega.tsv').set_index('branches')
    combos = {k: np.array(list(combinations(range(len(ids)), k))) for k in range(2, 7)}
    rows = []
    for label, rate in [('current', rates.s_rate), ('cv', rates.n_rate), ('oracle', true_rate)]:
        weights = []
        forecast = np.zeros_like(syn[ids])
        for i, group in enumerate(data.groups):
            if len(group) < 2:
                continue
            parent = state[data.parents[:, None, None], np.arange(len(rate))[None, :, None], group]
            w = (parent @ (1. - np.eye(len(group)))) * (rate * data.eligible)[:, :, None]
            # Destination rate per unit of parental opportunity. Retain the observed
            # branch-wide S total, replacing only its allocation among destinations.
            opportunity = w.sum(axis=1)
            exposure = effort[train] @ opportunity[train]
            counts = syn[train, i, :len(group)].sum(axis=0)
            coef = np.divide(counts, exposure, out=np.zeros_like(counts), where=exposure > 0)
            forecast[:, i, :len(group)] = opportunity[ids] * coef
            weights.append((i, w[ids]))
        assert (forecast.sum(axis=(1,2)) > 0).all()
        forecast *= (s_total[ids] / forecast.sum(axis=(1,2)))[:, None, None]
        np.testing.assert_allclose(forecast.sum(axis=(1,2)), s_total[ids])
        for allocation, totals in [('observed', syn[ids]), ('pooled', forecast)]:
            profiles = [omega.wallenius_inclusion_profiles(w[:, :, j], totals[:, i, j], g)
                        for i, w in weights for j in range(w.shape[2]) if totals[:, i, j].any()]
            for k, ix in combos.items():
                keys = [','.join(map(str, c)) for c in ids[ix]]
                target = truth.loc[keys, 'muSany2spe_null'].to_numpy()
                ec = sum(omega._calc_cb_site_overlap(ix, prof, np.float64) for prof in profiles)
                if label == 'current' and allocation == 'observed':
                    ref = pd.read_table(earlier/rep/FITS[1]/'urn/target/out'/f'csubst_cb_{k}.tsv',
                                        converters={'ECSany2spe': float})
                    ref.index = ref.filter(regex='^branch_id_').astype(str).agg(','.join,axis=1)
                    np.testing.assert_allclose(ec, ref.loc[keys, 'ECSany2spe'], rtol=1e-10, atol=6e-18)
                rows.append(dict(scenario=scenario, seed=seed, rate=label, allocation=allocation, k=k,
                                 estimated=ec.sum(), target=target.sum(), abs_error=np.abs(ec-target).sum()))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('panel', 'earlier', 'work'):
        p.add_argument(arg, type=Path)
    a = p.parse_args()
    panel, earlier, work = [x.resolve() for x in (a.panel, a.earlier, a.work)]
    work.mkdir(exist_ok=True)
    with ProcessPoolExecutor(3) as pool:
        groups = pool.map(score, [(panel, earlier, work, cond, seed) for cond in SCENARIOS for seed in SEEDS])
        pd.DataFrame(row for group in groups for row in group).to_csv(work/'results.tsv', sep='\t', index=False)
