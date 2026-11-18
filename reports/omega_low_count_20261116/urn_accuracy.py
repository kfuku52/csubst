"""Measure PEPC Wallenius urn estimates and sampled tail calls against simulation truth."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from csubst import endpoint_io, ete, tree

from pilot import EXPECTED, KEY, NULL, OBS, TRUTH


GENERATING = ['muNany2spe_sim', 'muSany2spe_sim']
PVALUE = 'pomegaCany2spe'
QVALUE = 'qomegaCany2spe'
DRAW_COUNTS = ['pvalue_n_any2spe', 'pvalue_undefined_any2spe']
SAMPLED = ('each', 'each_tau2')


def load_search(root, outdir):
    out = root / outdir
    manifest = json.loads((out / 'csubst_search_run.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Incomplete search: ' + str(out))
    endpoint_manifest = json.loads((out / 'csubst_endpoint_model.json').read_text())['codon']
    return pd.read_csv(out / 'csubst_cb_2.tsv', sep='\t'), endpoint_manifest


def search_argv(mode, seed, draws):
    profile = {'oracle_file': 'file', 'each_tau2': 'each'}.get(mode, mode)
    method = 'codon_model' if mode == 'codon_model' else 'urn'
    outdir = mode if method == 'codon_model' else 'urn-' + mode
    argv = ['search', '--alignment_file', 'sim.fa', '--rooted_tree_file', 'tree.nwk',
            '--threads', '1', '--blas_threads', '1', '--b', 'no', '--exclude_sister_pair', 'no',
            '--output_stat', 'any2spe', '--float_digit', '12', '--random_seed', str(seed),
            '--expectation_method', method, '--outdir', outdir]
    if method == 'urn':
        argv += ['--urn_model', 'wallenius', '--asrv', profile]
    for ext in ('iqtree', 'state', 'rate', 'treefile', 'log'):
        path = 'generating.rate' if mode == 'oracle_file' and ext == 'rate' else 'fit.' + ext
        argv += ['--iqtree_' + ext, path]
    if draws > 0 and mode in SAMPLED:
        argv += ['--calc_omegaC_pvalue', 'yes', '--omegaC_pvalue_null_model', 'hypergeom',
                 '--omegaC_pvalue_niter_schedule', str(draws), '--min_sub_pp', '0']
    if mode == 'each_tau2':
        argv += ['--asrv_concentration', '2']
    return outdir, argv


def run_search(root, mode, seed, draws, read_existing):
    outdir, argv = search_argv(mode, seed, draws)
    if read_existing:
        return load_search(root, outdir)
    log = root / (outdir + '.log')
    with log.open('w') as f:
        try:
            subprocess.run([sys.executable, '-m', 'csubst', *argv], cwd=root,
                           stdout=f, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'Urn search failed. See {log}.') from exc
    return load_search(root, outdir)


def summarize(frame, subsets, fitted, tag, sampling=None):
    truth = frame[TRUTH].to_numpy()
    obs = frame[OBS].to_numpy()
    null = frame[NULL].to_numpy()
    alt = frame[GENERATING].to_numpy()
    estimated = fitted[EXPECTED].to_numpy()
    if sampling is None:
        sampling = np.column_stack((np.full((len(frame), 2), np.nan), np.zeros((len(frame), 2))))
    for subset, include in subsets.items():
        if not include.any():
            continue
        ec = estimated[include]
        p, q, valid_draws, undefined_draws = sampling[include].T
        assessed = np.isfinite(p)
        candidates = (obs[include, 0] >= 2.) & (fitted['omegaCany2spe'][include].to_numpy() >= 5.)
        raw_hits, adjusted_hits = p <= .05, q <= .05
        result = dict(subset=subset, method=tag, n_pairs=int(include.sum()),
                      finite_omega=int(np.isfinite(fitted['omegaCany2spe'][include]).sum()),
                      zero_ecn=int((ec[:, 0] == 0).sum()), zero_ecs=int((ec[:, 1] == 0).sum()),
                      returned_p=int(assessed.sum()), p_le_005=int(raw_hits.sum()),
                      q_le_005=int(adjusted_hits.sum()), passing_search_cutoff=int(candidates.sum()),
                      q_after_search_cutoff=int((adjusted_hits & candidates).sum()),
                      q_with_obs_n_lt_one=int((adjusted_hits & (obs[include, 0] < 1.)).sum()),
                      q_with_true_n_zero=int((adjusted_hits & (truth[include, 0] == 0.)).sum()),
                      valid_draws=int(valid_draws.sum()), undefined_draws=int(undefined_draws.sum()))
        for name, data in [('truth', truth), ('obs', obs), ('q_null', null), ('q_sim', alt), ('expected', estimated)]:
            result.update({name + '_' + side: value for side, value in zip('ns', data[include].sum(axis=0))})
        result['ecn_sse'], result['ecs_sse'] = np.square(ec - null[include]).sum(axis=0)
        yield result


def run_case(args):
    rep, cases, work, modes, draws, read_existing = args
    scenario, seed_s = rep.name.rsplit('-', 1)
    seed = int(seed_s)
    existing = cases / rep.name
    if existing.is_dir():
        fit_report = existing / 'fit.iqtree'
    else:
        fit_report, = (rep / 'iqtree-default').glob('*.iqtree')
    root = work / rep.name
    if not read_existing:
        root.mkdir(parents=True)
        shutil.copyfile(rep / 'sim/sim.fa', root / 'sim.fa')
        shutil.copyfile(rep / 'sim/sim_true_asr.treefile', root / 'tree.nwk')
        shutil.copyfile(rep / 'sim/sim_true_asr.rate', root / 'generating.rate')
        for ext in ('iqtree', 'state', 'rate', 'treefile', 'log', 'ckp.gz'):
            shutil.copyfile(str(fit_report).removesuffix('.iqtree') + '.' + ext, root / ('fit.' + ext))
    codon, ref_model = run_search(root, 'codon_model', seed, draws, read_existing)
    truth = pd.read_csv(rep / 'sim/sim_expected_omega.tsv', sep='\t')
    tr = tree.add_numerical_node_labels(ete.PhyloNode(str(rep / 'sim/sim_true_asr.treefile'), format=1))
    parents, _ = endpoint_io._tree_arrays({'tree': tr}, structural=False)
    names = {int(ete.get_prop(node, 'numerical_label')): node.name for node in tr.traverse()}
    for i in (1, 2):
        if not truth[f'branch_id_{i}'].map(names).eq(truth[f'branch_name_{i}']).all():
            raise ValueError('Simulated and inferred branch labels differ.')
    paired = truth.merge(codon[KEY + EXPECTED + OBS + ['omegaCany2spe']], on=KEY, how='outer',
                         validate='one_to_one', indicator=True)
    if not paired.pop('_merge').eq('both').all():
        raise ValueError('Search pair set differs from truth.')
    paired = paired.dropna(subset=EXPECTED).sort_values(KEY)
    a, b = paired[KEY].to_numpy().T
    foreground = (paired.foreground_1.eq(1) & paired.foreground_2.eq(1)).to_numpy()
    sister = parents[a] == parents[b]
    null_omega = paired['omegaCany2spe_expected'].to_numpy()
    subsets = {
        'all': np.ones(len(paired), dtype=bool),
        'sisters': sister,
        'different_parents': ~sister,
        'two_foreground': foreground,
        'other_pairs': ~foreground,
        'enrichment_ge_1_5': null_omega >= 1.5,
        'near_null': np.isclose(null_omega, 1., rtol=.001),
        'low_expected_s': paired[NULL[1]].le(.1).to_numpy(),
    }
    summary = list(summarize(paired, subsets, paired, 'codon_model'))
    for mode in modes:
        if mode == 'oracle_file' and scenario != 'rate_mixture':
            continue
        table, model = run_search(root, mode, seed, draws, read_existing)
        for key in ('rates', 'weights', 'branch_lengths'):
            if model[key] != ref_model[key]:
                raise ValueError('The urn search changed the fitted ' + key)
        if model['parameter_source']['generator_sha256'] != ref_model['parameter_source']['generator_sha256']:
            raise ValueError('The urn search changed the fitted Q.')
        merged = paired[KEY].merge(table, on=KEY, how='left', validate='one_to_one')
        np.testing.assert_allclose(merged[OBS], paired[OBS], rtol=1e-9, atol=6e-12)
        if not np.isfinite(merged[EXPECTED]).all().all():
            raise ValueError('Urn EC is missing for an eligible paired row.')
        tests = merged[[PVALUE, QVALUE, *DRAW_COUNTS]].to_numpy() if draws > 0 and mode in SAMPLED else None
        if tests is not None:
            if not np.isfinite(tests[:, 0]).any():
                raise ValueError('The urn sampler returned no finite p-values.')
            np.testing.assert_array_equal(tests[:, 2:].sum(axis=1), draws)
        summary.extend(summarize(paired, subsets, merged, 'urn_' + mode, tests))
        print(rep.name, mode, len(merged), flush=True)
    return [dict(scenario=scenario, seed=seed, **entry) for entry in summary]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('panel', type=Path)
    p.add_argument('cases', type=Path)
    p.add_argument('work', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--jobs', type=int, default=3)
    p.add_argument('--draws', type=int, default=999, help='Use 0 for expected counts without sampled tails.')
    p.add_argument('--modes', nargs='+',
                   choices=['no', 'pool', 'sn', 'each', 'file', 'file_each', 'oracle_file', 'each_tau2'],
                   default=['each', 'file', 'no', 'oracle_file'])
    p.add_argument('--scenarios', nargs='+', choices=['baseline', 'biased', 'rate_mixture'],
                   default=['baseline', 'biased', 'rate_mixture'])
    p.add_argument('--seeds', nargs='+', type=int, default=[101, 202, 303])
    p.add_argument('--read-existing', action='store_true')
    args = p.parse_args()
    if args.draws < 0:
        p.error('--draws should be nonnegative')
    jobs = [(args.panel.resolve() / 'PEPC' / f'{scenario}-{seed}', args.cases.resolve(),
             args.work.resolve(), args.modes, args.draws, args.read_existing)
            for scenario in args.scenarios for seed in args.seeds]
    results = []
    with ThreadPoolExecutor(args.jobs) as pool:
        for rows in pool.map(run_case, jobs):
            results.extend(rows)
    pd.DataFrame(results).to_csv(args.output, sep='\t', index=False, float_format='%.12g', na_rep='NA')


if __name__ == '__main__':
    main()
