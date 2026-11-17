"""Fit alternate codon models to both PEPC panels and score their pair counts."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from csubst import endpoint_io, ete, tree

from asr_bootstrap import omega_score
from pilot import EXPECTED, KEY, NULL, OBS, TRUTH
from urn_accuracy import GENERATING, load_search
from urn_parent_opportunity import metrics


MODELS = {'hybrid': 'ECMK07_GY0K+F+R4', 'gy': 'GY+F+R4',
          'mgk': 'MGK+F3X4+R4', 'ecmk07': 'ECMK07+F+R4'}
SCENARIOS = ('baseline', 'rate_mixture', 'biased')


def fit_run(panel, rep, label, iqtree, work):
    case = work / panel / rep.name / label
    case.mkdir(parents=True)
    seed = rep.name.rsplit('-', 1)[1]
    log = case / 'run.log'
    args = [sys.executable, '-m', 'csubst', 'search', '--alignment_file', str(rep / 'sim/sim.fa'),
            '--rooted_tree_file', str(rep / 'sim/sim_true_asr.treefile'), '--iqtree_exe', iqtree,
            '--iqtree_model', MODELS[label], '--iqtree_outdir', 'iqtree', '--outdir', 'search',
            '--random_seed', seed, '--threads', '1', '--blas_threads', '1', '--max_arity', '2',
            '--exhaustive_until', '2', '--exclude_sister_pair', 'no', '--output_stat', 'any2spe',
            '--float_digit', '12', '--b', 'no']
    with log.open('w') as f:
        try:
            subprocess.run(args, cwd=case, stdout=f, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'{panel}/{rep.name}/{label} failed. See {log}.') from e
    print('fitted', panel, rep.name, label, flush=True)


def fit_statistics(source, out):
    if out == 'codon_model':
        report = source / 'fit.iqtree'
    else:
        folder = 'iqtree-default' if out == 'default' else 'iqtree'
        report, = (source / folder).glob('*.iqtree')
    payload = report.read_text()
    likelihood = re.search(r'^Log-likelihood of the tree:\s*([-\d.]+)', payload, re.MULTILINE)
    aic = re.search(r'^Akaike information criterion \(AIC\) score:\s*([-\d.]+)', payload, re.MULTILINE)
    if likelihood is None or aic is None:
        raise ValueError(f'Incomplete model fit report: {report}')
    return {'log_likelihood': float(likelihood[1]), 'aic': float(aic[1])}


def scored_case(panel, rep, paths, labels):
    tr = tree.add_numerical_node_labels(ete.PhyloNode(str(rep / 'sim/sim_true_asr.treefile'), format=1))
    parents, _ = endpoint_io._tree_arrays({'tree': tr}, structural=False)
    names = {int(ete.get_prop(node, 'numerical_label')): node.name for node in tr.traverse()}
    frame = pd.read_csv(rep / 'sim/sim_expected_omega.tsv', sep='\t')
    for side in (1, 2):
        assert frame[f'branch_id_{side}'].map(names).eq(frame[f'branch_name_{side}']).all()
    table, fit = {}, {}
    valid = np.ones(len(frame), dtype=bool)
    for label, (source, out) in zip(labels, paths):
        current, meta = load_search(source, out)
        model = meta['parameter_source']['model']
        if model != MODELS[label]:
            raise ValueError(f'Unexpected model in {source}: {model}')
        fit[label] = fit_statistics(source, out)
        columns = OBS + EXPECTED + ['omegaCany2spe']
        joined = frame[KEY].merge(current[KEY + columns], how='outer', on=KEY, validate='one_to_one', indicator=True)
        assert joined['_merge'].eq('both').all()
        joined = frame[KEY].merge(joined.drop(columns='_merge'), on=KEY, how='left', validate='one_to_one')
        np.testing.assert_array_equal(joined[KEY], frame[KEY])
        table[label] = joined
        valid &= np.isfinite(joined[EXPECTED]).all(axis=1).to_numpy()
    for fitted in table.values():
        np.testing.assert_array_equal(valid, np.isfinite(fitted[EXPECTED]).all(axis=1).to_numpy())
    a, b = frame[KEY].to_numpy().T
    foreground = (frame.foreground_1.eq(1) & frame.foreground_2.eq(1)).to_numpy()
    subsets = {'all': valid, 'two_foreground': valid & foreground, 'other_pairs': valid & ~foreground,
               'sisters': valid & (parents[a] == parents[b]), 'different_parents': valid & (parents[a] != parents[b]),
               'ecs_ge_0_5': valid & frame[NULL[1]].ge(.5).to_numpy(),
               'true_n_positive': valid & frame[TRUTH[0]].gt(0).to_numpy()}
    common = np.logical_and.reduce([np.isfinite(t['omegaCany2spe']) for t in table.values()])
    fixed_scores = {label: omega_score(t[OBS].to_numpy(), frame[NULL].to_numpy()) for label, t in table.items()}
    common_fixed = np.logical_and.reduce([np.isfinite(score) for score in fixed_scores.values()])
    with np.errstate(divide='ignore', invalid='ignore'):
        target = (frame[GENERATING[0]] / frame[NULL[0]]) / (frame[GENERATING[1]] / frame[NULL[1]])
    finite_target = np.isfinite(target)
    rows = []
    scenario, seed = rep.name.rsplit('-', 1)
    for label, fitted in table.items():
        working = frame.join(fitted[OBS])
        expected = fitted[EXPECTED].to_numpy()
        # Use the CLI's own omegaC column to preserve its exact zero-tolerance decision.
        observed_score = fitted['omegaCany2spe'].to_numpy()
        for scope, select in subsets.items():
            if not select.any():
                continue
            info = metrics(working, expected, select, common)
            finite = select & common & finite_target
            fixed = select & common_fixed & finite_target
            errors = np.log1p(observed_score[finite]) - np.log1p(target[finite])
            errors_fixed = np.log1p(fixed_scores[label][fixed]) - np.log1p(target[fixed])
            observed_errors = np.square(working.loc[select, OBS].to_numpy() - working.loc[select, TRUTH].to_numpy())
            info.update(finite_omega=int(finite.sum()), omega_sse=np.square(errors).sum(),
                        zero_omega=int((observed_score[finite] == 0.).sum()),
                        obs_sse_n=observed_errors[:, 0].sum(), obs_sse_s=observed_errors[:, 1].sum(),
                        finite_oracle_ec=int(fixed.sum()), omega_oracle_ec_sse=np.square(errors_fixed).sum())
            rows.append(dict(panel=panel, scenario=scenario, seed=int(seed), subset=scope, model=label,
                             **fit[label], **info))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archived_panel', type=Path)
    parser.add_argument('archived_fits', type=Path)
    parser.add_argument('fresh_panel', type=Path)
    parser.add_argument('work', type=Path)
    parser.add_argument('--iqtree', default='iqtree')
    parser.add_argument('--jobs', type=int, default=11)
    parser.add_argument('--reuse', action='store_true')
    parser.add_argument('--panels', nargs='+', choices=['archived', 'fresh_fits'], default=['archived', 'fresh_fits'])
    parser.add_argument('--models', nargs='+', choices=['gy', 'mgk', 'ecmk07'], default=['gy', 'mgk', 'ecmk07'])
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument('--seeds', nargs='+', type=int)
    args = parser.parse_args()
    work = args.work.resolve()
    archived = args.archived_fits.resolve()
    sources = {'archived': (args.archived_panel.resolve(), (101, 202, 303)),
               'fresh_fits': (args.fresh_panel.resolve(), (411, 522, 633))}
    cases = [(panel, folder / 'PEPC' / f'{scenario}-{seed}') for panel in args.panels
             for folder, seeds in (sources[panel],) for scenario in args.scenarios for seed in (args.seeds or seeds)]
    if not args.reuse:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            jobs = [pool.submit(fit_run, panel, rep, label, args.iqtree, work)
                    for panel, rep in cases for label in args.models]
            errors = []
            for job in as_completed(jobs):
                try:
                    job.result()
                except RuntimeError as e:
                    print(e, file=sys.stderr, flush=True)
                    errors.append(str(e))
            if errors:
                raise RuntimeError('\n'.join(errors))
    rows = []
    for panel, rep in cases:
        paths = [(archived / rep.name, 'codon_model') if panel == 'archived' else (rep, 'default')]
        paths.extend((work / panel / rep.name / label, 'search') for label in args.models)
        rows.extend(scored_case(panel, rep, paths, ['hybrid', *args.models]))
    pd.DataFrame(rows).to_csv(work / 'summary.tsv', index=False, sep='\t', float_format='%.12g')


if __name__ == '__main__':
    main()
