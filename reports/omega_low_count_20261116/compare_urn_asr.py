"""Compare two urn expectations after each saved codon-model fit on PEPC."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from asr_bootstrap import omega_score
from compare_codon_models import MODELS, SCENARIOS
from pilot import EXPECTED, KEY, OBS
from urn_accuracy import load_search
from urn_empirical_routes import crossfit_groups, estimate_group, parent_evidence, validate_rates
from urn_parent_opportunity import load_fitted_events, metrics, urn_expected


def urn_expectations(pairs, training_branches, model, tips, aa, observed):
    membership, parent, parent_rate, routes = parent_evidence(model, tips, aa, training_branches)
    totals = [channel.sum(axis=1) for channel in observed]
    np.testing.assert_allclose(routes[training_branches].sum(axis=1), totals[0][training_branches],
                               rtol=1e-10, atol=1e-12)

    # For complete tips, --asrv each has one common profile per destination.
    # Its Dirichlet concentration defaults to one count per site.
    standard = np.column_stack([
        urn_expected(pairs, np.broadcast_to(channel.sum(axis=0) + 1., channel.shape), mass)
        for channel, mass in zip(observed, totals)
    ])
    empirical = np.full_like(standard, np.nan)
    inputs = observed, totals, membership, parent, parent_rate, routes, model.lengths
    for keep, train in crossfit_groups(pairs, training_branches):
        result = estimate_group(pairs[keep], train, *inputs, extra=False, only_empirical_routes=True)
        empirical[keep] = np.column_stack(result['empirical_routes'])
    assert np.isfinite(standard).all() and np.isfinite(empirical).all()
    return {'urn_each': standard, 'empirical_routes_crossfit': empirical}


def analyze(task):
    panel, rep, label, source, outdir, report, tree_path, exclude_known_foreground = task
    codon, meta = load_search(source, outdir)
    if meta['parameter_source']['model'] != MODELS[label]:
        raise ValueError(f'Unexpected fitted model in {source}.')
    frame, _, model, tips, aa, observed, _ = load_fitted_events(rep, codon, meta, report, tree_path)
    pairs = frame[KEY].to_numpy()
    training_branches = np.unique(pairs)
    if exclude_known_foreground:
        foreground_ids = np.loadtxt(rep / 'sim/sim_foreground_branch.txt', dtype=int, ndmin=1)
        for i in (1, 2):
            np.testing.assert_array_equal(frame[f'foreground_{i}'].eq(1), np.isin(frame[f'branch_id_{i}'], foreground_ids))
        training_branches = training_branches[~np.isin(training_branches, foreground_ids)]
    counts = {'codon_model': frame[EXPECTED].to_numpy(),
              **urn_expectations(pairs, training_branches, model, tips, aa, observed)}
    obs = frame[OBS].to_numpy()
    common = np.logical_and.reduce([np.isfinite(omega_score(obs, expected)) for expected in counts.values()])
    # Use simulated foreground labels to score the biased subset after estimation.
    foreground = (frame['foreground_1'].eq(1) & frame['foreground_2'].eq(1)).to_numpy()
    scopes = {'all': np.ones(len(frame), dtype=bool), 'two_foreground': foreground}
    scenario, seed = rep.name.rsplit('-', 1)
    print(f'{panel}/{rep.name}/{label}: replayed {len(frame)} pairs from {len(training_branches)} training branches',
          flush=True)
    return [dict(panel=panel, scenario=scenario, seed=int(seed), model=label, subset=tag,
                 method=method, **metrics(frame, expected, subset, common))
            for tag, subset in scopes.items() if subset.any() for method, expected in counts.items()]


def fit_tasks(args):
    panels = {'archived': (args.archived_panel, (101, 202, 303)),
              'fresh_fits': (args.fresh_panel, (411, 522, 633))}
    for panel, (folder, seeds) in panels.items():
        for seed in seeds:
            if args.seeds and seed not in args.seeds:
                continue
            for scenario in args.scenarios:
                rep = folder / 'PEPC' / f'{scenario}-{seed}'
                for label in args.models:
                    tree_path = rep / 'sim/sim_true_asr.treefile'
                    if label != 'hybrid':
                        source = args.alternate_fits / panel / rep.name / label
                        report, = (source / 'iqtree').glob('*.iqtree')
                        outdir = 'search'
                    elif panel == 'fresh_fits':
                        source, outdir = rep, 'default'
                        report, = (rep / 'iqtree-default').glob('*.iqtree')
                    else:
                        source, outdir = args.archived_fits / rep.name, 'codon_model'
                        report, tree_path = source / 'fit.iqtree', source / 'tree.nwk'
                    yield panel, rep, label, source, outdir, report, tree_path


def fit_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('archived_panel', type=Path)
    parser.add_argument('archived_fits', type=Path)
    parser.add_argument('fresh_panel', type=Path)
    parser.add_argument('alternate_fits', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--jobs', type=int, default=6)
    return parser


def main():
    parser = fit_parser(__doc__)
    parser.add_argument('--exclude-known-foreground', action='store_true',
                        help='Reproduce the comparison that held out the true simulated foreground from every fit.')
    args = parser.parse_args()
    validate_rates()
    tasks = [(*task, args.exclude_known_foreground) for task in fit_tasks(args)]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = [row for batch in pool.map(analyze, tasks) for row in batch]
    pd.DataFrame(rows).to_csv(args.output, sep='\t', index=False, float_format='%.12g')


if __name__ == '__main__':
    main()
