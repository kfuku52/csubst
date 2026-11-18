"""Compare search expectations on simulations from unscaled ECMK07 exchangeabilities."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from csubst import genetic_code, parser_iqtree, parser_misc
from csubst.main_simulate import (get_nonsynonymous_codon_substitution_index, get_pyvolve_codon_order,
                                  get_synonymous_codon_substitution_index, get_total_Q)

from asr_bootstrap import omega_score
from pilot import EXPECTED, KEY, OBS
from urn_accuracy import load_search
from urn_parent_opportunity import metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'simulate_sensitivity_20261116'))
from run import CONDITIONS, SEARCH, prepare, run_cli, simulate


MODELS = ('ECMK07+F', 'ECMK07_GY0K+F', 'GY+F', 'MGK+F3X4')
SCENARIOS = ('baseline', 'rate_mixture', 'biased')
SEEDS = (741, 852, 963)


def background_omega(sources):
    path = sources / 'PEPC.alignment.fa.iqtree'
    codons = get_pyvolve_codon_order()
    g = parser_iqtree.read_iqtree({'path_iqtree_iqtree': str(path), 'codon_orders': codons, 'float_type': np.float64})
    assert g['substitution_model'].split('+')[0] == 'ECMK07'
    code = genetic_code.get_codon_table(1)
    syn = get_synonymous_codon_substitution_index({'codon_table': code}, codons)
    nonsyn = get_nonsynonymous_codon_substitution_index(syn)
    exchange = parser_misc.read_exchangeability_matrix('substitution_matrix/ECMunrest.dat', codons)
    pi = g['equilibrium_frequency']
    q = parser_misc.exchangeability2Q(exchange, pi / pi.sum())
    return get_total_Q(q, nonsyn) / get_total_Q(q, syn)


def make_case(sources, rep, scenario, seed, omega):
    rep.mkdir()
    condition = replace(CONDITIONS[scenario], omega=omega, foreground_omega=omega)
    simulate(rep, 'PEPC', sources, condition, seed)


def fitted_model(base, scenario):
    return base + '+R4' if scenario == 'rate_mixture' else base


def fit_case(rep, scenario, seed, base, iqtree):
    root = rep / base
    root.mkdir()
    start = perf_counter()
    run_cli(root, 'codon.stdout', [
        'search', '--alignment_file', rep / 'sim/sim.fa', '--rooted_tree_file', rep / 'sim/sim_true_asr.treefile',
        '--iqtree_exe', iqtree, '--iqtree_model', fitted_model(base, scenario), '--iqtree_outdir', 'iqtree',
        '--expectation_method', 'codon_model', '--outdir', 'codon', '--random_seed', seed, *SEARCH,
    ])
    (root / 'codon.wall').write_text(f'{perf_counter() - start:.6f}\n')
    print('Fitted', rep.name, base, flush=True)


def urn_case(rep, seed, base, method):
    # Keep the historical output name for the four-fold fit. Omit --asrv for
    # the pooled run so its provenance also verifies the search default.
    asrv_args = ['--asrv', 'empirical'] if method == 'urn' else []
    root = rep / base
    report, = (root / 'iqtree').glob('*.iqtree')
    fit_args = [token for ext in ('treefile', 'state', 'rate', 'iqtree', 'log')
                for token in ('--iqtree_' + ext, report.with_suffix('.' + ext))]
    start = perf_counter()
    run_cli(root, method + '.stdout', [
        'search', '--alignment_file', rep / 'sim/sim.fa', '--rooted_tree_file', rep / 'sim/sim_true_asr.treefile',
        *fit_args, '--expectation_method', 'urn', *asrv_args, '--outdir', method, '--random_seed', seed, *SEARCH,
    ])
    (root / (method + '.wall')).write_text(f'{perf_counter() - start:.6f}\n')
    print('Compared', rep.name, base, method, flush=True)


def comparison(rep, scenario, base, truth):
    root = rep / base
    codon, codon_meta = load_search(root, 'codon')
    assert codon_meta['parameter_source']['model'] == fitted_model(base, scenario)
    keep = codon[EXPECTED].notna().all(axis=1)
    frame = codon.loc[keep].merge(truth, on=KEY, how='left', validate='one_to_one')
    np.testing.assert_array_equal(frame[KEY], codon.loc[keep, KEY])
    counts = {'codon_model': codon.loc[keep, EXPECTED].to_numpy()}
    for method in ('urn', 'empirical_all'):
        urn, urn_meta = load_search(root, method)
        provenance = json.loads((root / method / 'csubst_urn_provenance.json').read_text())
        settings = provenance['empirical_urn']
        assert settings['n_draw_totals'] == 'pooled_branch_n_plus_s'
        if method == 'urn':
            assert provenance['asrv'] == 'empirical' and len(settings['folds']) == 10
        else:
            assert provenance['asrv'] == 'empirical_all'
            assert settings['training_scope'] == 'all_eligible_branches' and not settings['folds']
            assert len(set(provenance['training_branch_ids'])) == 138
        for key in ('rates', 'weights', 'branch_lengths'):
            assert codon_meta[key] == urn_meta[key]
        assert codon_meta['parameter_source']['generator_sha256'] == urn_meta['parameter_source']['generator_sha256']
        np.testing.assert_array_equal(codon[KEY], urn[KEY])
        np.testing.assert_allclose(codon[OBS], urn[OBS], rtol=1e-10, atol=6e-12)
        np.testing.assert_array_equal(keep, urn[EXPECTED].notna().all(axis=1))
        counts[method] = urn.loc[keep, EXPECTED].to_numpy()
    return frame, counts


def score_case(rep, scenario, seed):
    truth = pd.read_csv(rep / 'sim/sim_expected_omega.tsv', sep='\t')
    fits = {base: comparison(rep, scenario, base, truth) for base in MODELS}
    reference = fits[MODELS[0]][0]
    common = np.ones(len(reference), dtype=bool)
    for frame, counts in fits.values():
        np.testing.assert_array_equal(frame[KEY], reference[KEY])
        for expected in counts.values():
            common &= np.isfinite(omega_score(frame[OBS].to_numpy(), expected))
    select = (reference['foreground_1'].eq(1) & reference['foreground_2'].eq(1)
              if scenario == 'biased' else np.ones(len(reference), dtype=bool))
    return [dict(scenario=scenario, seed=seed, model=base, fit_model=fitted_model(base, scenario), method=method,
                 **metrics(frame, expected, select, common)) for base, (frame, counts) in fits.items()
            for method, expected in counts.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('--stage', choices=('fit', 'urn', 'empirical_all', 'score'), required=True)
    parser.add_argument('--iqtree', default='iqtree')
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    work = args.work.resolve()
    cases = [(work / f'{scenario}-{seed}', scenario, seed) for scenario in SCENARIOS for seed in SEEDS]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        if args.stage == 'fit':
            sources = prepare(work, 'PEPC')
            omega = background_omega(sources)
            (work / 'background-omega.txt').write_text(f'{omega:.17g}\n')
            for job in [pool.submit(make_case, sources, *case, omega) for case in cases]:
                job.result()
            for job in [pool.submit(fit_case, *case, base, args.iqtree) for case in cases for base in MODELS]:
                job.result()
        else:
            if args.stage in ('urn', 'empirical_all'):
                methods = ('urn', 'empirical_all') if args.stage == 'urn' else ('empirical_all',)
                for method in methods:
                    for job in [pool.submit(urn_case, rep, seed, base, method)
                                for rep, _, seed in cases for base in MODELS]:
                        job.result()
            jobs = [pool.submit(score_case, *case) for case in cases]
            rows = [row for job in jobs for row in job.result()]
            pd.DataFrame(rows).to_csv(work / 'summary.tsv', sep='\t', index=False, float_format='%.12g')


if __name__ == '__main__':
    main()
