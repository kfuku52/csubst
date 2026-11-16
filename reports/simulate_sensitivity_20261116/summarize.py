"""Summarize simulation expectations, realized counts, and fitted search outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from csubst import ete, tree
from run import CONDITIONS, MODEL, PEPC_CONDITIONS, PEPC_SEEDS, PGK_SEEDS, UNIFORM_REFIT


PAIR = ['branch_id_1', 'branch_id_2']
FIT = ['OCNany2spe', 'OCSany2spe', 'ECNany2spe', 'ECSany2spe']
SIM = ['muNany2spe_sim', 'muSany2spe_sim', 'muNany2spe_null', 'muSany2spe_null',
       'OCNany2spe_true', 'OCSany2spe_true']
OMEGA = 'omegaCany2spe'
GROUP = ['dataset', 'scenario', 'variant', 'pair_group']


def read_pairs(rep, variant):
    fits = rep / variant
    if json.loads((fits / 'csubst_search_run.json').read_text())['status'] != 'complete':
        raise ValueError(f'Incomplete search in {fits}')
    meta = json.loads((fits / 'csubst_endpoint_model.json').read_text())['codon']
    expected_model = MODEL + ('+R4' if variant == 'default' else '')
    if meta['parameter_source']['model'] != expected_model:
        raise ValueError(f'Unexpected fitted model in {fits}')
    axes = tree.add_numerical_node_labels(ete.PhyloNode(str(rep / 'sim/sim_true_asr.treefile'), format=1))
    names = {ete.get_prop(node, 'numerical_label'): node.name for node in axes.traverse()}
    root_ids = [ete.get_prop(child, 'numerical_label') for child in ete.get_children(axes)]
    generated = pd.read_csv(rep / 'sim/sim_expected_omega.tsv', sep='\t')
    fitted = pd.read_csv(fits / 'csubst_cb_2.tsv', sep='\t')
    joined = generated.merge(
        fitted[PAIR + FIT + [OMEGA]], on=PAIR, how='outer', indicator=True, validate='one_to_one',
    )
    if not joined.pop('_merge').eq('both').all():
        raise ValueError(f'Pair set differs in {fits}')
    for i in (1, 2):
        if not joined[f'branch_id_{i}'].map(names).eq(joined[f'branch_name_{i}']).all():
            raise ValueError(f'Branch mapping differs in {fits}')
    joined['foreground'] = joined.foreground_1 + joined.foreground_2
    joined['valid'] = joined[FIT].notna().all(axis=1)
    np.testing.assert_array_equal(joined.valid, ~joined[PAIR].isin(root_ids).any(axis=1))
    return joined, meta


def count_frame(frame):
    n_null, s_null = [frame[c].to_numpy() for c in SIM[2:4]]
    obs_n, obs_s, fit_n, fit_s = [frame[c].to_numpy() for c in FIT]
    real_n, real_s = [frame[c].to_numpy() for c in SIM[4:]]
    # Precision of the TSV is 12 decimal places. Only compute ratio errors
    # among pairs with realized N and S events and positive fitted counts.
    common = (real_n > 0) & (real_s > 0) & (obs_n > 0) & (obs_s > 0) & (fit_n > 0) & (fit_s > 0)
    estimated = frame[OMEGA].to_numpy()
    q_ratio = frame['omegaCany2spe_expected'].to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        fixed = (obs_n / obs_s) * (s_null / n_null)
        diff_asr = np.log(obs_n[common] / obs_s[common]) - np.log(real_n[common] / real_s[common])
        diff_ec = np.log(fit_s[common] / s_null[common]) - np.log(fit_n[common] / n_null[common])
        diff_sampling = np.log(frame['omegaCany2spe_true'].to_numpy()[common]) - np.log(q_ratio[common])
    calls = (obs_n >= 2.0) & (estimated >= 5.0)
    unknown_s = real_s == 0
    return {
        'pairs': len(frame), **frame[SIM + FIT].sum().to_dict(),
        'true_N0': int((real_n == 0).sum()), 'true_S0': int(unknown_s.sum()),
        'true_both0': int(((real_n == 0) & unknown_s).sum()),
        'true_S0_Npositive': int(((real_n > 0) & unknown_s).sum()),
        'finite_fitted_omega': int(np.isfinite(estimated).sum()),
        'finite_fitted_omega_with_true_S0': int((np.isfinite(estimated) & unknown_s).sum()),
        'heuristic_calls': int(calls.sum()), 'heuristic_calls_with_true_S0': int((calls & unknown_s).sum()),
        'heuristic_calls_true_N_below2': int((calls & (real_n < 2)).sum()),
        'pairs_with_positive_truth_and_fit_counts': int(common.sum()),
        'abs_log_asr': np.abs(diff_asr).sum(), 'abs_log_ec': np.abs(diff_ec).sum(),
        'abs_log_sampling': np.abs(diff_sampling).sum(),
        'abs_error_OCN': np.abs(obs_n - real_n).sum(),
        'abs_error_OCS': np.abs(obs_s - real_s).sum(),
        'abs_error_ECN': np.abs(fit_n - n_null).sum(),
        'abs_error_ECS': np.abs(fit_s - s_null).sum(),
        'fixed_omega_finite': int(np.isfinite(fixed).sum()),
    }


def ratios(frame):
    with np.errstate(divide='ignore', invalid='ignore'):
        return frame.assign(
            q_omega=lambda f: f[SIM[0]] * f[SIM[3]] / (f[SIM[1]] * f[SIM[2]]),
            true_omega=lambda f: f[SIM[4]] * f[SIM[3]] / (f[SIM[5]] * f[SIM[2]]),
            fixed_null_omega=lambda f: f[FIT[0]] * f[SIM[3]] / (f[FIT[1]] * f[SIM[2]]),
            fitted_omega=lambda f: f[FIT[0]] * f[FIT[3]] / (f[FIT[1]] * f[FIT[2]]),
            ecn_over_q=lambda f: f[FIT[2]] / f[SIM[2]],
            ecs_over_q=lambda f: f[FIT[3]] / f[SIM[3]],
            ocn_over_truth=lambda f: f[FIT[0]] / f[SIM[4]],
            ocs_over_truth=lambda f: f[FIT[1]] / f[SIM[5]],
        )


def summarize(work):
    rows, diagnostics, strata = [], [], []
    for dataset in ('PGK', 'PEPC'):
        for rep in sorted((work / dataset).glob('*-*')):
            scenario, seed = rep.name.rsplit('-', 1)
            for variant in ('default', 'uniform'):
                manifest = rep / variant / 'csubst_search_run.json'
                if not manifest.exists() or json.loads(manifest.read_text())['status'] != 'complete':
                    continue
                joined, meta = read_pairs(rep, variant)
                labels = dict(dataset=dataset, scenario=scenario, seed=int(seed), variant=variant)
                valid = joined.loc[joined.valid]
                for group, mask in (
                    ('all', joined.valid), ('foreground', joined.valid & joined.foreground.eq(2)),
                    ('background', joined.valid & joined.foreground.eq(0)),
                    ('mixed', joined.valid & joined.foreground.eq(1)),
                ):
                    rows.append(dict(**labels, pair_group=group, **count_frame(joined.loc[mask])))
                rates, weights = np.asarray(meta['rates']), np.asarray(meta['weights'])
                root_pairs = joined.loc[~joined.valid]
                diagnostics.append(dict(
                    **labels, sites=CONDITIONS[scenario].sites, num_pairs=len(joined),
                    num_masked_pairs=len(root_pairs), masked_q_N=root_pairs[SIM[2]].sum(),
                    masked_q_S=root_pairs[SIM[3]].sum(), q_N=joined[SIM[2]].sum(), q_S=joined[SIM[3]].sum(),
                    multiplier=meta['parameter_source']['omega'], fitted_rate_cv=np.sqrt(
                        np.average((rates - np.average(rates, weights=weights))**2, weights=weights)
                    ) / np.average(rates, weights=weights),
                    **{f'rate{i + 1}': r for i, r in enumerate(rates)},
                    **{f'weight{i + 1}': w for i, w in enumerate(weights)},
                ))
                edges = [-np.inf, 0.1, 1, 5, np.inf]
                for bounds, indices in valid.groupby(pd.cut(valid[SIM[3]], edges), observed=True).groups.items():
                    strata.append(dict(**labels, stratum=str(bounds), **count_frame(valid.loc[indices])))
    expected = {(dataset, scenario, seed, 'default')
                for dataset, scenarios, seeds in (('PGK', CONDITIONS, PGK_SEEDS), ('PEPC', PEPC_CONDITIONS, PEPC_SEEDS))
                for scenario in scenarios for seed in seeds}
    expected.update(('PGK', scenario, seed, 'uniform') for scenario in UNIFORM_REFIT for seed in PGK_SEEDS)
    actual = {(row['dataset'], row['scenario'], row['seed'], row['variant']) for row in diagnostics}
    if expected != actual:
        raise ValueError(f'Incomplete panel or unexpected runs: {sorted(expected.symmetric_difference(actual))}')
    frame = pd.DataFrame(rows)
    stratified = pd.DataFrame(strata)
    groups = ['dataset', 'scenario', 'variant', 'stratum']
    replicate_totals = frame.groupby(GROUP, sort=False)['seed'].nunique().rename('replicates')
    pooled = frame.drop(columns='seed').groupby(GROUP, sort=False).sum().join(replicate_totals)
    return {
        'by_seed.tsv': ratios(frame),
        'pooled.tsv': ratios(pooled.reset_index()),
        'fit_diagnostics.tsv': pd.DataFrame(diagnostics),
        'by_expected_S.tsv': ratios(stratified.drop(columns='seed').groupby(groups, as_index=False).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, frame in summarize(args.work).items():
        frame.to_csv(args.output / filename, sep='\t', index=False, na_rep='NA', float_format='%.10g')


if __name__ == '__main__':
    main()
