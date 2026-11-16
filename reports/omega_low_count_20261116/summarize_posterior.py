"""Summarize posterior calibration and branch-dependence diagnostics."""

import argparse
from pathlib import Path

import pandas as pd

from pilot import OBS, TRUTH


KEYS = ['dataset', 'scenario', 'seed', 'branch_id_1', 'branch_id_2']


def group_summary(frame, group):
    table = frame.copy()
    table['truth_undefined'] = table[TRUTH].eq(0).all(axis=1)
    table['truth_defined'] = ~table.truth_undefined
    table['both_counts_covered'] = table.n_covered & table.s_covered
    table['ratio_available'] = table.ratio_covered.notna()
    table['ratio_covered'] = table.ratio_covered.eq(True)
    table['ratio_upper_unbounded'] = table.ratio_upper.eq(float('inf'))
    table['predicted_undefined'] = table.p_undefined
    table['brier_undefined'] = (table.p_undefined - table.truth_undefined)**2
    table['both_width_zero'] = table.n_upper.eq(table.n_lower) & table.s_upper.eq(table.s_lower)
    for channel, truth in zip(('n', 's'), TRUTH):
        error = table[f'{channel}_mean'] - table[truth]
        table[f'{channel}_abs_error'] = error.abs()
        table[f'{channel}_sq_error'] = error**2
        table[f'{channel}_width'] = table[f'{channel}_upper'] - table[f'{channel}_lower']
        rank = table[f'{channel}_pit']
        table[f'{channel}_pit_central'] = rank.between(.025, .975)
    totals = [
        'truth_undefined', 'truth_defined', 'both_counts_covered', 'ratio_available',
        'ratio_covered', 'ratio_upper_unbounded', 'both_width_zero',
        'n_covered', 's_covered', 'n_pit_central', 's_pit_central',
    ]
    averaged = [
        'predicted_undefined', 'brier_undefined', 'n_abs_error', 's_abs_error', 'n_width', 's_width',
        'n_mean', 's_mean', 'n_sq_error', 's_sq_error',
    ]
    grouped = table.groupby(group, sort=False)
    return grouped[totals].sum().join(grouped[averaged].mean()).join(grouped.size().rename('rows')).reset_index()


def factorization_differences(table):
    raw = table.loc[table.method.eq('factorized'), KEYS + OBS]
    methods = table.loc[table.method.isin(('shared_rate', 'joint_fitted'))].merge(
        raw, on=KEYS, suffixes=('', '_raw'), validate='many_to_one',
    )
    columns = ['delta_n', 'delta_s', 'abs_delta_n', 'abs_delta_s']
    methods['delta_n'] = methods.n_mean - methods[OBS[0] + '_raw']
    methods['delta_s'] = methods.s_mean - methods[OBS[1] + '_raw']
    methods['abs_delta_n'] = methods.delta_n.abs()
    methods['abs_delta_s'] = methods.delta_s.abs()
    grouped = methods.groupby(['relationship', 'method'])
    return grouped[columns].agg(['mean', 'max'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pairs', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.pairs, sep='\t')
    for method, data in frame.groupby('method'):
        if data.duplicated(KEYS).any():
            raise ValueError(f'Duplicate branch pairs for {method}.')
    overall = group_summary(frame, ['method'])
    by_stratum = group_summary(frame, ['s_bin', 'method'])
    by_scenario = group_summary(frame, ['dataset', 'scenario', 'method'])
    by_foreground = group_summary(frame, ['foreground_pair', 'method'])
    by_relationship = group_summary(frame, ['relationship', 'method'])
    differences = factorization_differences(frame)
    differences.columns = [f'{quantity}_{stat}' for quantity, stat in differences.columns]
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in (
        ('posterior_summary.tsv', overall), ('posterior_by_support.tsv', by_stratum),
        ('posterior_by_scenario.tsv', by_scenario), ('posterior_factorization.tsv', differences.reset_index()),
        ('posterior_by_foreground.tsv', by_foreground),
        ('posterior_by_relationship.tsv', by_relationship),
    ):
        table.to_csv(args.output / name, sep='\t', index=False, float_format='%.10g')


if __name__ == '__main__':
    main()
