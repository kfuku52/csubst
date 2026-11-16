"""Summarize the six simulated PEPC runs described in README.md."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PAIR = ['branch_id_1', 'branch_id_2']
STAGES = {
    'true': 'omegaCany2spe_true',
    'search_fixed': 'omegaCany2spe_search_fixed',
    'search': 'omegaCany2spe',
}
COUNTS = {
    'Q': ('muNany2spe_sim', 'muSany2spe_sim'),
    'true': ('OCNany2spe_true', 'OCSany2spe_true'),
    'search': ('OCNany2spe', 'OCSany2spe'),
}
NULL = ['muNany2spe_null', 'muSany2spe_null']
FITTED = ['ECNany2spe', 'ECSany2spe']


def omega(n, s, null_n, null_s):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((null_n > 0) & (null_s > 0), (n / null_n) / (s / null_s), np.nan)


def load_runs(runs):
    tables = []
    for scenario in ('null', 'biased'):
        for seed in (101, 202, 303):
            directory = runs / f'{scenario}-{seed}'
            manifest = directory / 'search/csubst_search_run.json'
            if json.loads(manifest.read_text())['status'] != 'complete':
                raise ValueError(f'Incomplete search: {manifest}')
            sim = pd.read_csv(directory / 'sim/sim_expected_omega.tsv', sep='\t')
            search = pd.read_csv(directory / 'search/csubst_cb_2.tsv', sep='\t')
            joined = sim.merge(
                search[PAIR + list(COUNTS['search']) + FITTED + ['omegaCany2spe']],
                on=PAIR, how='outer', validate='one_to_one', indicator=True,
            )
            if not (joined.pop('_merge') == 'both').all():
                raise ValueError(f'Different branch pair sets: {directory}')
            joined['scenario'] = 'background' if scenario == 'null' else scenario
            joined['seed'] = seed
            joined['num_fg_branches'] = joined.foreground_1 + joined.foreground_2
            joined['omegaCany2spe_search_fixed'] = omega(
                *(joined[name].to_numpy() for name in list(COUNTS['search']) + NULL)
            )
            tables.append(joined)
    return pd.concat(tables, ignore_index=True)


def summarize(groups, keys):
    ratios, counts = [], []
    for group, table in groups:
        labels = dict(zip(keys, group))
        totals = table.select_dtypes(include='number').sum()
        expected = table.omegaCany2spe_expected
        valid_expected = np.isfinite(expected) & (expected > 0)
        for stage, name in STAGES.items():
            value = table[name]
            finite = np.isfinite(value)
            positive = valid_expected & finite & (value > 0)
            log_error = np.log10(value[positive]) - np.log10(expected[positive])
            ratio_counts = list(COUNTS['true' if stage == 'true' else 'search'])
            denominator = FITTED if stage == 'search' else NULL
            ratios.append(dict(
                **labels, stage=stage, n_rows=len(table), n_expected_positive=valid_expected.sum(),
                expected_median=expected[valid_expected].median(),
                n_positive_comparison=positive.sum(),
                n_zero=(value == 0).sum(), n_infinite=np.isinf(value).sum(), n_na=value.isna().sum(),
                n_ocs_zero=(table[ratio_counts[1]] == 0).sum(),
                median_log10_error=log_error.median(), median_abs_log10_error=log_error.abs().median(),
                pooled_expected=omega(*(totals[name] for name in list(COUNTS['Q']) + NULL)).item(),
                pooled_value=omega(*(totals[name] for name in ratio_counts + denominator)).item(),
            ))
        for channel, names, null_name, fitted_name in zip(('N', 'S'), zip(*COUNTS.values()), NULL, FITTED):
            predicted, true, inferred = (table[name] for name in names)
            counts.append(dict(
                **labels, channel=channel, expected_total=predicted.sum(),
                true_total=true.sum(), search_total=inferred.sum(),
                null_total=totals[null_name], fitted_expected_total=totals[fitted_name],
                mae_true_vs_expected=(true - predicted).abs().mean(),
                mae_search_vs_expected=(inferred - predicted).abs().mean(),
                mae_search_vs_true=(inferred - true).abs().mean(),
            ))
    return pd.DataFrame(ratios), pd.DataFrame(counts)


def plot_foreground(ratios, by_seed, output):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = ['Q expectation', 'True ancestors', 'Search, fixed EC', 'Search, fitted EC']
    colors = ['#31688e', '#4c956c', '#c17c2b', '#b14762']
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.9), sharey=True, layout='constrained')
    for ax, scenario in zip(axes, ('background', 'biased')):
        pooled = ratios.query('scenario == @scenario and num_fg_branches == 2').set_index('stage')
        repeated = by_seed.query('scenario == @scenario and num_fg_branches == 2')
        ax.axhline(1, color='#555555', ls='--', linewidth=1)
        for x, (label, color, key) in enumerate(zip(labels, colors, ('Q', *STAGES))):
            if key == 'Q':
                value = pooled.pooled_expected.iloc[0]
                replicates = repeated.query('stage == "true"').pooled_expected
            else:
                value = pooled.loc[key, 'pooled_value']
                replicates = repeated.query('stage == @key').pooled_value
            ax.bar(x, value, color=color, width=0.65, alpha=0.62)
            ax.scatter(x + np.array([-0.14, 0, 0.14]), replicates, color=color,
                       edgecolor='white', linewidth=0.6, zorder=3)
        ax.set_xticks(range(len(labels)), labels, rotation=28, ha='right')
        ax.set_title({'background': 'Background Q', 'biased': '20% biased sites'}[scenario])
    axes[0].set_ylabel('ωC from pooled counts')
    fig.suptitle('PEPC foreground × foreground pairs')
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    table = load_runs(args.runs)
    args.output.mkdir(parents=True, exist_ok=True)
    keys = ['scenario', 'num_fg_branches']
    omega_summary, count_summary = summarize(table.groupby(keys), keys)
    seed_keys = ['scenario', 'seed', 'num_fg_branches']
    omega_by_seed, count_by_seed = summarize(table.groupby(seed_keys), seed_keys)
    for name, data in [
        ('omega_summary.tsv', omega_summary), ('count_summary.tsv', count_summary),
        ('omega_by_seed.tsv', omega_by_seed), ('count_by_seed.tsv', count_by_seed),
        ('paired_results.tsv.gz', table),
    ]:
        data.to_csv(args.output / name, sep='\t', index=False, na_rep='NA', float_format='%.12g')
    plot_foreground(omega_summary, omega_by_seed, args.output / 'foreground_omega.png')


if __name__ == '__main__':
    main()
