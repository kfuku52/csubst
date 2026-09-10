#!/usr/bin/env python3
"""Reproduce long-tail set sensitivity and a known-parameter count-null study.

This is a statistical comparison, not a speed benchmark or an end-to-end
phylogenetic simulation. Legacy quantile mapping is transcribed from dd37bee;
the new frozen and independent-null transformations use production code.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from csubst.longtail import QuantileMap  # noqa: E402
from csubst.omega_statistics import _calc_raw_omega, _calibrate_dsc_matrix  # noqa: E402
from csubst.randomness import generator  # noqa: E402

METHODS = ('Uncalibrated', 'Legacy empirical', 'Frozen empirical', 'Independent null')
COLORS = {'Uncalibrated': '#4D4D4D', 'Legacy empirical': '#D55E00',
          'Frozen empirical': '#009E73', 'Independent null': '#0072B2'}
MARKERS = {'Uncalibrated': 'o', 'Legacy empirical': 's', 'Frozen empirical': 'D', 'Independent null': '^'}


def legacy_dsc(n, s):
    """Vectorized exact legacy formula, rows x draws; includes the one-row case."""
    n, s = np.asarray(n, float), np.asarray(s, float)
    q = (stats.rankdata(s, axis=0) - .5) / s.shape[0]
    sorted_n = np.sort(n, axis=0)
    ix = q * (s.shape[0] - 1)
    lo, hi = np.floor(ix).astype(int), np.ceil(ix).astype(int)
    a, b = np.take_along_axis(sorted_n, lo, axis=0), np.take_along_axis(sorted_n, hi, axis=0)
    weight = ix - lo
    mapped = np.where(weight >= .5, b - (b - a) * (1. - weight), a + (b - a) * weight)
    return np.maximum(s, mapped)


def omega(n, s):
    return _calc_raw_omega(n, s, 1e-12)


def upper_p(observed, null):
    return (1. + np.count_nonzero(null >= observed)) / (null.size + 1.)


def interval(k, n):
    return (0. if k == 0 else stats.beta.ppf(.025, k, n - k + 1),
            1. if k == n else stats.beta.ppf(.975, k + 1, n - k))


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'savefig.facecolor': 'white',
                         'axes.labelcolor': '#333333', 'text.color': '#222222'})


def save(fig, out, name):
    for suffix in ('png', 'pdf'):
        fig.savefig(out / (name + '.' + suffix), dpi=190, bbox_inches='tight')
    plt.close(fig)


def sensitivity(out, seed):
    # Fixed focal rates; display membership changes but reference does not.
    n_sets = [[10.], [10., 1., 1., 1.], [10., 100., 100., 100.]]
    names = ['Focal row only', '+ 3 low-N rows', '+ 3 high-N rows']
    frozen = QuantileMap.fit([1., 1., 1., 10.], [1., 1., 1., 1.])
    rng = generator(seed, 'set-example-fit')
    independent = QuantileMap.fit(rng.poisson(8., 1000) / 8., rng.poisson(8., 1000) / 8.)
    records = []
    for label, n in zip(names, n_sets):
        s = np.ones(len(n))
        records.extend([
            dict(case=label, method='Uncalibrated', focal_omega=10.),
            dict(case=label, method='Legacy empirical', focal_omega=10. / legacy_dsc(np.array(n)[:, None], s[:, None])[0, 0]),
            dict(case=label, method='Frozen empirical', focal_omega=10. / frozen.apply([1.])[0]),
            dict(case=label, method='Independent null', focal_omega=10. / independent.apply([1.])[0]),
        ])
    frame = pd.DataFrame(records)
    frame.to_csv(out / 'set_sensitivity.tsv', sep='\t', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.1), constrained_layout=True)
    for j, method in enumerate(METHODS):
        vals = frame[frame.method == method].focal_omega
        axes[0].plot(np.arange(3) + (j - 1.5) * .035, vals, marker=MARKERS[method],
                     color=COLORS[method], label=method, linewidth=1.4,
                     linestyle='--' if method == 'Frozen empirical' else '-')
    axes[0].set(xticks=np.arange(3), xticklabels=names, yscale='log', ylabel='Focal omegaC (log scale)',
                title='A  Same focal row, different displayed sets', ylim=(.07, 15))
    axes[0].text(.02, .05, 'Focal dNC = 10; dSC = 1\nFrozen references held constant', transform=axes[0].transAxes, fontsize=9)
    for i, scale in enumerate([1., 10.]):
        n, s = np.array([1., 2., 3.]) * scale, np.ones(3)
        corrected = omega(n, legacy_dsc(n[:, None], s[:, None])[:, 0])
        axes[1].plot([1, 2, 3], n, marker='o', color=COLORS['Uncalibrated'],
                     linestyle='-' if i else '--', label='Raw, N x{}'.format(int(scale)))
        axes[1].plot([1, 2, 3], corrected, marker='s', markersize=8 if i else 4,
                     markerfacecolor='none' if i else COLORS['Legacy empirical'],
                     color=COLORS['Legacy empirical'], linestyle='-' if i else '--',
                     label='Legacy, N x{}'.format(int(scale)))
    axes[1].set(xticks=[1, 2, 3], xlabel='Row', ylabel='omegaC (log scale)', yscale='log',
                title='B  A common N increase can disappear')
    axes[1].legend(fontsize=8, loc='center left')
    axes[0].legend(loc='lower left', bbox_to_anchor=(.02, .17), ncol=1, fontsize=8)
    fig.suptitle('Set dependence is distinct from false-positive-rate calibration', fontsize=14)
    save(fig, out, 'set_sensitivity')


def simulate(out, seed, replicates, fit_draws, test_draws):
    records = []
    checks = generator(seed, 'implementation-check')
    n, s = checks.poisson(8., (8, 19)) / 8., checks.poisson(2., (8, 19)) / 2.
    np.testing.assert_array_equal(legacy_dsc(n, s), _calibrate_dsc_matrix(n, s))
    for mu_s in [.5, 8.]:
        for rows in [1, 4, 32]:
            for fraction in [0., .5, 1.]:
                hits = np.zeros((2, len(METHODS)), dtype=int)
                threshold_hits = hits.copy()
                other_signal = int(round(fraction * (rows - 1)))
                for rep in range(replicates):
                    # Purpose-separated streams. The same focal data and null
                    # draws are retained as other rows/signals are added.
                    key = (float(mu_s), rep)
                    n_test = generator(seed, 'test-N', *key).poisson(8., (rows, test_draws)) / 8.
                    s_test = generator(seed, 'test-S', *key).poisson(mu_s, (rows, test_draws)) / mu_s
                    old_test = omega(n_test, legacy_dsc(n_test, s_test))[0]
                    raw_test = omega(n_test[0], s_test[0])
                    fit_n = generator(seed, 'fit-N', *key).poisson(8., fit_draws) / 8.
                    fit_s = generator(seed, 'fit-S', *key).poisson(mu_s, fit_draws) / mu_s
                    mapping = QuantileMap.fit(fit_n, fit_s)
                    new_test = omega(n_test[0], mapping.apply(s_test[0]))
                    tests = [raw_test, old_test, raw_test if rows == 1 else old_test, new_test]
                    other_means = np.full(rows - 1, 8.)
                    other_means[:other_signal] *= 4.
                    other_n = generator(seed, 'other-N', *key).poisson(other_means) / 8.
                    obs_s = generator(seed, 'obs-S', *key).poisson(mu_s, rows) / mu_s
                    for alternative in [0, 1]:
                        focal_n = generator(seed, 'focal-N', *key).poisson(32. if alternative else 8.) / 8.
                        obs_n = np.r_[focal_n, other_n]
                        raw = float(omega(focal_n, obs_s[0]))
                        old = float(omega(obs_n, legacy_dsc(obs_n[:, None], obs_s[:, None])[:, 0])[0])
                        frozen = float(omega(obs_n, QuantileMap.fit(obs_n, obs_s).apply(obs_s))[0])
                        new = float(omega(focal_n, mapping.apply(obs_s[:1])[0]))
                        for mi, (observed, null) in enumerate(zip([raw, old, frozen, new], tests)):
                            hits[alternative, mi] += upper_p(observed, null) <= .05
                            threshold_hits[alternative, mi] += observed >= 5.
                for alternative in [0, 1]:
                    for mi, method in enumerate(METHODS):
                        k = int(hits[alternative, mi])
                        low, high = interval(k, replicates)
                        records.append(dict(mu_N=8., mu_S=mu_s, rows=rows,
                                            other_signal_fraction=fraction, other_signal_rows=other_signal,
                                            focal_multiplier=4 if alternative else 1, method=method,
                                            replicates=replicates, rejections=k, rejection_rate=k / replicates,
                                            ci_low=low, ci_high=high,
                                            omega_ge5=int(threshold_hits[alternative, mi]) / replicates))
                print('Completed mu_S={}, rows={}, other-signal fraction={}'.format(mu_s, rows, fraction), flush=True)
    result = pd.DataFrame(records)
    stable = result[result.method.isin(['Uncalibrated', 'Independent null'])]
    assert stable.groupby(['mu_S', 'focal_multiplier', 'method']).rejections.nunique().max() == 1
    result.to_csv(out / 'count_null_summary.tsv', sep='\t', index=False)
    return result


def comparison_figures(result, out, replicates, fit_draws, test_draws):
    # Frozen empirical overlaps legacy for full tables with >=2 finite rows;
    # avoid duplicating identical lines in these power/FPR panels.
    methods = ['Uncalibrated', 'Legacy empirical', 'Independent null']
    fpr_top = max(8., float(result[(result.rows == 32) & (result.focal_multiplier == 1)].ci_high.max()) * 100 + 1.)
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4), constrained_layout=True)
    for i, mu_s in enumerate([.5, 8.]):
        for j, effect in enumerate([1, 4]):
            ax = axes[i, j]
            for mi, method in enumerate(methods):
                data = result[(result.mu_S == mu_s) & (result.rows == 32) & (result.focal_multiplier == effect) & (result.method == method)]
                x, y = data.other_signal_fraction.to_numpy() * 100, data.rejection_rate.to_numpy() * 100
                ax.errorbar(x + (mi - 1) * 1.5, y,
                            yerr=[y - data.ci_low.to_numpy() * 100, data.ci_high.to_numpy() * 100 - y],
                            color=COLORS[method], marker=MARKERS[method], capsize=3, label=method)
            if effect == 1:
                ax.axhline(5., color='#777777', linestyle=':', linewidth=1)
            ax.set(xlabel='Signal among other rows (%)', ylabel='Rejection rate (%)',
                   xticks=[0, 50, 100], ylim=(-.3, fpr_top if effect == 1 else 101),
                   title='{}  {} | expected S = {}'.format('ABCD'[i * 2 + j], 'Focal null (FPR)' if effect == 1 else 'Focal N x4 (power)', mu_s))
    axes[0, 0].legend(loc='upper center', fontsize=9)
    fig.suptitle('Known-parameter Poisson counts: 32 rows, P <= 0.05\n{} independent datasets / condition; {} fit + {} test draws; exact 95% binomial intervals'.format(replicates, fit_draws, test_draws), fontsize=13)
    save(fig, out, 'count_null_comparison')
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.1), constrained_layout=True)
    for ax, mu_s in zip(axes, [.5, 8.]):
        for mi, method in enumerate(METHODS):
            data = result[(result.mu_S == mu_s) & (result.focal_multiplier == 4) & (result.other_signal_fraction == 1.) & (result.method == method)]
            x, y = np.arange(3) + (mi - 1.5) * .04, data.rejection_rate.to_numpy() * 100
            ax.errorbar(x, y, yerr=[y - data.ci_low.to_numpy() * 100, data.ci_high.to_numpy() * 100 - y],
                        color=COLORS[method], marker=MARKERS[method], capsize=2, label=method,
                        linestyle='--' if method == 'Frozen empirical' else '-')
        ax.set(xticks=np.arange(3), xticklabels=['1', '4', '32'], xlabel='Rows in the analysis',
               ylabel='Power (%)', ylim=(-1, 101), title='Expected S = {}'.format(mu_s))
    axes[0].legend(loc='center left', fontsize=8)
    fig.suptitle('Widespread signal: all rows have N x4\nFrozen empirical skips a one-row reference; it matches legacy for full tables with >=2 finite rows', fontsize=13)
    save(fig, out, 'small_sample_power')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports' / 'longtail_20260910')
    parser.add_argument('--replicates', type=int, default=1000)
    parser.add_argument('--fit-draws', type=int, default=1000)
    parser.add_argument('--test-draws', type=int, default=999)
    parser.add_argument('--seed', type=int, default=20260910)
    args = parser.parse_args()
    if min(args.replicates, args.fit_draws, args.test_draws) < 2:
        parser.error('replicates and draw counts must be >= 2')
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    sensitivity(args.out, args.seed)
    result = simulate(args.out, args.seed, args.replicates, args.fit_draws, args.test_draws)
    comparison_figures(result, args.out, args.replicates, args.fit_draws, args.test_draws)
    metadata = dict(seed=args.seed, replicates=args.replicates, fit_draws=args.fit_draws,
                    test_draws=args.test_draws, python=platform.python_version(), numpy=np.__version__,
                    scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                    model='Independent Poisson counts, known E_N=8 and E_S in {0.5,8}; no tree/ASR/model refitting.',
                    sources={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                             ['csubst/longtail.py', 'csubst/omega_statistics.py', '.github/scripts/longtail_comparison.py']})
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
