#!/usr/bin/env python3
"""Plot measured PEPC cost and independent fixed-model calibration results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--performance', type=Path, required=True)
    parser.add_argument('--accuracy', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    performance = json.loads(args.performance.read_text())['summary']
    accuracy = json.loads(args.accuracy.read_text())['modes']
    modes, labels = ['legacy_default', 'joint', 'bridge'], ['Legacy', 'Joint', 'Bridge']
    colors = ['#617080', '#257991', '#bc602d']
    x = np.arange(3)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), layout='constrained')
    fig.suptitle('CTMC scan: computational cost and conditional calibration', fontsize=15)
    for ax, key, scale, title, unit in [
        (axes[0, 0], 'process_seconds', 1, 'PEPC: complete process time', 'Seconds'),
        (axes[0, 1], 'peak_rss_bytes', 1024**3, 'PEPC: peak resident memory', 'GiB'),
    ]:
        values = np.array([performance[mode][key]['median']/scale for mode in modes])
        low = np.array([performance[mode][key]['minimum']/scale for mode in modes])
        high = np.array([performance[mode][key]['maximum']/scale for mode in modes])
        bars = ax.bar(x, values, color=colors, width=.6)
        ax.errorbar(x, values, yerr=[values-low, high-values], fmt='none', color='#222222', capsize=4)
        ax.bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=7)
        ax.set_ylim(0, high.max()*1.22)
        ax.set_title(title + '\nNo calibration; median and range of 3 runs', fontsize=11)
        ax.set_ylabel(unit)
    ax = axes[1, 0]
    for offset, kind, alpha in [(-.18, 'nominal', .35), (.18, 'bootstrap', 1.)]:
        values = np.array([accuracy[m]['null_'+kind]['rate']*100 for m in modes])
        ci = np.array([accuracy[m]['null_'+kind]['interval_95'] for m in modes])*100
        ax.bar(x+offset, values, width=.32, color=colors, alpha=alpha, label=kind.capitalize())
        ax.errorbar(x+offset, values, yerr=[values-ci[:, 0], ci[:, 1]-values], fmt='none', color='#222222', capsize=3)
    ax.axhline(5, color='#555555', linestyle='--', linewidth=1)
    ax.set_ylabel('Family-wise rejection rate (%)')
    ax.set_title('Null simulation: false positives\n2,000 independent validation datasets', fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.set_ylim(0, 23)
    ax = axes[1, 1]
    values = np.array([accuracy[m]['alternative_bootstrap']['rate']*100 for m in modes])
    ci = np.array([accuracy[m]['alternative_bootstrap']['interval_95'] for m in modes])*100
    bars = ax.bar(x, values, color=colors, width=.6)
    ax.errorbar(x, values, yerr=[values-ci[:, 0], ci[:, 1]-values], fmt='none', color='#222222', capsize=4)
    ax.bar_label(bars, labels=[f'{v:.1f}%' for v in values], padding=10)
    ax.set_ylim(0, 55)
    ax.set_ylabel('Detection rate (%)')
    ax.set_title('Alternative simulation: calibrated power\n2,000 independent validation datasets', fontsize=11)
    for ax in axes.flat:
        ax.set_xticks(x, labels)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#e2e5e7', linewidth=.6)
    fig.supxlabel('Fixed model/tree scope. Simulation intervals are conditional on the shared 1,999-dataset calibration reference.', fontsize=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
