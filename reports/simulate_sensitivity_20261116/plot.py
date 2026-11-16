"""Plot multiplier profiling and the lack of synonymous convergence events."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


NULLS = (
    'baseline', 'short_alignment', 'long_alignment', 'short_branches',
    'long_branches', 'rate_mixture', 'strong_constraint', 'weak_constraint',
)
STRATA = ['(-inf, 0.1]', '(0.1, 1.0]', '(1.0, 5.0]', '(5.0, inf]']


def plot(folder):
    profile = pd.read_csv(folder / 'profile_omega.tsv', sep='\t')
    counts = pd.read_csv(folder / 'by_expected_S.tsv', sep='\t')
    counts = counts[(counts.variant == 'default') & counts.scenario.isin(NULLS)]
    pooled = counts.groupby(['dataset', 'stratum'])[['pairs', 'true_S0']].sum()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    for dataset, color in [('PGK', '#1c6393'), ('PEPC', '#208568')]:
        fits = profile.loc[(profile.dataset == dataset) & (profile.omega_iqtree != 1)]
        left.scatter(fits.omega_sim, fits.omega_iqtree, label=dataset, color=color, s=29, alpha=.68, zorder=2)
    stalled = profile.loc[profile.omega_iqtree == 1]
    offsets = np.random.default_rng(11).uniform(-.065, .065, size=len(stalled))
    x = stalled.omega_sim * np.exp(offsets)
    left.vlines(x, stalled.omega_profile, stalled.omega_iqtree, color='#c46537', alpha=.4, linewidth=1)
    left.scatter(x, stalled.omega_iqtree, color='#c46537', marker='x', s=36, label='Fit stayed at 1')
    left.scatter(x, stalled.omega_profile, facecolors='white', edgecolors='#c46537', s=32,
                 label='Same fit, multiplier profiled', zorder=3)
    left.plot([.045, 1.35], [.045, 1.35], linestyle='--', color='.55', linewidth=1, zorder=0)
    left.set(xscale='log', yscale='log', ylim=(.043, 1.45), xlim=(.049, 1.25),
             xlabel='Simulated nonsynonymous multiplier', ylabel='Estimated multiplier',
             title='Likelihood check on 72 fits')
    left.set_xticks([.05, .1, .3, 1], labels=['0.05', '0.1', '0.3', '1'])
    left.set_yticks([.05, .1, .3, 1], labels=['0.05', '0.1', '0.3', '1'])
    left.legend(loc='lower right', frameon=False, fontsize=9)
    grid = np.arange(len(STRATA))
    for dataset, color, shift in [('PGK', '#1c6393', -.18), ('PEPC', '#208568', .18)]:
        cols = pooled.loc[dataset].reindex(STRATA)
        right.bar(grid + shift, cols.true_S0 / cols.pairs, width=.34, color=color, label=dataset)
    right.set_xticks(grid, labels=['≤0.1', '0.1–1', '1–5', '>5'])
    right.set(xlabel='Q-based expected synonymous convergence count',
              ylabel='Pairs with zero true synonymous convergences',
              ylim=(0, 1.07), title='Null simulations by expected S count')
    right.yaxis.set_major_formatter(PercentFormatter(1))
    right.legend(frameon=False)
    for ax in (left, right):
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=10)
    fig.savefig(folder / 'sensitivity.png', dpi=190)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    plot(parser.parse_args().report)
