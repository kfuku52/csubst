"""Plot realized-count interval coverage by branch relationship."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


METHODS = [
    ('factorized', 'Separate branch posteriors', '#a85d38', 'x'),
    ('joint_fitted', 'Joint posterior, fitted model', '#165b8e', 'o'),
    ('joint_oracle', 'Joint posterior, simulation inputs', '#2a876e', 'D'),
]


def plot(results, output):
    inputs = [
        ('Background Q', results / 'null' / 'posterior_by_relationship.tsv'),
        ('Directionally biased sites', results / 'biased' / 'posterior_by_relationship.tsv'),
    ]
    fig, panels = plt.subplots(1, 2, figsize=(10.1, 3.7), sharey=True, layout='constrained')
    for ax, (label, path) in zip(panels, inputs):
        table = pd.read_csv(path, sep='\t').set_index(['relationship', 'method'])
        for method, name, color, marker in METHODS:
            if method not in table.index.get_level_values('method'):
                continue
            rows = table.xs(method, level='method').loc[['sisters', 'different_parents']]
            values = rows.s_covered / rows.rows
            ax.plot(values, [0, 1], marker=marker, linewidth=1.3, markersize=7, color=color, label=name)
        ax.axvline(.95, color='.5', linestyle='--', linewidth=1)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set(xlabel='Truth inside the nominal 95% count interval', title=label, ylim=(-.32, 1.36),
               xlim=(.75, 1.01), xticks=[.75, .8, .85, .9, .95, 1.])
        ax.spines[['top', 'right']].set_visible(False)
    panels[0].set_yticks([0, 1], labels=['Sister branches', 'Different parents'])
    handles, labels = panels[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False, fontsize=9)
    fig.savefig(output, dpi=190)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    plot(args.results, args.output)
