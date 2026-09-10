"""Render the retained full grid, without scenario selection."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
rows = json.loads((root / 'results.json').read_text())['results']
fig, axes = plt.subplots(1, 3, figsize=(12, 6), constrained_layout=True)
for ax, key, title in zip(axes, ['oracle_null', 'independent_null', 'oracle_alternative'],
                          ['Pair oracle: null rejection', 'Independent: null rejection', 'Pair oracle: alternative rejection']):
    values = np.array([r[key]['exact_rejection_probability'] for r in rows]).reshape(9, 3)
    ax.imshow(values, vmin=0, vmax=.85, cmap='Blues', aspect='auto')
    for i in range(9):
        for j in range(3):
            ax.text(j, i, f'{values[i, j]*100:.1f}%', ha='center', va='center',
                    color='white' if values[i, j]>.5 else 'black', fontsize=9)
    ax.set_xticks(range(3), ['Same\nstable', 'Same\nmismatched', 'Different\nparents'])
    ax.set_yticks(range(9), [f'J={j:g}, t={t:g}' for j in (0,.8,1.6) for t in (.1,.5,1.5)])
    ax.set_title(title, fontsize=11)
fig.suptitle('Exact rejection probabilities at nominal 5%\nKnown landscape and parents; 20 independent two-site blocks', fontsize=13)
fig.savefig(root / 'rejection_grid.png', dpi=180)
fig.savefig(root / 'rejection_grid.pdf')
