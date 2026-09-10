#!/usr/bin/env python3
"""Summarize frozen paired holdouts, retaining dataset-level uncertainty."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.directory / 'paired_results.csv', keep_default_na=False)
    groups = ['scenario', 'foregrounds', 'regime', 'filtered', 'method']
    records = []
    for key, frame in data.groupby(groups):
        n = len(frame)
        ci = binomtest(int(frame.false_any.sum()), n).proportion_ci()
        records.append(dict(zip(groups, key), replicates=n, false_any=frame.false_any.mean(),
            false_any_low=ci.low, false_any_high=ci.high, mean_fdp=frame.fdp.mean(),
            sensitivity=frame.detected.sum() / max(frame.signal.sum(), 1),
            endpoint_sensitivity=frame.endpoint_detected.sum() / max(frame.endpoints.sum(), 1),
            jump_sensitivity=frame.jump_detected.sum() / max(frame.jumps.sum(), 1),
            signal=int(frame.signal.sum()), endpoints=int(frame.endpoints.sum()),
            jumps=int(frame.jumps.sum())))
    pd.DataFrame(records).to_csv(args.directory / 'summary.csv', index=False)
    differences = []
    for key, frame in data[data.regime != 'null'].groupby(groups[:4]):
        fixed = frame[frame.method == 'fixed'].set_index('replicate')
        learned = frame[frame.method == 'learned'].set_index('replicate')
        delta = (learned.detected / learned.signal - fixed.detected / fixed.signal).to_numpy()
        se = delta.std(ddof=1) / np.sqrt(len(delta))
        differences.append(dict(zip(groups[:4], key), paired_sensitivity_difference=delta.mean(),
                                normal95_low=delta.mean()-1.96*se, normal95_high=delta.mean()+1.96*se))
    pd.DataFrame(differences).to_csv(args.directory / 'paired_differences.csv', index=False)
    lines = ['| Geometry | Foregrounds | Filter | Null any false: Poisson / fixed / learned | Process sensitivity: Poisson / fixed / learned | Realized endpoint sensitivity: Poisson / fixed / learned |',
             '|---|---:|---|---|---|---|']
    for key, frame in data.groupby(['scenario', 'foregrounds', 'filtered']):
        metrics = []
        for method in ['poisson', 'fixed', 'learned']:
            subset = frame[frame.method == method]
            null = subset[subset.regime == 'null']
            alt = subset[subset.regime != 'null']
            metrics.append([null.false_any.mean(), alt.detected.sum()/alt.signal.sum(),
                            alt.endpoint_detected.sum()/max(alt.endpoints.sum(), 1)])
        cells = [' / '.join(f'{100*x:.1f}%' for x in column) for column in zip(*metrics)]
        lines.append('| ' + ' | '.join(map(str, [*key, *cells])) + ' |')
    (args.directory / 'comparison.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
