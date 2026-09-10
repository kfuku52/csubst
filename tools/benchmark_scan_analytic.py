#!/usr/bin/env python3
"""Reproducible analytical scan calibration/power and paired runtime benchmark.

Simulation uses the independent four-codon CTMC/pruner from the existing scan
calibration validator. Both methods see the same simulated alignments and
candidate selection. Known null parameters are not estimated. Runtime compares
scan with/without the new annotation, in fresh processes; scientific outputs
intentionally differ but old candidate/diagnostic fields must remain identical.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from validate_scan_calibration import one_dataset  # noqa: E402
from csubst import pipeline_calibration, scan_analytic  # noqa: E402


def timed(arguments, analytical):
    start = time.perf_counter()
    result = one_dataset(arguments, analytical=analytical)
    return dict(seconds=time.perf_counter() - start,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--sites', type=int, default=12)
    parser.add_argument('--skip-runtime', action='store_true', help='Run statistical comparisons only.')
    parser.add_argument('--seed', type=int, default=51052026)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    summary = dict(analytical_alternatives=list(scan_analytic.ALTERNATIVES), python=sys.version, platform=platform.platform(), numpy=np.__version__, seed=args.seed,
                   replicates=args.replicates, sites=args.sites, workers=args.workers, conditions=[], runtime=[],
                   scope='known_parameter_four_codon_CTMC; estimated nuisance/model misspecification not validated')
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for scenario in ('sparse', 'unequal', 'uncertain'):
            for filtered in (False, True):
                for signal in (0, max(1, args.sites // 4)):
                    jobs = [(scenario, args.sites, args.seed + i, filtered, signal) for i in range(args.replicates)]
                    results = list(pool.map(partial(one_dataset, analytical=True), jobs, chunksize=10))
                    condition = dict(scenario=scenario, filtered=filtered, signal_sites=signal, metrics={})
                    for key in results[0]:
                        if key in ('score', 'null_score', 'candidates'):
                            continue
                        values = [r[key] for r in results]
                        metric = {'mean': float(np.mean(values))}
                        if all(v in (0, 1) for v in values):
                            metric['ci95'] = list(pipeline_calibration.binomial_interval(int(sum(values)), len(values)))
                        condition['metrics'][key] = metric
                    condition['mean_candidates'] = float(np.mean([r['candidates'] for r in results]))
                    summary['conditions'].append(condition)
                    (args.outdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
                    print(json.dumps(condition), flush=True)
    # Each measurement gets a fresh process so peak RSS and warmup are comparable.
    for sites in (() if args.skip_runtime else (100, 1000)):
        runs = {}
        for analytical in (False, True):
            series = []
            for repeat in range(4):
                with ProcessPoolExecutor(1) as pool:
                    pool.submit(timed, ('uncertain', 12, args.seed, False, 0), analytical).result()
                    series.append(pool.submit(timed, ('uncertain', sites, args.seed + 5000, False, 0), analytical).result())
            runs[str(analytical)] = series
        for before, after in zip(runs['False'], runs['True']):
            assert all(after['result'][k] == v for k, v in before['result'].items()), 'Baseline outputs changed.'
        summary['runtime'].append(dict(sites=sites, runs=runs))
    summary['elapsed_seconds'] = time.perf_counter() - start
    (args.outdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
