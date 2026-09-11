#!/usr/bin/env python3
"""Compare baseline and integrated scans with the same fitted PEPC model.

Peak RSS sums the live process tree (including IQ-TREE and bootstrap workers),
polled every 20 ms. Warmups are excluded; configuration order rotates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[2]


def run(source, mode, calibration, args, out):
    env = dict(os.environ, PYTHONPATH=str(source), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    command = [sys.executable, '-m', 'csubst', 'scan', '--alignment_file', str(args.alignment),
               '--rooted_tree_file', str(ROOT/'csubst/dataset/PEPC.tree.nwk'),
               '--foreground', str(ROOT/'reports/csubst_scan_pepc_20260625/PEPC.foreground.independent.txt'),
               '--iqtree_model', 'GY+FQ', '--iqtree_outdir', str(args.fit),
               '--scan_pvalue_calibration', calibration, '--scan_n_permutations', str(args.niter),
               '--scan_permutation_seed', '19051', '--scan_site_plot', 'no', '--threads', '1',
               '--blas_threads', '1', '--float_digit', '10', '--outdir', str(out)]
    command += ['--scan_observation', mode]
    if mode != 'marginal':
        command += ['--scan_rate_exposure', 'endpoint', '--scan_rate_length', 'raw']
    out.mkdir(parents=True)
    start = time.perf_counter()
    peak = 0
    with (out/'process.log').open('w') as log:
        process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
        monitor = psutil.Process(process.pid)
        while process.poll() is None:
            memory = 0
            try:
                family = [monitor] + monitor.children(recursive=True)
            except psutil.NoSuchProcess:
                family = []
            for item in family:
                try:
                    memory += item.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, memory)
            time.sleep(.02)
    if process.returncode:
        raise RuntimeError(str(out/'process.log'))
    record = dict(seconds=time.perf_counter()-start, peak_tree_rss_bytes=peak, command=command)
    frame = pd.read_csv(out/'csubst_scan.tsv', sep='\t')
    if calibration == 'parametric_bootstrap':
        summary = json.loads((out/'csubst_scan_inference.json').read_text())['bootstrap']
        if summary['failure_count'] or summary['success_count'] != args.niter:
            raise RuntimeError('Incomplete calibration: '+str(out))
    record.update(candidates=len(frame), finite_scores=int(np.isfinite(frame.score_rate_enrichment).sum()))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline', 'alignment', 'fit', 'outdir']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--niter', type=int, default=3)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    configs = [(label, source, mode, calibration)
               for label, source, mode in [('baseline', args.baseline, 'marginal'), ('joint', ROOT, 'joint'), ('bridge', ROOT, 'bridge')]
               for calibration in ['none', 'parametric_bootstrap']]
    results = dict(platform=platform.platform(), python=sys.version,
                   alignment_sha256=hashlib.sha256(args.alignment.read_bytes()).hexdigest(), records=[])
    for repeat in range(args.repeats+1):
        for index in range(len(configs)):
            label, source, mode, calibration = configs[(index+repeat) % len(configs)]
            out = args.outdir/f'{label}-{calibration}-r{repeat}'
            record = run(source, mode, calibration, args, out)
            record.update(label=label, calibration=calibration, repeat=repeat, warmup=repeat==0)
            results['records'].append(record)
            text = json.dumps(results, indent=2).replace(str(Path.home()), '${HOME}')
            (args.outdir/'benchmark.json').write_text(text+'\n')
            print(label, calibration, repeat, round(record['seconds'], 2), flush=True)
    summary = []
    for label, _, _, calibration in configs:
        rows = [r for r in results['records'] if r['label']==label and r['calibration']==calibration and not r['warmup']]
        summary.append(dict(label=label, calibration=calibration,
                            median_seconds=float(np.median([r['seconds'] for r in rows])),
                            median_peak_GiB=float(np.median([r['peak_tree_rss_bytes']/2**30 for r in rows]))))
    (args.outdir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
