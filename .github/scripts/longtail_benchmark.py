#!/usr/bin/env python3
"""Compare exact PEPC count draws with/without pass-local caches, then run search."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from csubst import cli, omega_calibration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=ROOT / 'reports/longtail_benchmark')
    parser.add_argument('--niter', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.niter < 100 or args.repeats < 1:
        parser.error('--niter must be >= 100 and --repeats must be >= 1')
    args.outdir.mkdir(parents=True, exist_ok=True)
    original = omega_calibration.apply_calibration
    data = ROOT / 'csubst/dataset'
    inputs = [data / ('PEPC.alignment.fa' + suffix)
              for suffix in ['', '.treefile', '.state', '.rate', '.iqtree', '.log']]
    inputs.append(data / 'PEPC.tree.nwk')
    record = dict(python=platform.python_version(), platform=platform.platform(),
                  input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  niter=args.niter, repeats=args.repeats, results=[])

    def benchmark(cb, g, ON_tensor=None, OS_tensor=None, reuse_reference=False):
        omega_calibration._base_seed(g)
        rows = [0, len(cb) // 2, len(cb) - 1]
        record.update(rows=rows, combinations=len(cb), exact_draw_equality=False)
        for model in ['poisson', 'hypergeom']:
            for repeat in range(args.repeats):
                base = dict(g, omega_pvalue_null_model=model)
                cached = omega_calibration._draw_config(cb, base)
                outputs = []
                for label, config in [('baseline', base), ('cached', cached)]:
                    start = time.perf_counter()
                    arrays = []
                    with contextlib.redirect_stdout(io.StringIO()):
                        for sub in ['any2spe', 'any2any']:
                            for index in rows:
                                arrays.extend(omega_calibration._draw_rates(
                                    cb.iloc[[index]], sub, 'fit', args.niter,
                                    ON_tensor, OS_tensor, config))
                    outputs.append(arrays)
                    record['results'].append(dict(model=model, repeat=repeat, path=label,
                                                  seconds=time.perf_counter() - start))
                for expected, actual in zip(*outputs):
                    np.testing.assert_array_equal(expected, actual)
        record['exact_draw_equality'] = True
        start = time.perf_counter()
        out = original(cb, g, ON_tensor, OS_tensor, reuse_reference)
        record['full_calibration_seconds'] = time.perf_counter() - start
        record['process_maxrss_platform_units'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.outdir / 'benchmark.json').write_text(json.dumps(record, indent=2) + '\n')
        return out

    command = ['csubst', 'search', '--alignment_file', str(data / 'PEPC.alignment.fa'),
               '--rooted_tree_file', str(data / 'PEPC.tree.nwk'),
               '--expectation_method', 'urn', '--output_stat', 'any2spe,any2any',
               '--calibrate_longtail', 'yes', '--random_seed', '20260910',
               '--omega_pvalue_null_model', 'poisson', '--threads', '1',
               '--max_arity', '2', '--longtail_null_niter', str(args.niter),
               '--outdir', str(args.outdir / 'search')]
    for kind in ['treefile', 'state', 'rate', 'iqtree', 'log']:
        command += ['--iqtree_' + kind, str(data / ('PEPC.alignment.fa.' + kind))]
    record['command'] = command
    saved_argv = sys.argv
    try:
        omega_calibration.apply_calibration = benchmark
        sys.argv = command
        cli.main()
    finally:
        omega_calibration.apply_calibration = original
        sys.argv = saved_argv


if __name__ == '__main__':
    main()
