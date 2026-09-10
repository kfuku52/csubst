#!/usr/bin/env python3
"""Compare a saved pre-optimization source tree with the current joint engine.

Both trees need compatible built extensions and the bundled PGK/PEPC fixtures.
Run without concurrent test/benchmark processes. Writes raw runs under workdir.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def worker(args):
    sys.path.insert(0, str(args.source_root))
    bench = runpy.run_path(str(args.source_root / '.github/scripts/benchmark_endpoints.py'))
    command = bench['cli_args'](args.dataset, args.mode, 64, args.outdir)
    command[command.index('--b') + 1] = 'yes' if args.branch_table else 'no'
    start = time.perf_counter()
    sys.argv = command
    runpy.run_module('csubst', run_name='__main__')
    seconds = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {'seconds': seconds, 'peak_rss_mib': rss / (1024**2 if sys.platform == 'darwin' else 1024),
              'command': command}
    Path(args.result).write_text(json.dumps(result, indent=2) + '\n')


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / 'csubst').rglob('*'))
            if p.is_file() and (p.suffix in ('.py', '.pyx', '.so') or
                               (p.parent.name == 'dataset' and p.name.startswith(('PGK.', 'PEPC.'))))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root', type=Path)
    p.add_argument('--workdir', type=Path, default=Path('/tmp/csubst-endpoint-optimization'))
    p.add_argument('--result', default='endpoint_optimization.json')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--source-root', type=Path)
    p.add_argument('--dataset', default='PGK')
    p.add_argument('--mode', default='joint')
    p.add_argument('--outdir', type=Path)
    p.add_argument('--branch-table', action='store_true')
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    if args.baseline_root is None or args.repeats < 1:
        p.error('--baseline-root and positive --repeats are required')
    import numpy as np
    import pandas as pd
    import scipy
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')
    args.workdir.mkdir(parents=True, exist_ok=True)
    output = {'environment': {'platform': platform.platform(), 'python': sys.version,
                              'numpy': np.__version__, 'scipy': scipy.__version__,
                              'threads': 1, 'warmups': 1, 'repeats': args.repeats},
              'baseline_sha256': hashes(args.baseline_root), 'optimized_sha256': hashes(ROOT), 'datasets': {}}
    configs = [('marginal', args.baseline_root, 'marginal'),
               ('joint_before', args.baseline_root, 'joint'), ('joint_after', ROOT, 'joint')]
    for dataset in ['PGK', 'PEPC']:
        runs = []
        for repeat in range(args.repeats + 1):
            for label, source, mode in (configs if repeat % 2 == 0 else configs[::-1]):
                tag = '{}-{}-{}'.format(dataset, label, repeat)
                result = args.workdir / (tag + '.json')
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--source-root', str(source),
                           '--dataset', dataset, '--mode', mode, '--result', str(result),
                           '--outdir', str(args.workdir / tag)]
                if args.branch_table:
                    command.append('--branch-table')
                with (args.workdir / (tag + '.log')).open('w') as log:
                    subprocess.run(command, cwd=args.workdir, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                row = json.loads(result.read_text())
                row.update(version=label, repeat=repeat, warmup=repeat == 0)
                runs.append(row)
                print(tag, round(row['seconds'], 3), round(row['peak_rss_mib'], 1), flush=True)
        summary = {}
        for label, _, _ in configs:
            selected = [r for r in runs if r['version'] == label and not r['warmup']]
            summary[label] = {key: {'median': float(np.median([r[key] for r in selected])),
                                    'min': min(r[key] for r in selected), 'max': max(r[key] for r in selected)}
                              for key in ['seconds', 'peak_rss_mib']}
        parity = {}
        files = ['csubst_cb_2.tsv'] + (['csubst_b.tsv'] if args.branch_table else [])
        for name in files:
            a = pd.read_csv(args.workdir / (dataset + '-joint_before-1') / name, sep='\t')
            b = pd.read_csv(args.workdir / (dataset + '-joint_after-1') / name, sep='\t')
            pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10, atol=1e-10)
            numeric = a.select_dtypes(include='number').columns
            delta = (a[numeric] - b[numeric]).to_numpy()
            finite = np.isfinite(delta)
            parity[name] = {'shape': list(a.shape), 'rtol': 1e-10, 'atol': 1e-10,
                            'max_absolute_difference_finite': float(np.max(np.abs(delta[finite]), initial=0))}
        output['datasets'][dataset] = {'runs': runs, 'summary': summary, 'joint_output_parity': parity}
    # Keep exported evidence independent of a particular user's home directory.
    text = json.dumps(output, indent=2).replace(str(Path.home()) + '/', '${HOME}/')
    Path(args.result).write_text(text + '\n')


if __name__ == '__main__':
    main()
