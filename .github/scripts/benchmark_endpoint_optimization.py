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
    # Preserve the unrounded table for verification. Both versions pay the
    # same capture cost; performance runs retain normal user-facing TSVs.
    from csubst import tsv
    original_writer = tsv.write_dataframe
    def capture(dataframe, output_path, **kwargs):
        if Path(output_path).name in ('csubst_cb_2.tsv', 'csubst_b.tsv'):
            dataframe.to_pickle(str(output_path) + '.unrounded.pkl')
        return original_writer(dataframe, output_path, **kwargs)
    tsv.write_dataframe = capture
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



def compare_frames(reference, candidate):
    """Check counts strictly and propagate their rounding differences to ratios.

    CoD divides by any2any-any2spe, so tiny summation differences can be
    amplified. This checks that ratio changes are fully explained by their
    already-validated inputs, not an arbitrary looser tolerance for all output.
    """
    import numpy as np
    import pandas as pd
    ratios = {}
    for name in reference.columns:
        for prefix, numerator, denominator in [('dNC', 'OCN', 'ECN'), ('dSC', 'OCS', 'ECS'),
                                                ('omegaC', 'dNC', 'dSC')]:
            if name.startswith(prefix):
                suffix = name[len(prefix):]
                ratios[name] = (numerator + suffix, denominator + suffix)
        if name in ('OCNCoD', 'OCSCoD'):
            ratios[name] = (name[:3] + 'any2spe', name[:3] + 'any2dif')
    pd.testing.assert_frame_equal(reference.drop(columns=list(ratios)), candidate.drop(columns=list(ratios)),
                                  check_exact=False, rtol=1e-10, atol=1e-10)
    details = {}
    for name, (numerator, denominator) in ratios.items():
        a, b = reference[name].to_numpy(), candidate[name].to_numpy()
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_array_equal(np.isposinf(a), np.isposinf(b))
        np.testing.assert_array_equal(np.isneginf(a), np.isneginf(b))
        # Identical masked zeros can have undefined rate inputs (0 / NaN).
        # They need no error propagation; changed outputs still must pass it.
        finite = np.isfinite(a) & np.isfinite(b) & (a != b)
        x, y = reference[numerator].to_numpy()[finite], candidate[numerator].to_numpy()[finite]
        u, v = reference[denominator].to_numpy()[finite], candidate[denominator].to_numpy()[finite]
        delta = np.abs(a[finite] - b[finite])
        bound = 8 * np.finfo(float).eps * np.maximum(1, np.maximum(np.abs(a[finite]), np.abs(b[finite])))
        positive = (u != 0) & (v != 0)
        bound[positive] += (np.abs(x[positive] - y[positive]) / np.abs(u[positive])
                            + np.abs(y[positive] / v[positive]) * np.abs((u[positive] - v[positive]) / u[positive]))
        if not np.all(delta <= bound):
            raise AssertionError('Unexplained ratio difference in ' + name)
        details[name] = {'max_absolute_difference': float(np.max(delta, initial=0)),
                         'max_relative_difference': float(np.max(delta / np.maximum(np.abs(a[finite]), 1e-300), initial=0))}
    return {'shape': list(reference.shape), 'count_rtol': 1e-10, 'count_atol': 1e-10,
            'ratio_input_error_propagation': True, 'ratios': details}

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
            before_path = args.workdir / (dataset + '-joint_before-1') / name
            after_path = args.workdir / (dataset + '-joint_after-1') / name
            # These pickles are created by the local workers above, never read
            # from downloaded or user-supplied files.
            a = pd.read_pickle(str(before_path) + '.unrounded.pkl')
            b = pd.read_pickle(str(after_path) + '.unrounded.pkl')
            parity[name] = compare_frames(a, b)
            rounded_a = pd.read_csv(before_path, sep='\t')
            rounded_b = pd.read_csv(after_path, sep='\t')
            numeric = rounded_a.select_dtypes(include='number').columns
            with np.errstate(invalid='ignore'):
                delta = (rounded_a[numeric] - rounded_b[numeric]).to_numpy()
            finite = np.isfinite(delta)
            parity[name]['rounded_numeric_cells_different'] = int(np.count_nonzero(delta[finite]))
            parity[name]['rounded_max_absolute_difference'] = float(np.max(np.abs(delta[finite]), initial=0))
        output['datasets'][dataset] = {'runs': runs, 'summary': summary, 'joint_output_parity': parity}
    # Keep exported evidence independent of a particular user's home directory.
    text = json.dumps(output, indent=2).replace(str(Path.home()) + '/', '${HOME}/')
    Path(args.result).write_text(text + '\n')


if __name__ == '__main__':
    main()
