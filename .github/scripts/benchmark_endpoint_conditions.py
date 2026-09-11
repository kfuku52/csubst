#!/usr/bin/env python3
"""Compare endpoint implementations across data, outputs, block sizes and arity.

Both roots require compatible native builds. Run sequentially without other
benchmark/test jobs. Raw tables, logs and profiles are written under workdir.
"""
import argparse
import cProfile
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
CASES = {
    'pgk_pair': ('PGK', {}),
    'pgk_sites': ('PGK', {'s': 'yes', 'bs': 'yes', 'cs': 'yes'}),
    'pgk_full': ('PGK', {'b': 'yes', 'output_stat': 'any2any,any2spe,spe2any,spe2spe'}),
    'pgk_recode': ('PGK', {'b': 'yes', 'nonsyn_recode': 'dayhoff6'}),
    'pepc_pair': ('PEPC', {}),
    'pepc_branch': ('PEPC', {'b': 'yes'}),
    'pepc_arity6': ('PEPC', {'b': 'yes', 'max_arity': '6',
                              'cutoff_stat': 'OCNany2spe,2.0|omegaCany2spe,5.0'}),
    'pepc_block8': ('PEPC', {'b': 'yes', 'endpoint_block_size': '8'}),
    'pepc_block256': ('PEPC', {'b': 'yes', 'endpoint_block_size': '256', 'threads': '4'}),
}


def worker(args):
    sys.path.insert(0, str(args.source_root))
    helper = runpy.run_path(str(args.source_root / '.github/scripts/benchmark_endpoints.py'))
    dataset, settings = CASES[args.case]
    command = helper['cli_args'](dataset, 'joint', 64, args.workdir)
    for name, value in settings.items():
        flag = '--' + name
        if flag in command:
            command[command.index(flag) + 1] = value
        else:
            command += [flag, value]
    from csubst import tsv
    original = tsv.write_dataframe

    def capture(frame, path, **kwargs):
        if Path(path).name.startswith('csubst_') and 'stats' not in Path(path).stem:
            frame.to_pickle(str(path) + '.unrounded.pkl')
        return original(frame, path, **kwargs)

    tsv.write_dataframe = capture
    sys.argv = command
    profile = cProfile.Profile() if args.profile else None
    cpu_start = time.process_time()
    start = time.perf_counter()
    if profile is not None:
        profile.enable()
    runpy.run_module('csubst', run_name='__main__')
    seconds = time.perf_counter() - start
    cpu_seconds = time.process_time() - cpu_start
    if profile is not None:
        profile.disable()
        profile.dump_stats(str(args.workdir) + '.prof')
    result = dict(seconds=seconds, cpu_seconds=cpu_seconds, command=command,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                  / (1024**2 if sys.platform == 'darwin' else 1024))
    Path(str(args.workdir) + '.json').write_text(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--result', type=Path)
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    parser.add_argument('--case', choices=list(CASES))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--profile', action='store_true', help='One diagnostic run per case, baseline only.')
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.baseline_root is None or args.result is None or args.repeats < 1:
        parser.error('--baseline-root, --result and positive --repeats are required')
    import numpy as np
    import pandas as pd
    import psutil
    helper = runpy.run_path(str(ROOT / '.github/scripts/benchmark_endpoint_optimization.py'))
    args.workdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED=os.environ.get('PYTHONHASHSEED', '0'))
    result = {'environment': {'python': sys.version, 'platform': platform.platform(),
                              'memory_gib': psutil.virtual_memory().total / 1024**3,
                              'numpy': np.__version__, 'logical_cpus': psutil.cpu_count(),
                              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                              'warmups': 0 if args.profile else 1, 'repeats': 1 if args.profile else args.repeats,
                              'python_hash_seed': env['PYTHONHASHSEED'], 'blas_threads': 1,
                              'profile': args.profile},
              'baseline_sha256': helper['hashes'](args.baseline_root),
              'candidate_sha256': helper['hashes'](ROOT), 'runs': [], 'summary': {}}

    def save():
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2).replace(str(Path.home()) + '/', '${HOME}/') + '\n')

    for case in args.cases:
        reference = None
        configs = [('before', args.baseline_root)] if args.profile else [('before', args.baseline_root), ('after', ROOT)]
        for repeat in range(1 if args.profile else args.repeats + 1):
            for label, source in configs if repeat % 2 == 0 else configs[::-1]:
                out = args.workdir / f'{case}-{label}-{repeat}'
                cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--case', case,
                       '--source-root', str(source), '--workdir', str(out)]
                if args.profile:
                    cmd += ['--profile']
                host_start = psutil.cpu_times()
                wall_start = time.perf_counter()
                with Path(str(out) + '.log').open('w') as log:
                    subprocess.run(cmd, cwd=args.workdir, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                wall = time.perf_counter() - wall_start
                host_end = psutil.cpu_times()
                row = json.loads(Path(str(out) + '.json').read_text())
                row.update(case=case, version=label, repeat=repeat, warmup=repeat == 0,
                           process_wall_seconds=wall,
                           host_active_cpu_cores=sum(getattr(host_end, key) - getattr(host_start, key)
                                                     for key in ['user', 'nice', 'system']) / wall)
                frames = {p.name: pd.read_pickle(p) for p in sorted(out.glob('*.unrounded.pkl'))}
                if case == 'pepc_arity6' and 'csubst_cb_6.tsv.unrounded.pkl' not in frames:
                    raise AssertionError('Arity 6 was not reached')
                if not frames:
                    raise AssertionError('No output tables: ' + case)
                if reference is None:
                    reference = frames
                if frames.keys() != reference.keys():
                    raise AssertionError('Output table set changed: ' + case)
                row['tables'] = {}
                for name, frame in frames.items():
                    row['tables'][name] = helper['compare_frames'](reference[name], frame)
                    row['tables'][name]['exact_frame_equal'] = reference[name].equals(frame)
                    if {'OCNany2spe', 'omegaCany2spe'} <= set(frame):
                        passed = (frame['OCNany2spe'] >= 2) & (frame['omegaCany2spe'] >= 5)
                        expected = (reference[name]['OCNany2spe'] >= 2) & (reference[name]['omegaCany2spe'] >= 5)
                        np.testing.assert_array_equal(passed, expected)
                        row['tables'][name]['passed_cutoffs'] = int(passed.sum())
                    if 'branch_id_1' in frame:
                        ids = [c for c in frame if c.startswith('branch_id_')]
                        np.testing.assert_array_equal(reference[name][ids], frame[ids])
                    if case == 'pepc_arity6' and name == 'csubst_cb_6.tsv.unrounded.pkl':
                        np.testing.assert_array_equal(frame.filter(regex='^branch_id_').to_numpy(), [[1, 26, 33, 35, 40, 102]])
                result['runs'].append(row)
                save()
                print(case, label, repeat, round(row['seconds'], 3), round(row['peak_rss_mib'], 1), flush=True)
        if not args.profile:
            result['summary'][case] = {}
            for label, _ in configs:
                rows = [r for r in result['runs'] if r['case'] == case and r['version'] == label and not r['warmup']]
                result['summary'][case][label] = {
                    metric: {'median': float(np.median([r[metric] for r in rows])),
                             'min': min(r[metric] for r in rows), 'max': max(r[metric] for r in rows)}
                    for metric in ['seconds', 'peak_rss_mib', 'process_wall_seconds']}
            save()


if __name__ == '__main__':
    main()
