"""Verify the integrated memory changes in fresh, isolated Python processes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import tempfile
import tracemalloc

for variable in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

from bench_common import benchmark_lock, environment  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def worker(source, case):
    sys.path.insert(0, str(source))
    if case == 'conservation':
        import numpy as np
        from csubst import substitution_scan as module

        state = np.random.default_rng(723).random((192, 8192, 20))
        state /= state.sum(axis=2, keepdims=True)
        state[:, ::29] = 0
        state.flags.writeable = False
        result = module._calc_conservation(state, np.arange(20), np.arange(192), 1e-12)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = peak if sys.platform == 'darwin' else peak * 1024
        digest = hashlib.sha256()
        for key, values in result.items():
            digest.update(key.encode())
            digest.update(json.dumps(values.tolist()).encode() if values.dtype == object else values.tobytes())
        measurement = {'peak_rss_bytes': peak}
    else:
        from csubst import combination as module, ete

        text = '(' * 8191 + 'L0:1' + ''.join(f',L{index}:1):1' for index in range(1, 8192))
        tree = ete.PhyloNode(text[:-2] + 'R;', format=1)
        tracemalloc.start()
        result = module.count_independent_branch_combinations(tree, 16, exclude_sister_pair=True)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        digest = hashlib.sha256(json.dumps(result).encode())
        measurement = {'peak_traced_bytes': peak}
    expected = (source / 'csubst' / Path(module.__file__).name).resolve()
    assert Path(module.__file__).resolve() == expected
    measurement['output_sha256'] = digest.hexdigest()
    measurement['source_sha256'] = hashlib.sha256(expected.read_bytes()).hexdigest()
    print(json.dumps(measurement))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, default=ROOT)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cases', nargs='+', choices=['conservation', 'combinations'],
                        default=['conservation', 'combinations'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    results = {'scope': 'Fresh-process memory; no runtime speed claim', 'repeats': args.repeats,
               'workloads': {
                   'conservation': '192 tips x 8192 sites x 20 float64 states, missing every 29th site; process RSS includes imports and input',
                   'combinations': '8192-tip pectinate tree, arity16, sister excluded; tracemalloc only during DP, excludes existing tree/imports; RSS not measured',
               }, 'cases': {}}
    if args.resume:
        results = json.loads(args.result.read_text())
    with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-round3-memory-') as directory:
        results['environment'] = environment()
        for case in args.cases:
            samples = {'before': [], 'after': []}
            for repeat in range(args.repeats):
                for label in (('before', 'after') if repeat % 2 == 0 else ('after', 'before')):
                    source = args.baseline if label == 'before' else args.candidate
                    command = [sys.executable, str(Path(__file__).resolve()), 'worker', str(source.resolve()), case]
                    output = subprocess.check_output(command, text=True, cwd=directory)
                    samples[label].append(json.loads(output.splitlines()[-1]))
            hashes = {sample['output_sha256'] for group in samples.values() for sample in group}
            assert len(hashes) == 1, (case, hashes)
            metric = 'peak_rss_bytes' if case == 'conservation' else 'peak_traced_bytes'
            results['cases'][case] = {'runs': samples, 'output_exact_digest': True, 'metric': metric,
                'summary': {label: {'median': statistics.median(sample[metric] for sample in group),
                                    'min': min(sample[metric] for sample in group),
                                    'max': max(sample[metric] for sample in group)}
                            for label, group in samples.items()}}
            args.result.write_text(json.dumps(results, indent=2) + '\n')
            print(case, results['cases'][case]['summary'], flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        worker(Path(sys.argv[2]), sys.argv[3])
    else:
        main()
