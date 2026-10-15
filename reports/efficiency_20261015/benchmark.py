"""Reproduce fixed-fit CLI and tree preparation comparisons in fresh processes.

CLI mode reuses the repository's all-column comparison harness (macOS time -l).
Tree mode times prepared inputs, excluding imports and tree parsing. Both modes
use one warmup and three measured repetitions, with identical Python/native code.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time


def tree_worker(case):
    import numpy as np
    from csubst import endpoint, ete, tree

    shape, size_text = case.rsplit('_', 1)
    size = int(size_text)
    if shape == 'endpoint':
        parents = np.arange(size) - 1
        lengths = np.ones(size)
        lengths[0] = 0

        def run():
            model = endpoint.EndpointModel(parents, lengths, [[-1., 1.], [1., -1.]], [.5, .5])
            return model.order
    else:
        parts = ['L{}:1'.format(i) for i in range(size)]
        if shape == 'balanced':
            while len(parts) > 1:
                parts = ['(' + ','.join(parts[i:i + 2]) + '):1' for i in range(0, len(parts), 2)]
        elif shape == 'ladder':
            text = parts[0]
            for part in parts[1:]:
                text = '(' + text + ',' + part + '):1'
            parts = [text]
        else:
            raise ValueError(shape)
        tr = ete.PhyloNode(parts[0] + ';', format=1)

        def run():
            tree.add_numerical_node_labels(tr)
            return tr

    timings = []
    for repeat in range(4):
        start = time.perf_counter()
        result = run()
        elapsed = time.perf_counter() - start
        if repeat:
            timings.append(elapsed)
    values = (result if shape == 'endpoint' else
              [ete.get_prop(node, 'numerical_label') for node in result.traverse()])
    checksum = hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(dict(seconds=timings, output_sha256=checksum,
                          peak_rss_mib=rss / (2**20 if sys.platform == 'darwin' else 1024))))


def compare_trees():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    output = dict(python=sys.version, platform=platform.platform(), warmups=1,
                  measured_repeats=3, runs=[], summary={})
    for case in ('balanced_2048', 'ladder_2048', 'endpoint_16384'):
        versions = {}
        for label, source in [('before', args.baseline_root), ('after', args.candidate_root)]:
            command = [sys.executable, str(Path(__file__).resolve()), 'tree-worker', case]
            proc = subprocess.run(command, cwd=source, env=dict(os.environ, PYTHONPATH=str(source)),
                                  capture_output=True, text=True, check=True)
            row = dict(case=case, version=label, command=command, **json.loads(proc.stdout))
            row['source_sha256'] = {
                name: hashlib.sha256((source / 'csubst' / name).read_bytes()).hexdigest()
                for name in ('tree.py', 'endpoint.py')
            }
            output['runs'].append(row)
            versions[label] = row
        assert versions['before']['output_sha256'] == versions['after']['output_sha256']
        medians = {label: statistics.median(row['seconds']) for label, row in versions.items()}
        output['summary'][case] = dict(seconds=medians,
                                      after_over_before=medians['after'] / medians['before'],
                                      identical_output=True)
        print(case, output['summary'][case], flush=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(output, indent=2).replace(str(Path.home()) + '/', '${HOME}/') + '\n')


def compare_scan_core():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(VECLIB_MAXIMUM_THREADS='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    import pandas as pd
    from csubst import cli, substitution_scan

    source = args.baseline_root / 'csubst/substitution_scan.py'
    spec = importlib.util.spec_from_file_location('scan_baseline', source)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    helper_path = Path(__file__).resolve().parents[1] / 'resource_audit_20260911/benchmark.py'
    spec = importlib.util.spec_from_file_location('cli_harness', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    args.workdir.mkdir(parents=True, exist_ok=False)
    original = substitution_scan.scan_substitutions
    output = dict(python=sys.version, platform=platform.platform(), warmups=1,
                  measured_repeats=3, runs=[], summary={}, source_sha256={})
    for label, path in [('before', source), ('after', Path(substitution_scan.__file__))]:
        output['source_sha256'][label] = hashlib.sha256(path.read_bytes()).hexdigest()
    for case in ('pgk_scan_defaults', 'pepc_scan_defaults'):
        def measure(g, ON_tensor, rate_ON_tensor=None):
            static = substitution_scan._build_scan_static_context(g, ON_tensor, rate_ON_tensor)
            for repeat in range(4):
                versions = [('before', baseline), ('after', substitution_scan)]
                if repeat % 2:
                    versions.reverse()
                frames = {}
                for label, module in versions:
                    wall, cpu = time.perf_counter(), time.process_time()
                    frame, units = module._scan_substitutions_core(
                        g, ON_tensor, rate_ON_tensor, scan_static=static)
                    cpu_elapsed, wall_elapsed = time.process_time() - cpu, time.perf_counter() - wall
                    frames[label] = (frame, units)
                    output['runs'].append(dict(case=case, version=label, repeat=repeat,
                                              warmup=repeat == 0, seconds=wall_elapsed,
                                              cpu_seconds=cpu_elapsed, rows=len(frame)))
                for before, after in zip(frames['before'], frames['after']):
                    pd.testing.assert_frame_equal(before, after, check_exact=True)
            return original(g, ON_tensor, rate_ON_tensor)

        substitution_scan.scan_substitutions = measure
        command = helper.command_for(case, 'before', args.baseline_root,
                                     args.baseline_root / 'csubst/dataset', args.workdir / case)
        try:
            with (args.workdir / (case + '.log')).open('w') as log, contextlib.redirect_stdout(log):
                status = cli.main(command[3:])
            assert status in (0, None)
        finally:
            substitution_scan.scan_substitutions = original
        summary = {}
        for metric in ('seconds', 'cpu_seconds'):
            medians = {
                label: statistics.median(r[metric] for r in output['runs']
                                         if r['case'] == case and r['version'] == label and not r['warmup'])
                for label in ('before', 'after')
            }
            summary[metric] = dict(medians=medians, after_over_before=medians['after'] / medians['before'])
        output['summary'][case] = dict(summary, identical_output=True, command=command)
        print(case, summary, flush=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(output, indent=2).replace(str(Path.home()) + '/', '${HOME}/') + '\n')


def compare_cli():
    source = Path(__file__).resolve().parents[1] / 'memory_optimization_20260911/benchmark.py'
    spec = importlib.util.spec_from_file_location('memory_benchmark', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    harness = module.harness
    harness.output_summary = module.output_summary
    harness.compare = module.compare_all
    harness.CASES.update({
        'pepc4_scan_defaults': ('PEPCx4', 'scan', 'defaults', {}),
        'pgk_scan_parallel20': ('PGK', 'scan', 'defaults', {
            'scan_pvalue_calibration': 'full_scan', 'scan_n_permutations': '20', 'threads': '2',
        }),
    })
    os.environ.update(VECLIB_MAXIMUM_THREADS='1', MPLBACKEND='Agg')
    harness.main()


def check_cli_result(path):
    source = Path(__file__).resolve().parents[1] / 'memory_optimization_20260911/benchmark.py'
    spec = importlib.util.spec_from_file_location('memory_benchmark', source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    result = json.loads(path.read_text())
    cases = {row['case'] for row in result['runs']}
    assert cases and cases == set(result['summary'])
    assert len(result['runs']) == len(cases) * 2 * (1 + result['environment']['measured_repeats'])
    assert all(row['exit_code'] == 0 for row in result['runs'])
    result['json_comparisons'] = {}
    result['repeat_comparisons'] = {}
    for case in result['summary']:
        pairs = {}
        for row in result['runs']:
            if row['case'] == case:
                command = row['command']
                outdir = Path(command[command.index('--outdir') + 1])
                reports = {}
                for report in outdir.glob('csubst_scan_*.json'):
                    data = json.loads(report.read_text())
                    data.pop('csubst_version', None)
                    reports[report.name] = data
                pairs.setdefault(row['repeat'], {})[row['version']] = (outdir, reports)
        result['repeat_comparisons'][case] = {}
        for repeat, versions in pairs.items():
            before, after = versions['before'], versions['after']
            assert before[1] == after[1], (case, repeat)
            result['repeat_comparisons'][case][repeat] = helper.compare_all(
                helper.output_summary(before[0]), helper.output_summary(after[0]))
        result['json_comparisons'][case] = dict(identical=True, compared_pairs=len(pairs),
                                              files=sorted(before[1]))
    path.write_text(json.dumps(result, indent=2) + '\n')
    print('All scan calibration/inference JSON records match (excluding version metadata).')


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    if mode == 'tree-worker':
        tree_worker(sys.argv[1])
    elif mode == 'trees':
        compare_trees()
    elif mode == 'cli':
        compare_cli()
    elif mode == 'scan-core':
        compare_scan_core()
    elif mode == 'check-results':
        check_cli_result(Path(sys.argv[1]))
    else:
        raise SystemExit('Use cli, trees or scan-core.')
