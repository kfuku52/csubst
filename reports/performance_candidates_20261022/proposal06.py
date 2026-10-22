"""Evaluate removing per-distribution pandas Series from scan candidates.

Run from repository root: python reports/performance_candidates_20261022/proposal06.py
This patches one function in this process only; production files are untouched.
"""

import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import cli, substitution_scan  # noqa: E402


def numpy_distribution(values, state_orders):
    """Mirror pandas descending nargsort, including its quicksort tie order."""
    if len(values) == 1:
        index, value = values[0]
        return '{}:{:.6g}'.format(str(state_orders[int(index)]), float(value))
    masses = np.asarray([mass for _, mass in values])
    mask = np.isnan(masses)
    positions = np.flatnonzero(~mask)
    # pandas reverses before ascending quicksort and reverses the sorted result.
    # Sorting -masses directly can produce a different order for equal values.
    order = positions[::-1][masses[positions][::-1].argsort(kind='quicksort')][::-1]
    if mask.any():
        order = np.concatenate((order, np.flatnonzero(mask)))
    return ','.join('{}:{:.6g}'.format(str(state_orders[int(values[i][0])]), float(values[i][1]))
                    for i in order)


def make_prototype():
    source = inspect.getsource(substitution_scan.build_candidates)
    before = """','.join('{}:{:.6g}'.format(str(state_orders[int(index)]), float(value))
                                  for index, value in pd.Series(
                                      [mass for _, mass in values], index=[index for index, _ in values]
                                  ).sort_values(ascending=False).items())"""
    assert source.count(before) == 1
    namespace = dict(substitution_scan.__dict__, numpy_distribution=numpy_distribution)
    exec(compile(source.replace(before, 'numpy_distribution(values, state_orders)'),
                 '<proposal06-prototype>', 'exec'), namespace)
    return namespace['build_candidates'], hashlib.sha256(source.encode()).hexdigest()


def baseline_distribution(values, state_orders):
    return ','.join('{}:{:.6g}'.format(str(state_orders[int(index)]), float(value))
                    for index, value in pd.Series(
                        [mass for _, mass in values], index=[index for index, _ in values]
                    ).sort_values(ascending=False).items())


def check_distribution():
    rng = np.random.default_rng(61)
    states = np.array(['S{}'.format(i) for i in range(128)], dtype=object)
    comparisons = 0
    for size in (1, 2, 3, 8, 16, 20, 32, 64, 128):
        for mode in ('ties', 'random', 'nan', 'all_nan', 'int64', 'float32'):
            for _ in range(10):
                masses = rng.integers(0, 5, size=size).astype(float)
                if mode == 'random':
                    masses = rng.random(size)
                elif mode == 'nan':
                    masses[rng.random(size) < 0.3] = np.nan
                elif mode == 'all_nan':
                    masses[:] = np.nan
                elif mode == 'int64':
                    masses = masses.astype(np.int64) + 2**54
                elif mode == 'float32':
                    masses = masses.astype(np.float32) / np.float32(3)
                values = list(zip(rng.permutation(size), masses))
                assert numpy_distribution(values, states) == baseline_distribution(values, states)
                comparisons += 1
    return comparisons


def synthetic_events():
    rng = np.random.default_rng(62)
    rows = 6000
    events = pd.DataFrame({
        'branch_id': rng.integers(1, 80, size=rows),
        'site': rng.integers(0, 300, size=rows),
        'from_state_id': rng.integers(0, 20, size=rows),
        'to_state_id': rng.integers(0, 20, size=rows),
        'event_pp': rng.choice([0.1, 0.2, 0.3, 0.5, 1.0], size=rows),
    })
    # Duplicate indices and shuffled event order should not affect parity.
    events.index = np.arange(rows) % 37
    return events


def candidate_checks(prototype):
    states = np.array(['S{}'.format(i) for i in range(20)], dtype=object)
    large = synthetic_events()
    tied = pd.DataFrame({
        'branch_id': np.arange(1, 21), 'site': np.zeros(20, dtype=int),
        'from_state_id': np.arange(20), 'to_state_id': np.ones(20, dtype=int),
        'event_pp': np.ones(20),
    })
    cases = {'empty': large.iloc[:0], 'singleton': large.iloc[:1],
             'tied': tied, 'fractional': large.iloc[:100], 'large': large}
    comparisons = 0
    for events in cases.values():
        for matches in ([match] for match in substitution_scan.SCAN_MATCHES):
            pd.testing.assert_frame_equal(substitution_scan.build_candidates(events, matches, states),
                                          prototype(events, matches, states), check_exact=True)
            comparisons += 1
        pd.testing.assert_frame_equal(substitution_scan.build_candidates(
            events, list(substitution_scan.SCAN_MATCHES), states),
            prototype(events, list(substitution_scan.SCAN_MATCHES), states), check_exact=True)
        comparisons += 1
    return {'candidate_frame_comparisons': comparisons,
            'distribution_string_comparisons': check_distribution()}, large, states


def paired_candidates(events, matches, states, prototype):
    functions = {'before': substitution_scan.build_candidates, 'numpy': prototype}
    runs = {label: [] for label in functions}
    returned_bytes = None
    rows = None
    for repeat in range(4):
        outputs = {}
        for label in (('before', 'numpy') if repeat % 2 == 0 else ('numpy', 'before')):
            outputs[label], timing = measure(functions[label], events, matches, states)
            if repeat:
                runs[label].append(timing)
        pd.testing.assert_frame_equal(outputs['before'], outputs['numpy'], check_exact=True)
        rows = len(outputs['before'])
        returned_bytes = int(outputs['before'].memory_usage(index=True, deep=True).sum())
    return {'event_rows': len(events), 'candidate_rows': rows,
            'returned_frame_bytes': returned_bytes, 'exact_frame_parity': True,
            'runs': runs, 'summary': {label: summarize(samples) for label, samples in runs.items()}}


class Captured(Exception):
    pass


def real_case(dataset, prototype, workdir):
    original = substitution_scan.scan_substitutions
    original_builder = substitution_scan.build_candidates
    output = {}

    def run(g, ON_tensor, rate_ON_tensor=None):
        static = substitution_scan._build_scan_static_context(g, ON_tensor, rate_ON_tensor)
        captured = []

        def save(events, scan_matches, state_orders):
            captured.append((events.copy(), np.array(state_orders, copy=True)))
            return original_builder(events, scan_matches, state_orders)

        substitution_scan.build_candidates = save
        try:
            reference = substitution_scan._scan_substitutions_core(
                g, ON_tensor, rate_ON_tensor, scan_static=static)
        finally:
            substitution_scan.build_candidates = original_builder
        assert len(captured) == 1
        events, states = captured[0]
        for label, matches in [('any2spe', ['any2spe']), ('all', list(substitution_scan.SCAN_MATCHES))]:
            output[label] = paired_candidates(events, matches, states, prototype)
        runs = {'before': [], 'numpy': []}
        for repeat in range(4):
            frames = {}
            for label in (('before', 'numpy') if repeat % 2 == 0 else ('numpy', 'before')):
                substitution_scan.build_candidates = original_builder if label == 'before' else prototype
                try:
                    frames[label], timing = measure(substitution_scan._scan_substitutions_core,
                                                    g, ON_tensor, rate_ON_tensor, scan_static=static)
                finally:
                    substitution_scan.build_candidates = original_builder
                if repeat:
                    runs[label].append(timing)
            for left, right in zip(frames['before'], frames['numpy']):
                pd.testing.assert_frame_equal(left, right, check_exact=True)
            for left, right in zip(reference, frames['numpy']):
                pd.testing.assert_frame_equal(left, right, check_exact=True)
        output['scan_core_any2spe'] = {
            'runs': runs, 'summary': {label: summarize(samples) for label, samples in runs.items()},
            'scan_rows': len(reference[0]), 'exact_frame_parity': True,
            'scope': 'prepared scan core; excludes inference, input preparation, outputs, calibration'}
        raise Captured

    source_data = ROOT / 'csubst/dataset'
    data = workdir / (dataset + '_inputs')
    data.mkdir()
    for name in ([dataset + '.alignment.fa', dataset + '.tree.nwk', dataset + '.foreground.txt']
                 + [dataset + '.alignment.fa.' + key for key in ('treefile', 'state', 'rate', 'iqtree', 'log')]):
        shutil.copy2(source_data / name, data / name)
    argv = ['scan', '--alignment_file', str(data / (dataset + '.alignment.fa')),
            '--rooted_tree_file', str(data / (dataset + '.tree.nwk')),
            '--foreground', str(data / (dataset + '.foreground.txt')),
            '--outdir', str(workdir / dataset), '--threads', '1', '--blas_threads', '1',
            '--scan_site_plot', 'no', '--scan_pvalue_calibration', 'none',
            '--random_seed', '8', '--float_digit', '12']
    for key in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        argv += ['--iqtree_' + key, str(data / (dataset + '.alignment.fa.' + key))]
    substitution_scan.scan_substitutions = run
    try:
        with (workdir / (dataset + '.log')).open('w') as log, contextlib.redirect_stdout(log):
            try:
                cli.main(argv)
            except Captured:
                pass
    finally:
        substitution_scan.scan_substitutions = original
        substitution_scan.build_candidates = original_builder
    output['input_sha256'] = {
        'csubst/dataset/' + path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(data.glob(dataset + '.*')) if path.is_file()}
    return output


def main():
    prototype, source_hash = make_prototype()
    result = {'proposal': 6, 'environment': environment(),
              'source_function_sha256': source_hash, 'warmups': 1, 'measured_repeats': 3,
              'thread_control': 'VECLIB/OPENBLAS/OMP/MKL=1 before NumPy/SciPy import',
              'memory_scope': 'returned DataFrame payload; peak RSS not measured', 'cases': {}}
    with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-proposal06-') as directory:
        result['edge_checks'], large, states = candidate_checks(prototype)
        for dataset in ('PGK', 'PEPC'):
            result['cases'][dataset] = real_case(dataset, prototype, Path(directory))
            print(dataset, {label: {version: item['wall_s']['median']
                                   for version, item in result['cases'][dataset][label]['summary'].items()}
                            for label in ('any2spe', 'all', 'scan_core_any2spe')}, flush=True)
        result['cases']['synthetic6000_all'] = paired_candidates(
            large, list(substitution_scan.SCAN_MATCHES), states, prototype)
    (HERE / 'proposal06.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
