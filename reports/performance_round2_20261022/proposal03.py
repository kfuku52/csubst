"""Evaluate an event-vector sidecar inside the existing bounded scan cache.

Run: python reports/performance_round2_20261022/proposal03.py
Production modules are patched only within this process, then restored.
"""

import argparse
import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import cli, substitution_scan as scan  # noqa: E402


class EventRecord:
    """Private immutable-by-contract sidecar; the public frame is preserved."""

    __slots__ = ('frame', 'values')

    def __init__(self, frame, num_branch):
        self.frame = frame
        self.values = np.zeros(num_branch, dtype=np.float64)
        ids = frame['branch_id'].to_numpy(dtype=np.int64)
        assert frame['branch_id'].is_unique
        assert ((ids >= 0) & (ids < num_branch)).all()
        self.values[ids] = frame['event_pp'].fillna(0).to_numpy(dtype=np.float64)
        self.values.flags.writeable = False

    def size(self):
        return (int(self.frame.memory_usage(index=True, deep=True).sum())
                + sys.getsizeof(self.values) + sys.getsizeof(self))


def numeric_events(candidate_events, branch_ids):
    if isinstance(candidate_events, EventRecord):
        values = candidate_events.values
        valid = (branch_ids >= 0) & (branch_ids < len(values))
        result = np.zeros(len(branch_ids), dtype=np.float64)
        result[valid] = values[branch_ids[valid]]
        return result
    # Public DataFrame and called-event paths keep the existing algorithm,
    # especially pandas' compensated summation for duplicate branch rows.
    event_ids = candidate_events['branch_id']
    if event_ids.is_unique:
        mapping = dict(zip(event_ids, candidate_events['event_pp'].fillna(0)))
    else:
        mapping = candidate_events.groupby('branch_id')['event_pp'].sum().to_dict()
    return np.array([float(mapping.get(int(bid), 0.0)) for bid in branch_ids], dtype=np.float64)


def cached_events(scan_static, sub_tensor, site, from_ids, to_ids, float_tol):
    projection = scan_static.get('rate_event_projection') if scan_static is not None else None

    def build():
        frame = scan.extract_candidate_posterior_events(
            sub_tensor, site, from_ids, to_ids, float_tol, projection)
        return EventRecord(frame, int(sub_tensor.shape[0]))

    cache = scan._scan_candidate_cache(scan_static, '_event_cache')
    if cache is None:
        # One-pass calls gain no reuse, so retain the ordinary frame.
        return scan.extract_candidate_posterior_events(
            sub_tensor, site, from_ids, to_ids, float_tol, projection)
    key = ('event_record', id(sub_tensor), int(site), tuple(from_ids),
           tuple(to_ids), float_tol, id(projection))
    return cache.obtain(key, build, lambda value: value.size())


def make_prototype():
    source = inspect.getsource(scan._rate_summary)
    start = source.index('    event_ids = candidate_events["branch_id"]')
    end = source.index('    length_col = _length_column(rate_length)', start)
    rewritten = source[:start] + '    event_values = numeric_events(candidate_events, branch_ids)\n' + source[end:]
    namespace = dict(scan.__dict__, numeric_events=numeric_events)
    exec(compile(rewritten, '<proposal03-event-vector>', 'exec'), namespace)
    reducer = namespace['_rate_summary']
    namespace = dict(scan.__dict__, _cached_posterior_events=cached_events, _rate_summary=reducer)
    fixed_source = inspect.getsource(scan._candidate_fixed_permutation_scores)
    exec(compile(fixed_source, '<proposal03-fixed-scores>', 'exec'), namespace)
    return reducer, namespace['_candidate_fixed_permutation_scores'], {
        'rate_summary': hashlib.sha256(source.encode()).hexdigest(),
        'fixed_scores': hashlib.sha256(fixed_source.encode()).hexdigest(),
    }


def outcome(function, options):
    try:
        return ('result', function(**options))
    except Exception as exc:
        return ('error', type(exc).__name__, str(exc))


def edge_checks(reducer):
    branches = pd.DataFrame({'branch_id': [9, 2, 5, -1, 1000000000000],
                             'parent_id': [0] * 5, 'raw_length': [0.1] * 5,
                             'sn_rescaled_length': [0.2] * 5,
                             'n_rescaled_length': [0.3] * 5})
    base = dict(branch_meta=branches, state_nsy=np.full((10, 2, 2), 0.5), site=0,
                from_ids=[0], to_ids=[1], target_branch_ids=[2, 9], other_branch_ids=[2, 5],
                rate_length='raw', q_matrix=np.array([[-1., 1.], [1., -1.]]))
    checks = []
    for mode, frame in (
        ('reordered_gaps_nan', pd.DataFrame({'branch_id': [5, 9, 2], 'event_pp': [0.1, np.nan, 0.2]})),
        ('empty', pd.DataFrame({'branch_id': [], 'event_pp': []})),
        ('negative', pd.DataFrame({'branch_id': [2], 'event_pp': [-1.]})),
        ('nonfinite', pd.DataFrame({'branch_id': [2], 'event_pp': [np.inf]})),
        ('duplicate_compensated', pd.DataFrame({'branch_id': [2] * 3, 'event_pp': [1e16, 1., -1e16]}))):
        prepared = frame if not frame['branch_id'].is_unique else EventRecord(frame, 10)
        for exposure in ('raw_branch_length', 'state_aware', 'q_weighted'):
            for score_only in (False, True):
                options = dict(base, candidate_events=frame, rate_exposure=exposure, score_only=score_only)
                before = outcome(scan._rate_summary, options)
                after = outcome(reducer, dict(options, candidate_events=prepared))
                np.testing.assert_equal(before, after)
                checks.append({'case': mode, 'exposure': exposure, 'score_only': score_only,
                               'exact_outcome': True, 'outcome': before[0]})
    original_expected = scan.scan_endpoint.expected_events
    try:
        for mode, exposure in (('normal', [0.1] * 5), ('positive_zero', [0.1, 0., 0.1, 0.1, 0.1]),
                               ('zero_control', [0.1, 0.1, 0., 0.1, 0.1])):
            def expected(*args, values=exposure):
                return np.array(values), np.zeros(5, dtype=bool), np.full(5, 'ok', dtype='U24')

            scan.scan_endpoint.expected_events = expected
            frame = pd.DataFrame({'branch_id': [2], 'event_pp': [0.5]})
            context = {'branch_ids': branches['branch_id'].to_numpy()}
            for score_only in (False, True):
                options = dict(base, candidate_events=frame, rate_exposure='endpoint',
                               endpoint_context=context, score_only=score_only)
                before = outcome(scan._rate_summary, options)
                after = outcome(reducer, dict(options, candidate_events=EventRecord(frame, 10)))
                np.testing.assert_equal(before, after)
                checks.append({'case': mode, 'exposure': 'endpoint_stub', 'score_only': score_only,
                               'exact_outcome': True, 'outcome': before[0]})
        for bad in ('length', 'order', 'no_context'):
            options = dict(base, candidate_events=frame, rate_exposure='endpoint', endpoint_context=context)
            if bad == 'length':
                options['rate_length'] = 'n_rescaled'
            elif bad == 'order':
                options['endpoint_context'] = {'branch_ids': np.array([1, 2, 3, 4, 5])}
            else:
                options['endpoint_context'] = None
            before = outcome(scan._rate_summary, options)
            after = outcome(reducer, dict(options, candidate_events=EventRecord(frame, 10)))
            np.testing.assert_equal(before, after)
            assert before[0] == 'error'
            checks.append({'case': bad, 'exposure': 'endpoint_validation', 'exact_outcome': True})
    finally:
        scan.scan_endpoint.expected_events = original_expected
    frame = pd.DataFrame({'branch_id': [2], 'event_pp': [0.5]})
    record = EventRecord(frame, 10)
    try:
        record.values[2] = 2
    except ValueError:
        pass
    else:
        raise AssertionError('numeric sidecar was writable')
    # Existing cache enforces both limits with the sidecar included in size.
    cache = scan._CandidateCache(max_bytes=record.size() + 1, max_items=2)
    cache.obtain('old', lambda: record, lambda item: item.size())
    cache.obtain('new', lambda: record, lambda item: item.size())
    assert list(cache.entries) == ['new'] and cache.nbytes <= cache.max_bytes
    checks.append({'case': 'readonly_and_byte_eviction', 'exact_outcome': True})
    cache = scan._CandidateCache(max_bytes=100 * record.size(), max_items=2)
    for key in ('first', 'second', 'third'):
        cache.obtain(key, lambda: record, lambda item: item.size())
    assert list(cache.entries) == ['second', 'third']
    checks.append({'case': 'item_eviction', 'exact_outcome': True})
    tensor = np.zeros((10, 2, 1, 2, 2))
    tensor[2, 0, 0, 0, 1] = 0.5
    static = {}
    first = cached_events(static, tensor, 0, [0], [1], 1e-12)
    again = cached_events(static, tensor, 0, [0], [1], 1e-12)
    other_site = cached_events(static, tensor, 1, [0], [1], 1e-12)
    other_tensor = cached_events(static, tensor.copy(), 0, [0], [1], 1e-12)
    assert first is again and first is not other_site and first is not other_tensor
    disabled = cached_events({'_cache_candidates': False}, tensor, 0, [0], [1], 1e-12)
    assert isinstance(disabled, pd.DataFrame)
    pd.testing.assert_frame_equal(first.frame, disabled, check_exact=True)
    checks.append({'case': 'reuse_fit_site_keys_and_disabled_cache_frame', 'exact_outcome': True})
    return checks


class Captured(Exception):
    pass


def real_case(dataset, prototype, workdir):
    result = {}
    original_chunk = scan._run_scan_permutation_chunk
    original_fixed = scan._candidate_fixed_permutation_scores

    def compare(*args, **kwargs):
        values = inspect.signature(original_chunk).bind(*args, **kwargs).arguments
        assert values['calibration'] == 'candidate_fixed'
        runs = {'before': [], 'event_vector': []}
        for repeat in range(4):
            outputs = {}
            for label in (('before', 'event_vector') if repeat % 2 == 0 else ('event_vector', 'before')):
                scan._candidate_fixed_permutation_scores = original_fixed if label == 'before' else prototype
                try:
                    outputs[label], timing = measure(original_chunk, *args, **kwargs)
                finally:
                    scan._candidate_fixed_permutation_scores = original_fixed
                if repeat:
                    runs[label].append(timing)
            np.testing.assert_equal(outputs['before'], outputs['event_vector'])
        # Populate an isolated local cache to measure actual retained array and
        # frame bytes for the observed candidates, outside timed repetitions.
        _, unpacked = scan._unpack_scan_worker_context(values['g'], values['scan_static'])
        tensor = scan._unpack_scan_tensor_for_worker(values['rate_ON_tensor'])
        for row in values['observed_df'].to_dict('records'):
            cached_events(unpacked, tensor, int(row['site']), scan._parse_id_list(row['from_state_ids']),
                          scan._parse_id_list(row['to_state_ids']), float(values['g'].get('float_tol', 1e-12)))
        event_cache = unpacked.get('_event_cache')
        entries = [entry[0] for entry in event_cache.entries.values()] if event_cache is not None else []
        result.update(candidate_rows=len(values['observed_df']), permutations=len(values['permutation_indices']),
                      success_count=outputs['before']['success_count'], failure_count=outputs['before']['failure_count'],
                      exact_scores_keys_diagnostics=True, runs=runs,
                      summary={label: summarize(samples) for label, samples in runs.items()},
                      retained={'cache_entries': len(entries),
                                'numeric_array_bytes': sum(entry.values.nbytes for entry in entries),
                                'array_object_and_record_bytes': sum(sys.getsizeof(entry.values) - entry.values.nbytes
                                                                    + sys.getsizeof(entry) for entry in entries),
                                'accounted_cache_bytes': 0 if event_cache is None else event_cache.nbytes,
                                'max_cache_bytes': 32 * 1024**2, 'max_cache_items': 4096})
        # Worker transport must continue to drop the existing cache key.
        empty_g = dict(values['g'])
        _, fresh = scan._unpack_scan_worker_context(empty_g, unpacked)
        assert '_event_cache' not in fresh
        result['worker_unpack_drops_cache'] = True
        raise Captured

    data = workdir / (dataset + '_data')
    data.mkdir()
    for original in (ROOT / 'csubst/dataset').glob(dataset + '.*'):
        if original.is_file():
            shutil.copyfile(original, data / original.name)
    argv = ['scan', '--alignment_file', str(data / (dataset + '.alignment.fa')),
            '--rooted_tree_file', str(data / (dataset + '.tree.nwk')),
            '--foreground', str(data / (dataset + '.foreground.txt')),
            '--outdir', str(workdir / dataset), '--threads', '1', '--blas_threads', '1',
            '--scan_site_plot', 'no', '--scan_pvalue_calibration', 'candidate_fixed',
            '--scan_n_permutations', '50', '--random_seed', '8', '--float_digit', '12']
    for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        argv += ['--iqtree_' + suffix, str(data / (dataset + '.alignment.fa.' + suffix))]
    scan._run_scan_permutation_chunk = compare
    try:
        with (workdir / (dataset + '.log')).open('w') as log, contextlib.redirect_stdout(log):
            try:
                cli.main(argv)
            except Captured:
                pass
    finally:
        scan._run_scan_permutation_chunk = original_chunk
        scan._candidate_fixed_permutation_scores = original_fixed
    assert result
    result['input_sha256'] = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted((ROOT / 'csubst/dataset').glob(dataset + '.*')) if path.is_file()}
    return result


def copy_inputs(dataset, workdir):
    data = workdir / (dataset + '_inputs')
    data.mkdir()
    names = [dataset + '.alignment.fa', dataset + '.tree.nwk', dataset + '.foreground.txt']
    names += [dataset + '.alignment.fa.' + suffix for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log')]
    for name in names:
        shutil.copyfile(ROOT / 'csubst/dataset' / name, data / name)
    return data, {name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in names}


def argv_for(dataset, data, outdir):
    argv = ['scan', '--alignment_file', str(data / (dataset + '.alignment.fa')),
            '--rooted_tree_file', str(data / (dataset + '.tree.nwk')),
            '--foreground', str(data / (dataset + '.foreground.txt')), '--outdir', str(outdir),
            '--threads', '1', '--blas_threads', '1', '--scan_site_plot', 'no',
            '--scan_pvalue_calibration', 'candidate_fixed', '--scan_n_permutations', '50',
            '--random_seed', '8', '--float_digit', '12']
    for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        argv += ['--iqtree_' + suffix, str(data / (dataset + '.alignment.fa.' + suffix))]
    return argv


def cli_case(dataset, workdir):
    # The isolated worker/time protocol matches proposal02, which tests a
    # different optimization independently. No proposal02 patch is installed.
    data, hashes = copy_inputs(dataset, workdir)
    runs = {'before': [], 'event_vector': []}
    commands = {}
    for repeat in range(4):
        outputs = {}
        for label in (('before', 'event_vector') if repeat % 2 == 0 else ('event_vector', 'before')):
            outdir = workdir / f'{dataset}_{repeat}_{label}'
            command = ['/usr/bin/time', '-p', '-l', sys.executable, str(Path(__file__).resolve()),
                       '--worker', label, '--dataset', dataset, '--data', str(data), '--outdir', str(outdir)]
            with (workdir / f'{dataset}_{repeat}_{label}.log').open('w') as log:
                completed, timing = measure(subprocess.run, command, stdout=log,
                                            stderr=subprocess.PIPE, text=True, check=True)
            usage = completed.stderr
            timing['cpu_s'] = sum(float(re.search(rf'^{part}\s+([\d.]+)$', usage, re.MULTILINE).group(1))
                                  for part in ('user', 'sys'))
            timing['peak_rss_bytes'] = int(re.search(r'(\d+)\s+maximum resident set size', usage).group(1))
            if repeat:
                runs[label].append(timing)
            outputs[label] = outdir
            commands[label] = ['$PYTHON', 'reports/performance_round2_20261022/proposal03.py',
                               '--worker', label, '--dataset', dataset, '--data', '$COPIED_INPUTS',
                               '--outdir', '$FRESH_OUTDIR']
        for name in ('csubst_scan.tsv', 'csubst_scan_units.tsv'):
            pd.testing.assert_frame_equal(pd.read_csv(outputs['before'] / name, sep='\t'),
                                          pd.read_csv(outputs['event_vector'] / name, sep='\t'), check_exact=True)
        for name in ('csubst_scan_calibration.json', 'csubst_scan_inference.json'):
            before = json.loads((outputs['before'] / name).read_text())
            after = json.loads((outputs['event_vector'] / name).read_text())
            np.testing.assert_equal(before, after)
    summaries = {label: summarize(samples) for label, samples in runs.items()}
    for label in summaries:
        rss = [sample['peak_rss_bytes'] for sample in runs[label]]
        summaries[label]['peak_rss_bytes'] = {'min': min(rss), 'max': max(rss), 'median': float(np.median(rss))}
    return {'scope': 'whole isolated Python subprocess, imports and wrapper startup included',
            'input_sha256': hashes, 'commands': commands,
            'all_scan_and_unit_table_columns_types_order_exact': True,
            'calibration_and_inference_json_exact': True, 'runs': runs, 'summary': summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('before', 'event_vector'))
    parser.add_argument('--dataset', choices=('PGK', 'PEPC'))
    parser.add_argument('--data', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--cli-only', action='store_true')
    args = parser.parse_args()
    reducer, prototype, hashes = make_prototype()
    if args.worker:
        if args.worker == 'event_vector':
            scan._candidate_fixed_permutation_scores = prototype
        cli.main(argv_for(args.dataset, args.data, args.outdir))
        return
    result = {'proposal': 3, 'environment': environment(), 'source_function_sha256': hashes,
              'scope': 'candidate_fixed50 worker chunk, cold caches per trial; not whole CLI',
              'warmups': 1, 'measured_repeats': 3,
              'memory_scope': 'worker retained event-vector/cache bytes; CLI subprocess peak RSS measured separately',
              'cases': {}}
    if args.cli_only:
        result = json.loads((HERE / 'proposal03.json').read_text())
        assert result['source_function_sha256'] == hashes
    result['memory_scope'] = 'worker retained event-vector/cache bytes; CLI subprocess peak RSS measured separately'
    with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-round2-p03-') as directory:
        if not args.cli_only:
            result['edge_checks'] = edge_checks(reducer)
            for dataset in ('PGK', 'PEPC'):
                result['cases'][dataset] = real_case(dataset, prototype, Path(directory))
                print(dataset, result['cases'][dataset]['summary'], flush=True)
        result['cli_cases'] = {}
        for dataset in ('PGK', 'PEPC'):
            subdir = Path(directory) / (dataset + '_cli')
            subdir.mkdir()
            result['cli_cases'][dataset] = cli_case(dataset, subdir)
            print(dataset, 'CLI', result['cli_cases'][dataset]['summary'], flush=True)
    (HERE / 'proposal03.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
