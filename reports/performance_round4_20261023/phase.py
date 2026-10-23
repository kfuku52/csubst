"""Measure isolated changes on arguments captured from real fixed-fit CLI runs.

python reports/performance_round4_20261023/phase.py --baseline <SOURCE> --result <JSON>
The baseline contains csubst/, including the same native binaries as the checkout.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['MPLBACKEND'] = 'Agg'

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'reports/performance_round3_20261023'))
from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from variants import (SELECTION_METHOD, compile_variant, diagnostic_source,  # noqa: E402
                      endpoint_source, scalar_source)


def comparator():
    path = ROOT / 'reports/performance_round2_implementation_20261022/benchmark.py'
    spec = importlib.util.spec_from_file_location('round4_cli', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_timed(before, after, count, repeats):
    before()  # Excluded warmup of both implementations, including lazy caches.
    after()
    samples = {'before': [], 'after': []}
    functions = {'before': before, 'after': after}
    for repeat in range(repeats):
        for label in (('before', 'after') if repeat % 2 == 0 else ('after', 'before')):
            def run():
                for _ in range(count):
                    functions[label]()
            _, sample = measure(run)
            samples[label].append(sample)
    return {'iterations_per_sample': count, 'repeats': repeats, 'warmups': 1,
            'runs': samples, 'summary': {k: summarize(v) for k, v in samples.items()}}


def count_allocations(before, after, count, repeats):
    """Partial constructor counters; these are neither live memory nor RSS."""
    import numpy as np
    before()
    after()
    original = {name: getattr(np, name) for name in ('ones_like', 'full', 'zeros')}
    result = {}
    for label, function in (('before', before), ('after', after)):
        counts = {name: {'calls': 0, 'cumulative_bytes': 0} for name in original}
        def counted(name):
            def wrapper(*a, **kw):
                value = original[name](*a, **kw)
                counts[name]['calls'] += 1
                counts[name]['cumulative_bytes'] += value.nbytes
                return value
            return wrapper
        try:
            for name in original:
                setattr(np, name, counted(name))
            function()
        finally:
            for name, constructor in original.items():
                setattr(np, name, constructor)
        result[label] = counts
    return {'constructor_counts_one_captured_pass': result}


def compare_endpoint(old_iter, arguments, compare, repeats):
    import numpy as np
    a, kw = arguments
    new_iter = compile_variant(old_iter, endpoint_source)
    records = 0
    for first, second in zip(old_iter(*a, **kw), new_iter(*a, **kw), strict=True):
        np.testing.assert_equal(vars(first), vars(second))
        records += 1
    def consume(function):
        return lambda: sum(record.child for record in function(*a, **kw))
    result = compare(consume(old_iter), consume(new_iter), 1, repeats)
    result['exact_records'] = records
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--allocations-only', action='store_true')
    parser.add_argument('--cases', nargs='+', default=['pepc_fixed200', 'pgk_marginal200'],
                        choices=['pepc_fixed200', 'pgk_marginal200', 'pgk_search', 'pepc_search'])
    args = parser.parse_args()
    compare = count_allocations if args.allocations_only else compare_timed
    sys.path.insert(0, str(args.baseline))
    import numpy as np
    from csubst import cli, endpoint, scan_statistics, substitution_scan as scan

    old_rate = scan._rate_summary
    old_iter = endpoint.EndpointModel.iter_blocks
    scope = ('partial NumPy constructor counts, one captured pass; not live memory or RSS'
             if args.allocations_only else 'in-process phase timings, excluding CLI startup')
    result = {'environment': environment(), 'scope': scope,
              'source_sha256': {name: hashlib.sha256((args.baseline / 'csubst' / name).read_bytes()).hexdigest()
                                for name in ('endpoint.py', 'substitution_scan.py', 'scan_statistics.py')},
              'cases': {}}
    b = comparator()
    b.CASES['pgk_marginal200'] = ('PGK', 'scan', {
        'substitution_posterior': 'marginal', 'scan_observation': 'marginal',
        'scan_rate_exposure': 'raw_branch_length', 'scan_rate_length': 'raw',
        'scan_pvalue_calibration': 'candidate_fixed', 'scan_n_permutations': '200'})
    with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-round4-phase-') as temporary:
        for case in args.cases:
            calls, iter_calls, scores = [], [], []
            def capture(*a, **kw):
                value = old_rate(*a, **kw)
                if kw.get('score_only') and len(calls) < 206:
                    calls.append((a, kw))
                    scores.append(value)
                return value
            def capture_iter(*a, **kw):
                if not iter_calls:
                    iter_calls.append((a, kw))
                return old_iter(*a, **kw)
            scan._rate_summary = capture
            endpoint.EndpointModel.iter_blocks = capture_iter
            directory = Path(temporary) / case
            directory.mkdir()
            data, hashes = b.copy_inputs(b.CASES[case][0], args.baseline, directory)
            argv = b.command(case, args.baseline, data, directory / 'output', 'none')[5:]
            with (directory / 'cli.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                cli.main(argv)
            scan._rate_summary = old_rate
            endpoint.EndpointModel.iter_blocks = old_iter
            result['cases'][case] = {'input_sha256': hashes, 'captured_rate_calls': len(calls)}
            if case.endswith('_search'):
                result['cases'][case]['endpoint'] = compare_endpoint(old_iter, iter_calls[0], compare, args.repeats)
                continue
            if case == 'pepc_fixed200':
                original_metadata = scan._ScanRateMetadata
                namespace = dict(scan.__dict__)
                exec(compile(textwrap.dedent(SELECTION_METHOD), '<selection>', 'exec'), namespace)
                original_metadata.select_events = namespace['select_events']
                for a, kw in calls:
                    kw['prepared_branch'][0]._event_selection = None
                selected = compile_variant(old_rate, lambda s: s.replace(
                    'event_values = candidate_events.select(branch_ids)',
                    'event_values = branch_data.select_events(candidate_events)'))
                for a, kw in calls:
                    np.testing.assert_equal(old_rate(*a, **kw), selected(*a, **kw))
                def run_rates(function):
                    return lambda: [function(*a, **kw) for a, kw in calls]
                result['cases'][case]['selection'] = compare(
                    run_rates(old_rate), run_rates(selected), 100, args.repeats)
                old_score = scan_statistics.calibration_score
                new_score = compile_variant(old_score, scalar_source)
                for row in scores:
                    np.testing.assert_equal(old_score(row), new_score(row))
                result['cases'][case]['scalar_score'] = compare(
                    lambda: [old_score(row) for row in scores],
                    lambda: [new_score(row) for row in scores], 1000, args.repeats)
                result['cases'][case]['endpoint'] = compare_endpoint(old_iter, iter_calls[0], compare, args.repeats)
            else:
                new_rate = compile_variant(old_rate, diagnostic_source)
                for a, kw in calls:
                    np.testing.assert_equal(old_rate(*a, **kw), new_rate(*a, **kw))
                result['cases'][case]['diagnostics'] = compare(
                    lambda: [old_rate(*a, **kw) for a, kw in calls],
                    lambda: [new_rate(*a, **kw) for a, kw in calls], 100, args.repeats)
        args.result.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({name: {key: value['summary'] for key, value in case.items()
                                if isinstance(value, dict) and 'summary' in value}
                          for name, case in result['cases'].items()}, indent=2))


if __name__ == '__main__':
    main()
