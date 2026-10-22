"""Evaluate a score-only reducer for fixed-candidate assignment calibration.

Run from repository root: python reports/performance_candidates_20261022/proposal07.py
Production files remain untouched; only this process patches one call boundary.
"""

import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
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
from csubst import cli, scan_statistics, substitution_scan as scan  # noqa: E402

REQUIRED = ('score_rate_enrichment', 'target_event_count', 'other_event_count',
            'target_exposure', 'other_exposure')


def make_prototype():
    original = inspect.getsource(scan._rate_summary)
    begin = original.index('    target_rate = target_event / target_exposure')
    stop = original.index('    score, pvalue = scan_statistics.rate_score(')
    source = original[:begin] + original[stop:]
    begin = source.index('    if impossible[is_target | is_other].any():')
    source = source[:begin] + '''    if rate_exposure == "endpoint" and impossible[is_target | is_other].any():
        score = np.nan
    return {
        "score_rate_enrichment": score,
        "target_event_count": target_event,
        "other_event_count": other_event,
        "target_exposure": target_exposure,
        "other_exposure": other_exposure,
    }
'''
    namespace = dict(scan.__dict__)
    exec(compile(source, '<proposal07-score-only>', 'exec'), namespace)
    reducer = namespace['_rate_summary']
    fixed_source = inspect.getsource(scan._candidate_fixed_permutation_scores)
    namespace = dict(scan.__dict__, _rate_summary=reducer)
    exec(compile(fixed_source, '<proposal07-fixed-candidates>', 'exec'), namespace)
    return reducer, namespace['_candidate_fixed_permutation_scores'], {
        'rate_summary': hashlib.sha256(original.encode()).hexdigest(),
        'fixed_scores': hashlib.sha256(fixed_source.encode()).hexdigest(),
    }


def score_outcome(function, options):
    try:
        row = function(**options)
        return ('score', scan_statistics.calibration_score(row)), row
    except Exception as exc:
        return ('error', type(exc).__name__, str(exc)), None


def edge_checks(reducer):
    branch = pd.DataFrame({'branch_id': [1, 2, 3], 'parent_id': [0, 0, 0],
                           'raw_length': [0.1, 0.2, 0.3],
                           'sn_rescaled_length': [0.2, 0.4, 0.6],
                           'n_rescaled_length': [0.3, 0.6, 0.9]})
    states = np.full((4, 2, 2), 0.5)
    base = dict(candidate_events=pd.DataFrame({'branch_id': [1, 2], 'event_pp': [0.75, 0.1]}),
                branch_meta=branch, state_nsy=states, site=0, from_ids=np.array([0]),
                to_ids=np.array([1]), target_branch_ids=np.array([1]),
                other_branch_ids=np.array([2, 3]), rate_length='raw', rate_exposure='raw_branch_length')
    cases = []
    for exposure in ('raw_branch_length', 'state_aware', 'q_weighted'):
        for mode in ('normal', 'zero_target', 'zero_control', 'missing', 'negative', 'infinite', 'empty_events'):
            options = dict(base, rate_exposure=exposure, branch_meta=branch.copy(), state_nsy=states.copy(),
                           q_matrix=np.array([[-1., 1.], [1., -1.]]))
            options['candidate_events'] = base['candidate_events'].copy()
            if mode == 'zero_target':
                options['branch_meta'].loc[0, 'raw_length'] = 0
            elif mode == 'zero_control':
                options['branch_meta'].loc[1:, 'raw_length'] = 0
            elif mode == 'missing':
                options['state_nsy'][0, 0] = 0
            elif mode == 'negative':
                options['candidate_events'].loc[0, 'event_pp'] = -0.5
            elif mode == 'infinite':
                options['candidate_events'].loc[0, 'event_pp'] = np.inf
            elif mode == 'empty_events':
                options['candidate_events'] = options['candidate_events'].iloc[:0]
            before, full = score_outcome(scan._rate_summary, options)
            after, compact = score_outcome(reducer, options)
            np.testing.assert_equal(before, after)
            if full is not None and compact is not None:
                np.testing.assert_equal({key: full[key] for key in REQUIRED}, compact)
            cases.append({'exposure': exposure, 'case': mode, 'outcome': before[0],
                          'score': None if before[0] != 'score' or not np.isfinite(before[1]) else before[1]})
    # Endpoint-specific validation and positive-event/zero-exposure handling are
    # retained verbatim. A stub supplies each valid/invalid exposure outcome.
    old_expected = scan.scan_endpoint.expected_events
    try:
        context = {'branch_ids': np.array([1, 2, 3]), 'model': 'synthetic'}
        for mode, exposure, missing in (
                ('normal', [0.1, 0.2, 0.3], [False] * 3),
                ('positive_zero', [0., 0.2, 0.3], [False] * 3),
                ('missing', [0., 0.2, 0.3], [True, False, False]),
                ('zero_control', [0.1, 0., 0.], [False] * 3)):
            def expected(*args, values=exposure, absent=missing):
                return np.array(values), np.array(absent), np.full(3, 'ok', dtype='U24')

            scan.scan_endpoint.expected_events = expected
            options = dict(base, rate_exposure='endpoint', endpoint_context=context)
            before, full = score_outcome(scan._rate_summary, options)
            after, compact = score_outcome(reducer, options)
            np.testing.assert_equal(before, after)
            if full is not None:
                np.testing.assert_equal({key: full[key] for key in REQUIRED}, compact)
            cases.append({'exposure': 'endpoint_stub', 'case': mode, 'outcome': before[0]})
        for bad in ('length', 'order', 'no_context'):
            options = dict(base, rate_exposure='endpoint', endpoint_context=context)
            if bad == 'length':
                options['rate_length'] = 'n_rescaled'
            elif bad == 'order':
                options['endpoint_context'] = dict(context, branch_ids=np.array([3, 2, 1]))
            else:
                options['endpoint_context'] = None
            before, _ = score_outcome(scan._rate_summary, options)
            after, _ = score_outcome(reducer, options)
            np.testing.assert_equal(before, after)
            assert before[0] == 'error'
            cases.append({'exposure': 'endpoint_validation', 'case': bad, 'outcome': 'error'})
    finally:
        scan.scan_endpoint.expected_events = old_expected
    return cases


class Captured(Exception):
    pass


def real_case(dataset, reducer, prototype, workdir):
    original_chunk = scan._run_scan_permutation_chunk
    original_fixed = scan._candidate_fixed_permutation_scores
    original_summary = scan._rate_summary
    result = {}

    def compare(*args, **kwargs):
        bound = inspect.signature(original_chunk).bind(*args, **kwargs)
        values = bound.arguments
        assert values['calibration'] == 'candidate_fixed'
        runs = {'before': [], 'score_only': []}
        example = []

        def save_summary(**options):
            if not example:
                example.append(options)
            return original_summary(**options)

        for repeat in range(4):
            outputs = {}
            for label in (('before', 'score_only') if repeat % 2 == 0 else ('score_only', 'before')):
                scan._candidate_fixed_permutation_scores = original_fixed if label == 'before' else prototype
                if repeat == 0 and label == 'before':
                    scan._rate_summary = save_summary
                try:
                    outputs[label], timing = measure(original_chunk, *args, **kwargs)
                finally:
                    scan._candidate_fixed_permutation_scores = original_fixed
                    scan._rate_summary = original_summary
                if repeat:
                    runs[label].append(timing)
            np.testing.assert_equal(outputs['before'], outputs['score_only'])
        result.update(candidate_rows=len(values['observed_df']),
                      permutations=len(values['permutation_indices']),
                      success_count=outputs['before']['success_count'],
                      failure_count=outputs['before']['failure_count'],
                      exact_scores_keys_diagnostics=True,
                      runs=runs, summary={key: summarize(samples) for key, samples in runs.items()})
        # Empty candidate families are successful no-test trials, not errors.
        empty = dict(values, observed_df=values['observed_df'].iloc[:0], observed_keys=set())
        before = original_chunk(**empty)
        scan._candidate_fixed_permutation_scores = prototype
        try:
            after = original_chunk(**empty)
        finally:
            scan._candidate_fixed_permutation_scores = original_fixed
        np.testing.assert_equal(before, after)
        result['empty_family_exact_parity'] = True
        state_checks = []
        options = example[0]
        for mode in ('negative', 'nonfinite', 'unnormalized', 'missing_child'):
            altered = dict(options, exposure_cache=None, state_cdn=np.array(options['state_cdn']))
            context = options['endpoint_context']
            node = int(context['parent_ids'][0])
            if mode == 'missing_child':
                node = int(context['branch_ids'][0])
            row = altered['state_cdn'][node, options['site']]
            if mode == 'negative':
                row[0] = -1
            elif mode == 'nonfinite':
                row[0] = np.nan
            elif mode == 'unnormalized':
                row[:] = 2
            else:
                row[:] = 0
            left, full = score_outcome(original_summary, altered)
            right, compact = score_outcome(reducer, altered)
            np.testing.assert_equal(left, right)
            if full is not None:
                np.testing.assert_equal({key: full[key] for key in REQUIRED}, compact)
            if mode != 'missing_child':
                assert left[0] == 'error'
            state_checks.append({'case': mode, 'outcome': left[0]})
        result['real_endpoint_state_checks'] = state_checks
        raise Captured

    data = ROOT / 'csubst/dataset'
    argv = ['scan', '--alignment_file', str(data / (dataset + '.alignment.fa')),
            '--rooted_tree_file', str(data / (dataset + '.tree.nwk')),
            '--foreground', str(data / (dataset + '.foreground.txt')),
            '--outdir', str(workdir / dataset), '--threads', '1', '--blas_threads', '1',
            '--scan_site_plot', 'no', '--scan_pvalue_calibration', 'candidate_fixed',
            '--scan_n_permutations', '10', '--random_seed', '8', '--float_digit', '12']
    for key in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        argv += ['--iqtree_' + key, str(data / (dataset + '.alignment.fa.' + key))]
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
        scan._rate_summary = original_summary
    assert result
    result['input_sha256'] = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(data.glob(dataset + '.*')) if path.is_file()}
    return result


def main():
    reducer, fixed, hashes = make_prototype()
    result = {'proposal': 7, 'environment': environment(), 'source_function_sha256': hashes,
              'scope': 'fixed-candidate worker chunk including one-time candidate parsing; not CLI',
              'warmups': 1, 'measured_repeats': 3,
              'revision': 'ID parsing already hoisted in current code; evaluate score-only reducer instead',
              'thread_control': 'VECLIB/OPENBLAS/OMP/MKL=1 before NumPy/SciPy import',
              'memory_scope': 'mapping item count only; peak RSS not measured', 'cases': {}}
    with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-proposal07-') as directory:
        result['edge_checks'] = edge_checks(reducer)
        for dataset in ('PGK', 'PEPC'):
            result['cases'][dataset] = real_case(dataset, reducer, fixed, Path(directory))
            print(dataset, result['cases'][dataset]['summary'], flush=True)
    (HERE / 'proposal07.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
