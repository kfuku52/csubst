"""Measure bounded batch scoring of fixed-candidate assignments in isolation.

Run with Python 3.14.7 from the repository root. CSUBST_PERF_BASELINE selects
an extracted 624bb8a snapshot; the session pointer is the fallback.
"""

import argparse
import contextlib
import difflib
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import warnings

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['MPLBACKEND'] = 'Agg'

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASELINE = os.environ.get('CSUBST_PERF_BASELINE')
if not BASELINE:
    BASELINE = Path('/tmp/csubst-round3-baseline-path').read_text().strip()
sys.path.insert(0, BASELINE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import cli, scan_statistics as stats, substitution_scan as scan  # noqa: E402


BLOCK_SIZE = 256


def rate_scores_batch(values):
    """Return scalar-equivalent score/P arrays for an (N, 4) float matrix.

    Scalar math.log/exp deliberately preserve the established libm rounding.
    All inputs here are already float aggregates from _rate_summary.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError('Batch rate scores require an (N, 4) array.')
    size = values.shape[0]
    scores = np.full(size, np.nan)
    pvalues = np.full(size, np.nan)
    valid = np.isfinite(values).all(axis=1)
    valid &= (values[:, (0, 2)] >= 0).all(axis=1)
    valid &= (values[:, (1, 3)] > 0).all(axis=1)
    ids = np.flatnonzero(valid)
    if not ids.size:
        return scores, pvalues
    x, a, y, b = values[ids].T
    positive_x = x > 0
    log_x = np.zeros(len(ids))
    log_y = np.zeros(len(ids))
    log_x[positive_x] = np.fromiter(map(math.log, x[positive_x]), float)
    positive_y = y > 0
    log_y[positive_y] = np.fromiter(map(math.log, y[positive_y]), float)
    log_a = np.fromiter(map(math.log, a), float)
    log_b = np.fromiter(map(math.log, b), float)
    enriched = positive_x & (~positive_y | (log_x - log_a > log_y - log_b))
    scores[ids[~enriched]] = 0.
    pvalues[ids[~enriched]] = 1.
    ids = ids[enriched]
    if not ids.size:
        return scores, pvalues
    x, y = x[enriched], y[enriched]
    log_x, log_y = log_x[enriched], log_y[enriched]
    log_a, log_b = log_a[enriched], log_b[enriched]
    positive_y = y > 0
    log_p = -np.logaddexp(0., log_b - log_a)
    log_q = -np.logaddexp(0., log_a - log_b)
    fraction_x = np.zeros(len(ids))
    fraction_x[positive_y] = -np.logaddexp(0., log_y[positive_y] - log_x[positive_y])
    with np.errstate(over='ignore', invalid='ignore'):
        terms = x * (fraction_x - log_p)
        terms[positive_y] += y[positive_y] * (
            -np.logaddexp(0., log_x[positive_y] - log_y[positive_y]) - log_q[positive_y])
        # Python max(0., nan) returns 0.; fmax preserves that scalar behavior.
        deviance = np.fmax(0., 2 * terms)
    tails = stats.log_ndtr(-np.sqrt(deviance))
    scores[ids] = -tails / math.log(10)
    pvalues[ids] = np.fromiter(map(math.exp, tails), float)
    return scores, pvalues


def flush_batch(pending, out):
    if not pending:
        return
    values = [(row['target_event_count'], row['target_exposure'],
               row['other_event_count'], row['other_exposure']) for _, row in pending]
    scores, _ = rate_scores_batch(values)
    for (key, row), score in zip(pending, scores):
        row['score_rate_enrichment'] = np.nan if row.pop('_invalid_endpoint') else float(score)
        out[key] = stats.calibration_score(row)
    pending.clear()


def prototype():
    summary_source = inspect.getsource(scan._rate_summary)
    before = '''    prepared_branch=None,
):'''
    after = '''    prepared_branch=None,
    defer_score=False,
):'''
    assert summary_source.count(before) == 1
    summary_source = summary_source.replace(before, after)
    before = '''    score, pvalue = scan_statistics.rate_score(
        x_target=target_event,
        l_target=target_exposure,
        x_other=other_event,
        l_other=other_exposure,
    )'''
    after = '''    if defer_score:
        score = pvalue = np.nan
    else:
        score, pvalue = scan_statistics.rate_score(
            x_target=target_event,
            l_target=target_exposure,
            x_other=other_event,
            l_other=other_exposure,
        )'''
    assert summary_source.count(before) == 1
    summary_source = summary_source.replace(before, after)
    before = '''        return {
            "score_rate_enrichment": score,'''
    after = '''        result = {
            "score_rate_enrichment": score,'''
    assert summary_source.count(before) == 1
    summary_source = summary_source.replace(before, after)
    before = '''            "other_exposure": other_exposure,
        }
    target_rate'''
    after = '''            "other_exposure": other_exposure,
        }
        if defer_score:
            result['_invalid_endpoint'] = invalid_endpoint
        return result
    target_rate'''
    assert summary_source.count(before) == 1
    summary_source = summary_source.replace(before, after)
    fixed_source = inspect.getsource(scan._candidate_fixed_permutation_scores)
    fixed_source = fixed_source.replace('    out = {}\n', '    out = {}\n    pending = []\n')
    fixed_source = fixed_source.replace('            score_only=True,\n',
                                        '            score_only=True, defer_score=True,\n')
    before = '''        out[_scan_row_key(row)] = scan_statistics.calibration_score(rate)
    return out'''
    after = '''        pending.append((_scan_row_key(row), rate))
        if len(pending) >= BLOCK_SIZE:
            flush_batch(pending, out)
    flush_batch(pending, out)
    return out'''
    assert fixed_source.count(before) == 1
    fixed_source = fixed_source.replace(before, after)
    namespace = dict(scan.__dict__, flush_batch=flush_batch, BLOCK_SIZE=BLOCK_SIZE)
    exec(compile(summary_source, '<proposal04-summary>', 'exec'), namespace)
    exec(compile(fixed_source, '<proposal04-fixed>', 'exec'), namespace)
    namespace['_candidate_fixed_permutation_scores']._proposal04_source = fixed_source
    scalar_source = inspect.getsource(stats.rate_score).replace(
        'not np.isfinite([x, a, y, b]).all()',
        'not (math.isfinite(x) and math.isfinite(a) and math.isfinite(y) and math.isfinite(b))')
    scalar_namespace = dict(stats.__dict__)
    exec(compile(scalar_source, '<proposal03-scalar-control>', 'exec'), scalar_namespace)
    # Review artifact, deliberately not applied to production.
    module_source = (Path(BASELINE) / 'csubst/substitution_scan.py').read_text()
    altered = module_source.replace(inspect.getsource(scan._rate_summary), summary_source)
    altered = altered.replace(inspect.getsource(scan._candidate_fixed_permutation_scores), fixed_source)
    altered = altered.replace('def _candidate_fixed_permutation_scores(',
                              'BLOCK_SIZE = 256\n\n\n' + inspect.getsource(rate_scores_batch)
                              + '\n\n' + inspect.getsource(flush_batch).replace('stats.', 'scan_statistics.')
                              + '\n\ndef _candidate_fixed_permutation_scores(')
    # The helper's globals must refer to the existing imported statistics module.
    altered = altered.replace('tails = stats.log_ndtr(', 'tails = scan_statistics.log_ndtr(')
    patch = ''.join(difflib.unified_diff(module_source.splitlines(keepends=True),
                    altered.splitlines(keepends=True), fromfile='a/csubst/substitution_scan.py',
                    tofile='b/csubst/substitution_scan.py', n=0))
    (HERE / 'proposal04.patch').write_text(patch)
    return namespace['_candidate_fixed_permutation_scores'], scalar_namespace['rate_score']


def edge_checks():
    grid = (-0.0, 0., np.nextafter(0., 1.), 0.1, 1., 1e-300, 1e300,
            np.finfo(float).max, -1., np.inf, -np.inf, np.nan)
    values = list(itertools.product(grid, repeat=4))
    rng = np.random.default_rng(404)
    with np.errstate(over='ignore'):
        values.extend(10. ** rng.uniform(-323., 308., size=(5000, 4)))
    for scale in (np.nextafter(0., 1.), 1e-300, 0.1, 1., 1e300, np.finfo(float).max):
        with np.errstate(over='ignore'):
            adjacent = (scale, np.nextafter(scale, 0.), np.nextafter(scale, np.inf))
        for other in adjacent:
            values.append((scale, 1., other, 1.))
    matrix = np.asarray(values, float)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        before = np.array([stats.rate_score(*row) for row in matrix])
        after = np.column_stack(rate_scores_batch(matrix))
    exact = (before == after) | (np.isnan(before) & np.isnan(after))
    finite_pair = np.isfinite(before) & np.isfinite(after)
    difference = np.zeros_like(before)
    difference[finite_pair] = np.abs(before[finite_pair] - after[finite_pair])
    mismatches = np.flatnonzero(~exact.all(axis=1))
    examples = [{'values': matrix[i].tolist(), 'before': before[i].tolist(),
                 'after': after[i].tolist()} for i in mismatches[:5]]
    empty = rate_scores_batch(np.empty((0, 4)))
    assert empty[0].shape == empty[1].shape == (0,)
    shape_cases = (np.empty((0,)), np.ones(4), np.ones((2, 3)), np.ones((2, 4, 1)))
    for value in shape_cases:
        try:
            rate_scores_batch(value)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid batch shape accepted.')
    # The deferred summary's explicit endpoint-invalid flag and calibration
    # validator must retain zero-exposure no-tests and invalid-row failures.
    semantics = []
    for raw in ((0., 1., 0., 1.), (1., 0., 0., 1.), (-1., 1., 0., 1.),
                (np.nan, 1., 0., 1.), (1., 1., 0., 1.)):
        for endpoint_invalid in (False, True):
            x, a, y, b = raw
            row = dict(score_rate_enrichment=stats.rate_score(*raw)[0],
                       target_event_count=x, target_exposure=a,
                       other_event_count=y, other_exposure=b)
            if endpoint_invalid:
                row['score_rate_enrichment'] = np.nan
            try:
                expected = ('result', stats.calibration_score(row))
            except ValueError as exc:
                expected = ('error', str(exc))
            deferred = dict(row, _invalid_endpoint=endpoint_invalid)
            try:
                out = {}
                flush_batch([('key', deferred)], out)
                actual = ('result', out['key'])
            except ValueError as exc:
                actual = ('error', str(exc))
            assert expected == actual
            semantics.append(expected[0])
    return dict(cases=len(values), exact_score_and_p_cases=int(exact.all(axis=1).sum()),
                mismatch_cases=len(mismatches), mismatch_examples=examples,
                max_absolute_score_difference=float(difference[:, 0].max()),
                max_absolute_p_difference=float(difference[:, 1].max()),
                warning_count=len(captured), empty_shapes_preserved=True,
                invalid_shapes_rejected=len(shape_cases),
                no_test_failure_endpoint_cases=len(semantics), seed=404)


def compare_timed(functions, warmups, repeats):
    runs = {label: [] for label in functions}
    labels = list(functions)
    mismatch = {}
    reference = None
    for repeat in range(warmups + repeats):
        order = labels if repeat % 2 == 0 else labels[::-1]
        outputs = {}
        for label in order:
            outputs[label], timing = measure(functions[label])
            if repeat >= warmups:
                runs[label].append(timing)
        reference = outputs[labels[0]]
        for label in labels[1:]:
            try:
                np.testing.assert_equal(outputs[labels[0]], outputs[label])
            except AssertionError as exc:
                mismatch[label] = str(exc)[:1000]
    return dict(runs=runs, summary={label: summarize(samples) for label, samples in runs.items()},
                exact_outputs=not mismatch, mismatch=mismatch), reference


def failure_order_check(candidate):
    """Exercise both real candidate loops with two independent invalid rows."""
    class Metadata:
        def __init__(self, _):
            pass

        def prepare(self, *_):
            return None

    def summary(**kwargs):
        if kwargs['site'] == 2:
            raise RuntimeError('Later candidate exposure failed.')
        row = dict(score_rate_enrichment=np.nan, target_event_count=1.,
                   target_exposure=1., other_event_count=0., other_exposure=1.)
        if kwargs.get('defer_score'):
            row['_invalid_endpoint'] = True
        return row

    replacements = dict(
        _rate_summary=summary, _ScanRateMetadata=Metadata,
        _cached_posterior_events=lambda **_: None,
        _target_branch_ids_from_maps=lambda *_: [0],
        _other_branch_ids_from_maps=lambda *_, **__: [1],
        _scan_candidate_cache=lambda *_: None)
    rows = [dict(trait='trait', target_class='all', scan_match='any2any', site=site,
                 from_state_ids='1', to_state_ids='2') for site in (1, 2)]
    fixed = [(row, [1], [2]) for row in rows]
    args = dict(g={'scan_rate_event_mode': 'posterior_sum', 'state_nsy': np.zeros((2, 2, 2))},
                observed_df=pd.DataFrame(rows),
                scan_context={'fg_ids': {}, 'units': pd.DataFrame({'trait': []})},
                branch_meta=pd.DataFrame(), valid_branch_ids=np.array([0, 1]),
                ON_tensor=None, rate_ON_tensor=None,
                scan_static={'rate_exposure': 'endpoint', '_fixed_candidates': fixed,
                             'q_context': dict(q_matrix=None, state_cdn=None, codon_q_matrix=None,
                                               codon_state_ids=None, endpoint_context=None)})
    outcomes = {}
    for label, source, globals_dict in (
            ('baseline', inspect.getsource(scan._candidate_fixed_permutation_scores), scan.__dict__),
            ('batch04', candidate._proposal04_source, candidate.__globals__)):
        namespace = dict(globals_dict, **replacements)
        exec(compile(source, '<proposal04-failure-order>', 'exec'), namespace)
        try:
            namespace['_candidate_fixed_permutation_scores'](**args)
        except Exception as exc:
            outcomes[label] = '{}: {}'.format(type(exc).__name__, str(exc))
        else:
            raise AssertionError('Expected invalid candidate failure.')
    return dict(outcomes, same_failure_reason=outcomes['baseline'] == outcomes['batch04'],
                consequence='Both runs fail, but batching can replace the first invalid-score reason with a later exposure failure.')


class Captured(Exception):
    pass


def phase_case(dataset, candidate, scalar_control, workdir, warmups, repeats, permutations):
    original_score = stats.rate_score
    original_fixed = scan._candidate_fixed_permutation_scores
    original_chunk = scan._run_scan_permutation_chunk
    result = {}

    def capture(*args, **kwargs):
        values = inspect.signature(original_chunk).bind(*args, **kwargs).arguments
        assert values['calibration'] == 'candidate_fixed'

        def run(label):
            stats.rate_score = scalar_control if label == 'scalar03' else original_score
            scan._candidate_fixed_permutation_scores = candidate if label == 'batch04' else original_fixed
            try:
                return original_chunk(*args, **kwargs)
            finally:
                stats.rate_score = original_score
                scan._candidate_fixed_permutation_scores = original_fixed

        functions = {label: lambda label=label: run(label)
                     for label in ('baseline', 'scalar03', 'batch04')}
        timing, output = compare_timed(functions, warmups, repeats)
        result.update(timing, candidates=len(values['observed_df']),
                      permutations=len(values['permutation_indices']),
                      success_count=output['success_count'], failure_count=output['failure_count'],
                      scope='candidate_fixed permutation chunk; fresh worker caches; not CLI')
        raise Captured

    data = workdir / (dataset + '_inputs')
    data.mkdir()
    paths = [dataset + suffix for suffix in ('.alignment.fa', '.tree.nwk', '.foreground.txt')]
    paths.extend(dataset + '.alignment.fa.' + suffix
                 for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log'))
    hashes = {}
    for name in paths:
        shutil.copyfile(Path(BASELINE) / 'csubst/dataset' / name, data / name)
        hashes[name] = hashlib.sha256((data / name).read_bytes()).hexdigest()
    argv = ['scan', '--alignment_file', str(data / (dataset + '.alignment.fa')),
            '--rooted_tree_file', str(data / (dataset + '.tree.nwk')),
            '--foreground', str(data / (dataset + '.foreground.txt')),
            '--outdir', str(workdir / (dataset + '_output')),
            '--threads', '1', '--blas_threads', '1', '--scan_site_plot', 'no',
            '--scan_pvalue_calibration', 'candidate_fixed', '--scan_n_permutations', str(permutations),
            '--random_seed', '8', '--float_digit', '12']
    for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        argv.extend(['--iqtree_' + suffix, str(data / (dataset + '.alignment.fa.' + suffix))])
    scan._run_scan_permutation_chunk = capture
    try:
        with (workdir / (dataset + '.log')).open('w') as log, contextlib.redirect_stdout(log):
            try:
                cli.main(argv)
            except Captured:
                pass
    finally:
        scan._run_scan_permutation_chunk = original_chunk
        stats.rate_score = original_score
        scan._candidate_fixed_permutation_scores = original_fixed
    assert result
    return dict(result, input_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', action='store_true',
                        help='Run optional rejected-prototype timings after correctness checks.')
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--permutations', type=int, default=50)
    args = parser.parse_args()
    candidate, scalar_control = prototype()
    result = dict(proposal=4, baseline_commit='624bb8a6dae536d0fa10891f666684ef1ad3dab0',
                  environment=environment(),
                  benchmark_configuration=dict(warmups=args.warmups, repeats=args.repeats,
                                               permutations=args.permutations),
                  benchmark_executed=args.benchmark,
                  block_size=BLOCK_SIZE,
                  command='python reports/performance_round3_20261023/proposal04.py',
                  source_sha256={name: hashlib.sha256((Path(BASELINE) / 'csubst' / name).read_bytes()).hexdigest()
                                 for name in ('scan_statistics.py', 'substitution_scan.py')},
                  memory='At most 256 pending row dictionaries plus O(256) numeric temporaries; peak RSS not measured.',
                  cases={})
    result['edge_checks'] = edge_checks()
    result['failure_order_check'] = failure_order_check(candidate)
    result['decision'] = 'reject'
    result['reason'] = 'Deferred validation changes the first reported failure reason.'
    result['performance_status'] = (
        'Optional rejected-prototype timings requested.' if args.benchmark else
        'Not run: correctness regression rejected the prototype before measurement. '
        'An earlier timing process was terminated while waiting for the shared lock.')
    if args.benchmark:
        print('Proposal04 waiting for shared benchmark lock.', flush=True)
        with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-p04-') as temporary:
            for dataset in ('PGK', 'PEPC'):
                result['cases'][dataset] = phase_case(dataset, candidate, scalar_control, Path(temporary),
                                                      args.warmups, args.repeats, args.permutations)
                (HERE / 'proposal04.json').write_text(json.dumps(result, indent=2) + '\n')
                print('Proposal04', dataset, result['cases'][dataset]['summary'], flush=True)
    (HERE / 'proposal04.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Proposal04 rejected:', result['failure_order_check'], flush=True)


if __name__ == '__main__':
    main()
