"""Measure scalar rate-score validation without editing the production module.

Run: python reports/performance_round3_20261023/proposal03.py
Set CSUBST_PERF_BASELINE to an extracted 624bb8a snapshot, or use the current
session's /tmp/csubst-round3-baseline-path pointer.
"""

import argparse
import contextlib
from decimal import Decimal
import difflib
from fractions import Fraction
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

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import cli, scan_statistics as stats, substitution_scan as scan  # noqa: E402


OLD = '    if not np.isfinite([x, a, y, b]).all() or min(x, y) < 0 or min(a, b) <= 0:'
NEW = '''    if (not (math.isfinite(x) and math.isfinite(a) and math.isfinite(y) and math.isfinite(b))
            or min(x, y) < 0 or min(a, b) <= 0):'''


def prototype():
    source = inspect.getsource(stats.rate_score)
    assert source.count(OLD) == 1
    namespace = dict(stats.__dict__)
    exec(compile(source.replace(OLD, NEW), '<proposal03-scalar-finite>', 'exec'), namespace)
    module_path = Path(BASELINE) / 'csubst/scan_statistics.py'
    module_source = module_path.read_text()
    patch = ''.join(difflib.unified_diff(
        module_source.splitlines(keepends=True),
        module_source.replace(OLD, NEW).splitlines(keepends=True),
        fromfile='a/csubst/scan_statistics.py', tofile='b/csubst/scan_statistics.py', n=0))
    (HERE / 'proposal03.patch').write_text(patch)
    return namespace['rate_score'], hashlib.sha256(source.encode()).hexdigest()


def outcome(function, values):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        try:
            value = ('result', function(*values))
        except Exception as exc:
            value = ('error', type(exc).__name__, str(exc))
    return value, [(type(w.message).__name__, str(w.message)) for w in captured]


def edge_checks(candidate):
    cases = []
    grid = (-0.0, 0.0, np.nextafter(0., 1.), 0.1, 1., 1e-300, 1e300,
            np.finfo(float).max, -1., np.inf, -np.inf, np.nan)
    cases.extend(itertools.product(grid, repeat=4))
    # Retain the original conversion failure boundary and finite scalar types.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        overflowing_longdouble = np.longdouble('1e4000')
    unusual = (None, 'bad', '0.5', 'nan', 'inf', 10 ** 1000,
               Decimal('1e9999'), Decimal('NaN'), Decimal('0.5'),
               Fraction(1, 3), np.float32(0.5), overflowing_longdouble,
               np.int64(1), True, 1 + 0j, [1.], np.array([1., 2.]))
    for index in range(4):
        for value in unusual:
            values = [2., 1., 0., 1.]
            values[index] = value
            cases.append(values)
    # Very uneven exposures and posterior mass; no count/exposure division.
    rng = np.random.default_rng(303)
    for _ in range(5000):
        values = (10. ** rng.uniform(-323., 308., size=4)).tolist()
        if rng.random() < 0.15:
            values[int(rng.integers(0, 4))] = 0.
        cases.append(values)
    # Exact and adjacent enrichment boundaries, including subnormal counts.
    for scale in (np.nextafter(0., 1.), 1e-300, 0.1, 1., 1e300, np.finfo(float).max):
        with np.errstate(over='ignore'):
            adjacent = (scale, np.nextafter(scale, 0.), np.nextafter(scale, np.inf))
        for other in adjacent:
            cases.append((scale, 1., other, 1.))
    counts = {'cases': len(cases), 'result': 0, 'error': 0, 'warnings': 0}
    for values in cases:
        before, after = outcome(stats.rate_score, values), outcome(candidate, values)
        np.testing.assert_equal(before, after)
        counts[before[0][0]] += 1
        counts['warnings'] += len(before[1])
    return dict(counts, exact_scores_pvalues_errors_and_warnings=True,
                input_types='float/int/NumPy scalar/Decimal/Fraction/string/invalid conversion',
                seed=303)


def compare_timed(before, after, warmups, repeats):
    runs = {'before': [], 'after': []}
    output = None
    for repeat in range(warmups + repeats):
        outputs = {}
        for label in (('before', 'after') if repeat % 2 == 0 else ('after', 'before')):
            outputs[label], timing = measure(before if label == 'before' else after)
            if repeat >= warmups:
                runs[label].append(timing)
        np.testing.assert_equal(outputs['before'], outputs['after'])
        output = outputs['before']
    summaries = {label: summarize(samples) for label, samples in runs.items()}
    changes = {metric: [100 * (1 - b[metric] / a[metric])
                        for a, b in zip(runs['before'], runs['after'])]
               for metric in ('wall_s', 'cpu_s')}
    return dict(runs=runs, summary=summaries, paired_reduction_percent=changes), output


def micro_case(candidate, warmups, repeats):
    original = stats.rate_score
    inputs = [(2., 1., 0., 1.), (0.7, 0.1, 0.2, 10.), (1., 2., 2., 1.),
              (2., 1e-300, 0., 1e300), (np.nan, 1., 0., 1.), (1., 0., 0., 1.)]

    def run(function):
        checksum = 0.
        for _ in range(5000):
            for values in inputs:
                score, pvalue = function(*values)
                if math.isfinite(score):
                    checksum += score + pvalue
        return checksum

    result, _ = compare_timed(lambda: run(original), lambda: run(candidate), warmups, repeats)
    return dict(result, scope='30,000 calls to rate_score; no CLI/ASR/permutation overhead',
                input_cases=len(inputs), calls_per_sample=30000)


class Captured(Exception):
    pass


def phase_case(dataset, candidate, workdir, warmups, repeats, permutations):
    original_score = stats.rate_score
    original_chunk = scan._run_scan_permutation_chunk
    result = {}

    def capture(*args, **kwargs):
        values = inspect.signature(original_chunk).bind(*args, **kwargs).arguments
        assert values['calibration'] == 'candidate_fixed'

        def run(function):
            stats.rate_score = function
            try:
                return original_chunk(*args, **kwargs)
            finally:
                stats.rate_score = original_score

        timing, output = compare_timed(lambda: run(original_score), lambda: run(candidate),
                                       warmups, repeats)
        result.update(timing, candidates=len(values['observed_df']),
                      permutations=len(values['permutation_indices']),
                      success_count=output['success_count'], failure_count=output['failure_count'],
                      exact_scores_candidate_keys_maxima_diagnostics=True,
                      scope='candidate_fixed permutation chunk; fresh worker caches each sample; not CLI')
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
    assert result
    return dict(result, input_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--permutations', type=int, default=50)
    args = parser.parse_args()
    candidate, source_hash = prototype()
    result = dict(proposal=3, baseline_commit='624bb8a6dae536d0fa10891f666684ef1ad3dab0',
                  environment=environment(), warmups=args.warmups, repeats=args.repeats,
                  rate_score_source_sha256=source_hash,
                  command='python reports/performance_round3_20261023/proposal03.py',
                  memory='No retained storage or algorithmic memory change; peak RSS not measured.',
                  cases={})
    print('Proposal03 waiting for shared benchmark lock.', flush=True)
    with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-p03-') as temporary:
        result['edge_checks'] = edge_checks(candidate)
        print('Proposal03 exact validation comparisons passed.', flush=True)
        result['cases']['micro'] = micro_case(candidate, args.warmups, args.repeats)
        for dataset in ('PGK', 'PEPC'):
            result['cases'][dataset] = phase_case(dataset, candidate, Path(temporary),
                                                   args.warmups, args.repeats, args.permutations)
            (HERE / 'proposal03.json').write_text(json.dumps(result, indent=2) + '\n')
            print('Proposal03', dataset, result['cases'][dataset]['summary'], flush=True)
    (HERE / 'proposal03.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
