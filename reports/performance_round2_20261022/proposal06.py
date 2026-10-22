"""Evaluate prepared common maxT tails; production modules remain unchanged.

Run: python reports/performance_round2_20261022/proposal06.py
"""

import contextlib
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

for variable in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import cli, scan_statistics, substitution_scan as scan  # noqa: E402


class PreparedTail:
    """One private, independent snapshot of a validated common reference."""

    def __init__(self, reference, expected_count=None, exact=False):
        values = np.asarray(reference, dtype=float)
        if values.ndim != 1 or not values.size or np.isnan(values).any() or np.isposinf(values).any():
            raise ValueError('A nonempty null reference without undefined replicates is required.')
        if expected_count is not None and values.size != expected_count:
            raise ValueError('Missing null replicates; success-conditioned calibration is not supported.')
        self.values = np.sort(values)
        self.values.flags.writeable = False
        self.offset = 0 if exact else 1
        self.corrections = 0

    @staticmethod
    def extreme(values, observed):
        return (values >= observed) | np.isclose(values, observed, rtol=1e-12, atol=1e-12)

    def correct_boundary(self, observed):
        # For finite positive observed, the inclusive tail predicate is
        # monotone in the sorted reference, including its floating subtraction.
        low, high = 0, len(self.values)
        while low < high:
            middle = (low + high) // 2
            if self.extreme(self.values[middle], observed):
                high = middle
            else:
                low = middle + 1
        return low

    def query(self, observed):
        # This private batch boundary receives float64 scan score arrays.
        # The public scalar entry point is deliberately left untouched.
        observed = np.asarray(observed, dtype=float)
        if observed.ndim != 1:
            raise ValueError('Prepared score queries require one dimension.')
        out = np.full(observed.shape, np.nan)
        out[observed <= 0] = 1.
        keep = np.isfinite(observed) & (observed > 0)
        scores = observed[keep]
        if not len(scores):
            return out
        thresholds = scores - (1e-12 + 1e-12 * np.abs(scores))
        indices = np.searchsorted(self.values, thresholds, side='left')
        # Testing both adjacent values catches a rounded threshold that lands
        # on the wrong side of a representable reference value. No absolute or
        # relative tolerance is changed; troublesome queries use the original
        # predicate itself in the binary search.
        wrong = np.zeros(len(scores), dtype=bool)
        valid = indices < len(self.values)
        wrong[valid] |= ~self.extreme(self.values[indices[valid]], scores[valid])
        valid = indices > 0
        wrong[valid] |= self.extreme(self.values[indices[valid] - 1], scores[valid])
        for index in np.flatnonzero(wrong):
            indices[index] = self.correct_boundary(scores[index])
        self.corrections += int(wrong.sum())
        out[keep] = (self.offset + len(self.values) - indices) / (self.offset + len(self.values))
        return out


def baseline(observed, reference, expected_count, exact=False):
    return np.array([scan_statistics.empirical_pvalue(value, reference, expected_count, exact)
                     for value in observed])


def candidate(observed, reference, expected_count, exact=False):
    return PreparedTail(reference, expected_count, exact).query(observed)


def pipeline_candidate(observed, reference, expected_count, exact=False):
    # Preserve substitution_scan's earlier denominator error before the
    # scalar-statistics validation if this batch is inserted there later.
    if len(reference) != expected_count:
        raise ValueError('Each scan configuration must contribute exactly one null statistic.')
    return candidate(observed, reference, expected_count, exact)


def assert_same(before, after):
    np.testing.assert_array_equal(before, after, strict=True)
    assert before.tobytes() == after.tobytes()


def edge_checks():
    rng = np.random.default_rng(12606)
    observed = np.r_[np.nextafter(0., 1.), np.finfo(float).tiny,
                     np.geomspace(1e-300, 1e300, 81),
                     np.nextafter(1e-12, 0.), 1e-12, np.nextafter(1e-12, np.inf),
                     np.finfo(float).max, 10. ** rng.uniform(-320, 308, 160)]
    checked = corrected = naive_wrong = 0
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        for score in observed:
            threshold = score - (1e-12 + 1e-12 * abs(score))
            adjacent = [threshold]
            lower = upper = threshold
            for _ in range(4):
                lower = np.nextafter(lower, -np.inf)
                upper = np.nextafter(upper, np.inf)
                adjacent.extend((lower, upper))
            reference = np.r_[np.repeat(adjacent, 7), -np.inf, -np.finfo(float).max,
                              np.finfo(float).max, 0., score]
            for exact in (False, True):
                prepared = PreparedTail(reference, len(reference), exact)
                before = baseline([score], reference, len(reference), exact)
                assert_same(before, prepared.query([score]))
                corrected += prepared.corrections
                checked += 1
            naive_count = np.count_nonzero(reference >= threshold)
            original_count = np.count_nonzero(PreparedTail.extreme(reference, score))
            naive_wrong += int(naive_count != original_count)
        reference = np.r_[rng.normal(2, 10, 2000), -np.inf, 0, 1, 1e-12]
        queries = np.r_[rng.normal(2, 10, 2000), -np.inf, np.inf, np.nan, -0., 0.]
        for exact in (False, True):
            prepared = PreparedTail(reference, len(reference), exact)
            assert_same(baseline(queries, reference, len(reference), exact), prepared.query(queries))
            checked += len(queries)
        # Validation must precede even an observed NaN or a no-enrichment query.
        invalid = [(1., [1., np.nan], 2), (0., [np.inf], 1),
                   (np.nan, [], 0), (np.nan, [1., 2.], 1),
                   (1., [[1., 2.]], 2), (1., ['bad'], 1)]
        for score, reference, expected_count in invalid:
            outcomes = []
            for function in (baseline, candidate):
                try:
                    function([score], reference, expected_count)
                except Exception as exc:
                    outcomes.append((type(exc).__name__, str(exc)))
                else:
                    raise AssertionError('Invalid reference accepted')
            assert outcomes[0] == outcomes[1]
            checked += 1
        for score, reference, expected_count in invalid:
            outcomes = []
            for function in (lambda values, ref, n: np.array([
                    scan._empirical_p_from_scores(v, ref, n) for v in values]), pipeline_candidate):
                try:
                    function([score], reference, expected_count)
                except Exception as exc:
                    outcomes.append((type(exc).__name__, str(exc)))
                else:
                    raise AssertionError('Invalid pipeline reference accepted')
            assert outcomes[0] == outcomes[1]
            checked += 1
        reference = np.array([-np.inf, 1., 2., 2.])
        prepared = PreparedTail(reference, 4)
        before = prepared.query([1., 2., 3.])
        reference[:] = 100
        assert_same(before, prepared.query([1., 2., 3.]))
        assert not prepared.values.flags.writeable
        checked += 1
    assert corrected > 0 and naive_wrong > 0
    return {'query_or_invalid_case_comparisons': checked,
            'boundary_corrected_queries': corrected,
            'adversarial_references_where_naive_threshold_count_was_wrong': naive_wrong,
            'strict_array_and_bytes_equal': True,
            'reference_snapshot_ownership_and_readonly': True,
            'orchestration_denominator_error_order_preserved': True,
            'public_scalar_function_unchanged': True}


def benchmark(observed, reference, expected_count, exact=False):
    expected, initial = measure(baseline, observed, reference, expected_count, exact)
    assert_same(expected, candidate(observed, reference, expected_count, exact))
    inner = min(1000, max(1, int(.03 / max(initial['wall_s'], 1e-9))))
    runs = {'before': [], 'prepared': []}

    def repeat(function):
        for _ in range(inner):
            output = function(observed, reference, expected_count, exact)
        return output

    for iteration in range(6):
        for label in (('before', 'prepared') if iteration % 2 == 0 else ('prepared', 'before')):
            gc.collect()
            result, timing = measure(repeat, baseline if label == 'before' else candidate)
            assert_same(expected, result)
            if iteration:
                runs[label].append({key: value / inner for key, value in timing.items()})
    return {'queries': len(observed), 'reference_draws': expected_count,
            'exact_enumeration': exact, 'inner_batches_per_sample': inner,
            'scope': 'common maxT final-tail batch; validation, float conversion and sort preparation included',
            'runs': runs, 'summary': {label: summarize(samples) for label, samples in runs.items()},
            'strict_array_and_bytes_equal': True,
            'sorted_reference_bytes': expected_count * 8,
            'returned_output_bytes': len(observed) * 8,
            'memory_scope': 'array sizes only, not process peak RSS; query scratch also scales with query count'}


def copy_inputs(dataset, workdir):
    data = workdir / 'inputs'
    data.mkdir()
    names = [dataset + '.alignment.fa', dataset + '.tree.nwk', dataset + '.foreground.txt']
    names += [dataset + '.alignment.fa.' + suffix for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log')]
    for name in names:
        shutil.copyfile(ROOT / 'csubst/dataset' / name, data / name)
    return data, {name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in names}


def real_case(dataset, workdir):
    data, hashes = copy_inputs(dataset, workdir)
    groups = {}
    original = scan._empirical_p_from_scores

    def capture(observed, values, denominator_count, exact=False):
        value = original(observed, values, denominator_count, exact)
        group = groups.setdefault(id(values), {'reference': values, 'queries': [], 'expected': [],
                                              'count': denominator_count, 'exact': exact})
        assert group['reference'] is values
        group['queries'].append(observed)
        group['expected'].append(value)
        return value

    argv = ['scan', '--alignment_file', str(data / (dataset + '.alignment.fa')),
            '--rooted_tree_file', str(data / (dataset + '.tree.nwk')),
            '--foreground', str(data / (dataset + '.foreground.txt')),
            '--outdir', str(workdir / 'output'), '--threads', '1', '--blas_threads', '1',
            '--scan_site_plot', 'no', '--scan_pvalue_calibration', 'full_scan',
            '--scan_n_permutations', '20', '--random_seed', '8', '--float_digit', '12']
    for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        argv += ['--iqtree_' + suffix, str(data / (dataset + '.alignment.fa.' + suffix))]
    scan._empirical_p_from_scores = capture
    try:
        with (workdir / 'capture.log').open('w') as log, contextlib.redirect_stdout(log):
            cli.main(argv)
    finally:
        scan._empirical_p_from_scores = original
    common = max(groups.values(), key=lambda group: len(group['queries']))
    observed = np.asarray(common['queries'], dtype=float)
    reference = list(common['reference'])
    assert len(observed) > 1
    assert_same(np.asarray(common['expected']), candidate(observed, reference, common['count'], common['exact']))
    result = benchmark(observed, reference, common['count'], common['exact'])
    result['input_sha256'] = hashes
    result['captured_from_real_full_scan'] = True
    result['candidate_rows_plus_global_query'] = len(observed)
    diagnostic = json.loads((workdir / 'output/csubst_scan_calibration.json').read_text())
    assert diagnostic['status'] == 'conditional_assignment'
    assert all(trial['status'] == 'success' for trial in diagnostic['trials'])
    result['capture_successful_trials'] = len(diagnostic['trials'])
    return result


def main():
    result = {'proposal': '06: prepared sorted common maxT reference with exact boundary checks',
              'environment': environment(), 'baseline_commit': '88528ce',
              'production_modified': False, 'warmups': 1, 'measured_repeats': 5,
              'source_function_sha256': hashlib.sha256(
                  inspect.getsource(scan_statistics.empirical_pvalue).encode()).hexdigest(),
              'scope': 'final maxT query stage only; not permutation evaluation or whole CLI',
              'cases': {}}
    with benchmark_lock():
        result['edge_checks'] = edge_checks()
        for dataset in ('PGK', 'PEPC'):
            with tempfile.TemporaryDirectory(prefix='csubst-round2-p06-') as directory:
                result['cases'][dataset] = real_case(dataset, Path(directory))
        rng = np.random.default_rng(80606)
        for count, queries in ((1000, 103), (1000, 10000), (10000, 2000)):
            reference = rng.exponential(2, count)
            reference[::19] = -np.inf
            observed = rng.exponential(3, queries)
            case = benchmark(observed, reference.tolist(), count)
            case['synthetic'] = True
            result['cases'][f'synthetic_{queries}_queries_{count}_draws'] = case
    (HERE / 'proposal06.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: case['summary'] for name, case in result['cases'].items()}, indent=2))


if __name__ == '__main__':
    main()
