"""Compare exact Fisher log-DP with a site-vectorized exclusion reduction."""

import inspect
import itertools
import json
import os
from pathlib import Path
import sys
from fractions import Fraction

for _key in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
             'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from csubst import omega  # noqa: E402
from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402


def make_prototype():
    source = inspect.getsource(omega._calc_fisher_inclusion_probabilities)
    start = source.index('    inclusion = np.zeros(num_positive, dtype=np.float64)')
    stop = source.index('    out[positive_mask] = np.clip(inclusion', start)
    replacement = '''    inclusion = np.full(num_positive, -np.inf, dtype=np.float64)
    workspace = np.empty(num_positive, dtype=np.float64)
    for left_pick in range(draw_size):
        right_pick = draw_size - 1 - left_pick
        np.add(prefix[:-1, left_pick], suffix[1:, right_pick], out=workspace)
        np.logaddexp(inclusion, workspace, out=inclusion)
    np.add(log_weights, inclusion, out=inclusion)
    np.subtract(inclusion, log_denominator, out=inclusion)
    np.exp(inclusion, out=inclusion)
'''
    source = source[:start] + replacement + source[stop:]
    namespace = {'np': np}
    exec(compile(source, '<fisher-vectorized-prototype>', 'exec'), namespace)
    return namespace['_calc_fisher_inclusion_probabilities']


def enumerate_reference(weights, draw):
    """Product-weighted subsets, independently enumerated using rational arithmetic."""
    positive = [index for index, weight in enumerate(weights) if weight > 0]
    draw = min(draw, len(positive))
    if draw == 0:
        return np.zeros(len(weights))
    denominator = Fraction(0)
    inclusion = [Fraction(0) for _ in weights]
    for selected in itertools.combinations(positive, draw):
        mass = Fraction(1)
        for index in selected:
            mass *= Fraction(float(weights[index]))
        denominator += mass
        for index in selected:
            inclusion[index] += mass
    return np.array([float(value / denominator) for value in inclusion])


def assert_bitwise_equal(before, after):
    assert before.dtype == after.dtype
    assert before.shape == after.shape
    np.testing.assert_array_equal(after, before)
    assert before.tobytes() == after.tobytes()


def check_small(prototype):
    rng = np.random.default_rng(20261022)
    examples = [[], [0., 0.], [1.], [0., 1., 0.], [1., 1., 1., 1.],
                [.8, .1, .05, .05, 0.], [1e-100, 1e-40, 1., 1e40, 1e100]]
    examples += [rng.integers(0, 31, size=n).astype(float).tolist()
                 for n in range(2, 10) for _ in range(3)]
    comparisons = 0
    maximum_absolute = maximum_reference_error = 0.
    dtype_checks = 0
    for values in examples:
        weights = np.asarray(values)
        for draw in range(len(weights) + 2):
            expected = enumerate_reference(weights, draw)
            for dtype in (np.float32, np.float64):
                before = omega._calc_fisher_inclusion_probabilities(weights, draw, dtype)
                after = prototype(weights, draw, dtype)
                assert before.dtype == after.dtype == dtype
                assert_bitwise_equal(before, after)
                if len(weights):
                    maximum_absolute = max(maximum_absolute, float(np.max(np.abs(before - after))))
                    maximum_reference_error = max(maximum_reference_error,
                                                  float(np.max(np.abs(after.astype(float) - expected))))
                np.testing.assert_allclose(after, expected, rtol=3e-6 if dtype == np.float32 else 3e-13,
                                           atol=1e-7 if dtype == np.float32 else 1e-14)
                comparisons += 1
                dtype_checks += 1
    error_checks = []
    for values, draw in [([1., -1.], 1), ([1., 2.], -1)]:
        messages = []
        for function in (omega._calc_fisher_inclusion_probabilities, prototype):
            try:
                function(values, draw)
            except ValueError as exc:
                messages.append(str(exc))
        assert len(messages) == 2 and messages[0] == messages[1]
        error_checks.append(messages[0])
    readonly = np.array([1., 0., 2., 4., 0.])
    readonly.flags.writeable = False
    readonly_bytes = readonly.tobytes()
    assert_bitwise_equal(omega._calc_fisher_inclusion_probabilities(readonly, 2),
                         prototype(readonly, 2))
    assert readonly.tobytes() == readonly_bytes and not readonly.flags.writeable
    return {'old_new_exact_comparisons': comparisons, 'dtype_checks': dtype_checks,
            'old_new_dtype_shape_bytes_comparisons': comparisons,
            'maximum_old_new_absolute_difference': maximum_absolute,
            'maximum_oracle_absolute_difference_including_float32': maximum_reference_error,
            'matched_error_messages': error_checks, 'readonly_input': 'dtype/shape/bytes passed',
            'readonly_input_bytes_and_flag_unchanged': True,
            'oracle': 'exhaustive product-weighted subsets using fractions.Fraction'}


def cases():
    rng = np.random.default_rng(20261022)
    for size in (417, 971):
        for distribution in ('uniform', 'skew_zero'):
            weights = np.ones(size)
            if distribution == 'skew_zero':
                weights = np.exp(rng.uniform(-10., 5., size))
                weights[::7] = 0
            for draw in (1, 10, 100):
                yield f'{size}_{distribution}_draw{draw}', weights, draw
    weights = np.exp(rng.uniform(-12., 3., 4096))
    weights[::11] = 0
    yield '4096_skew_zero_draw128', weights, 128


def benchmark_case(prototype, weights, draw):
    baseline = omega._calc_fisher_inclusion_probabilities
    outputs = [baseline(weights, draw), prototype(weights, draw)]
    assert_bitwise_equal(outputs[0], outputs[1])
    samples = {'before': [], 'after': []}
    for repeat in range(3):
        order = ('before', 'after') if repeat % 2 == 0 else ('after', 'before')
        for label in order:
            function = baseline if label == 'before' else prototype
            result, timing = measure(function, weights, draw)
            assert_bitwise_equal(outputs[0], result)
            samples[label].append(timing)
    summaries = {label: summarize(values) for label, values in samples.items()}
    positive = int(np.count_nonzero(weights > 0))
    return {'sites': len(weights), 'positive_sites': positive, 'draw_size': draw,
            'samples': samples, 'summary': summaries,
            'wall_reduction_percent': 100 * (1 - summaries['after']['wall_s']['median']
                                             / summaries['before']['wall_s']['median']),
            'speedup': summaries['before']['wall_s']['median']
                       / summaries['after']['wall_s']['median'],
            'exact_old_new_match': True,
            'sum_inclusion_minus_draw': float(outputs[0].sum() - draw),
            'unchanged_prefix_suffix_bytes': 2 * (positive + 1) * (draw + 1) * 8,
            'additional_workspace_bytes': positive * 8}


def main():
    prototype = make_prototype()
    result = {'proposal': 9, 'environment': environment(),
              'scope': 'Fisher inclusion function only; synthetic site weights, not an end-to-end urn run',
              'thread_control': 'BLAS environment variables set to 1 before importing NumPy/SciPy',
              'warmups': 1, 'measured_repeats': 3,
              'memory_scope': 'explicit ndarray workspace bytes, not measured peak RSS',
              'cases': {}}
    with benchmark_lock():
        result['correctness'] = check_small(prototype)
        for name, weights, draw in cases():
            result['cases'][name] = benchmark_case(prototype, weights, draw)
            print(name, result['cases'][name]['speedup'], flush=True)
    (HERE / 'proposal09.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
