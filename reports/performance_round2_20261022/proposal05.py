"""Evaluate bounded-column omega tail transforms, without product edits.

Run: python reports/performance_round2_20261022/proposal05.py
"""

import argparse
import contextlib
from fractions import Fraction
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
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
from csubst import cli, omega, omega_statistics as statistics  # noqa: E402

ORIGINAL = statistics._calc_omega_empirical_upper_tail_counts
MAX_FLOAT_MATRIX_BYTES = 4 * 1024**2


def make_prototype():
    source = inspect.getsource(ORIGINAL)
    cut = source.index('    perm_omega = _calc_permutation_omega_matrix(')
    replacement = '''    shape = perm_count_N.shape
    width = max(1, MAX_FLOAT_MATRIX_BYTES // max(1, 8 * shape[0]))
    if shape[1] <= width or not shape[0]:
        return ORIGINAL(obs_omega, exp_N, exp_S, perm_count_N, perm_count_S,
                        float_tol, calibrate_dsc_transformation, alphas)
    prepared_alphas = None
    if alphas is not None:
        try:
            if len(alphas) != 4:
                raise ValueError
            prepared_alphas = [np.broadcast_to(np.asarray(value), shape) for value in alphas]
        except (TypeError, ValueError):
            # General array-like errors retain the original validation wording/order.
            return ORIGINAL(obs_omega, exp_N, exp_S, perm_count_N, perm_count_S,
                            float_tol, calibrate_dsc_transformation, alphas)
    extreme = np.zeros(shape[0], dtype=np.int64)
    valid = np.zeros(shape[0], dtype=np.int64)
    for start in range(0, shape[1], width):
        stop = min(start + width, shape[1])
        local_alphas = (None if prepared_alphas is None else
                        tuple(value[:, start:stop] for value in prepared_alphas))
        values = _calc_permutation_omega_matrix(
            exp_N, exp_S, perm_count_N[:, start:stop], perm_count_S[:, start:stop],
            float_tol, calibrate_dsc_transformation, local_alphas)
        more_extreme, more_valid = _calc_omega_empirical_upper_tail_counts_from_perm(obs_omega, values)
        extreme += more_extreme
        valid += more_valid
    return extreme, valid
'''
    namespace = dict(statistics.__dict__, ORIGINAL=ORIGINAL,
                     MAX_FLOAT_MATRIX_BYTES=MAX_FLOAT_MATRIX_BYTES)
    exec(compile(source[:cut] + replacement, '<proposal05-column-chunks>', 'exec'), namespace)
    return namespace[ORIGINAL.__name__]


CANDIDATE = make_prototype()


def digest(result):
    return hashlib.sha256(b''.join(value.tobytes() for value in result)).hexdigest()


def sample(rows, columns, seed=19):
    rng = np.random.default_rng(seed)
    n = rng.poisson(2.5, (rows, columns)).astype(float)
    s = rng.poisson(1.5, (rows, columns)).astype(float)
    e_n = rng.uniform(.1, 5., rows)
    e_s = rng.uniform(.1, 3., rows)
    obs = rng.uniform(0, 8, rows)
    return dict(obs_omega=obs, exp_N=e_n, exp_S=e_s, perm_count_N=n,
                perm_count_S=s, float_tol=1e-12)


def outcome(function, options):
    try:
        return 'ok', function(**options)
    except Exception as exc:
        return 'error', type(exc).__name__, str(exc)


def edge_checks():
    checks = []
    # Force several chunks even for small cases; only chunk width changes here.
    old_limit = CANDIDATE.__globals__['MAX_FLOAT_MATRIX_BYTES']
    CANDIDATE.__globals__['MAX_FLOAT_MATRIX_BYTES'] = 128
    try:
        for rows, columns in ((0, 4), (5, 0), (1, 11), (9, 13)):
            for transform in (None, 'quantile', 'gamma'):
                options = sample(rows, columns)
                for alpha in (None, (0.,) * 4, (.1,) * 4,
                              tuple(np.full((rows, 1), .2) for _ in range(4)),
                              tuple(np.linspace(0, 1, columns) for _ in range(4)),
                              tuple(np.zeros((rows, columns)) for _ in range(4))):
                    case = dict(options, calibrate_dsc_transformation=transform, alphas=alpha)
                    with np.errstate(all='ignore'):
                        before, after = outcome(ORIGINAL, case), outcome(CANDIDATE, case)
                    np.testing.assert_equal(before, after)
                    checks.append(dict(rows=rows, columns=columns, transformation=transform,
                                       alpha_shape=None if alpha is None else np.asarray(alpha[0]).shape,
                                       exact_outcome=True, outcome=before[0]))
        for mode in ('nonfinite', 'short_exp_N', 'short_exp_S', 'mismatch_rows', 'mismatch_matrices',
                     'wrong_ndim', 'invalid_transformation', 'wrong_alpha_shape', 'short_alphas'):
            case = sample(9, 13)
            if mode == 'nonfinite':
                case['obs_omega'][:5] = [0., np.nan, np.inf, -np.inf, 1e-300]
                case['perm_count_N'][:5, :4] = [0., np.nan, np.inf, 1e-300]
                case['perm_count_S'][:5, :4] = [0., np.nan, np.inf, 1e-300]
                case['exp_N'][:3] = [0., np.inf, np.nan]
            elif mode.startswith('short_exp'):
                key = mode.removeprefix('short_')
                case[key] = case[key][:-1]
            elif mode == 'mismatch_rows':
                case['obs_omega'] = case['obs_omega'][:-1]
            elif mode == 'mismatch_matrices':
                case['perm_count_S'] = case['perm_count_S'][:-1]
            elif mode == 'wrong_ndim':
                case['perm_count_N'] = np.ones(9)
                case['perm_count_S'] = np.ones(9)
            elif mode == 'invalid_transformation':
                case['calibrate_dsc_transformation'] = 'unsupported'
            elif mode == 'wrong_alpha_shape':
                case['alphas'] = (np.ones(7),) * 4
            elif mode == 'short_alphas':
                case['alphas'] = (.1,)
            with np.errstate(all='ignore'):
                before, after = outcome(ORIGINAL, case), outcome(CANDIDATE, case)
            np.testing.assert_equal(before, after)
            checks.append(dict(case=mode, exact_outcome=True, outcome=before[0]))
        # Nonzero mixed alpha and strided inputs exercise full-shape broadcast views.
        case = sample(9, 26)
        for key in ('perm_count_N', 'perm_count_S'):
            case[key] = case[key][:, ::2]
            case[key].flags.writeable = False
        case['alphas'] = (np.arange(9)[:, None] / 20, 0., np.arange(13) / 10, .1)
        np.testing.assert_equal(ORIGINAL(**case), CANDIDATE(**case))
        checks.append(dict(case='readonly_strided_mixed_alphas', exact_outcome=True))
        # Independent rational oracle: no copied NumPy rate/tie implementation.
        numerator = np.tile([0., 1., 2., 3., 4., 5.], (4, 5))
        denominator = np.tile([0., 2., 0., 3., 2., 1.], (4, 5))
        observations = [Fraction(1, 2), Fraction(1), Fraction(2), np.inf]
        case = dict(obs_omega=np.array([float(value) for value in observations]),
                    exp_N=np.ones(4), exp_S=np.ones(4), perm_count_N=numerator,
                    perm_count_S=denominator, float_tol=1e-12)
        oracle = []
        for observed, n_values, s_values in zip(observations, numerator, denominator):
            ranks = 0
            for n_value, s_value in zip(n_values, s_values):
                ratio = (Fraction(0) if n_value == 0 else
                         np.inf if s_value == 0 else Fraction(int(n_value), int(s_value)))
                ranks += int(ratio >= observed)
            oracle.append(ranks)
        expected = np.asarray(oracle, dtype=np.int64), np.full(4, 30, dtype=np.int64)
        np.testing.assert_equal(expected, ORIGINAL(**case))
        np.testing.assert_equal(expected, CANDIDATE(**case))
        checks.append(dict(case='independent_fraction_tail_oracle', exact_outcome=True,
                           expected_extreme=oracle, expected_valid=30))
        # Prove the optimized branch executed and every calibrated chunk kept all rows.
        old_matrix = CANDIDATE.__globals__['_calc_permutation_omega_matrix']
        calls = []

        def matrix(*args, **kwargs):
            calls.append(args[2].shape)
            return old_matrix(*args, **kwargs)

        CANDIDATE.__globals__['_calc_permutation_omega_matrix'] = matrix
        try:
            case = sample(9, 13)
            case['calibrate_dsc_transformation'] = 'quantile'
            CANDIDATE(**case)
        finally:
            CANDIDATE.__globals__['_calc_permutation_omega_matrix'] = old_matrix
        width = 128 // (9 * 8)
        assert len(calls) == 13 and all(rows == 9 and columns == width for rows, columns in calls)
        checks.append(dict(case='runtime_chunk_execution_full_row_population', exact_outcome=True,
                           chunk_shapes=calls))
    finally:
        CANDIDATE.__globals__['MAX_FLOAT_MATRIX_BYTES'] = old_limit
    return checks


def bench(options, repeats=3):
    expected = ORIGINAL(**options)
    np.testing.assert_equal(expected, CANDIDATE(**options))
    runs = {'before': [], 'chunked': []}
    for repeat in range(repeats + 1):
        for label in (('before', 'chunked') if repeat % 2 == 0 else ('chunked', 'before')):
            gc.collect()
            function = ORIGINAL if label == 'before' else CANDIDATE
            result, timing = measure(function, **options)
            np.testing.assert_array_equal(result[0], expected[0])
            np.testing.assert_array_equal(result[1], expected[1])
            if repeat:
                runs[label].append(timing)
    shape = options['perm_count_N'].shape
    width = max(1, MAX_FLOAT_MATRIX_BYTES // max(1, shape[0] * 8))
    return dict(shape=shape, chunk_columns=min(width, shape[1]), exact_counts=True,
                digest=digest(expected), runs=runs,
                summary={label: summarize(times) for label, times in runs.items()},
                input_count_matrix_bytes=sum(options[k].nbytes for k in ('perm_count_N', 'perm_count_S')),
                maximum_single_float_workspace_bytes=shape[0] * min(width, shape[1]) * 8)


class Captured(BaseException):
    pass


def real_case(dataset, workdir):
    workdir.mkdir()
    data = workdir / 'data'
    data.mkdir()
    names = [dataset + '.alignment.fa', dataset + '.tree.nwk']
    names += [dataset + '.alignment.fa.' + suffix for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log')]
    for name in names:
        shutil.copyfile(ROOT / 'csubst/dataset' / name, data / name)
    argv = ['search', '--alignment_file', str(data / (dataset + '.alignment.fa')),
            '--rooted_tree_file', str(data / (dataset + '.tree.nwk')), '--outdir', str(workdir / 'out'),
            '--threads', '1', '--blas_threads', '1', '--max_arity', '2', '--random_seed', '8',
            '--expectation_method', 'urn', '--output_stat', 'any2spe', '--calc_omega_pvalue', 'yes',
            '--omega_pvalue_null_model', 'poisson', '--omega_pvalue_niter_schedule', '1000',
            '--calibrate_longtail', 'no']
    for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        argv += ['--iqtree_' + suffix, str(data / (dataset + '.alignment.fa.' + suffix))]
    result = {}
    original = omega._calc_omega_empirical_upper_tail_counts

    def intercept(*args, **kwargs):
        bound = inspect.signature(ORIGINAL).bind(*args, **kwargs)
        bound.apply_defaults()
        result.update(bench(bound.arguments))
        raise Captured

    omega._calc_omega_empirical_upper_tail_counts = intercept
    try:
        with (workdir / 'run.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                cli.main(argv)
            except Captured:
                pass
    finally:
        omega._calc_omega_empirical_upper_tail_counts = original
    assert result, 'The real urn CLI did not call the targeted tail transform.'
    result['input_sha256'] = {name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in names}
    result['scope'] = 'first any2spe marginal Poisson tail transform from real joint urn CLI; full 1000-draw stage'
    result['command_options'] = {'action': 'search', 'expectation_method': 'urn', 'output_stat': 'any2spe',
                                 'calc_omega_pvalue': 'yes', 'omega_pvalue_null_model': 'poisson',
                                 'omega_pvalue_niter_schedule': '1000', 'calibrate_longtail': 'no',
                                 'max_arity': 2, 'threads': 1, 'blas_threads': 1, 'random_seed': 8}
    return result


def memory_worker(label):
    options = sample(6000, 1000)
    function = ORIGINAL if label == 'before' else CANDIDATE
    result, timing = measure(function, **options)
    # On macOS ru_maxrss is bytes, on Linux KiB. Import/startup/input allocation included.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    timing['peak_rss_bytes'] = int(rss if sys.platform == 'darwin' else rss * 1024)
    return dict(timing=timing, digest=digest(result))


def memory_bench():
    runs = {'before': [], 'chunked': []}
    expected_digest = None
    for repeat in range(4):
        for label in (('before', 'chunked') if repeat % 2 == 0 else ('chunked', 'before')):
            command = [sys.executable, str(Path(__file__).resolve()), '--memory-worker', label]
            completed = subprocess.run(command, capture_output=True, text=True, check=True)
            result = json.loads(completed.stdout)
            if expected_digest is None:
                expected_digest = result['digest']
            assert result['digest'] == expected_digest
            if repeat:
                runs[label].append(result['timing'])
    summary = {label: summarize(times) for label, times in runs.items()}
    for label, times in runs.items():
        values = [entry['peak_rss_bytes'] for entry in times]
        summary[label]['peak_rss_bytes'] = dict(median=float(np.median(values)), min=min(values), max=max(values))
    return dict(shape=[6000, 1000], input_count_matrix_bytes=96_000_000,
                scope='synthetic existing count matrices; isolated child process peak RSS includes imports and inputs',
                timed_scope='transform and inclusive-tail counts only; excludes input generation and process startup',
                exact_digest=True, digest=expected_digest, runs=runs, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-worker', choices=('before', 'chunked'))
    parser.add_argument('--checks-only', action='store_true')
    args = parser.parse_args()
    if args.memory_worker:
        print(json.dumps(memory_worker(args.memory_worker)))
        return
    if args.checks_only:
        destination = HERE / 'proposal05.json'
        result = json.loads(destination.read_text())
        with benchmark_lock():
            result['edge_checks'] = edge_checks()
        destination.write_text(json.dumps(result, indent=2) + '\n')
        print('Supplemented edge checks:', len(result['edge_checks']))
        return
    result = dict(proposal=5, environment=environment(),
                  source_function_sha256=hashlib.sha256(inspect.getsource(ORIGINAL).encode()).hexdigest(),
                  command='python reports/performance_round2_20261022/proposal05.py',
                  workspace_budget=MAX_FLOAT_MATRIX_BYTES,
                  workspace_definition='one float transform matrix, not total process or all intermediates',
                  warmups=1, measured_repeats=3)
    with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-round2-proposal05-') as temporary:
        result['edge_checks'] = edge_checks()
        result['cases'] = {dataset: real_case(dataset, Path(temporary) / dataset) for dataset in ('PGK', 'PEPC')}
        result['synthetic_memory'] = memory_bench()
    (HERE / 'proposal05.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value['summary'] for key, value in result['cases'].items()}, indent=2))
    print(json.dumps(result['synthetic_memory']['summary'], indent=2))


if __name__ == '__main__':
    main()
