"""Evaluate fold-local log-prior and response reuse in structure-weighting CV.

Run: python reports/performance_round2_20261022/proposal10.py
All inputs are synthetic; production files are never changed.
"""

import contextlib
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import epistasis  # noqa: E402

ORIGINAL_SELECT = epistasis._select
ORIGINAL_PREDICT = epistasis._predict_score
CACHE_LIMIT = 32 * 1024**2
STATS = []


class Prior:
    """Replace private base storage with its log only after predictor validation."""

    conversions = 0

    def __init__(self, values):
        self.values = values
        self.ready = False

    @property
    def nbytes(self):
        return self.values.nbytes

    def log(self, mask):
        if not self.ready:
            values = np.full_like(self.values, -np.inf)
            np.log(self.values, out=values, where=(self.values > 0) & mask)
            values.flags.writeable = False
            self.values, self.ready = values, True
            Prior.conversions += 1
        return self.values


class Heldout:
    """Only the current inner fold's scoring responses; never training data."""

    builds = 0

    def __init__(self, counts):
        self.total = counts.sum()
        self.positive = counts > 0
        self.values = counts[self.positive]
        Heldout.builds += 1

    @property
    def nbytes(self):
        return self.positive.nbytes + self.values.nbytes + 8

    def log_score(self, probabilities):
        values = probabilities[self.positive]
        if np.any(values <= 0):
            return -np.inf
        return float(np.sum(self.values * np.log(values)))


def make_prototype():
    predict_source = inspect.getsource(ORIGINAL_PREDICT)
    source = predict_source.replace('def _predict_score(base, score, beta, clip, mask):',
                                    'def _predict_prior(prior, score, beta, clip, mask):')
    old = ('    logbase = np.full_like(base, -np.inf)\n'
           '    np.log(base, out=logbase, where=(base > 0) & mask)\n')
    assert source.count(old) == 1
    source = source.replace(old, '    logbase = prior.log(mask)\n')
    assert 'prior.log(mask)' in source and 'np.log(base,' not in source
    namespace = dict(epistasis.__dict__, Prior=Prior, Heldout=Heldout,
                     STATS=STATS, CACHE_LIMIT=CACHE_LIMIT)
    exec(compile(source, '<proposal10-predict-prior>', 'exec'), namespace)
    select_source = inspect.getsource(ORIGINAL_SELECT)
    source = select_source.replace('    cached_bytes = 0\n',
                                   "    cached_bytes = 0\n    stats = dict(hits=0, misses=0, evictions=0, peak_array_bytes=0)\n")
    source = source.replace('        if key in cache:\n',
                            "        if key in cache:\n            stats['hits'] += 1\n")
    source = source.replace('        value = build()\n',
                            "        stats['misses'] += 1\n        value = build()\n")
    source = source.replace('32 * 1024**2', 'CACHE_LIMIT')
    source = source.replace('                cached_bytes -= cache.popitem(last=False)[1].nbytes\n',
                            "                cached_bytes -= cache.popitem(last=False)[1].nbytes\n                stats['evictions'] += 1\n")
    source = source.replace('            cached_bytes += value.nbytes\n',
                            "            cached_bytes += value.nbytes\n            stats['peak_array_bytes'] = max(stats['peak_array_bytes'], cached_bytes)\n")
    source = source.replace('            mask = masks[test]\n',
                            "            mask = cached(('mask', fold), lambda: masks[test])\n")
    old = "            base = cached(('base', fold, alpha), lambda: base_probabilities(counts, train, mask, alpha))\n"
    assert source.count(old) == 1
    source = source.replace(old,
                            "            base = cached(('prior', fold, alpha), lambda: Prior(base_probabilities(counts, train, mask, alpha)))\n")
    old = ('            p = _predict_score(base, score, beta, clip, mask)\n'
           '            total = counts[test].sum()\n'
           '            # All branches (including zero-event rows) remain in the folds.\n'
           '            scores.append(log_score(counts[test], p) / total if total > 0 else 0.)\n')
    assert source.count(old) == 1
    source = source.replace(old,
                            "            p = _predict_prior(base, score, beta, clip, mask)\n"
                            "            response = cached(('heldout', fold), lambda: Heldout(counts[test]))\n"
                            "            scores.append(response.log_score(p) / response.total if response.total > 0 else 0.)\n")
    source = source.replace("    return best, {'selection':", "    STATS.append(stats)\n    return best, {'selection':")
    assert "Prior(base_probabilities(" in source and "response.log_score(p)" in source
    assert '_predict_prior(base,' in source and "STATS.append(stats)" in source
    exec(compile(source, '<proposal10-select>', 'exec'), namespace)
    return namespace['_select'], namespace['_predict_prior'], {
        'select': hashlib.sha256(select_source.encode()).hexdigest(),
        'predict_score': hashlib.sha256(predict_source.encode()).hexdigest(),
    }


CANDIDATE, CANDIDATE_PREDICT, SOURCE_HASHES = make_prototype()


def arrays(branches, sites, seed=1210):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(.12, (branches, sites)).astype(float)
    context = rng.normal(size=(branches, 2))
    features = rng.normal(size=(sites, 2))
    masks = rng.random((branches, sites)) > .03
    groups = np.arange(branches) % 5
    partitions = np.zeros(branches, dtype=int)
    return [counts, context, features, masks, groups, partitions]


def config(**overrides):
    return dict({'epistasis_beta_auto': True, 'epistasis_joint_auto': True,
                 'epistasis_clip_auto': True}, **overrides)


def crossfit(function, values, settings):
    previous = epistasis._select
    epistasis._select = function
    try:
        return epistasis.crossfit(*values, settings)
    finally:
        epistasis._select = previous


def assert_exact(first, second):
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_exact(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert type(first) is type(second) and len(first) == len(second)
        for a, b in zip(first, second):
            assert_exact(a, b)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second, strict=True)
        assert first.tobytes() == second.tobytes()
    else:
        np.testing.assert_equal(first, second)


def outcome(function, values, settings):
    try:
        return 'ok', crossfit(function, values, settings)
    except Exception as exc:
        return 'error', type(exc).__name__, str(exc)


def edge_checks():
    settings = config(epistasis_joint_alpha_grid=[0., 1.], epistasis_joint_clip_grid=[1.5, 3.])
    checks = []
    for kind in ('normal', 'zero_context', 'zero_events', 'masked', 'buffer', 'depth_bins',
                 'readonly', 'undefined', 'nonfinite', 'overflow_beta_zero', 'fixed', 'insufficient'):
        values = arrays(30, 7)
        local = dict(settings)
        if kind == 'zero_context':
            values[1][:] = 0
        elif kind == 'zero_events':
            values[0][:] = 0
        elif kind == 'masked':
            values[3][:2] = False
        elif kind == 'buffer':
            values[4][:2] = -1
        elif kind == 'depth_bins':
            values[5] = np.arange(30) % 2
        elif kind == 'readonly':
            for value in values:
                value.flags.writeable = False
        elif kind == 'undefined':
            values = [np.eye(3), np.ones((3, 1)), np.arange(3.)[:, None],
                      np.ones((3, 3), bool), np.arange(3), np.zeros(3, int)]
            local = config(epistasis_joint_auto=False, epistasis_clip_auto=False,
                           epistasis_clip_value=3., asrv_dirichlet_alpha=0.)
        elif kind == 'nonfinite':
            values[1][0, 0] = np.nan
        elif kind == 'overflow_beta_zero':
            values[1][:] = 1e308
            values[2][:] = 1e308
            local = config(epistasis_beta_auto=False, epistasis_beta_value=0.,
                           epistasis_clip_auto=False, epistasis_clip_value=3.)
        elif kind == 'fixed':
            local = config(epistasis_beta_auto=False, epistasis_beta_value=.4,
                           epistasis_joint_auto=False, epistasis_clip_auto=False,
                           epistasis_clip_value=3.)
        elif kind == 'insufficient':
            values[4] %= 2
        with np.errstate(over='ignore', invalid='ignore'):
            before = outcome(ORIGINAL_SELECT, values, local)
            after = outcome(CANDIDATE, values, local)
        assert_exact(before, after)
        if kind in ('nonfinite', 'undefined', 'insufficient', 'overflow_beta_zero'):
            assert before[0] == 'error'
        if kind == 'zero_context':
            np.testing.assert_array_equal(after[1]['beta_by_branch'], 0.)
        checks.append(dict(case=kind, exact_outcome=True, outcome=before[0]))
    values = arrays(30, 7)
    first = crossfit(CANDIDATE, values, settings)
    changed = [a.copy() for a in values]
    heldout = values[4] == 0
    changed[0][heldout] = 999
    second = crossfit(CANDIDATE, changed, settings)
    for key in ('probabilities', 'null_probabilities', 'beta_by_branch', 'alpha_by_branch', 'clip_by_branch'):
        assert_exact(first[key][heldout], second[key][heldout])
    checks.append(dict(case='outer_heldout_response_independence', exact_outcome=True))
    values[4][:2] = -1
    first = crossfit(CANDIDATE, values, settings)
    values[0][:2] = 999
    second = crossfit(CANDIDATE, values, settings)
    for key in ('probabilities', 'null_probabilities', 'beta_by_branch', 'alpha_by_branch', 'clip_by_branch'):
        assert_exact(first[key], second[key])
    checks.append(dict(case='buffer_response_independence', exact_outcome=True))
    old_limit = CANDIDATE.__globals__['CACHE_LIMIT']
    CANDIDATE.__globals__['CACHE_LIMIT'] = 1024
    STATS.clear()
    try:
        values = arrays(30, 7)
        assert_exact(crossfit(ORIGINAL_SELECT, values, settings), crossfit(CANDIDATE, values, settings))
        assert STATS and max(x['peak_array_bytes'] for x in STATS) <= 1024
        assert sum(x['evictions'] for x in STATS) > 0
    finally:
        CANDIDATE.__globals__['CACHE_LIMIT'] = old_limit
    checks.append(dict(case='combined_cache_byte_eviction', exact_outcome=True))
    base = np.array([[.2, .3, .5]])
    score = np.array([[1., 0., -1.]])
    mask = np.ones_like(base, dtype=bool)
    predicted = CANDIDATE_PREDICT(Prior(base), score, .8, 3., mask)
    oracle = base * np.exp(.8 * score)
    oracle /= oracle.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(predicted, oracle, rtol=1e-14, atol=0)
    checks.append(dict(case='independent_direct_softmax', rtol=1e-14, exact_outcome=False))
    return checks


def benchmark(branches, sites):
    values, settings = arrays(branches, sites), config()
    before = crossfit(ORIGINAL_SELECT, values, settings)
    Prior.conversions = Heldout.builds = 0
    STATS.clear()
    after = crossfit(CANDIDATE, values, settings)
    assert_exact(before, after)
    stats = dict(prior_log_conversions=Prior.conversions, heldout_builds=Heldout.builds,
                 selection_calls=len(STATS), cache_hits=sum(x['hits'] for x in STATS),
                 cache_misses=sum(x['misses'] for x in STATS),
                 evictions=sum(x['evictions'] for x in STATS),
                 peak_retained_array_bytes=max(x['peak_array_bytes'] for x in STATS),
                 cache_limit_array_bytes=CACHE_LIMIT)
    assert stats['cache_hits'] > 0 and stats['prior_log_conversions'] > 0
    assert stats['peak_retained_array_bytes'] <= CACHE_LIMIT
    assert np.all(values[3].any(axis=1))
    # In this single-partition fixture every row is eligible and no entry is
    # evicted. The original cache retains one base per alpha and one score
    # for every inner test row, which partitions the outer training rows.
    training_rows = max(np.count_nonzero(values[4] != group) for group in range(5))
    alphas = len({alpha for _, alpha, _ in epistasis.candidate_grid(settings)})
    baseline_payload = training_rows * sites * 8 * (alphas + 1)
    stats['baseline_retained_array_bytes_derived'] = baseline_payload
    stats['additional_retained_array_bytes_derived'] = stats['peak_retained_array_bytes'] - baseline_payload
    runs = {'before': [], 'cached': []}
    for repeat in range(4):
        for label in (('before', 'cached') if repeat % 2 == 0 else ('cached', 'before')):
            function = ORIGINAL_SELECT if label == 'before' else CANDIDATE
            output, timing = measure(crossfit, function, values, settings)
            assert_exact(before, output)
            STATS.clear()
            if repeat:
                runs[label].append(timing)
    return dict(shape=[branches, sites], input='entirely synthetic Poisson counts and independent Gaussian context/features',
                groups=5, partitions=1, context_dimensions=2,
                input_array_sha256={name: hashlib.sha256(value.tobytes()).hexdigest()
                                    for name, value in zip(('counts', 'context', 'features', 'masks', 'groups', 'partitions'), values)},
                grid_candidates=len(before['beta_diag']['candidates_beta_alpha_clip']),
                exact_all_crossfit_arrays_and_diagnostics=True, cache=stats, runs=runs,
                summary={label: summarize(times) for label, times in runs.items()})


def existing_tests():
    import pytest

    previous = epistasis._select
    epistasis._select = CANDIDATE
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            code = pytest.main(['-q', 'tests/unit/test_omega_epistasis.py', '--disable-warnings'])
    finally:
        epistasis._select = previous
    assert code == 0, log.getvalue()
    # Persist only the final summary, not pytest's machine-specific paths.
    return {'command': 'pytest -q tests/unit/test_omega_epistasis.py --disable-warnings',
            'backend': 'private _select replaced with proposal10', 'exit_code': int(code),
            'summary': log.getvalue().strip().splitlines()[-1]}


def main():
    with benchmark_lock():
        checks = edge_checks()
        result = dict(proposal=10, environment=environment(), source_function_sha256=SOURCE_HASHES,
                      command='python reports/performance_round2_20261022/proposal10.py',
                      scope='full nested crossfit stage; fully synthetic inputs, not a CLI or biological calibration',
                      warmups=1, measured_repeats=3,
                      memory_scope='cache retained array payload bytes; Python metadata and process peak RSS not measured',
                      edge_checks=checks,
                      cases={'PGK_shape': benchmark(64, 417), 'PEPC_shape': benchmark(140, 971)},
                      existing_tests=existing_tests())
    (HERE / 'proposal10.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, case in result['cases'].items():
        print(name, case['summary'], case['cache'])
    print(result['existing_tests'])


if __name__ == '__main__':
    main()
