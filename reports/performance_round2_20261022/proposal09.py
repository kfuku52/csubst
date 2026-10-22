"""Evaluate reachability-aware uniformization, without changing product code.

Run: python reports/performance_round2_20261022/proposal09.py
"""

from decimal import Decimal, localcontext
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile

for variable in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np  # noqa: E402
from scipy.linalg import expm  # noqa: E402

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import endpoint  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ORIGINAL = endpoint.EndpointModel


class GraphModel(ORIGINAL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        k = self.pi.size
        distance = np.where(self._uniform_kernel > 0, 1, k + 1).astype(np.int64)
        np.fill_diagonal(distance, 0)
        for via in range(k):
            np.minimum(distance, distance[:, via, None] + distance[None, via, :], out=distance)
        self._reachable = distance <= k
        self._minimum_order = int(distance[self._reachable].max())
        self.statistics = None

    def _uniform_transition(self, t):
        tau = self._uniform_rate * t
        if tau == 0:
            return np.eye(self.pi.size)
        capacity = self._uniform_powers.shape[0]
        if tau > 32 or capacity < self.pi.size:
            return None
        coefficients = np.empty(capacity)
        coefficients[0] = np.exp(-tau)
        target = 1e-18
        for n in range(1, capacity):
            coefficients[n] = coefficients[n - 1] * tau / n
            if n < self._minimum_order or n + 1 <= tau:
                continue
            tail = coefficients[n] * tau / (n + 1 - tau)
            if tail > target:
                continue
            while self._uniform_count <= n:
                i = self._uniform_count
                self._uniform_powers[i] = self._uniform_powers[i - 1] @ self._uniform_kernel
                self._uniform_count += 1
            matrix = (coefficients[:n + 1] @ self._uniform_powers[:n + 1].reshape(n + 1, -1)).reshape(self.q.shape)
            # Graph reachability does not guarantee a representable positive
            # partial: the first path may underflow, while a later path does not.
            if (matrix[self._reachable] <= 0).any():
                if self.statistics is not None:
                    self.statistics['zero_guards'] += 1
                continue
            target = float(matrix[matrix > 0].min() * np.finfo(float).eps * 0.25)
            if tail <= target:
                if self.statistics is not None:
                    self.statistics['accepted_orders'].append(n)
                return matrix
        return None


def clone(model, cls=ORIGINAL):
    return cls(model.parents, model.lengths, model.q, model.pi, model.rates, model.weights)


def fields(record):
    for name in ('node', 'joint', 'predictive', 'category_node'):
        value = getattr(record, name)
        if value is not None:
            yield name, value
    for name in ('reduced', 'reduced_predictive'):
        for key, value in (getattr(record, name) or {}).items():
            yield (name, key), value


def compare_arrays(left, right):
    np.testing.assert_allclose(left, right, rtol=3e-14, atol=1e-300)
    positive = left != 0
    return {'values': int(left.size), 'changed': int(np.count_nonzero(left != right)),
            'max_absolute': float(np.max(np.abs(left - right), initial=0)),
            'max_relative': float(np.max(np.abs((left[positive] - right[positive]) / left[positive]), initial=0))}


def compare_records(before, after, tips, options):
    from itertools import zip_longest
    total = changed = threshold_crossings = records = 0
    maximum_absolute = maximum_relative = 0.
    for a, b in zip_longest(before.iter_blocks(tips, 64, **options), after.iter_blocks(tips, 64, **options)):
        assert a is not None and b is not None
        assert (a.start, a.stop, a.child, a.parent) == (b.start, b.stop, b.child, b.parent)
        aa, bb = dict(fields(a)), dict(fields(b))
        assert aa.keys() == bb.keys()
        for key, values in aa.items():
            metrics = compare_arrays(values, bb[key])
            total += metrics['values']
            changed += metrics['changed']
            maximum_absolute = max(maximum_absolute, metrics['max_absolute'])
            maximum_relative = max(maximum_relative, metrics['max_relative'])
            for threshold in (1e-12, .5):
                threshold_crossings += int(np.count_nonzero((values >= threshold) != (bb[key] >= threshold)))
        records += 1
    assert threshold_crossings == 0
    return dict(records=records, values=total, changed=changed, maximum_absolute=maximum_absolute,
                maximum_relative=maximum_relative, threshold_crossings=threshold_crossings,
                rtol=3e-14, atol=1e-300)


def all_transitions(model):
    total = 0.
    for child in model.order[1:]:
        for category in range(len(model.rates)):
            total += float(model.transition(child, category).sum())
    return total


def consume(model, tips, options):
    count = 0
    for record in model.iter_blocks(tips, 64, **options):
        count += record.node.size
    return count


def timed_models(source, tips, options):
    cold, warm = ({label: [] for label in ('before', 'graph')} for _ in range(2))
    models = {'before': clone(source), 'graph': clone(source, GraphModel)}
    for model in models.values():
        all_transitions(model)
    for repeat in range(4):
        for label in (('before', 'graph') if repeat % 2 == 0 else ('graph', 'before')):
            cls = ORIGINAL if label == 'before' else GraphModel
            gc.collect()
            def prepare():
                model = clone(source, cls)
                return all_transitions(model)
            _, sample = measure(prepare)
            if repeat:
                cold[label].append(sample)
            _, sample = measure(consume, models[label], tips, options)
            if repeat:
                warm[label].append(sample)
    return {'cold_model_and_all_transitions': {'runs': cold, 'summary': {k: summarize(v) for k, v in cold.items()}},
            'warm_endpoint_stage': {'runs': warm, 'summary': {k: summarize(v) for k, v in warm.items()}}}


def capture_module():
    path = ROOT / 'reports/performance_candidates_20261022/proposal02.py'
    spec = importlib.util.spec_from_file_location('round2_endpoint_capture09', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def real_case(dataset, directory):
    source, tips, options, hashes = capture_module().capture(dataset, directory)
    before, after = clone(source), clone(source, GraphModel)
    after.statistics = dict(accepted_orders=[], zero_guards=0)
    transition_metrics = []
    for child in source.order[1:]:
        for category in range(len(source.rates)):
            transition_metrics.append(compare_arrays(before.transition(child, category), after.transition(child, category)))
    parity = compare_records(before, after, tips, options)
    timings = timed_models(source, tips, options)
    orders = after.statistics['accepted_orders']
    assert orders and min(orders) < source.pi.size - 1, 'Optimized early return did not execute.'
    return dict(input_sha256=hashes, shape=[source.parents.size, len(next(iter(tips.values()))), source.pi.size],
                rates=source.rates.tolist(), graph_minimum_order=after._minimum_order,
                actual_accepted_min_order=min(orders), actual_accepted_max_order=max(orders),
                accepted_transition_count=len(orders), reachable_pair_count=int(after._reachable.sum()),
                extra_retained_array_bytes=after._reachable.nbytes,
                transient_distance_array_bytes=after._reachable.size * 8,
                transition_maximum_absolute=max(v['max_absolute'] for v in transition_metrics),
                transition_maximum_relative=max(v['max_relative'] for v in transition_metrics),
                record_parity=parity, **timings)


def decimal_expm(q, length):
    k = len(q)
    with localcontext() as context:
        context.prec = 420
        operator = [[Decimal(str(q[i, j])) * Decimal(str(length)) for j in range(k)] for i in range(k)]
        term = [[Decimal(int(i == j)) for j in range(k)] for i in range(k)]
        total = [row.copy() for row in term]
        for n in range(1, 60):
            term = [[sum(term[i][z] * operator[z][j] for z in range(k)) / n for j in range(k)] for i in range(k)]
            total = [[total[i][j] + term[i][j] for j in range(k)] for i in range(k)]
        return np.array(total, dtype=float)


def edge_checks():
    checks = []
    rng = np.random.default_rng(1909)
    dense = rng.random((8, 8))
    dense = (dense + dense.T) / 2
    np.fill_diagonal(dense, 0)
    chain = np.zeros((8, 8))
    for i in range(7):
        chain[i, i + 1] = chain[i + 1, i] = .7
    reducible = np.zeros((8, 8))
    reducible[:4, :4] = dense[:4, :4]
    reducible[4:, 4:] = dense[4:, 4:]
    directed = np.zeros((8, 8))
    for i in range(8):
        directed[i, (i + 1) % 8] = .7
    underflow = np.full((4, 4), 1e-320)
    for i in range(3):
        underflow[i, i + 1] = underflow[i + 1, i] = .7
    for name, base in (('dense', dense), ('chain', chain), ('reducible', reducible),
                       ('directed_cycle', directed), ('underflow_first_path', underflow),
                       ('zero_generator', np.zeros((4, 4)))):
        q = base.copy()
        np.fill_diagonal(q, 0)
        np.fill_diagonal(q, -q.sum(axis=1))
        k = len(q)
        for length in (0., 1e-100, 1e-10, 1e-5, .1, 1., 100.):
            before = ORIGINAL([-1, 0], [0, length], q, np.ones(k) / k)
            after = GraphModel([-1, 0], [0, length], q, np.ones(k) / k)
            after.statistics = dict(accepted_orders=[], zero_guards=0)
            expected = decimal_expm(q, length) if 0 < length <= 1 else expm(q * length)
            actual = after.transition(1, 0)
            np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=1e-300)
            assert not np.any((expected > 0) & (actual == 0)), (name, length, 'lost representable probability')
            compare_arrays(before.transition(1, 0), actual)
            if name == 'underflow_first_path' and length == 1e-100:
                assert after.statistics['zero_guards'] > 0
            checks.append(dict(model=name, length=length, reference='Decimal420/59terms' if 0 < length <= 1 else 'scipy.linalg.expm',
                               graph_order=after._minimum_order, zero_guards=after.statistics['zero_guards'],
                               accepted_orders=after.statistics['accepted_orders'], smallest_positive=float(actual[actual > 0].min())))
    # Mixed rates, ambiguity, predictive records, selected/unselected transforms.
    q = np.array([[-1., 1.], [1., -1.]])
    tips = {1: np.array([[1., 0.], [1., 1.], [.3, .7]]),
            2: np.array([[0., 1.], [1., 1.], [1., 0.]])}
    before = ORIGINAL([-1, 0, 0], [0, 0, .4], q, [.5, .5], [0., .5, 2.], [.2, .3, .5])
    after = GraphModel([-1, 0, 0], [0, 0, .4], q, [.5, .5], [0., .5, 2.], [.2, .3, .5])
    for predictive in (False, True):
        metrics = compare_records(before, after, tips, dict(joint=True, predictive=predictive, category_nodes=True))
        checks.append(dict(model='ambiguous_mixed_zero_rate_records', predictive=predictive, **metrics))
    return checks


def supplementary_checks():
    k = 128
    pi = np.ones(k) / k
    q = np.tile(pi, (k, 1)) - np.eye(k)
    model = GraphModel([-1, 0], [0, .1], q, pi)
    assert model._uniform_powers.shape[0] < k
    assert model._uniform_transition(.1) is None
    np.testing.assert_allclose(model.transition(1, 0), expm(q * .1), rtol=3e-14, atol=1e-300)
    checks = [dict(model='capacity_fallback_128_states', unchanged_fallback=True)]
    q = np.zeros((6, 6))
    for i in range(5):
        q[i, i + 1] = .7
    np.fill_diagonal(q, -q.sum(axis=1))
    pi = np.array([0., 0., 0., 0., 0., 1.])
    model = GraphModel([-1, 0], [0, 1e-10], q, pi)
    expected = decimal_expm(q, 1e-10)
    np.testing.assert_allclose(model.transition(1, 0), expected, rtol=3e-14, atol=1e-300)
    checks.append(dict(model='directed_absorbing_chain', graph_order=model._minimum_order,
                       reference='Decimal420/59terms', smallest_reachable_probability=float(expected[expected > 0].min())))
    return checks


def existing_tests():
    import pytest

    class Counts:
        def __init__(self):
            self.passed = self.failed = self.skipped = 0

        def pytest_runtest_logreport(self, report):
            if report.when == 'call':
                self.passed += int(report.passed)
                self.failed += int(report.failed)
            if report.skipped:
                self.skipped += 1

    counts = Counts()
    endpoint.EndpointModel = GraphModel
    try:
        with tempfile.TemporaryDirectory(prefix='csubst-round2-09-pytest-') as directory:
            code = pytest.main(['-q', 'tests/unit/test_endpoint.py', '--basetemp', directory], plugins=[counts])
    finally:
        endpoint.EndpointModel = ORIGINAL
    assert code == 0
    return dict(passed=counts.passed, failed=counts.failed, skipped=counts.skipped,
                scope='existing source endpoint tests with GraphModel injected, not installed wheel')


def main():
    if '--verify-only' in sys.argv:
        result = json.loads((HERE / 'proposal09.json').read_text())
        with benchmark_lock():
            result['supplementary_checks'] = supplementary_checks()
            result['existing_endpoint_tests'] = existing_tests()
        result['prototype_class_sha256'] = hashlib.sha256(inspect.getsource(GraphModel).encode()).hexdigest()
        (HERE / 'proposal09.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        return
    result = dict(proposal='reachability_aware_uniformization_start_order', environment=environment(),
                  source_function_sha256=hashlib.sha256(inspect.getsource(ORIGINAL._uniform_transition).encode()).hexdigest(),
                  prototype_class_sha256=hashlib.sha256(inspect.getsource(GraphModel).encode()).hexdigest(),
                  command='python reports/performance_round2_20261022/proposal09.py',
                  scope='cold model+all transitions and warm endpoint stage; not full CLI', warmups=1, repeats=3,
                  tolerances=dict(rtol=3e-14, atol=1e-300), memory_scope='array bytes, not RSS', cases={})
    with benchmark_lock():
        result['edge_checks'] = edge_checks()
        result['supplementary_checks'] = supplementary_checks()
        result['existing_endpoint_tests'] = existing_tests()
        for dataset in ('PGK', 'PEPC'):
            with tempfile.TemporaryDirectory(prefix='csubst-round2-09-') as directory:
                result['cases'][dataset] = real_case(dataset, Path(directory))
    (HERE / 'proposal09.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({name: {scope: value[scope]['summary'] for scope in ('cold_model_and_all_transitions', 'warm_endpoint_stage')}
                      for name, value in result['cases'].items()}, indent=2))


if __name__ == '__main__':
    main()
