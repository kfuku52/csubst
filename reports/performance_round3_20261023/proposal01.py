"""Evaluate bounded per-generator endpoint transition references.

Run from the repository root: python reports/performance_round3_20261023/proposal01.py
The product module and tests are never edited by this harness.
"""

import gc
import hashlib
import importlib.util
import inspect
import json
import os
import statistics
from pathlib import Path
import sys
import tempfile
import textwrap
import tracemalloc

for variable in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np  # noqa: E402

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASELINE = os.environ.get('CSUBST_PERF_BASELINE')
if BASELINE is None:
    BASELINE = Path('/tmp/csubst-round3-baseline-path').read_text().strip()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASELINE_ENDPOINT_SHA256 = '4d64021b56d99b84655962cd88616d30a812c008bbbfaf119027296a46aa9d1a'
assert hashlib.sha256((Path(BASELINE) / 'csubst/endpoint.py').read_bytes()).hexdigest() == BASELINE_ENDPOINT_SHA256
endpoint = load(Path(BASELINE) / 'csubst/endpoint.py', 'proposal01_endpoint_baseline')
ORIGINAL = endpoint.EndpointModel.iter_blocks


def prototype():
    source = textwrap.dedent(inspect.getsource(ORIGINAL))
    old = '    capacity = min(int(block_size), first.shape[0])\n'
    new = '''    # Only borrow matrices that fit together in the existing bounded cache.
    transitions = ([[None] * n for _ in range(nc)]
                   if n * nc <= min(self._cache_items, 8192) else None)
    capacity = min(int(block_size), first.shape[0])
'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = '                        post[c, child] = inside[c, child] @ self.transition(child, c).T\n'
    new = '''                        if transitions is None:
                            transition = self.transition(child, c)
                        else:
                            transition = transitions[c][child]
                            if transition is None:
                                transition = self.transition(child, c)
                                transitions[c][child] = transition
                        post[c, child] = inside[c, child] @ transition.T
'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = '                transition = self.transition(child, c)\n'
    # The exact indent also occurs within the newly inserted lazy block.
    old = '\n                transition = self.transition(child, c)\n'
    new = '''
                transition = (self.transition(child, c) if transitions is None
                              else transitions[c][child])
'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    namespace = dict(endpoint.__dict__)
    exec(compile(source, '<proposal01-bounded-transitions>', 'exec'), namespace)
    return namespace['iter_blocks'], source


CANDIDATE, CANDIDATE_SOURCE = prototype()


def clone(model):
    return endpoint.EndpointModel(model.parents, model.lengths, model.q, model.pi,
                                  model.rates, model.weights)


def arrays(record):
    for name in ('node', 'joint', 'predictive', 'category_node'):
        value = getattr(record, name)
        if value is not None:
            yield name, value
    for name in ('reduced', 'reduced_predictive'):
        for key, value in (getattr(record, name) or {}).items():
            yield (name, key), value


def compare(model, tips, options, block):
    from itertools import zip_longest
    left_model, right_model = clone(model), clone(model)
    left_model._cache_items = right_model._cache_items = model._cache_items
    values = records = 0
    for left, right in zip_longest(ORIGINAL(left_model, tips, block_size=block, **options),
                                  CANDIDATE(right_model, tips, block_size=block, **options)):
        assert left is not None and right is not None
        assert (left.start, left.stop, left.child, left.parent) == (
            right.start, right.stop, right.child, right.parent)
        first, second = dict(arrays(left)), dict(arrays(right))
        assert first.keys() == second.keys()
        for key, array in first.items():
            np.testing.assert_array_equal(array, second[key], strict=True)
            assert array.tobytes() == second[key].tobytes()
            values += array.size
        records += 1
    assert len(left_model._transitions) <= model._cache_items
    assert len(right_model._transitions) <= model._cache_items
    assert set(left_model._transitions) == set(right_model._transitions)
    for key in left_model._transitions:
        np.testing.assert_array_equal(left_model._transitions[key], right_model._transitions[key])
    return {'records': records, 'values': values, 'bitwise_exact': True,
            'transition_cache_keyset_exact': True,
            'transition_cache_lru_order_exact': list(left_model._transitions) == list(right_model._transitions)}


def consume(function, model, tips, options, block):
    checksum = 0.
    count = 0
    for record in function(model, tips, block_size=block, **options):
        checksum += float(record.node.sum())
        count += 1
    return count, checksum


def call_count(function, model, tips, options, block):
    original = model.transition
    count = 0

    def counted(child, category):
        nonlocal count
        count += 1
        return original(child, category)

    model.transition = counted
    try:
        consume(function, model, tips, options, block)
    finally:
        del model.transition
    return count


def benchmark(model, tips, options, block, repeats=9):
    models = {'baseline': clone(model), 'candidate': clone(model)}
    functions = {'baseline': ORIGINAL, 'candidate': CANDIDATE}
    samples = {'baseline': [], 'candidate': []}
    checksums = {}
    for repeat in range(repeats + 1):
        order = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
        for name in order:
            gc.collect()
            value, timing = measure(consume, functions[name], models[name], tips, options, block)
            checksums.setdefault(name, value)
            assert value == checksums[name]
            if repeat:
                samples[name].append(timing)
    assert checksums['baseline'] == checksums['candidate']
    result = {name: {'samples': sample, 'summary': summarize(sample)}
              for name, sample in samples.items()}
    for name in result:
        result[name]['transition_calls'] = call_count(functions[name], models[name], tips, options, block)
        gc.collect()
        tracemalloc.start()
        consume(functions[name], models[name], tips, options, block)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result[name]['warmed_stage_tracemalloc_peak_bytes'] = peak
    result['paired_wall_ratios'] = [candidate['wall_s'] / baseline['wall_s']
                                   for baseline, candidate in zip(samples['baseline'], samples['candidate'])]
    return result


def synthetic():
    rng = np.random.default_rng(9021)
    parents = [-1, 0, 0, 1, 1, 2, 2]
    q = np.ones((4, 4)) / 3
    np.fill_diagonal(q, -1)
    model = endpoint.EndpointModel(parents, [0, .1, 0, .2, 1e-6, .1, .3], q, np.ones(4) / 4,
                                   rates=[0, .2, 1.1, 3], weights=[.1, .2, .3, .4])
    tips = {leaf: np.eye(4)[rng.integers(4, size=131)] for leaf in model.leaves}
    checks = []
    for case in ('one_hot', 'ambiguous_missing_scaled', 'empty'):
        if case == 'ambiguous_missing_scaled':
            for values in tips.values():
                values[5] = 1
                values[17] = [.5, .5, 0, 0]
                values[90] *= 2
        if case == 'empty':
            tips = {leaf: values[:0] for leaf, values in tips.items()}
        for limit in (1, 1000):
            model._cache_items = limit
            for block in (1, 64, 200):
                for transformed in (False, True):
                    options = {'predictive': True, 'category_nodes': True, 'branch_ids': [1, 3, 5]}
                    if transformed:
                        def transform(left, right, transition):
                            return {'events': left[:, :, None] * transition * right[:, None, :]}
                        options.update(transform=transform, unselected_transform=transform)
                    checks.append({'case': case, 'cache_items': limit, 'block': block,
                                   'transformed': transformed,
                                   **compare(model, tips, options, block)})
    invalid_model = endpoint.EndpointModel([-1, 0, 0], [0, 0, 0], [[-1, 1], [1, -1]], [.5, .5])
    failures = []
    for tips, options in [({1: np.array([[1., 0]]), 2: np.array([[0., 1]])}, {}),
                          ({1: np.array([[1., 0]]), 2: np.array([[1., 0]])}, {'block_size': 0}),
                          ({1: np.array([[-1., 0]]), 2: np.array([[1., 0]])}, {}),
                          ({1: np.array([[np.nan, 0]]), 2: np.array([[1., 0]])}, {}),
                          ({1: np.array([[1., 0]])}, {})]:
        errors = []
        for function in (ORIGINAL, CANDIDATE):
            try:
                list(function(invalid_model, tips, **options))
            except ValueError as error:
                errors.append(str(error))
        assert len(errors) == 2 and errors[0] == errors[1]
        failures.append({'error': errors[0], 'exact': True})
    empty_tips = {1: np.empty((0, 2)), 2: np.empty((0, 2))}
    assert call_count(CANDIDATE, invalid_model, empty_tips, {}, 64) == 0
    fallback = endpoint.EndpointModel([-1, 0, 0], [0, .1, .2], [[-1, 1], [1, -1]], [.5, .5])
    fallback._cache_items = 1
    observations = {1: np.ones((4, 2)), 2: np.ones((4, 2))}
    counts = [call_count(function, fallback, observations, {}, 2)
              for function in (ORIGINAL, CANDIDATE)]
    assert counts == [8, 8]
    return {'exact_cases': checks, 'failure_cases': failures, 'empty_calls_zero': True,
            'fallback_transition_calls': counts}


def finalize(result):
    result['baseline_endpoint_sha256'] = BASELINE_ENDPOINT_SHA256
    result['decision'] = 'reject: mixed CPU pairs and noisy wall times do not establish real-stage improvement'
    result['contention_caveat'] = 'Unrelated CPU-heavy jobs were observed outside the shared repository lock; they were not modified.'
    result['omitted_checks'] = ['Full CLI repeats: stage effects were small and mixed; no whole-CLI speed claim.',
                                'Installed wheel, source-only native-reducer fallback, Linux and scientific calibration: no product change applied.']
    for entry in result['cases'].values():
        benchmark = entry['benchmark']
        ratios = [candidate['cpu_s'] / baseline['cpu_s']
                  for baseline, candidate in zip(benchmark['baseline']['samples'],
                                                 benchmark['candidate']['samples'])]
        benchmark['paired_cpu_ratios'] = ratios
        benchmark['paired_cpu_median_ratio'] = statistics.median(ratios)
        benchmark['paired_cpu_faster_count'] = sum(value < 1 for value in ratios)
    return result


def main():
    result = {'proposal': 'Bounded endpoint transition references per generator',
              'baseline_commit': '624bb8a', 'environment': environment(),
              'command': 'python reports/performance_round3_20261023/proposal01.py',
              'scope': 'EndpointModel.iter_blocks stage only; fixed fit native scan transforms; no CLI claim',
              'warmups': 1, 'measured_repeats': 9, 'block_size': 64,
              'memory_metric': 'warm stage tracemalloc peak; not process RSS', 'cases': {}}
    with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-round3-transitions-') as directory:
        result['synthetic'] = synthetic()
        capture = load(ROOT / 'reports/performance_candidates_20261022/proposal02.py',
                       'proposal01_capture')
        for dataset in ('PGK', 'PEPC'):
            work = Path(directory) / dataset
            work.mkdir()
            captured, tips, options, hashes = capture.capture(dataset, work)
            model = clone(captured)
            entry = {'inputs_sha256': hashes, 'nodes': model.parents.size,
                     'sites': len(next(iter(tips.values()))), 'states': model.pi.size,
                     'categories': model.rates.size, 'cache_items': model._cache_items,
                     'pointer_table_bytes': (sys.getsizeof([[None] * model.parents.size
                                                           for _ in range(model.rates.size)])
                                             + model.rates.size * sys.getsizeof([None] * model.parents.size)),
                     'parity': compare(model, tips, options, 64),
                     'benchmark': benchmark(model, tips, options, 64)}
            result['cases'][dataset] = entry
            (HERE / 'proposal01.json').write_text(json.dumps(result, indent=2) + '\n')
            print(dataset, {name: entry['benchmark'][name]['summary']
                            for name in ('baseline', 'candidate')}, flush=True)
    (HERE / 'proposal01.json').write_text(json.dumps(finalize(result), indent=2) + '\n')


if __name__ == '__main__':
    main()
