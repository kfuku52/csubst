"""Evaluate exact one-hot leaf message gathers; production code is never edited.

Run: python reports/performance_round2_20261022/proposal01.py
"""

import gc
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap

for variable in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np  # noqa: E402

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import endpoint  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ORIGINAL = endpoint.EndpointModel.iter_blocks


def prototype():
    """Specialize only completely one-hot tip blocks after original validation."""
    source = textwrap.dedent(inspect.getsource(ORIGINAL))
    old = '    n, k, nc = self.parents.size, self.pi.size, self.rates.size\n'
    new = '''    tip_info = {}
    for node, values in arrays.items():
        codes = values.argmax(axis=1)
        valid = ((np.count_nonzero(values, axis=1) == 1)
                 & (values[np.arange(values.shape[0]), codes] == 1))
        tip_info[node] = (codes, valid)
    n, k, nc = self.parents.size, self.pi.size, self.rates.size
'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = '        size = stop - start\n'
    new = '''        size = stop - start
        block_codes = {node: codes[start:stop]
                       for node, (codes, valid) in tip_info.items()
                       if valid[start:stop].all()}
'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = '                        post[c, child] = inside[c, child] @ self.transition(child, c).T\n'
    new = '''                        if child in block_codes:
                            post[c, child] = self.transition(child, c).T[block_codes[child]]
                        else:
                            post[c, child] = inside[c, child] @ self.transition(child, c).T
'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    gather = 'post[c, child] = self.transition(child, c).T[block_codes[child]]'
    assert source.count(gather) == 1
    gather_line = next(i for i, line in enumerate(source.splitlines(), 1) if gather in line)
    namespace = dict(endpoint.__dict__)
    exec(compile(source, '<one_hot_leaf_prototype>', 'exec'), namespace)
    function = namespace['iter_blocks']
    function._gather_line = gather_line
    return function


CANDIDATE = prototype()


def load_capture():
    # Import only; its main and fixture-generating harness main are not invoked.
    path = ROOT / 'reports/performance_candidates_20261022/proposal02.py'
    spec = importlib.util.spec_from_file_location('endpoint_capture_round2', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arrays_of(record):
    for name in ('node', 'joint', 'predictive', 'category_node'):
        value = getattr(record, name)
        if value is not None:
            yield name, value
    for name in ('reduced', 'reduced_predictive'):
        for key, value in (getattr(record, name) or {}).items():
            yield (name, key), value


def compare(model, tips, options, block_size):
    from itertools import zip_longest
    before = ORIGINAL(model, tips, block_size=block_size, **options)
    after = CANDIDATE(model, tips, block_size=block_size, **options)
    count = records = 0
    for left, right in zip_longest(before, after):
        assert left is not None and right is not None
        assert (left.start, left.stop, left.child, left.parent) == (
            right.start, right.stop, right.child, right.parent)
        left_arrays, right_arrays = dict(arrays_of(left)), dict(arrays_of(right))
        assert left_arrays.keys() == right_arrays.keys()
        for key, value in left_arrays.items():
            np.testing.assert_array_equal(value, right_arrays[key], strict=True)
            assert value.tobytes() == right_arrays[key].tobytes()
            count += value.size
        records += 1
    return {'records': records, 'asserted_values': count, 'bitwise_exact': True}


def consume(function, model, tips, options, block_size):
    count = checksum = 0
    for record in function(model, tips, block_size=block_size, **options):
        count += 1
        checksum += float(record.node.sum())
    return count, checksum


def verify_gather_execution(model, tips, options, block_size=64):
    """Trace the actual gather line outside timing; preparation alone cannot pass."""
    count = 0

    def local_trace(frame, event, argument):
        nonlocal count
        if event == 'line' and frame.f_lineno == CANDIDATE._gather_line:
            count += 1
        return local_trace

    def global_trace(frame, event, argument):
        if frame.f_code is CANDIDATE.__code__:
            return local_trace
        return None

    previous = sys.gettrace()
    try:
        sys.settrace(global_trace)
        consume(CANDIDATE, model, tips, options, block_size)
    finally:
        sys.settrace(previous)
    expected = coverage(tips, block_size)['gather_eligible_blocks'] * model.rates.size
    assert count == expected and count > 0
    return {'gather_line_executions': count, 'expected': int(expected),
            'tracing_scope': 'one candidate pass outside measured samples',
            'execution_verified': True}


def benchmark(model, tips, options, block_size=64):
    sample = {'baseline': [], 'candidate': []}
    functions = {'baseline': ORIGINAL, 'candidate': CANDIDATE}
    expected = None
    for repeat in range(6):
        order = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
        for name in order:
            gc.collect()
            result, timing = measure(consume, functions[name], model, tips, options, block_size)
            if expected is None:
                expected = result
            assert expected == result
            if repeat:
                sample[name].append(timing)
    return {name: {'samples': samples, 'summary': summarize(samples)}
            for name, samples in sample.items()}


def coverage(tips, block_size):
    total = eligible = metadata = 0
    for values in tips.values():
        codes = values.argmax(axis=1)
        valid = ((np.count_nonzero(values, axis=1) == 1)
                 & (values[np.arange(values.shape[0]), codes] == 1))
        metadata += codes.nbytes + valid.nbytes
        for start in range(0, len(values), block_size):
            total += 1
            eligible += int(valid[start:start + block_size].all())
    return {'leaf_blocks': total, 'gather_eligible_blocks': eligible,
            'retained_tip_metadata_bytes': metadata,
            'temporary_gather_bytes_upper_bound': min(block_size, len(values)) * values.shape[1] * 8,
            'memory_metric': 'array bytes, not peak RSS; one gather replaces one matmul output'}


def synthetic_checks():
    rng = np.random.default_rng(9021)
    parents = np.array([-1, 0, 0, 1, 1, 2, 2])
    lengths = np.array([0, .1, 0, .2, 1e-6, .1, .3])
    q = np.ones((4, 4)) / 3
    np.fill_diagonal(q, -1)
    model = endpoint.EndpointModel(parents, lengths, q, np.ones(4) / 4,
                                   rates=[0, .2, 1.1, 3], weights=[.1, .2, .3, .4])
    tips = {leaf: np.eye(4)[rng.integers(4, size=131)] for leaf in model.leaves}
    out = []
    for case in ('one_hot', 'ambiguous_missing_scaled', 'empty'):
        if case == 'ambiguous_missing_scaled':
            for leaf in tips:
                tips[leaf][5] = 1  # Unobserved tip: retain original likelihoods.
                tips[leaf][17] = [.5, .5, 0, 0]
                tips[leaf][90] *= 2  # Nonunit singleton is not a gather.
        if case == 'empty':
            tips = {leaf: values[:0] for leaf, values in tips.items()}
        for block in (1, 32, 64, 200):
            for transformed in (False, True):
                options = {'predictive': True, 'category_nodes': True, 'branch_ids': [1, 3, 5]}
                if transformed:
                    def transform(left, right, transition):
                        return {'events': left[:, :, None] * transition * right[:, None, :]}
                    options['transform'] = transform
                    options['unselected_transform'] = transform
                result = compare(model, tips, options, block)
                out.append({'case': case, 'block': block, 'transform': transformed, **result})
    invalid_cases = []
    model = endpoint.EndpointModel([-1, 0, 0], [0, 0, 0], [[-1, 1], [1, -1]], [.5, .5])
    checks = [({1: np.array([[1., 0]]), 2: np.array([[0., 1]])}, {}, 'Zero likelihood'),
              ({1: np.array([[1., 0]]), 2: np.array([[1., 0]])}, {'block_size': 0}, 'block size'),
              ({1: np.array([[-1., 0]]), 2: np.array([[1., 0]])}, {}, 'nonnegative'),
              ({1: np.array([[np.nan, 0]]), 2: np.array([[1., 0]])}, {}, 'finite'),
              ({1: np.array([[1., 0]])}, {}, 'every leaf')]
    for values, options, label in checks:
        errors = []
        for function in (ORIGINAL, CANDIDATE):
            try:
                list(function(model, values, **options))
            except ValueError as error:
                errors.append(str(error))
        assert len(errors) == 2 and errors[0] == errors[1]
        invalid_cases.append({'case': label, 'error_equal': True})
    return {'exact_cases': out, 'failure_cases': invalid_cases}


def main():
    results = {'proposal': 'exact one-hot leaf message gathers', 'environment': environment(),
               'command': 'python reports/performance_round2_20261022/proposal01.py',
               'scope': 'EndpointModel.iter_blocks stage only; warmed transition caches; no CLI speed claim',
               'harness_correction': {'initial_run_valid': False,
                   'reason': 'Initial prototype missed the final source.replace; it added preparation but never executed gathers.',
                   'resolution': 'Discarded initial measurements and verdict. Replaced gather statement and traced its execution before fresh parity and alternating measurements.',
                   'initial_measurements_included': False},
               'sources': ['https://academic.oup.com/sysbio/article/66/2/205/2420631',
                           'https://numpy.org/doc/stable/user/how-to-index.html'], 'real_cases': {}}
    with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-round2-leaf-') as directory:
        capture = load_capture()
        results['synthetic'] = synthetic_checks()
        for dataset in ('PGK', 'PEPC'):
            workdir = Path(directory) / dataset
            workdir.mkdir()
            model, tips, options, hashes = capture.capture(dataset, workdir)
            entry = {'inputs_sha256': hashes, 'sites': len(next(iter(tips.values()))),
                     'states': model.pi.size, 'nodes': model.parents.size,
                     'categories': model.rates.size,
                     'coverage': coverage(tips, 64),
                     'gather_execution': verify_gather_execution(model, tips, options, 64),
                     'parity': compare(model, tips, options, 64)}
            entry['benchmark'] = benchmark(model, tips, options)
            results['real_cases'][dataset] = entry
    (HERE / 'proposal01.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({name: entry['benchmark'] for name, entry in results['real_cases'].items()}, indent=2))


if __name__ == '__main__':
    sys.exit(main())
