"""Reproduce a selected-pair NumPy endpoint-event fallback investigation.

Run from the repository root: python reports/performance_round3_20261023/proposal02.py
No production modules are modified. Native dispatch and non-partition mappings
retain the baseline behavior. The baseline source tree can be supplied with
--baseline-root; otherwise the parent's saved snapshot marker is used.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np  # noqa: E402

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from csubst import genetic_code  # noqa: E402


def grouped_event_transform(module, g, kinds, mappings):
    """Keep native dispatch; group only the selected pair products in NumPy."""
    native = getattr(module.substitution_sparse.substitution_sparse_cy,
                     'project_endpoint_events_double', None)
    specs = {}
    for kind in kinds:
        if kind == 'S':
            ng, ns = len(g['amino_acid_orders']), g['max_synonymous_size']
            pairs = [(group * ns * ns + a * ns + d, int(ca), int(cd))
                     for group, aa in enumerate(g['amino_acid_orders'])
                     for a, ca in enumerate(g['synonymous_indices'][aa])
                     for d, cd in enumerate(g['synonymous_indices'][aa]) if a != d]
        else:
            mapping = mappings[kind]
            ng, ns = 1, mapping.shape[1]
            if not np.all((mapping == 0) | (mapping == 1)) or not np.all(mapping.sum(axis=1) == 1):
                specs[kind] = None
                continue
            groups = mapping.argmax(axis=1)
            pairs = [(int(a * ns + d), ca, cd)
                     for ca, a in enumerate(groups) for cd, d in enumerate(groups) if a != d]
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 3)
        if native is None and len(pairs):
            pairs = pairs[np.argsort(pairs[:, 0], kind='stable')]
            starts = np.r_[0, np.flatnonzero(np.diff(pairs[:, 0])) + 1]
            targets = pairs[starts, 0]
        else:
            starts, targets = np.empty(0, dtype=int), np.empty(0, dtype=int)
        specs[kind] = pairs, ng, ns, starts, targets

    def transform(left, right, transition):
        out = {}
        raw = None
        for kind in kinds:
            spec = specs[kind]
            if native is not None and spec is not None:
                pairs, ng, ns, _, _ = spec
                events = native(left, right, transition, pairs, ng, ns)
            elif spec is not None:
                pairs, ng, ns, starts, targets = spec
                events = np.zeros((left.shape[0], ng * ns * ns))
                if len(pairs):
                    _, ca, cd = pairs.T
                    values = left[:, ca] * transition[ca, cd]
                    values *= right[:, cd]
                    if len(starts) == len(pairs):
                        events[:, targets] = values
                    else:
                        events[:, targets] = np.add.reduceat(values, starts, axis=1)
                events = events.reshape(left.shape[0], ng, ns, ns)
            else:
                if raw is None:
                    raw = left[:, :, None] * transition * right[:, None, :]
                events = module._events(raw, kind, g, mappings[kind])
            out[kind, 'events'] = events
        return out
    return transform


def candidate_event_transform(module, g, kinds, mappings):
    # Build exactly the proposed patch in an isolated namespace once. No live
    # production function or file is changed, including during validation.
    if not hasattr(module, '_proposal02_narrow'):
        source = inspect.getsource(module._event_transform)
        source = source.replace('    def transform(left, right, transition):', '''    # Structural joint endpoints use one-to-one reported states. Select their
    # off-diagonal products directly when the optional accelerator is absent.
    selected = (kinds == ['N'] and specs['N'] is not None
                and mappings['N'].shape[0] == mappings['N'].shape[1]
                and np.all(mappings['N'].sum(axis=0) == 1))

    def transform(left, right, transition):''', 1)
        source = source.replace('''                events = native(left, right, transition, pairs, ng, ns)
            else:''', '''                events = native(left, right, transition, pairs, ng, ns)
            elif selected:
                pairs, ng, ns = spec
                event, ca, cd = pairs.T
                values = left[:, ca] * transition[ca, cd] * right[:, cd]
                events = np.zeros((left.shape[0], ng * ns * ns),
                                  dtype=np.result_type(values.dtype, mappings[kind].dtype))
                events[:, event] = values
                events = events.reshape(left.shape[0], ng, ns, ns)
            else:''', 1)
        namespace = dict(module.__dict__)
        exec(compile(source, '<proposal02_narrow>', 'exec'), namespace)
        module._proposal02_narrow = namespace['_event_transform']
        module._proposal02_narrow_source = source
    return module._proposal02_narrow(g, kinds, mappings)


def load_baseline(path):
    spec = importlib.util.spec_from_file_location('proposal02_baseline', path / 'csubst' / 'endpoint_io.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context(states=61, recoded=False):
    if states == 61:
        table = [(aa, codon) for aa, codon in genetic_code.get_codon_table(1) if aa != '*']
        table.sort(key=lambda item: item[1])
        orders = sorted({aa for aa, _ in table})
        indices = {aa: [i for i, entry in enumerate(table) if entry[0] == aa] for aa in orders}
    else:
        orders = ['X', 'Y']
        indices = {'X': [0, 1], 'Y': [2]}
    g = {'amino_acid_orders': orders, 'max_synonymous_size': max(map(len, indices.values())),
         'synonymous_indices': indices}
    mapping = np.zeros((states, 6 if recoded else len(orders)))
    for i, aa in enumerate(orders):
        mapping[indices[aa], i % 6 if recoded else i] = 1
    return g, {'S': None, 'N': mapping}


def inputs(size, states, seed, strided=False):
    rng = np.random.default_rng(seed)
    left = rng.random((size, states))
    right = rng.random((size, states))
    # Include resolved, missing, and partial-ambiguity likelihood vectors.
    left[::11] = 0
    left[::11, 0] = 1
    right[::13] = 1
    right[::17] = 0
    right[::17, :2] = 1
    transition = rng.random((states, states))
    transition /= transition.sum(axis=1, keepdims=True)
    if strided:
        def stride(array):
            expanded = np.zeros(tuple(2 * dim for dim in array.shape))
            expanded[::2, ::2] = array
            return expanded[::2, ::2]
        left, right, transition = map(stride, (left, right, transition))
    for array in (left, right, transition):
        array.flags.writeable = False
    return left, right, transition


def independent(left, right, transition, g, mappings, kinds):
    result = {}
    for kind in kinds:
        if kind == 'S':
            ns, ng = g['max_synonymous_size'], len(g['amino_acid_orders'])
        else:
            ns, ng = mappings[kind].shape[1], 1
        events = np.zeros((len(left), ng, ns, ns))
        for s in range(len(left)):
            for ca in range(left.shape[1]):
                for cd in range(left.shape[1]):
                    weight = left[s, ca] * transition[ca, cd] * right[s, cd]
                    if kind == 'S':
                        for group, aa in enumerate(g['amino_acid_orders']):
                            indices = g['synonymous_indices'][aa]
                            if ca in indices and cd in indices and ca != cd:
                                events[s, group, indices.index(ca), indices.index(cd)] += weight
                    else:
                        for a in range(ns):
                            for d in range(ns):
                                if a != d:
                                    events[s, 0, a, d] += weight * mappings[kind][ca, a] * mappings[kind][cd, d]
        result[kind, 'events'] = events
    return result


def validate(module, factory=grouped_event_transform):
    module.substitution_sparse.substitution_sparse_cy = None
    checks, maximum = [], 0.0
    for states in (3, 61):
        for size in (0, 1, 8, 64, 129, 256):
            for strided in (False, True):
                for recoded in (False, True) if states == 61 else (False,):
                    g, mappings = context(states, recoded)
                    data = inputs(size, states, 31415 + size, strided)
                    old = module._event_transform(g, ['S', 'N'], mappings)(*data)
                    new = factory(module, g, ['S', 'N'], mappings)(*data)
                    for key in old:
                        np.testing.assert_allclose(new[key], old[key], atol=1e-15, rtol=1e-14)
                        maximum = max(maximum, float(np.max(np.abs(new[key] - old[key]), initial=0)))
                    checks.append([states, size, strided, recoded])
    g, mappings = context(3)
    data = inputs(7, 3, 11, True)
    for mapping in (mappings['N'], np.array([[.5, .5], [1., 0], [0, 1.]]),
                    np.ones((3, 1)), np.eye(3), np.array([[1., 1], [0, 1], [1., 0]])):
        mappings['N'] = mapping
        expected = independent(*data, g, mappings, ['S', 'N'])
        actual = factory(module, g, ['S', 'N'], mappings)(*data)
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-14, atol=1e-15)
    # Native dispatch is preserved, including argument order and output object.
    class Sentinel:
        calls = []
        @classmethod
        def project_endpoint_events_double(cls, *args):
            cls.calls.append(args)
            return cls
    module.substitution_sparse.substitution_sparse_cy = Sentinel
    native = factory(module, g, ['S', 'N'], {'S': None, 'N': np.eye(3)})(*data)
    assert len(Sentinel.calls) == 2 and all(value is Sentinel for value in native.values())
    module.substitution_sparse.substitution_sparse_cy = None
    return {'comparison_cases': len(checks), 'max_abs_diff': maximum,
            'tolerance': {'rtol': 1e-14, 'atol': 1e-15},
            'independent_cases': 5, 'native_dispatch_preserved': True,
            'readonly_strided_empty_singleton_fractional_overlapping_mappings': 'passed'}


def execute(transform, blocks, cycles):
    result = None
    for _ in range(cycles):
        for block in blocks:
            result = transform(*block)
    return result


def benchmark(module, repeats):
    timings = []
    for label, total_sites, block_size, kinds, recoded in (
        ('PGK_shape_SN64', 417, 64, ['S', 'N'], False),
        ('PEPC_shape_SN64', 971, 64, ['S', 'N'], False),
        ('PEPC_shape_SN8', 971, 8, ['S', 'N'], False),
        ('PEPC_shape_SN256', 971, 256, ['S', 'N'], False),
        ('PEPC_shape_S_only64', 971, 64, ['S'], False),
        ('PEPC_shape_recode64', 971, 64, ['S', 'N'], True)):
        g, mappings = context(61, recoded)
        blocks = [inputs(min(block_size, total_sites - start), 61, start + 42)
                  for start in range(0, total_sites, block_size)]
        old = module._event_transform(g, kinds, mappings)
        new = grouped_event_transform(module, g, kinds, mappings)
        cycles = 24 if block_size >= 64 else 8
        # Both versions receive one complete excluded warmup.
        execute(old, blocks, cycles)
        execute(new, blocks, cycles)
        samples = {'baseline': [], 'candidate': []}
        for repeat in range(repeats):
            order = [('baseline', old), ('candidate', new)]
            if repeat % 2:
                order.reverse()
            for name, function in order:
                _, sample = measure(execute, function, blocks, cycles)
                samples[name].append(sample)
        summary = {key: summarize(value) for key, value in samples.items()}
        speedup = summary['baseline']['wall_s']['median'] / summary['candidate']['wall_s']['median']
        timings.append({'name': label, 'sites': total_sites, 'states': 61,
                        'block_size': block_size, 'kinds': kinds, 'cycles': cycles,
                        'calls_per_sample': len(blocks) * cycles,
                        'samples': samples, 'summary': summary, 'speedup': speedup})
        print(label, f'{speedup:.3f}x', flush=True)
    return timings


def benchmark_narrow(module, repeats):
    timings = []
    for label, block_size in [('3di20_identity64', 64), ('3di20_identity256', 256)]:
        mappings = {'N': np.eye(20)}
        blocks = [inputs(min(block_size, 971 - start), 20, start + 42)
                  for start in range(0, 971, block_size)]
        old = module._event_transform({}, ['N'], mappings)
        new = candidate_event_transform(module, {}, ['N'], mappings)
        maximum = 0.
        for block in blocks:
            a, b = old(*block), new(*block)
            np.testing.assert_array_equal(a['N', 'events'], b['N', 'events'])
            maximum = max(maximum, float(np.max(np.abs(a['N', 'events'] - b['N', 'events']))))
        cycles = 160
        execute(old, blocks, cycles)
        execute(new, blocks, cycles)
        samples = {'baseline': [], 'candidate': []}
        for repeat in range(repeats):
            order = [('baseline', old), ('candidate', new)]
            if repeat % 2:
                order.reverse()
            for name, function in order:
                _, sample = measure(execute, function, blocks, cycles)
                samples[name].append(sample)
        summary = {key: summarize(value) for key, value in samples.items()}
        speedup = summary['baseline']['wall_s']['median'] / summary['candidate']['wall_s']['median']
        timings.append({'name': label, 'sites': 971, 'states': 20,
                        'block_size': block_size, 'cycles': cycles, 'calls_per_sample': len(blocks) * cycles,
                        'samples': samples, 'summary': summary, 'speedup': speedup,
                        'max_abs_diff': maximum, 'array_equal': True})
        print(label, f'{speedup:.3f}x', flush=True)
    return timings


def validate_narrow(module):
    module.substitution_sparse.substitution_sparse_cy = None
    cases = 0
    for size in (0, 1, 8, 129):
        for strided in (False, True):
            data = inputs(size, 20, 1928 + size, strided)
            for mapping in (np.eye(20), np.eye(20)[::-1]):
                for dtype in (np.float64, np.float32, np.int16, np.complex128):
                    values = tuple(array.astype(dtype, copy=False) for array in data)
                    spec = {'N': mapping.astype(dtype, copy=False)}
                    a = module._event_transform({}, ['N'], spec)(*values)['N', 'events']
                    b = candidate_event_transform(module, {}, ['N'], spec)(*values)['N', 'events']
                    assert a.dtype == b.dtype
                    np.testing.assert_array_equal(a, b)
                    cases += 1
                # Dtype promotion must follow left*transition*right, then the
                # mapping matrices. Mixed inputs exercise the cast boundary.
                for dtypes in ((np.float32, np.float64, np.float32, np.float32),
                               (np.float32, np.float32, np.float32, np.complex128)):
                    left, right, transition = [array.astype(dtype) for array, dtype in zip(data, dtypes[:3])]
                    spec = {'N': mapping.astype(dtypes[3])}
                    a = module._event_transform({}, ['N'], spec)(left, right, transition)['N', 'events']
                    b = candidate_event_transform(module, {}, ['N'], spec)(left, right, transition)['N', 'events']
                    assert a.dtype == b.dtype
                    np.testing.assert_array_equal(a, b)
                    cases += 1
    # Independently enumerate three-state one-to-one permutation events.
    data = inputs(7, 3, 11, True)
    mapping = {'N': np.eye(3)[::-1]}
    expected = independent(*data, {}, mapping, ['N'])
    actual = candidate_event_transform(module, {}, ['N'], mapping)(*data)
    np.testing.assert_array_equal(actual['N', 'events'], expected['N', 'events'])
    return {'identity_permutation_dtype_cases': cases, 'independent_permutation': 'passed',
            'array_equal': True, 'dtype_equal': True, 'mixed_float_and_complex': 'passed'}


def memory_worker(module, version, size, kind):
    module.substitution_sparse.substitution_sparse_cy = None
    g, mappings = context()
    function = module._event_transform(g, kind.split(','), mappings) if version == 'baseline' else grouped_event_transform(module, g, kind.split(','), mappings)
    data = inputs(size, 61, 1024)
    result = function(*data)
    assert all(np.isfinite(value).all() for value in result.values())
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({'version': version, 'sites': size, 'kinds': kind,
                      'peak_rss_bytes': rss if sys.platform == 'darwin' else rss * 1024}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-root', '--baseline', type=Path,
                        default=os.environ.get('CSUBST_PERF_BASELINE'))
    parser.add_argument('--phase', choices=['initial', 'narrow'], default='initial')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--rss-worker', choices=['baseline', 'candidate'])
    parser.add_argument('--rss-sites', type=int, default=4096)
    parser.add_argument('--rss-kinds', default='S,N')
    args = parser.parse_args()
    baseline = args.baseline_root
    if baseline is None:
        baseline = Path('/tmp/csubst-round3-baseline-path').read_text().strip()
        baseline = Path(baseline)
    module = load_baseline(baseline)
    if args.rss_worker:
        memory_worker(module, args.rss_worker, args.rss_sites, args.rss_kinds)
        return
    if args.repeats < 5:
        parser.error('Use at least five interleaved repetitions.')
    if args.phase == 'narrow':
        with benchmark_lock():
            path = Path(__file__).with_suffix('.json')
            result = json.loads(path.read_text())
            result['narrow_validation'] = validate(module, candidate_event_transform)
            result['narrow_identity_validation'] = validate_narrow(module)
            result['narrow_timings'] = benchmark_narrow(module, args.repeats)
            result['narrow_measured_implementation_sha256'] = hashlib.sha256(
                module._proposal02_narrow_source.encode()).hexdigest()
            result['narrow_timing_status'] = 'exact final patch measured'
            path.write_text(json.dumps(result, indent=2) + '\n')
        return
    with benchmark_lock():
        result = {'candidate': 'NumPy selected classified endpoint pair reduction',
                  'baseline_commit': '624bb8a', 'environment': environment(),
                  'validation': validate(module), 'timings': benchmark(module, args.repeats),
                  'peak_rss': []}
        for kind in ('S,N', 'S'):
            for sites in (64, 4096):
                for version in ('baseline', 'candidate'):
                    command = [sys.executable, str(Path(__file__).resolve()), '--baseline-root', str(baseline),
                               '--rss-worker', version, '--rss-sites', str(sites), '--rss-kinds', kind]
                    output = subprocess.check_output(command, text=True)
                    result['peak_rss'].append(json.loads(output.splitlines()[-1]))
        Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
