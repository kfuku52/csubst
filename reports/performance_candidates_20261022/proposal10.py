"""Compare explicit endpoint matmul outputs, without changing the source tree.

Run from the repository root: python reports/performance_candidates_20261022/proposal10.py
"""

import contextlib
import gc
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
import textwrap

for variable in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np  # noqa: E402

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import endpoint  # noqa: E402
from proposal02 import capture, fields  # noqa: E402

HERE = Path(__file__).resolve().parent
ORIGINAL = endpoint.EndpointModel.iter_blocks


def make_prototype():
    source = textwrap.dedent(inspect.getsource(ORIGINAL))
    replacements = {
        'post[c, child] = inside[c, child] @ self.transition(child, c).T':
            'np.matmul(inside[c, child], self.transition(child, c).T, out=post[c, child])',
        'post[c, child] = (left @ transition) * right':
            'np.matmul(left, transition, out=post[c, child])\n'
            '                np.multiply(post[c, child], right, out=post[c, child])',
    }
    for before, after in replacements.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    namespace = dict(endpoint.__dict__)
    exec(compile(source, '<proposal10-out-buffers>', 'exec'), namespace)
    return namespace['iter_blocks']


PROTOTYPE = make_prototype()


def compare(model, tips, options, block=64, retained=False):
    before = ORIGINAL(model, tips, block, **options)
    after = PROTOTYPE(model, tips, block, **options)
    if retained:
        before, after = list(before), list(after)
    maximum = 0.0
    count = 0
    records = 0
    for old, new in zip(before, after, strict=True):
        assert (old.start, old.stop, old.child, old.parent) == (
            new.start, new.stop, new.child, new.parent)
        old_values, new_values = fields(old), fields(new)
        assert old_values.keys() == new_values.keys()
        for key, value in old_values.items():
            candidate = new_values[key]
            np.testing.assert_allclose(candidate, value, rtol=1e-12, atol=1e-13)
            maximum = max(maximum, float(np.max(np.abs(candidate - value), initial=0)))
            count += value.size
        records += 1
    return {'asserted_values': count, 'records': records,
            'maximum_absolute_difference': maximum, 'retained_until_exhaustion': retained}


def consume(function, model, tips, options):
    checksum, count = 0.0, 0
    for record in function(model, tips, 64, **options):
        for value in fields(record).values():
            checksum += float(value.sum())
            count += value.size
    return checksum, count


def paired(model, tips, options):
    expected = consume(ORIGINAL, model, tips, options)
    assert consume(PROTOTYPE, model, tips, options) == expected
    samples = {'before': [], 'after': []}
    for repeat in range(3):
        order = (('before', ORIGINAL), ('after', PROTOTYPE))
        if repeat % 2:
            order = tuple(reversed(order))
        for name, function in order:
            result, timing = measure(consume, function, model, tips, options)
            assert result == expected
            samples[name].append(timing)
    sites = len(next(iter(tips.values())))
    n, k, c = model.parents.size, model.pi.size, model.rates.size
    return {
        'samples': samples,
        'summary': {name: summarize(values) for name, values in samples.items()},
        'principal_buffer_bytes_unchanged': int(c * n * min(sites, 64) * (2 * k + 1) * 8),
        'explicit_output_allocation_volume_avoided_bytes': int(3 * c * (n - 1) * sites * k * 8),
        'maximum_two_explicit_result_arrays_bytes': int(2 * min(sites, 64) * k * 8),
        'extra_retained_workspace_bytes': 0,
    }


def edge_checks():
    rng = np.random.default_rng(810)
    parents = np.array([-1, 0, 1, 1, 0])
    lengths = np.array([0, 0, 1e-10, .3, .2])
    pi = np.array([.2, .3, .5])
    q = np.tile(pi, (3, 1)) - np.eye(3)
    regular = {leaf: rng.uniform(.1, 1, size=(67, 3)) for leaf in (2, 3, 4)}
    for value in regular.values():
        value[4:7] = 1
        value[9] = [1, 1, 0]
    checks = {}
    for rate_name, rates, weights in (
            ('uniform', [1], [1]), ('invariant_mixture', [0, .3, 2], [.1, .3, .6])):
        for input_name in ('regular', 'read_only_strided'):
            tips = regular
            if input_name == 'read_only_strided':
                tips = {}
                for key, value in regular.items():
                    backing = np.zeros((134, 6))
                    backing[::2, ::2] = value
                    tips[key] = backing[::2, ::2]
                    tips[key].flags.writeable = False
            model = endpoint.EndpointModel(parents, lengths, q, pi, rates, weights)
            options = {'predictive': True, 'category_nodes': True, 'branch_ids': [1, 3]}
            for block in (1, 4, 64):
                name = f'{rate_name}_{input_name}_{block}'
                checks[name] = compare(model, tips, options, block)
            checks[f'{rate_name}_{input_name}_retained'] = compare(
                model, tips, options, 4, retained=True)

            def alias_transform(left, right, transition):
                return {'left_alias': left, 'right_alias': right,
                        'mass': np.einsum('si,ij,sj->s', left, transition, right)}

            options.update(transform=alias_transform, unselected_transform=alias_transform)
            checks[f'{rate_name}_{input_name}_transform_alias_retained'] = compare(
                model, tips, options, 4, retained=True)
    model = endpoint.EndpointModel(parents, lengths, q, pi)
    checks['empty_sites'] = compare(model, {key: value[:0] for key, value in regular.items()}, {})
    return checks


def existing_tests():
    import pytest

    endpoint.EndpointModel.iter_blocks = PROTOTYPE
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = int(pytest.main(['-q', 'tests/unit/test_endpoint.py']))
    finally:
        endpoint.EndpointModel.iter_blocks = ORIGINAL
    assert code == 0, output.getvalue()
    return {'command': 'python -m pytest -q tests/unit/test_endpoint.py (prototype substituted in process)',
            'exit_code': code, 'output': output.getvalue()}


def main():
    result = {
        'proposal': 10, 'environment': environment(), 'warmups_per_implementation': 1,
        'measured_repeats': 3, 'block_size': 64,
        'scope': 'Endpoint inference and classified reduction; not CLI',
        'thread_control': 'VECLIB/OPENBLAS/OMP/MKL/BLIS env=1 before NumPy import',
        'memory_scope': 'Explicit ndarray buffers and allocation volume; peak RSS and BLAS workspace not measured',
        'cases': {},
    }
    with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-proposal10-') as directory:
        workdir = Path(directory)
        for dataset in ('PGK', 'PEPC'):
            model, tips, options, hashes = capture(dataset, workdir)
            cases = [(dataset, tips)]
            if dataset == 'PEPC':
                cases.append(('PEPCx4', {key: np.tile(value, (4, 1)) for key, value in tips.items()}))
            for name, values in cases:
                result['cases'][name] = {
                    'sites': len(next(iter(values.values()))), 'nodes': int(model.parents.size),
                    'states': int(model.pi.size), 'rates': model.rates.tolist(),
                    'weights': model.weights.tolist(), 'input_sha256': hashes,
                    'parity': compare(model, values, options),
                    'timing': paired(model, values, options),
                }
                print(name, result['cases'][name]['timing']['summary'], flush=True)
            del model, tips, cases
            gc.collect()
        result['edge_checks'] = edge_checks()
        result['existing_tests'] = existing_tests()
    (HERE / 'proposal10.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
