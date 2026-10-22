"""Evaluate endpoint site blocking on copied fixed-fit PGK/PEPC inputs.

Run from the repository root: python reports/performance_candidates_20261022/proposal02.py
"""

import contextlib
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

for variable in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np  # noqa: E402 -- thread limits must precede NumPy initialization

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import cli, endpoint  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BLOCKS = (16, 32, 64, 128, 256)


class Captured(BaseException):
    def __init__(self, model, tips, options):
        self.model, self.tips, self.options = model, tips, options


def capture(dataset, workdir):
    data = workdir / 'data'
    data.mkdir(exist_ok=True)
    files = [dataset + '.alignment.fa', dataset + '.tree.nwk', dataset + '.foreground.txt']
    files += [dataset + '.alignment.fa.' + suffix
              for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log')]
    for name in files:
        shutil.copyfile(ROOT / 'csubst/dataset' / name, data / name)
    spec = importlib.util.spec_from_file_location(
        'fixed_fit_harness', ROOT / 'reports/resource_audit_20260911/benchmark.py')
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    command = harness.command_for(dataset.lower() + '_scan_defaults', 'before', ROOT,
                                  data, workdir / (dataset.lower() + '_capture'))
    original = endpoint.EndpointModel.iter_blocks

    def intercept(model, tips, *args, **options):
        options.pop('block_size', None)
        raise Captured(model, tips, options)

    endpoint.EndpointModel.iter_blocks = intercept
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cli.main(command[3:])
    except Captured as captured:
        return captured.model, captured.tips, captured.options, {
            name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in files}
    finally:
        endpoint.EndpointModel.iter_blocks = original
    raise AssertionError('CLI did not reach joint inference')


def fields(record):
    values = {'node': record.node}
    for name in ('joint', 'predictive'):
        if getattr(record, name) is not None:
            values[name] = getattr(record, name)
    if record.category_node is not None:
        values['category_node'] = record.category_node.transpose(1, 0, 2)
    for name in ('reduced', 'reduced_predictive'):
        for key, value in (getattr(record, name) or {}).items():
            values[(name, str(key))] = value
    return values


def reference(model, tips, options):
    sites = len(next(iter(tips.values())))
    outputs = {}
    for record in model.iter_blocks(tips, 32, **options):
        for key, value in fields(record).items():
            index = record.child, key
            if index not in outputs:
                outputs[index] = np.empty((sites,) + value.shape[1:], dtype=value.dtype)
            outputs[index][record.start:record.stop] = value
    return outputs


def assert_reference(model, tips, options, block, outputs, original_sites):
    maximum = 0.0
    count = 0
    for record in model.iter_blocks(tips, block, **options):
        indices = np.arange(record.start, record.stop) % original_sites
        for key, value in fields(record).items():
            expected = outputs[record.child, key][indices]
            np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-13)
            maximum = max(maximum, float(np.max(np.abs(value - expected), initial=0)))
            count += value.size
    return {'asserted_values': count, 'maximum_absolute_difference': maximum,
            'rtol': 1e-12, 'atol': 1e-13}


def consume(model, tips, options, block):
    checksum = 0.0
    count = 0
    for record in model.iter_blocks(tips, block, **options):
        for value in fields(record).values():
            checksum += float(value.sum())
            count += value.size
    return checksum, count


def compare_blocks(model, tips, options, repeats=3):
    # All block variants receive the identical prepared model, warmed transition
    # cache, tip likelihoods, transforms and rate-category treatment.
    consume(model, tips, options, 64)
    samples = {block: [] for block in BLOCKS}
    for repeat in range(repeats + 1):
        order = BLOCKS if repeat % 2 == 0 else tuple(reversed(BLOCKS))
        for block in order:
            _, timing = measure(consume, model, tips, options, block)
            if repeat:
                samples[block].append(timing)
    sites = len(next(iter(tips.values())))
    n, c, k = model.parents.size, model.rates.size, model.pi.size
    return {str(block): {
        'samples': samples[block], 'summary': summarize(samples[block]),
        'principal_buffer_bytes': int(n * c * min(sites, block) * (2 * k + 1) * 8),
        'scan_write_buffer_bytes_if_disk_backed': int(n * min(sites, block) * (20**2 + c * k) * 8),
    } for block in BLOCKS}


def synthetic_checks():
    parents = np.array([-1, 0, 0, 1, 1, 2, 2])
    lengths = np.array([0, .1, 0, .2, 1e-6, .1, .3])
    q = np.ones((4, 4)) / 3
    np.fill_diagonal(q, -1)
    rng = np.random.default_rng(802)
    tips = {leaf: np.eye(4)[rng.integers(4, size=263)] for leaf in (3, 4, 5, 6)}
    for leaf in tips:
        tips[leaf][17:20] = 1
        tips[leaf][9, :2] = 1
    checks = {}
    for name, rates, weights in (
            ('uniform', [1], [1]), ('invariant_mixture', [0, .5, 1.5], [.1, .3, .6])):
        # Site 9 is compatible with the invariant class; other sites can put
        # that class at zero posterior without removing it from the model.
        model = endpoint.EndpointModel(parents, lengths, q, np.ones(4) / 4, rates, weights)
        options = {'branch_ids': [3, 5], 'predictive': True, 'category_nodes': True}
        outputs = reference(model, tips, options)
        for block in BLOCKS:
            # Joint and predictive arrays are checked separately here because
            # this fixture does not use a classified transform.
            checks[name + '_' + str(block)] = assert_reference(
                model, tips, options, block, outputs, 263)
    return checks


def main():
    numpy_config = np.show_config(mode='dicts')
    blas_config = numpy_config['Build Dependencies']['blas']
    result = {'environment': environment(), 'proposal': 2,
              'scope': 'EndpointModel inference with native scan classified-event transform; not CLI',
              'warmups': 1, 'measured_repeats': 3, 'cases': {},
              'thread_control': 'VECLIB/OPENBLAS/OMP/MKL/BLIS env = 1 before NumPy import; no threadpoolctl installed',
              'numpy_configuration': {
                  'blas': {key: blas_config[key] for key in ('name', 'version', 'found')},
                  'simd': numpy_config['SIMD Extensions']},
              'blas_runtime_library': (Path(sys.prefix) / 'lib/libblas.3.dylib').resolve().name,
              'memory_scope': 'inside_buffer + post_buffer + scale_buffer exact ndarray bytes; not RSS'}
    with benchmark_lock(), tempfile.TemporaryDirectory(
            prefix='csubst-proposal02-') as directory:
        workdir = Path(directory)
        for dataset in ('PGK', 'PEPC'):
            model, tips, options, hashes = capture(dataset, workdir)
            original_sites = len(next(iter(tips.values())))
            outputs = reference(model, tips, options)
            cases = [(dataset, tips)]
            if dataset == 'PEPC':
                cases.append(('PEPCx4', {key: np.tile(value, (4, 1)) for key, value in tips.items()}))
            for case, observations in cases:
                checks = {str(block): assert_reference(
                    model, observations, options, block, outputs, original_sites) for block in BLOCKS}
                # Reference outputs deliberately stay outside the measured
                # function and are not read by it.
                result['cases'][case] = {
                    'nodes': int(model.parents.size), 'leaves': len(tips),
                    'sites': len(next(iter(observations.values()))), 'states': int(model.pi.size),
                    'rates': model.rates.tolist(), 'weights': model.weights.tolist(),
                    'input_sha256': hashes, 'parity': checks,
                    'blocks': compare_blocks(model, observations, options),
                    'current_effective_scan_block': min(
                        64, max(1, (32 * 1024**2) // (
                            model.parents.size * (20**2 + model.rates.size * model.pi.size) * 8))),
                }
                print(case, {block: item['summary']['wall_s']['median']
                             for block, item in result['cases'][case]['blocks'].items()}, flush=True)
            del outputs
            gc.collect()
        result['edge_case_parity'] = synthetic_checks()
    (HERE / 'proposal02.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
