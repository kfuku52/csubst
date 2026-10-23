"""Measure bounded conservation-site slicing against baseline 624bb8a.

Run from the repository root with python reports/performance_round3_20261023/proposal07.py.
Only this proposal's report files are written. All workloads hold the shared lock.
"""

import ast
import difflib
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import tracemalloc

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASELINE = os.environ.get('CSUBST_PERF_BASELINE')
if not BASELINE:
    BASELINE = Path('/tmp/csubst-round3-baseline-path').read_text().strip()
BLOCK_MIB = int(os.environ.get('CSUBST_CONSERVATION_BLOCK_MIB', '16'))


def prototype(np):
    source = (Path(BASELINE) / 'csubst/substitution_scan.py').read_text()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == '_calc_conservation')
    original = '\n'.join(source.splitlines()[node.lineno - 1:node.end_lineno]) + '\n'
    anchor = '    block = state[tip_ids, :, :]\n'
    insertion = '''    # Bound the advanced-indexing copy while retaining tip summation order.
    # One selected site is the minimum block, even if it exceeds the target.
    max_block_bytes = BLOCK_TARGET_MIB * 1024 * 1024
    bytes_per_site = tip_ids.shape[0] * state.shape[2] * state.dtype.itemsize
    sites_per_block = max(1, max_block_bytes // max(bytes_per_site, 1))
    if num_site > sites_per_block:
        for start in range(0, num_site, sites_per_block):
            stop = min(start + sites_per_block, num_site)
            values = _calc_conservation(state[:, start:stop, :], orders, tip_ids, float_tol)
            for key in prefix:
                prefix[key][start:stop] = values[key]
        return prefix
'''.replace('BLOCK_TARGET_MIB', str(BLOCK_MIB))
    assert original.count(anchor) == 1
    candidate = original.replace(anchor, insertion + anchor)
    changed = source.replace(original, candidate)
    assert changed != source
    (HERE / 'proposal07.patch').write_text(''.join(difflib.unified_diff(
        source.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile='a/csubst/substitution_scan.py', tofile='b/csubst/substitution_scan.py', n=0)))
    old_namespace = {'np': np, 'math': math}
    new_namespace = {'np': np, 'math': math}
    exec(compile(original, '<proposal07-baseline>', 'exec'), old_namespace)
    exec(compile(candidate, '<proposal07-candidate>', 'exec'), new_namespace)
    return old_namespace['_calc_conservation'], new_namespace['_calc_conservation']


def equality(np, old, new, args):
    try:
        first = old(*args)
    except Exception as old_error:
        try:
            new(*args)
        except Exception as new_error:
            assert type(old_error) is type(new_error)
            return {'equal_exception': type(old_error).__name__}
        raise AssertionError('candidate did not raise') from old_error
    second = new(*args)
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key], strict=True)
        if first[key].dtype != object:
            assert first[key].tobytes() == second[key].tobytes(), key
    return {'arrays_exact': len(first), 'sites': len(first['entropy'])}


def cases(np, old, new):
    rng = np.random.default_rng(70623)
    results = []
    for dtype in (np.float32, np.float64):
        for sites in (0, 1, 17, 2000):
            # 180 tips x 2000 sites crosses the block bound for both dtypes.
            state = rng.random((180, sites, 20), dtype=dtype)
            if sites:
                state[::3, ::4] = 0
                state[:, ::7] = 0
                state[0, ::5] = 1 / 20  # Explicit ambiguity.
                state[2, ::9] = state[1, ::9]  # Tie-like repeated observations.
            for ids in ([], [0], list(range(180)), [179, 0, 4, 4, 1, 0] * 31):
                for tol in (1e-12, 0.2, -1, float('nan')):
                    args = (state, np.array(list('ACDEFGHIKLMNPQRSTVWY'), object), ids, tol)
                    results.append(equality(np, old, new, args))
            if sites:
                state[1, 0, 1] = np.nan
                state[3, 0, 4] = np.inf
                for view in (state[:, ::-1], state[::-1, :, ::-1]):
                    results.append(equality(np, old, new, (view, range(20), range(180), 1e-12)))
                state.flags.writeable = False
                results.append(equality(np, old, new, (state, range(20), range(180), 1e-12)))
        del state
    # Missing-only rows, exact major-state ties, tolerance boundaries, integer
    # input, invalid tip indexes, empty states, and wrong order cardinality.
    tied = np.tile(np.array([0.5, 0.5, 0., 0.], dtype=np.float64), (200, 3000, 1))
    tied[0, ::2] = 0
    for state, orders, ids, tol in (
        (tied, ['first', 'second', 'third', 'fourth'], range(200), 0.5),
        (tied, ['first', 'second', 'third', 'fourth'], range(200), 1),
        (tied, ['first', 'second', 'third', 'fourth'], [-1, 0] * 100, 0),
        (tied.astype(np.int64), range(4), range(200), 0),
        (np.zeros((2, 0, 4)), range(4), [2], 1e-12),
        (np.zeros((2, 0, 0)), [], [0, 1], 1e-12),
        (np.zeros((2, 3, 0)), [], [0, 1], 1e-12),
        (tied, [], range(200), 1e-12),
        (tied, range(4), [200], 1e-12),
    ):
        results.append(equality(np, old, new, (state, orders, ids, tol)))
    del tied
    with tempfile.TemporaryDirectory(prefix='csubst-proposal07-') as directory:
        path = Path(directory) / 'state.npy'
        mapped = np.lib.format.open_memmap(path, mode='w+', dtype='float64', shape=(180, 1000, 20))
        mapped[:] = rng.random(mapped.shape)
        mapped.flush()
        del mapped
        mapped = np.load(path, mmap_mode='r')
        results.append(equality(np, old, new, (mapped, range(20), range(180), 1e-12)))
    return {'cases': len(results), 'exact_arrays': sum(r.get('arrays_exact', 0) for r in results),
            'equal_exception_cases': [r for r in results if 'equal_exception' in r],
            'float_bitwise_exact': True}


def read_alignment(np, name, amino):
    # Deterministic observed-tip encoding from real bundled alignment symbols.
    code_source = (Path(BASELINE) / 'csubst/genetic_code.py').read_text()
    namespace = {}
    exec(compile(code_source, '<genetic-code>', 'exec'), namespace)
    table = namespace['get_codon_table'](1)
    mapping = {codon: aa for aa, codon in table if aa != '*'}
    orders = sorted(set(mapping.values())) if amino else sorted(mapping)
    index = {value: i for i, value in enumerate(orders)}
    sequences = []
    for line in (ROOT / 'csubst/dataset' / (name + '.alignment.fa')).read_text().splitlines():
        if line.startswith('>'):
            sequences.append('')
        elif line.strip():
            sequences[-1] += line.strip().upper()
    assert len({len(seq) for seq in sequences}) == 1
    sites = len(sequences[0]) // 3
    state = np.zeros((len(sequences), sites, len(orders)), dtype=np.float64)
    for row, seq in enumerate(sequences):
        for site in range(sites):
            codon = seq[site * 3:site * 3 + 3]
            if codon in mapping:
                symbol = mapping[codon] if amino else codon
                state[row, site, index[symbol]] = 1
    return state, orders, np.arange(len(sequences)), 1e-12


def timed_case(np, old, new, args, repeats=7):
    parity = equality(np, old, new, args)
    _, first_warmup = measure(old, *args)
    _, second_warmup = measure(new, *args)
    batch_size = max(1, min(50, int(0.05 / max(first_warmup['cpu_s'], second_warmup['cpu_s']))))

    def batch(function):
        for _ in range(batch_size):
            function(*args)

    samples = {'baseline': [], 'candidate': []}
    for repeat in range(repeats):
        for name, function in ([('baseline', old), ('candidate', new)]
                               if repeat % 2 == 0 else [('candidate', new), ('baseline', old)]):
            _, timing = measure(batch, function)
            samples[name].append({key: value / batch_size for key, value in timing.items()})
    memory = {}
    for name, function in (('baseline', old), ('candidate', new)):
        gc.collect()
        tracemalloc.start()
        value = function(*args)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory[name] = {'peak_traced_bytes': peak, 'returned_traced_bytes': current}
        del value
    summary = {name: summarize(records) for name, records in samples.items()}
    ratio = summary['candidate']['wall_s']['median'] / summary['baseline']['wall_s']['median']
    return {'shape': list(args[0].shape), 'dtype': str(args[0].dtype),
            'calls_per_repeat': batch_size,
            'parity': parity, 'samples': samples, 'summary': summary,
            'candidate_to_baseline_wall': ratio, 'memory': memory,
            'memory_method': 'tracemalloc includes NumPy data allocations; excludes preexisting input, imported modules and native allocator/RSS overhead'}


def main():
    with benchmark_lock():
        import numpy as np
        old, new = prototype(np)
        with np.errstate(invalid='ignore'):
            parity = cases(np, old, new)
        result = {'proposal': 7, 'baseline': '624bb8a', 'environment': environment(),
                  'exactness': parity, 'block_bytes': BLOCK_MIB * 1024 * 1024,
                  'method': 'One warmup per implementation; seven alternating ordered repeats batched toward 50 ms CPU per implementation (1–50 calls); samples normalized per call; inputs/preparation outside timed section; all operations hold shared benchmark lock',
                  'benchmarks': {}}
        for dataset in ('PGK', 'PEPC'):
            for amino in (True, False):
                args = read_alignment(np, dataset, amino)
                key = dataset + ('_amino_observed' if amino else '_codon_observed')
                result['benchmarks'][key] = timed_case(np, old, new, args)
                del args
        rng = np.random.default_rng(70623)
        for dtype in (np.float32, np.float64):
            state = rng.random((192, 8192, 20), dtype=dtype)
            state[::6, ::9] = 0
            args = state, list('ACDEFGHIKLMNPQRSTVWY'), np.arange(192), 1e-12
            result['benchmarks']['large_' + np.dtype(dtype).name] = timed_case(np, old, new, args)
            del state, args
            gc.collect()
        result['source_sha256'] = hashlib.sha256((Path(BASELINE) / 'csubst/substitution_scan.py').read_bytes()).hexdigest()
        (HERE / 'proposal07.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'exactness': parity, 'benchmarks': {
            key: {'wall_ratio': value['candidate_to_baseline_wall'],
                  'baseline_peak_bytes': value['memory']['baseline']['peak_traced_bytes'],
                  'candidate_peak_bytes': value['memory']['candidate']['peak_traced_bytes']}
            for key, value in result['benchmarks'].items()}}, indent=2))


if __name__ == '__main__':
    main()
