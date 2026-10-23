"""Evaluate skipping finite sparse product rows after a zero intermediate.

Run from the repository root: python reports/performance_round3_20261023/proposal08.py
The immutable baseline snapshot is required (CSUBST_PERF_BASELINE overrides the
shared pointer). Only this proposal's four report files are written.
"""

import ast
import contextlib
import difflib
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import shutil
import tempfile
import tracemalloc

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402

HERE = Path(__file__).resolve().parent
BASELINE = Path(os.environ.get('CSUBST_PERF_BASELINE') or
                Path('/tmp/csubst-round3-baseline-path').read_text().strip())


def prototype(substitution):
    source = (BASELINE / 'csubst/substitution.py').read_text()
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == 'aggregate_sparse_branches')
    original = '\n'.join(source.splitlines()[node.lineno - 1:node.end_lineno]) + '\n'
    anchor = '            matrix = matrix.multiply(sub_tensor.matrix.getrow(int(branch_id))).tocsr()\n'
    replacement = '''            row = sub_tensor.matrix.getrow(int(branch_id))
            # Multiplying an empty row by finite values stays empty. Retain
            # getrow validation and nonfinite 0 * NaN/Inf sparse semantics.
            if matrix.nnz == 0 and np.isfinite(row.data).all():
                continue
            matrix = matrix.multiply(row).tocsr()
'''
    assert original.count(anchor) == 1
    candidate = original.replace(anchor, replacement)
    namespace = dict(substitution.__dict__)
    exec(compile(candidate, '<proposal08-product>', 'exec'), namespace)
    changed = source.replace(original, candidate)
    (HERE / 'proposal08.patch').write_text(''.join(difflib.unified_diff(
        source.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile='a/csubst/substitution.py', tofile='b/csubst/substitution.py', n=0)))
    return substitution.aggregate_sparse_branches, namespace['aggregate_sparse_branches'], source


def signature(function, args):
    try:
        return function(*args), None
    except Exception as error:
        return None, (type(error).__name__, str(error))


def exact(np, old, new, tensor, ids, operation, dense_reference=False):
    args = tensor, ids, operation
    first, first_error = signature(old, args)
    second, second_error = signature(new, args)
    assert first_error == second_error, (ids, operation, first_error, second_error)
    if first_error:
        return {'exception': first_error[0]}
    assert first.shape == second.shape and first.dtype == second.dtype
    for attribute in ('data', 'indices', 'indptr'):
        before, after = getattr(first.matrix, attribute), getattr(second.matrix, attribute)
        np.testing.assert_array_equal(before, after, strict=True)
        assert before.tobytes() == after.tobytes(), attribute
        assert not after.flags.writeable
    assert second.matrix.has_canonical_format and second.matrix.has_sorted_indices
    assert not np.any(second.matrix.data == 0)
    # Integer/bool aggregation intentionally retains the input dtype; sparse
    # empty selection returns zeros even for product or unknown operation.
    if dense_reference:
        normalized = old.__globals__['_normalize_branch_ids'](ids)
        dense = tensor.to_dense()
        if not len(normalized):
            reference = np.zeros((1,) + tensor.shape[1:], dtype=tensor.dtype)
        elif operation == 'sum':
            reference = dense[normalized].sum(axis=0, dtype=tensor.dtype, keepdims=True)
        else:
            reference = dense[normalized].prod(axis=0, dtype=tensor.dtype, keepdims=True)
        np.testing.assert_array_equal(first.to_dense(), reference, strict=True)
    return {'exact': True, 'dense_reference': dense_reference}


def parity(np, sp, sparse, old, new):
    rng = np.random.default_rng(82623)
    records = []
    for dtype in (np.bool_, np.int8, np.int32, np.uint64, np.float32, np.float64, np.complex128):
        dense = rng.integers(0, 4, size=(8, 7, 1, 2, 2)).astype(dtype)
        dense[rng.random(dense.shape) < 0.9] = 0
        dense[0] = 0  # Empty running product from the first row.
        dense[1, 1, 0, 0, 1] = 1
        dense[2, 2, 0, 0, 1] = 2  # Empty after multiplying disjoint support.
        tensor = sparse.SparseSubstitutionTensor.from_dense(dense)
        snapshots = [getattr(tensor.matrix, attribute).tobytes()
                     for attribute in ('data', 'indices', 'indptr')]
        for ids in (None, [], [0], [1], [1, 2], [1, 2, 3, 4], list(range(8)),
                    [7, 2, 1, 1, 0, 6], np.array([[1, 2], [3, 4]]), ['1.0', '+2', '3']):
            for operation in ('sum', 'product'):
                records.append(exact(np, old, new, tensor, ids, operation, dense_reference=True))
        for ids in ([], [1], [0, 2, 8], [8], [-1], [-8], [-9], [0, 2, -1],
                    [True], [1.5], [np.nan], ['bad'], [10 ** 40]):
            for operation in ('sum', 'product', 'unsupported'):
                records.append(exact(np, old, new, tensor, ids, operation))
        assert snapshots == [getattr(tensor.matrix, attribute).tobytes()
                             for attribute in ('data', 'indices', 'indptr')]
        assert all(not getattr(tensor.matrix, attribute).flags.writeable
                   for attribute in ('data', 'indices', 'indptr'))
    for dtype in (np.float32, np.float64, np.complex128):
        dense = np.zeros((8, 5, 1, 2, 2), dtype=dtype)
        dense[1, 0, 0, 0, 1] = 2
        dense[2, 1, 0, 0, 1] = 4
        dense[3, 0, 0, 0, 1] = np.nan
        dense[4, 1, 0, 0, 1] = np.inf
        dense[5, 2, 0, 0, 1] = -np.inf
        dense[6, 3, 0, 0, 1] = -1
        dense[7, 4, 0, 0, 1] = -0.0
        tensor = sparse.SparseSubstitutionTensor.from_dense(dense)
        for ids in ([0, 3], [0, 4], [0, 5], [1, 2, 3], [1, 2, 4],
                    [1, 2, 3, 0, 4, 5, 6], [0, 6, 3], [5, 4, 0], [3, 3, 0]):
            for operation in ('sum', 'product'):
                # Sparse 0 * nonfinite semantics are the existing reference;
                # signed zero payloads are removed by tensor construction.
                records.append(exact(np, old, new, tensor, ids, operation))
                if operation == 'product':
                    reference = scalar_product_reference(np, tensor, ids)
                    np.testing.assert_array_equal(old(tensor, ids, operation).matrix.toarray()[0], reference, strict=True)
    # Independent dense references for finite cancellation, dtype overflow,
    # repeated IDs (not deduplicated), exact zero elimination, all-empty CSR.
    dense = np.zeros((5, 3, 1, 2, 2), dtype=np.float64)
    dense[:, 1, 0, 0, 1] = [2 ** 30, 0.5, -(2 ** 30), -0.5, 0]
    tensor = sparse.SparseSubstitutionTensor.from_dense(dense)
    for ids in ([0, 1, 2, 3], [1, 1, 1], [4, 4, 4], []):
        for operation in ('sum', 'product'):
            records.append(exact(np, old, new, tensor, ids, operation, dense_reference=True))
    # Rectangular event axes and zero site/state dimensions retain their shape.
    for shape in ((0, 3, 2, 3, 4), (4, 0, 2, 3, 4), (4, 3, 0, 2, 2), (4, 3, 2, 3, 4)):
        tensor = sparse.SparseSubstitutionTensor(shape, np.float64,
                  matrix=sp.csr_matrix((shape[0], int(np.prod(shape[1:])))))
        for ids in ([], list(range(shape[0])), [shape[0]]):
            for operation in ('sum', 'product'):
                records.append(exact(np, old, new, tensor, ids, operation))
    for operation in ('sum', 'product'):
        records.append(exact(np, old, new, np.zeros((2, 3, 1, 2, 2)), [0], operation))
    return {'cases': len(records), 'exact_outputs': sum('exact' in r for r in records),
            'dense_references': sum(r.get('dense_reference', False) for r in records),
            'independent_nonfinite_sparse_product_references': 27,
            'equal_exceptions': sum('exception' in r for r in records),
            'bitwise_csr_payload_equal': True, 'input_storage_unchanged_and_readonly': True}



def scalar_product_reference(np, tensor, ids):
    # Independent sparse support-union reference. Missing entries are zero;
    # explicit NaN/Inf rows must still interact with missing zero entries.
    result = None
    for branch in ids:
        begin, end = tensor.matrix.indptr[branch:branch + 2]
        row = dict(zip(tensor.matrix.indices[begin:end].tolist(),
                       tensor.matrix.data[begin:end]))
        if result is None:
            result = row
            continue
        output = {}
        for column in result.keys() | row.keys():
            value = np.multiply(result.get(column, tensor.dtype.type(0)),
                                row.get(column, tensor.dtype.type(0)), dtype=tensor.dtype)
            if value != 0:
                output[column] = value
        result = output
    out = np.zeros(tensor.matrix.shape[1], dtype=tensor.dtype)
    for column, value in (result or {}).items():
        out[column] = value
    return out


class Captured(Exception):
    pass


def real_case(np, old, new, dataset):
    from csubst import cli, main_sites
    original = main_sites.plot_state
    result = {}
    with tempfile.TemporaryDirectory(prefix='csubst-proposal08-') as directory:
        work = Path(directory)
        inputs = work / 'inputs'
        inputs.mkdir()
        filenames = [dataset + suffix for suffix in ('.alignment.fa', '.tree.nwk')]
        filenames += [dataset + '.alignment.fa.' + suffix
                      for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log')]
        hashes = {}
        for name in filenames:
            shutil.copyfile(BASELINE / 'csubst/dataset' / name, inputs / name)
            hashes[name] = hashlib.sha256((inputs / name).read_bytes()).hexdigest()
        def capture(ON_tensor, OS_tensor, branch_ids, g):
            assert len(branch_ids) == 16
            for kind, tensor in (('N', ON_tensor), ('S', OS_tensor)):
                assert hasattr(tensor, 'matrix'), type(tensor)
                for label, ids in (('pair', branch_ids[:2]), ('sixteen', branch_ids)):
                    entry = timed(np, old, new, tensor, ids, 'product')
                    running = tensor.matrix.getrow(int(ids[0]))
                    counts = [int(running.nnz)]
                    for branch in ids[1:]:
                        running = running.multiply(tensor.matrix.getrow(int(branch))).tocsr()
                        counts.append(int(running.nnz))
                    entry['running_product_nnz'] = counts
                    result[kind + '_' + label] = entry
            raise Captured
        argv = ['sites', '--alignment_file', str(inputs / (dataset + '.alignment.fa')),
                '--rooted_tree_file', str(inputs / (dataset + '.tree.nwk')),
                '--branch_id', ','.join(map(str, range(1, 17))), '--mode', 'intersection',
                '--outdir', str(work / 'output'), '--threads', '1', '--blas_threads', '1',
                '--tree_site_plot', 'no', '--site_summary_plot', 'no', '--site_state_plot', 'yes',
                '--random_seed', '8', '--float_digit', '12']
        for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log'):
            argv += ['--iqtree_' + suffix, str(inputs / (dataset + '.alignment.fa.' + suffix))]
        main_sites.plot_state = capture
        try:
            with (work / 'capture.log').open('w') as log, contextlib.redirect_stdout(log):
                try:
                    cli.main(argv)
                except Captured:
                    pass
                except Exception:
                    print((work / 'capture.log').read_text(), file=sys.stderr)
                    raise
        finally:
            main_sites.plot_state = original
        assert len(result) == 4
        return {'benchmarks': result, 'input_sha256': hashes,
                'scope': 'Actual observed PGK/PEPC endpoint substitution tensors captured at intersection plot_state; only aggregate product timed; pair and explicit 16 branch selection 1..16; not full CLI timing'}


def make_input(np, sp, sparse, branches, sites, per_row, dtype=None, common=0):
    dtype = np.float64 if dtype is None else dtype
    rng = np.random.default_rng(82323 + sites + branches)
    width = sites * 400
    rows, columns, values = [], [], []
    for branch in range(branches):
        selected = rng.choice(width, size=per_row, replace=False)
        selected = np.union1d(selected, np.arange(common))
        rows.append(np.full(len(selected), branch, dtype=np.int32))
        columns.append(selected)
        values.append(rng.random(len(selected)).astype(dtype))
    matrix = sp.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                           shape=(branches, width), dtype=dtype)
    return sparse.SparseSubstitutionTensor((branches, sites, 1, 20, 20), dtype, matrix=matrix)


def timed(np, old, new, tensor, ids, operation):
    parity_result = exact(np, old, new, tensor, ids, operation)
    _, warmup_old = measure(old, tensor, ids, operation)
    _, warmup_new = measure(new, tensor, ids, operation)
    batch_count = max(1, min(100, int(0.04 / max(warmup_old['cpu_s'], warmup_new['cpu_s']))))
    def batch(function):
        for _ in range(batch_count):
            function(tensor, ids, operation)
    samples = {'baseline': [], 'candidate': []}
    for repeat in range(7):
        for name, function in ([('baseline', old), ('candidate', new)] if repeat % 2 == 0
                               else [('candidate', new), ('baseline', old)]):
            _, timing = measure(batch, function)
            samples[name].append({key: value / batch_count for key, value in timing.items()})
    memory = {}
    for name, function in (('baseline', old), ('candidate', new)):
        gc.collect()
        tracemalloc.start()
        value = function(tensor, ids, operation)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory[name] = {'peak_traced_bytes': peak, 'returned_traced_bytes': current}
        del value
    summary = {name: summarize(records) for name, records in samples.items()}
    return {'shape': list(tensor.shape), 'nnz': tensor.nnz, 'dtype': str(tensor.dtype),
            'selected_branches': list(map(int, ids)), 'operation': operation,
            'parity': parity_result, 'calls_per_repeat': batch_count,
            'samples': samples, 'summary': summary, 'memory': memory,
            'candidate_to_baseline_wall': summary['candidate']['wall_s']['median'] / summary['baseline']['wall_s']['median'],
            'candidate_to_baseline_cpu': summary['candidate']['cpu_s']['median'] / summary['baseline']['cpu_s']['median']}


def main():
    with benchmark_lock():
        sys.path.insert(0, str(BASELINE))
        import numpy as np
        import scipy.sparse as sp
        from csubst import substitution, substitution_sparse
        assert Path(substitution.__file__).parent.parent.resolve() == BASELINE.resolve()
        old, new, source = prototype(substitution)
        with np.errstate(invalid='ignore', over='ignore'):
            exactness = parity(np, sp, substitution_sparse, old, new)
        result = {'proposal': 8, 'baseline': '624bb8a', 'environment': environment(),
                  'decision': 'rejected: shortcut never fires on captured real-data intersections; synthetic-only gain and control regressions',
                  'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                  'exactness': exactness, 'benchmarks': {},
                  'real_data': {},
                  'method': 'All import/preparation/parity and measurement holds shared lock; native threads=1 before imports; one warmup per implementation; seven alternating repeats; each repeat 1-100 calls toward 40 ms CPU; per-call samples; memory tracemalloc excludes input/native allocator and is not peak RSS'}
        for label, branches, sites, per_row, common, dtype in (
                ('small_pair', 8, 4, 120, 0, np.float64),
                ('small_many', 16, 4, 120, 0, np.float64),
                ('wide_pair', 16, 5000, 5000, 0, np.float64),
                ('wide_many', 16, 5000, 5000, 0, np.float64),
                ('wide_common_support', 16, 5000, 5000, 20, np.float64),
                ('very_wide_many_f32', 32, 25000, 5000, 0, np.float32)):
            tensor = make_input(np, sp, substitution_sparse, branches, sites, per_row, dtype, common)
            ids = list(range(2 if label.endswith('pair') else branches))
            result['benchmarks'][label] = timed(np, old, new, tensor, ids, 'product')
            if label in ('small_pair', 'wide_many'):
                result['benchmarks'][label + '_sum_control'] = timed(np, old, new, tensor, ids, 'sum')
            del tensor
        for dataset in ('PGK', 'PEPC'):
            result['real_data'][dataset] = real_case(np, old, new, dataset)
        (HERE / 'proposal08.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'exactness': exactness, 'real_data': {dataset: {key: {'cpu_ratio': entry['candidate_to_baseline_cpu'], 'wall_ratio': entry['candidate_to_baseline_wall'], 'running_nnz': entry['running_product_nnz']} for key, entry in item['benchmarks'].items()} for dataset, item in result['real_data'].items()}, 'benchmarks': {
            key: {'wall_ratio': value['candidate_to_baseline_wall'],
                  'cpu_ratio': value['candidate_to_baseline_cpu'],
                  'baseline_wall_median': value['summary']['baseline']['wall_s']['median'],
                  'candidate_wall_median': value['summary']['candidate']['wall_s']['median']}
            for key, value in result['benchmarks'].items()}}, indent=2))


if __name__ == '__main__':
    main()
