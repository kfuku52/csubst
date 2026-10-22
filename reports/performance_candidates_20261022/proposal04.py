"""Evaluate bounded-lifetime mmap reads for endpoint CSR spool finalization.

Run from the repository root with ``python reports/performance_candidates_20261022/proposal04.py``.
Only investigation code; production modules are never edited.
"""

from contextlib import ExitStack, redirect_stdout
import gc
import hashlib
import json
import mmap
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import cli, endpoint_io  # noqa: E402

ORIGINAL = endpoint_io._Spool


def finish_mmap(spool):
    """Map scratch payloads only while scattering them into final CSR arrays."""
    indptr = np.r_[0, np.cumsum(spool.counts)]
    index = np.empty(spool.offset, dtype=spool.index_dtype)
    data = np.empty(spool.offset, dtype=spool.dtype)
    cursor = indptr[:-1].copy()
    if spool.offset:
        spool.indices.flush()
        spool.values.flush()
        with mmap.mmap(spool.indices.fileno(), 0, access=mmap.ACCESS_READ) as imap:
            with mmap.mmap(spool.values.fileno(), 0, access=mmap.ACCESS_READ) as vmap:
                source_index = np.ndarray((spool.offset,), dtype=spool.index_dtype, buffer=imap)
                source_data = np.ndarray((spool.offset,), dtype=spool.dtype, buffer=vmap)
                try:
                    for row, offset, count in spool.fragments:
                        sl = slice(cursor[row], cursor[row] + count)
                        index[sl] = source_index[offset:offset + count]
                        data[sl] = source_data[offset:offset + count]
                        cursor[row] += count
                finally:
                    del source_index, source_data
    result = sparse.csr_matrix((data, index, indptr),
                               shape=(spool.rows, spool.columns), copy=False)
    result.sort_indices()
    return result


def equal(before, after):
    assert before.shape == after.shape
    assert before.dtype == after.dtype
    for field in ('indices', 'indptr', 'data'):
        np.testing.assert_array_equal(getattr(before, field), getattr(after, field))
    assert before.has_sorted_indices == after.has_sorted_indices


def paired(function_before, function_after):
    samples = {'before': [], 'mmap': []}
    digest = None
    for repeat in range(4):
        out = {}
        order = ('before', 'mmap') if repeat % 2 == 0 else ('mmap', 'before')
        for label in order:
            gc.collect()
            function = function_before if label == 'before' else function_after
            out[label], timing = measure(function)
            if repeat:
                samples[label].append(timing)
        equal(out['before'], out['mmap'])
        digest = hashlib.sha256(out['before'].data.tobytes() +
                                out['before'].indices.tobytes() +
                                out['before'].indptr.tobytes()).hexdigest()
    return {'runs': samples, 'summary': {k: summarize(v) for k, v in samples.items()},
            'exact_csr_equal': True, 'csr_sha256': digest}


def build(rows, columns, dtype, calls, use_mmap):
    with ExitStack() as stack:
        spool = ORIGINAL(stack, rows, columns, dtype)
        for row, block, start, sites in calls:
            spool.append(row, block, start, sites)
        return finish_mmap(spool) if use_mmap else spool.finish()


class CaptureComplete(Exception):
    pass


def real_dataset(dataset, workdir):
    """Measure actual spools inside a fixed-fit higher-arity preparation."""
    results = []

    class RecordingSpool(ORIGINAL):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = []

        def append(self, row, block, start, sites):
            self.calls.append((row, np.array(block, copy=True), start, sites))
            return super().append(row, block, start, sites)

        def finish(self):
            finish = paired(lambda: ORIGINAL.finish(self), lambda: finish_mmap(self))
            total = paired(lambda: build(self.rows, self.columns, self.dtype, self.calls, False),
                           lambda: build(self.rows, self.columns, self.dtype, self.calls, True))
            result = ORIGINAL.finish(self)
            results.append({'rows': self.rows, 'columns': self.columns,
                            'index_dtype': str(self.index_dtype), 'dtype': str(self.dtype),
                            'fragments': len(self.fragments), 'nnz': self.offset,
                            'payload_bytes': self.offset * (self.dtype.itemsize + self.index_dtype.itemsize),
                            'max_fragment_bytes': max((c for _, _, c in self.fragments), default=0) *
                            (self.dtype.itemsize + self.index_dtype.itemsize),
                            'finish': finish, 'append_and_finish': total})
            return result

    original_build = endpoint_io._build

    def stop_after_build(g, structural=False):
        original_build(g, structural)
        raise CaptureComplete

    data = ROOT / 'csubst/dataset'
    argv = ['search', '--alignment_file', str(data / (dataset + '.alignment.fa')),
            '--rooted_tree_file', str(data / (dataset + '.tree.nwk')),
            '--outdir', str(workdir / dataset), '--threads', '1', '--blas_threads', '1',
            '--max_arity', '3', '--calibrate_longtail', 'no', '--random_seed', '8',
            '--drop_invariant_tip_sites', 'no', '--b', 'no', '--s', 'no', '--cs', 'no',
            '--bs', 'no', '--cbs', 'no']
    for key in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        argv += ['--iqtree_' + key, str(data / (dataset + '.alignment.fa.' + key))]
    endpoint_io._Spool = RecordingSpool
    endpoint_io._build = stop_after_build
    try:
        with (workdir / (dataset + '.log')).open('w') as log, redirect_stdout(log):
            try:
                cli.main(argv)
            except CaptureComplete:
                pass
    finally:
        endpoint_io._Spool = ORIGINAL
        endpoint_io._build = original_build
    assert results, 'The selected real workflow must actually use spools.'
    return results


def edge_checks():
    cases = []
    for dtype in (np.float32, np.float64):
        for columns in (15, 2**31 + 15):
            with ExitStack() as stack:
                spool = ORIGINAL(stack, 2, columns, dtype)
                zero = finish_mmap(spool)
                equal(spool.finish(), zero)
                block = np.arange(30., dtype=dtype).reshape(2, 5, 3)
                block.flags.writeable = False
                for start, stop in ((0, 2), (2, 5)):
                    for branch in (1, 0):
                        spool.append(branch, block[branch, start:stop], start, 5)
                expected = spool.finish()
                actual = finish_mmap(spool)
                equal(expected, actual)
                # Both finish methods are repeatable; returned CSR owns its payload.
                equal(actual, finish_mmap(spool))
            equal(expected, actual)
            cases.append({'dtype': np.dtype(dtype).name, 'columns': columns,
                          'index_dtype': str(actual.indices.dtype), 'exact_equal': True,
                          'csr_survives_closed_spool': True})
    return cases


def rss_worker(label):
    rows, sites, features, block_size = 256, 8192, 20, 64
    rng = np.random.default_rng(20261022)
    block = rng.random((block_size, features))
    block[block < .9] = 0
    with ExitStack() as stack:
        spool = ORIGINAL(stack, rows, sites * features, np.float64)
        for start in range(0, sites, block_size):
            for row in range(rows):
                spool.append(row, block, start, sites)
        result, timing = measure(finish_mmap if label == 'mmap' else ORIGINAL.finish, spool)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Hashing allocates a byte copy; exclude that verification cost from RSS.
        digest = hashlib.sha256(result.data.tobytes()).hexdigest()
        print(json.dumps({'label': label, 'peak_rss_mib': peak / (2**20 if sys.platform == 'darwin' else 1024),
                          'shape': [rows, sites * features], 'nnz': result.nnz,
                          'payload_bytes': spool.offset * 12, 'finish': timing, 'sha256': digest}))


def rss_measurements():
    samples = []
    for repeat in range(3):
        for label in (('before', 'mmap') if repeat % 2 == 0 else ('mmap', 'before')):
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--rss-worker', label],
                                  cwd=ROOT, capture_output=True, text=True, check=True)
            samples.append({'repeat': repeat, **json.loads(proc.stdout)})
    assert len({x['sha256'] for x in samples}) == 1
    return samples


def aggregate(spools):
    result = {'spools': len(spools), 'fragments': sum(x['fragments'] for x in spools),
              'nnz': sum(x['nnz'] for x in spools),
              'payload_bytes': sum(x['payload_bytes'] for x in spools),
              'scope': 'Per-replicate sum of eight separately timed spool operations'}
    for scope in ('finish', 'append_and_finish'):
        samples = {label: [{metric: sum(x[scope]['runs'][label][i][metric] for x in spools)
                            for metric in ('wall_s', 'cpu_s')} for i in range(3)]
                   for label in ('before', 'mmap')}
        result[scope] = {'runs': samples, 'summary': {k: summarize(v) for k, v in samples.items()}}
    return result


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--rss-worker':
        rss_worker(sys.argv[2])
        return
    destination = Path(__file__).with_suffix('.json')
    if sys.argv[1:] == ['--rss-only']:
        result = json.loads(destination.read_text())
        with benchmark_lock():
            result['synthetic_rss'] = rss_measurements()
        result['real_aggregate'] = {k: aggregate(v) for k, v in result['real'].items()}
        destination.write_text(json.dumps(result, indent=2) + '\n')
        return
    result = {'environment': environment(), 'real': {}, 'input_sha256': {},
              'scope': 'Actual higher-arity spool finish and append+finish, not full CLI',
              'rss_scope': 'Fresh subprocess synthetic spool; macOS process peak RSS, not total machine RAM'}
    with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-spool-proposal04-') as work:
        result['edge_checks'] = edge_checks()
        for dataset in ('PGK', 'PEPC'):
            result['real'][dataset] = real_dataset(dataset, Path(work))
            print(dataset, 'spools', len(result['real'][dataset]), flush=True)
            for extension in ('alignment.fa', 'tree.nwk', 'alignment.fa.iqtree', 'alignment.fa.state'):
                path = ROOT / 'csubst/dataset' / (dataset + '.' + extension)
                result['input_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        result['synthetic_rss'] = rss_measurements()
        result['real_aggregate'] = {k: aggregate(v) for k, v in result['real'].items()}
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print('saved', destination.name)


if __name__ == '__main__':
    main()
