"""Compare mmap-backed site-cache fills while preserving public read copies.

Run from repository root: python reports/performance_candidates_20261022/proposal08.py
"""

import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import shutil
import subprocess
import sys
import tempfile

for variable in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import cli, site_storage, substitution_scan  # noqa: E402

ORIGINAL_READ = site_storage.SiteArray.read_site
ORIGINAL_STATE = site_storage.SiteArray.__getstate__


def mapped_read_site(self, site):
    site = int(site)
    if site == self._cached_site:
        return self._cached_values
    if not self._sealed:
        raise ValueError('Cannot read an unfinished site array.')
    if not 0 <= site < self.shape[self.site_axis]:
        raise IndexError('Site range is outside the stored array.')
    first = site * self.site_size
    last = first + self.site_size
    file_size = os.path.getsize(self.filename)
    if last * self.dtype.itemsize > file_size:
        raise ValueError('Truncated site-array payload.')
    if self.site_size:
        mapped = getattr(self, '_mapped_values', None)
        if mapped is None or len(mapped) < last:
            mapped = np.memmap(self.filename, dtype=self.dtype, mode='r',
                               shape=(file_size // self.dtype.itemsize,))
            self._mapped_values = mapped
        # The copy is intentional: a returned cache entry must retain its
        # snapshot even if the scratch file later changes or is truncated.
        values = np.array(mapped[first:last], copy=True).reshape(self.site_shape)
    else:
        values = np.empty(self.site_shape, dtype=self.dtype)
    threshold = getattr(self, 'min_sub_pp', 0)
    if threshold:
        values[values < threshold] = 0
    values.flags.writeable = False
    self._cached_values, self._cached_site = values, site
    return values


def mapped_getstate(self):
    state = ORIGINAL_STATE(self)
    state.pop('_mapped_values', None)
    return state


@contextlib.contextmanager
def patched(mapped):
    site_storage.SiteArray.read_site = mapped_read_site if mapped else ORIGINAL_READ
    site_storage.SiteArray.__getstate__ = mapped_getstate if mapped else ORIGINAL_STATE
    try:
        yield
    finally:
        site_storage.SiteArray.read_site = ORIGINAL_READ
        site_storage.SiteArray.__getstate__ = ORIGINAL_STATE


def reset(stores):
    for stored in stores:
        stored._cached_site = None
        stored._cached_values = None
        stored.__dict__.pop('_mapped_values', None)


def write_store(shape, dtype=float):
    stored = site_storage.SiteEventTensor(shape, dtype=dtype)
    rng = np.random.default_rng(81)
    for start in range(0, shape[1], 32):
        values = rng.random((min(32, shape[1] - start),) + stored.site_shape).astype(dtype)
        diagonal = np.arange(shape[-1])
        values[:, :, 0, diagonal, diagonal] = 0
        stored.write_block(start, values)
    stored.seal()
    return stored


def checks():
    comparisons = 0
    for dtype in (np.float32, np.float64):
        stored = write_store((5, 9, 1, 4, 4), dtype)
        for threshold in (0, .3, .9):
            tensor = stored.thresholded(threshold)
            for site in (0, 8, 3, 3, 2, 0):
                reset([tensor])
                expected = ORIGINAL_READ(tensor, site).copy()
                reset([tensor])
                with patched(True):
                    actual = tensor.read_site(site)
                    np.testing.assert_array_equal(actual, expected)
                    assert not actual.flags.writeable
                    assert tensor.read_site(site) is actual
                    state = tensor.__getstate__()
                    assert '_mapped_values' not in state
                    payload = pickle.dumps(tensor)
                    assert len(payload) < tensor.nbytes
                    borrowed = pickle.loads(payload)
                    np.testing.assert_array_equal(borrowed.read_site(site), expected)
                    block = tensor.read_sites(site, site + 1)
                    assert block.flags.writeable
                    block[:] = -1
                    np.testing.assert_array_equal(tensor.read_site(site), expected)
                comparisons += 1
    # Cached snapshots and worker payloads must not alias the mapped file.
    stored = site_storage.SiteArray((2, 3), float, 0)
    stored.write_block(0, np.arange(6, dtype=float).reshape(2, 3))
    stored.seal()
    with patched(True):
        old = stored.read_site(0)
        with open(stored.filename, 'r+b') as handle:
            np.full(3, 9., dtype=float).tofile(handle)
        np.testing.assert_array_equal(old, np.arange(3, dtype=float))
        np.testing.assert_array_equal(stored.read_site(1), np.arange(3, 6, dtype=float))
        with open(stored.filename, 'r+b') as handle:
            handle.truncate(8)
        np.testing.assert_array_equal(old, np.arange(3, dtype=float))
        try:
            stored.read_site(0)
        except ValueError as error:
            assert 'Truncated' in str(error)
        else:
            raise AssertionError('truncated file accepted')
    partial = site_storage.SiteArray((2, 3), float, 0)
    with patched(True):
        try:
            partial.read_site(0)
        except ValueError as error:
            assert 'unfinished' in str(error)
        else:
            raise AssertionError('unfinished file accepted')
    # A raw zero-copy mapped view demonstrably violates old snapshot semantics.
    with tempfile.TemporaryFile() as handle:
        np.arange(6, dtype=float).tofile(handle)
        view = np.memmap(handle, dtype=float, mode='r', shape=(6,))[:3]
        before = view.copy()
        handle.seek(0)
        np.ones(3).tofile(handle)
        handle.flush()
        zero_copy_snapshot_changed = not np.array_equal(before, view)
        assert zero_copy_snapshot_changed
    return {'exact_site_comparisons': comparisons, 'public_block_independent_writable': True,
            'cache_readonly_and_same_site_identity': True,
            'pickle_excludes_mapping_and_payload': True,
            'snapshot_survives_file_rewrite_and_truncation': True,
            'unfinished_and_truncated_errors': True,
            'zero_copy_public_view_rejected_due_to_snapshot_aliasing': zero_copy_snapshot_changed}


def read_sequence(stores, accesses):
    reset(stores)
    checksum = 0.
    for index, site in accesses:
        checksum += float(stores[index].read_site(site).sum())
    return checksum


def paired(stores, accesses):
    runs = {'before': [], 'mmap_copy': []}
    for repeat in range(4):
        outputs = {}
        for label in (('before', 'mmap_copy') if repeat % 2 == 0 else ('mmap_copy', 'before')):
            with patched(label == 'mmap_copy'):
                outputs[label], timing = measure(read_sequence, stores, accesses)
            if repeat:
                runs[label].append(timing)
        assert outputs['before'] == outputs['mmap_copy']
    reset(stores)
    return {'runs': runs, 'summary': {label: summarize(rows) for label, rows in runs.items()},
            'access_count': len(accesses), 'exact_checksum_parity': True,
            'map_initialization_and_cache_reset_included': True}


class Captured(BaseException):
    pass


def real_case(dataset, workdir):
    original = substitution_scan.scan_substitutions
    output = {}

    def capture(g, ON_tensor, rate_ON_tensor=None):
        static = substitution_scan._build_scan_static_context(g, ON_tensor, rate_ON_tensor)
        stores, accesses = [], []

        def logged(stored, site):
            if not any(stored is item for item in stores):
                stores.append(stored)
            index = next(i for i, item in enumerate(stores) if item is stored)
            accesses.append((index, int(site)))
            return ORIGINAL_READ(stored, site)

        site_storage.SiteArray.read_site = logged
        try:
            reference = substitution_scan._scan_substitutions_core(
                g, ON_tensor, rate_ON_tensor, scan_static=static)
        finally:
            site_storage.SiteArray.read_site = ORIGINAL_READ
        output['stores'] = [{'shape': list(store.shape), 'nbytes': store.nbytes,
                             'threshold': getattr(store, 'min_sub_pp', None)} for store in stores]
        output['read_sequence'] = paired(stores, accesses)
        runs = {'before': [], 'mmap_copy': []}
        for repeat in range(4):
            frames = {}
            for label in (('before', 'mmap_copy') if repeat % 2 == 0 else ('mmap_copy', 'before')):
                reset(stores)
                with patched(label == 'mmap_copy'):
                    frames[label], timing = measure(substitution_scan._scan_substitutions_core,
                                                    g, ON_tensor, rate_ON_tensor, scan_static=static)
                if repeat:
                    runs[label].append(timing)
            for pair in zip(frames['before'], frames['mmap_copy'], reference):
                pd.testing.assert_frame_equal(pair[0], pair[1], check_exact=True)
                pd.testing.assert_frame_equal(pair[0], pair[2], check_exact=True)
        output['scan_core'] = {'runs': runs,
                               'summary': {label: summarize(rows) for label, rows in runs.items()},
                               'scan_rows': len(reference[0]), 'exact_frame_parity': True}
        reset(stores)
        raise Captured

    data = workdir / (dataset + '_inputs')
    data.mkdir()
    for name in ([dataset + '.alignment.fa', dataset + '.tree.nwk', dataset + '.foreground.txt']
                 + [dataset + '.alignment.fa.' + key for key in ('treefile', 'state', 'rate', 'iqtree', 'log')]):
        shutil.copy2(ROOT / 'csubst/dataset' / name, data / name)
    argv = ['scan', '--alignment_file', str(data / (dataset + '.alignment.fa')),
            '--rooted_tree_file', str(data / (dataset + '.tree.nwk')),
            '--foreground', str(data / (dataset + '.foreground.txt')),
            '--outdir', str(workdir / dataset), '--threads', '1', '--blas_threads', '1',
            '--scan_site_plot', 'no', '--scan_pvalue_calibration', 'none',
            '--random_seed', '8', '--float_digit', '12']
    for key in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        argv += ['--iqtree_' + key, str(data / (dataset + '.alignment.fa.' + key))]
    substitution_scan.scan_substitutions = capture
    try:
        with (workdir / (dataset + '.log')).open('w') as log, contextlib.redirect_stdout(log):
            try:
                cli.main(argv)
            except Captured:
                pass
    finally:
        substitution_scan.scan_substitutions = original
        site_storage.SiteArray.read_site = ORIGINAL_READ
    output['input_sha256'] = {
        'csubst/dataset/' + path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(data.iterdir()) if path.is_file()}
    return output


def memory_child(filename, shape, mapped):
    stored = site_storage.SiteEventTensor.__new__(site_storage.SiteEventTensor)
    stored.shape = tuple(shape)
    stored.site_axis = 1
    stored.site_shape = tuple(shape[:1] + shape[2:])
    stored.site_size = int(np.prod(stored.site_shape))
    stored.dtype = np.dtype(float)
    stored.filename, stored._sealed = filename, True
    stored._cached_site = stored._cached_values = None
    stored.min_sub_pp = 0
    with patched(mapped):
        result, timing = measure(read_sequence, [stored], [(0, i) for i in range(shape[1])])
    return {'checksum': result, **timing,
            'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
            (1024**2 if sys.platform == 'darwin' else 1024)}


def paired_lifecycle(shape):
    rng = np.random.default_rng(82)
    blocks = [rng.random((min(32, shape[1] - start), shape[0], *shape[2:]))
              for start in range(0, shape[1], 32)]

    def lifecycle():
        stored = site_storage.SiteEventTensor(shape)
        for index, block in enumerate(blocks):
            stored.write_block(index * 32, block)
        stored.seal()
        try:
            return read_sequence([stored], [(0, i) for i in range(shape[1])])
        finally:
            reset([stored])
            stored._directory.cleanup()

    runs = {'before': [], 'mmap_copy': []}
    for repeat in range(4):
        outputs = {}
        for label in (('before', 'mmap_copy') if repeat % 2 == 0 else ('mmap_copy', 'before')):
            with patched(label == 'mmap_copy'):
                outputs[label], timing = measure(lifecycle)
            if repeat:
                runs[label].append(timing)
        assert outputs['before'] == outputs['mmap_copy']
    return {'shape': list(shape), 'scope': 'file create, write, seal, full site reads, unmap, delete',
            'input_block_generation_excluded': True, 'exact_checksum_parity': True,
            'runs': runs, 'summary': {label: summarize(rows) for label, rows in runs.items()}}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--check-existing-tests':
        import pytest
        with benchmark_lock(), patched(True):
            raise SystemExit(pytest.main(['tests/unit/test_site_storage.py', '-q']))
    if len(sys.argv) > 1 and sys.argv[1] == '--memory-child':
        print(json.dumps(memory_child(sys.argv[2], json.loads(sys.argv[3]), sys.argv[4] == 'mmap_copy')))
        return
    result = {'proposal': 8, 'environment': environment(), 'cases': {},
              'warmups': 1, 'measured_repeats': 3,
              'scope': 'warm OS page cache, fresh mapping per sample; public block reads unchanged',
              'thread_control': 'VECLIB/OPENBLAS/OMP/MKL/BLIS=1 before NumPy import'}
    with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-proposal08-') as directory:
        result['checks'] = checks()
        for dataset in ('PGK', 'PEPC'):
            result['cases'][dataset] = real_case(dataset, Path(directory))
            print(dataset, {key: {label: value['wall_s']['median'] for label, value in
                                 result['cases'][dataset][key]['summary'].items()}
                            for key in ('read_sequence', 'scan_core')}, flush=True)
        result['synthetic_lifecycle'] = paired_lifecycle((141, 971, 1, 20, 20))
        stored = write_store((141, 971, 1, 20, 20))
        result['memory_sequential_synthetic'] = {'shape': list(stored.shape), 'payload_bytes': stored.nbytes,
                                                'before': [], 'mmap_copy': []}
        for repeat in range(4):
            for label in (('before', 'mmap_copy') if repeat % 2 == 0 else ('mmap_copy', 'before')):
                command = [sys.executable, str(Path(__file__).resolve()), '--memory-child',
                           stored.filename, json.dumps(list(stored.shape)), label]
                row = json.loads(subprocess.check_output(command, text=True))
                if repeat:
                    result['memory_sequential_synthetic'][label].append(row)
        assert len({row['checksum'] for label in ('before', 'mmap_copy')
                    for row in result['memory_sequential_synthetic'][label]}) == 1
        del stored
        gc.collect()
    (HERE / 'proposal08.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
