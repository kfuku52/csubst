"""Benchmark inference-local synonymous transition products; product unchanged.

Run: python reports/performance_round2_20261022/proposal07.py
"""

from collections import OrderedDict
import contextlib
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import cli, ete, scan_ctmc, site_storage  # noqa: E402

ORIGINAL = scan_ctmc._joint_infer


class Products:
    """Cache only private immutable transition/mask products, per infer call."""

    def __init__(self, mask=None, limit=32 * 1024**2):
        self.mask = mask
        self.limit = limit
        self.entries = OrderedDict()
        self.nbytes = self.peak = self.cross_bytes = self.hits = self.misses = 0

    def get(self, transition, mask=None):
        key = id(transition)
        if key in self.entries:
            original, cross = self.entries[key]
            assert original is transition
            self.entries.move_to_end(key)
            self.hits += 1
            return cross
        self.misses += 1
        cross = transition * (self.mask if mask is None else mask)
        size = transition.nbytes + cross.nbytes
        if size <= self.limit:
            while self.entries and self.nbytes + size > self.limit:
                original, old = self.entries.popitem(last=False)[1]
                self.nbytes -= original.nbytes + old.nbytes
                self.cross_bytes -= old.nbytes
            self.entries[key] = transition, cross
            self.nbytes += size
            self.cross_bytes += cross.nbytes
            self.peak = max(self.peak, self.nbytes)
        return cross

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, retained_product_bytes=self.cross_bytes,
                    accounted_array_bytes=self.nbytes, peak_accounted_array_bytes=self.peak,
                    limit_array_bytes=self.limit, entries=len(self.entries))


def prototype():
    source = inspect.getsource(ORIGINAL)
    marker = '    def transform(left, right, transition):\n'
    assert source.count(marker) == 1
    source = source.replace(marker, '    products = Products() if summaries is not None else None\n\n' + marker)
    old = "(transition * summaries['synonymous_mask'])"
    assert source.count(old) == 1
    source = source.replace(old, "products.get(transition, summaries['synonymous_mask'])")
    old = '    return posterior, tensor\n'
    assert source.count(old) == 1
    source = source.replace(old, "    if summaries is not None:\n        summaries['_proposal07_cache'] = products.stats()\n" + old)
    assert "left @ products.get(transition, summaries['synonymous_mask'])" in source
    namespace = dict(scan_ctmc.__dict__, Products=Products)
    exec(compile(source, '<synonymous-product-cache>', 'exec'), namespace)
    return namespace['_joint_infer'], hashlib.sha256(inspect.getsource(ORIGINAL).encode()).hexdigest()


CANDIDATE, SOURCE_HASH = prototype()


class Captured(BaseException):
    pass


def capture(dataset, workdir):
    data = workdir / 'data'
    data.mkdir()
    names = [dataset + suffix for suffix in ('.alignment.fa', '.tree.nwk', '.foreground.txt')]
    names += [dataset + '.alignment.fa.' + suffix for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log')]
    for name in names:
        shutil.copyfile(ROOT / 'csubst/dataset' / name, data / name)
    argv = ['scan', '--alignment_file', str(data / (dataset + '.alignment.fa')),
            '--rooted_tree_file', str(data / (dataset + '.tree.nwk')),
            '--foreground', str(data / (dataset + '.foreground.txt')),
            '--outdir', str(workdir / 'scan'), '--threads', '1', '--blas_threads', '1',
            '--scan_site_plot', 'no', '--scan_pvalue_calibration', 'none',
            '--scan_n_permutations', '0', '--random_seed', '8', '--float_digit', '12']
    for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        argv += ['--iqtree_' + suffix, str(data / (dataset + '.alignment.fa.' + suffix))]
    found = {}

    def intercept(*args, **kwargs):
        bound = inspect.signature(ORIGINAL).bind(*args, **kwargs)
        bound.apply_defaults()
        found.update(bound.arguments)
        raise Captured

    scan_ctmc._joint_infer = intercept
    try:
        with (workdir / 'capture.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                cli.main(argv)
            except Captured:
                pass
    finally:
        scan_ctmc._joint_infer = ORIGINAL
    assert found and found['disk_backed']
    return found, {name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in names}


def run(function, options):
    call = dict(options)
    summary = options['summaries']
    call['summaries'] = None if summary is None else dict(summary)
    result = function(**call)
    return result, call['summaries']


def compare_array(left, right):
    if isinstance(left, site_storage.SiteArray):
        assert left.shape == right.shape and left.dtype == right.dtype
        count = 0
        for (start, a), (other, b) in zip(left.iter_site_blocks(), right.iter_site_blocks()):
            assert start == other
            np.testing.assert_array_equal(a, b, strict=True)
            assert a.tobytes() == b.tobytes()
            count += a.size
        return count
    np.testing.assert_array_equal(left, right, strict=True)
    assert np.asarray(left).tobytes() == np.asarray(right).tobytes()
    return np.asarray(left).size


def parity(options):
    before, bs = run(ORIGINAL, options)
    after, ps = run(CANDIDATE, options)
    count = sum(compare_array(a, b) for a, b in zip(before, after))
    if bs is not None:
        assert set(ps) == set(bs) | {'_proposal07_cache'}
        for key, value in bs.items():
            if value is None:
                assert ps[key] is None
            elif isinstance(value, (np.ndarray, site_storage.SiteArray)):
                count += compare_array(value, ps[key])
            else:
                assert value == ps[key]
        stats = ps['_proposal07_cache']
        assert stats['peak_accounted_array_bytes'] <= stats['limit_array_bytes']
    else:
        assert ps is None
        stats = None
    return {'all_outputs_bitwise_exact': True, 'asserted_values': int(count), 'cache': stats}


def benchmark(options):
    runs = {'before': [], 'cached': []}
    checksums = {}
    stats = None
    for repeat in range(4):
        order = ('before', 'cached') if repeat % 2 == 0 else ('cached', 'before')
        for label in order:
            gc.collect()
            (result, summary), timing = measure(run, ORIGINAL if label == 'before' else CANDIDATE, options)
            checksums[label] = (float(result[0].sum()), summary['synonymous_counts'].tobytes())
            if label == 'cached':
                stats = summary['_proposal07_cache']
            if repeat:
                runs[label].append(timing)
            del result, summary
        assert checksums['before'] == checksums['cached']
    assert stats['hits'] > 0 and stats['misses'] > 0
    return {'runs': runs, 'summary': {name: summarize(values) for name, values in runs.items()}, 'cache': stats}


def edges():
    tree = ete.PhyloNode('((A:0.2,B:0.3)X:0.1,C:0.4)R;', format=1)
    for index, node in enumerate(tree.traverse('preorder')):
        ete.set_prop(node, 'numerical_label', index)
    shape = (len(list(tree.traverse())), 9, 3)
    states = np.zeros(shape)
    for node in tree.traverse():
        if ete.is_leaf(node):
            states[int(ete.get_prop(node, 'numerical_label')), :, 0] = 1
    leaf_id = int(ete.get_prop(next(node for node in tree.traverse() if ete.is_leaf(node)), 'numerical_label'))
    q = np.full((3, 3), 0.5)
    np.fill_diagonal(q, -1)
    base = dict(tree=tree, tip_states=states, q=q, pi=np.full(3, 1 / 3), groups=np.array([0, 0, 1]),
                block_size=3, summaries={'synonymous_mask': np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])},
                rates=np.array([0., 1., 2.]), weights=np.array([.1, .5, .4]), disk_backed=False)
    records = []
    for mode in ('normal', 'missing', 'ambiguous', 'zero_q', 'no_summary', 'eligible'):
        for disk in (False, True):
            options = dict(base, disk_backed=disk, tip_states=states.copy())
            if mode == 'missing':
                options['tip_states'][:, 1] = 0
            elif mode == 'ambiguous':
                options['tip_states'][:, 2, :] = [.2, .3, .5]
            elif mode == 'zero_q':
                options['q'] = np.zeros((3, 3))
            elif mode == 'no_summary':
                options['summaries'] = None
            elif mode == 'eligible':
                eligible = np.ones(shape[:2], dtype=bool)
                eligible[:, 2] = False
                options['summaries'] = dict(base['summaries'], eligible=eligible)
            records.append(dict(mode=mode, disk_backed=disk, **parity(options)))
    for mode in ('negative_tip', 'nan_tip', 'bad_q', 'bad_pi', 'bad_block'):
        options = dict(base, tip_states=states.copy())
        if mode == 'negative_tip':
            options['tip_states'][leaf_id, 0, 0] = -1
        elif mode == 'nan_tip':
            options['tip_states'][leaf_id, 0, 0] = np.nan
        elif mode == 'bad_q':
            options['q'] = q + 1
        elif mode == 'bad_pi':
            options['pi'] = [1, 1, 1]
        else:
            options['block_size'] = 0
        errors = []
        for function in (ORIGINAL, CANDIDATE):
            try:
                run(function, options)
            except Exception as exc:
                errors.append((type(exc).__name__, str(exc)))
        assert len(errors) == 2 and errors[0] == errors[1]
        records.append({'mode': mode, 'same_error': list(errors[0])})
    small = Products(np.ones((3, 3)), limit=3 * 3 * 8 * 2)
    first, second = np.eye(3), np.full((3, 3), .2)
    small.get(first)
    small.get(first)
    small.get(second)
    assert small.hits == 1 and small.misses == 2 and len(small.entries) == 1
    assert next(iter(small.entries.values()))[0] is second
    assert small.nbytes <= small.limit
    records.append({'mode': 'identity_byte_eviction', 'cache': small.stats()})
    return records


def main():
    report = dict(proposal=7, environment=environment(), source_function_sha256=SOURCE_HASH,
                  scope='joint inference stage including cold product/transition construction and disk output; not whole CLI',
                  warmups=1, measured_repeats=3,
                  memory_scope='cache array byte accounting, not process RSS; both retained transition and product charged',
                  cases={})
    with benchmark_lock():
        report['edge_checks'] = edges()
        for dataset in ('PGK', 'PEPC'):
            with tempfile.TemporaryDirectory(prefix='csubst-round2-p07-') as directory:
                options, hashes = capture(dataset, Path(directory))
                report['cases'][dataset] = dict(input_sha256=hashes, shape=list(options['tip_states'].shape),
                                                rates=options['rates'].tolist(), weights=options['weights'].tolist(),
                                                requested_block_size=options['block_size'], disk_backed=True,
                                                parity=parity(options), **benchmark(options))
    (HERE / 'proposal07.json').write_text(json.dumps(report, indent=2) + '\n')
    for dataset, case in report['cases'].items():
        print(dataset, case['summary'], case['cache'])


if __name__ == '__main__':
    main()
