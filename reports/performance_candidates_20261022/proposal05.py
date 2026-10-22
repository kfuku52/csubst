"""Bounded per-site endpoint exposure preparation; independent research prototype."""

from collections import OrderedDict
import contextlib
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

for variable in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import cli, scan_endpoint, site_storage  # noqa: E402


class PreparedSnapshot:
    """Own state copies; never infer immutability from array identity alone."""

    def __init__(self, context, state_cdn, state_nsy, budget=32 * 1024**2):
        self.context = {key: value.copy() if isinstance(value, np.ndarray) else value
                        for key, value in context.items()}
        self.state_cdn = state_cdn.copy()
        self.state_nsy = state_nsy.copy()
        for array in (self.state_cdn, self.state_nsy, *self.context.values()):
            if isinstance(array, np.ndarray):
                array.flags.writeable = False
        # Category file objects are sealed immutable stores owned by this run.
        # ndarray category states were copied above with all other context arrays.
        self.budget, self.cache, self.cache_bytes = budget, OrderedDict(), 0
        self.peak_cache_bytes, self.hits, self.misses, self.evictions = 0, 0, 0, 0
        self.snapshot_bytes = (self.state_cdn.nbytes + self.state_nsy.nbytes
                               + sum(value.nbytes for value in self.context.values()
                                     if isinstance(value, np.ndarray)))

    def site(self, site):
        site = int(site)
        if site in self.cache:
            self.hits += 1
            self.cache.move_to_end(site)
            return self.cache[site][0]
        self.misses += 1
        context = self.context
        parent = np.asarray(self.state_cdn[context['parent_ids'], site, :], dtype=float)
        child = np.asarray(self.state_cdn[context['branch_ids'], site, :], dtype=float)
        nsy_parent = np.asarray(self.state_nsy[context['parent_ids'], site, :], dtype=float)
        nsy_child = np.asarray(self.state_nsy[context['branch_ids'], site, :], dtype=float)
        for values in (parent, child, nsy_parent, nsy_child):
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('Scan endpoint posterior states must be finite and nonnegative.')
            total = values.sum(axis=1)
            if not np.all((total == 0) | np.isclose(total, 1, atol=1e-4, rtol=0)):
                raise ValueError('Scan endpoint posterior rows must sum to one (or zero for missing states).')
        mass = parent.sum(axis=1)
        missing = ((mass == 0) | (child.sum(axis=1) == 0)
                   | (nsy_parent.sum(axis=1) == 0) | (nsy_child.sum(axis=1) == 0))
        if 'eligible' in context:
            missing |= ~context['eligible'][context['branch_ids'], site]
        normalized = np.divide(parent, mass[:, None], out=np.zeros_like(parent), where=mass[:, None] > 0)
        weighted_parent = None
        if 'category_groups' in context:
            category_states = context.get('category_states')
            if category_states is None:
                raise ValueError('Mixture exposure requires category-conditional joint posteriors.')
            weighted_parent = (category_states.read_site(site)[:, context['parent_ids'], :]
                               if isinstance(category_states, site_storage.SiteArray) else
                               category_states[:, context['parent_ids'], site, :])
            if (not np.isfinite(weighted_parent).all() or (weighted_parent < 0).any()
                    or not np.allclose(weighted_parent.sum(axis=0), parent, atol=1e-10, rtol=1e-8)):
                raise ValueError('Category posterior states do not match scan parent marginals.')
        values = parent, missing, normalized, weighted_parent
        size = sum(value.nbytes for value in values if isinstance(value, np.ndarray))
        for value in values:
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        if size <= self.budget:
            while self.cache and self.cache_bytes + size > self.budget:
                _, (_, old_size) = self.cache.popitem(last=False)
                self.cache_bytes -= old_size
                self.evictions += 1
            self.cache[site] = values, size
            self.cache_bytes += size
            self.peak_cache_bytes = max(self.peak_cache_bytes, self.cache_bytes)
        return values

    def expected(self, site, source, destination):
        return PREPARED_EVENTS(self.context, self.state_cdn, self.state_nsy,
                               site, source, destination, self)


def make_prepared_function():
    """Retain every candidate-dependent calculation verbatim from the baseline."""
    source = inspect.getsource(scan_endpoint.expected_events)
    source = source.replace('def expected_events(context, state_cdn, state_nsy, site, from_ids, to_ids):',
                            'def prepared_events(context, state_cdn, state_nsy, site, from_ids, to_ids, prepared):')
    start = source.index('    parent = np.asarray(')
    stop = source.index("    if 'category_groups' in context:", start)
    source = source[:start] + '    parent, missing, normalized, weighted_parent = prepared.site(site)\n' + source[stop:]
    start = source.index("        category_states = context.get('category_states')")
    stop = source.index('        category_weights =', start)
    source = source[:start] + source[stop:]
    namespace = {'np': np}
    exec(compile(source, '<proposal05-prepared-exposure>', 'exec'), namespace)
    return namespace['prepared_events']


PREPARED_EVENTS = make_prepared_function()


def capture(dataset, match, workdir):
    data = workdir / 'data'
    data.mkdir(exist_ok=True)
    names = [dataset + '.alignment.fa', dataset + '.tree.nwk', dataset + '.foreground.txt']
    names += [dataset + '.alignment.fa.' + suffix for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log')]
    for name in names:
        shutil.copyfile(ROOT / 'csubst/dataset' / name, data / name)
    calls, captured = [], {}
    original = scan_endpoint.expected_events

    def observe(context, state_cdn, state_nsy, site, source, destination):
        if not captured:
            owned_context = dict(context)
            category = context.get('category_states')
            if isinstance(category, site_storage.SiteArray):
                cloned = copy.copy(category)
                destination_path = workdir / (dataset + '-' + match + '-category.bin')
                shutil.copyfile(category.filename, destination_path)
                cloned.filename = str(destination_path)
                cloned._cached_site = cloned._cached_values = None
                owned_context['category_states'] = cloned
            captured.update(context=owned_context, state_cdn=state_cdn, state_nsy=state_nsy)
        calls.append((int(site), np.asarray(source).copy(), np.asarray(destination).copy()))
        return original(context, state_cdn, state_nsy, site, source, destination)

    args = ['scan', '--alignment_file', str(data / (dataset + '.alignment.fa')),
            '--rooted_tree_file', str(data / (dataset + '.tree.nwk')),
            '--foreground', str(data / (dataset + '.foreground.txt')),
            '--outdir', str(workdir / (dataset + '-' + match)), '--threads', '1',
            '--blas_threads', '1', '--scan_site_plot', 'no', '--scan_match', match,
            '--scan_pvalue_calibration', 'none', '--random_seed', '8']
    for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        args += ['--iqtree_' + suffix, str(data / (dataset + '.alignment.fa.' + suffix))]
    scan_endpoint.expected_events = observe
    try:
        with (workdir / (dataset + '-' + match + '.log')).open('w') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                cli.main(args)
    finally:
        scan_endpoint.expected_events = original
    assert calls, 'No endpoint exposure calls reached'
    return captured, calls, {name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in names}


def replay(config, calls, prepared, budget=32 * 1024**2):
    snapshot = PreparedSnapshot(**config, budget=budget) if prepared else None
    outputs = []
    for site, source, destination in calls:
        result = (snapshot.expected(site, source, destination) if snapshot else
                  scan_endpoint.expected_events(**config, site=site, from_ids=source, to_ids=destination))
        outputs.append(result)
    info = ({'hits': snapshot.hits, 'misses': snapshot.misses, 'evictions': snapshot.evictions,
             'peak_cache_bytes': snapshot.peak_cache_bytes, 'snapshot_bytes': snapshot.snapshot_bytes}
            if snapshot else {})
    return outputs, info


def equal(before, after):
    maximum = 0.
    assert len(before) == len(after)
    for expected, actual in zip(before, after):
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)
            if left.dtype.kind == 'f':
                maximum = max(maximum, float(np.max(np.abs(left - right), initial=0.)))
    return maximum


def small_checks():
    import importlib.util

    spec = importlib.util.spec_from_file_location('endpoint_test_helpers', ROOT / 'tests/integration/test_scan_endpoint.py')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    checks = {}
    for mode in ('uniform', 'missing', 'zero_length', 'zero_source', 'unreachable', 'eligible', 'mixture', 'bridge'):
        g, meta, ids = helpers.context((0., 1.) if mode == 'zero_length' else (1., 2.))
        if mode == 'missing':
            g['state_cdn'][2] = 0
        if mode == 'unreachable':
            g['instantaneous_codon_rate_matrix'] = np.array([[-1.5, 1.5, 0], [1.5, -1.5, 0], [0, 0, 0]])
        if mode == 'eligible':
            g['event_eligible'] = np.array([[True], [True], [False]])
        if mode == 'bridge':
            g['scan_observation'] = 'bridge'
        context = scan_endpoint.build_context(g, meta, ids)
        if mode == 'mixture':
            context['category_groups'] = np.stack([context['transition_groups']] * 2)
            context['category_states'] = np.stack([g['state_cdn'] * .4, g['state_cdn'] * .6])
        config = dict(context=context, state_cdn=g['state_cdn'], state_nsy=g['state_nsy'])
        source = [2] if mode == 'zero_source' else [0]
        calls = [(0, np.array(source), np.array(destination)) for destination in ([1], [2], [0, 1, 2])]
        before, _ = replay(config, calls, False)
        after, info = replay(config, calls, True)
        checks[mode] = {'max_abs_difference': equal(before, after), **info}
    g, meta, ids = helpers.context()
    context = scan_endpoint.build_context(g, meta, ids)
    config = dict(context=context, state_cdn=g['state_cdn'], state_nsy=g['state_nsy'])
    snapshot = PreparedSnapshot(**config)
    old = snapshot.expected(0, [0], [2])
    g['state_cdn'][0, 0] = [0, 1, 0]
    equal([old], [snapshot.expected(0, [0], [2])])
    rebuilt = PreparedSnapshot(**config)
    actual = scan_endpoint.expected_events(**config, site=0, from_ids=[0], to_ids=[2])
    equal([actual], [rebuilt.expected(0, [0], [2])])
    assert not np.array_equal(actual[0], old[0])
    checks['mutation_rebuild'] = 'owned snapshot unaffected; rebuilt snapshot matches changed original'
    invalid = {}
    for kind in ('nan', 'negative', 'row_sum', 'category_mismatch', 'candidate_range'):
        g, meta, ids = helpers.context()
        context = scan_endpoint.build_context(g, meta, ids)
        if kind == 'nan':
            g['state_cdn'][0, 0, 0] = np.nan
        elif kind == 'negative':
            g['state_nsy'][0, 0] = [-.1, .6, .5]
        elif kind == 'row_sum':
            g['state_cdn'][0, 0, 0] = .8
        elif kind == 'category_mismatch':
            context['category_groups'] = np.stack([context['transition_groups']] * 2)
            context['category_states'] = np.zeros((2,) + g['state_cdn'].shape)
        config = dict(context=context, state_cdn=g['state_cdn'], state_nsy=g['state_nsy'])
        destination = [99] if kind == 'candidate_range' else [2]
        messages = []
        for prepared in (False, True):
            try:
                replay(config, [(0, np.array([0]), np.array(destination))], prepared)
            except ValueError as error:
                messages.append(str(error))
        assert len(messages) == 2 and messages[0] == messages[1]
        invalid[kind] = messages[0]
    checks['invalid_input_matching_errors'] = invalid
    # Force eviction and revalidation at the same site after another site.
    config['context'].pop('category_groups', None)
    config['context'].pop('category_states', None)
    config['state_cdn'] = np.tile(np.nan_to_num(config['state_cdn']), (1, 2, 1))
    config['state_nsy'] = np.tile(config['state_nsy'], (1, 2, 1))
    calls = [(site, np.array([0]), np.array([2])) for site in (0, 1, 0)]
    baseline, _ = replay(config, calls, False)
    candidate, info = replay(config, calls, True, budget=110)
    equal(baseline, candidate)
    assert info['evictions'] == 2 and info['peak_cache_bytes'] <= 110
    checks['bounded_eviction'] = info
    return checks


def main():
    result = {'proposal': 5, 'environment': environment(), 'cases': {},
              'scope': 'captured expected_events calls replay, including fresh snapshot copies and cold site cache; not CLI',
              'warmups': 1, 'measured_repeats': 3, 'cache_budget_bytes': 32 * 1024**2,
              'memory_scope': 'explicit snapshot and cached ndarray payload, not peak RSS'}
    with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-proposal05-') as directory:
        workdir = Path(directory)
        for dataset in ('PGK', 'PEPC'):
            for match in ('any2spe', 'all'):
                config, calls, hashes = capture(dataset, match, workdir)
                before, _ = replay(config, calls, False)
                after, info = replay(config, calls, True)
                maximum = equal(before, after)
                samples = {'baseline': [], 'prepared': []}
                for repeat in range(3):
                    pair = {}
                    for prepared in ((False, True) if repeat % 2 == 0 else (True, False)):
                        (values, info), timing = measure(replay, config, calls, prepared)
                        samples['prepared' if prepared else 'baseline'].append(timing)
                        pair[prepared] = values
                    maximum = max(maximum, equal(pair[False], pair[True]))
                result['cases'][dataset + '_' + match] = {
                    'calls': len(calls), 'unique_sites': len({call[0] for call in calls}),
                    'max_abs_difference': maximum, 'cache': info, 'input_sha256': hashes,
                    'samples': samples, 'summary': {key: summarize(values) for key, values in samples.items()},
                }
                print(dataset, match, result['cases'][dataset + '_' + match]['summary'], flush=True)
        result['edge_checks'] = small_checks()
    (HERE / 'proposal05.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
