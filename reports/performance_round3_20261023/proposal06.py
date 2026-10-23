"""Profile bounded per-site endpoint posterior preparation against 624bb8a.

Run: python reports/performance_round3_20261023/proposal06.py
CSUBST_PERF_BASELINE selects an extracted baseline; otherwise use the session's
/tmp/csubst-round3-baseline-path pointer. Production source is never modified.
"""

import argparse
import contextlib
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import types

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['MPLBACKEND'] = 'Agg'

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASELINE = os.environ.get('CSUBST_PERF_BASELINE')
if not BASELINE:
    BASELINE = Path('/tmp/csubst-round3-baseline-path').read_text().strip()
sys.path.insert(0, BASELINE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import cli, scan_endpoint, site_storage, substitution_scan as scan  # noqa: E402


def prototype():
    endpoint_source = (Path(BASELINE) / 'csubst/scan_endpoint.py').read_text()
    scan_source = (Path(BASELINE) / 'csubst/substitution_scan.py').read_text()
    start = endpoint_source.index('    parent = np.asarray(state_cdn[context["parent_ids"]')
    stop = endpoint_source.index('        category_weights = np.einsum', start)
    block = endpoint_source[start:stop]
    split = block.index("    if 'category_groups' in context:")
    helper = ('def _posterior_site(context, state_cdn, state_nsy, site):\n'
              '    """Prepare one site of an immutable scan fit; never retain input arrays."""\n'
              + block[:split] + '    weighted_parent = None\n' + block[split:]
              + '    return parent, normalized, missing, weighted_parent\n\n\n')
    replacement = '''    def prepare():
        return _posterior_site(context, state_cdn, state_nsy, site)
    if _site_cache is None:
        parent, normalized, missing, weighted_parent = prepare()
    else:
        key = ('posterior_site', id(context), id(state_cdn), id(state_nsy), int(site))
        parent, normalized, missing, weighted_parent = _site_cache.obtain(
            key, prepare, lambda values: sum(value.nbytes for value in values if value is not None))
    if 'category_groups' in context:
'''
    candidate_endpoint_source = endpoint_source[:start] + replacement + endpoint_source[stop:]
    signature = 'def expected_events(context, state_cdn, state_nsy, site, from_ids, to_ids):'
    assert candidate_endpoint_source.count(signature) == 1
    candidate_endpoint_source = candidate_endpoint_source.replace(
        signature, helper + signature[:-2] + ', _site_cache=None):')
    old_call = 'return scan_endpoint.expected_events(endpoint_context, state_cdn, state_nsy, site, from_ids, to_ids)'
    new_call = ('return scan_endpoint.expected_events(endpoint_context, state_cdn, state_nsy, site, '
                'from_ids, to_ids, exposure_cache)')
    assert scan_source.count(old_call) == 1
    candidate_scan_source = scan_source.replace(old_call, new_call)
    patches = []
    for filename, before, after in (
            ('scan_endpoint.py', endpoint_source, candidate_endpoint_source),
            ('substitution_scan.py', scan_source, candidate_scan_source)):
        patches.append(''.join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile='a/csubst/' + filename, tofile='b/csubst/' + filename, n=0)))
    (HERE / 'proposal06.patch').write_text(''.join(patches))
    endpoint_ns = dict(scan_endpoint.__dict__)
    exec(compile(candidate_endpoint_source, '<proposal06-endpoint>', 'exec'), endpoint_ns)
    scan_ns = dict(scan.__dict__)
    exec(compile(candidate_scan_source, '<proposal06-scan>', 'exec'), scan_ns)
    scan_ns['scan_endpoint'] = types.SimpleNamespace(**endpoint_ns)
    return endpoint_ns, scan_ns


def equal(first, second):
    if isinstance(first, pd.DataFrame):
        pd.testing.assert_frame_equal(first, second, check_exact=True)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            equal(left, right)
    else:
        np.testing.assert_equal(first, second)


def outcome(function, *arguments):
    try:
        return 'ok', function(*arguments)
    except Exception as exc:
        return 'error', type(exc).__name__, str(exc)


def correctness(candidate):
    fixture = runpy.run_path(str(ROOT / 'tests/integration/test_scan_endpoint.py'))['context']
    cases = []
    for length in (0., 1e-8, 1., 100.):
        for groups in ((0, 1, 2), (0, 0, 1)):
            for mode in ('ordinary', 'missing', 'eligibility', 'bridge', 'mixture', 'mixture_sitearray'):
                g, meta, ids = fixture((length, length), groups)
                context = scan_endpoint.build_context(g, meta, ids)
                if mode == 'missing':
                    g['state_cdn'][2] = 0
                elif mode == 'eligibility':
                    context['eligible'] = np.array([[True], [False], [True]])
                elif mode == 'bridge':
                    g['scan_observation'] = 'bridge'
                    context = scan_endpoint.build_context(g, meta, ids)
                elif mode.startswith('mixture'):
                    # Synthetic category state partition preserves each marginal.
                    context['category_groups'] = np.repeat(context['transition_groups'][None], 2, axis=0)
                    categories = np.stack([g['state_cdn'] * .25, g['state_cdn'] * .75])
                    if mode.endswith('sitearray'):
                        storage = site_storage.SiteArray(categories.shape, categories.dtype, site_axis=2)
                        storage.write_block(0, np.moveaxis(categories, 2, 0))
                        storage.seal()
                        categories = storage
                    context['category_states'] = categories
                cache = scan._CandidateCache()
                for source, dest in (([0], [max(ids)]), (list(range(max(ids) + 1)), [max(ids)]),
                                     ([], []), ([0], [0]), ([0, 0], [max(ids), max(ids)])):
                    args = context, g['state_cdn'], g['state_nsy'], 0, source, dest
                    equal(scan_endpoint.expected_events(*args), candidate['expected_events'](*args, cache))
                cases.append(dict(length=length, groups=list(groups), mode=mode))
    errors = []
    for scenario in ('range_first', 'negative', 'nonfinite', 'row_sum', 'child_row_sum',
                     'nsy_negative', 'bad_site', 'missing_categories', 'wrong_categories'):
        g, meta, ids = fixture()
        context = scan_endpoint.build_context(g, meta, ids)
        source, site = [0], 0
        if scenario in ('range_first', 'negative'):
            g['state_cdn'][0, 0, 0] = -1
            if scenario == 'range_first':
                source = [-1]
        elif scenario == 'nonfinite':
            g['state_cdn'][0, 0, 0] = np.nan
        elif scenario in ('row_sum', 'child_row_sum'):
            g['state_cdn'][0 if scenario == 'row_sum' else 1, 0] = .2
        elif scenario == 'nsy_negative':
            g['state_nsy'][0, 0, 0] = -1
        elif scenario == 'bad_site':
            site = 999
        elif scenario in ('missing_categories', 'wrong_categories'):
            context['category_groups'] = context['transition_groups'][None]
            context['category_states'] = None if scenario == 'missing_categories' else g['state_cdn'][None] * .5
        args = context, g['state_cdn'], g['state_nsy'], site, source, [2]
        before = outcome(scan_endpoint.expected_events, *args)
        for cache in (None, scan._CandidateCache()):
            after = outcome(candidate['expected_events'], *args, cache)
            equal(before, after)
        assert before[0] == 'error'
        errors.append(dict(scenario=scenario, error=list(before[1:])))
    # Ordinary public calls always re-read mutable states after successful calls.
    g, meta, ids = fixture()
    context = scan_endpoint.build_context(g, meta, ids)
    args = context, g['state_cdn'], g['state_nsy'], 0, [0], [2]
    candidate['expected_events'](*args, scan._CandidateCache())
    g['state_cdn'][0, 0, 0] = -1
    equal(outcome(scan_endpoint.expected_events, *args), outcome(candidate['expected_events'], *args))
    # Bounded memory, no retention for overbudget sites, and LRU eviction.
    g, meta, ids = fixture()
    context = scan_endpoint.build_context(g, meta, ids)
    args = context, g['state_cdn'], g['state_nsy'], 0, [0], [2]
    tiny = scan._CandidateCache(max_bytes=1)
    candidate['expected_events'](*args, tiny)
    assert tiny.nbytes == 0 and not tiny.entries
    bounded = scan._CandidateCache(max_bytes=220, max_items=1)
    candidate['expected_events'](*args, bounded)
    bounded.obtain(('vector',), lambda: np.zeros(20), lambda value: value.nbytes)
    assert bounded.nbytes <= 220 and len(bounded.entries) == 1
    return dict(exact_cases=cases, exact_errors=errors, public_mutable_inputs_revalidated=True,
                overbudget_not_retained=True, shared_budget_lru=True)


class Captured(Exception):
    pass


def scan_timed(namespace, g, tensor, rate_tensor):
    # Test instrumentation retains counters only; it does not keep cached arrays.
    counters = dict(site_hit=0, site_build=0, vector_hit=0, vector_build=0,
                    peak_payload_bytes=0, peak_site_bytes=0)
    base_class = namespace['_CandidateCache']
    core = namespace['_scan_substitutions_core']
    initial = {}

    class RecordingCache(base_class):
        def obtain(self, key, build, size_of):
            kind = 'site' if key[0] == 'posterior_site' else 'vector'
            counters[kind + ('_hit' if key in self.entries else '_build')] += 1
            value = super().obtain(key, build, size_of)
            counters['peak_payload_bytes'] = max(counters['peak_payload_bytes'], self.nbytes)
            site_bytes = sum(size for item_key, (_, size) in self.entries.items()
                             if item_key[0] == 'posterior_site')
            counters['peak_site_bytes'] = max(counters['peak_site_bytes'], site_bytes)
            return value

    def timed_core(*args, **kwargs):
        result, timing = measure(core, *args, **kwargs)
        initial.update(timing)
        return result

    namespace['_CandidateCache'] = RecordingCache
    namespace['_scan_substitutions_core'] = timed_core
    try:
        with open(os.devnull, 'w') as stream, contextlib.redirect_stdout(stream):
            result, timing = measure(namespace['scan_substitutions'], g, tensor, rate_tensor)
    finally:
        namespace['_CandidateCache'] = base_class
        namespace['_scan_substitutions_core'] = core
    return result, dict(total=timing, initial=initial), counters


def real_case(dataset, matches, candidate, workdir, warmups, repeats, permutations):
    saved = scan.scan_substitutions
    result = {}

    def capture(g, ON_tensor, rate_ON_tensor=None):
        runs = {label: {phase: [] for phase in ('initial', 'total')}
                for label in ('baseline', 'cached')}
        counters = {}
        for index in range(warmups + repeats):
            outputs, diagnostics = {}, {}
            order = ('baseline', 'cached') if index % 2 == 0 else ('cached', 'baseline')
            for label in order:
                namespace = scan.__dict__ if label == 'baseline' else candidate
                namespace['scan_substitutions'] = saved if label == 'baseline' else candidate_function
                run_g = copy.deepcopy(g)
                outputs[label], timings, counters[label] = scan_timed(
                    namespace, run_g, ON_tensor, rate_ON_tensor)
                diagnostics[label] = run_g.get('scan_calibration_diagnostics')
                if index >= warmups:
                    for phase, values in timings.items():
                        runs[label][phase].append(values)
            equal(outputs['baseline'], outputs['cached'])
            equal(diagnostics['baseline'], diagnostics['cached'])
        frame = outputs['baseline'][0]
        result.update(runs=runs, summary={label: {phase: summarize(values)
                      for phase, values in phases.items()} for label, phases in runs.items()},
                      counters=counters, candidates=len(frame), exact_outputs_and_diagnostics=True,
                      distinct_candidate_sites=int(frame.site.nunique()) if len(frame) else 0,
                      completed_permutations=diagnostics['baseline'].get('success_count'),
                      scope='initial scan core and total scan_substitutions stage; excludes CLI loading and inference')
        raise Captured

    candidate_function = candidate['scan_substitutions']
    data = workdir / (dataset + '_' + matches.replace(',', '_'))
    data.mkdir()
    names = [dataset + suffix for suffix in ('.alignment.fa', '.tree.nwk', '.foreground.txt')]
    names.extend(dataset + '.alignment.fa.' + suffix for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log'))
    hashes = {}
    for name in names:
        shutil.copyfile(Path(BASELINE) / 'csubst/dataset' / name, data / name)
        hashes[name] = hashlib.sha256((data / name).read_bytes()).hexdigest()
    argv = ['scan', '--alignment_file', str(data / (dataset + '.alignment.fa')),
            '--rooted_tree_file', str(data / (dataset + '.tree.nwk')),
            '--foreground', str(data / (dataset + '.foreground.txt')),
            '--outdir', str(data / 'output'), '--threads', '1', '--blas_threads', '1',
            '--scan_site_plot', 'no', '--scan_match', matches,
            '--scan_pvalue_calibration', 'candidate_fixed', '--scan_n_permutations', str(permutations),
            '--random_seed', '8', '--float_digit', '12']
    for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        argv.extend(['--iqtree_' + suffix, str(data / (dataset + '.alignment.fa.' + suffix))])
    scan.scan_substitutions = capture
    try:
        with (data / 'run.log').open('w') as stream, contextlib.redirect_stdout(stream):
            try:
                cli.main(argv)
            except Captured:
                pass
    finally:
        scan.scan_substitutions = saved
    assert result
    return dict(result, input_sha256=hashes, scan_match=matches, requested_permutations=permutations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--permutations', type=int, default=10)
    args = parser.parse_args()
    endpoint, candidate = prototype()
    result = dict(proposal=6, baseline_commit='624bb8a6dae536d0fa10891f666684ef1ad3dab0',
                  environment=environment(), warmups=args.warmups, repeats=args.repeats,
                  command='python reports/performance_round3_20261023/proposal06.py',
                  decision='reject',
                  decision_reason=('Actual site reuse is outweighed by retention and eviction of existing '
                                   'candidate vectors on PEPC; no uninstrumented benefit has been established.'),
                  measurement_caveat=('Cache hit/build and retained-payload profiling is inside the timed region. '
                                      'The per-obtain retained-site traversal adds overhead, especially when the '
                                      'site cache causes vector eviction. These are instrumented timings, not '
                                      'production runtime claims; wall time is also highly variable.'),
                  memory_scope=('High-water retained cache array payload, not peak process RSS. Site preparations '
                                'and candidate exposure vectors share the existing 32 MiB/4,096-entry LRU. '
                                'No additional long-lived cache; worker transport drops it with the exposure cache.'),
                  source_sha256={name: hashlib.sha256((Path(BASELINE) / 'csubst' / name).read_bytes()).hexdigest()
                                 for name in ('scan_endpoint.py', 'substitution_scan.py')}, cases={})
    print('Proposal06 ready; waiting for benchmark lock.', flush=True)
    with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-p06-') as temporary:
        result['correctness'] = correctness(endpoint)
        for matches in ('any2spe', 'any2spe,spe2spe'):
            for dataset in ('PGK', 'PEPC'):
                key = dataset + '_' + matches
                result['cases'][key] = real_case(dataset, matches, candidate, Path(temporary),
                                                 args.warmups, args.repeats, args.permutations)
                (HERE / 'proposal06.json').write_text(json.dumps(result, indent=2) + '\n')
                print('Proposal06', key, result['cases'][key]['summary'],
                      result['cases'][key]['counters'], flush=True)
    (HERE / 'proposal06.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
