"""Compare the implementation with 4382e1f, using fixed inputs and exact outputs."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

os.environ.update(VECLIB_MAXIMUM_THREADS='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MPLBACKEND='Agg')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def paired(before, after, check):
    runs = []
    for repeat in range(4):
        functions = [('before', before), ('after', after)]
        if repeat % 2:
            functions.reverse()
        outputs = {}
        for label, function in functions:
            start, cpu = time.perf_counter(), time.process_time()
            outputs[label] = function()
            runs.append(dict(version=label, warmup=repeat == 0, seconds=time.perf_counter() - start,
                             cpu_seconds=time.process_time() - cpu))
        check(outputs['before'], outputs['after'])
    summary = {}
    for metric in ('seconds', 'cpu_seconds'):
        medians = {label: statistics.median(row[metric] for row in runs if not row['warmup']
                                          and row['version'] == label) for label in ('before', 'after')}
        summary[metric] = dict(medians=medians, ratio=medians['after'] / medians['before'])
    return dict(runs=runs, summary=summary, equivalent_pairs=4)


def phases():
    import numpy as np
    import pandas as pd
    from csubst import epistasis, ete, foreground, main_inspect, parser_iqtree, scan_statistics, substitution_scan, tree
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--only', nargs='+')
    args = parser.parse_args()
    old = {name: load(args.baseline_root / 'csubst' / (name + '.py'), 'baseline_' + name)
           for name in ('epistasis', 'foreground', 'main_inspect', 'parser_iqtree', 'scan_statistics', 'substitution_scan', 'tree')}
    result = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__,
                  baseline='4382e1f84d01b7bff56855965c4c6e55c3512c51', experiments={}, clade_payload={})
    def record(name, before, after, check):
        if args.only and name not in args.only:
            return
        result['experiments'][name] = paired(before, after, check)
        print(name, json.dumps(result['experiments'][name]['summary']), flush=True)
    def masks_equal(a, b):
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    def rows_equal(a, b):
        pd.testing.assert_frame_equal(pd.DataFrame(a[0]), pd.DataFrame(b[0]), check_exact=True)
        assert a[1] == b[1]
    for shape in ('balanced', 'ladder'):
        size = 2048
        parts = ['L{}:1'.format(i) for i in range(size)]
        if shape == 'balanced':
            while len(parts) > 1:
                parts = ['(' + ','.join(parts[i:i + 2]) + '):1' for i in range(0, len(parts), 2)]
        else:
            text = parts[0]
            for part in parts[1:]:
                text = '(' + text + ',' + part + '):1'
            parts = [text]
        tr = tree.add_numerical_node_labels(ete.PhyloNode(parts[0] + ';', format=1))
        states = np.zeros((2 * size - 1, 64, 4))
        rng = np.random.default_rng(42)
        for leaf in ete.iter_leaves(tr):
            states[int(ete.get_prop(leaf, 'numerical_label')), :, 0] = rng.random(64) > .2
        record('missingness_' + shape, lambda: old['parser_iqtree'].get_internal_site_nonzero_masks(tr, states),
               lambda: parser_iqtree.get_internal_site_nonzero_masks(tr, states), masks_equal)
        record('inspect_' + shape, lambda: old['main_inspect']._collect_branch_rows(tr),
               lambda: main_inspect._collect_branch_rows(tr), rows_equal)
        def clades_equal(a, b):
            for key in ('branch_ids', 'size', 'is_fg_stem'):
                np.testing.assert_array_equal(a[key], b[key])
            owner = b['descendant_indices_by_index'].index
            linear = sum(value.nbytes for value in vars(owner).values() if isinstance(value, np.ndarray))
            result['clade_payload'][shape] = dict(
                before_numpy=sum(value.nbytes for key in ('descendant_indices_by_index', 'descendant_branch_ids_by_index')
                                 for value in a[key]), after_numpy=linear,
                listing_cache_limit=8 * 1024**2)
            for index in range(len(a['branch_ids'])):
                for key in ('descendant_indices_by_index', 'descendant_branch_ids_by_index'):
                    np.testing.assert_array_equal(a[key][index], b[key][index])
                assert a['leaf_names_by_index'][index] == b['leaf_names_by_index'][index]
        record('clades_' + shape, lambda: old['foreground']._build_trait_clade_permutation_cache({'tree': tr}, 'audit'),
               lambda: foreground._build_trait_clade_permutation_cache({'tree': tr}, 'audit'), clades_equal)
        def partitions_equal(a, b):
            names = sorted(set(ete.get_leaf_names(tr)))
            converted = {frozenset(frozenset(name for index, name in enumerate(names) if part & (1 << index))
                                   for part in parts): node.name for node, parts in b.items()}
            assert a == converted
        record('root_partitions_' + shape, lambda: old['tree']._internal_node_partition_names(tr),
               lambda: tree._internal_node_partition_bits(tr), partitions_equal)
    rng = np.random.default_rng(91)
    events = pd.DataFrame(dict(site=rng.integers(0, 300, 4000), from_state_id=rng.integers(0, 20, 4000),
                               to_state_id=rng.integers(0, 20, 4000), event_pp=rng.uniform(.5, 1., 4000)))
    for selected in ('any2spe', 'all'):
        matches = substitution_scan.normalize_scan_matches(selected)
        record('candidates_' + selected,
               lambda: old['substitution_scan'].build_candidates(events, matches, np.arange(20)),
               lambda: substitution_scan.build_candidates(events, matches, np.arange(20)),
               lambda a, b: pd.testing.assert_frame_equal(a, b, check_exact=True))
    state_path = Path('csubst/dataset/PEPC.alignment.fa.state')
    columns = parser_iqtree._read_state_header_columns(state_path)[3:]
    names = sorted({row[0] for row in parser_iqtree._iter_state_rows(state_path)})
    nsite = max(int(row[1]) for row in parser_iqtree._iter_state_rows(state_path))
    arrays = [np.zeros((len(names), nsite, len(columns))) for _ in range(2)]
    node_ids = {name: index for index, name in enumerate(names)}
    def parse(module, array):
        loaded = module._load_internal_state_rows_one_pass(array, state_path, columns, nsite, node_ids,
                                                          set(names), set(names), np.float64)
        return loaded, array
    def states_equal(a, b):
        assert a[0] == b[0]
        np.testing.assert_array_equal(a[1], b[1])
    record('pepc_state_parser', lambda: parse(old['parser_iqtree'], arrays[0]),
           lambda: parse(parser_iqtree, arrays[1]), states_equal)
    result['state_input_sha256'] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    n, sites = 80, 256
    counts = rng.poisson(.25, size=(n, sites)).astype(float)
    context, features = rng.normal(size=(n, 2)), rng.normal(size=(sites, 2))
    masks = rng.random((n, sites)) > .1
    counts[~masks] = 0
    groups, rows = np.repeat(np.arange(5), n // 5), np.arange(n)
    candidates = epistasis.candidate_grid(dict(epistasis_beta_auto=True, epistasis_joint_auto=True, epistasis_clip_auto=True))
    def same(a, b):
        assert a == b
    record('structure_inner_cv', lambda: old['epistasis']._select(counts, context, features, masks, groups, rows, candidates),
           lambda: epistasis._select(counts, context, features, masks, groups, rows, candidates), same)
    frame = pd.DataFrame({'score_rate_enrichment': np.arange(10000, dtype=float)})
    record('maximum_score_10000', lambda: old['scan_statistics'].maximum_score(frame),
           lambda: scan_statistics.maximum_score(frame), same)
    args.result.write_text(json.dumps(result, indent=2) + '\n')


def cli():
    root = Path(__file__).resolve().parents[1]
    helper = load(root / 'memory_optimization_20260911/benchmark.py', 'memory_benchmark')
    harness = helper.harness
    harness.output_summary, harness.compare = helper.output_summary, helper.compare_all
    harness.CASES.update({
        'pepc_scan_all': ('PEPC', 'scan', 'defaults', {'scan_match': 'all'}),
        'pepc_fixed10': ('PEPC', 'scan', 'defaults', {'scan_pvalue_calibration': 'candidate_fixed', 'scan_n_permutations': '10'}),
        'pepc_marginal': ('PEPC', 'search', 'defaults', {'substitution_posterior': 'marginal'}),
        'pgk_scan_parallel20': ('PGK', 'scan', 'defaults', {
            'scan_pvalue_calibration': 'full_scan', 'scan_n_permutations': '20', 'threads': '2'}),
    })
    harness.main()


def cache_diagnostic():
    import contextlib
    from csubst import cli, substitution_scan as scan
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=False)
    harness = load(Path(__file__).resolve().parents[1] / 'resource_audit_20260911/benchmark.py', 'diagnostic_harness')
    harness.CASES['all'] = ('PEPC', 'scan', 'defaults', {'scan_match': 'all'})
    build_original, obtain_original = scan._build_scan_static_context, scan._CandidateCache.obtain
    seen = {}
    def force_caching(*values, **options):
        result = build_original(*values, **options)
        result['_cache_candidates'] = True
        return result
    def counted(self, key, build, size_of):
        stats = seen.setdefault(id(self), dict(calls=0, hits=0, retained_bytes=0))
        stats['calls'] += 1
        stats['hits'] += int(key in self.entries)
        result = obtain_original(self, key, build, size_of)
        stats['retained_bytes'] = self.nbytes
        return result
    scan._build_scan_static_context, scan._CandidateCache.obtain = force_caching, counted
    command = harness.command_for('all', 'after', Path.cwd(), Path('csubst/dataset'), args.workdir / 'output')
    try:
        with (args.workdir / 'stdout.log').open('w') as log, contextlib.redirect_stdout(log):
            assert cli.main(command[3:]) in (0, None)
    finally:
        scan._build_scan_static_context, scan._CandidateCache.obtain = build_original, obtain_original
    result = dict(case='PEPC all matches, no calibration', caches=list(seen.values()),
                  note='Diagnostic forces the pre-refinement always-cache policy only inside this process.')
    args.result.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    if mode == 'phases':
        phases()
    elif mode == 'cli':
        cli()
    elif mode == 'check-results':
        helper = load(Path(__file__).resolve().parents[1] / 'efficiency_20261015/benchmark.py', 'earlier_benchmark')
        helper.check_cli_result(Path(sys.argv[1]))
    elif mode == 'cache-diagnostic':
        cache_diagnostic()
    else:
        raise SystemExit('Use phases, cli, check-results or cache-diagnostic')
