"""Investigation-only prototypes. Never install them into production modules.

Run from the checkout with --data-dir csubst/dataset and a fresh --workdir.
Measured phases exclude CLI imports; endpoint mode includes inference.
Prototypes are not release-ready.
"""

import argparse
from collections import Counter, OrderedDict
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

os.environ.update(VECLIB_MAXIMUM_THREADS='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')

import numpy as np
import pandas as pd

from csubst import cli, endpoint, endpoint_io, ete, foreground, main_inspect, parser_iqtree, scan_endpoint
from csubst import scan_statistics, substitution_scan as scan, tree


def paired(before, after, check, repeats=3):
    runs = []
    for repeat in range(repeats + 1):
        variants = [('before', before), ('prototype', after)]
        if repeat % 2:
            variants.reverse()
        outputs = {}
        for label, function in variants:
            wall, cpu = time.perf_counter(), time.process_time()
            output = function()
            elapsed_cpu, elapsed_wall = time.process_time() - cpu, time.perf_counter() - wall
            outputs[label] = output
            runs.append(dict(version=label, warmup=repeat == 0, seconds=elapsed_wall,
                             cpu_seconds=elapsed_cpu))
        check(outputs['before'], outputs['prototype'])
    summary = {}
    for metric in ('seconds', 'cpu_seconds'):
        medians = {label: statistics.median(r[metric] for r in runs
                                          if r['version'] == label and not r['warmup'])
                   for label in ('before', 'prototype')}
        summary[metric] = dict(medians=medians, ratio=medians['prototype'] / medians['before'])
    return dict(runs=runs, summary=summary, compared_pairs=repeats + 1)


def shared_candidate_groups(events, scan_matches, state_orders):
    """Reuse identical summaries for the four grouping shapes of nine matches."""
    if events.empty:
        return pd.DataFrame()
    rows, summaries = [], {}
    for match in scan_matches:
        columns = tuple(scan._candidate_group_columns(match))
        if columns not in summaries:
            records = []
            for _, group in events.groupby(list(columns), sort=True):
                records.append((group, scan._candidate_state_sets('any2any', group, len(state_orders)),
                                scan._state_distribution_label(group, 'from_state_id', state_orders),
                                scan._state_distribution_label(group, 'to_state_id', state_orders),
                                float(group['event_pp'].sum())))
            summaries[columns] = records
        anc, der = match.split('2')
        for group, state_sets, from_distribution, to_distribution, total in summaries[columns]:
            all_from, all_to, observed_from, observed_to = state_sets
            if ((anc == 'spe' and len(observed_from) != 1) or
                    (anc == 'dif' and len(observed_from) < 2) or
                    (der == 'spe' and len(observed_to) != 1) or
                    (der == 'dif' and len(observed_to) < 2)):
                continue
            from_ids = all_from if anc == 'any' else observed_from
            to_ids = all_to if der == 'any' else observed_to
            site = int(group['site'].iloc[0])
            from_label = scan._state_set_label(from_ids, state_orders)
            to_label = scan._state_set_label(to_ids, state_orders)
            rows.append(dict(
                scan_match=match, site=site,
                from_state_ids=','.join(map(str, from_ids.tolist())),
                to_state_ids=','.join(map(str, to_ids.tolist())),
                observed_from_state_ids=','.join(map(str, observed_from.tolist())),
                observed_to_state_ids=','.join(map(str, observed_to.tolist())),
                from_state=from_label, to_state=to_label,
                from_state_distribution=from_distribution, to_state_distribution=to_distribution,
                state_change=scan._format_state_change(from_label, to_label, site + 1, match),
                candidate_event_pp_sum=total))
    return pd.DataFrame(rows)


def inspect_counts(tr):
    counts = {}
    for node in tr.traverse('postorder'):
        children = ete.get_children(node)
        counts[node] = sum(counts[c] for c in children) if children else 1
    nodes = {int(ete.get_prop(n, 'numerical_label')): n for n in tr.traverse()}
    rows = []
    for bid, node in sorted(nodes.items()):
        rows.append(dict(branch_id=bid,
                         parent_branch_id=np.nan if ete.is_root(node) else int(ete.get_prop(node.up, 'numerical_label')),
                         is_root=bool(ete.is_root(node)), is_leaf=bool(ete.is_leaf(node)),
                         node_name='' if node.name is None else str(node.name),
                         branch_length=0. if node.dist is None else float(node.dist),
                         num_descendant_leaves=counts[node]))
    return rows, nodes


def subtree_masks(tr, states):
    nonmissing = {}
    for node in tr.traverse('postorder'):
        if ete.is_leaf(node):
            nonmissing[node] = states[int(ete.get_prop(node, 'numerical_label'))].sum(axis=1) != 0
        else:
            nonmissing[node] = np.logical_or.reduce([nonmissing[c] for c in ete.get_children(node)])
    out = {}
    for node in tr.traverse():
        if ete.is_leaf(node) or ete.is_root(node):
            continue
        groups = list(ete.get_children(node)) + list(ete.get_sisters(node))
        if len(groups) >= 2:
            out[int(ete.get_prop(node, 'numerical_label'))] = np.sum(
                [nonmissing[group] for group in groups], axis=0) >= 2
    return out


def maximum_score(frame):
    if frame.empty:
        return -np.inf
    values = frame['score_rate_enrichment'].to_numpy(dtype=float)
    valid = np.isfinite(values) & (values >= 0)
    values = values.copy()
    for index in np.flatnonzero(~valid):
        values[index] = scan_statistics.calibration_score(frame.iloc[index])
    return float(values.max())


def predictive_events(g, kinds, mappings, fallback, stats):
    """Group constant predictive transition kernels; right likelihood is one."""
    cache = OrderedDict()
    retained = 0
    syn_ids = [np.asarray(g['synonymous_indices'][aa], dtype=int) for aa in g['amino_acid_orders']]
    nsyn = int(g['max_synonymous_size'])
    syn_index = np.zeros((len(syn_ids), nsyn), dtype=int)
    syn_valid = np.zeros_like(syn_index, dtype=bool)
    for group, indices in enumerate(syn_ids):
        syn_index[group, :len(indices)] = indices
        syn_valid[group, :len(indices)] = True
    specs = {}
    for kind in kinds:
        if kind == 'S':
            continue
        mapping = mappings[kind]
        if not np.all((mapping == 0) | (mapping == 1)) or not np.all(mapping.sum(axis=1) == 1):
            specs[kind] = None
            continue
        groups = mapping.argmax(axis=1)
        members = [np.flatnonzero(groups == group) for group in range(mapping.shape[1])]
        index = np.zeros((mapping.shape[1], max(map(len, members))), dtype=int)
        valid = np.zeros_like(index, dtype=bool)
        for group, indices in enumerate(members):
            index[group, :len(indices)] = indices
            valid[group, :len(indices)] = True
        specs[kind] = (index, valid)

    def kernels(transition):
        nonlocal retained
        stats['calls'] += 1
        key = id(transition)
        if key in cache:
            stats['hits'] += 1
            cache.move_to_end(key)
            return cache[key][1]
        values = {}
        for kind in kinds:
            if kind == 'S':
                kernel = transition[syn_index[:, :, None], syn_index[:, None, :]].copy()
                kernel *= syn_valid[:, :, None] & syn_valid[:, None, :]
                kernel[:, np.arange(nsyn), np.arange(nsyn)] = 0
                values[kind] = kernel
            elif specs[kind] is not None:
                index, valid = specs[kind]
                mapping = mappings[kind]
                # Sum positive cross-group transitions directly, without cancellation.
                derived = (transition * (mapping @ mapping.T == 0)) @ mapping
                values[kind] = derived[index] * valid[:, :, None]
        size = transition.nbytes + sum(value.nbytes for value in values.values())
        while cache and retained + size > 32 * 1024**2:
            retained -= cache.popitem(last=False)[1][2]
        if size <= 32 * 1024**2:
            cache[key] = (transition, values, size)
            retained += size
        stats['retained_bytes'] = retained
        return values

    def transform(left, right, transition):
        values, out = kernels(transition), {}
        for kind in kinds:
            if kind == 'S':
                events = left[:, syn_index, None] * values[kind]
            elif specs[kind] is not None:
                index, valid = specs[kind]
                events = np.einsum('sac,acd->sad', left[:, index] * valid, values[kind])[:, None]
            else:
                out[kind, 'events'] = fallback(left, right, transition)[kind, 'events']
                continue
            out[kind, 'events'] = events
        return out
    return transform


def endpoint_experiments(args, result):
    harness = load_harness()
    build_original = endpoint_io._build
    for case in ('pepc_full',):
        def measure(g, structural=False):
            assert not structural
            counters = []
            def before():
                return build_original(g, structural)
            def prototype():
                stats = Counter()
                transform_original = endpoint_io._event_transform
                iter_original = endpoint.EndpointModel.iter_blocks
                transforms = {}
                def factory(config, kinds, mappings):
                    transform = transform_original(config, kinds, mappings)
                    transforms[transform] = predictive_events(config, kinds, mappings, transform, stats)
                    return transform
                def blocks(model, *values, **options):
                    predictive = options.get('predictive_transform')
                    if predictive in transforms:
                        options['predictive_transform'] = transforms[predictive]
                    yield from iter_original(model, *values, **options)
                endpoint_io._event_transform = factory
                endpoint.EndpointModel.iter_blocks = blocks
                try:
                    return build_original(g, structural)
                finally:
                    endpoint_io._event_transform = transform_original
                    endpoint.EndpointModel.iter_blocks = iter_original
                    counters.append(dict(stats))
            def snapshot(function):
                function()
                observed = g['_endpoint_tensors']
                predicted = g['_endpoint_reducers']
                return ({kind: {**{key: value.copy() for key, value in tensor.pairwise.items()},
                                'branch_site': tensor.branch_site.copy()}
                         for kind, tensor in observed.items() if kind != 'AA'},
                        {kind: {key: value.copy() for key, value in values['pairwise'].items()}
                         for kind, values in predicted.items()},
                        json.dumps(g['_endpoint_manifest'], sort_keys=True))
            def check(a, b):
                for before_part, after_part in zip(a[:2], b[:2]):
                    assert before_part.keys() == after_part.keys()
                    for kind in before_part:
                        assert before_part[kind].keys() == after_part[kind].keys()
                        for stat in before_part[kind]:
                            np.testing.assert_allclose(before_part[kind][stat], after_part[kind][stat],
                                                       rtol=1e-12, atol=1e-13)
                assert a[2] == b[2]
            result['experiments']['predictive_kernel_' + case] = paired(
                lambda: snapshot(before), lambda: snapshot(prototype), check)
            result['experiments']['predictive_kernel_' + case]['cache_counters'] = counters
            return build_original(g, structural)
        endpoint_io._build = measure
        command = harness.command_for(case, 'before', Path.cwd(), args.data_dir, args.workdir / case)
        try:
            with (args.workdir / (case + '.log')).open('w') as log, contextlib.redirect_stdout(log):
                assert cli.main(command[3:]) in (0, None)
        finally:
            endpoint_io._build = build_original
        result['commands'][case] = command
        print('done', case, flush=True)


def make_tree(shape, size):
    parts = ['L{}:1'.format(i) for i in range(size)]
    if shape == 'balanced':
        while len(parts) > 1:
            parts = ['(' + ','.join(parts[i:i + 2]) + '):1' for i in range(0, len(parts), 2)]
    else:
        text = parts[0]
        for part in parts[1:]:
            text = '(' + text + ',' + part + '):1'
        parts = [text]
    return tree.add_numerical_node_labels(ete.PhyloNode(parts[0] + ';', format=1))


def synthetic(result):
    for shape in ('balanced', 'ladder'):
        for size in (512, 2048):
            tr = make_tree(shape, size)
            name = '{}_{}'.format(shape, size)
            def check_rows(before, after):
                pd.testing.assert_frame_equal(pd.DataFrame(before[0]), pd.DataFrame(after[0]), check_exact=True)
                assert before[1] == after[1]
            result['experiments']['inspect_' + name] = paired(
                lambda: main_inspect._collect_branch_rows(tr), lambda: inspect_counts(tr), check_rows)
            states = np.zeros((2 * size - 1, 64, 4))
            rng = np.random.default_rng(42)
            for leaf in ete.iter_leaves(tr):
                states[int(ete.get_prop(leaf, 'numerical_label')), :, 0] = rng.random(64) > .2
            def check_masks(before, after):
                assert before.keys() == after.keys()
                for key in before:
                    np.testing.assert_array_equal(before[key], after[key])
            result['experiments']['missingness_' + name] = paired(
                lambda: parser_iqtree.get_internal_site_nonzero_masks(tr, state_tensor=states),
                lambda: subtree_masks(tr, states), check_masks)
            start = time.perf_counter()
            cache = foreground._build_trait_clade_permutation_cache({'tree': tr}, 'audit')
            arrays = cache['descendant_branch_ids_by_index'] + cache['descendant_indices_by_index']
            result['clade_cache'][name] = dict(seconds=time.perf_counter() - start,
                                               numpy_descendant_bytes=sum(a.nbytes for a in arrays),
                                               descendant_entries=sum(len(a) for a in arrays),
                                               leaf_name_references=sum(len(v) for v in cache['leaf_names_by_index']),
                                               interval_integer_payload_bytes=3 * (2 * size - 1) * 8,
                                               note='Interval payload is a design estimate, not an API-equivalent implementation.')
            del cache
            print('done', name, flush=True)
    for size in (1000, 10000):
        frame = pd.DataFrame({'score_rate_enrichment': np.arange(size, dtype=float)})
        result['experiments']['maximum_score_' + str(size)] = paired(
            lambda: scan_statistics.maximum_score(frame), lambda: maximum_score(frame),
            lambda a, b: np.testing.assert_equal(a, b))


def load_harness():
    source = Path(__file__).resolve().parents[1] / 'resource_audit_20260911/benchmark.py'
    spec = importlib.util.spec_from_file_location('cli_harness', source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def cached_scan(g, tensor, rate_tensor, stats):
    """Cache immutable per-fit endpoint exposure and event vectors, 32 MiB LRU each."""
    original_events, original_exposure = scan.extract_candidate_posterior_events, scan_endpoint.expected_events
    caches = {'events': OrderedDict(), 'exposure': OrderedDict()}
    sizes = Counter()

    def obtain(kind, key, function, size_of):
        stats[kind + '_calls'] += 1
        cache = caches[kind]
        if key in cache:
            stats[kind + '_hits'] += 1
            cache.move_to_end(key)
            return cache[key][0]
        value = function()
        size = size_of(value)
        while cache and sizes[kind] + size > 32 * 1024**2:
            sizes[kind] -= cache.popitem(last=False)[1][1]
        if size <= 32 * 1024**2:
            cache[key] = (value, size)
            sizes[kind] += size
        return value

    def events(*args, **kwargs):
        bound = event_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        key = (id(a['sub_tensor']), a['site'], tuple(a['from_ids']), tuple(a['to_ids']),
               a['float_tol'], id(a['projection']))
        return obtain('events', key, lambda: original_events(*args, **kwargs),
                      lambda frame: int(frame.memory_usage(index=True, deep=True).sum()))

    def exposure(context, state_cdn, state_nsy, site, from_ids, to_ids):
        key = (id(context), id(state_cdn), id(state_nsy), site, tuple(from_ids), tuple(to_ids))
        return obtain('exposure', key,
                      lambda: original_exposure(context, state_cdn, state_nsy, site, from_ids, to_ids),
                      lambda values: sum(v.nbytes for v in values))

    import inspect
    event_signature = inspect.signature(original_events)
    scan.extract_candidate_posterior_events, scan_endpoint.expected_events = events, exposure
    try:
        return scan_original(g, tensor, rate_tensor)
    finally:
        scan.extract_candidate_posterior_events, scan_endpoint.expected_events = original_events, original_exposure
        stats.update({kind + '_retained_bytes': size for kind, size in sizes.items()})


scan_original = scan.scan_substitutions


def scan_experiments(args, result):
    harness = load_harness()
    cases = {
        'pepc_all_matches': ('PEPC', 'scan', 'defaults', {'scan_match': 'all'}),
        'pgk_full20': ('PGK', 'scan', 'defaults', {'scan_pvalue_calibration': 'full_scan', 'scan_n_permutations': '20'}),
        'pepc_fixed10': ('PEPC', 'scan', 'defaults', {'scan_pvalue_calibration': 'candidate_fixed', 'scan_n_permutations': '10'}),
    }
    harness.CASES.update(cases)
    for case in cases:
        def measure(g, ON_tensor, rate_ON_tensor=None):
            if case == 'pepc_all_matches':
                captured = []
                original = scan.build_candidates
                def capture(*values, **options):
                    captured.append((values, options))
                    return original(*values, **options)
                scan.build_candidates = capture
                try:
                    frame, units = scan_original(g, ON_tensor, rate_ON_tensor)
                finally:
                    scan.build_candidates = original
                for index, (values, options) in enumerate(captured):
                    result['experiments']['candidate_groups_' + str(index)] = paired(
                        lambda: original(*values, **options),
                        lambda: shared_candidate_groups(*values, **options),
                        lambda a, b: pd.testing.assert_frame_equal(a, b, check_exact=True))
                    events = options.get('events', values[0] if values else None)
                    result['experiments']['candidate_groups_' + str(index)]['input_events'] = len(events)
                return frame, units
            counters = []
            def before():
                out = scan_original(g, ON_tensor, rate_ON_tensor)
                return out, json.dumps(g['scan_calibration_diagnostics'], sort_keys=True)
            def prototype():
                stats = Counter()
                out = cached_scan(g, ON_tensor, rate_ON_tensor, stats)
                counters.append(dict(stats))
                return out, json.dumps(g['scan_calibration_diagnostics'], sort_keys=True)
            def check(before, after):
                for a, b in zip(before[0], after[0]):
                    pd.testing.assert_frame_equal(a, b, check_exact=True)
                assert before[1] == after[1]
            measured = paired(before, prototype, check)
            measured['cache_counters'] = counters
            result['experiments']['static_scan_cache_' + case] = measured
            return scan_original(g, ON_tensor, rate_ON_tensor)
        scan.scan_substitutions = measure
        command = harness.command_for(case, 'before', Path.cwd(), args.data_dir, args.workdir / case)
        try:
            with (args.workdir / (case + '.log')).open('w') as log, contextlib.redirect_stdout(log):
                assert cli.main(command[3:]) in (0, None)
        finally:
            scan.scan_substitutions = scan_original
        result['commands'][case] = command
        print('done', case, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('synthetic', 'scan', 'endpoint'), required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('csubst/dataset'))
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=False)
    result = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                  pandas=pd.__version__, repeats=3, warmups=1, experiments={}, clade_cache={}, commands={})
    if args.mode == 'synthetic':
        synthetic(result)
    elif args.mode == 'scan':
        scan_experiments(args, result)
    else:
        endpoint_experiments(args, result)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    for case, command in result['commands'].items():
        command[0] = 'python'
        result['commands'][case] = [str(value).replace(str(args.data_dir), 'csubst/dataset')
                                   .replace(str(args.workdir), '${WORKDIR}') for value in command]
    args.result.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: record['summary'] for name, record in result['experiments'].items()}, indent=2))


if __name__ == '__main__':
    main()
