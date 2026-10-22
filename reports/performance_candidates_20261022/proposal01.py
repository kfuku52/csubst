"""Investigation of exact endpoint site-pattern compression, not production code.

Run from repository root: python reports/performance_candidates_20261022/proposal01.py
"""

import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402 - thread limits must precede native imports

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import cli, endpoint, endpoint_io, fitted_model  # noqa: E402


class Captured(Exception):
    pass


def capture(dataset, workdir):
    """Stop the real fixed-fit CLI immediately before endpoint inference."""
    data = ROOT / 'csubst/dataset'
    captured = {}
    original = endpoint_io._build

    def stop(g, structural=False):
        assert not structural
        captured['g'] = g
        raise Captured

    argv = ['search', '--alignment_file', str(data / (dataset + '.alignment.fa')),
            '--rooted_tree_file', str(data / (dataset + '.tree.nwk')),
            '--outdir', str(workdir / dataset), '--threads', '1', '--blas_threads', '1',
            '--max_arity', '2', '--calibrate_longtail', 'no', '--random_seed', '8',
            '--b', 'no', '--s', 'no', '--cs', 'no', '--bs', 'no', '--cbs', 'no']
    for key in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        argv += ['--iqtree_' + key, str(data / (dataset + '.alignment.fa.' + key))]
    endpoint_io._build = stop
    try:
        with (workdir / (dataset + '.log')).open('w') as log, contextlib.redirect_stdout(log):
            try:
                cli.main(argv)
            except Captured:
                pass
    finally:
        endpoint_io._build = original
    g = captured['g']
    state = g['state_cdn']
    parents, lengths = endpoint_io._tree_arrays(g, False)
    q = g['instantaneous_codon_rate_matrix']
    pi = g.get('equilibrium_frequency', g.get('empirical_eq_freq'))
    rates, weights = endpoint_io.model_rates(g)
    model = endpoint.EndpointModel(parents, lengths, q, pi, rates, weights)
    observed, eligible = fitted_model.observation_masks(g['tree'], state)
    tips = {}
    for leaf in sorted(model.leaves):
        values = np.array(state[leaf], dtype=float)
        values[~observed[leaf]] = 1
        tips[leaf] = values
    mapping = endpoint_io._mapping(g, 'N', len(pi))
    return dict(parents=parents, lengths=lengths, q=q, pi=pi, rates=rates, weights=weights,
                tips=tips, eligible=eligible, mapping=mapping)


def patterns(tips):
    leaf_ids = list(tips)
    values = np.ascontiguousarray(np.stack([tips[i] for i in leaf_ids], axis=1))
    flat = values.reshape(len(values), -1)
    keys = flat.view(np.dtype((np.void, flat.dtype.itemsize * flat.shape[1]))).ravel()
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    compressed = {leaf: tips[leaf][first] for leaf in leaf_ids}
    return compressed, first, inverse, values.nbytes + keys.nbytes + first.nbytes + inverse.nbytes


def infer(config, compress):
    original_sites = next(iter(config['tips'].values())).shape[0]
    if compress:
        tips, _, inverse, pattern_bytes = patterns(config['tips'])
    else:
        tips = config['tips']
        inverse = None
        pattern_bytes = 0
    sites = next(iter(tips.values())).shape[0]
    model = endpoint.EndpointModel(config['parents'], config['lengths'], config['q'],
                                   config['pi'], config['rates'], config['weights'])
    n, k, nc = len(config['parents']), len(config['pi']), len(config['rates'])
    groups = config['mapping'].shape[1]
    nodes = np.empty((n, sites, k))
    events = np.zeros((n, sites, groups, groups))
    category = np.empty((n, sites, nc, k)) if nc > 1 else None
    transform = endpoint_io._event_transform({}, ['N'], {'N': config['mapping']})
    for rec in model.iter_blocks(tips, 64, joint=False, transform=transform, category_nodes=nc > 1):
        sl = slice(rec.start, rec.stop)
        nodes[rec.child, sl] = rec.node
        if category is not None:
            category[rec.child, sl] = rec.category_node.transpose(1, 0, 2)
        if rec.parent >= 0:
            events[rec.child, sl] = rec.reduced['N', 'events'][:, 0]
    if compress:
        nodes = nodes[:, inverse]
        events = events[:, inverse]
        if category is not None:
            category = category[:, inverse]
    events *= config['eligible'][:, :, None, None]
    output_bytes = nodes.nbytes + events.nbytes + (0 if category is None else category.nbytes)
    return dict(nodes=nodes, events=events, category=category, original_sites=original_sites,
                unique_sites=sites, pattern_temporary_upper_bytes=pattern_bytes,
                returned_array_bytes=output_bytes)


def check(before, after):
    errors = {}
    for key in ('nodes', 'events', 'category'):
        if before[key] is None:
            assert after[key] is None
            continue
        np.testing.assert_allclose(before[key], after[key], rtol=1e-12, atol=1e-14)
        errors[key] = float(np.max(np.abs(before[key] - after[key]), initial=0))
    return errors


def run_case(config):
    runs = {'before': [], 'compressed': []}
    metadata, errors = {}, {}
    for repeat in range(4):
        results = {}
        for label in (('before', 'compressed') if repeat % 2 == 0 else ('compressed', 'before')):
            gc.collect()
            results[label], timing = measure(infer, config, label == 'compressed')
            if repeat:
                runs[label].append(timing)
        errors = check(results['before'], results['compressed'])
        metadata = {key: value for key, value in results['compressed'].items()
                    if key not in ('nodes', 'events', 'category')}
        del results
    summaries = {key: summarize(values) for key, values in runs.items()}
    return dict(runs=runs, summary=summaries, max_abs_errors=errors, **metadata)


def synthetic():
    rng = np.random.default_rng(20261022)
    parents = np.array([-1, 0, 0, 1, 1, 2, 2])
    pi = np.array([.2, .1, .3, .4])
    q = np.ones((4, 4)) * pi
    np.fill_diagonal(q, 0)
    np.fill_diagonal(q, -q.sum(axis=1))
    base = np.eye(4)[rng.integers(4, size=(4, 11))]
    base[0, 0] = [1, 1, 0, 0]
    base[:, 1] = 1  # Completely missing site.
    base[2, 2] = [.2, .5, .1, .9]  # Exact nonbinary likelihoods must be preserved.
    order = np.array([4, 0, 1, 4, 2, 10, 1, 0, 6, 7, 2, 5, 5, 5, 0])
    tips = {leaf: base[i, order] for i, leaf in enumerate((3, 4, 5, 6))}
    eligible = rng.random((7, len(order))) > .2
    eligible[0] = False
    return dict(parents=parents, lengths=np.array([0, 0, .1, 1e-5, .2, .3, .4]),
                q=q, pi=pi, rates=np.array([0., .2, 1., 2.]),
                weights=np.array([.1, .2, .3, .4]), tips=tips,
                eligible=eligible, mapping=np.eye(4))


def main():
    result = {'environment': environment(), 'cases': {}, 'input_sha256': {},
              'scope': 'Endpoint inference plus full node/event expansion and eligibility; not end-to-end CLI',
              'memory_metric': 'Array payload bytes; peak RSS was not measured'}
    with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-patterns-') as work:
        result['synthetic_equivalence'] = check(infer(synthetic(), False), infer(synthetic(), True))
        for dataset in ('PGK', 'PEPC'):
            config = capture(dataset, Path(work))
            result['cases'][dataset] = run_case(config)
            print(dataset, result['cases'][dataset]['summary'], flush=True)
            for extension in ('alignment.fa', 'tree.nwk', 'alignment.fa.iqtree', 'alignment.fa.state'):
                path = ROOT / 'csubst/dataset' / (dataset + '.' + extension)
                result['input_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            if dataset == 'PEPC':
                repeated = dict(config)
                repeated['tips'] = {leaf: np.tile(values, (4, 1)) for leaf, values in config['tips'].items()}
                repeated['eligible'] = np.tile(config['eligible'], (1, 4))
                result['cases']['PEPC_x4_repeated'] = run_case(repeated)
                print('PEPC_x4_repeated', result['cases']['PEPC_x4_repeated']['summary'], flush=True)
    destination = Path(__file__).with_suffix('.json')
    destination.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
