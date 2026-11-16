"""Exact observed any2spe totals with a batched contraction at each common ancestor."""

from collections import Counter, defaultdict
from itertools import groupby
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from csubst import endpoint, endpoint_io, ete, fitted_model, genetic_code, parser_iqtree, parser_misc, sequence, tree
from endpoint_posterior import ConditionalEdges, pair_paths, pair_probabilities


PAIR = ['branch_id_1', 'branch_id_2']
COUNTS = ['OCNany2spe', 'OCSany2spe']


def make_plan(parents, pairs):
    split_paths = defaultdict(list)
    for i, pair in enumerate(pairs):
        ancestor, a, b = pair_paths(parents, pair)
        # Each disjoint child-subtree pair is one weighted cross product.
        if a[-1] > b[-1]:
            a, b = b, a
        split_paths[ancestor, a[-1], b[-1]].append((i, a, b))
    groups = []
    for (ancestor, _, _), split in split_paths.items():
        left, right = [sorted(set(paths)) for paths in zip(*[(a, b) for _, a, b in split])]
        left_index = {path: i for i, path in enumerate(left)}
        right_index = {path: i for i, path in enumerate(right)}
        requested = np.array([(i, left_index[a], right_index[b]) for i, a, b in split])
        groups.append((ancestor, left, right, requested.T))
    return groups


def exact_counts(model, tips, amino, pairs, block_size, *, release=True):
    """Stream site blocks and accumulate, with no per-pair site tensor."""
    groups = make_plan(model.parents, pairs)
    if release:
        order = {node: i for i, node in enumerate(model.order)}
        groups.sort(key=lambda group: order[group[0]], reverse=True)
    all_paths = [path for _, left, right, _ in groups for path in (*left, *right)]
    prefixes = {path[:i] for path in all_paths for i in range(2, len(path) + 1)}
    uses = Counter(all_paths)
    uses.update(path[:-1] for path in prefixes)
    focal = {path[0] for path in all_paths}
    internal = {edge for path in all_paths for edge in path[1:]}
    required_nodes = {ancestor for ancestor, *_ in groups}
    transform = ConditionalEdges(len(model.rates))
    membership = np.eye(amino.max() + 1)[amino]
    nonsyn = amino[:, None] != amino
    syn = ~nonsyn & ~np.eye(len(amino), dtype=bool)
    sections = (slice(None, membership.shape[1]), slice(membership.shape[1], None))
    totals = np.zeros((len(pairs), 2))
    records = model.iter_blocks(
        tips, block_size=block_size, branch_ids=focal | internal, joint=False,
        transform=transform, transform_rate_context=True, category_nodes=True,
    )
    for _, block in groupby(records, key=lambda record: record.start):
        posteriors, edges, direct = {}, {}, {}
        for record in block:
            if record.child in required_nodes:
                posteriors[record.child] = np.sqrt(record.category_node)
            if record.child in focal or record.child in internal:
                kernel = np.stack([record.reduced[i] for i in range(len(model.rates))])
                if record.child in focal:
                    direct[record.child] = np.concatenate(((kernel * nonsyn) @ membership, kernel * syn), axis=-1)
                if record.child in internal:
                    edges[record.child] = kernel

        cached = {}
        remaining = uses.copy()

        def projected(path):
            if len(path) == 1:
                return direct[path[0]]
            if path not in cached:
                cached[path] = edges[path[-1]] @ projected(path[:-1])
            if release:
                remaining[path] -= 1
                if remaining[path] == 0:
                    return cached.pop(path)
            return cached[path]

        for ancestor, left, right, (target, a, b) in groups:
            weight = posteriors[ancestor][None, :, :, :, None]
            x, y = (np.stack([projected(path) for path in side]) * weight for side in (left, right))
            for j, section in enumerate(sections):
                lflat = np.ascontiguousarray(x[..., section]).reshape(len(x), -1)
                rflat = np.ascontiguousarray(y[..., section]).reshape(len(y), -1)
                totals[target, j] += (lflat @ rflat.T)[a, b]
        del x, y, lflat, rflat
    return totals


def model_and_pairs():
    codon_table = genetic_code.get_codon_table(1)
    g = {
        'alignment_file': 'inputs/sim.fa', 'float_type': np.float64, 'codon_table': codon_table,
        **{f'path_iqtree_{ext}': 'inputs/fit.' + ext for ext in ('state', 'log', 'iqtree')},
    }
    parser_iqtree.read_state_header(g)
    parser_iqtree.read_iqtree(g)
    parser_iqtree.read_log(g)
    codons = g['codon_orders']
    g['exchangeability_matrix'] = parser_misc.read_exchangeability_matrix(
        'substitution_matrix/ECMunrest.dat', codons,
    )
    fitted_model.prepare(g)
    q, pi = g['instantaneous_codon_rate_matrix'], g['equilibrium_frequency']
    metadata = json.loads(Path('search-0/csubst_endpoint_model.json').read_text())['codon']
    digest = hashlib.sha256()
    for value in (q, pi):
        digest.update(np.asarray(value, dtype='<f8').tobytes())
    if digest.hexdigest() != metadata['parameter_source']['generator_sha256']:
        raise ValueError('The fitted generator differs from the search model.')
    tr = tree.add_numerical_node_labels(ete.PhyloNode('inputs/tree.nwk', format=1))
    parents, _ = endpoint_io._tree_arrays({'tree': tr}, structural=False)
    model = endpoint.EndpointModel(parents, metadata['branch_lengths'], q, pi, metadata['rates'], metadata['weights'])
    seq = sequence.read_fasta('inputs/sim.fa')
    lookup = {codon: i for i, codon in enumerate(codons)}
    eye = np.eye(len(codons))
    tips = {}
    for leaf in ete.iter_leaves(tr):
        code = seq[leaf.name]
        idx = int(ete.get_prop(leaf, 'numerical_label'))
        tips[idx] = eye[[lookup[code[i:i+3]] for i in range(0, len(code), 3)]]
    translation = {codon: aa for aa, codon in codon_table}
    amino = np.unique([translation[c] for c in codons], return_inverse=True)[1]
    search = pd.read_csv('search-0/csubst_cb_2.tsv', sep='\t')
    return model, tips, amino, search


def benchmark(spec):
    model, tips, amino, search = model_and_pairs()
    pairs = search[PAIR].to_numpy()
    if spec['mode'] == 'targeted':
        rng = np.random.default_rng(940)
        pairs = pairs[np.sort(rng.choice(len(pairs), 64, replace=False))]
    started = time.perf_counter()
    if spec['mode'] in {'optimized', 'composed'}:
        from pair_counts_compact import compact_counts
        values = compact_counts(model, tips, amino, pairs, spec['block_size'], compose=spec['mode'] == 'composed')
    else:
        values = exact_counts(model, tips, amino, pairs, spec['block_size'], release=spec['mode'] != 'exact')
    print(json.dumps(dict(kernel_s=time.perf_counter()-started, pairs=len(pairs), sites=len(next(iter(tips.values()))))))
    if not np.isfinite(values).all():
        raise ValueError('Exact counts contain nonfinite values.')
    # Compare the fast path against the independently verified per-pair
    # algorithm in a separate worker, outside the timed kernel.
    result = pd.DataFrame(pairs, columns=PAIR).assign(**dict(zip(COUNTS, values.T)))
    result.to_csv(spec['outdir'] + '.tsv', sep='\t', index=False, float_format='%.12g')


def check_replay(output):
    model, tips, amino, search = model_and_pairs()
    saved = pd.read_csv(output, sep='\t')
    sample = saved.sample(min(len(saved), 64), random_state=941)
    pairs = sample[PAIR].to_numpy()
    direct = pair_probabilities(model, tips, amino, pairs).sum(axis=2)
    expected = sample.merge(search[PAIR+COUNTS], on=PAIR, validate='one_to_one', suffixes=('_exact', '_search'))
    for i, suffix in enumerate(('search', 'exact')):
        np.testing.assert_allclose(direct[2*i], expected[[col+'_'+suffix for col in COUNTS]],
                                   rtol=1e-9, atol=6e-12)
    differences = (direct[2] - direct[0]).max(axis=0)
    print(json.dumps(dict(pairs=len(pairs), max_n_difference=differences[0], max_s_difference=differences[1])))


def check_saved_totals(reference, observed):
    expected, actual = (pd.read_csv(path, sep='\t').set_index(PAIR, verify_integrity=True).sort_index()
                        for path in (reference, observed))
    pd.testing.assert_index_equal(actual.index, expected.index)
    np.testing.assert_allclose(actual[COUNTS], expected[COUNTS], rtol=1e-9, atol=6e-12)
    differences = (actual[COUNTS] - expected[COUNTS]).abs().max()
    print(json.dumps(dict(pairs=len(actual), max_abs_difference=differences.to_dict())))
