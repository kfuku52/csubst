"""Previous compact endpoint-pair benchmark implementation."""

from collections import Counter, defaultdict
from itertools import groupby

import numpy as np

from csubst.endpoint_pair import _called_codons, _compress_patterns, _cross, _pack, count_any2spe, make_plan
from endpoint_posterior import ConditionalEdges


def compact_counts(model, tips, amino, pairs, block_size, *, compose=False):
    """Propagate only supported terminal destinations and stream blocks of site patterns."""
    if compose:
        return count_any2spe(model, tips, amino, pairs, block_size)
    totals = np.zeros((len(pairs), 2))
    if not len(pairs):
        return totals
    groups = make_plan(model.parents, pairs)
    called = {leaf: codes for leaf, values in tips.items() if (codes := _called_codons(values)) is not None}
    tips, called, multiplicity = _compress_patterns(tips, called)
    paths = [path for _, left, right, _ in groups for path in (*left, *right)]
    prefixes = {path[:i] for path in paths for i in range(2, len(path) + 1)}
    uses = Counter(paths)
    focal = {path[0] for path in paths}
    internal = set()

    called = {leaf: codons for leaf, codons in called.items() if leaf in focal}
    labels = {leaf: np.column_stack((amino[codons], codons)) for leaf, codons in called.items()}
    states = np.arange(len(amino))
    rate_count = len(model.rates)
    transitions = {leaf: np.stack([model.transition(leaf, c) > 0 for c in range(rate_count)]) for leaf in called}

    projections = defaultdict(lambda: defaultdict(list))
    for path in sorted(prefixes):
        start = len(path) - 1
        source, segment = path[:start], path[start:]
        internal.update(segment)
        uses[source] += 1
        projections[model.parents[path[-1]]][segment, path[0] in called].append((path, source))
    groups_by_node = defaultdict(list)
    for group in groups:
        groups_by_node[group[0]].append(group)
    schedule = [(node, projections[node], groups_by_node[node]) for node in reversed(model.order)
                if node in projections or node in groups_by_node]
    required_nodes = {ancestor for ancestor, *_ in groups}

    amino_count = amino.max() + 1
    membership = np.eye(amino_count)[amino]
    nonsyn = amino[:, None] != amino
    syn = ~nonsyn & ~np.eye(len(amino), dtype=bool)
    destinations = slice(0, amino_count), slice(amino_count, amino_count + len(amino))
    reducer = ConditionalEdges(rate_count)
    selected = (focal - called.keys()) | internal
    records = model.iter_blocks(
        tips, block_size=block_size, branch_ids=selected, joint=False,
        transform=reducer, transform_rate_context=True, category_nodes=True,
    )
    for start, block in groupby(records, key=lambda record: record.start):
        weights, edges, direct = {}, {}, {}
        for record in block:
            sites = slice(start, record.stop)
            child = record.child
            if child in required_nodes:
                weights[child] = np.sqrt(record.category_node * multiplicity[None, sites, None])
            if child in called:
                codons = called[child][sites]
                support = transitions[child][:, :, codons].transpose(0, 2, 1)
                changes = amino[codons, None] != amino
                direct[child] = np.stack((support & changes, support & ~changes & (codons[:, None] != states)),
                                         axis=-2).astype(float)
            elif child in focal or child in internal:
                kernel = np.stack([record.reduced[c] for c in range(rate_count)])
                if child in internal:
                    edges[child] = kernel.swapaxes(-1, -2)
                if child in focal:
                    values = np.empty((*kernel.shape[:2], amino_count + len(amino), len(amino)))
                    values[..., destinations[0], :] = ((kernel * nonsyn) @ membership).swapaxes(-1, -2)
                    values[..., destinations[1], :] = (kernel * syn).swapaxes(-1, -2)
                    direct[child] = values

        cached = {}
        remaining = uses.copy()

        def consume(path):
            if len(path) == 1:
                return direct[path[0]]
            remaining[path] -= 1
            if remaining[path] == 0:
                return cached.pop(path)
            return cached[path]

        for ancestor, steps, crosses in schedule:
            for (segment, _), pending in steps.items():
                transform = edges[segment[0]]
                # All paths in a tile use the same transition along the segment.
                # Limit tile width so a deep subtree cannot retain one enormous array.
                for i in range(0, len(pending), 16):
                    tile = pending[i:i + 16]
                    inputs = [consume(source) for _, source in tile]
                    width = inputs[0].shape[-2]
                    rhs = np.concatenate(inputs, axis=-2) if len(inputs) > 1 else inputs[0]
                    moved = rhs @ transform
                    for j, (path, _) in enumerate(tile):
                        cached[path] = moved[..., j * width:(j + 1) * width, :]
                    del inputs, rhs, moved
            for _, left, right, (target, a, b) in crosses:
                x, y = ([consume(path) for path in side] for side in (left, right))
                for channel, section in enumerate(destinations):
                    ldata = _pack(left, x, labels, weights[ancestor], channel, sites, section)
                    rdata = _pack(right, y, labels, weights[ancestor], channel, sites, section)
                    totals[target, channel] += _cross(ldata, rdata)[a, b]
                    del ldata, rdata
                del x, y
    return totals
