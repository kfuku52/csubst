"""Joint observed convergence counts for two branches under a fixed codon model."""

from collections import Counter, defaultdict
from itertools import groupby

import numpy as np

from csubst.endpoint import _conditional_transition


BranchPath = tuple[int, ...]


def pair_paths(parents, pair):
    paths = []
    for child in pair:
        path = [int(child)]
        while parents[path[-1]] >= 0:
            path.append(int(parents[path[-1]]))
        paths.append(path)
    left, right = paths
    ancestor = next(node for node in left if node in right)
    a = tuple(left[:left.index(ancestor)])
    b = tuple(right[:right.index(ancestor)])
    if not a or not b:
        raise ValueError('Focal branches must lie on separate LCA child paths.')
    return ancestor, a, b


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


def _called_codons(values):
    sites = np.arange(len(values))
    codons = values.argmax(axis=1)
    if np.count_nonzero(values) == len(values) and (values[sites, codons] == 1).all():
        return codons
    return None


class _EdgeEvidence:
    def __init__(self, rate_count):
        self.rate_count = rate_count
        self.index = 0

    def __call__(self, left, right, transition, time, weight):
        category = self.index % self.rate_count
        self.index += 1
        return {category: right.copy()}


def _edge_kernel(transitions, right):
    kernel = transitions[:, None] * right[..., None, :]
    mass = kernel.sum(axis=-1, keepdims=True)
    floor = np.sqrt(np.finfo(float).tiny)
    inverse = np.divide(1., mass, out=np.zeros_like(mass), where=mass >= floor)
    kernel *= inverse
    rare = ((mass > 0) & (mass < floor)).any(axis=(2, 3))
    for category, site in np.argwhere(rare):
        with np.errstate(divide='ignore'):
            kernel[category, site] = _conditional_transition(transitions[category], np.log(right[category, site]))
    return kernel


def _compress_patterns(tips, called):
    """Identical tip patterns have identical posteriors under the shared rate mixture."""
    sites = len(next(iter(tips.values())))
    if called.keys() == tips.keys():
        encoded = np.stack(list(called.values()), axis=1)
        _, first, counts = np.unique(encoded, axis=0, return_index=True, return_counts=True)
        if len(first) < sites:
            return ({leaf: values[first] for leaf, values in tips.items()},
                    {leaf: codes[first] for leaf, codes in called.items()}, counts)
    return tips, called, np.ones(sites)


def _pack(paths, features, labels, weight, channel, sites, destinations):
    terminal_ids = [i for i, path in enumerate(paths) if path[0] in labels]
    dense_ids = [i for i, path in enumerate(paths) if path[0] not in labels]
    rates, site_count, states = weight.shape
    width = destinations.stop - destinations.start
    by_site = weight.transpose(1, 0, 2)
    dense = np.empty((len(dense_ids), site_count, width, rates, states))
    for row, i in enumerate(dense_ids):
        values = features[i][..., destinations, :].transpose(1, 2, 0, 3)
        np.multiply(values, by_site[:, None], out=dense[row])
    terminal = np.empty((site_count, len(terminal_ids), rates, states))
    codons = np.empty((site_count, len(terminal_ids)), dtype=np.intp)
    for row, i in enumerate(terminal_ids):
        values = features[i][..., channel, :].transpose(1, 0, 2)
        np.multiply(values, by_site, out=terminal[:, row])
        codons[:, row] = labels[paths[i][0]][sites, channel]
    return (dense_ids, dense.reshape(len(dense_ids), site_count, width, rates * states),
            terminal_ids, codons, terminal.reshape(site_count, len(terminal_ids), rates * states))


def _cross_terminal_dense(terminal, destinations, dense):
    out = np.zeros((terminal.shape[1], len(dense)))
    by_destination = dense.transpose(1, 2, 0, 3)
    for site, labels in enumerate(destinations):
        order = labels.argsort()
        ordered_labels = labels[order]
        stops: np.ndarray = np.flatnonzero(ordered_labels[1:] != ordered_labels[:-1]) + 1
        for indices in np.split(order, stops):
            out[indices] += terminal[site, indices] @ by_destination[site, labels[indices[0]]].T
    return out


def _cross(left, right):
    li, ld, lt, lc, lv = left
    ri, rd, rt, rc, rv = right
    out = np.empty((len(li) + len(lt), len(ri) + len(rt)))
    if li and ri:
        out[np.ix_(li, ri)] = ld.reshape(len(li), -1) @ rd.reshape(len(ri), -1).T
    if lt and rt:
        dot = lv @ rv.transpose(0, 2, 1)
        out[np.ix_(lt, rt)] = (dot * (lc[:, :, None] == rc[:, None, :])).sum(axis=0)
    if lt and ri:
        out[np.ix_(lt, ri)] = _cross_terminal_dense(lv, lc, rd)
    if li and rt:
        out[np.ix_(li, rt)] = _cross_terminal_dense(rv, rc, ld).T
    return out


def _pack_dense(features, weight, destinations):
    size = destinations.stop - destinations.start
    packed = np.empty((len(features), *weight.shape[:2], size, weight.shape[-1]))
    for row, values in zip(packed, features):
        np.multiply(values[..., destinations, :], weight[..., None, :], out=row)
    return packed.reshape(len(packed), -1)


def _split_terminal_queries(groups, called, parents, order):
    """Move mixed pairs into a downward pass when one side contains only called tips."""
    ordinary, queries = [], defaultdict(list)
    focal, edges, projected_tips = set(), set(), []
    for ancestor, left, right, (target, a, b) in groups:
        left_called = np.array([path[0] in called for path in left])
        right_called = np.array([path[0] in called for path in right])
        if left_called.all() == right_called.all():
            ordinary.append((ancestor, left, right, (target, a, b)))
            continue
        if right_called.all():
            left, right, a, b = right, left, b, a
            right_called = left_called
        keep = right_called[b]
        mixed = ~keep
        probe_ids, positions = np.unique(a[mixed], return_inverse=True)
        probes = [left[i] for i in probe_ids]
        tasks, branches = defaultdict(list), set()
        mixed_targets, dense_ids = target[mixed], b[mixed]
        for j in np.unique(dense_ids):
            matches = dense_ids == j
            path = right[j]
            child = path[0]
            tasks[parents[child]].append((mixed_targets[matches], child, positions[matches]))
            branches.update(path[1:])
            focal.add(child)
        nodes = sorted(branches | {ancestor}, key=order.__getitem__)
        uses = Counter(parents[node] for node in branches)
        queries[ancestor].append((probes, [(node, tasks[node]) for node in nodes], uses))
        projected_tips.extend(probes)
        edges.update(branches)
        if keep.any():
            li, aa = np.unique(a[keep], return_inverse=True)
            ri, bb = np.unique(b[keep], return_inverse=True)
            ordinary.append((ancestor, [left[i] for i in li], [right[i] for i in ri], (target[keep], aa, bb)))
    return ordinary, queries, projected_tips, focal, edges


def _accumulate_terminal_query(probes, nodes, uses, measure, edges, direct, labels, sites, amino_count,
                               parents, totals):
    """Propagate joint mass down the opposite subtree, one destination per event and probe."""
    destination = np.stack([labels[probe[0]][sites] for probe in probes], axis=1) + [0, amino_count]
    s = np.arange(len(destination))[:, None, None]
    root = nodes[0][0]
    mass = {root: measure}
    remaining = uses.copy()
    for node, tasks in nodes:
        if node != root:
            parent = parents[node]
            mass[node] = mass[parent] @ edges[node].swapaxes(-1, -2)
            remaining[parent] -= 1
            if remaining[parent] == 0:
                del mass[parent]
        values = mass[node].reshape(*measure.shape[:2], len(probes), 2, measure.shape[-1])
        for idx, child, pos in tasks:
            match = direct[child][:, s, destination[:, pos], :]
            totals[idx] += np.einsum('cstdk,cstdk->td', values[:, :, pos], match)
        if remaining[node] == 0:
            del mass[node]


def count_any2spe(model, tips, amino, pairs, block_size, *, tip_missing=None, eligible=None):
    """Propagate only supported terminal destinations and stream blocks of site patterns."""
    totals = np.zeros((len(pairs), 2))
    if not len(pairs):
        return totals
    groups = make_plan(model.parents, pairs)
    site_count = len(next(iter(tips.values())))
    missing = {} if tip_missing is None else tip_missing
    called = {leaf: codes for leaf, values in tips.items()
              if (leaf not in missing or not missing[leaf].any()) and (codes := _called_codons(values)) is not None}
    if eligible is None or eligible[np.arange(len(model.parents)) != model.root].all():
        tips, called, multiplicity = _compress_patterns(tips, called)
        eligible = None
        if len(multiplicity) < site_count:
            tip_missing = None  # Pattern compression requires every tip site to be called.
    else:
        multiplicity = np.ones(site_count)
    order = {node: i for i, node in enumerate(model.order)}
    groups, queries, probe_paths, probe_focal, probe_edges = _split_terminal_queries(groups, called, model.parents, order)
    paths = [path for _, left, right, _ in groups for path in (*left, *right)] + probe_paths
    complete_paths = set(paths)
    prefixes = {path[:i] for path in paths if path[0] in called for i in range(2, len(path) + 1)}
    projected = prefixes | {path for path in complete_paths if len(path) > 1}
    uses = Counter(paths)
    focal = {path[0] for path in paths} | probe_focal
    internal = set(probe_edges)

    called = {leaf: codons for leaf, codons in called.items() if leaf in focal}
    labels = {leaf: np.column_stack((amino[codons], codons)) for leaf, codons in called.items()}
    states = np.arange(len(amino))
    rate_count = len(model.rates)
    transitions = {leaf: np.stack([model.transition(leaf, c) > 0 for c in range(rate_count)]) for leaf in called}

    projections: defaultdict[int, defaultdict[tuple[BranchPath, bool], list[tuple[BranchPath, BranchPath]]]]
    projections = defaultdict(lambda: defaultdict(list))
    for path in sorted(projected):
        start = len(path) - 1
        if path[0] not in called:
            while start > 1 and path[:start] not in complete_paths:
                start -= 1
        source, segment = path[:start], path[start:]
        internal.update(segment)
        uses[source] += 1
        projections[model.parents[path[-1]]][segment, path[0] in called].append((path, source))
    groups_by_node = defaultdict(list)
    for group in groups:
        groups_by_node[group[0]].append(group)
    schedule = [(node, projections[node], groups_by_node[node], queries.get(node, ())) for node in reversed(model.order)
                if node in projections or node in groups_by_node or node in queries]
    required_nodes = {ancestor for ancestor, *_ in groups} | queries.keys()

    amino_count = amino.max() + 1
    membership = np.eye(amino_count)[amino]
    nonsyn = amino[:, None] != amino
    syn = ~nonsyn & ~np.eye(len(amino), dtype=bool)
    destinations = slice(0, amino_count), slice(amino_count, amino_count + len(amino))
    selected = (focal - called.keys()) | internal
    precached = {child: np.stack([model.transition(child, c) for c in range(rate_count)]) for child in selected}
    records = model.iter_blocks(
        tips, block_size=block_size, branch_ids=selected, joint=False,
        transform=_EdgeEvidence(rate_count), transform_rate_context=True, category_nodes=True,
        tip_missing=tip_missing,
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
                kernel = _edge_kernel(precached[child], kernel)
                if child in internal:
                    edges[child] = kernel.swapaxes(-1, -2)
                if child in focal:
                    values = np.empty((*kernel.shape[:2], amino_count + len(amino), len(amino)))
                    values[..., destinations[0], :] = ((kernel * nonsyn) @ membership).swapaxes(-1, -2)
                    values[..., destinations[1], :] = (kernel * syn).swapaxes(-1, -2)
                    direct[child] = values
            if eligible is not None and child in direct:
                invalid = ~eligible[child, sites]
                direct[child][:, invalid] = 0

        cached: dict[BranchPath, np.ndarray] = {}
        composite = {}
        remaining = uses.copy()

        def consume(path):
            if len(path) == 1:
                return direct[path[0]]
            remaining[path] -= 1
            if remaining[path] == 0:
                return cached.pop(path)
            return cached[path]

        def transition(segment):
            if len(segment) == 1:
                return edges[segment[0]]
            if segment not in composite:
                composite[segment] = transition(segment[:-1]) @ edges[segment[-1]]
            return composite[segment]

        for ancestor, steps, crosses, probes in schedule:
            for (segment, _), pending in steps.items():
                transform = transition(segment)
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
            for paths_to_probe, nodes, query_uses in probes:
                features = np.concatenate([consume(path) for path in paths_to_probe], axis=-2)
                measure = features * weights[ancestor][..., None, :] ** 2
                _accumulate_terminal_query(paths_to_probe, nodes, query_uses, measure, edges, direct, labels, sites,
                                           amino_count, model.parents, totals)
            for _, left, right, (target, a, b) in crosses:
                x, y = ([consume(path) for path in side] for side in (left, right))
                dense_only = all(path[0] not in called for path in (*left, *right))
                for channel, section in enumerate(destinations):
                    if dense_only:
                        ldata = _pack_dense(x, weights[ancestor], section)
                        rdata = _pack_dense(y, weights[ancestor], section)
                        totals[target, channel] += (ldata @ rdata.T)[a, b]
                    else:
                        ldata = _pack(left, x, labels, weights[ancestor], channel, sites, section)
                        rdata = _pack(right, y, labels, weights[ancestor], channel, sites, section)
                        totals[target, channel] += _cross(ldata, rdata)[a, b]
                    del ldata, rdata
                del x, y
    return totals
