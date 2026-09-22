"""Model/input and sparse-output adapters for joint endpoint inference."""

from collections import OrderedDict
from contextlib import ExitStack
import json
import hashlib
import re
import tempfile

import numpy as np
from scipy import sparse

from csubst import endpoint, ete, fitted_model, runtime, substitution_sparse


BASE_STATS = ('any2any', 'spe2any', 'any2spe', 'spe2spe')


def enabled(g):
    return (g.get('subcommand') != 'scan'
            and str(g.get('substitution_posterior', 'marginal')).lower() == 'joint')


def validate_codon_model(name):
    pattern = r'(?:ECMK07|ECMrest|GY)(?:\+(?:F(?:O|Q|1X4|3X4)?|G\d*|R\d*|I))*'
    mg_pattern = r'MGK?(?:\+(?:F1X4|F3X4|G\d*|R\d*|I))*'
    if not re.fullmatch(pattern, name) and not re.fullmatch(mg_pattern, name):
        raise ValueError('Joint codon endpoints support ECMK07, ECMrest, GY and MG/MGK with counted F1X4/F3X4; unsupported modifiers and mixtures of Q matrices require their own verified likelihood models.')


def model_rates(g):
    """Fitted category rates/priors, never posterior mean rates from .rate."""
    name = str(g.get('substitution_model', ''))
    validate_codon_model(name)
    path = g.get('path_iqtree_iqtree')
    if path:
        with open(path) as handle:
            return read_rate_mixture(handle.read())
    if re.search(r'\+(?:G\d*|R\d*|I)(?:\+|$)', name):
        raise ValueError('Rate mixtures require the fitted IQ-TREE rate-category table.')
    values = np.asarray(g.get('iqtree_rate_values', [1.]), dtype=float)
    if not np.all(values == 1):
        raise ValueError('Uniform endpoint models require unit site rates.')
    return np.ones(1), np.ones(1)


def validate_options(g):
    method = str(g.get('substitution_posterior', 'marginal')).lower()
    if method not in ('marginal', 'joint'):
        raise ValueError('--substitution_posterior must be marginal or joint.')
    size = g.get('endpoint_block_size', 64)
    if isinstance(size, bool) or int(size) != size or size < 1:
        raise ValueError('--endpoint_block_size must be a positive integer.')
    if not enabled(g):
        return
    if str(g.get('ml_anc', False)).lower() in ('true', 'yes', '1'):
        raise ValueError('--substitution_posterior joint requires --ml_anc no.')
    if g.get('nonsyn_recode') == '3di20' and g.get('sa_asr_mode', 'direct') != 'direct':
        raise ValueError('Joint 3Di endpoints require direct ASR with fitted uniform GTR.')


def read_rate_mixture(report):
    """Read the fitted category distribution, never posterior-mean site rates.

    IQ-TREE text reports round parameter values. Normalize rounding in weights;
    retain reported rates and explicitly record this precision in the manifest.
    Missing/unsupported mixture information must not fall back to uniform rates.
    """
    if re.search(r'Model of rate heterogeneity:\s*Uniform\b', report, re.I):
        return np.ones(1), np.ones(1)
    header = re.search(r'Category\s+Relative_rate\s+Proportion\s*\n', report)
    # IQ-TREE omits the category table for a pure +I model when its fitted
    # invariant proportion is zero. This is the unit-rate model, although
    # the report still calls it "Invar". Do not discard a positive invariant
    # component or mistake Invar+Gamma/FreeRate for this degenerate case.
    if header is None and re.search(r'^Model of rate heterogeneity:\s*Invar\s*$', report, re.M | re.I):
        proportion = re.search(r'^Proportion of invariable sites:\s*(\S+)\s*$', report, re.M | re.I)
        if proportion is not None:
            try:
                invariant = float(proportion[1])
            except ValueError:
                invariant = np.nan
            if invariant == 0:
                return np.ones(1), np.ones(1)
    if header is None:
        raise ValueError('Joint endpoints require an IQ-TREE uniform model or rate-category table.')
    rows = []
    for line in report[header.end():].splitlines():
        match = re.fullmatch(r'\s*\d+\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*', line)
        if match is None:
            break
        rows.append([float(match[1]), float(match[2])])
    if not rows:
        raise ValueError('Empty IQ-TREE rate-category table for joint endpoints.')
    values = np.array(rows)
    if (not np.isfinite(values).all() or np.any(values < 0)
            or np.any(values[:, 1] == 0) or abs(values[:, 1].sum() - 1) > 0.002):
        raise ValueError('Invalid or insufficiently precise IQ-TREE rate-category table.')
    return values[:, 0], values[:, 1] / values[:, 1].sum()


class _Spool:
    """Stream unordered CSR row fragments to disk; allocate final payload once."""

    def __init__(self, stack, rows, columns, dtype):
        self.rows, self.columns, self.dtype = rows, columns, np.dtype(dtype)
        self.index_dtype = np.dtype(np.int32 if columns < 2**31 else np.int64)
        self.indices = stack.enter_context(tempfile.TemporaryFile())
        self.values = stack.enter_context(tempfile.TemporaryFile())
        self.fragments = []
        self.counts = np.zeros(rows, dtype=np.int64)
        self.offset = 0

    def append(self, row, block, start, num_site):
        # block is site x event (or site x group x from x to).
        values = np.asarray(block).reshape(block.shape[0], -1).T
        event, site = np.nonzero(values)
        if not site.size:
            return
        indices = (event * num_site + start + site).astype(self.index_dtype)
        payload = values[event, site].astype(self.dtype)
        indices.tofile(self.indices)
        payload.tofile(self.values)
        self.fragments.append((int(row), self.offset, site.size))
        self.offset += site.size
        self.counts[row] += site.size

    def finish(self):
        indptr = np.r_[0, np.cumsum(self.counts)]
        index: np.ndarray = np.empty(self.offset, dtype=self.index_dtype)
        data = np.empty(self.offset, dtype=self.dtype)
        cursor = indptr[:-1].copy()
        for row, offset, count in self.fragments:
            sl = slice(cursor[row], cursor[row] + count)
            self.indices.seek(offset * self.index_dtype.itemsize)
            self.values.seek(offset * self.dtype.itemsize)
            index[sl] = np.fromfile(self.indices, dtype=self.index_dtype, count=count)
            data[sl] = np.fromfile(self.values, dtype=self.dtype, count=count)
            cursor[row] += count
        result = sparse.csr_matrix((data, index, indptr), shape=(self.rows, self.columns), copy=False)
        result.sort_indices()
        return result



class _PairwiseAccumulator:
    """Accumulate a branch Gram matrix, retaining one site block of features."""

    def __init__(self, rows, block_size, feature_indices):
        self.features = np.asarray(feature_indices, dtype=int)
        self.block = np.zeros((rows, block_size, len(self.features)))
        self.gram = np.zeros((rows, rows))
        self.start = None

    def _flush(self):
        if self.start is not None and self.features.size:
            matrix = self.block.reshape(self.block.shape[0], -1)
            self.gram += matrix @ matrix.T

    def append(self, row, block, start, num_site):
        if start != self.start:
            self._flush()
            self.block.fill(0)
            self.start = start
        values = np.asarray(block).reshape(block.shape[0], -1)
        self.block[row, :len(values)] = values[:, self.features]

    def finish(self):
        self._flush()
        self.start = None
        return self.gram



def _pairwise_storage_bound(n, num_site, shapes, kinds, stats, block_size, g):
    observed = set(stats) | {'any2any'}
    streams = [(kind, stat) for kind in kinds for stat in observed]
    streams += [(kind, stat) for kind in kinds if kind != 'AA' for stat in stats]
    features = sum(len(_pairwise_features(kind, stat, shapes[kind], g)) for kind, stat in streams)
    return 8 * ((len(streams) + 1) * n * n + n * min(block_size, num_site) * features
                + len(kinds) * n * num_site * (3 if g.get('b', False) else 1))

def _pairwise_features(kind, stat, shape, g):
    ng, ns = shape[2:4]
    if kind != 'S':
        if stat == 'spe2spe':
            return [a * ns + d for a in range(ns) for d in range(ns) if a != d]
        return np.arange(ng if stat == 'any2any' else ng * ns)
    # Omit padding and singleton synonymous groups, whose change mass is zero.
    sizes = [len(g['synonymous_indices'][aa]) for aa in g['amino_acid_orders']]
    if stat == 'any2any':
        return [sg for sg, size in enumerate(sizes) if size > 1]
    if stat == 'spe2spe':
        return [sg * ns * ns + a * ns + d for sg, size in enumerate(sizes)
                for a in range(size) for d in range(size) if a != d]
    return [sg * ns + state for sg, size in enumerate(sizes) if size > 1
            for state in range(size)]

def _tree_arrays(g, structural):
    nodes = list(g['tree'].traverse())
    n = len(nodes)
    labels = {int(ete.get_prop(node, 'numerical_label')) for node in nodes}
    if labels != set(range(n)):
        raise ValueError('Joint endpoints require a complete, contiguous tree node axis.')
    parents: np.ndarray = np.full(n, -1, dtype=int)
    lengths = np.zeros(n)
    for node in nodes:
        idx = int(ete.get_prop(node, 'numerical_label'))
        if not ete.is_root(node):
            parents[idx] = int(ete.get_prop(node.up, 'numerical_label'))
            lengths[idx] = float(g['3di_branch_lengths'][idx] if structural else node.dist)
    return parents, lengths


def _mapping(g, kind, k):
    if kind == '3di':
        return np.eye(k)
    orders = g['amino_acid_orders'] if kind == 'AA' else g['nonsyn_state_orders']
    groups = g['synonymous_indices'] if kind == 'AA' else g['nonsynonymous_indices']
    result = np.zeros((k, len(orders)))
    for i, name in enumerate(orders):
        result[np.asarray(groups[name], dtype=int), i] = 1
    if not np.all(result.sum(axis=1) == 1):
        raise ValueError('Endpoint recoding must partition the codon state space.')
    return result


def _events(raw, kind, g, mapping):
    size = raw.shape[0]
    if kind == 'S':
        result = np.zeros((size, len(g['amino_acid_orders']),
                           g['max_synonymous_size'], g['max_synonymous_size']))
        for sg, aa in enumerate(g['amino_acid_orders']):
            idx = np.asarray(g['synonymous_indices'][aa], dtype=int)
            result[:, sg, :len(idx), :len(idx)] = raw[:, idx[:, None], idx]
    else:
        result = (mapping.T @ raw @ mapping)[:, None, :, :]
    diagonal = np.arange(result.shape[-1])
    result[:, :, diagonal, diagonal] = 0
    return result


def _projections(events):
    return {'any2any': events.sum(axis=(2, 3)),
            'spe2any': events.sum(axis=3),
            'any2spe': events.sum(axis=2),
            'spe2spe': events}



def _pairwise_search(g):
    return (int(g.get('max_arity', 0)) == 2
            and not g.get('site_filter_report', False)
            and int(g.get('fg_clade_permutation', 0)) == 0)


def _use_projected_search(g, selected_stats):
    # These consumers require individual events or additional null summaries.
    # Keep their full, exact representation rather than approximate a missing
    # statistic from a coarser projection.
    return (g.get('subcommand') in ('search', 'analyze', 'benchmark')
            and g.get('expectation_method', 'codon_model') == 'codon_model'
            and not any(g.get(key, False) for key in (
                'cs', 'cbs', 'calc_omega_pvalue', 'asrv_report', 'calibrate_longtail'))
            and not g.get('epistasis_requested', False)
            and g.get('asrv_training_branches', 'all') == 'all'
            and float(g.get('min_sub_pp', 0)) == 0
            and ('spe2spe' not in selected_stats or _pairwise_search(g)))


def _projection_transform(g, kinds, mappings, stats, predictive=False, cache_bytes=32 * 1024**2):
    """Contract only within-S pairs and cross-group N transition marginals.

    The cross-group mask sums positive terms directly, avoiding cancellation
    from subtracting the (often almost unit) unchanged probability. Predictive
    calls require unit right likelihoods and immutable transition arrays.
    """
    syn_indices = [np.asarray(g['synonymous_indices'][aa], dtype=int)
                   for aa in g['amino_acid_orders']] if 'S' in kinds else []
    syn_pairs = [(sg, a, d, int(ca), int(cd))
                 for sg, idx in enumerate(syn_indices)
                 for a, ca in enumerate(idx) for d, cd in enumerate(idx) if a != d]
    pairs: np.ndarray = np.asarray(syn_pairs, dtype=np.int64).reshape(-1, 5)
    syn_axes = pairs.T
    cython_project = getattr(substitution_sparse.substitution_sparse_cy, 'project_endpoint_syn_double', None)
    masks = {kind: (mapping @ mapping.T == 0) for kind, mapping in mappings.items()
             if mapping is not None}
    kernels: OrderedDict = OrderedDict()
    cached_bytes = 0

    def prediction_kernel(kind, transition):
        # EndpointModel transition arrays are immutable during inference. Keep
        # the object with the key so Python cannot reuse its identity in cache.
        nonlocal cached_bytes
        key = (kind, id(transition))
        if key in kernels:
            kernels.move_to_end(key)
            return kernels[key][1:3]
        cross = transition * masks[kind]
        derived = cross @ mappings[kind]
        ancestral = cross.sum(axis=1)
        size = transition.nbytes + derived.nbytes + ancestral.nbytes
        if size <= cache_bytes:
            while kernels and (cached_bytes + size > cache_bytes or len(kernels) >= 1024):
                cached_bytes -= kernels.popitem(last=False)[1][3]
            kernels[key] = (transition, derived, ancestral, size)
            cached_bytes += size
        return derived, ancestral

    def transform(left, right, transition):
        out = {}
        for kind in kinds:
            if kind == 'S' and cython_project is not None:
                total, ancestral, derived = cython_project(
                    left, right, transition, pairs, len(syn_indices),
                    g['max_synonymous_size'], 'spe2any' in stats, 'any2spe' in stats)
                out[kind, 'any2any'] = total
                if 'spe2any' in stats:
                    out[kind, 'spe2any'] = ancestral
                if 'any2spe' in stats:
                    out[kind, 'any2spe'] = derived
            elif kind == 'S':
                events = np.zeros((left.shape[0], len(syn_indices),
                                   g['max_synonymous_size'], g['max_synonymous_size']))
                sg, a, d, ca, cd = syn_axes
                events[:, sg, a, d] = left[:, ca] * transition[ca, cd] * right[:, cd]
                out.update({(kind, stat): value for stat, value in _projections(events).items()
                            if stat in stats})
            else:
                if predictive:
                    # Predictive right likelihoods are one. Contract constant
                    # transition/group axes once instead of at every site block.
                    kernel, ancestral = prediction_kernel(kind, transition)
                    derived = left @ kernel
                else:
                    cross = transition * masks[kind]
                    derived = ((left @ cross) * right) @ mappings[kind]
                out[kind, 'any2any'] = derived.sum(axis=1, keepdims=True)
                if 'any2spe' in stats:
                    out[kind, 'any2spe'] = derived[:, None, :]
                if 'spe2any' in stats:
                    if predictive:
                        values = (left * ancestral) @ mappings[kind]
                    else:
                        values = ((right @ cross.T) * left) @ mappings[kind]
                    out[kind, 'spe2any'] = values[:, None, :]
        return out
    return transform

def _event_transform(g, kinds, mappings):
    """Accumulate only classified events; maxima must follow rate mixing."""
    native = getattr(substitution_sparse.substitution_sparse_cy, 'project_endpoint_events_double', None)
    specs: dict = {}
    for kind in kinds:
        if kind == 'S':
            ng, ns = len(g['amino_acid_orders']), g['max_synonymous_size']
            pairs = [(group * ns * ns + a * ns + d, int(ca), int(cd))
                     for group, aa in enumerate(g['amino_acid_orders'])
                     for a, ca in enumerate(g['synonymous_indices'][aa])
                     for d, cd in enumerate(g['synonymous_indices'][aa]) if a != d]
        else:
            mapping = mappings[kind]
            ng, ns = 1, mapping.shape[1]
            if not np.all((mapping == 0) | (mapping == 1)) or not np.all(mapping.sum(axis=1) == 1):
                specs[kind] = None
                continue
            groups = mapping.argmax(axis=1)
            pairs = [(int(a * ns + d), ca, cd)
                     for ca, a in enumerate(groups) for cd, d in enumerate(groups) if a != d]
        specs[kind] = (np.asarray(pairs, dtype=np.int64).reshape(-1, 3), ng, ns)

    def transform(left, right, transition):
        out = {}
        raw = None
        for kind in kinds:
            spec = specs[kind]
            if native is not None and spec is not None:
                pairs, ng, ns = spec
                events = native(left, right, transition, pairs, ng, ns)
            else:
                if raw is None:
                    raw = left[:, :, None] * transition * right[:, None, :]
                events = _events(raw, kind, g, mappings[kind])
            out[kind, 'events'] = events
        return out
    return transform


def _build(g, structural=False):
    source = g['state_nsy'] if structural else g['state_cdn']
    if source.dtype.kind != 'f':
        raise ValueError('Joint endpoint inference requires floating-point state tensors (--ml_anc no).')
    n, num_site, k = source.shape
    parents, lengths = _tree_arrays(g, structural)
    if n != parents.size:
        raise ValueError('Endpoint state and tree axes differ.')
    if structural:
        from csubst import expectation_3di
        expectation_3di.validate_context(g, n, num_site)
        q, pi = g['3di_q'], g['3di_pi']
        rates, weights = np.ones(1), np.ones(1)
        kinds = ['N']
        mappings: dict[str, np.ndarray | None] = {'N': np.eye(k)}
    else:
        validate_codon_model(str(g.get('substitution_model', '')))
        q = g['instantaneous_codon_rate_matrix']
        pi = g.get('equilibrium_frequency', g.get('empirical_eq_freq'))
        rates, weights = model_rates(g)
        kinds = ['S']
        mappings = {'S': None}
        if g.get('nonsyn_recode', 'no') != '3di20':
            kinds.append('N')
            mappings['N'] = _mapping(g, 'N', k)
        # The amino-acid stream is also used by VESM and codon exposure.
        if g.get('nonsyn_recode', 'no') != 'no':
            kinds.append('AA')
            mappings['AA'] = _mapping(g, 'AA', k)
    model = endpoint.EndpointModel(parents, lengths, q, pi, rates, weights)
    tips = {}
    observed, eligible = fitted_model.observation_masks(g["tree"], source)
    for leaf in model.leaves:
        values = np.array(source[leaf], dtype=float)
        values[~observed[leaf]] = 1
        tips[leaf] = values
    expected = (str(g.get('expectation_method', 'codon_model')) == 'codon_model'
                and g.get('subcommand', 'search') in ('search', 'analyze', 'benchmark'))
    cache = g.setdefault('_endpoint_tensors', {})
    expected_cache = g.setdefault('_endpoint_reducers', {})
    shapes = {}
    aa_mapping = None if structural else _mapping(g, 'AA', k)
    for kind in kinds:
        if kind == 'S':
            ng, ns = len(g['amino_acid_orders']), g['max_synonymous_size']
        else:
            mapping = mappings[kind]
            if mapping is None:
                raise ValueError('Missing endpoint state mapping for ' + kind)
            ng, ns = 1, mapping.shape[1]
        shapes[kind] = (n, num_site, ng, ns, ns)
    from csubst import omega, output_stat
    selected_stats = output_stat.get_required_base_stats(omega._resolve_requested_output_stats(g))
    projected = _use_projected_search(g, selected_stats)
    pairwise = projected and _pairwise_search(g)
    block_size = min(int(g.get('endpoint_block_size', 64)), max(1, num_site))
    observed_stats = set(selected_stats) | {'any2any'}
    if pairwise:
        # A full from/to channel is larger than a marginal channel. Reduce
        # the site workspace before giving up the streaming pair reducer.
        while block_size > 1 and _pairwise_storage_bound(
                n, num_site, shapes, kinds, selected_stats, block_size, g) > 64 * 1024 * 1024:
            block_size = max(1, block_size // 2)
        pairwise = _pairwise_storage_bound(n, num_site, shapes, kinds, selected_stats,
                                           block_size, g) <= 64 * 1024 * 1024
    if 'spe2spe' in selected_stats and not pairwise:
        projected = False
    direct = projected and 'AA' not in kinds and 'spe2spe' not in selected_stats
    retained_branches = g.get('_endpoint_retained_branches') if g.get('subcommand') == 'sites' else None
    if retained_branches is not None:
        retained_branches = frozenset(int(v) for v in retained_branches)
        if projected or float(g.get('min_sub_pp', 0)) != 0:
            raise ValueError('Selected endpoint branches require unthresholded sites events.')
    classified = not direct or g.get('b', False)
    coarse_transform = _projection_transform(g, kinds, mappings, observed_stats) if direct else None
    transform = _event_transform(g, kinds, mappings) if classified else coarse_transform
    totals_transform = (_projection_transform(g, kinds, mappings, {'any2any'})
                        if retained_branches is not None else None)
    predictive_transform = (_projection_transform(g, kinds, mappings, observed_stats, predictive=True)
                            if direct else transform)
    with ExitStack() as stack:
        builders = {kind: _Spool(stack, n, int(np.prod(shape[1:])), source.dtype)
                    for kind, shape in shapes.items()} if not projected else {}
        obuilders = {}
        maxima = {}
        branch_sites = {}
        if retained_branches is not None:
            branch_sites = {kind: np.zeros((n, num_site)) for kind in kinds}

        def builder(kind, stat, count):
            if pairwise:
                return _PairwiseAccumulator(n, block_size, _pairwise_features(kind, stat, shapes[kind], g))
            return _Spool(stack, n, num_site * count, source.dtype)
        if projected:
            for kind, shape in shapes.items():
                ng, ns = shape[2:4]
                features = {'any2any': ng, 'spe2any': ng * ns, 'any2spe': ng * ns,
                            'spe2spe': ng * ns * ns}
                obuilders[kind] = {stat: builder(kind, stat, features[stat])
                                   for stat in observed_stats}
                if pairwise:
                    branch_sites[kind] = np.zeros((n, num_site))
                if g.get('b', False):
                    maxima[kind] = (np.zeros((n, num_site)), np.zeros((n, num_site), dtype=np.int32),
                                    np.zeros((n, num_site), dtype=np.int32))
        ebuilders = {}
        totals = {kind: 0.0 for kind in kinds}
        for kind in kinds:
            if expected and kind != 'AA':
                ng, ns = shapes[kind][2:4]
                features = {'any2any': ng, 'spe2any': ng * ns, 'any2spe': ng * ns, 'spe2spe': ng * ns * ns}
                ebuilders[kind] = {stat: builder(kind, stat, count)
                                   for stat, count in features.items() if stat in selected_stats}
        for record in model.iter_blocks(tips, block_size=block_size, predictive=expected,
                                        transform=transform,
                                        predictive_transform=predictive_transform,
                                        branch_ids=retained_branches, unselected_transform=totals_transform):
            sl = slice(record.start, record.stop)
            node = record.child
            if node not in model.leaves:
                # Keep tip observation likelihoods intact for future filtered
                # reconstructions. Internal marginals include the inserted root.
                pp = record.node * observed[node, sl, None]
                source[node, sl] = pp
                if not structural:
                    g['state_pep'][node, sl] = pp @ aa_mapping
                    if 'N' in mappings:
                        g['state_nsy'][node, sl] = pp @ mappings['N']
            if record.parent < 0:
                continue
            valid = eligible[node, sl]
            if record.joint is not None:
                record.joint[~valid] = 0
            if record.predictive is not None:
                record.predictive[~valid] = 0
            for kind in kinds:
                if retained_branches is not None and node not in retained_branches:
                    branch_sites[kind][node, sl] = record.reduced[kind, 'any2any'].sum(axis=1) * valid
                    continue
                if classified:
                    events = record.reduced[kind, 'events']
                    events[~valid] = 0
                    if projected:
                        projections = _projections(events)
                else:
                    projections = {stat: record.reduced[kind, stat] for stat in observed_stats}
                    for values in projections.values():
                        values[~valid] = 0
                if projected:
                    if pairwise:
                        branch_sites[kind][node, sl] = projections['any2any'].sum(axis=1)
                    for stat, accumulator in obuilders[kind].items():
                        accumulator.append(node, projections[stat], record.start, num_site)
                    if kind in maxima:
                        flat = events.reshape(events.shape[0], -1)
                        index = flat.argmax(axis=1)
                        ns = shapes[kind][-1]
                        maxima[kind][0][node, sl] = flat[np.arange(len(index)), index]
                        maxima[kind][1][node, sl] = (index // ns) % ns
                        maxima[kind][2][node, sl] = index % ns
                else:
                    if retained_branches is not None:
                        branch_sites[kind][node, sl] = events.sum(axis=(1, 2, 3))
                    if retained_branches is None or node in retained_branches:
                        builders[kind].append(node, events, record.start, num_site)
                if kind in ebuilders:
                    if direct:
                        prediction_projections = {stat: record.reduced_predictive[kind, stat]
                                                  for stat in observed_stats}
                        for values in prediction_projections.values():
                            values[~valid] = 0
                    else:
                        prediction = record.reduced_predictive[kind, 'events']
                        prediction[~valid] = 0
                        prediction_projections = _projections(prediction)
                    totals[kind] += float(prediction_projections['any2any'].sum())
                    for stat, accumulator in ebuilders[kind].items():
                        accumulator.append(node, prediction_projections[stat], record.start, num_site)
        for kind in kinds:
            if pairwise:
                cache[kind] = substitution_sparse.PairwiseSubstitutionSummary(
                    shapes[kind], source.dtype,
                    {stat: accumulator.finish() for stat, accumulator in obuilders[kind].items()},
                    branch_sites[kind], maxima.get(kind))
            elif projected:
                cache[kind] = substitution_sparse.ProjectedSubstitutionTensor(
                    shapes[kind], source.dtype,
                    {stat: builder.finish() for stat, builder in obuilders[kind].items()},
                    maxima.get(kind))
            else:
                matrix = builders[kind].finish()
                if retained_branches is not None:
                    cache[kind] = substitution_sparse.SelectedBranchSubstitutionTensor(
                        shapes[kind], source.dtype, matrix, branch_sites[kind], retained_branches)
                else:
                    cache[kind] = substitution_sparse.SparseSubstitutionTensor(
                        shapes[kind], source.dtype, matrix=matrix)
            if kind in ebuilders:
                projections = {stat: builder.finish() for stat, builder in ebuilders[kind].items()}
                expected_cache[kind] = {
                    'projections': {} if pairwise else projections,
                    **({'pairwise': projections} if pairwise else {}),
                    'total': totals[kind],
                    'storage': sum(x.nbytes if pairwise else x.data.nbytes + x.indices.nbytes + x.indptr.nbytes
                                   for x in projections.values()),
                    'mode': 'nsy' if kind == 'N' else 'cdn',
                }
    for kind in kinds:
        cache[kind].eligible = eligible
    if not structural and g.get('nonsyn_recode', 'no') == 'no':
        cache['AA'] = cache['N']
    manifest = g.setdefault('_endpoint_manifest', {})
    manifest['3di' if structural else 'codon'] = {
        'observed_storage': ('selected_branches' if retained_branches is not None else
                             ('pairwise' if pairwise else ('projections' if projected else 'full_events'))),
        'retained_branches': sorted(retained_branches) if retained_branches is not None else None,
        'direct_projection': direct,
        'block_size': block_size,
        'rates': np.asarray(rates).tolist(), 'weights': np.asarray(weights).tolist(),
        'branch_lengths': lengths.tolist(),
        'parameter_source': '3di_checkpoint' if structural else g.get('fitted_model_provenance', 'provided_model_matrix'),
        'missing_events': 'excluded_from_reporting; latent_states_integrated',
    }


def _input_fingerprint(g):
    """Invalidate in-process outputs when model, emissions or projection change."""
    digest = hashlib.sha256()
    def add(value, sparse=False):
        array = np.asarray(value)
        digest.update(str((array.shape, array.dtype, sparse)).encode())
        if sparse:
            # Tip emissions are mostly exact zeros. Hash their lossless packed
            # support and values, avoiding repeated hashing of a dense codon axis.
            support = array != 0
            digest.update(np.packbits(support).tobytes())
            digest.update(np.ascontiguousarray(array[support]).tobytes())
        else:
            digest.update(np.ascontiguousarray(array).tobytes())
    add(g['instantaneous_codon_rate_matrix'])
    add(g.get('equilibrium_frequency', g.get('empirical_eq_freq')))
    for structural in ([False, True] if g.get('nonsyn_recode') == '3di20' else [False]):
        for value in _tree_arrays(g, structural):
            add(value)
        source = g['state_nsy'] if structural else g['state_cdn']
        for node in g['tree'].traverse():
            if ete.is_leaf(node):
                add(source[int(ete.get_prop(node, 'numerical_label'))], sparse=True)
        if structural:
            add(g['3di_q'])
            add(g['3di_pi'])
    for value in model_rates(g):
        add(value)
    for key in ('subcommand', 'expectation_method', 'nonsyn_recode', 'nonsynonymous_indices',
                'synonymous_indices', 'amino_acid_orders', 'nonsyn_state_orders', 'output_stats',
                'b', 'cs', 'cbs', 'max_arity', 'calibrate_longtail', 'min_sub_pp', '_endpoint_retained_branches',
                'calc_omega_pvalue', 'asrv_report', 'epistasis_requested', 'asrv_training_branches',
                'site_filter_report', 'fg_clade_permutation', 'max_synonymous_size'):
        digest.update(str((key, g.get(key))).encode())
    return digest.hexdigest()


def prepare(g):
    if not enabled(g):
        return
    validate_options(g)
    fingerprint = _input_fingerprint(g)
    if '_endpoint_tensors' in g:
        if fingerprint == g.get('_endpoint_input_fingerprint'):
            return
        invalidate(g)
    print('Joint endpoint inference: blocked pruning, block_size={}; no ancestral-history sampling.'.format(
        g.get('endpoint_block_size', 64)), flush=True)
    try:
        _build(g)
        if g.get('nonsyn_recode') == '3di20':
            _build(g, structural=True)
    except Exception:
        invalidate(g)
        raise
    g['_endpoint_input_fingerprint'] = fingerprint
    if g.get('outdir'):
        manifest = dict(g['_endpoint_manifest'])
        manifest.update({'schema_version': 2, 'input_sha256': fingerprint, 'substitution_posterior': 'joint',
                         'quantity': 'endpoint_state_difference',
                         'expectation': 'fitted_parent_and_rate_posterior_predictive',
                         'branch_combinations': 'factorized_edge_scores',
                         'block_size': int(g.get('endpoint_block_size', 64)),
                         'codon_parameter_precision': g.get('fitted_model_provenance', 'provided_model_matrix')})
        with open(runtime.output_path(g, 'endpoint_model.json'), 'w') as handle:
            json.dump(manifest, handle, indent=2)
            handle.write('\n')


def invalidate(g):
    for key in ('_endpoint_tensors', '_endpoint_reducers', '_endpoint_manifest', '_endpoint_input_fingerprint', 'EN_reducer', 'ES_reducer'):
        g.pop(key, None)


def observed_tensor(g, state, mode):
    if mode == 'syn':
        kind = 'S'
    elif state is g.get('state_nsy'):
        kind = 'N'
    elif state is g.get('state_pep'):
        kind = 'AA'
    else:
        raise ValueError('Joint endpoints require the configured codon/AA/recoded state tensors.')
    prepare(g)
    return g['_endpoint_tensors'][kind]


def expected_reducer(g, mode, selected):
    prepare(g)
    kind = 'S' if mode == 'cdn' else 'N'
    if kind not in g['_endpoint_reducers']:
        raise ValueError('Joint endpoint model expectations were not prepared for this analysis.')
    result = dict(g['_endpoint_reducers'][kind])
    key = 'pairwise' if 'pairwise' in result else 'projections'
    result[key] = {stat: result[key][stat] for stat in selected}
    return result


def release_expected(g, kind):
    """Drop the endpoint cache's final reference once no subsequent reuse is needed."""
    if (enabled(g) and g.get('_release_state_after_expected_reducer', False)
            and int(g.get('fg_clade_permutation', 0)) == 0):
        g.get('_endpoint_reducers', {}).pop(kind, None)
