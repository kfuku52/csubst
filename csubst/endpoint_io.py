"""Model/input and sparse-output adapters for joint endpoint inference."""

from contextlib import ExitStack
import json
import re
import tempfile

import numpy as np
from scipy import sparse

from csubst import endpoint, ete, runtime, substitution_sparse


BASE_STATS = ('any2any', 'spe2any', 'any2spe', 'spe2spe')


def enabled(g):
    return str(g.get('substitution_posterior', 'marginal')).lower() == 'joint'


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
        index = np.empty(self.offset, dtype=self.index_dtype)
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


def _tree_arrays(g, structural):
    nodes = list(g['tree'].traverse())
    n = len(nodes)
    labels = {int(ete.get_prop(node, 'numerical_label')) for node in nodes}
    if labels != set(range(n)):
        raise ValueError('Joint endpoints require a complete, contiguous tree node axis.')
    parents = np.full(n, -1, dtype=int)
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
            and 'spe2spe' not in selected_stats)


def _projection_transform(g, kinds, mappings, stats):
    """Contract only within-S pairs and cross-group N transition marginals.

    The cross-group mask sums positive terms directly, avoiding cancellation
    from subtracting the (often almost unit) unchanged probability.
    """
    syn_indices = [np.asarray(g['synonymous_indices'][aa], dtype=int)
                   for aa in g['amino_acid_orders']] if 'S' in kinds else []
    syn_pairs = [(sg, a, d, int(ca), int(cd))
                 for sg, idx in enumerate(syn_indices)
                 for a, ca in enumerate(idx) for d, cd in enumerate(idx) if a != d]
    pairs = np.asarray(syn_pairs, dtype=np.int64).reshape(-1, 5)
    syn_axes = pairs.T
    cython_project = getattr(substitution_sparse.substitution_sparse_cy, 'project_endpoint_syn_double', None)
    masks = {kind: (mapping @ mapping.T == 0) for kind, mapping in mappings.items()
             if mapping is not None}

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
                cross = transition * masks[kind]
                derived = ((left @ cross) * right) @ mappings[kind]
                out[kind, 'any2any'] = derived.sum(axis=1, keepdims=True)
                if 'any2spe' in stats:
                    out[kind, 'any2spe'] = derived[:, None, :]
                if 'spe2any' in stats:
                    out[kind, 'spe2any'] = (((right @ cross.T) * left) @ mappings[kind])[:, None, :]
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
        mappings = {'N': np.eye(k)}
    else:
        model_name = str(g.get('substitution_model', ''))
        if not re.fullmatch(r'(?:ECMK07|ECMrest|GY)(?:\+(?:F(?:O|Q|1X4|3X4)?|G\d*|R\d*|I))*', model_name):
            raise ValueError('Joint codon endpoints currently support ECMK07, ECMrest and GY models; '
                             'unsupported modifiers, MG, model mixtures and ascertainment schemes '
                             'require their own verified likelihood models.')
        q = g['instantaneous_codon_rate_matrix']
        pi = g.get('equilibrium_frequency', g.get('empirical_eq_freq'))
        with open(g['path_iqtree_iqtree']) as handle:
            rates, weights = read_rate_mixture(handle.read())
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
    observed = np.zeros((n, num_site), dtype=bool)
    for leaf in model.leaves:
        values = np.array(source[leaf], dtype=float)
        observed[leaf] = values.sum(axis=1) > 0
        values[~observed[leaf]] = 1
        tips[leaf] = values
    for node in reversed(model.order):
        for child in model.children[node]:
            observed[node] |= observed[child]
    expected = (str(g.get('expectation_method', 'codon_model')) == 'codon_model'
                and g.get('subcommand', 'search') in ('search', 'analyze', 'benchmark'))
    cache = g.setdefault('_endpoint_tensors', {})
    expected_cache = g.setdefault('_endpoint_reducers', {})
    shapes = {}
    aa_mapping = None if structural else _mapping(g, 'AA', k)
    for kind in kinds:
        ng, ns = ((len(g['amino_acid_orders']), g['max_synonymous_size']) if kind == 'S'
                  else (1, mappings[kind].shape[1]))
        shapes[kind] = (n, num_site, ng, ns, ns)
    from csubst import omega, output_stat
    selected_stats = output_stat.get_required_base_stats(omega._resolve_requested_output_stats(g))
    projected = _use_projected_search(g, selected_stats)
    observed_stats = set(selected_stats) | {'any2any'}
    direct = projected and not g.get('b', False) and 'AA' not in kinds
    transform = _projection_transform(g, kinds, mappings, observed_stats) if direct else None
    with ExitStack() as stack:
        builders = {kind: _Spool(stack, n, int(np.prod(shape[1:])), source.dtype)
                    for kind, shape in shapes.items()} if not projected else {}
        obuilders = {}
        maxima = {}
        if projected:
            for kind, shape in shapes.items():
                ng, ns = shape[2:4]
                features = {'any2any': ng, 'spe2any': ng * ns, 'any2spe': ng * ns}
                obuilders[kind] = {stat: _Spool(stack, n, num_site * features[stat], source.dtype)
                                   for stat in observed_stats}
                if g.get('b', False):
                    maxima[kind] = (np.zeros((n, num_site)), np.zeros((n, num_site), dtype=np.int32),
                                    np.zeros((n, num_site), dtype=np.int32))
        ebuilders = {}
        totals = {kind: 0.0 for kind in kinds}
        for kind in kinds:
            if expected and kind != 'AA':
                ng, ns = shapes[kind][2:4]
                features = {'any2any': ng, 'spe2any': ng * ns, 'any2spe': ng * ns, 'spe2spe': ng * ns * ns}
                ebuilders[kind] = {stat: _Spool(stack, n, num_site * count, source.dtype)
                                   for stat, count in features.items() if stat in selected_stats}
        for record in model.iter_blocks(tips, block_size=g.get('endpoint_block_size', 64), predictive=expected, transform=transform):
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
            valid = observed[node, sl] & observed[record.parent, sl]
            if record.joint is not None:
                record.joint[~valid] = 0
            if record.predictive is not None:
                record.predictive[~valid] = 0
            for kind in kinds:
                if direct:
                    projections = {stat: record.reduced[kind, stat] for stat in observed_stats}
                    for values in projections.values():
                        values[~valid] = 0
                else:
                    events = _events(record.joint, kind, g, mappings[kind])
                    projections = _projections(events)
                if projected:
                    for stat, builder in obuilders[kind].items():
                        builder.append(node, projections[stat], record.start, num_site)
                    if kind in maxima:
                        flat = events.reshape(events.shape[0], -1)
                        index = flat.argmax(axis=1)
                        ns = shapes[kind][-1]
                        maxima[kind][0][node, sl] = flat[np.arange(len(index)), index]
                        maxima[kind][1][node, sl] = (index // ns) % ns
                        maxima[kind][2][node, sl] = index % ns
                else:
                    builders[kind].append(node, events, record.start, num_site)
                if kind in ebuilders:
                    if direct:
                        prediction_projections = {stat: record.reduced_predictive[kind, stat]
                                                  for stat in observed_stats}
                        for values in prediction_projections.values():
                            values[~valid] = 0
                    else:
                        prediction = _events(record.predictive, kind, g, mappings[kind])
                        prediction_projections = _projections(prediction)
                    totals[kind] += float(prediction_projections['any2any'].sum())
                    for stat, builder in ebuilders[kind].items():
                        builder.append(node, prediction_projections[stat], record.start, num_site)
        for kind in kinds:
            if projected:
                cache[kind] = substitution_sparse.ProjectedSubstitutionTensor(
                    shapes[kind], source.dtype,
                    {stat: builder.finish() for stat, builder in obuilders[kind].items()},
                    maxima.get(kind))
            else:
                cache[kind] = substitution_sparse.SparseSubstitutionTensor(
                    shapes[kind], source.dtype, matrix=builders[kind].finish())
            if kind in ebuilders:
                projections = {stat: builder.finish() for stat, builder in ebuilders[kind].items()}
                expected_cache[kind] = {
                    'projections': projections,
                    'total': totals[kind],
                    'storage': sum(x.data.nbytes + x.indices.nbytes + x.indptr.nbytes for x in projections.values()),
                    'mode': 'nsy' if kind == 'N' else 'cdn',
                }
    if not structural and g.get('nonsyn_recode', 'no') == 'no':
        cache['AA'] = cache['N']
    manifest = g.setdefault('_endpoint_manifest', {})
    manifest['3di' if structural else 'codon'] = {
        'observed_storage': 'projections' if projected else 'full_events',
        'direct_projection': direct,
        'rates': np.asarray(rates).tolist(), 'weights': np.asarray(weights).tolist(),
        'branch_lengths': lengths.tolist(),
        'parameter_source': '3di_checkpoint' if structural else 'iqtree_report_and_model_matrix',
    }


def prepare(g):
    if not enabled(g):
        return
    validate_options(g)
    if '_endpoint_tensors' in g:
        return
    print('Joint endpoint inference: blocked pruning, block_size={}; no ancestral-history sampling.'.format(
        g.get('endpoint_block_size', 64)), flush=True)
    try:
        _build(g)
        if g.get('nonsyn_recode') == '3di20':
            _build(g, structural=True)
    except Exception:
        invalidate(g)
        raise
    if g.get('outdir'):
        manifest = dict(g['_endpoint_manifest'])
        manifest.update({'schema_version': 1, 'substitution_posterior': 'joint',
                         'quantity': 'endpoint_state_difference',
                         'expectation': 'fitted_parent_and_rate_posterior_predictive',
                         'branch_combinations': 'factorized_edge_scores',
                         'block_size': int(g.get('endpoint_block_size', 64)),
                         'codon_parameter_precision': 'rounded_IQ_TREE_report; conditional_on_parsed_parameters'})
        with open(runtime.output_path(g, 'endpoint_model.json'), 'w') as handle:
            json.dump(manifest, handle, indent=2)
            handle.write('\n')


def invalidate(g):
    for key in ('_endpoint_tensors', '_endpoint_reducers', '_endpoint_manifest', 'EN_reducer', 'ES_reducer'):
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
    result['projections'] = {stat: result['projections'][stat] for stat in selected}
    return result
