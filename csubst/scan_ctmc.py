"""Scaled pruning, exact edge posteriors, and posterior CTMC jump counts.

Stationary codon models with discrete rate categories, conditional on fitted Q/tree.
No sampled ancestral states or products of separately inferred marginals.
"""
from functools import cached_property, lru_cache, partial

import numpy as np
from scipy.linalg import eigh, expm, expm_frechet
from scipy.sparse.csgraph import shortest_path

from csubst import ete

_SCAN_IN_MEMORY_BYTES = 64 * 1024**2
_SHORT_BRIDGE_TAU = .05
_VERY_SHORT_BRIDGE_TAU = .001
_BRIDGE_MAX_UNIFORMIZATION_ORDER = 128


def _normalize(x):
    total = x.sum(axis=-1, keepdims=True)
    if not np.isfinite(x).all() or (x < 0).any() or (total <= 0).any():
        raise ValueError("Zero or invalid likelihood in CTMC pruning.")
    return x / total


def bridge_kernel(q, length, left, right):
    """Unnormalized expected jump masses for batches of boundary messages.

    Adjoint Frechet derivative: Q_ij d[a exp(Qt)b]/dQ_ij, with Q's
    diagonal held fixed. This counts every jump, including return paths.
    """
    out = np.empty((len(left), len(q), len(q)))
    for k, (a, b) in enumerate(zip(left, right)):
        out[k] = q * expm_frechet((q * length).T, np.outer(a, b) * length,
                                  compute_expm=False)
    idx = np.arange(len(q))
    out[:, idx, idx] = 0
    return out


def _bridge_spectral(q, pi):
    if (pi <= 0).any():
        raise ValueError("CTMC bridge requires positive stationary frequencies.")
    root = np.sqrt(pi)
    symmetric = root[:, None] * q / root[None, :]
    if (pi <= 0).any() or not np.allclose(symmetric, symmetric.T, atol=1e-10):
        raise ValueError("CTMC bridge requires positive stationary frequencies and reversible Q.")
    values, vectors = eigh(symmetric)
    return values, vectors / root[:, None], vectors.T * root[None, :]


def _integrals(values, length):
    x = values[:, None] * length
    y = values[None, :] * length
    delta = np.abs(x - y)
    ratio = np.ones_like(delta)
    np.divide(-np.expm1(-delta), delta, out=ratio, where=delta > 1e-14)
    return length * np.exp(np.maximum(x, y)) * ratio


@lru_cache(maxsize=6)
def _quadrature_nodes(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (nodes + 1) * .5, weights * .5


def _bridge_integral_kernel(values, length):
    """Compress the divided-difference matrix with checked quadrature factors."""
    integrals = _integrals(values, length)
    if len(values) < 16 or integrals.min() <= 0:
        return integrals
    scaled = (values * length)[:, None]
    for order in (2, 4, 6, 8, 12, 16):
        if order * 3 >= len(values):
            break
        nodes, weights = _quadrature_nodes(order)
        scale = np.sqrt(length * weights)
        # Each quadrature node contributes a separable exponential product.
        left = np.exp(scaled * nodes) * scale
        right = np.exp(scaled * (1 - nodes)) * scale
        if np.all(np.abs(left @ right.T - integrals) <= 8e-15 * integrals):
            return left, right
    return integrals


class _ShortBridge:
    """Evaluate short-branch jump integrals with nonnegative powers."""

    def __init__(self, q):
        self.rate = -q.diagonal().min()
        self.jump_q = np.where(q > 0, q, 0.)
        np.fill_diagonal(self.jump_q, 0.)
        self.transition = np.eye(len(q))
        if self.rate > 0:
            self.transition += q / self.rate
        self.row_norm = float(self.transition.sum(axis=1).max())
        self.nonnegative = bool((self.transition >= 0).all())
        self.coefficients_at = lru_cache(maxsize=256)(self._coefficients)
        # Each entry holds at most (129 x 129) float64 values.
        self.convolution_at = lru_cache(maxsize=128)(self._convolution)

    def _coefficients(self, length):
        tau = self.rate * length
        weights = np.empty(_BRIDGE_MAX_UNIFORMIZATION_ORDER + 1)
        weights[0] = np.exp(-tau)
        for order in range(1, len(weights)):
            weights[order] = weights[order - 1] * tau / order
        return weights

    def _convolution(self, length, order):
        weights = self.coefficients_at(length)[:order + 1] / np.arange(1, order + 2)
        counts = np.arange(order + 1)
        total = counts[:, None] + counts
        return np.where(total <= order, weights[np.minimum(total, order)], 0.)

    @cached_property
    def support(self):
        edges = self.jump_q > 0
        distances = shortest_path(edges, directed=True, unweighted=True)
        reachable = np.isfinite(distances)
        max_distance = int(np.max(distances, where=reachable, initial=0.))
        return edges, reachable, max(6, 2 * max_distance)

    def mass(self, length, left, right):
        tau_bound = self.rate * length * self.row_norm
        if not 0 < tau_bound <= _SHORT_BRIDGE_TAU or not self.nonnegative:
            return None
        edges, reachable, minimum_order = self.support
        if reachable.all():
            possible = edges
        else:
            source = (left > 0) @ reachable
            target = (right > 0) @ reachable.T
            possible = edges & source[:, :, None] & target[:, None, :]
        forward = [left]
        backward = [right]
        norm = left.sum(axis=1) * right.max(axis=1)
        tolerance = np.finfo(float).eps / 4
        probabilities = self.coefficients_at(length)
        for order in range(minimum_order, len(probabilities), 4):
            while len(forward) <= order:
                forward.append(forward[-1] @ self.transition)
                backward.append(backward[-1] @ self.transition.T)
            forward_array = np.stack(forward, axis=-1)
            evidence = np.einsum('bki,bk,i->b', forward_array, right, probabilities[:order + 1])
            tail = probabilities[order] * self.row_norm**order * tau_bound / (order + 1 - tau_bound)
            error = norm * tail
            if (evidence <= 0).any() or (error > tolerance * evidence).any():
                continue
            # Work with the mean integrand so t does not underflow before
            # conditioning on rare endpoint evidence.
            mean = (forward_array @ self.convolution_at(length, order)) @ np.stack(backward, axis=1)
            too_small = (mean <= 0) | (error[:, None, None] > tolerance * mean)
            if (possible & too_small).any():
                continue
            mass = (mean / evidence[:, None, None]) * (length * self.jump_q)
            if np.isfinite(mass).all():
                return mass
        return None


def _conditioned_bridge_mass(q, length, a, b, denom, v, w, integrals, diagonal, *, short_bridge):
    minimum_evidence = denom.min()
    if minimum_evidence <= 0:
        raise ValueError("Impossible edge evidence under fitted CTMC.")
    tau = length * short_bridge.rate
    if 0 < tau <= _VERY_SHORT_BRIDGE_TAU:
        # Even for likely endpoints, individual excursions on very short
        # branches can be rare enough to cancel in the spectral sum.
        mass = short_bridge.mass(length, a, b)
        if mass is not None:
            return mass
    left_message, right_message = a @ v, b @ w.T
    if isinstance(integrals, tuple):
        left, right = integrals
        x = w.T @ (left_message[:, :, None] * left)
        y = v @ (right_message[:, :, None] * right)
        mass = x @ y.swapaxes(1, 2)
    else:
        coefficients = left_message[:, :, None] * right_message[:, None, :] * integrals
        # Use one matrix multiplication for the right transform across sites.
        mass = (((w.T @ coefficients).reshape(-1, len(q))) @ v.T).reshape(coefficients.shape)
    mass *= q
    mass[:, diagonal, diagonal] = 0
    # Cancellation can dominate the spectral result when evidence is small.
    unstable = (mass.min(axis=(1, 2)) < -1e-10 * denom) | (denom < 1e-10)
    has_unstable = unstable.any()
    replacement = None
    if (has_unstable or minimum_evidence < 1e-8) and 0 < tau <= _SHORT_BRIDGE_TAU:
        short = unstable | (denom < 1e-8)
        if tau > _VERY_SHORT_BRIDGE_TAU or not short.all():
            replacement = short_bridge.mass(length, a[short], b[short])
    if replacement is None:
        if has_unstable:
            mass[unstable] = bridge_kernel(q, length, a[unstable], b[unstable])
        mass /= denom[:, None, None]
    else:
        np.divide(mass, denom[:, None, None], out=mass, where=~short[:, None, None])
        mass[short] = replacement
    if mass.min() < -1e-8 or not np.isfinite(mass).all():
        raise ValueError("Numerically invalid CTMC posterior event mass.")
    return mass


def _mixture_bridge_transform(q, pi, projection, synonymous_mask):
    values, v, w = _bridge_spectral(q, pi)
    short_bridge = _ShortBridge(q)
    state_diagonal = np.arange(len(q))
    group_diagonal = np.arange(projection.shape[1])
    group_events = partial(_group_bridge_events, projection, group_diagonal)
    if synonymous_mask is not None:
        syn_from, syn_to = np.nonzero(synonymous_mask)
        syn_weights = synonymous_mask[syn_from, syn_to]
        if np.all(syn_weights == 1):
            syn_weights = None

    @lru_cache(maxsize=max(1, (32 * 1024**2) // q.nbytes))
    def integral_at(length):
        return _bridge_integral_kernel(values, length)

    def transform(left, right, transition, length, category_weight):
        active = category_weight > 0
        if length == 0 or not active.any():
            result = {'events': np.zeros((len(left), projection.shape[1], projection.shape[1]))}
            if synonymous_mask is not None:
                result['synonymous'] = np.zeros(len(left))
            return result
        partial = not active.all()
        if partial:
            left, right, category_weight = left[active], right[active], category_weight[active]
        # Normalize the boundary message before forming the integral.
        a = left / left.max(axis=1, keepdims=True)
        denom = ((a @ transition) * right).sum(axis=1)
        mass = _conditioned_bridge_mass(q, length, a, right, denom, v, w, integral_at(length), state_diagonal,
                                        short_bridge=short_bridge)
        np.maximum(mass, 0, out=mass)
        mass *= category_weight[:, None, None]
        events = group_events(mass)
        if partial:
            grouped = np.zeros((len(active), projection.shape[1], projection.shape[1]))
            grouped[active] = events
        else:
            grouped = events
        result = {'events': grouped}
        if synonymous_mask is not None:
            pairs = mass[:, syn_from, syn_to]
            if syn_weights is not None:
                pairs *= syn_weights
            values = pairs.sum(axis=1)
            if partial:
                synonymous = np.zeros(len(active))
                synonymous[active] = values
            else:
                synonymous = values
            result['synonymous'] = synonymous
        return result

    return transform


def _group_bridge_events(projection, diagonal, mass):
    # Project destinations in one matrix multiplication across the site axis.
    by_destination = mass.reshape(-1, projection.shape[0]) @ projection
    events = projection.T @ by_destination.reshape(len(mass), *projection.shape)
    events[:, diagonal, diagonal] = 0
    return events


def _endpoint_model(nodes, n, q, pi, rates, weights):
    from csubst import endpoint
    parents = np.full(n, -1, dtype=int)
    lengths = np.zeros(n)
    for node in nodes:
        if not ete.is_root(node):
            i = int(ete.get_prop(node, "numerical_label"))
            parents[i] = int(ete.get_prop(node.up, "numerical_label"))
            lengths[i] = float(node.dist)
    return endpoint.EndpointModel(parents, lengths, q, pi, rates, weights)


def _infer_endpoint_blocks(tree, tip_states, q, pi, groups, block_size, summaries, rates, weights,
                           disk_backed=False, bridge=False, model=None):
    from csubst import endpoint_io, site_storage
    nodes = list(tree.traverse("preorder"))
    n, sites, states = tip_states.shape
    if model is None:
        model = _endpoint_model(nodes, n, q, pi, rates, weights)
    tips = {}
    missing = {}
    for node in nodes:
        if not ete.get_children(node):
            i = int(ete.get_prop(node, "numerical_label"))
            tips[i] = np.asarray(tip_states[i], float)
            missing[i] = tips[i].sum(axis=1) == 0
    groups = np.asarray(groups, int)
    projection = np.eye(int(groups.max()) + 1)[groups]
    posterior = np.zeros_like(tip_states, dtype=float)
    event_shape = (n, sites, 1, projection.shape[1], projection.shape[1])
    category_shape = (len(rates), n, sites, states)
    retain_categories = summaries is not None and len(rates) > 1
    if disk_backed:
        # Bound the writing workspace independently of alignment length.
        per_site = n * (projection.shape[1]**2 + (len(rates) * states if retain_categories else 0)) * 8
        block_size = min(block_size, max(1, (32 * 1024**2) // per_site))
        tensor = site_storage.SiteEventTensor(event_shape)
        category_states = site_storage.SiteArray(category_shape, np.float64, 2) if retain_categories else None
        event_block = np.zeros((block_size, n, 1, projection.shape[1], projection.shape[1]))
        category_block = np.zeros((block_size, len(rates), n, states)) if retain_categories else None
    else:
        tensor = np.zeros(event_shape)
        category_states = np.zeros(category_shape) if retain_categories else None
    if summaries is not None:
        summaries['synonymous_counts'] = np.zeros(n)

    if bridge:
        transform = _mixture_bridge_transform(model.q, model.pi, projection,
                                              summaries['synonymous_mask'] if summaries is not None else None)
    else:
        # Reuse the classified endpoint reducer instead of materializing every
        # codon pair and contracting it through two one-hot matrices.
        event_transform = endpoint_io._event_transform({}, ['N'], {'N': projection})

        def transform(left, right, transition):
            grouped = event_transform(left, right, transition)['N', 'events'][:, 0]
            result = {'events': grouped}
            if summaries is not None:
                result['synonymous'] = ((left @ (transition * summaries['synonymous_mask'])) * right).sum(axis=1)
            return result

    block_start = None
    block_length = 0

    def flush_block():
        tensor.write_block(block_start, event_block[:block_length])
        if category_states is not None:
            assert category_block is not None
            category_states.write_block(block_start, category_block[:block_length])

    for record in model.iter_blocks(tips, block_size, transform=transform,
                                    category_nodes=category_states is not None, transform_rate_context=bridge,
                                    tip_missing=missing):
        if disk_backed and record.start != block_start:
            if block_start is not None:
                flush_block()
            block_start, block_length = record.start, record.stop - record.start
            event_block.fill(0)
            if category_block is not None:
                category_block.fill(0)
        sl = slice(record.start, record.stop)
        posterior[record.child, sl] = record.node
        if category_states is not None:
            if disk_backed:
                assert category_block is not None
                category_block[:block_length, :, record.child] = record.category_node.transpose(1, 0, 2)
            else:
                category_states[:, record.child, sl] = record.category_node
        if record.parent >= 0:
            valid = summaries['eligible'][record.child, sl] if summaries is not None and 'eligible' in summaries else np.ones(record.stop-record.start, dtype=bool)
            events = record.reduced['events'] * valid[:, None, None]
            if disk_backed:
                event_block[:block_length, record.child, 0] = events
            else:
                tensor[record.child, sl, 0] = events
            if summaries is not None:
                summaries['synonymous_counts'][record.child] += (record.reduced['synonymous'] * valid).sum()
    if disk_backed:
        if block_start is not None:
            flush_block()
        tensor.seal()
        if category_states is not None:
            category_states.seal()
    if summaries is not None:
        summaries['category_states'] = category_states
    return posterior, tensor


def infer(tree, tip_states, q, pi, groups, mode="joint", block_size=32, summaries=None,
          rates=(1.,), weights=(1.,), disk_backed=False, *, endpoint_model=None):
    """Return all-node marginals and a branch/site/group-pair event tensor.

    Input leaf rows are emission likelihoods; zero rows mean missing (ones).
    Internal input rows are ignored. Node labels index the first tensor axis.
    Missing leaves are integrated out; they are not forced to have zero events.
    """
    if mode not in ("joint", "bridge"):
        raise ValueError("Unknown CTMC observation mode.")
    if endpoint_model is not None and not (
        np.array_equal(endpoint_model.q, q) and np.array_equal(endpoint_model.pi, pi)
        and np.array_equal(endpoint_model.rates, rates) and np.array_equal(endpoint_model.weights, weights)
    ):
        raise ValueError('Fitted transition parameters changed during fixed-model calibration.')
    if mode == 'joint':
        return _infer_endpoint_blocks(tree, tip_states, q, pi, groups, block_size, summaries, rates, weights,
                                      disk_backed, model=endpoint_model)
    if len(rates) != 1 or float(rates[0]) != 1 or len(weights) != 1 or float(weights[0]) != 1:
        return _infer_endpoint_blocks(tree, tip_states, q, pi, groups, block_size, summaries, rates, weights,
                                      disk_backed, bridge=True, model=endpoint_model)
    return _bridge_infer(tree, tip_states, q, pi, groups, block_size, summaries, disk_backed, model=endpoint_model)


def _bridge_infer(tree, tip_states, q, pi, groups, block_size, summaries, disk_backed, model=None):
    from csubst import site_storage

    nodes = list(tree.traverse("preorder"))
    n, sites, states = tip_states.shape
    if model is None:
        model = _endpoint_model(nodes, n, q, pi, (1.,), (1.,))
    q, pi = model.q, model.pi
    order = [int(ete.get_prop(node, "numerical_label")) for node in nodes]
    children = {i: [int(ete.get_prop(child, "numerical_label")) for child in ete.get_children(node)]
                for i, node in zip(order, nodes)}
    lengths = {i: float(node.dist) for i, node in zip(order[1:], nodes[1:])}
    values, v, w = _bridge_spectral(q, pi)
    short_bridge = _ShortBridge(q)
    length_to_child = {length: child for child, length in lengths.items()}
    kernels = {length: (model.transition(child, 0), _bridge_integral_kernel(values, length))
               for length, child in length_to_child.items()}
    groups = np.asarray(groups, int)
    num_groups = int(groups.max()) + 1
    projection = np.eye(num_groups)[groups]
    shape = (n, sites, 1, num_groups, num_groups)
    posterior = np.zeros_like(tip_states, dtype=float)
    if disk_backed:
        per_site = n * num_groups**2 * 8
        block_size = min(block_size, max(1, (32 * 1024**2) // per_site))
        tensor = site_storage.SiteEventTensor(shape)
        event_buffer = np.zeros((min(block_size, sites), n, *shape[2:]))
    else:
        tensor = np.zeros(shape)
    if summaries is not None:
        summaries["synonymous_counts"] = np.zeros(n)

    workspace = (n, min(block_size, sites), states)
    down_buffer = np.empty(workspace)
    outside_buffer = np.empty(workspace)
    messages_buffer = np.empty(workspace)
    state_diagonal = np.arange(states)
    group_diagonal = np.arange(num_groups)
    for start in range(0, sites, block_size):
        stop = min(start + block_size, sites)
        sl = slice(start, stop)
        size = stop - start
        down = down_buffer[:, :size]
        outside = outside_buffer[:, :size]
        messages = messages_buffer[:, :size]
        events = event_buffer[:size].swapaxes(0, 1) if disk_backed else tensor[:, sl]
        for i in reversed(order):
            if not children[i]:
                emission = np.asarray(tip_states[i, sl], float).copy()
                emission[emission.sum(axis=1) == 0] = 1
                down[i] = _normalize(emission)
            else:
                down[i].fill(1)
                for child in children[i]:
                    transition, _ = kernels[lengths[child]]
                    messages[child] = down[child] @ transition.T
                    down[i] = _normalize(down[i] * messages[child])
        outside[order[0]] = pi
        for i in order:
            posterior[i, sl] = _normalize(outside[i] * down[i])
            for child in children[i]:
                a = outside[i].copy()
                for sibling in children[i]:
                    if sibling != child:
                        a = _normalize(a * messages[sibling])
                a = _normalize(a)
                length = lengths[child]
                transition, integrals = kernels[length]
                child_message = a @ transition
                outside[child] = _normalize(child_message)
                b = down[child]
                denom = (child_message * b).sum(axis=1)
                if length == 0:
                    if (denom <= 0).any():
                        raise ValueError("Impossible edge evidence under fitted CTMC.")
                    events[child, :, 0] = 0
                    continue
                mass = _conditioned_bridge_mass(q, length, a, b, denom, v, w, integrals, state_diagonal,
                                                short_bridge=short_bridge)
                if summaries is not None and "eligible" in summaries:
                    mass[~summaries["eligible"][child, sl]] = 0
                np.maximum(mass, 0, out=mass)
                if summaries is not None:
                    summaries["synonymous_counts"][child] += np.einsum("sij,ij->", mass, summaries["synonymous_mask"])
                events[child, :, 0] = _group_bridge_events(projection, group_diagonal, mass)
        if disk_backed:
            tensor.write_block(start, event_buffer[:size])
    if disk_backed:
        tensor.seal()
    return posterior, tensor


def _prepare_3di(g):
    """Infer N on the native structural model and S on the independent codon fit."""
    from csubst import endpoint_io, fitted_model, scan_endpoint
    state = np.asarray(g['state_nsy'])
    groups = np.arange(state.shape[2])
    scan_endpoint.validate_model(g, None, groups)
    structural_tree = g['tree'].copy()
    for node in structural_tree.traverse():
        node.dist = float(g['3di_branch_lengths'][int(ete.get_prop(node, 'numerical_label'))])
    observed, eligible = fitted_model.observation_masks(structural_tree, state)
    summaries = {'eligible': eligible, 'synonymous_mask': np.zeros((len(groups), len(groups)), dtype=bool)}
    disk_backed = state.shape[0] * state.shape[1] * len(groups)**2 * 8 > _SCAN_IN_MEMORY_BYTES
    posterior, tensor = infer(structural_tree, state, g['3di_q'], g['3di_pi'], groups, 'joint',
                              summaries=summaries, rates=np.ones(1), weights=np.ones(1),
                              block_size=g.get('endpoint_block_size', 64), disk_backed=disk_backed)
    # Codon synonymous counts retain their own topology lengths, Q and rate mixture.
    codon = np.asarray(g['state_cdn'])
    aa_groups = np.full(codon.shape[2], -1, dtype=int)
    for index, aa in enumerate(g['amino_acid_orders']):
        aa_groups[g['synonymous_indices'][aa]] = index
    if (aa_groups < 0).any():
        raise ValueError('CTMC scan requires the complete codon-to-amino-acid map.')
    codon_observed, codon_eligible = fitted_model.observation_masks(g['tree'], codon)
    codon_summary = {'eligible': codon_eligible,
                     'synonymous_mask': (aa_groups[:, None] == aa_groups[None, :]) & ~np.eye(len(aa_groups), dtype=bool)}
    rates, weights = endpoint_io.model_rates(g)
    codon_posterior, _ = infer(g['tree'], codon, g['instantaneous_codon_rate_matrix'],
                               g['equilibrium_frequency'], np.zeros(len(aa_groups), dtype=int), 'joint',
                               summaries=codon_summary, rates=rates, weights=weights,
                               block_size=g.get('endpoint_block_size', 64))
    aa_projection = np.eye(len(g['amino_acid_orders']))[aa_groups]
    updated = dict(g)
    updated.update(tree=structural_tree, state_nsy=posterior, state_cdn=codon_posterior,
                   state_pep=codon_posterior @ aa_projection, event_eligible=eligible,
                   scan_observed_state_nsy=state * observed[:, :, None],
                   scan_observed_state_pep=(codon @ aa_projection) * codon_observed[:, :, None],
                   scan_ctmc_synonymous_counts=codon_summary['synonymous_counts'],
                   scan_ctmc_model='GTRX+FQ (3Di)', scan_category_states=None,
                   scan_endpoint_metadata={'model': 'GTRX+FQ (3Di)', 'rates': [1.], 'weights': [1.],
                                           'category_weighting': 'posterior_given_all_tips',
                                           'missing_tip_events': 'excluded_from_reporting; latent_states_integrated',
                                           'parameter_source': '3di_checkpoint',
                                           'storage': 'site_files' if disk_backed else 'memory'})
    return updated, tensor


def prepare(g, *, endpoint_model=None):
    if g.get('nonsyn_recode') == '3di20':
        return _prepare_3di(g)
    from csubst import endpoint_io, fitted_model, scan_endpoint, substitution_scan
    groups = substitution_scan._build_codon_state_ids(g)
    model_name = scan_endpoint.validate_model(g, None, groups)[0]
    aa_groups = np.full(len(groups), -1)
    if "synonymous_indices" in g:
        for index, aa in enumerate(g["amino_acid_orders"]):
            aa_groups[g["synonymous_indices"][aa]] = index
    elif g.get("nonsyn_recode", "no") == "no":
        aa_groups = groups
    if (aa_groups < 0).any():
        raise ValueError("CTMC scan requires the complete codon-to-amino-acid map.")
    observed, eligible = fitted_model.observation_masks(g["tree"], g["state_cdn"])
    summaries = {"eligible": eligible, "synonymous_mask": (aa_groups[:, None] == aa_groups[None, :]) & ~np.eye(len(groups), dtype=bool)}
    rates, weights = endpoint_io.model_rates(g)
    n, sites, states = g['state_cdn'].shape
    stored_bytes = n * sites * (int(groups.max() + 1)**2 + (len(rates) * states if len(rates) > 1 else 0)) * 8
    disk_backed = stored_bytes > _SCAN_IN_MEMORY_BYTES
    if disk_backed:
        print('Scan CTMC events: storing {:,} bytes in site-major temporary files; bounded site reads.'.format(stored_bytes), flush=True)
    posterior, tensor = infer(g["tree"], g["state_cdn"], g["instantaneous_codon_rate_matrix"],
                              g["equilibrium_frequency"], groups, g["scan_observation"], summaries=summaries,
                              rates=rates, weights=weights, block_size=g.get('endpoint_block_size', 64),
                              disk_backed=disk_backed, endpoint_model=endpoint_model)
    updated = dict(g)
    updated["event_eligible"] = eligible
    updated["scan_tip_emissions"] = g["state_cdn"]
    updated["state_cdn"] = posterior
    projection = np.eye(g["state_nsy"].shape[2])[groups]
    updated["scan_observed_state_nsy"] = (g["state_cdn"] @ projection) * observed[:, :, None]
    updated["state_nsy"] = posterior @ projection
    if g.get("nonsyn_recode", "no") == "no":
        updated["state_pep"] = updated["state_nsy"]
    elif "synonymous_indices" in g:
        aa_projection = np.zeros((len(groups), len(g["amino_acid_orders"])))
        for index, aa in enumerate(g["amino_acid_orders"]):
            aa_projection[g["synonymous_indices"][aa], index] = 1
        updated["state_pep"] = posterior @ aa_projection
    aa_projection = np.eye(len(g["amino_acid_orders"]))[aa_groups]
    updated["scan_observed_state_pep"] = (updated["scan_observed_state_nsy"]
                                           if np.array_equal(groups, aa_groups) else
                                           (g["state_cdn"] @ aa_projection) * observed[:, :, None])
    updated["scan_ctmc_synonymous_counts"] = summaries["synonymous_counts"]
    updated["scan_ctmc_model"] = model_name
    updated['scan_category_states'] = summaries.get('category_states')
    updated['scan_endpoint_metadata'] = {
        'model': model_name, 'rates': rates.tolist(), 'weights': weights.tolist(),
        'category_weighting': 'posterior_given_all_tips',
        'missing_tip_events': 'excluded_from_reporting; latent_states_integrated',
        'parameter_source': g.get('scan_ctmc_model_precision', 'parsed_iqtree_model_and_category_table'),
        'storage': 'site_files' if disk_backed else 'memory',
    }
    return updated, tensor


def set_branch_length_summaries(g, tensor):
    """Populate reporting-only S/N lengths from the same posterior event model."""
    from csubst import substitution
    count_n = substitution.get_branch_sub_counts(tensor)
    count_s = g["scan_ctmc_synonymous_counts"]
    sites = g["event_eligible"].sum(axis=1) if "event_eligible" in g else np.full(tensor.shape[0], tensor.shape[1])
    for node in g["tree"].traverse():
        i = int(ete.get_prop(node, "numerical_label"))
        ete.set_prop(node, "Ndist", float(count_n[i] / sites[i]) if sites[i] else 0.)
        ete.set_prop(node, "Sdist", float(count_s[i] / sites[i]) if sites[i] else 0.)
        ete.set_prop(node, "SNdist", float((count_s[i] + count_n[i]) / sites[i]) if sites[i] else 0.)
    return g


def _transition_cdf(transition):
    p = np.maximum(transition, 0.)
    p /= p.sum(axis=1, keepdims=True)
    last_positive = p.shape[1] - 1 - (p[:, ::-1] > 0).argmax(axis=1)
    np.cumsum(p, axis=1, out=p)
    np.minimum(p, 1., out=p)
    # Close the final supported interval without assigning rounding residue
    # to any impossible states at the end of the row.
    p[np.arange(p.shape[1]) >= last_positive[:, None]] = 1.
    return p


def _tip_transition_cdf(q, length, rate):
    return _transition_cdf(expm(q * length * rate))


def _sample_transition_states(cdf, parent_states, rng):
    nstates = cdf.shape[1]
    binary = nstates >= 32 and len(parent_states) >= 512
    # Limit the gathered CDFs or search workspace to about 8 MiB.
    bytes_per_site = 48 if binary else nstates * (cdf.itemsize + 1)
    block_size = max(1, (8 * 1024**2) // bytes_per_site)
    states: np.ndarray = np.empty(len(parent_states), dtype=np.int64)
    for start in range(0, len(parent_states), block_size):
        stop = min(start + block_size, len(parent_states))
        parent = parent_states[start:stop]
        uniforms = rng.random(stop - start)
        # The first cumulative probability strictly above u is selected.
        # Zero-width intervals never generate a state.
        if not binary:
            states[start:stop] = (uniforms[:, None] >= cdf[parent]).sum(axis=1)
            continue
        position = states[start:stop]
        position.fill(0)
        step = 1 << ((nstates - 1).bit_length() - 1)
        while step:
            proposal = position + step
            bound = cdf[parent, np.minimum(proposal, nstates) - 1]
            position += step * ((proposal < nstates) & (uniforms >= bound))
            step >>= 1
    return states


def simulate_tips(g, rng, ambiguity_partitions=None, *, transition_cdf=None):
    """Simulate whole alignments; retain the observed missing-tip mask."""
    if ambiguity_partitions is None:
        ambiguity_partitions = validate_parametric_inputs(g)
    result = dict(g)
    original = g.get("scan_tip_emissions", g["state_cdn"])
    state = np.zeros_like(original)
    nsite = original.shape[1]
    if transition_cdf is None:
        transition_cdf = partial(_tip_transition_cdf, g["instantaneous_codon_rate_matrix"])
    pi = g["equilibrium_frequency"]
    from csubst import endpoint_io
    rates, weights = endpoint_io.model_rates(g)
    if len(rates) > 1:
        categories = rng.choice(len(rates), size=nsite, p=weights)
        category_sites = tuple(np.flatnonzero(categories == k) for k in range(len(rates)))
    else:
        category_sites = (np.arange(nsite),)
    draws = {}
    for node in g["tree"].traverse("preorder"):
        i = int(ete.get_prop(node, "numerical_label"))
        if ete.is_root(node):
            draws[i] = rng.choice(len(pi), size=nsite, p=pi)
        else:
            parent = int(ete.get_prop(node.up, "numerical_label"))
            draws[i] = np.zeros(nsite, dtype=int)
            for rate, selected in zip(rates, category_sites):
                if not len(selected):
                    continue
                cdf = transition_cdf(float(node.dist), float(rate))
                draws[i][selected] = _sample_transition_states(cdf, draws[parent][selected], rng)
        if not ete.get_children(node):
            state[i, np.arange(nsite), draws[i]] = 1
            state[i, original[i].sum(axis=1) == 0] = 0
            for sites, emission_lookup in ambiguity_partitions.get(i, []):
                state[i, sites] = emission_lookup[draws[i][sites]]
    result["state_cdn"] = state
    return result


def validate_parametric_inputs(g):
    """Build a deterministic nucleotide-coarsening model for partial ambiguity.

    Resolved positions stay resolved; an ambiguous IUPAC subset and its
    complement form two categories (N is one category). The observed codon
    support must be exactly reproducible under this partition of sense codons.
    """
    if bool(g.get("drop_invariant_tip_sites", False)):
        raise ValueError("Parametric calibration requires --drop_invariant_tip_sites no; data-dependent site filtering is not yet replayed.")
    emissions = g.get("scan_tip_emissions", g["state_cdn"])
    partitions: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    codons = np.asarray([list(str(c)) for c in g.get("codon_orders", [])])
    for node in ete.iter_leaves(g["tree"]):
        branch = int(ete.get_prop(node, "numerical_label"))
        values = emissions[branch]
        ambiguous = np.flatnonzero((values > 0).sum(axis=1) > 1)
        if not len(ambiguous):
            continue
        if codons.shape != (values.shape[1], 3):
            raise ValueError("Partial tip ambiguity requires codon_orders for its observation model.")
        supports, inverse = np.unique(values[ambiguous] > 0, axis=0, return_inverse=True)
        partitions[branch] = []
        for index, support in enumerate(supports):
            sites = ambiguous[inverse == index]
            normalized = values[sites] / values[sites].sum(axis=1, keepdims=True)
            if not np.allclose(normalized, support / support.sum(), atol=1e-8):
                raise ValueError("Parametric tip likelihoods must be uniform over an IUPAC codon support.")
            labels = []
            for position in range(3):
                allowed = np.unique(codons[support, position])
                if len(allowed) == 1:
                    labels.append(codons[:, position])
                else:
                    labels.append(np.isin(codons[:, position], allowed).astype(str))
            labels = np.asarray(labels).T
            same_class = (labels[:, None, :] == labels[None, :, :]).all(axis=2)
            observed_class = same_class[np.flatnonzero(support)[0]]
            if not np.array_equal(support, observed_class):
                raise ValueError("Tip ambiguity cannot be represented by nucleotide coarsening.")
            lookup = same_class / same_class.sum(axis=1, keepdims=True)
            partitions[branch].append((sites, lookup))
    return partitions


def calibrate(g, observed):
    """Fixed-model ASR/discovery calibration using the common maximum score."""
    from csubst import endpoint_io, scan_statistics, substitution_scan
    ambiguity_partitions = validate_parametric_inputs(g)
    count = int(g.get("scan_n_permutations", 1000))
    if count < 1:
        raise ValueError("parametric calibration requires positive replicate count.")
    trials: list[dict[str, object]] = []
    diagnostic = dict(schema_version=1, calibration="parametric", status="fixed_model_parametric_bootstrap",
                      null="fixed_codon_ctmc", sampling="monte_carlo", scope=scan_statistics.MAXIMUM_SCOPE,
                      requested_count=count, seed=int(g.get("scan_permutation_seed", 1)),
                      trials=trials, fixed=["Q", "tree", "branch_lengths", "rate_categories_and_priors", "foreground"],
                      endpoint_model=g.get('scan_endpoint_metadata'),
                      repeated=["ASR", "candidate_discovery", "support_filter", "maximum_score"])
    result = observed.copy()
    result["p_rate_enrichment_empirical"] = np.nan
    result["p_rate_enrichment_empirical_maxT"] = np.nan
    if not len(observed):
        diagnostic.update(status="no_observed_candidates", global_pvalue=1.)
        return substitution_scan._finish_scan_calibration(g, result, diagnostic)
    observed_maximum = scan_statistics.maximum_score(observed)
    rng = np.random.default_rng(diagnostic["seed"])
    maxima = []
    q = g["instantaneous_codon_rate_matrix"]
    cache_items = max(1, (32 * 1024**2) // q.nbytes)
    transition_cdf = lru_cache(maxsize=cache_items)(partial(_tip_transition_cdf, q))
    rates, weights = endpoint_io.model_rates(g)
    # Only the fitted transition matrices carry over between replicates.
    model = _endpoint_model(list(g["tree"].traverse("preorder")), g["state_cdn"].shape[0],
                            q.copy(), np.array(g["equilibrium_frequency"]), rates, weights)
    exposure_cache: dict = {}
    for index in range(count):
        simulated = simulate_tips(g, rng, ambiguity_partitions, transition_cdf=transition_cdf)
        simulated["scan_pvalue_calibration"] = "none"
        simulated["_scan_fixed_exposure"] = exposure_cache
        simulated, tensor = prepare(simulated, endpoint_model=model)
        simulated = set_branch_length_summaries(simulated, tensor)
        frame, _ = substitution_scan._scan_substitutions_core(simulated, tensor, tensor)
        maximum = scan_statistics.maximum_score(frame)
        maxima.append(maximum)
        trials.append(dict(status="success", sampling_attempts=1, configuration_id=None,
                                         failure_reason="", maximum_score=None if maximum == -np.inf else maximum))
        del tensor, simulated
    result["p_rate_enrichment_empirical_maxT"] = scan_statistics.empirical_pvalues(
        result["score_rate_enrichment"], maxima, count,
    )
    # This reference corrects the whole scan; do not apply candidate-wise BH to it.
    diagnostic["global_pvalue"] = scan_statistics.empirical_pvalue(observed_maximum, maxima, count)
    diagnostic["pvalue_resolution"] = 1 / (count + 1)
    result["scan_permutation_backend"] = "serial_parametric"
    result["scan_permutation_n_jobs"] = 1
    return substitution_scan._finish_scan_calibration(g, result, diagnostic)
