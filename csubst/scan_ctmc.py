"""Scaled pruning, exact edge posteriors, and posterior CTMC jump counts.

Stationary codon models with discrete rate categories, conditional on fitted Q/tree.
No sampled ancestral states or products of separately inferred marginals.
"""
import numpy as np
from scipy.linalg import eigh, expm, expm_frechet

from csubst import ete

_SCAN_IN_MEMORY_BYTES = 64 * 1024**2


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


def _joint_infer(tree, tip_states, q, pi, groups, block_size, summaries, rates, weights, disk_backed=False):
    from csubst import endpoint, endpoint_io, site_storage
    nodes = list(tree.traverse("preorder"))
    n, sites, states = tip_states.shape
    parents = np.full(n, -1, dtype=int)
    lengths = np.zeros(n)
    tips = {}
    for node in nodes:
        i = int(ete.get_prop(node, "numerical_label"))
        if not ete.is_root(node):
            parents[i] = int(ete.get_prop(node.up, "numerical_label"))
            lengths[i] = float(node.dist)
        if not ete.get_children(node):
            values = np.asarray(tip_states[i], float).copy()
            values[values.sum(axis=1) == 0] = 1
            tips[i] = values
    model = endpoint.EndpointModel(parents, lengths, q, pi, rates, weights)
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
                                    category_nodes=category_states is not None):
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
          rates=(1.,), weights=(1.,), disk_backed=False):
    """Return all-node marginals and a branch/site/group-pair event tensor.

    Input leaf rows are emission likelihoods; zero rows mean missing (ones).
    Internal input rows are ignored. Node labels index the first tensor axis.
    Missing leaves are integrated out; they are not forced to have zero events.
    """
    if mode not in ("joint", "bridge"):
        raise ValueError("Unknown CTMC observation mode.")
    if mode == 'joint':
        return _joint_infer(tree, tip_states, q, pi, groups, block_size, summaries, rates, weights, disk_backed)
    if len(rates) != 1 or float(rates[0]) != 1 or len(weights) != 1 or float(weights[0]) != 1:
        raise ValueError('CTMC bridge currently requires uniform unit rates; use joint for discrete rate mixtures.')
    q, pi = np.asarray(q, float), np.asarray(pi, float)
    nodes = list(tree.traverse("preorder"))
    def label(node):
        return int(ete.get_prop(node, "numerical_label"))
    n, sites, states = tip_states.shape
    groups = np.asarray(groups, int)
    projection = np.eye(int(groups.max()) + 1)[groups]
    transition = {label(node): expm(q * float(node.dist)) for node in nodes if not ete.is_root(node)}
    down = np.ones((n, sites, states))
    messages = {}
    for node in reversed(nodes):
        i = label(node)
        children = ete.get_children(node)
        if not children:
            emission = np.asarray(tip_states[i], float).copy()
            emission[emission.sum(axis=1) == 0] = 1
            down[i] = _normalize(emission)
        else:
            for child in children:
                j = label(child)
                messages[j] = down[j] @ transition[j].T
                down[i] = _normalize(down[i] * messages[j])
    outside = np.zeros_like(down)
    root_id = label(tree)
    outside[root_id] = pi
    posterior = np.zeros_like(down)
    tensor = np.zeros((n, sites, 1, projection.shape[1], projection.shape[1]))
    if summaries is not None:
        summaries["synonymous_counts"] = np.zeros(n)
    spectral = _bridge_spectral(q, pi) if mode == "bridge" else None
    for node in nodes:
        i = label(node)
        posterior[i] = _normalize(outside[i] * down[i])
        children = ete.get_children(node)
        for child in children:
            j = label(child)
            a = outside[i].copy()
            for sibling in children:
                if sibling is not child:
                    a = _normalize(a * messages[label(sibling)])
            a = _normalize(a)
            outside[j] = _normalize(a @ transition[j])
            b = down[j]
            denom = np.einsum("si,ij,sj->s", a, transition[j], b)
            if (denom <= 0).any():
                raise ValueError("Impossible edge evidence under fitted CTMC.")
            if spectral is not None:
                values, v, w = spectral
                integrals = _integrals(values, float(child.dist))
            for start in range(0, sites, block_size):
                stop = min(start + block_size, sites)
                aa, bb = a[start:stop], b[start:stop]
                if mode == "joint":
                    mass = aa[:, :, None] * transition[j] * bb[:, None, :]
                else:
                    coefficients = (aa @ v)[:, :, None] * (bb @ w.T)[:, None, :] * integrals
                    mass = np.einsum("ia,sab,bj->sij", w.T, coefficients, v.T, optimize=True) * q
                    ix = np.arange(states)
                    mass[:, ix, ix] = 0
                    # Ill-conditioned evidence needs a stable direct Frechet derivative.
                    unstable = (mass.min(axis=(1, 2)) < -1e-10 * denom[start:stop]) | (denom[start:stop] < 1e-10)
                    if unstable.any():
                        mass[unstable] = bridge_kernel(q, float(child.dist), aa[unstable], bb[unstable])
                mass /= denom[start:stop, None, None]
                if mass.min() < -1e-8 or not np.isfinite(mass).all():
                    raise ValueError("Numerically invalid CTMC posterior event mass.")
                if summaries is not None and "eligible" in summaries:
                    mass[~summaries["eligible"][j, start:stop]] = 0
                if summaries is not None:
                    summaries["synonymous_counts"][j] += np.einsum("sij,ij->", np.maximum(mass, 0), summaries["synonymous_mask"])
                grouped = np.einsum("ig,sij,jh->sgh", projection, np.maximum(mass, 0), projection, optimize=True)
                ix = np.arange(projection.shape[1])
                grouped[:, ix, ix] = 0
                tensor[j, start:stop, 0] = grouped
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


def prepare(g):
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
    disk_backed = g['scan_observation'] == 'joint' and stored_bytes > _SCAN_IN_MEMORY_BYTES
    if disk_backed:
        print('Scan endpoints: storing {:,} bytes in site-major temporary files; bounded site reads.'.format(stored_bytes), flush=True)
    posterior, tensor = infer(g["tree"], g["state_cdn"], g["instantaneous_codon_rate_matrix"],
                              g["equilibrium_frequency"], groups, g["scan_observation"], summaries=summaries,
                              rates=rates, weights=weights, block_size=g.get('endpoint_block_size', 64),
                              disk_backed=disk_backed)
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


def simulate_tips(g, rng, ambiguity_partitions=None):
    """Simulate whole alignments; retain the observed missing-tip mask."""
    if ambiguity_partitions is None:
        ambiguity_partitions = validate_parametric_inputs(g)
    result = dict(g)
    original = g.get("scan_tip_emissions", g["state_cdn"])
    state = np.zeros_like(original)
    nsite = original.shape[1]
    q = g["instantaneous_codon_rate_matrix"]
    pi = g["equilibrium_frequency"]
    from csubst import endpoint_io
    rates, weights = endpoint_io.model_rates(g)
    categories = rng.choice(len(rates), size=nsite, p=weights) if len(rates) > 1 else np.zeros(nsite, dtype=int)
    draws = {}
    for node in g["tree"].traverse("preorder"):
        i = int(ete.get_prop(node, "numerical_label"))
        if ete.is_root(node):
            draws[i] = rng.choice(len(pi), size=nsite, p=pi)
        else:
            parent = int(ete.get_prop(node.up, "numerical_label"))
            draws[i] = np.zeros(nsite, dtype=int)
            for category, rate in enumerate(rates):
                selected = np.flatnonzero(categories == category)
                p = np.maximum(expm(q * float(node.dist) * rate), 0)
                p /= p.sum(axis=1, keepdims=True)
                draws[i][selected] = (rng.random(len(selected))[:, None] > np.cumsum(p[draws[parent][selected]], axis=1)).sum(axis=1)
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
    from csubst import scan_statistics, substitution_scan
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
    for index in range(count):
        simulated = simulate_tips(g, rng, ambiguity_partitions)
        simulated["scan_pvalue_calibration"] = "none"
        simulated, tensor = prepare(simulated)
        simulated = set_branch_length_summaries(simulated, tensor)
        frame, _ = substitution_scan._scan_substitutions_core(simulated, tensor, tensor)
        maximum = scan_statistics.maximum_score(frame)
        maxima.append(maximum)
        trials.append(dict(status="success", sampling_attempts=1, configuration_id=None,
                                         failure_reason="", maximum_score=None if maximum == -np.inf else maximum))
        del tensor, simulated
    result["p_rate_enrichment_empirical_maxT"] = [
        scan_statistics.empirical_pvalue(float(v), maxima, count) for v in result["score_rate_enrichment"]]
    # This reference corrects the whole scan; do not apply candidate-wise BH to it.
    diagnostic["global_pvalue"] = scan_statistics.empirical_pvalue(observed_maximum, maxima, count)
    diagnostic["pvalue_resolution"] = 1 / (count + 1)
    result["scan_permutation_backend"] = "serial_parametric"
    result["scan_permutation_n_jobs"] = 1
    return substitution_scan._finish_scan_calibration(g, result, diagnostic)
