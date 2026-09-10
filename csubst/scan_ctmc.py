"""Scaled pruning, exact edge posteriors, and posterior CTMC jump counts.

Uniform stationary reversible codon models, conditional on fitted Q/tree.
No sampled ancestral states or products of separately inferred marginals.
"""
import numpy as np
from scipy.linalg import eigh, expm, expm_frechet

from csubst import ete


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


def infer(tree, tip_states, q, pi, groups, mode="joint", block_size=32, summaries=None):
    """Return all-node marginals and a branch/site/group-pair event tensor.

    Input leaf rows are emission likelihoods; zero rows mean missing (ones).
    Internal input rows are ignored. Node labels index the first tensor axis.
    Missing leaves are integrated out; they are not forced to have zero events.
    """
    if mode not in ("joint", "bridge"):
        raise ValueError("Unknown CTMC observation mode.")
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
                if summaries is not None:
                    summaries["synonymous_counts"][j] += np.einsum("sij,ij->", np.maximum(mass, 0), summaries["synonymous_mask"])
                grouped = np.einsum("ig,sij,jh->sgh", projection, np.maximum(mass, 0), projection, optimize=True)
                ix = np.arange(projection.shape[1])
                grouped[:, ix, ix] = 0
                tensor[j, start:stop, 0] = grouped
    return posterior, tensor


def prepare(g):
    from csubst import scan_endpoint, substitution_scan
    groups = substitution_scan._build_codon_state_ids(g)
    context = scan_endpoint.build_context(g, substitution_scan.build_branch_metadata(g), groups)
    aa_groups = np.full(len(groups), -1)
    if "synonymous_indices" in g:
        for index, aa in enumerate(g["amino_acid_orders"]):
            aa_groups[g["synonymous_indices"][aa]] = index
    elif g.get("nonsyn_recode", "no") == "no":
        aa_groups = groups
    if (aa_groups < 0).any():
        raise ValueError("CTMC scan requires the complete codon-to-amino-acid map.")
    summaries = {"synonymous_mask": (aa_groups[:, None] == aa_groups[None, :]) & ~np.eye(len(groups), dtype=bool)}
    posterior, tensor = infer(g["tree"], g["state_cdn"], g["instantaneous_codon_rate_matrix"],
                              g["equilibrium_frequency"], groups, g["scan_observation"], summaries=summaries)
    updated = dict(g)
    updated["state_cdn"] = posterior
    projection = np.eye(g["state_nsy"].shape[2])[groups]
    updated["state_nsy"] = posterior @ projection
    if g.get("nonsyn_recode", "no") == "no":
        updated["state_pep"] = updated["state_nsy"]
    elif "synonymous_indices" in g:
        aa_projection = np.zeros((len(groups), len(g["amino_acid_orders"])))
        for index, aa in enumerate(g["amino_acid_orders"]):
            aa_projection[g["synonymous_indices"][aa], index] = 1
        updated["state_pep"] = posterior @ aa_projection
    updated["scan_ctmc_synonymous_counts"] = summaries["synonymous_counts"]
    updated["scan_ctmc_model"] = context["model"]
    return updated, tensor


def set_branch_length_summaries(g, tensor):
    """Populate reporting-only S/N lengths from the same posterior event model."""
    count_n = tensor.sum(axis=(1, 2, 3, 4))
    count_s = g["scan_ctmc_synonymous_counts"]
    sites = tensor.shape[1]
    for node in g["tree"].traverse():
        i = int(ete.get_prop(node, "numerical_label"))
        ete.set_prop(node, "Ndist", float(count_n[i] / sites))
        ete.set_prop(node, "Sdist", float(count_s[i] / sites))
        ete.set_prop(node, "SNdist", float((count_s[i] + count_n[i]) / sites))
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
    draws = {}
    for node in g["tree"].traverse("preorder"):
        i = int(ete.get_prop(node, "numerical_label"))
        if ete.is_root(node):
            draws[i] = rng.choice(len(pi), size=nsite, p=pi)
        else:
            parent = int(ete.get_prop(node.up, "numerical_label"))
            p = np.maximum(expm(q * float(node.dist)), 0)
            p /= p.sum(axis=1, keepdims=True)
            draws[i] = (rng.random(nsite)[:, None] > np.cumsum(p[draws[parent]], axis=1)).sum(axis=1)
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
                      trials=trials, fixed=["Q", "tree", "branch_lengths", "foreground"],
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
