"""Finite-time codon endpoint opportunities for scan (not Markov jump counts).

The first implementation deliberately uses uniform fitted codon models only.
Exponentiate the full generator before aggregating destination state groups;
the grouped instantaneous matrix is generally not a lumped CTMC generator.
"""

import re

import numpy as np
from scipy.linalg import expm

from csubst import ete


def validate_options(g):
    if str(g.get("scan_rate_length", "n_rescaled")) != "raw":
        raise ValueError("--scan_rate_exposure endpoint requires --scan_rate_length raw; "
                         "rescaled endpoint counts are not CTMC time.")
    if str(g.get("scan_rate_event_mode", "posterior_sum")) != "posterior_sum":
        raise ValueError("--scan_rate_exposure endpoint requires --scan_rate_event_mode posterior_sum.")
    if str(g.get("nonsyn_recode", "no")) == "3di20":
        raise ValueError("Scan endpoint exposure currently supports codon-derived states only; "
                         "native 3Di endpoint context is not yet supported.")


def build_context(g, branch_meta, codon_state_ids):
    validate_options(g)
    model = str(g.get("substitution_model", ""))
    if not re.fullmatch(r"(?:GY|MG|ECMK07|ECMrest)(?:\+F(?:1X4|3X4|Q)?)?", model):
        raise ValueError("Scan endpoint exposure requires a uniform GY/MG/ECMK07/ECMrest "
                         "codon model, optionally with +F/+F1X4/+F3X4/+FQ; "
                         "rate mixtures are not supported.")
    state = np.asarray(g["state_cdn"])
    q = np.asarray(g["instantaneous_codon_rate_matrix"], dtype=np.float64)
    ids = np.asarray(codon_state_ids, dtype=np.int64)
    if state.ndim != 3 or q.shape != (state.shape[2], state.shape[2]) or ids.shape != (state.shape[2],):
        raise ValueError("Scan endpoint codon Q/state/group axes do not match.")
    off = q.copy()
    np.fill_diagonal(off, 0)
    if (not np.isfinite(q).all() or (off < 0).any()
            or not np.allclose(q.sum(axis=1), 0, atol=1e-10, rtol=0)):
        raise ValueError("Scan endpoint requires a finite CTMC generator with nonnegative off-diagonals and zero row sums.")
    pi = np.asarray(g.get("equilibrium_frequency", []), dtype=float)
    if (pi.shape != (len(q),) or not np.isfinite(pi).all() or (pi < 0).any()
            or not np.isclose(pi.sum(), 1) or not np.allclose(pi @ q, 0, atol=1e-10, rtol=0)
            or not np.isclose(-pi @ q.diagonal(), 1)):
        raise ValueError("Scan endpoint Q must match normalized stationary codon frequencies and model time units.")
    if "tree" in g:
        for node in g["tree"].traverse():
            if ete.is_root(node):
                continue
            length = float(node.dist or 0)
            if not np.isfinite(length) or length < 0:
                raise ValueError("Scan endpoint requires finite nonnegative model branch lengths.")
    rates = np.asarray(g.get("iqtree_rate_values", []), dtype=float)
    if rates.shape != (state.shape[1],) or not np.all(rates == 1):
        raise ValueError("Scan endpoint requires one unit fitted site rate per analyzed site.")
    lengths = branch_meta["raw_length"].to_numpy(dtype=float)
    if not np.isfinite(lengths).all() or (lengths < 0).any():
        raise ValueError("Scan endpoint requires finite nonnegative model branch lengths.")
    num_group = np.asarray(g["state_nsy"]).shape[2]
    if (ids < 0).any() or (ids >= num_group).any():
        raise ValueError("Scan endpoint has invalid codon state groups.")
    # Only a branch x codon x group tensor: no branch x site x codon-pair tensor.
    group_projection = np.eye(num_group, dtype=float)[ids]
    reachable = off > 0
    np.fill_diagonal(reachable, True)
    for via in range(len(q)):
        reachable |= reachable[:, via, None] & reachable[None, via, :]
    reachable_groups = (reachable.astype(int) @ group_projection) > 0
    unique_lengths, inverse = np.unique(lengths, return_inverse=True)
    projected = np.empty((len(unique_lengths), len(q), num_group), dtype=float)
    for index, length in enumerate(unique_lengths):
        p = expm(q * length)
        if (not np.isfinite(p).all() or (p < -1e-12).any()
                or not np.allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0)):
            raise ValueError("Invalid scan endpoint transition probabilities.")
        # Remove only negative floating-point roundoff; preserve tiny positive paths.
        projected[index] = np.maximum(p, 0) @ group_projection
    extra = {}
    if g.get("scan_observation") == "bridge":
        # Integral of transition probabilities, computed without inverting singular Q.
        occupation = []
        for length in unique_lengths:
            augmented = np.zeros((2 * len(q), 2 * len(q)))
            augmented[:len(q), :len(q)] = q
            augmented[:len(q), len(q):] = np.eye(len(q))
            integrated = expm(augmented * length)[:len(q), len(q):]
            if (not np.isfinite(integrated).all() or (integrated < -1e-10).any()
                    or not np.allclose(integrated.sum(axis=1), length, atol=1e-12, rtol=1e-9)):
                raise ValueError("Invalid integrated CTMC transition probabilities.")
            occupation.append(np.maximum(integrated, 0))
        extra = {"occupation": np.asarray(occupation), "jump_q": off, "reachable_states": reachable}
    return {**extra, "transition_groups": projected, "length_indices": inverse,
            "raw_lengths": lengths, "reachable_groups": reachable_groups,
            "branch_ids": branch_meta["branch_id"].to_numpy(dtype=np.int64),
            "parent_ids": branch_meta["parent_id"].to_numpy(dtype=np.int64),
            "codon_state_ids": ids, "model": model}


def expected_events(context, state_cdn, state_nsy, site, from_ids, to_ids):
    """Return expected endpoint mass and branch/site missingness.

    Normalize nonmissing parent codon rows to absorb ASR text rounding. Both
    endpoints must have posterior mass on this site, as on the observed side.
    """
    ids = context["codon_state_ids"]
    groups = context["transition_groups"]
    num_group = groups.shape[2]
    from_ids = np.unique(np.asarray(from_ids, dtype=np.int64))
    to_ids = np.unique(np.asarray(to_ids, dtype=np.int64))
    if any((arr < 0).any() or (arr >= num_group).any() for arr in (from_ids, to_ids)):
        raise ValueError("Scan endpoint candidate state is out of range.")
    allowed = np.isin(ids, from_ids)
    # Sum off-group probabilities directly, avoiding 1 - P_ii cancellation
    # on very short branches. Synonymous endpoint changes are excluded too.
    candidate_mask = allowed[:, None] & (ids[:, None] != to_ids[None, :])
    weights = np.einsum("lcg,cg->lc", groups[:, :, to_ids], candidate_mask)
    parent = np.asarray(state_cdn[context["parent_ids"], int(site), :], dtype=float)
    child = np.asarray(state_cdn[context["branch_ids"], int(site), :], dtype=float)
    nsy_parent = np.asarray(state_nsy[context["parent_ids"], int(site), :], dtype=float)
    nsy_child = np.asarray(state_nsy[context["branch_ids"], int(site), :], dtype=float)
    for values in (parent, child, nsy_parent, nsy_child):
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Scan endpoint posterior states must be finite and nonnegative.")
        total = values.sum(axis=1)
        if not np.all((total == 0) | np.isclose(total, 1, atol=1e-4, rtol=0)):
            raise ValueError("Scan endpoint posterior rows must sum to one (or zero for missing states).")
    mass = parent.sum(axis=1)
    missing = (mass == 0) | (child.sum(axis=1) == 0) | (nsy_parent.sum(axis=1) == 0) | (nsy_child.sum(axis=1) == 0)
    normalized = np.divide(parent, mass[:, None], out=np.zeros_like(parent), where=mass[:, None] > 0)
    if "occupation" in context:
        jump_rates = (context["jump_q"][:, np.isin(ids, to_ids)] *
                      (ids[:, None] != ids[np.isin(ids, to_ids)][None, :])).sum(axis=1) * allowed
        jump_weights = context["occupation"] @ jump_rates
        expected = np.einsum("bi,bi->b", normalized, jump_weights[context["length_indices"]])
    else:
        expected = np.einsum("bi,bi->b", normalized, weights[context["length_indices"]])
    expected[missing] = 0
    if not np.isfinite(expected).all() or (expected < -1e-12).any() or ("occupation" not in context and (expected > 1 + 1e-10).any()):
        raise ValueError("Scan endpoint expected mass is outside [0, 1].")
    expected = np.maximum(expected, 0)
    reasons = np.full(len(expected), "ok", dtype="U24")
    zero = expected == 0
    reachable_candidate = (context["reachable_groups"][:, to_ids] & candidate_mask).any(axis=1)
    if "occupation" in context:
        reachable_candidate = context["reachable_states"][:, jump_rates > 0].any(axis=1)
    reachable_parent = ((parent > 0) & reachable_candidate[None, :]).any(axis=1)
    source_mass = parent[:, allowed].sum(axis=1)
    reasons[zero] = "numerical_zero"
    reasons[zero & ~reachable_parent] = "unreachable"
    if "occupation" not in context:
        reasons[zero & (source_mass == 0)] = "zero_source_mass"
    reasons[zero & (context["raw_lengths"] == 0)] = "zero_model_length"
    reasons[missing] = "missing_state"
    return expected, missing, reasons
