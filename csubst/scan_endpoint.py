"""Finite-time fitted endpoint opportunities for scan (not Markov jump counts).

Codon models support fitted rate mixtures; native 3Di uses uniform GTR.
Exponentiate the full generator before aggregating destination state groups;
the grouped instantaneous matrix is generally not a lumped CTMC generator.
"""

import numpy as np

from csubst import endpoint, endpoint_io, ete, expectation_3di, site_storage


def validate_options(g):
    if str(g.get("scan_rate_length", "n_rescaled")) != "raw":
        raise ValueError("--scan_rate_exposure endpoint requires --scan_rate_length raw; "
                         "rescaled endpoint counts are not CTMC time.")
    if str(g.get("scan_rate_event_mode", "posterior_sum")) != "posterior_sum":
        raise ValueError("--scan_rate_exposure endpoint requires --scan_rate_event_mode posterior_sum.")
    if str(g.get("nonsyn_recode", "no")) == "3di20":
        expectation_3di.validate_options(g)
        if g.get("scan_observation") != "joint":
            raise ValueError("Native 3Di endpoint exposure requires joint observations.")
        if g.get("scan_pvalue_calibration") in ("parametric", "parametric_bootstrap"):
            raise ValueError("Native 3Di scan does not yet support parametric calibration.")


def validate_model(g, branch_meta, codon_state_ids):
    """Validate the fitted model without constructing exposure transitions."""
    validate_options(g)
    if g.get("nonsyn_recode") == "3di20":
        state = np.asarray(g["state_nsy"])
        expectation_3di.validate_context(g, *state.shape[:2])
        q = np.asarray(g["3di_q"], dtype=float)
        ids = np.arange(state.shape[2])
        if not np.array_equal(codon_state_ids, ids):
            raise ValueError("Native 3Di endpoint groups must use the fitted state order.")
        indices = ([int(ete.get_prop(node, "numerical_label")) for node in g["tree"].traverse()
                    if not ete.is_root(node)] if branch_meta is None else
                   branch_meta["branch_id"].to_numpy(dtype=int))
        lengths = np.asarray(g["3di_branch_lengths"])[indices]
        off = q.copy()
        np.fill_diagonal(off, 0)
        return "GTRX+FQ (3Di)", np.ones(1), np.ones(1), q, ids, len(ids), lengths, off
    model = str(g.get("substitution_model", ""))
    rates, priors = endpoint_io.model_rates(g)
    if len(rates) > 1 and g.get('scan_observation', 'marginal') not in ('joint', 'bridge'):
        raise ValueError('Discrete-rate endpoint exposure requires joint or bridge observations.')
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
    # Before joint reconstruction, internal state rows may intentionally be
    # empty. Validate model lengths directly, without filtering branches by
    # their not-yet-inferred posterior states.
    if branch_meta is None:
        lengths = np.asarray([float(node.dist or 0) for node in g['tree'].traverse()
                              if not ete.is_root(node)])
    else:
        lengths = branch_meta["raw_length"].to_numpy(dtype=float)
    if not np.isfinite(lengths).all() or (lengths < 0).any():
        raise ValueError("Scan endpoint requires finite nonnegative model branch lengths.")
    num_group = np.asarray(g["state_nsy"]).shape[2]
    if (ids < 0).any() or (ids >= num_group).any():
        raise ValueError("Scan endpoint has invalid codon state groups.")
    return model, rates, priors, q, ids, num_group, lengths, off


def _exposure_matrices(q, rates, unique_lengths, group_projection, bridge):
    num_group = group_projection.shape[1]
    extra: dict = {}
    series = endpoint.TransitionSeries(q)
    if bridge:
        # Integral in rate-scaled time. This incorporates the category's rate
        # multiplier on Q when computing jump opportunities.
        occupation = np.empty((len(rates), len(unique_lengths), len(q), len(q)))
        for category, rate in enumerate(rates):
            for index, length in enumerate(unique_lengths):
                time = length * rate
                integrated = series.occupation(time)
                if (not np.isfinite(integrated).all() or (integrated < -1e-10).any()
                        or not np.allclose(integrated.sum(axis=1), time, atol=1e-12, rtol=1e-9)):
                    raise ValueError("Invalid integrated CTMC transition probabilities.")
                occupation[category, index] = np.maximum(integrated, 0)
        extra = {"occupation": occupation if len(rates) > 1 else occupation[0]}
    else:
        projected = np.empty((len(rates), len(unique_lengths), len(q), num_group), dtype=float)
        for category, rate in enumerate(rates):
            for index, length in enumerate(unique_lengths):
                p = series.transition(length * rate)
                if (not np.isfinite(p).all() or (p < -1e-12).any()
                        or not np.allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0)):
                    raise ValueError("Invalid scan endpoint transition probabilities.")
                projected[category, index] = np.maximum(p, 0) @ group_projection
        extra['transition_groups'] = projected[0]
        if len(rates) > 1:
            extra['category_groups'] = projected
    return extra


def build_context(g, branch_meta, codon_state_ids):
    model, rates, priors, q, ids, num_group, lengths, off = validate_model(g, branch_meta, codon_state_ids)
    # Only a branch x model-state x group tensor: no branch x site x codon-pair tensor.
    group_projection = np.eye(num_group, dtype=float)[ids]
    reachable = off > 0
    np.fill_diagonal(reachable, True)
    for via in range(len(q)):
        reachable |= reachable[:, via, None] & reachable[None, via, :]
    reachable_groups = (reachable.astype(int) @ group_projection) > 0
    unique_lengths, inverse = np.unique(lengths, return_inverse=True)
    bridge = g.get("scan_observation") == "bridge"
    cache = g.get("_scan_fixed_exposure")
    extra = None
    if cache is not None:
        key = (bridge, q.shape, q.tobytes(), rates.tobytes(), unique_lengths.tobytes(), ids.tobytes(), num_group)
        if cache.get("key") == key:
            extra = cache["matrices"].copy()
    if extra is None:
        extra = _exposure_matrices(q, rates, unique_lengths, group_projection, bridge)
        if cache is not None:
            cache.clear()
            if sum(values.nbytes for values in extra.values()) <= 32 * 1024**2:
                cache.update(key=key, matrices=extra.copy())
    if bridge:
        extra.update(jump_q=off, reachable_states=reachable)
    if len(rates) > 1:
        extra['category_states'] = g.get('scan_category_states')
    if "event_eligible" in g:
        extra["eligible"] = g["event_eligible"]
    return {**extra, "num_group": num_group, "length_indices": inverse,
            "category_rates": rates, "category_priors": priors,
            "raw_lengths": lengths, "reachable_groups": reachable_groups,
            "branch_ids": branch_meta["branch_id"].to_numpy(dtype=np.int64),
            "parent_ids": branch_meta["parent_id"].to_numpy(dtype=np.int64),
            "codon_state_ids": ids, "model": model}


def expected_events(context, state_cdn, state_nsy, site, from_ids, to_ids):
    """Return expected endpoint mass and branch/site missingness.

    Normalize nonmissing parent model-state rows to absorb ASR text rounding. Both
    endpoints must have posterior mass on this site, as on the observed side.
    """
    ids = context["codon_state_ids"]
    num_group = context["num_group"]
    from_ids = np.unique(np.asarray(from_ids, dtype=np.int64))
    to_ids = np.unique(np.asarray(to_ids, dtype=np.int64))
    if any((arr < 0).any() or (arr >= num_group).any() for arr in (from_ids, to_ids)):
        raise ValueError("Scan endpoint candidate state is out of range.")
    allowed = np.isin(ids, from_ids)
    # Sum off-group probabilities directly, avoiding 1 - P_ii cancellation
    # on very short branches. Synonymous endpoint changes are excluded too.
    candidate_mask = allowed[:, None] & (ids[:, None] != to_ids[None, :])
    parent = np.asarray(state_cdn[context["parent_ids"], int(site), :], dtype=float)
    child = np.asarray(state_cdn[context["branch_ids"], int(site), :], dtype=float)
    nsy_parent = np.asarray(state_nsy[context["parent_ids"], int(site), :], dtype=float)
    nsy_child = np.asarray(state_nsy[context["branch_ids"], int(site), :], dtype=float)
    row_totals = []
    for values in (parent, child, nsy_parent, nsy_child):
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Scan endpoint posterior states must be finite and nonnegative.")
        total = values.sum(axis=1)
        if not np.all((total == 0) | np.isclose(total, 1, atol=1e-4, rtol=0)):
            raise ValueError("Scan endpoint posterior rows must sum to one (or zero for missing states).")
        row_totals.append(total)
    mass, child_mass, nsy_parent_mass, nsy_child_mass = row_totals
    missing = (mass == 0) | (child_mass == 0) | (nsy_parent_mass == 0) | (nsy_child_mass == 0)
    if "eligible" in context:
        missing |= ~context["eligible"][context["branch_ids"], int(site)]
    if 'category_states' in context:
        category_states = context['category_states']
        if category_states is None:
            raise ValueError('Mixture exposure requires category-conditional joint posteriors.')
        weighted_parent = (category_states.read_site(site)[:, context['parent_ids'], :]
                           if isinstance(category_states, site_storage.SiteArray) else
                           category_states[:, context['parent_ids'], int(site), :])
        if (not np.isfinite(weighted_parent).all() or (weighted_parent < 0).any()
                or not np.allclose(weighted_parent.sum(axis=0), parent, atol=1e-10, rtol=1e-8)):
            raise ValueError('Category posterior states do not match scan parent marginals.')
    else:
        weighted_parent = None
        normalized = np.divide(parent, mass[:, None], out=np.zeros_like(parent), where=mass[:, None] > 0)
    if "occupation" in context:
        target = np.isin(ids, to_ids)
        jump_rates = (context["jump_q"][:, target] *
                      (ids[:, None] != ids[target][None, :])).sum(axis=1) * allowed
        jump_weights = context["occupation"] @ jump_rates
        if weighted_parent is not None:
            expected = np.einsum('cbi,cbi->b', weighted_parent, jump_weights[:, context['length_indices']])
        else:
            expected = np.einsum("bi,bi->b", normalized, jump_weights[context["length_indices"]])
    elif weighted_parent is not None:
        category_weights = np.einsum('clig,ig->cli', context['category_groups'][:, :, :, to_ids], candidate_mask)
        expected = np.einsum('cbi,cbi->b', weighted_parent,
                             category_weights[:, context['length_indices']])
    else:
        groups = context["transition_groups"]
        weights = np.einsum("lcg,cg->lc", groups[:, :, to_ids], candidate_mask)
        expected = np.einsum("bi,bi->b", normalized, weights[context["length_indices"]])
    expected[missing] = 0
    if not np.isfinite(expected).all() or (expected < -1e-12).any() or ("occupation" not in context and (expected > 1 + 1e-10).any()):
        raise ValueError("Scan endpoint expected mass is outside [0, 1].")
    expected = np.maximum(expected, 0)
    reasons: np.ndarray = np.full(len(expected), "ok", dtype="U24")
    zero = expected == 0
    if not zero.any():
        return expected, missing, reasons
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
