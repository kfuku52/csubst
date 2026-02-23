import numpy as np
import pytest
from matplotlib import image as mpimg

from csubst import recoding


def _toy_grouping_g():
    amino_acids = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object)
    codon_orders = np.array(["C{:02d}".format(i) for i in range(amino_acids.shape[0])], dtype=object)
    synonymous_indices = {aa: [i] for i, aa in enumerate(amino_acids.tolist())}
    matrix_groups = {aa: [codon_orders[i]] for i, aa in enumerate(amino_acids.tolist())}
    return {
        "amino_acid_orders": amino_acids,
        "codon_orders": codon_orders,
        "synonymous_indices": synonymous_indices,
        "matrix_groups": matrix_groups,
    }


def _toy_auto_grouping_g():
    g = _toy_grouping_g()
    aa_orders = [str(aa) for aa in g["amino_acid_orders"].tolist()]
    aa_matrix = np.vstack(
        [
            np.arange(20, dtype=np.int16),
            np.roll(np.arange(20, dtype=np.int16), 1),
            np.roll(np.arange(20, dtype=np.int16), 5),
            np.roll(np.arange(20, dtype=np.int16), 10),
        ]
    )
    base = np.arange(1, 21, dtype=np.float64)
    fmat = np.vstack(
        [
            base / base.sum(),
            base[::-1] / base.sum(),
            np.roll(base, 5) / base.sum(),
            np.roll(base, 10) / base.sum(),
        ]
    )
    nsitev = np.array([400, 400, 400, 400], dtype=np.int64)
    fr = (fmat * nsitev[:, np.newaxis]).sum(axis=0)
    fr = fr / fr.sum()
    g["alignment_file"] = ""
    g["nonsyn_recode_seed"] = 7
    g["nonsyn_recode_random_starts"] = 24
    g["_nonsyn_recode_alignment_cache"] = {
        "alignment_file": "",
        "aa_orders": tuple(aa_orders),
        "aa_matrix": aa_matrix,
        "fmat": fmat,
        "fr": fr,
        "nsitev": nsitev,
    }
    return g


def test_normalize_nonsyn_recode_accepts_aliases():
    assert recoding.normalize_nonsyn_recode("no") == "no"
    assert recoding.normalize_nonsyn_recode("dayhoff-6") == "dayhoff6"
    assert recoding.normalize_nonsyn_recode("SR_6") == "sr6"
    assert recoding.normalize_nonsyn_recode("sr-chi-sq") == "srchisq6"
    assert recoding.normalize_nonsyn_recode("kgb-auto") == "kgbauto6"


def test_normalize_nonsyn_recode_rejects_unknown_value():
    with pytest.raises(ValueError, match="--nonsyn_recode should be one of"):
        recoding.normalize_nonsyn_recode("unknown")
    with pytest.raises(ValueError, match="--nonsyn_recode should be one of"):
        recoding.normalize_nonsyn_recode("none")


def test_initialize_nonsyn_groups_no_copies_amino_acid_groups():
    g = _toy_grouping_g()
    g["nonsyn_recode"] = "no"
    out = recoding.initialize_nonsyn_groups(g)
    assert out["nonsyn_recode"] == "no"
    assert out["nonsyn_state_orders"].tolist() == out["amino_acid_orders"].tolist()
    assert out["max_nonsynonymous_size"] == 1
    for aa in out["amino_acid_orders"]:
        assert out["nonsynonymous_indices"][aa] == out["synonymous_indices"][aa]


def test_initialize_nonsyn_groups_dayhoff6_builds_expected_membership():
    g = _toy_grouping_g()
    g["nonsyn_recode"] = "dayhoff6"
    out = recoding.initialize_nonsyn_groups(g)
    assert out["nonsyn_state_orders"].tolist() == ["AGPST", "DENQ", "HKR", "ILMV", "FWY", "C"]
    assert out["max_nonsynonymous_size"] == 5
    for aa in list("AGPST"):
        assert out["nonsyn_aa_to_state"][aa] == "AGPST"
    for aa in list("DENQ"):
        assert out["nonsyn_aa_to_state"][aa] == "DENQ"
    expected = sorted([g["synonymous_indices"][aa][0] for aa in list("AGPST")])
    assert out["nonsynonymous_indices"]["AGPST"] == expected


def test_initialize_nonsyn_groups_requires_grouping_keys():
    with pytest.raises(ValueError, match="Missing required key"):
        recoding.initialize_nonsyn_groups({"amino_acid_orders": np.array(["A"], dtype=object)})


@pytest.mark.parametrize("scheme_name", ["srchisq6", "kgbauto6"])
def test_initialize_nonsyn_groups_auto_recode_builds_six_states(scheme_name):
    g = _toy_auto_grouping_g()
    g["nonsyn_recode"] = scheme_name
    out = recoding.initialize_nonsyn_groups(g)
    assert len(out["nonsyn_state_orders"]) == 6
    aa_orders = [str(aa) for aa in out["amino_acid_orders"].tolist()]
    grouped = "".join([str(state) for state in out["nonsyn_state_orders"].tolist()])
    assert sorted(grouped) == sorted("".join(aa_orders))
    assert set(out["nonsyn_aa_to_state"].keys()) == set(aa_orders)


@pytest.mark.parametrize("scheme_name", ["srchisq6", "kgbauto6"])
def test_initialize_nonsyn_groups_auto_recode_threaded_matches_single_thread(scheme_name):
    g1 = _toy_auto_grouping_g()
    g1["nonsyn_recode"] = scheme_name
    g1["threads"] = 1
    g1["parallel_backend"] = "threading"
    g1["nonsyn_recode_parallel_min_starts_per_job"] = 1
    out1 = recoding.initialize_nonsyn_groups(g1)

    g2 = _toy_auto_grouping_g()
    g2["nonsyn_recode"] = scheme_name
    g2["threads"] = 4
    g2["parallel_backend"] = "threading"
    g2["nonsyn_recode_parallel_min_starts_per_job"] = 1
    out2 = recoding.initialize_nonsyn_groups(g2)

    assert out1["nonsyn_state_orders"].tolist() == out2["nonsyn_state_orders"].tolist()
    assert out1["nonsyn_recode_auto_score"] == pytest.approx(out2["nonsyn_recode_auto_score"], abs=1e-12)


def test_initialize_nonsyn_groups_auto_recode_is_deterministic():
    g1 = _toy_auto_grouping_g()
    g1["nonsyn_recode"] = "srchisq6"
    out1 = recoding.initialize_nonsyn_groups(g1)
    g2 = _toy_auto_grouping_g()
    g2["nonsyn_recode"] = "srchisq6"
    out2 = recoding.initialize_nonsyn_groups(g2)
    assert out1["nonsyn_state_orders"].tolist() == out2["nonsyn_state_orders"].tolist()


def test_random_bin_assignments_matches_sequential_generator():
    n_random = 200
    num_item = 20
    num_bin = 6
    seed = 31
    rng_batch = np.random.default_rng(seed=seed)
    out_batch = recoding._random_bin_assignments(
        num_item=num_item,
        num_bin=num_bin,
        rng=rng_batch,
        n_random=n_random,
    )
    rng_ref = np.random.default_rng(seed=seed)
    out_ref = np.vstack(
        [
            recoding._random_bin_assignment(num_item=num_item, num_bin=num_bin, rng=rng_ref)
            for _ in range(n_random)
        ]
    ).astype(np.int64, copy=False)
    assert out_batch.shape == (n_random, num_item)
    assert np.array_equal(out_batch, out_ref)


def test_resolve_auto_recode_parallel_n_jobs_uses_work_scale():
    g = {
        "threads": 8,
        "parallel_backend": "auto",
        "nonsyn_recode_parallel_min_total_starts": 30000,
        "nonsyn_recode_parallel_min_starts_per_job": 5000,
    }
    n1 = recoding._resolve_auto_recode_parallel_n_jobs(g=g, n_random=1000, work_scale=1)
    n2 = recoding._resolve_auto_recode_parallel_n_jobs(g=g, n_random=1000, work_scale=40)
    assert n1 == 1
    assert n2 > 1


def test_resolve_auto_recode_parallel_backend_prefers_threading_for_auto():
    g = {"parallel_backend": "auto"}
    out = recoding._resolve_auto_recode_parallel_backend(g=g, prefer_threading=True)
    assert out == "threading"


def test_resolve_auto_recode_chunk_factor_adaptive_defaults():
    assert recoding._resolve_auto_recode_chunk_factor(g={"threads": 8}, total_work_units=5_000_000) == 1
    assert recoding._resolve_auto_recode_chunk_factor(g={"threads": 8}, total_work_units=6_500_000) == 4
    assert recoding._resolve_auto_recode_chunk_factor(g={"threads": 8}, total_work_units=12_000_000) == 8


def test_resolve_auto_recode_chunk_factor_respects_user_setting():
    g = {"threads": 8, "parallel_chunk_factor": 2}
    out = recoding._resolve_auto_recode_chunk_factor(g=g, total_work_units=20_000_000)
    assert out == 2


def _estimate_empirical_transition_matrix_reference(aa_matrix, num_state):
    pair_counts = np.zeros((num_state, num_state), dtype=np.float64)
    num_taxa = aa_matrix.shape[0]
    for i in np.arange(num_taxa - 1):
        seq_i = aa_matrix[i, :]
        for j in np.arange(i + 1, num_taxa):
            seq_j = aa_matrix[j, :]
            valid = (seq_i >= 0) & (seq_j >= 0)
            if not np.any(valid):
                continue
            idx_i = seq_i[valid].astype(np.int64, copy=False)
            idx_j = seq_j[valid].astype(np.int64, copy=False)
            np.add.at(pair_counts, (idx_i, idx_j), 1.0)
            np.add.at(pair_counts, (idx_j, idx_i), 1.0)
    np.fill_diagonal(pair_counts, 0.0)
    pair_counts = pair_counts + recoding._AA_PSEUDOCOUNT
    np.fill_diagonal(pair_counts, 0.0)
    q = np.zeros_like(pair_counts)
    row_sum = pair_counts.sum(axis=1)
    valid_row = row_sum > 0
    q[valid_row, :] = pair_counts[valid_row, :] / row_sum[valid_row, np.newaxis]
    return q


def test_estimate_empirical_transition_matrix_matches_reference():
    rng = np.random.default_rng(seed=91)
    aa_matrix = rng.integers(low=-1, high=20, size=(37, 251), endpoint=False).astype(np.int16, copy=False)
    out = recoding._estimate_empirical_transition_matrix(aa_matrix=aa_matrix, num_state=20)
    ref = _estimate_empirical_transition_matrix_reference(aa_matrix=aa_matrix, num_state=20)
    assert np.allclose(out, ref, atol=1e-12, rtol=0.0)


def test_chisq_max_criterion_matches_reference_implementation():
    rng = np.random.default_rng(seed=3)
    n_taxa = 7
    n_state = 20
    n_bin = 6
    fmat = rng.random((n_taxa, n_state))
    fmat = fmat / fmat.sum(axis=1, keepdims=True)
    fr = rng.random((n_state,))
    fr = fr / fr.sum()
    nsitev = rng.integers(low=50, high=500, size=(n_taxa,), endpoint=False).astype(np.float64)
    bins = rng.integers(low=0, high=n_bin, size=(n_state,), endpoint=False).astype(np.int64)
    # Ensure non-empty bins for reference stability.
    bins[:n_bin] = np.arange(n_bin, dtype=np.int64)

    out = recoding._chisq_max_criterion(bin_assignment=bins, fmat=fmat, fr=fr, nsitev=nsitev, num_bin=n_bin)

    frb = np.bincount(bins, weights=fr, minlength=n_bin).astype(np.float64)
    ref = 0.0
    for k in range(n_taxa):
        frt = np.bincount(bins, weights=fmat[k, :], minlength=n_bin).astype(np.float64)
        chisq = float((((frt - frb) ** 2) / frb).sum() * nsitev[k])
        if chisq > ref:
            ref = chisq
    assert out == pytest.approx(ref, abs=1e-12)


def _hill_climb_bins_chisq_reference(initial_bins, num_bin, fmat, fr, nsitev, tol=1e-8):
    bins = np.asarray(initial_bins, dtype=np.int64).copy()
    counts = np.bincount(bins, minlength=num_bin).astype(np.int64, copy=False)
    frb = np.bincount(bins, weights=fr, minlength=num_bin).astype(np.float64, copy=False)
    n_taxa = int(fmat.shape[0])
    frt = np.zeros((n_taxa, num_bin), dtype=np.float64)
    for b in range(num_bin):
        mask = bins == b
        if np.any(mask):
            frt[:, b] = fmat[:, mask].sum(axis=1)
    term = ((frt - frb[np.newaxis, :]) ** 2) / frb[np.newaxis, :]
    taxon_sum = term.sum(axis=1)
    crit = float((taxon_sum * nsitev).max())
    while True:
        improved = False
        for el in range(int(bins.shape[0])):
            src = int(bins[el])
            if counts[src] <= 1:
                continue
            fr_el = float(fr[el])
            fvec = fmat[:, el]
            for dst in range(int(num_bin)):
                if dst == src:
                    continue
                frb_src_new = float(frb[src] - fr_el)
                frb_dst_new = float(frb[dst] + fr_el)
                if (frb_src_new <= 0.0) or (frb_dst_new <= 0.0):
                    continue
                old_src = term[:, src]
                old_dst = term[:, dst]
                frt_src_new = frt[:, src] - fvec
                frt_dst_new = frt[:, dst] + fvec
                new_src = ((frt_src_new - frb_src_new) ** 2) / frb_src_new
                new_dst = ((frt_dst_new - frb_dst_new) ** 2) / frb_dst_new
                taxon_sum_new = taxon_sum - old_src - old_dst + new_src + new_dst
                crit_new = float((taxon_sum_new * nsitev).max())
                if crit_new < (crit - tol):
                    bins[el] = dst
                    counts[src] -= 1
                    counts[dst] += 1
                    frb[src] = frb_src_new
                    frb[dst] = frb_dst_new
                    frt[:, src] = frt_src_new
                    frt[:, dst] = frt_dst_new
                    term[:, src] = new_src
                    term[:, dst] = new_dst
                    taxon_sum = taxon_sum_new
                    crit = crit_new
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return bins, crit


def test_hill_climb_bins_chisq_matches_reference_implementation():
    rng = np.random.default_rng(seed=17)
    n_taxa = 11
    n_state = 20
    n_bin = 6
    for _ in range(10):
        fmat = rng.random((n_taxa, n_state))
        fmat = fmat / fmat.sum(axis=1, keepdims=True)
        fr = rng.random((n_state,))
        fr = fr / fr.sum()
        nsitev = rng.integers(low=50, high=600, size=(n_taxa,), endpoint=False).astype(np.float64)
        bins = rng.integers(low=0, high=n_bin, size=(n_state,), endpoint=False).astype(np.int64)
        bins[:n_bin] = np.arange(n_bin, dtype=np.int64)

        out_bins, out_crit = recoding._hill_climb_bins_chisq(
            initial_bins=bins,
            num_bin=n_bin,
            fmat=fmat,
            fr=fr,
            nsitev=nsitev,
        )
        ref_bins, ref_crit = _hill_climb_bins_chisq_reference(
            initial_bins=bins,
            num_bin=n_bin,
            fmat=fmat,
            fr=fr,
            nsitev=nsitev,
        )
        assert out_bins.tolist() == ref_bins.tolist()
        assert out_crit == pytest.approx(ref_crit, abs=1e-12)


def test_hill_climb_bins_chisq_cython_matches_python_when_available():
    cython_fn = None
    if getattr(recoding, "recoding_cy", None) is not None:
        cython_fn = getattr(recoding.recoding_cy, "hill_climb_bins_chisq_double", None)
    if cython_fn is None:
        pytest.skip("recoding_cy is unavailable")
    rng = np.random.default_rng(seed=23)
    n_taxa = 13
    n_state = 20
    n_bin = 6
    for _ in range(8):
        fmat = rng.random((n_taxa, n_state))
        fmat = np.ascontiguousarray(fmat / fmat.sum(axis=1, keepdims=True), dtype=np.float64)
        fr = np.ascontiguousarray(rng.random((n_state,)), dtype=np.float64)
        fr = fr / fr.sum()
        nsitev = np.ascontiguousarray(
            rng.integers(low=50, high=800, size=(n_taxa,), endpoint=False).astype(np.float64),
            dtype=np.float64,
        )
        bins = rng.integers(low=0, high=n_bin, size=(n_state,), endpoint=False).astype(np.int64)
        bins[:n_bin] = np.arange(n_bin, dtype=np.int64)
        bins = np.ascontiguousarray(bins, dtype=np.int64)

        py_bins, py_crit = recoding._hill_climb_bins_chisq(
            initial_bins=bins,
            num_bin=n_bin,
            fmat=fmat,
            fr=fr,
            nsitev=nsitev,
        )
        cy_bins, cy_crit = cython_fn(
            initial_bins=bins,
            num_bin=n_bin,
            fmat=fmat,
            fr=fr,
            nsitev=nsitev,
            obj_eps=1e-8,
        )
        assert cy_bins.tolist() == py_bins.tolist()
        assert cy_crit == pytest.approx(py_crit, abs=1e-12)


def test_search_initial_bins_chunk_chisq_cython_matches_python_when_available():
    cython_fn = None
    if getattr(recoding, "recoding_cy", None) is not None:
        cython_fn = getattr(recoding.recoding_cy, "search_initial_bins_chunk_chisq_double", None)
    if cython_fn is None:
        pytest.skip("recoding_cy is unavailable")
    rng = np.random.default_rng(seed=97)
    n_taxa = 23
    n_state = 20
    n_bin = 6
    n_start = 81
    fmat = np.ascontiguousarray(rng.random((n_taxa, n_state)), dtype=np.float64)
    fmat = np.ascontiguousarray(fmat / fmat.sum(axis=1, keepdims=True), dtype=np.float64)
    fr = np.ascontiguousarray(rng.random((n_state,)), dtype=np.float64)
    fr = fr / fr.sum()
    nsitev = np.ascontiguousarray(
        rng.integers(low=40, high=600, size=(n_taxa,), endpoint=False).astype(np.float64),
        dtype=np.float64,
    )
    initial_bins_chunk = np.vstack(
        [recoding._random_bin_assignment(num_item=n_state, num_bin=n_bin, rng=rng) for _ in range(n_start)]
    ).astype(np.int64, copy=False)
    start_index = 17

    py_bins, py_crit, py_start = recoding._search_initial_bins_chunk_chisq(
        initial_bins_chunk=initial_bins_chunk,
        start_index=start_index,
        num_bin=n_bin,
        fmat=fmat,
        fr=fr,
        nsitev=nsitev,
        use_cython=False,
    )
    cy_bins, cy_crit, cy_offset = cython_fn(
        initial_bins_chunk=initial_bins_chunk,
        num_bin=n_bin,
        fmat=fmat,
        fr=fr,
        nsitev=nsitev,
        obj_eps=1e-8,
    )
    assert np.array_equal(np.asarray(cy_bins, dtype=np.int64), np.asarray(py_bins, dtype=np.int64))
    assert cy_crit == pytest.approx(py_crit, abs=1e-12)
    assert int(start_index + int(cy_offset)) == int(py_start)


def test_random_bin_assignments_cython_matches_python_when_available():
    cython_fn = None
    if getattr(recoding, "recoding_cy", None) is not None:
        cython_fn = getattr(recoding.recoding_cy, "random_bin_assignments_int64", None)
    if cython_fn is None:
        pytest.skip("recoding_cy is unavailable")
    num_item = 20
    num_bin = 6
    n_random = 300
    seed = 109
    out_cy = cython_fn(
        num_item=num_item,
        num_bin=num_bin,
        rng=np.random.default_rng(seed=seed),
        n_random=n_random,
    )
    # Build a true sequential Python reference with identical RNG consumption.
    rng_ref = np.random.default_rng(seed=seed)
    out_ref = np.vstack(
        [
            recoding._random_bin_assignment(num_item=num_item, num_bin=num_bin, rng=rng_ref)
            for _ in range(n_random)
        ]
    ).astype(np.int64, copy=False)
    assert np.array_equal(np.asarray(out_cy, dtype=np.int64), out_ref)


def test_conductance_criterion_matches_reference_implementation():
    rng = np.random.default_rng(seed=5)
    n_state = 20
    n_bin = 6
    bins = rng.integers(low=0, high=n_bin, size=(n_state,), endpoint=False).astype(np.int64)
    bins[:n_bin] = np.arange(n_bin, dtype=np.int64)
    pi = rng.random((n_state,))
    pi = pi / pi.sum()
    q = rng.random((n_state, n_state))
    np.fill_diagonal(q, 0.0)
    row_sum = q.sum(axis=1)
    q = q / row_sum[:, np.newaxis]
    weighted_q = pi[:, np.newaxis] * q

    out = recoding._conductance_criterion(bin_assignment=bins, pi=pi, weighted_q=weighted_q, num_bin=n_bin)

    cap = np.zeros((n_bin,), dtype=np.float64)
    flow = np.zeros((n_bin, n_bin), dtype=np.float64)
    for i in range(n_state):
        bi = int(bins[i])
        cap[bi] += pi[i]
        for j in range(n_state):
            bj = int(bins[j])
            if bi == bj:
                continue
            flow[bi, bj] += pi[i] * q[i, j]
    phi = flow / cap[:, np.newaxis]
    np.fill_diagonal(phi, 0.0)
    ref = float(phi.sum())
    assert out == pytest.approx(ref, abs=1e-12)


def _hill_climb_bins_conductance_reference(initial_bins, num_bin, pi, weighted_q, tol=1e-8):
    bins = np.asarray(initial_bins, dtype=np.int64).copy()
    counts = np.bincount(bins, minlength=num_bin).astype(np.int64, copy=False)
    membership = np.zeros((bins.shape[0], num_bin), dtype=np.float64)
    membership[np.arange(bins.shape[0]), bins] = 1.0
    flow = membership.T @ weighted_q @ membership
    cap = np.bincount(bins, weights=pi, minlength=num_bin).astype(np.float64, copy=False)
    row_sum = flow.sum(axis=1)
    diag = np.diag(flow)
    crit = float(((row_sum - diag) / cap).sum())
    while True:
        improved = False
        for el in range(int(bins.shape[0])):
            src = int(bins[el])
            if counts[src] <= 1:
                continue
            row_bin_sum = np.bincount(bins, weights=weighted_q[el, :], minlength=num_bin).astype(np.float64, copy=False)
            col_bin_sum = np.bincount(bins, weights=weighted_q[:, el], minlength=num_bin).astype(np.float64, copy=False)
            pi_el = float(pi[el])
            for dst in range(int(num_bin)):
                if dst == src:
                    continue
                cap_tmp = cap.copy()
                cap_tmp[src] -= pi_el
                cap_tmp[dst] += pi_el
                if cap_tmp[src] <= 0:
                    continue
                flow_tmp = flow.copy()
                flow_tmp[src, :] -= row_bin_sum
                flow_tmp[dst, :] += row_bin_sum
                flow_tmp[:, src] -= col_bin_sum
                flow_tmp[:, dst] += col_bin_sum
                row_sum_tmp = flow_tmp.sum(axis=1)
                diag_tmp = np.diag(flow_tmp)
                crit_new = float(((row_sum_tmp - diag_tmp) / cap_tmp).sum())
                if crit_new < (crit - tol):
                    bins[el] = dst
                    counts[src] -= 1
                    counts[dst] += 1
                    cap = cap_tmp
                    flow = flow_tmp
                    crit = crit_new
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return bins, crit


def test_hill_climb_bins_conductance_matches_reference_implementation():
    rng = np.random.default_rng(seed=11)
    n_state = 20
    n_bin = 6
    for _ in range(8):
        bins = rng.integers(low=0, high=n_bin, size=(n_state,), endpoint=False).astype(np.int64)
        bins[:n_bin] = np.arange(n_bin, dtype=np.int64)
        pi = rng.random((n_state,))
        pi = pi / pi.sum()
        q = rng.random((n_state, n_state))
        np.fill_diagonal(q, 0.0)
        q = q / q.sum(axis=1, keepdims=True)
        weighted_q = pi[:, np.newaxis] * q

        out_bins, out_crit = recoding._hill_climb_bins_conductance(
            initial_bins=bins,
            num_bin=n_bin,
            pi=pi,
            weighted_q=weighted_q,
        )
        ref_bins, ref_crit = _hill_climb_bins_conductance_reference(
            initial_bins=bins,
            num_bin=n_bin,
            pi=pi,
            weighted_q=weighted_q,
        )
        assert out_bins.tolist() == ref_bins.tolist()
        assert out_crit == pytest.approx(ref_crit, abs=1e-12)


def test_hill_climb_bins_conductance_cython_matches_python_when_available():
    cython_fn = None
    if getattr(recoding, "recoding_cy", None) is not None:
        cython_fn = getattr(recoding.recoding_cy, "hill_climb_bins_conductance_double", None)
    if cython_fn is None:
        pytest.skip("recoding_cy is unavailable")
    rng = np.random.default_rng(seed=29)
    n_state = 20
    n_bin = 6
    for _ in range(8):
        bins = rng.integers(low=0, high=n_bin, size=(n_state,), endpoint=False).astype(np.int64)
        bins[:n_bin] = np.arange(n_bin, dtype=np.int64)
        bins = np.ascontiguousarray(bins, dtype=np.int64)
        pi = np.ascontiguousarray(rng.random((n_state,)), dtype=np.float64)
        pi = pi / pi.sum()
        q = rng.random((n_state, n_state))
        np.fill_diagonal(q, 0.0)
        q = q / q.sum(axis=1, keepdims=True)
        weighted_q = np.ascontiguousarray(pi[:, np.newaxis] * q, dtype=np.float64)

        py_bins, py_crit = recoding._hill_climb_bins_conductance(
            initial_bins=bins,
            num_bin=n_bin,
            pi=pi,
            weighted_q=weighted_q,
        )
        cy_bins, cy_crit = cython_fn(
            initial_bins=bins,
            num_bin=n_bin,
            pi=pi,
            weighted_q=weighted_q,
            obj_eps=1e-8,
        )
        assert cy_bins.tolist() == py_bins.tolist()
        assert cy_crit == pytest.approx(py_crit, abs=1e-12)


def test_search_initial_bins_chunk_conductance_cython_matches_python_when_available():
    cython_fn = None
    if getattr(recoding, "recoding_cy", None) is not None:
        cython_fn = getattr(recoding.recoding_cy, "search_initial_bins_chunk_conductance_double", None)
    if cython_fn is None:
        pytest.skip("recoding_cy is unavailable")
    rng = np.random.default_rng(seed=121)
    n_state = 20
    n_bin = 6
    n_start = 67
    pi = np.ascontiguousarray(rng.random((n_state,)), dtype=np.float64)
    pi = pi / pi.sum()
    q = rng.random((n_state, n_state))
    np.fill_diagonal(q, 0.0)
    q = q / q.sum(axis=1, keepdims=True)
    weighted_q = np.ascontiguousarray(pi[:, np.newaxis] * q, dtype=np.float64)
    initial_bins_chunk = np.vstack(
        [recoding._random_bin_assignment(num_item=n_state, num_bin=n_bin, rng=rng) for _ in range(n_start)]
    ).astype(np.int64, copy=False)
    start_index = 23

    py_bins, py_crit, py_start = recoding._search_initial_bins_chunk_conductance(
        initial_bins_chunk=initial_bins_chunk,
        start_index=start_index,
        num_bin=n_bin,
        pi=pi,
        weighted_q=weighted_q,
        use_cython=False,
    )
    cy_bins, cy_crit, cy_offset = cython_fn(
        initial_bins_chunk=initial_bins_chunk,
        num_bin=n_bin,
        pi=pi,
        weighted_q=weighted_q,
        obj_eps=1e-8,
    )
    assert np.array_equal(np.asarray(cy_bins, dtype=np.int64), np.asarray(py_bins, dtype=np.int64))
    assert cy_crit == pytest.approx(py_crit, abs=1e-12)
    assert int(start_index + int(cy_offset)) == int(py_start)


def test_write_nonsyn_recoding_table_writes_non_none_scheme(tmp_path):
    g = _toy_grouping_g()
    g["nonsyn_recode"] = "dayhoff6"
    g = recoding.initialize_nonsyn_groups(g)
    output_path = tmp_path / "csubst_nonsyn_recoding.tsv"
    returned = recoding.write_nonsyn_recoding_table(g, output_path=str(output_path))
    assert returned == str(output_path)
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("recode\tstate_id\tstate_label")
    assert len(lines) == 1 + len(g["amino_acid_orders"])
    assert any([line.split("\t")[4] == "A" for line in lines[1:]])


def test_write_nonsyn_recoding_table_skips_no(tmp_path):
    g = _toy_grouping_g()
    g["nonsyn_recode"] = "no"
    g = recoding.initialize_nonsyn_groups(g)
    output_path = tmp_path / "csubst_nonsyn_recoding.tsv"
    returned = recoding.write_nonsyn_recoding_table(g, output_path=str(output_path))
    assert returned is None
    assert output_path.exists() is False


def test_write_nonsyn_recoding_pca_plot_writes_png_for_fixed_recode(tmp_path):
    g = _toy_grouping_g()
    g["nonsyn_recode"] = "dayhoff6"
    g = recoding.initialize_nonsyn_groups(g)
    output_path = tmp_path / "csubst_nonsyn_recoding_pca.png"
    returned = recoding.write_nonsyn_recoding_pca_plot(g, output_path=str(output_path))
    assert returned == str(output_path)
    assert output_path.exists() is True
    assert output_path.stat().st_size > 0
    img = mpimg.imread(str(output_path))
    assert img.shape[0] == 720
    assert img.shape[1] == 720


def test_write_nonsyn_recoding_pca_plot_writes_png_for_auto_recode(tmp_path):
    g = _toy_auto_grouping_g()
    g["nonsyn_recode"] = "srchisq6"
    g = recoding.initialize_nonsyn_groups(g)
    output_path = tmp_path / "csubst_nonsyn_recoding_pca.png"
    returned = recoding.write_nonsyn_recoding_pca_plot(g, output_path=str(output_path))
    assert returned == str(output_path)
    assert output_path.exists() is True
    assert output_path.stat().st_size > 0


def test_get_scheme_groups_for_pca_includes_auto_schemes_when_data_available():
    g = _toy_auto_grouping_g()
    g["nonsyn_recode"] = "dayhoff6"
    g = recoding.initialize_nonsyn_groups(g)
    groups_by_scheme = recoding._get_scheme_groups_for_pca(g)
    assert "no" in groups_by_scheme
    assert groups_by_scheme["no"] == tuple(list("ACDEFGHIKLMNPQRSTVWY"))
    assert "srchisq6" in groups_by_scheme
    assert "kgbauto6" in groups_by_scheme
    assert len(groups_by_scheme["srchisq6"]) == 6
    assert len(groups_by_scheme["kgbauto6"]) == 6


def test_write_nonsyn_recoding_pca_plot_writes_png_for_no(tmp_path):
    g = _toy_grouping_g()
    g["nonsyn_recode"] = "no"
    g = recoding.initialize_nonsyn_groups(g)
    output_path = tmp_path / "csubst_nonsyn_recoding_pca.png"
    returned = recoding.write_nonsyn_recoding_pca_plot(g, output_path=str(output_path))
    assert returned == str(output_path)
    assert output_path.exists() is True
    assert output_path.stat().st_size > 0
