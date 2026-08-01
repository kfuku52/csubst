import numpy as np
import pytest

from csubst import recoding


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
