import numpy as np
import pytest

from csubst import recoding


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
    for _ in range(2):
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
    n_random = 30
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
