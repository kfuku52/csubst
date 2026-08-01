import numpy as np
import pytest

from csubst import recoding


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
