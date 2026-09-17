import numpy as np
import pytest

from csubst import recoding


def test_hill_climb_bins_conductance_cython_matches_python_when_available():
    cython_fn = None
    if getattr(recoding, "recoding_cy", None) is not None:
        cython_fn = getattr(recoding.recoding_cy, "hill_climb_bins_conductance_double", None)
    if cython_fn is None:
        pytest.skip("recoding_cy is unavailable")
    rng = np.random.default_rng(seed=29)
    n_state = 20
    n_bin = 6
    for _ in range(2):
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
