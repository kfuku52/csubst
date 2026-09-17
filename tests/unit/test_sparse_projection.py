from sparse_fixtures import large_sparse_reducer_tensor as _large_sparse_reducer_tensor
from sparse_fixtures import toy_reducer_tensor as _toy_reducer_tensor

import itertools
import numpy as np
import scipy.sparse as sp
import pytest

from csubst import substitution








def test_sub_tensor2cb_sparse_gram_fastpath_matches_python_fallback_with_unsorted_pairs(monkeypatch):
    sparse_tensor = _large_sparse_reducer_tensor(num_branch=40, num_site=12)
    ids = np.array(list(itertools.combinations(range(40), 2)), dtype=np.int64)
    ids_unsorted = ids.copy()
    ids_unsorted[1::2, :] = ids_unsorted[1::2, ::-1]
    selected = ["any2any", "spe2any", "any2spe", "spe2spe"]

    monkeypatch.setattr(substitution, "_can_use_sparse_cb_projection_gram", lambda *args, **kwargs: False)
    monkeypatch.setattr(substitution, "_can_use_sparse_cb_summary_gram", lambda *args, **kwargs: False)
    expected = substitution.sub_tensor2cb_sparse(
        ids_unsorted,
        sparse_tensor,
        mmap=False,
        df_mmap=None,
        mmap_start=0,
        float_type=np.float64,
        selected_base_stats=selected,
    )

    monkeypatch.setattr(substitution, "_can_use_sparse_cb_projection_gram", lambda *args, **kwargs: False)
    monkeypatch.setattr(substitution, "_can_use_sparse_cb_summary_gram", lambda *args, **kwargs: True)
    observed = substitution.sub_tensor2cb_sparse(
        ids_unsorted,
        sparse_tensor,
        mmap=False,
        df_mmap=None,
        mmap_start=0,
        float_type=np.float64,
        selected_base_stats=selected,
    )
    np.testing.assert_allclose(observed, expected, atol=1e-12)


def test_sub_tensor2cb_sparse_projection_gram_matches_fallback_all_stats(monkeypatch):
    sparse_tensor = _large_sparse_reducer_tensor(num_branch=40, num_site=12)
    ids = np.array(list(itertools.combinations(range(40), 2)), dtype=np.int64)
    ids[1::2, :] = ids[1::2, ::-1]
    selected = ["any2any", "spe2any", "any2spe", "spe2spe"]

    monkeypatch.setattr(substitution, "_can_use_sparse_cb_projection_gram", lambda *args, **kwargs: False)
    monkeypatch.setattr(substitution, "_can_use_sparse_cb_summary_gram", lambda *args, **kwargs: False)
    expected = substitution.sub_tensor2cb_sparse(
        ids,
        sparse_tensor,
        float_type=np.float64,
        selected_base_stats=selected,
    )

    monkeypatch.setattr(substitution, "_can_use_sparse_cb_projection_gram", lambda *args, **kwargs: True)
    observed = substitution.sub_tensor2cb_sparse(
        ids,
        sparse_tensor,
        float_type=np.float64,
        selected_base_stats=selected,
    )
    np.testing.assert_allclose(observed, expected, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 5])
def test_cython_csr_gram_matches_dense_randomized(seed):
    cython_module = substitution.substitution_sparse_cy
    if cython_module is None or not hasattr(cython_module, "calc_csr_gram_dense_double"):
        pytest.skip("Cython CSR Gram extension is unavailable")
    rng = np.random.default_rng(seed)
    num_row = int(rng.integers(2, 18))
    num_column = int(rng.integers(1, 180))
    dense = rng.normal(size=(num_row, num_column))
    dense[rng.random(size=dense.shape) > rng.uniform(0.02, 0.8)] = 0.0
    projection = sp.csr_matrix(dense, dtype=np.float64)

    observed = cython_module.calc_csr_gram_dense_double(
        projection.indptr,
        projection.indices,
        projection.data,
        num_column,
    )

    np.testing.assert_allclose(observed, dense @ dense.T, rtol=1e-12, atol=1e-12)


def test_sparse_projection_gram_is_disabled_for_nonfinite_values():
    dense = _large_sparse_reducer_tensor(num_branch=40, num_site=12).to_dense()
    dense[0, 0, 0, 0, 1] = np.nan
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array(list(itertools.combinations(range(40), 2)), dtype=np.int64)
    assert not substitution._can_use_sparse_cb_projection_gram(ids, sparse_tensor)




def test_sub_tensor2cb_sparse_projection_failure_warns_and_uses_bounded_fallback(monkeypatch):
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array([[2, 0], [1, 2]], dtype=np.int64)
    selected = ["any2any", "any2spe"]

    monkeypatch.setattr(substitution, "_can_use_sparse_projection_product", lambda *args, **kwargs: False)
    expected = substitution.sub_tensor2cb_sparse(
        ids,
        sparse_tensor,
        mmap=False,
        df_mmap=None,
        mmap_start=0,
        float_type=np.float64,
        selected_base_stats=selected,
    )

    monkeypatch.setattr(substitution, "_CYTHON_FALLBACK_WARNED", set())
    monkeypatch.setattr(substitution, "_can_use_sparse_projection_product", lambda *args, **kwargs: True)

    def _raise_projection(*args, **kwargs):
        raise RuntimeError("forced-projection-failure")

    monkeypatch.setattr(substitution, "_calc_sparse_projection_products", _raise_projection)
    with pytest.warns(RuntimeWarning, match='Sparse fast path "sub_tensor2cb_sparse_projection_product" failed'):
        observed = substitution.sub_tensor2cb_sparse(
            ids,
            sparse_tensor,
            mmap=False,
            df_mmap=None,
            mmap_start=0,
            float_type=np.float64,
            selected_base_stats=selected,
        )
    np.testing.assert_allclose(observed, expected, atol=1e-12)
