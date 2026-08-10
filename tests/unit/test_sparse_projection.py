from sparse_fixtures import large_sparse_reducer_tensor as _large_sparse_reducer_tensor
from sparse_fixtures import toy_reducer_tensor as _toy_reducer_tensor

import itertools
import numpy as np
import scipy.sparse as sp
import pytest

from csubst import substitution








def test_sub_tensor2cb_sparse_gram_fastpath_matches_python_fallback(monkeypatch):
    sparse_tensor = _large_sparse_reducer_tensor(num_branch=40, num_site=12)
    ids = np.array(list(itertools.combinations(range(40), 2)), dtype=np.int64)
    selected = ["any2any", "any2spe"]

    monkeypatch.setattr(substitution, "_can_use_sparse_cb_projection_gram", lambda *args, **kwargs: False)
    monkeypatch.setattr(substitution, "_can_use_sparse_cb_summary_gram", lambda *args, **kwargs: False)
    expected = substitution.sub_tensor2cb_sparse(
        ids,
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
        ids,
        sparse_tensor,
        mmap=False,
        df_mmap=None,
        mmap_start=0,
        float_type=np.float64,
        selected_base_stats=selected,
    )
    np.testing.assert_allclose(observed, expected, atol=1e-12)


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


def test_sparse_projection_gram_switches_sparse_and_dense_per_projection(monkeypatch):
    row_ids = np.array([0, 1], dtype=np.int64)
    col_ids = np.array([1, 0], dtype=np.int64)
    sparse_projection = sp.csr_matrix(
        np.array([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    )
    dense_projection = sp.csr_matrix(np.ones((2, 4), dtype=np.float64))
    monkeypatch.setattr(substitution, "_SPARSE_CB_PROJECTION_DENSE_CUTOFF", 0.5)

    sparse_values, _density, sparse_backend = substitution._calc_projection_gram_pair_values(
        projection=sparse_projection,
        row_ids=row_ids,
        col_ids=col_ids,
    )
    dense_values, _density, dense_backend = substitution._calc_projection_gram_pair_values(
        projection=dense_projection,
        row_ids=row_ids,
        col_ids=col_ids,
    )

    assert sparse_backend == "sparse"
    assert dense_backend == "dense"
    np.testing.assert_allclose(sparse_values, [2.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(dense_values, [4.0, 4.0], atol=1e-12)


@pytest.mark.parametrize("seed", range(8))
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


def test_adaptive_projection_gram_uses_pair_coverage_and_density():
    num_branch = 20
    dense_projection = sp.csr_matrix(np.ones((num_branch, 10), dtype=np.float64))
    few_pairs = np.array(list(itertools.combinations(range(num_branch), 2))[:10], dtype=np.int64)
    use_gram, coverage, threshold = substitution._should_use_adaptive_projection_gram(
        few_pairs,
        dense_projection,
    )
    assert not use_gram
    assert coverage < threshold

    all_pairs = np.array(list(itertools.combinations(range(num_branch), 2)), dtype=np.int64)
    use_gram, coverage, threshold = substitution._should_use_adaptive_projection_gram(
        all_pairs,
        dense_projection,
    )
    assert use_gram
    assert coverage >= threshold

    sparse_projection = sp.csr_matrix(
        (np.ones(2), ([0, 1], [0, 1])),
        shape=(num_branch, 10),
    )
    use_gram, coverage, threshold = substitution._should_use_adaptive_projection_gram(
        few_pairs,
        sparse_projection,
    )
    assert use_gram
    assert coverage >= threshold


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
