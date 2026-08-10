from sparse_fixtures import toy_dense_tensor as _toy_dense_tensor

import numpy as np
import scipy.sparse as sp
import pytest

from csubst import substitution
from csubst import substitution_sparse


try:
    from csubst import substitution_sparse_cy
except ImportError:  # pragma: no cover - optional Cython extension
    substitution_sparse_cy = None




def test_dense_sparse_roundtrip_preserves_values_and_shape():
    dense = _toy_dense_tensor()
    sparse_tensor = substitution_sparse.dense_to_sparse_substitution_tensor(dense)
    restored = substitution_sparse.sparse_to_dense_substitution_tensor(sparse_tensor)

    assert sparse_tensor.shape == dense.shape
    assert sparse_tensor.nnz == int(np.count_nonzero(dense))
    np.testing.assert_allclose(restored, dense, atol=1e-12)


def test_sparse_tensor_reports_payload_and_dense_storage_bytes():
    dense = _toy_dense_tensor()
    sparse_tensor = substitution_sparse.dense_to_sparse_substitution_tensor(dense)
    expected_payload = (
        sparse_tensor.matrix.data.nbytes
        + sparse_tensor.matrix.indices.nbytes
        + sparse_tensor.matrix.indptr.nbytes
    )
    assert sparse_tensor.nbytes == expected_payload
    assert sparse_tensor.dense_nbytes == dense.nbytes
    assert sparse_tensor.compression_ratio == pytest.approx(dense.nbytes / expected_payload)


def test_clear_sparse_cb_projection_cache_releases_cached_payload():
    sparse_tensor = substitution_sparse.dense_to_sparse_substitution_tensor(_toy_dense_tensor())
    projection = substitution._get_sparse_cb_projection(sparse_tensor, stat="any2spe")
    expected_nbytes = projection.data.nbytes + projection.indices.nbytes + projection.indptr.nbytes

    released_nbytes = substitution.clear_sparse_cb_projection_cache(sparse_tensor)

    assert released_nbytes == expected_nbytes
    assert sparse_tensor._cb_sparse_projection_cache == {}
    assert substitution.clear_sparse_cb_projection_cache(sparse_tensor) == 0


def test_sparse_tensor_blocks_and_csr_payload_are_read_only():
    sparse_tensor = substitution_sparse.dense_to_sparse_substitution_tensor(_toy_dense_tensor())
    key = next(iter(sparse_tensor.blocks))
    mat = sparse_tensor.blocks[key]

    with pytest.raises(TypeError):
        sparse_tensor.blocks[(0, 0, 0)] = sp.csr_matrix(mat.shape)
    with pytest.raises(ValueError):
        mat.data[0] = 0.0


def test_dense_to_sparse_cython_accepts_read_only_input():
    if substitution_sparse_cy is None:
        pytest.skip("Cython substitution_sparse fast path is unavailable")
    dense = _toy_dense_tensor()
    dense.setflags(write=False)
    sparse_tensor = substitution_sparse.dense_to_sparse_substitution_tensor(dense)
    np.testing.assert_allclose(sparse_tensor.to_dense(), dense, atol=1e-12)


def test_dense_to_sparse_applies_tolerance():
    dense = _toy_dense_tensor()
    nnz_before = int(np.count_nonzero(dense))
    dense[0, 0, 0, 2, 2] = 1e-12
    sparse_tensor = substitution_sparse.dense_to_sparse_substitution_tensor(dense, tol=1e-9)
    restored = sparse_tensor.to_dense()

    assert sparse_tensor.nnz == nnz_before
    assert restored[0, 0, 0, 2, 2] == 0


def test_dense_to_sparse_preserves_nan_values_with_tolerance():
    dense = np.zeros((2, 2, 1, 1, 1), dtype=np.float64)
    dense[0, 0, 0, 0, 0] = np.nan
    dense[1, 1, 0, 0, 0] = 1e-12

    sparse_tensor = substitution_sparse.dense_to_sparse_substitution_tensor(dense, tol=1e-9)
    restored = sparse_tensor.to_dense()

    assert np.isnan(restored[0, 0, 0, 0, 0])
    assert restored[1, 1, 0, 0, 0] == 0


def test_sparse_sum_matches_dense_without_materializing_full_tensor(monkeypatch):
    dense = _toy_dense_tensor()
    dense[2, 2, 0, 1, 2] = np.nan
    sparse_tensor = substitution_sparse.dense_to_sparse_substitution_tensor(dense, tol=0)
    monkeypatch.setattr(
        substitution_sparse.SparseSubstitutionTensor,
        "to_dense",
        lambda self: (_ for _ in ()).throw(AssertionError("sum should not call to_dense")),
    )

    axes_to_check = [
        None,
        0,
        1,
        2,
        3,
        4,
        -1,
        (0, 1),
        (0, 2, 4),
        (1, 3, 4),
        (2, 3, 4),
        (0, 1, 2, 3, 4),
    ]
    for axis in axes_to_check:
        observed = sparse_tensor.sum(axis=axis)
        expected = dense.sum(axis=axis)
        np.testing.assert_allclose(observed, expected, atol=1e-12, equal_nan=True)


def test_sparse_sum_bool_tensor_matches_numpy_count_dtype():
    dense = _toy_dense_tensor() > 0
    sparse_tensor = substitution_sparse.dense_to_sparse_substitution_tensor(dense, tol=0)

    observed = sparse_tensor.sum(axis=(0, 1))
    expected = dense.sum(axis=(0, 1))

    assert observed.dtype == expected.dtype
    np.testing.assert_array_equal(observed, expected)




def test_sparse_projections_match_dense_reductions():
    dense = _toy_dense_tensor()
    sparse_tensor = substitution_sparse.SparseSubstitutionTensor.from_dense(dense)

    for sg in range(dense.shape[2]):
        observed_any2any = sparse_tensor.project_any2any(sg).toarray()
        expected_any2any = dense[:, :, sg, :, :].sum(axis=(2, 3))
        np.testing.assert_allclose(observed_any2any, expected_any2any, atol=1e-12)

        for a in range(dense.shape[3]):
            observed_spe2any = sparse_tensor.project_spe2any(sg, a).toarray()
            expected_spe2any = dense[:, :, sg, a, :].sum(axis=2)
            np.testing.assert_allclose(observed_spe2any, expected_spe2any, atol=1e-12)

        for d in range(dense.shape[4]):
            observed_any2spe = sparse_tensor.project_any2spe(sg, d).toarray()
            expected_any2spe = dense[:, :, sg, :, d].sum(axis=2)
            np.testing.assert_allclose(observed_any2spe, expected_any2spe, atol=1e-12)


def test_packed_sitewise_max_cython_matches_python_fallback(monkeypatch):
    if substitution_sparse_cy is None or not hasattr(
        substitution_sparse_cy,
        "scan_packed_sitewise_max_row_double",
    ):
        pytest.skip("Cython packed sitewise scan is unavailable")
    sparse_tensor = substitution_sparse.SparseSubstitutionTensor.from_dense(_toy_dense_tensor())
    observed = substitution._get_sparse_branch_sitewise_max_indices(
        sparse_tensor,
        branch_id=0,
        min_sitewise_pp=0.1,
    )
    monkeypatch.setattr(substitution, "substitution_sparse_cy", None)
    expected = substitution._get_sparse_branch_sitewise_max_indices(
        sparse_tensor,
        branch_id=0,
        min_sitewise_pp=0.1,
    )
    for observed_array, expected_array in zip(observed, expected):
        np.testing.assert_array_equal(observed_array, expected_array)


def test_substitution_helpers_convert_dense_and_sparse():
    dense = _toy_dense_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    restored = substitution.sparse_to_dense_sub_tensor(sparse_tensor)
    np.testing.assert_allclose(restored, dense, atol=1e-12)


def test_sparse_summary_matches_dense_axis_sums():
    dense = _toy_dense_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)

    sub_bg, sub_sg = substitution_sparse.summarize_sparse_sub_tensor(sparse_tensor, mode="spe2spe")
    np.testing.assert_allclose(sub_bg, dense.sum(axis=1), atol=1e-12)
    np.testing.assert_allclose(sub_sg, dense.sum(axis=0), atol=1e-12)

    sub_bg, sub_sg = substitution_sparse.summarize_sparse_sub_tensor(sparse_tensor, mode="spe2any")
    np.testing.assert_allclose(sub_bg, dense.sum(axis=(1, 4)), atol=1e-12)
    np.testing.assert_allclose(sub_sg, dense.sum(axis=(0, 4)), atol=1e-12)

    sub_bg, sub_sg = substitution_sparse.summarize_sparse_sub_tensor(sparse_tensor, mode="any2spe")
    np.testing.assert_allclose(sub_bg, dense.sum(axis=(1, 3)), atol=1e-12)
    np.testing.assert_allclose(sub_sg, dense.sum(axis=(0, 3)), atol=1e-12)

    sub_bg, sub_sg = substitution_sparse.summarize_sparse_sub_tensor(sparse_tensor, mode="any2any")
    np.testing.assert_allclose(sub_bg, dense.sum(axis=(1, 3, 4)), atol=1e-12)
    np.testing.assert_allclose(sub_sg, dense.sum(axis=(0, 3, 4)), atol=1e-12)


def test_sparse_summary_bool_tensor_accumulates_counts_without_cast_errors():
    dense = (_toy_dense_tensor() > 0)
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)

    sub_bg, sub_sg = substitution_sparse.summarize_sparse_sub_tensor(sparse_tensor, mode="spe2spe")
    np.testing.assert_allclose(sub_bg, dense.sum(axis=1), atol=1e-12)
    np.testing.assert_allclose(sub_sg, dense.sum(axis=0), atol=1e-12)

    sub_bg, sub_sg = substitution_sparse.summarize_sparse_sub_tensor(sparse_tensor, mode="spe2any")
    np.testing.assert_allclose(sub_bg, dense.sum(axis=(1, 4)), atol=1e-12)
    np.testing.assert_allclose(sub_sg, dense.sum(axis=(0, 4)), atol=1e-12)

    sub_bg, sub_sg = substitution_sparse.summarize_sparse_sub_tensor(sparse_tensor, mode="any2spe")
    np.testing.assert_allclose(sub_bg, dense.sum(axis=(1, 3)), atol=1e-12)
    np.testing.assert_allclose(sub_sg, dense.sum(axis=(0, 3)), atol=1e-12)

    sub_bg, sub_sg = substitution_sparse.summarize_sparse_sub_tensor(sparse_tensor, mode="any2any")
    np.testing.assert_allclose(sub_bg, dense.sum(axis=(1, 3, 4)), atol=1e-12)
    np.testing.assert_allclose(sub_sg, dense.sum(axis=(0, 3, 4)), atol=1e-12)
