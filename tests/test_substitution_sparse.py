import numpy

from csubst import substitution
from csubst import substitution_sparse


def _toy_dense_tensor():
    # shape = [branch, site, group, from, to]
    sub = numpy.zeros((3, 4, 2, 3, 3), dtype=numpy.float64)
    sub[0, 0, 0, 0, 1] = 0.2
    sub[1, 0, 0, 0, 1] = 0.3
    sub[2, 1, 0, 2, 1] = 1.1
    sub[0, 3, 1, 1, 2] = 0.8
    sub[1, 2, 1, 1, 2] = 0.6
    sub[2, 3, 1, 0, 0] = 0.5
    return sub


def test_dense_sparse_roundtrip_preserves_values_and_shape():
    dense = _toy_dense_tensor()
    sparse_tensor = substitution_sparse.dense_to_sparse_substitution_tensor(dense)
    restored = substitution_sparse.sparse_to_dense_substitution_tensor(sparse_tensor)

    assert sparse_tensor.shape == dense.shape
    assert sparse_tensor.nnz == int(numpy.count_nonzero(dense))
    numpy.testing.assert_allclose(restored, dense, atol=1e-12)


def test_dense_to_sparse_applies_tolerance():
    dense = _toy_dense_tensor()
    nnz_before = int(numpy.count_nonzero(dense))
    dense[0, 0, 0, 2, 2] = 1e-12
    sparse_tensor = substitution_sparse.dense_to_sparse_substitution_tensor(dense, tol=1e-9)
    restored = sparse_tensor.to_dense()

    assert sparse_tensor.nnz == nnz_before
    assert restored[0, 0, 0, 2, 2] == 0


def test_sparse_projections_match_dense_reductions():
    dense = _toy_dense_tensor()
    sparse_tensor = substitution_sparse.SparseSubstitutionTensor.from_dense(dense)

    for sg in range(dense.shape[2]):
        observed_any2any = sparse_tensor.project_any2any(sg).toarray()
        expected_any2any = dense[:, :, sg, :, :].sum(axis=(2, 3))
        numpy.testing.assert_allclose(observed_any2any, expected_any2any, atol=1e-12)

        for a in range(dense.shape[3]):
            observed_spe2any = sparse_tensor.project_spe2any(sg, a).toarray()
            expected_spe2any = dense[:, :, sg, a, :].sum(axis=2)
            numpy.testing.assert_allclose(observed_spe2any, expected_spe2any, atol=1e-12)

        for d in range(dense.shape[4]):
            observed_any2spe = sparse_tensor.project_any2spe(sg, d).toarray()
            expected_any2spe = dense[:, :, sg, :, d].sum(axis=2)
            numpy.testing.assert_allclose(observed_any2spe, expected_any2spe, atol=1e-12)


def test_substitution_helpers_convert_dense_and_sparse():
    dense = _toy_dense_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    restored = substitution.sparse_to_dense_sub_tensor(sparse_tensor)
    numpy.testing.assert_allclose(restored, dense, atol=1e-12)
