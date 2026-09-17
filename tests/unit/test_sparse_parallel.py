from sparse_fixtures import toy_reducer_tensor as _toy_reducer_tensor

import numpy as np

from csubst import substitution

def test_get_cb_threads_setting_matches_single_thread_for_dense_and_sparse():
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array([[2, 0], [1, 2], [0, 1]], dtype=np.int64)
    g_single = {"threads": 1, "float_type": np.float64}
    g_thread = {"threads": 2, "float_type": np.float64}
    out_dense_single = substitution.get_cb(ids, dense, g_single, attr="OCN")
    out_dense_thread = substitution.get_cb(ids, dense, g_thread, attr="OCN")
    out_sparse_single = substitution.get_cb(ids, sparse_tensor, g_single, attr="OCN")
    out_sparse_thread = substitution.get_cb(ids, sparse_tensor, g_thread, attr="OCN")
    np.testing.assert_allclose(out_dense_thread.values, out_dense_single.values, atol=1e-12)
    np.testing.assert_allclose(out_sparse_thread.values, out_sparse_single.values, atol=1e-12)


def test_get_cbs_sparse_arity3_projection_product_matches_dense_without_5d_reconstruction(monkeypatch):
    rng = np.random.default_rng(33)
    dense = rng.random((4, 6, 2, 3, 3), dtype=np.float64)
    dense[dense < 0.7] = 0
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    g = {"threads": 1}
    expected = substitution.get_cbs(ids, dense, attr="N", g=g)

    monkeypatch.setattr(
        substitution,
        "_get_sparse_combination_group_tensor",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("5D reconstruction should not run")),
    )
    observed = substitution.get_cbs(ids, sparse_tensor, attr="N", g=g)

    np.testing.assert_allclose(observed.values, expected.values, atol=1e-12)


def test_get_cbs_threads_setting_matches_single_thread_for_dense_and_sparse():
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array([[2, 0], [1, 2], [0, 1]], dtype=np.int64)
    g_single = {"threads": 1}
    g_thread = {"threads": 2}
    out_dense_single = substitution.get_cbs(ids, dense, attr="N", g=g_single)
    out_dense_thread = substitution.get_cbs(ids, dense, attr="N", g=g_thread)
    out_sparse_single = substitution.get_cbs(ids, sparse_tensor, attr="N", g=g_single)
    out_sparse_thread = substitution.get_cbs(ids, sparse_tensor, attr="N", g=g_thread)
    np.testing.assert_allclose(out_dense_thread.values, out_dense_single.values, atol=1e-12)
    np.testing.assert_allclose(out_sparse_thread.values, out_sparse_single.values, atol=1e-12)
