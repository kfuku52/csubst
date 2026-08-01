from sparse_fixtures import large_sparse_reducer_tensor as _large_sparse_reducer_tensor
from sparse_fixtures import toy_dense_tensor as _toy_dense_tensor
from sparse_fixtures import toy_reducer_tensor as _toy_reducer_tensor

import itertools
import numpy as np

from csubst import substitution
from csubst import substitution_sparse








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


def test_get_cb_auto_parallel_matches_single_thread_for_dense_and_sparse():
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array([[2, 0], [1, 2], [0, 1]], dtype=np.int64)
    g_single = {"threads": 1, "float_type": np.float64}
    g_auto = {"threads": 2, "float_type": np.float64}
    out_dense_single = substitution.get_cb(ids, dense, g_single, attr="OCN")
    out_dense_auto = substitution.get_cb(ids, dense, g_auto, attr="OCN")
    out_sparse_single = substitution.get_cb(ids, sparse_tensor, g_single, attr="OCN")
    out_sparse_auto = substitution.get_cb(ids, sparse_tensor, g_auto, attr="OCN")
    np.testing.assert_allclose(out_dense_auto.values, out_dense_single.values, atol=1e-12)
    np.testing.assert_allclose(out_sparse_auto.values, out_sparse_single.values, atol=1e-12)


def test_sparse_projection_gram_scheduler_uses_one_worker(monkeypatch):
    sparse_tensor = _large_sparse_reducer_tensor(num_branch=40, num_site=12)
    ids = np.array(list(itertools.combinations(range(40), 2)), dtype=np.int64)
    g = {"threads": 8}
    monkeypatch.setattr(substitution.parallel, "resolve_task_n_jobs", lambda **_kwargs: 8)
    observed = substitution._resolve_cb_n_jobs(
        id_combinations=ids,
        sub_tensor=sparse_tensor,
        g=g,
        writer=substitution.sub_tensor2cb_sparse,
        selected=["any2any", "any2spe"],
    )
    assert observed == 1


def test_dense_arity3_projection_scheduler_uses_one_worker(monkeypatch):
    dense = np.ones((6, 5, 1, 2, 2), dtype=np.float64)
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.tile(np.array([[0, 1, 2]], dtype=np.int64), (20, 1))
    g = {"threads": 8}
    monkeypatch.setattr(substitution.parallel, "resolve_task_n_jobs", lambda **_kwargs: 8)
    monkeypatch.setattr(substitution, "_SPARSE_CB_PROJECTION_ARITY3_DENSE_MIN_COMBINATIONS", 1)
    observed = substitution._resolve_cb_n_jobs(
        id_combinations=ids,
        sub_tensor=sparse_tensor,
        g=g,
        writer=substitution.sub_tensor2cb_sparse,
        selected=["any2any"],
    )

    assert observed == 1


def test_resolve_dense_cython_n_jobs_prefers_single_for_small_workload():
    ids = np.zeros((1200, 2), dtype=np.int64)
    sub = np.zeros((10, 100, 1, 4, 4), dtype=np.float64)
    out = substitution._resolve_dense_cython_n_jobs(
        n_jobs=8,
        id_combinations=ids,
        sub_tensor=sub,
        task="cb",
    )
    assert out == 1


def test_resolve_dense_cython_n_jobs_allows_parallel_for_large_workload():
    ids = np.zeros((200000, 2), dtype=np.int64)
    sub = np.zeros((10, 500, 1, 4, 4), dtype=np.float64)
    out = substitution._resolve_dense_cython_n_jobs(
        n_jobs=8,
        id_combinations=ids,
        sub_tensor=sub,
        task="cb",
    )
    assert out >= 2


def test_sparse_group_tensor_packed_path_ignores_legacy_block_cache():
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array([2, 0], dtype=np.int64)
    uncached = substitution._get_sparse_combination_group_tensor(
        sub_tensor=sparse_tensor,
        branch_ids=ids,
        sg=0,
        data_type=np.float64,
    )
    group_block_index = substitution._get_sparse_group_block_index(sparse_tensor)
    row_cache = dict()
    cached = substitution._get_sparse_combination_group_tensor(
        sub_tensor=sparse_tensor,
        branch_ids=ids,
        sg=0,
        data_type=np.float64,
        group_block_index=group_block_index,
        row_cache=row_cache,
    )
    np.testing.assert_allclose(cached, uncached, atol=1e-12)
    assert row_cache == {}
    assert substitution._get_sparse_group_block_index(sparse_tensor) is group_block_index


def test_sparse_site_vectors_cache_matches_uncached():
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array([2, 0], dtype=np.int64)
    uncached = substitution._get_sparse_site_vectors(
        sub_tensor=sparse_tensor,
        branch_ids=ids,
        data_type=np.float64,
    )
    cached = substitution._get_sparse_site_vectors(
        sub_tensor=sparse_tensor,
        branch_ids=ids,
        data_type=np.float64,
        group_block_index=substitution._get_sparse_group_block_index(sparse_tensor),
        row_cache=dict(),
    )
    for observed, expected in zip(cached, uncached):
        np.testing.assert_allclose(observed, expected, atol=1e-12)


def test_get_cbs_sparse_matches_dense():
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array([[2, 0], [1, 2]], dtype=np.int64)
    g = {"threads": 1}
    out_dense = substitution.get_cbs(ids, dense, attr="N", g=g)
    out_sparse = substitution.get_cbs(ids, sparse_tensor, attr="N", g=g)
    np.testing.assert_allclose(out_sparse.values, out_dense.values, atol=1e-12)


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


def test_get_cbs_auto_parallel_matches_single_thread_for_dense_and_sparse():
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array([[2, 0], [1, 2], [0, 1]], dtype=np.int64)
    g_single = {"threads": 1}
    g_auto = {"threads": 2}
    out_dense_single = substitution.get_cbs(ids, dense, attr="N", g=g_single)
    out_dense_auto = substitution.get_cbs(ids, dense, attr="N", g=g_auto)
    out_sparse_single = substitution.get_cbs(ids, sparse_tensor, attr="N", g=g_single)
    out_sparse_auto = substitution.get_cbs(ids, sparse_tensor, attr="N", g=g_auto)
    np.testing.assert_allclose(out_dense_auto.values, out_dense_single.values, atol=1e-12)
    np.testing.assert_allclose(out_sparse_auto.values, out_sparse_single.values, atol=1e-12)


def test_estimate_sub_tensor_density_matches_dense_and_sparse():
    dense = _toy_dense_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    expected = np.count_nonzero(dense) / dense.size
    assert substitution.estimate_sub_tensor_density(dense) == expected
    assert substitution.estimate_sub_tensor_density(sparse_tensor) == expected


def test_get_reducer_sub_tensor_converts_and_caches_sparse():
    dense = _toy_dense_tensor()
    g = {"float_tol": 0.0}
    sparse1 = substitution.get_reducer_sub_tensor(dense, g=g, label="test")
    sparse2 = substitution.get_reducer_sub_tensor(dense, g=g, label="test")
    assert isinstance(sparse1, substitution_sparse.SparseSubstitutionTensor)
    assert sparse1 is sparse2
