import numpy as np
from pathlib import Path

from csubst import substitution
from csubst import substitution_sparse
from csubst import tree
from csubst import ete


def _toy_dense_tensor():
    # shape = [branch, site, group, from, to]
    sub = np.zeros((3, 4, 2, 3, 3), dtype=np.float64)
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
    assert sparse_tensor.nnz == int(np.count_nonzero(dense))
    np.testing.assert_allclose(restored, dense, atol=1e-12)


def test_dense_to_sparse_applies_tolerance():
    dense = _toy_dense_tensor()
    nnz_before = int(np.count_nonzero(dense))
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
        np.testing.assert_allclose(observed_any2any, expected_any2any, atol=1e-12)

        for a in range(dense.shape[3]):
            observed_spe2any = sparse_tensor.project_spe2any(sg, a).toarray()
            expected_spe2any = dense[:, :, sg, a, :].sum(axis=2)
            np.testing.assert_allclose(observed_spe2any, expected_spe2any, atol=1e-12)

        for d in range(dense.shape[4]):
            observed_any2spe = sparse_tensor.project_any2spe(sg, d).toarray()
            expected_any2spe = dense[:, :, sg, :, d].sum(axis=2)
            np.testing.assert_allclose(observed_any2spe, expected_any2spe, atol=1e-12)


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


def _toy_reducer_tensor():
    # shape = [branch, site, group, from, to]
    sub = np.zeros((3, 2, 1, 2, 2), dtype=np.float64)
    sub[0, 0, 0, :, :] = [[0.0, 0.2], [0.1, 0.0]]
    sub[1, 0, 0, :, :] = [[0.0, 0.5], [0.2, 0.0]]
    sub[2, 0, 0, :, :] = [[0.0, 0.4], [0.3, 0.0]]
    sub[0, 1, 0, :, :] = [[0.0, 0.1], [0.0, 0.0]]
    sub[1, 1, 0, :, :] = [[0.0, 0.1], [0.3, 0.0]]
    sub[2, 1, 0, :, :] = [[0.0, 0.2], [0.1, 0.0]]
    return sub


def test_get_cs_sparse_matches_dense():
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array([[2, 0], [1, 2]], dtype=np.int64)
    out_dense = substitution.get_cs(ids, dense, attr="N")
    out_sparse = substitution.get_cs(ids, sparse_tensor, attr="N")
    np.testing.assert_allclose(out_sparse.values, out_dense.values, atol=1e-12)


def test_get_cb_sparse_matches_dense():
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array([[2, 0], [1, 2]], dtype=np.int64)
    g = {"threads": 1, "float_type": np.float64}
    out_dense = substitution.get_cb(ids, dense, g, attr="OCN")
    out_sparse = substitution.get_cb(ids, sparse_tensor, g, attr="OCN")
    np.testing.assert_allclose(out_sparse.values, out_dense.values, atol=1e-12)


def test_get_cb_threading_matches_single_thread_for_dense_and_sparse():
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array([[2, 0], [1, 2], [0, 1]], dtype=np.int64)
    g_single = {"threads": 1, "float_type": np.float64}
    g_thread = {
        "threads": 2,
        "float_type": np.float64,
        "parallel_backend": "threading",
        "parallel_chunk_factor_reducer": 2,
    }
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
    g_auto = {"threads": 2, "float_type": np.float64, "parallel_backend": "auto"}
    out_dense_single = substitution.get_cb(ids, dense, g_single, attr="OCN")
    out_dense_auto = substitution.get_cb(ids, dense, g_auto, attr="OCN")
    out_sparse_single = substitution.get_cb(ids, sparse_tensor, g_single, attr="OCN")
    out_sparse_auto = substitution.get_cb(ids, sparse_tensor, g_auto, attr="OCN")
    np.testing.assert_allclose(out_dense_auto.values, out_dense_single.values, atol=1e-12)
    np.testing.assert_allclose(out_sparse_auto.values, out_sparse_single.values, atol=1e-12)


def test_resolve_dense_cython_n_jobs_prefers_single_for_small_workload():
    ids = np.zeros((1200, 2), dtype=np.int64)
    sub = np.zeros((10, 100, 1, 4, 4), dtype=np.float64)
    g = {
        "parallel_dense_cython_min_combos_per_job": 5000,
        "parallel_dense_cython_min_ops_per_job": 500000000,
    }
    out = substitution._resolve_dense_cython_n_jobs(
        n_jobs=8,
        id_combinations=ids,
        sub_tensor=sub,
        g=g,
        task="cb",
    )
    assert out == 1


def test_resolve_dense_cython_n_jobs_allows_parallel_for_large_workload():
    ids = np.zeros((200000, 2), dtype=np.int64)
    sub = np.zeros((10, 500, 1, 4, 4), dtype=np.float64)
    g = {
        "parallel_dense_cython_min_combos_per_job": 5000,
        "parallel_dense_cython_min_ops_per_job": 500000000,
    }
    out = substitution._resolve_dense_cython_n_jobs(
        n_jobs=8,
        id_combinations=ids,
        sub_tensor=sub,
        g=g,
        task="cb",
    )
    assert out >= 2


def test_sparse_group_tensor_cache_matches_uncached():
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
    assert len(row_cache) > 0
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


def test_get_cbs_threading_matches_single_thread_for_dense_and_sparse():
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array([[2, 0], [1, 2], [0, 1]], dtype=np.int64)
    g_single = {"threads": 1}
    g_thread = {"threads": 2, "parallel_backend": "threading", "parallel_chunk_factor_reducer": 2}
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
    g_auto = {"threads": 2, "parallel_backend": "auto"}
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


def test_resolve_reducer_backend_auto_threshold_switches_dense_sparse():
    dense = _toy_dense_tensor()
    high_cutoff = {"sub_tensor_backend": "auto", "sub_tensor_sparse_density_cutoff": 1.0}
    low_cutoff = {"sub_tensor_backend": "auto", "sub_tensor_sparse_density_cutoff": 0.00001}
    assert substitution.resolve_reducer_backend(high_cutoff, dense, label="x") == "sparse"
    assert substitution.resolve_reducer_backend(low_cutoff, dense, label="y") == "dense"


def test_get_reducer_sub_tensor_converts_and_caches_sparse():
    dense = _toy_dense_tensor()
    g = {"sub_tensor_backend": "sparse", "float_tol": 0.0}
    sparse1 = substitution.get_reducer_sub_tensor(dense, g=g, label="test")
    sparse2 = substitution.get_reducer_sub_tensor(dense, g=g, label="test")
    assert isinstance(sparse1, substitution_sparse.SparseSubstitutionTensor)
    assert sparse1 is sparse2


def test_get_substitution_tensor_sparse_asis_matches_dense():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    state = np.zeros((3, 2, 2), dtype=float)
    state[labels["R"], :, :] = [[1.0, 0.0], [0.5, 0.5]]
    state[labels["A"], :, :] = [[0.0, 1.0], [1.0, 0.0]]
    state[labels["B"], :, :] = [[1.0, 0.0], [0.0, 1.0]]
    base_g = {"tree": tr, "ml_anc": "yes", "float_tol": 1e-12}
    dense_mmap = Path("tmp.csubst.sub_tensor.toy_dense_asis.mmap")
    sparse_mmap = Path("tmp.csubst.sub_tensor.toy_sparse_asis.mmap")
    try:
        dense = substitution.get_substitution_tensor(
            state_tensor=state,
            mode="asis",
            g=dict(base_g, sub_tensor_backend="dense"),
            mmap_attr="toy_dense_asis",
        )
        sparse = substitution.get_substitution_tensor(
            state_tensor=state,
            mode="asis",
            g=dict(base_g, sub_tensor_backend="sparse"),
            mmap_attr="toy_sparse_asis",
        )
        assert isinstance(sparse, substitution_sparse.SparseSubstitutionTensor)
        np.testing.assert_allclose(sparse.to_dense(), dense, atol=1e-12)
    finally:
        if dense_mmap.exists():
            dense_mmap.unlink()
        if sparse_mmap.exists():
            sparse_mmap.unlink()


def test_apply_min_sub_pp_sparse_matches_dense():
    dense = _toy_reducer_tensor()
    sparse = substitution.dense_to_sparse_sub_tensor(dense.copy(), tol=0)
    g = {"min_sub_pp": 0.25, "ml_anc": False}
    dense_out = substitution.apply_min_sub_pp(g, dense.copy())
    sparse_out = substitution.apply_min_sub_pp(g, sparse)
    np.testing.assert_allclose(sparse_out.to_dense(), dense_out, atol=1e-12)


def test_get_group_state_totals_matches_dense_and_sparse():
    dense = _toy_dense_tensor()
    sparse = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    gad_d, ga_d, gd_d = substitution.get_group_state_totals(dense)
    gad_s, ga_s, gd_s = substitution.get_group_state_totals(sparse)
    np.testing.assert_allclose(gad_d, dense.sum(axis=(0, 1)), atol=1e-12)
    np.testing.assert_allclose(ga_d, dense.sum(axis=(0, 1, 4)), atol=1e-12)
    np.testing.assert_allclose(gd_d, dense.sum(axis=(0, 1, 3)), atol=1e-12)
    np.testing.assert_allclose(gad_s, gad_d, atol=1e-12)
    np.testing.assert_allclose(ga_s, ga_d, atol=1e-12)
    np.testing.assert_allclose(gd_s, gd_d, atol=1e-12)
