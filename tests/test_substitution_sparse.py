import numpy as np
from pathlib import Path
import pytest

from csubst import substitution
from csubst import substitution_sparse
from csubst import substitution_cy
from csubst import tree
from csubst import ete

try:
    from csubst import substitution_sparse_cy
except ImportError:  # pragma: no cover - optional Cython extension
    substitution_sparse_cy = None


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


def test_dense_to_sparse_preserves_nan_values_with_tolerance():
    dense = np.zeros((2, 2, 1, 1, 1), dtype=np.float64)
    dense[0, 0, 0, 0, 0] = np.nan
    dense[1, 1, 0, 0, 0] = 1e-12

    sparse_tensor = substitution_sparse.dense_to_sparse_substitution_tensor(dense, tol=1e-9)
    restored = sparse_tensor.to_dense()

    assert np.isnan(restored[0, 0, 0, 0, 0])
    assert restored[1, 1, 0, 0, 0] == 0


def test_dense_to_sparse_cython_path_matches_python_fallback(monkeypatch):
    if (substitution_sparse_cy is None) or (not hasattr(substitution_sparse_cy, "dense_block_to_csr_arrays_double")):
        pytest.skip("Cython substitution_sparse fast path is unavailable")
    dense = _toy_dense_tensor()
    dense[0, 0, 0, 2, 2] = 1e-12

    monkeypatch.setattr(substitution_sparse, "_can_use_cython_dense_block_to_csr", lambda *args, **kwargs: False)
    expected = substitution_sparse.dense_to_sparse_substitution_tensor(dense, tol=1e-9)

    monkeypatch.setattr(substitution_sparse, "_can_use_cython_dense_block_to_csr", lambda *args, **kwargs: True)
    observed = substitution_sparse.dense_to_sparse_substitution_tensor(dense, tol=1e-9)

    np.testing.assert_allclose(observed.to_dense(), expected.to_dense(), atol=1e-12)
    assert observed.nnz == expected.nnz


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


def test_get_cb_sparse_selective_base_stats_matches_dense():
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array([[2, 0], [1, 2]], dtype=np.int64)
    g = {"threads": 1, "float_type": np.float64}
    selected = ["any2any", "any2spe"]
    out_dense = substitution.get_cb(ids, dense, g, attr="OCN", selected_base_stats=selected)
    out_sparse = substitution.get_cb(ids, sparse_tensor, g, attr="OCN", selected_base_stats=selected)
    assert out_sparse.columns.tolist() == ["branch_id_1", "branch_id_2", "OCNany2any", "OCNany2spe"]
    np.testing.assert_allclose(out_sparse.values, out_dense.values, atol=1e-12)


def test_sparse_cb_summary_arrays_match_dense_reductions():
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    selected = ["any2any", "spe2any", "any2spe"]
    total, from_site, to_site, pair_site = substitution._get_sparse_cb_summary_arrays(sparse_tensor, selected)

    expected_total = dense.sum(axis=(3, 4)).transpose(0, 2, 1)
    expected_from_site = dense.sum(axis=4).transpose(0, 2, 3, 1)
    expected_to_site = dense.sum(axis=3).transpose(0, 2, 3, 1)
    np.testing.assert_allclose(total, expected_total, atol=1e-12)
    np.testing.assert_allclose(from_site, expected_from_site, atol=1e-12)
    np.testing.assert_allclose(to_site, expected_to_site, atol=1e-12)
    assert pair_site is None


def test_sparse_cb_summary_arrays_include_spe2spe_channel():
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    selected = ["spe2spe"]
    total, from_site, to_site, pair_site = substitution._get_sparse_cb_summary_arrays(sparse_tensor, selected)

    assert total is None
    assert from_site is None
    assert to_site is None
    expected_pair_site = dense.reshape(dense.shape[0], dense.shape[1], dense.shape[2], -1).transpose(0, 2, 3, 1)
    np.testing.assert_allclose(pair_site, expected_pair_site, atol=1e-12)


def test_sparse_cb_summary_arrays_cython_accumulator_matches_python_fallback(monkeypatch):
    if (substitution_sparse_cy is None) or (not hasattr(substitution_sparse_cy, "accumulate_sparse_summary_block_csr_double")):
        pytest.skip("Cython sparse-summary accumulator is unavailable")
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    selected = ["any2any", "spe2any", "any2spe", "spe2spe"]

    monkeypatch.setattr(substitution, "_can_use_cython_sparse_summary_csr_accumulator", lambda *args, **kwargs: False)
    monkeypatch.setattr(substitution, "_can_use_cython_sparse_summary_accumulator", lambda *args, **kwargs: False)
    expected = substitution._get_sparse_cb_summary_arrays(sparse_tensor, selected)
    substitution._clear_sparse_cb_summary_arrays(sparse_tensor, selected)

    monkeypatch.setattr(substitution, "_can_use_cython_sparse_summary_csr_accumulator", lambda *args, **kwargs: True)
    monkeypatch.setattr(substitution, "_can_use_cython_sparse_summary_accumulator", lambda *args, **kwargs: True)
    observed = substitution._get_sparse_cb_summary_arrays(sparse_tensor, selected)

    for exp_arr, obs_arr in zip(expected, observed):
        if exp_arr is None:
            assert obs_arr is None
            continue
        np.testing.assert_allclose(obs_arr, exp_arr, atol=1e-12)


def test_sparse_cb_summary_arrays_csr_cython_accumulator_matches_existing_cython_path(monkeypatch):
    if (substitution_sparse_cy is None) or (not hasattr(substitution_sparse_cy, "accumulate_sparse_summary_block_csr_double")):
        pytest.skip("Cython sparse-summary CSR accumulator is unavailable")
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    selected = ["any2any", "spe2any", "any2spe", "spe2spe"]

    monkeypatch.setattr(substitution, "_can_use_cython_sparse_summary_csr_accumulator", lambda *args, **kwargs: False)
    monkeypatch.setattr(substitution, "_can_use_cython_sparse_summary_accumulator", lambda *args, **kwargs: True)
    expected = substitution._get_sparse_cb_summary_arrays(sparse_tensor, selected)
    substitution._clear_sparse_cb_summary_arrays(sparse_tensor, selected)

    monkeypatch.setattr(substitution, "_can_use_cython_sparse_summary_csr_accumulator", lambda *args, **kwargs: True)
    monkeypatch.setattr(substitution, "_can_use_cython_sparse_summary_accumulator", lambda *args, **kwargs: False)
    observed = substitution._get_sparse_cb_summary_arrays(sparse_tensor, selected)

    for exp_arr, obs_arr in zip(expected, observed):
        if exp_arr is None:
            assert obs_arr is None
            continue
        np.testing.assert_allclose(obs_arr, exp_arr, atol=1e-12)


def test_sparse_csr_count_and_scatter_cython_kernels_match_python_fallback(monkeypatch):
    if (substitution_sparse_cy is None) or (not hasattr(substitution_sparse_cy, "accumulate_branch_sub_counts_csr_double")):
        pytest.skip("Cython sparse CSR helper kernels are unavailable")
    dense = _toy_dense_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    branch_id = 1

    monkeypatch.setattr(substitution, "_is_sparse_csr_cython_compatible", lambda mat: False)
    expected_branch = substitution.get_branch_sub_counts(sparse_tensor)
    expected_site = substitution.get_site_sub_counts(sparse_tensor)
    expected_branch_site = substitution.get_branch_site_sub_counts(sparse_tensor, branch_id=branch_id)
    expected_branch_tensor = substitution._get_sparse_branch_tensor(sparse_tensor, branch_id=branch_id)

    monkeypatch.setattr(substitution, "_is_sparse_csr_cython_compatible", lambda mat: True)
    observed_branch = substitution.get_branch_sub_counts(sparse_tensor)
    observed_site = substitution.get_site_sub_counts(sparse_tensor)
    observed_branch_site = substitution.get_branch_site_sub_counts(sparse_tensor, branch_id=branch_id)
    observed_branch_tensor = substitution._get_sparse_branch_tensor(sparse_tensor, branch_id=branch_id)

    np.testing.assert_allclose(observed_branch, expected_branch, atol=1e-12)
    np.testing.assert_allclose(observed_site, expected_site, atol=1e-12)
    np.testing.assert_allclose(observed_branch_site, expected_branch_site, atol=1e-12)
    np.testing.assert_allclose(observed_branch_tensor, expected_branch_tensor, atol=1e-12)


def test_get_b_sitewise_cython_scan_matches_python_fallback(monkeypatch):
    if (substitution_sparse_cy is None) or (not hasattr(substitution_sparse_cy, "scan_sitewise_max_indices_double")):
        pytest.skip("Cython sitewise max scan kernel is unavailable")
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    label_by_name = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse() if n.name != ""}
    num_node = max(label_by_name.values()) + 1
    dense = np.zeros((num_node, 3, 1, 3, 3), dtype=np.float64)
    dense[label_by_name["A"], 0, 0, 0, 1] = 0.6
    dense[label_by_name["A"], 0, 0, 1, 2] = 0.6
    dense[label_by_name["A"], 1, 0, 2, 1] = 0.4
    dense[label_by_name["A"], 2, 0, 1, 0] = np.nan
    dense[label_by_name["B"], 1, 0, 2, 0] = 0.8
    dense[label_by_name["N1"], 2, 0, 1, 2] = 0.7
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    g = {
        "tree": tr,
        "num_node": num_node,
        "amino_acid_orders": np.array(["A", "B", "C"], dtype=object),
    }

    monkeypatch.setattr(substitution, "_can_use_cython_sitewise_max_scan", lambda *args, **kwargs: False)
    expected = substitution.get_b(g=g, sub_tensor=sparse_tensor, attr="N", sitewise=True, min_sitewise_pp=0.5)

    monkeypatch.setattr(substitution, "_can_use_cython_sitewise_max_scan", lambda *args, **kwargs: True)
    observed = substitution.get_b(g=g, sub_tensor=sparse_tensor, attr="N", sitewise=True, min_sitewise_pp=0.5)

    assert observed.columns.tolist() == expected.columns.tolist()
    assert observed.loc[:, "branch_name"].tolist() == expected.loc[:, "branch_name"].tolist()
    assert observed.loc[:, "N_sitewise"].tolist() == expected.loc[:, "N_sitewise"].tolist()
    np.testing.assert_allclose(
        observed.loc[:, ["branch_id", "N_sub"]].to_numpy(dtype=float),
        expected.loc[:, ["branch_id", "N_sub"]].to_numpy(dtype=float),
        atol=1e-12,
    )


def test_sub_tensor2cb_sparse_cython_fastpath_matches_python_fallback(monkeypatch):
    if not hasattr(substitution_cy, "calc_combinatorial_sub_sparse_summary_double_arity2"):
        pytest.skip("Cython sparse-summary reducer fast path is unavailable")
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array([[2, 0], [1, 2], [0, 1]], dtype=np.int64)
    selected = ["any2any", "spe2any", "any2spe"]

    monkeypatch.setattr(substitution, "_can_use_cython_sparse_cb_summary", lambda *args, **kwargs: False)
    expected = substitution.sub_tensor2cb_sparse(
        ids,
        sparse_tensor,
        mmap=False,
        df_mmap=None,
        mmap_start=0,
        float_type=np.float64,
        selected_base_stats=selected,
    )

    monkeypatch.setattr(substitution, "_can_use_cython_sparse_cb_summary", lambda *args, **kwargs: True)
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


def test_sub_tensor2cb_sparse_cython_fastpath_supports_spe2spe(monkeypatch):
    if not hasattr(substitution_cy, "calc_combinatorial_sub_sparse_summary_double_arity2"):
        pytest.skip("Cython sparse-summary reducer fast path is unavailable")
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array([[2, 0], [1, 2], [0, 1]], dtype=np.int64)
    selected = ["any2any", "spe2any", "any2spe", "spe2spe"]

    monkeypatch.setattr(substitution, "_can_use_cython_sparse_cb_summary", lambda *args, **kwargs: False)
    expected = substitution.sub_tensor2cb_sparse(
        ids,
        sparse_tensor,
        mmap=False,
        df_mmap=None,
        mmap_start=0,
        float_type=np.float64,
        selected_base_stats=selected,
    )

    monkeypatch.setattr(substitution, "_can_use_cython_sparse_cb_summary", lambda *args, **kwargs: True)
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


def test_sub_tensor2cb_sparse_cython_failure_warns_and_falls_back(monkeypatch):
    if not hasattr(substitution_cy, "calc_combinatorial_sub_sparse_summary_double_arity2"):
        pytest.skip("Cython sparse-summary reducer fast path is unavailable")
    dense = _toy_reducer_tensor()
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array([[2, 0], [1, 2]], dtype=np.int64)
    selected = ["any2any", "any2spe"]

    monkeypatch.setattr(substitution, "_can_use_cython_sparse_cb_summary", lambda *args, **kwargs: False)
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
    monkeypatch.setattr(substitution, "_can_use_cython_sparse_cb_summary", lambda *args, **kwargs: True)

    def _raise_cython(*args, **kwargs):
        raise RuntimeError("forced-sparse-fastpath-failure")

    monkeypatch.setattr(substitution_cy, "calc_combinatorial_sub_sparse_summary_double_arity2", _raise_cython)
    with pytest.warns(RuntimeWarning, match='Cython fast path "sub_tensor2cb_sparse" failed'):
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


def test_get_substitution_tensor_cython_asis_fill_matches_python_fallback(monkeypatch):
    if not hasattr(substitution_cy, "fill_sub_tensor_asis_branch_double"):
        pytest.skip("Cython asis fill kernel is unavailable")
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    state = np.zeros((3, 3, 3), dtype=np.float64)
    state[labels["R"], :, :] = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.1, 0.2, 0.7]]
    state[labels["A"], :, :] = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.3, 0.5]]
    state[labels["B"], :, :] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, 0.1, 0.3]]
    base_g = {"tree": tr, "ml_anc": "yes", "float_tol": 1e-12, "sub_tensor_backend": "dense"}
    fallback_mmap = Path("tmp.csubst.sub_tensor.toy_asis_fallback.mmap")
    cython_mmap = Path("tmp.csubst.sub_tensor.toy_asis_cython.mmap")
    try:
        monkeypatch.setattr(substitution, "_can_use_cython_asis_sub_tensor_fill", lambda *args, **kwargs: False)
        expected = substitution.get_substitution_tensor(
            state_tensor=state,
            mode="asis",
            g=dict(base_g),
            mmap_attr="toy_asis_fallback",
        )
        monkeypatch.setattr(substitution, "_can_use_cython_asis_sub_tensor_fill", lambda *args, **kwargs: True)
        observed = substitution.get_substitution_tensor(
            state_tensor=state,
            mode="asis",
            g=dict(base_g),
            mmap_attr="toy_asis_cython",
        )
        np.testing.assert_allclose(observed, expected, atol=1e-12)
    finally:
        if fallback_mmap.exists():
            fallback_mmap.unlink()
        if cython_mmap.exists():
            cython_mmap.unlink()


def test_get_substitution_tensor_cython_syn_fill_matches_python_fallback(monkeypatch):
    if not hasattr(substitution_cy, "fill_sub_tensor_syn_branch_double"):
        pytest.skip("Cython syn fill kernel is unavailable")
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    state = np.zeros((3, 3, 4), dtype=np.float64)
    state[labels["R"], :, :] = [
        [0.6, 0.4, 0.0, 0.0],
        [0.1, 0.9, 0.0, 0.0],
        [0.0, 0.0, 0.3, 0.7],
    ]
    state[labels["A"], :, :] = [
        [0.2, 0.8, 0.0, 0.0],
        [0.7, 0.3, 0.0, 0.0],
        [0.0, 0.0, 0.4, 0.6],
    ]
    state[labels["B"], :, :] = [
        [0.9, 0.1, 0.0, 0.0],
        [0.2, 0.8, 0.0, 0.0],
        [0.0, 0.0, 0.6, 0.4],
    ]
    base_g = {
        "tree": tr,
        "ml_anc": "yes",
        "float_tol": 1e-12,
        "sub_tensor_backend": "dense",
        "amino_acid_orders": np.array(["AA0", "AA1"], dtype=object),
        "synonymous_indices": {"AA0": [0, 1], "AA1": [2, 3]},
        "max_synonymous_size": 2,
    }
    fallback_mmap = Path("tmp.csubst.sub_tensor.toy_syn_fallback.mmap")
    cython_mmap = Path("tmp.csubst.sub_tensor.toy_syn_cython.mmap")
    try:
        monkeypatch.setattr(substitution, "_can_use_cython_syn_sub_tensor_fill", lambda *args, **kwargs: False)
        expected = substitution.get_substitution_tensor(
            state_tensor=state,
            mode="syn",
            g=dict(base_g),
            mmap_attr="toy_syn_fallback",
        )
        monkeypatch.setattr(substitution, "_can_use_cython_syn_sub_tensor_fill", lambda *args, **kwargs: True)
        observed = substitution.get_substitution_tensor(
            state_tensor=state,
            mode="syn",
            g=dict(base_g),
            mmap_attr="toy_syn_cython",
        )
        np.testing.assert_allclose(observed, expected, atol=1e-12)
    finally:
        if fallback_mmap.exists():
            fallback_mmap.unlink()
        if cython_mmap.exists():
            cython_mmap.unlink()


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
