from sparse_fixtures import toy_reducer_tensor as _toy_reducer_tensor

import numpy as np
import pytest

from csubst import substitution
from csubst import tree
from csubst import ete


try:
    from csubst import substitution_sparse_cy
except ImportError:  # pragma: no cover - optional Cython extension
    substitution_sparse_cy = None








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


def test_get_cb_sparse_arity3_projection_product_matches_dense_without_5d_reconstruction(monkeypatch):
    rng = np.random.default_rng(31)
    dense = rng.random((4, 7, 2, 3, 3), dtype=np.float64)
    dense[dense < 0.72] = 0
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    ids = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    g = {"threads": 1, "float_type": np.float64}
    expected = substitution.get_cb(ids, dense, g, attr="OCN")

    monkeypatch.setattr(
        substitution,
        "_get_sparse_combination_group_tensor",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("5D reconstruction should not run")),
    )
    observed = substitution.get_cb(ids, sparse_tensor, g, attr="OCN")

    np.testing.assert_allclose(observed.values, expected.values, atol=1e-12)


def test_sparse_projection_product_cython_matches_python_for_totals_and_sites(monkeypatch):
    rng = np.random.default_rng(32)
    dense = rng.random((5, 9, 2, 3, 3), dtype=np.float64)
    dense[dense < 0.68] = 0
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    projection = substitution._get_sparse_cb_projection(sparse_tensor, "any2spe")
    ids = np.array([[0, 1, 2], [1, 3, 4], [0, 2, 4]], dtype=np.int64)

    observed_total = substitution._calc_sparse_projection_products(projection, ids)
    observed_site = substitution._calc_sparse_projection_products(projection, ids, num_site=dense.shape[1])
    monkeypatch.setattr(substitution, "_can_use_cython_sparse_projection_product", lambda *args, **kwargs: False)
    expected_total = substitution._calc_sparse_projection_products(projection, ids)
    expected_site = substitution._calc_sparse_projection_products(projection, ids, num_site=dense.shape[1])

    np.testing.assert_allclose(observed_total, expected_total, atol=1e-12)
    np.testing.assert_allclose(observed_site, expected_site, atol=1e-12)
    np.testing.assert_allclose(observed_total, observed_site.sum(axis=1), atol=1e-12)


def test_dense_arity3_projection_selector_matches_sparse_intersection(monkeypatch):
    rng = np.random.default_rng(34)
    dense = rng.random((6, 8, 2, 3, 3), dtype=np.float64)
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    projection = substitution._get_sparse_cb_projection(sparse_tensor, "any2spe")
    ids = np.vstack([rng.choice(6, size=3, replace=False) for _ in range(20)]).astype(np.int64)
    expected = substitution._calc_sparse_projection_products_python(projection, ids)
    invoked = {"dense": False}
    original = substitution._calc_dense_arity3_projection_products

    def _wrapped_dense(*args, **kwargs):
        invoked["dense"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(substitution, "_SPARSE_CB_PROJECTION_ARITY3_DENSE_MIN_COMBINATIONS", 1)
    monkeypatch.setattr(substitution, "_SPARSE_CB_PROJECTION_ARITY3_DENSE_CUTOFF", 0.0)
    monkeypatch.setattr(substitution, "_calc_dense_arity3_projection_products", _wrapped_dense)
    observed = substitution._calc_sparse_projection_products(projection, ids)

    assert invoked["dense"] is True
    np.testing.assert_allclose(observed, expected, atol=1e-12)


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


def test_get_b_sitewise_sparse_path_matches_dense_reference():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse() if n.name != ""}
    num_node = max(int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()) + 1
    dense = np.zeros((num_node, 4, 1, 3, 3), dtype=np.float64)
    dense[labels["A"], 0, 0, 0, 1] = 0.7
    dense[labels["A"], 0, 0, 1, 2] = 0.7
    dense[labels["A"], 2, 0, 2, 1] = 0.9
    dense[labels["B"], 1, 0, 1, 0] = 0.8
    dense[labels["B"], 3, 0, 2, 0] = 0.6
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0.0)
    g = {
        "tree": tr,
        "num_node": num_node,
        "amino_acid_orders": np.array(["A", "B", "C"], dtype=object),
    }

    dense_ref = substitution.get_b(g=g, sub_tensor=dense, attr="N", sitewise=True, min_sitewise_pp=0.5)
    sparse_obs = substitution.get_b(g=g, sub_tensor=sparse_tensor, attr="N", sitewise=True, min_sitewise_pp=0.5)

    assert sparse_obs.loc[:, "branch_name"].tolist() == dense_ref.loc[:, "branch_name"].tolist()
    assert sparse_obs.loc[:, "N_sitewise"].tolist() == dense_ref.loc[:, "N_sitewise"].tolist()
    np.testing.assert_allclose(
        sparse_obs.loc[:, ["branch_id", "N_sub"]].to_numpy(dtype=float),
        dense_ref.loc[:, ["branch_id", "N_sub"]].to_numpy(dtype=float),
        atol=1e-12,
    )


def test_get_b_sitewise_sparse_path_uses_nonsyn_state_orders():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse() if n.name != ""}
    num_node = max(int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()) + 1
    dense = np.zeros((num_node, 1, 1, 2, 2), dtype=np.float64)
    dense[labels["A"], 0, 0, 0, 1] = 0.9
    sparse_tensor = substitution.dense_to_sparse_sub_tensor(dense, tol=0.0)
    g = {
        "tree": tr,
        "num_node": num_node,
        "amino_acid_orders": np.array(["A", "V", "T", "I"], dtype=object),
        "nonsyn_state_orders": np.array(["AGPST", "C"], dtype=object),
    }

    out = substitution.get_b(g=g, sub_tensor=sparse_tensor, attr="N", sitewise=True, min_sitewise_pp=0.5)

    assert out.loc[out["branch_id"] == labels["A"], "N_sitewise"].iloc[0] == "AGPST1C"
