import numpy as np
import pytest

from csubst import omega
from csubst import substitution
from csubst import tree
from csubst import ete


def test_fused_expected_sparse_tensor_matches_materialized_path():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    num_node = max(labels.values()) + 1
    state = np.zeros((num_node, 3, 2), dtype=np.float64)
    state[labels["R"], :, 0] = [1.0, 0.75, 0.25]
    state[labels["R"], :, 1] = [0.0, 0.25, 0.75]
    state[labels["A"], :, :] = state[labels["R"], :, :]
    state[labels["B"], :, :] = state[labels["R"], :, :]
    inst = np.array([[-0.6, 0.6], [0.4, -0.4]], dtype=np.float64)
    g = {
        "tree": tr,
        "state_nsy": state,
        "instantaneous_nsy_rate_matrix": inst,
        "iqtree_rate_values": np.array([0.5, 1.0, 2.0], dtype=np.float64),
        "float_type": np.float64,
        "float_tol": 1e-12,
        "threads": 1,
        "expected_state_backend": "eigen",
        "ml_anc": False,
    }
    for node in tr.traverse():
        if not ete.is_root(node):
            ete.set_prop(node, "Ndist", 0.3)

    fused = omega._get_fused_expected_sparse_substitution_tensor(g=g, mode="nsy")
    materialized_state = omega.get_exp_state(g=g, mode="nsy")
    materialized = substitution.get_substitution_tensor(
        materialized_state,
        state,
        mode="asis",
        g=g,
        mmap_attr="EN",
    )

    assert fused is not None
    np.testing.assert_allclose(fused.to_dense(), materialized.to_dense(), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("expected_state_backend", ["auto", "eigen", "expm"])
def test_tensor_free_expected_reducer_matches_materialized_projections(expected_state_backend):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    state = np.zeros((max(labels.values()) + 1, 3, 2), dtype=np.float64)
    state[labels["R"], :, :] = [[1.0, 0.0], [0.75, 0.25], [0.25, 0.75]]
    state[labels["A"], :, :] = state[labels["R"], :, :]
    state[labels["B"], :, :] = state[labels["R"], :, :]
    g = {
        "tree": tr,
        "state_nsy": state,
        "instantaneous_nsy_rate_matrix": np.array([[-0.6, 0.6], [0.4, -0.4]], dtype=np.float64),
        "iqtree_rate_values": np.array([0.5, 1.0, 2.0], dtype=np.float64),
        "float_type": np.float64,
        "float_tol": 1e-12,
        "threads": 1,
        "expected_state_backend": expected_state_backend,
        "ml_anc": False,
    }
    for node in tr.traverse():
        if not ete.is_root(node):
            ete.set_prop(node, "Ndist", 0.3)

    reducer = omega._get_fused_expected_sparse_reducer(
        g=g,
        mode="nsy",
        selected_base_stats=["any2any", "spe2any", "any2spe", "spe2spe"],
    )
    expected_state = omega.get_exp_state(g=g, mode="nsy")
    materialized = substitution.get_substitution_tensor(
        expected_state,
        state,
        mode="asis",
        g=g,
        mmap_attr="EN",
    )

    assert reducer["total"] == pytest.approx(substitution.get_total_substitution(materialized), abs=1e-12)
    for stat, projection in reducer["projections"].items():
        expected_projection = substitution._build_sparse_cb_projection(materialized, stat)
        np.testing.assert_allclose(projection.toarray(), expected_projection.toarray(), atol=1e-12, rtol=1e-12)


def test_expected_projection_cython_payloads_match_synonymous_dense_layout():
    if omega.omega_cy is None or not hasattr(
        omega.omega_cy,
        "build_expected_projection_rows_double",
    ):
        pytest.skip("Cython expected projection-row builder is unavailable")
    parent = np.array(
        [[0.6, 0.4, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]],
        dtype=np.float64,
    )
    child = np.array(
        [[0.3, 0.7, 0.0, 0.0], [0.4, 0.3, 0.2, 0.1]],
        dtype=np.float64,
    )
    groups = [np.array([0, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64)]
    selected = ["any2any", "spe2any", "any2spe", "spe2spe"]
    state_indices = omega._build_expected_projection_state_indices(
        sub_mode="syn",
        num_group=2,
        num_state=2,
        syn_indices_list=groups,
    )
    payloads, total = omega._get_expected_branch_projection_payloads(
        parent_state=parent,
        expected_state=child,
        sub_mode="syn",
        num_group=2,
        num_state=2,
        syn_indices_list=groups,
        state_indices=state_indices,
        selected=selected,
    )
    dense_values, total_by_site = omega._get_expected_branch_projection_values(
        parent_state=parent,
        expected_state=child,
        sub_mode="syn",
        num_group=2,
        num_state=2,
        syn_indices_list=groups,
        selected=selected,
    )
    assert total == pytest.approx(total_by_site.sum(), abs=1e-12)
    for stat in selected:
        expected = dense_values[stat].T.reshape(-1)
        observed = np.zeros_like(expected)
        if stat in payloads:
            indices, data = payloads[stat]
            observed[indices] = data
        np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-12)


def test_get_exp_state_rejects_unknown_mode():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    num_node = len(list(tr.traverse()))
    state = np.zeros((num_node, 1, 2), dtype=np.float64)
    g = {
        "tree": tr,
        "state_pep": state.copy(),
        "state_cdn": state.copy(),
        "instantaneous_aa_rate_matrix": np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float64),
        "instantaneous_codon_rate_matrix": np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float64),
        "iqtree_rate_values": np.array([1.0], dtype=np.float64),
        "float_type": np.float64,
        "float_tol": 1e-12,
    }
    with pytest.raises(ValueError, match="Unsupported expected-state mode"):
        omega.get_exp_state(g=g, mode="unknown")


def test_get_static_sub_sites_if_available_uses_asrv_mode():
    sub_sg = np.zeros((2, 1), dtype=np.float64)
    g_each = {"asrv": "each", "sub_sites": {"each": np.ones((2, 2), dtype=np.float64)}}
    assert omega._get_static_sub_sites_if_available(g_each, sub_sg, "any2any", "OCNany2any") is None
    g_file_each = {"asrv": "file_each", "sub_sites": {}}
    assert omega._get_static_sub_sites_if_available(g_file_each, sub_sg, "any2any", "OCNany2any") is None

    g_sn = {
        "asrv": "sn",
        "sub_sites": {
            "S": np.array([[0.1, 0.9]], dtype=np.float64),
            "N": np.array([[0.3, 0.7]], dtype=np.float64),
        },
    }
    out_s = omega._get_static_sub_sites_if_available(g_sn, sub_sg, "any2any", "OCSany2any")
    out_n = omega._get_static_sub_sites_if_available(g_sn, sub_sg, "any2any", "OCNany2any")
    np.testing.assert_allclose(out_s, g_sn["sub_sites"]["S"], atol=1e-12)
    np.testing.assert_allclose(out_n, g_sn["sub_sites"]["N"], atol=1e-12)


def test_get_exp_state_matches_numpy_fallback(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1):1,C:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    num_node = max(labels.values()) + 1
    rng = np.random.default_rng(1)
    state = rng.random((num_node, 6, 4), dtype=np.float64)
    state /= state.sum(axis=2, keepdims=True)

    g = {
        "tree": tr,
        "state_pep": state.copy(),
        "state_cdn": state.copy(),
        "instantaneous_aa_rate_matrix": np.array(
            [
                [-1.5, 0.5, 0.5, 0.5],
                [0.2, -1.2, 0.6, 0.4],
                [0.4, 0.3, -1.1, 0.4],
                [0.3, 0.5, 0.2, -1.0],
            ],
            dtype=np.float64,
        ),
        "instantaneous_codon_rate_matrix": np.array(
            [
                [-1.5, 0.5, 0.5, 0.5],
                [0.2, -1.2, 0.6, 0.4],
                [0.4, 0.3, -1.1, 0.4],
                [0.3, 0.5, 0.2, -1.0],
            ],
            dtype=np.float64,
        ),
        "iqtree_rate_values": np.array([0.5, 1.0, 1.0, 0.25, 0.5, 1.0], dtype=np.float64),
        "float_type": np.float64,
        "float_tol": 1e-12,
    }

    for node in tr.traverse():
        if ete.is_root(node):
            continue
        ete.set_prop(node, "Ndist", 0.2)
        ete.set_prop(node, "SNdist", 0.2)

    out_default = omega.get_exp_state(g=g, mode="pep")
    monkeypatch.setattr(omega, "_can_use_cython_expected_state", lambda *_args, **_kwargs: False)
    out_numpy = omega.get_exp_state(g=g, mode="pep")
    np.testing.assert_allclose(out_default, out_numpy, atol=1e-12)


def test_get_exp_state_parallel_projection_matches_single_thread(monkeypatch):
    tr = tree.add_numerical_node_labels(
        ete.PhyloNode("((((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1):1,I:1)R;", format=1)
    )
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    num_node = max(labels.values()) + 1
    rng = np.random.default_rng(8)
    state = rng.random((num_node, 120, 4), dtype=np.float64)
    state /= state.sum(axis=2, keepdims=True)
    g_parallel = {
        "tree": tr,
        "state_pep": state.copy(),
        "instantaneous_aa_rate_matrix": np.array(
            [
                [-1.5, 0.5, 0.5, 0.5],
                [0.2, -1.2, 0.6, 0.4],
                [0.4, 0.3, -1.1, 0.4],
                [0.3, 0.5, 0.2, -1.0],
            ],
            dtype=np.float64,
        ),
        "iqtree_rate_values": np.array(([0.5, 1.0, 1.5, 2.0] * 30), dtype=np.float64),
        "float_type": np.float64,
        "float_tol": 1e-12,
        "threads": 4,
    }
    for node in tr.traverse():
        if ete.is_root(node):
            continue
        ete.set_prop(node, "Ndist", 0.2)
    g_single = dict(g_parallel)
    g_single["threads"] = 1
    monkeypatch.setattr(
        omega.parallel,
        "resolve_task_n_jobs",
        lambda num_items, threads, task: min(int(threads), max(1, int(num_items))),
    )
    out_parallel = omega.get_exp_state(g=g_parallel, mode="pep")
    out_single = omega.get_exp_state(g=g_single, mode="pep")
    np.testing.assert_allclose(out_parallel, out_single, atol=1e-12)
