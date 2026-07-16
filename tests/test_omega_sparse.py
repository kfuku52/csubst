import numpy as np
import pandas as pd
import pytest

from csubst import omega
from csubst import parallel
from csubst import substitution
from csubst import tree
from csubst import ete


def _toy_sub_tensor():
    # shape = [branch, site, group, from, to]
    sub = np.zeros((3, 3, 2, 2, 2), dtype=np.float64)
    sub[0, 0, 0, 0, 1] = 0.3
    sub[1, 0, 0, 0, 1] = 0.4
    sub[2, 0, 0, 0, 1] = 0.5
    sub[0, 1, 0, 1, 0] = 0.2
    sub[1, 1, 0, 1, 0] = 0.3
    sub[2, 1, 0, 1, 0] = 0.1
    sub[0, 2, 1, 0, 0] = 0.6
    sub[1, 2, 1, 0, 0] = 0.4
    sub[2, 2, 1, 0, 0] = 0.2
    return sub


def _toy_cb():
    return pd.DataFrame(
        {
            "branch_id_1": [0, 1],
            "branch_id_2": [1, 2],
        }
    )


def _toy_g(sub_tensor):
    num_branch = sub_tensor.shape[0]
    num_site = sub_tensor.shape[1]
    return {
        "threads": 1,
        "float_type": np.float64,
        "asrv": "no",
        "sub_sites": {"no": np.ones((num_branch, num_site), dtype=np.float64) / num_site},
        "N_ind_nomissing_gad": np.where(sub_tensor.sum(axis=(0, 1)) != 0),
        "N_ind_nomissing_ga": np.where(sub_tensor.sum(axis=(0, 1, 4)) != 0),
        "N_ind_nomissing_gd": np.where(sub_tensor.sum(axis=(0, 1, 3)) != 0),
    }


def test_calc_E_stat_mean_sparse_matches_dense_for_all_modes():
    dense = _toy_sub_tensor()
    sparse = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    cb = _toy_cb()
    g = _toy_g(dense)
    modes = ["spe2spe", "spe2any", "any2spe", "any2any"]
    for mode in modes:
        out_dense = omega.calc_E_stat(cb=cb, sub_tensor=dense, mode=mode, stat="mean", SN="N", g=g)
        out_sparse = omega.calc_E_stat(cb=cb, sub_tensor=sparse, mode=mode, stat="mean", SN="N", g=g)
        np.testing.assert_allclose(out_sparse, out_dense, atol=1e-12)


def test_resolve_E_stat_n_jobs_keeps_pepc_like_workload_single_thread():
    n_jobs, estimated_work = omega._resolve_E_stat_n_jobs(
        num_cb_rows=8446,
        num_site=956,
        num_categories=59,
        g={"threads": 3},
    )

    assert estimated_work == 8446 * 956 * 59
    assert n_jobs == 1


def test_resolve_E_stat_n_jobs_parallelizes_large_category_workload():
    n_jobs, estimated_work = omega._resolve_E_stat_n_jobs(
        num_cb_rows=50000,
        num_site=2000,
        num_categories=512,
        g={"threads": 4},
    )

    assert estimated_work == 50000 * 2000 * 512
    assert n_jobs == 4


def test_calc_E_stat_parallel_chunks_match_single_thread(monkeypatch):
    dense = _toy_sub_tensor()
    cb = _toy_cb()
    g_single = _toy_g(dense)
    g_parallel = _toy_g(dense)
    g_parallel["threads"] = 2
    monkeypatch.setattr(omega, "_DEFAULT_E_STAT_MIN_ITEMS_FOR_PARALLEL", 1)
    monkeypatch.setattr(omega, "_DEFAULT_E_STAT_MIN_CATEGORIES_PER_JOB", 1)
    invoked = {"parallel": False}
    orig_run_starmap = parallel.run_starmap

    def _wrapped_run_starmap(*args, **kwargs):
        invoked["parallel"] = True
        return orig_run_starmap(*args, **kwargs)

    monkeypatch.setattr(parallel, "run_starmap", _wrapped_run_starmap)
    out_parallel = omega.calc_E_stat(cb=cb, sub_tensor=dense, mode="any2spe", stat="mean", SN="N", g=g_parallel)
    out_single = omega.calc_E_stat(cb=cb, sub_tensor=dense, mode="any2spe", stat="mean", SN="N", g=g_single)

    assert invoked["parallel"] is True
    np.testing.assert_allclose(out_parallel, out_single, atol=1e-12)


def test_calc_E_stat_requires_g():
    with pytest.raises(ValueError, match="g is required"):
        omega.calc_E_stat(
            cb=_toy_cb(),
            sub_tensor=_toy_sub_tensor(),
            mode="any2any",
            stat="mean",
            SN="N",
            g=None,
        )


def test_calc_E_stat_rejects_unknown_mode():
    dense = _toy_sub_tensor()
    g = _toy_g(dense)
    with pytest.raises(ValueError, match="Unsupported E-stat mode"):
        omega.calc_E_stat(
            cb=_toy_cb(),
            sub_tensor=dense,
            mode="unknown",
            stat="mean",
            SN="N",
            g=g,
        )


def test_calc_E_stat_rejects_unknown_summary_statistic():
    dense = _toy_sub_tensor()
    g = _toy_g(dense)
    with pytest.raises(ValueError, match="Unsupported E-stat summary statistic"):
        omega.calc_E_stat(
            cb=_toy_cb(),
            sub_tensor=dense,
            mode="any2any",
            stat="median",
            SN="N",
            g=g,
        )


def test_get_exp_state_uses_branch_distance_props():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    num_node = max(labels.values()) + 1

    state = np.zeros((num_node, 1, 2), dtype=np.float64)
    state[labels["R"], 0, 0] = 1.0

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

    a_node = [n for n in tr.traverse() if n.name == "A"][0]
    ete.set_prop(a_node, "Ndist", 0.5)
    ete.set_prop(a_node, "SNdist", 0.5)
    # B keeps missing Ndist/SNdist to confirm default=0 behavior.

    pep = omega.get_exp_state(g=g, mode="pep")
    cdn = omega.get_exp_state(g=g, mode="cdn")

    assert pep[labels["A"], 0, :].sum() > 0
    assert cdn[labels["A"], 0, :].sum() > 0
    np.testing.assert_allclose(pep[labels["B"], 0, :], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(cdn[labels["B"], 0, :], [0.0, 0.0], atol=1e-12)


def test_collect_expected_state_branch_jobs_collapses_state_less_synthetic_parent():
    iqtree_like = ete.PhyloNode("(A:1,B:1,(C:1,D:1)Y:1)R;", format=1)
    rooted = ete.PhyloNode("(A:1,(B:1,(C:1,D:1)Y:1):1)RR;", format=1)
    tr = tree.add_numerical_node_labels(tree.transfer_root(tree_to=iqtree_like, tree_from=rooted))
    for node in tr.traverse():
        ete.set_prop(node, "Ndist", float(node.dist or 0.0))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse() if n.name}
    synthetic_node = [n for n in tr.traverse() if (not ete.is_leaf(n)) and (not ete.is_root(n)) and (n.name == "")][0]
    num_node = max(int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()) + 1
    state_has_mass = np.zeros((num_node,), dtype=bool)
    state_has_mass[labels["R"]] = True
    state_has_mass[labels["Y"]] = True
    state_has_mass[labels["A"]] = True
    state_has_mass[labels["B"]] = True
    state_has_mass[labels["C"]] = True
    state_has_mass[labels["D"]] = True
    jobs = omega._collect_expected_state_branch_jobs(
        tree=tr,
        mode="pep",
        num_node=num_node,
        float_tol=1e-12,
        state_has_mass=state_has_mass,
    )
    job_by_child = {child: (parent, branch_length) for child, parent, branch_length in jobs}
    expected_length = float([n for n in tr.traverse() if n.name == "B"][0].dist or 0.0) + float(synthetic_node.dist or 0.0)
    assert job_by_child[labels["B"]][0] == labels["R"]
    assert pytest.approx(job_by_child[labels["B"]][1], abs=1e-12) == expected_length


def test_get_exp_state_nsy_uses_nonsynonymous_rate_matrix():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    num_node = max(labels.values()) + 1
    state = np.zeros((num_node, 1, 2), dtype=np.float64)
    state[labels["R"], 0, 0] = 1.0
    g = {
        "tree": tr,
        "state_nsy": state.copy(),
        "instantaneous_nsy_rate_matrix": np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float64),
        "iqtree_rate_values": np.array([1.0], dtype=np.float64),
        "float_type": np.float64,
        "float_tol": 1e-12,
    }
    a_node = [n for n in tr.traverse() if n.name == "A"][0]
    ete.set_prop(a_node, "Ndist", 0.5)
    out = omega.get_exp_state(g=g, mode="nsy")
    assert out[labels["A"], 0, :].sum() > 0
    np.testing.assert_allclose(out[labels["B"], 0, :], [0.0, 0.0], atol=1e-12)


def test_reversible_expected_state_projector_matches_expm():
    stationary = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    exchangeability = np.array(
        [
            [0.0, 0.5, 0.2, 0.1],
            [0.5, 0.0, 0.3, 0.4],
            [0.2, 0.3, 0.0, 0.6],
            [0.1, 0.4, 0.6, 0.0],
        ],
        dtype=np.float64,
    )
    inst = exchangeability * stationary[None, :]
    np.fill_diagonal(inst, -inst.sum(axis=1))
    projector = omega._build_reversible_expected_state_projector(
        inst=inst,
        float_tol=1e-12,
        stationary=stationary,
    )
    assert projector is not None

    rng = np.random.default_rng(14)
    state = rng.random((2, 7, 4), dtype=np.float64)
    state /= state.sum(axis=2, keepdims=True)
    state_eigen = np.zeros_like(state)
    rates = np.array([0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0], dtype=np.float64)
    omega._project_expected_state_chunk_eigen(
        branch_jobs=[(1, 0, 0.7)],
        state=state,
        stateE=state_eigen,
        site_rates=rates,
        projector=projector,
        float_tol=1e-12,
    )
    expected = np.vstack([
        state[0, i, :] @ omega.expm(inst * 0.7 * rate)
        for i, rate in enumerate(rates)
    ])
    np.testing.assert_allclose(state_eigen[1, :, :], expected, atol=1e-12, rtol=1e-12)


def test_reversible_expected_state_projector_rejects_nonreversible_matrix():
    inst = np.array(
        [
            [-2.0, 2.0, 0.0],
            [0.0, -2.0, 2.0],
            [2.0, 0.0, -2.0],
        ],
        dtype=np.float64,
    )
    projector = omega._build_reversible_expected_state_projector(
        inst=inst,
        float_tol=1e-12,
    )
    assert projector is None


def test_general_expected_state_projector_matches_expm_for_nonreversible_matrix():
    inst = np.array(
        [
            [-2.0, 2.0, 0.0],
            [0.0, -2.0, 2.0],
            [2.0, 0.0, -2.0],
        ],
        dtype=np.float64,
    )
    projector = omega._build_expected_state_projector(inst=inst, float_tol=1e-12)
    assert projector is not None
    assert projector["kind"] == "general"
    parent = np.array([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5]], dtype=np.float64)
    rates = np.array([0.25, 1.5], dtype=np.float64)
    parent_eigen = omega._transform_parent_state_to_eigen(parent, projector)
    observed = omega._project_parent_eigen_state(
        parent_eigen_state=parent_eigen,
        branch_length=0.4,
        site_rates=rates,
        projector=projector,
        float_tol=1e-12,
    )
    expected = np.vstack([
        parent[i, :] @ omega.expm(inst * 0.4 * rate)
        for i, rate in enumerate(rates)
    ])
    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-12)


def test_get_exp_state_falls_back_to_expm_when_eigen_projection_is_unstable(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    state = np.zeros((max(labels.values()) + 1, 2, 2), dtype=np.float64)
    state[labels["R"], :, 0] = 1.0
    for node in tr.traverse():
        if not ete.is_root(node):
            ete.set_prop(node, "Ndist", 0.4)
    g = {
        "tree": tr,
        "state_nsy": state,
        "instantaneous_nsy_rate_matrix": np.array([[-0.5, 0.5], [0.25, -0.25]], dtype=np.float64),
        "iqtree_rate_values": np.array([0.5, 1.5], dtype=np.float64),
        "float_type": np.float64,
        "float_tol": 1e-12,
        "threads": 1,
        "expected_state_backend": "auto",
    }
    original_project = omega._project_parent_eigen_state

    def _fail_eigen(*_args, **_kwargs):
        raise FloatingPointError("synthetic instability")

    monkeypatch.setattr(omega, "_project_parent_eigen_state", _fail_eigen)
    with pytest.warns(RuntimeWarning, match="using scipy.linalg.expm"):
        fallback = omega.get_exp_state(g=g, mode="nsy")
    monkeypatch.setattr(omega, "_project_parent_eigen_state", original_project)
    g["expected_state_backend"] = "expm"
    expected = omega.get_exp_state(g=g, mode="nsy")
    np.testing.assert_allclose(fallback, expected, atol=1e-12, rtol=1e-12)


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


def test_project_expected_state_block_matches_numpy_fallback(monkeypatch):
    rng = np.random.default_rng(0)
    parent = rng.random((7, 4), dtype=np.float64)
    parent /= parent.sum(axis=1, keepdims=True)
    trans = rng.random((4, 4), dtype=np.float64)
    trans /= trans.sum(axis=1, keepdims=True)

    out_default = omega._project_expected_state_block(
        parent_state_block=parent,
        transition_prob=trans,
        float_tol=1e-12,
    )
    monkeypatch.setattr(omega, "_can_use_cython_expected_state", lambda *_args, **_kwargs: False)
    out_numpy = omega._project_expected_state_block(
        parent_state_block=parent,
        transition_prob=trans,
        float_tol=1e-12,
    )
    np.testing.assert_allclose(out_default, out_numpy, atol=1e-12)


def test_calc_tmp_E_sum_matches_numpy_fallback(monkeypatch):
    rng = np.random.default_rng(4)
    sub_sites = rng.random((9, 31), dtype=np.float64)
    sub_branches = rng.random(9, dtype=np.float64)
    cb_ids = rng.integers(0, 9, size=(12, 3), dtype=np.int64)
    out_default = omega._calc_tmp_E_sum(
        cb_ids=cb_ids,
        sub_sites=sub_sites,
        sub_branches=sub_branches,
        float_type=np.float64,
    )
    monkeypatch.setattr(omega, "_can_use_cython_tmp_E_sum", lambda *_args, **_kwargs: False)
    out_numpy = omega._calc_tmp_E_sum(
        cb_ids=cb_ids,
        sub_sites=sub_sites,
        sub_branches=sub_branches,
        float_type=np.float64,
    )
    np.testing.assert_allclose(out_default, out_numpy, atol=1e-12)


@pytest.mark.parametrize("arity", [1, 2, 3])
def test_calc_tmp_E_sum_with_cached_site_overlap_matches_direct(arity):
    rng = np.random.default_rng(14 + arity)
    sub_sites = rng.random((9, 31), dtype=np.float64)
    sub_branches = rng.random(9, dtype=np.float64)
    cb_ids = rng.integers(0, 9, size=(12, arity), dtype=np.int64)
    site_overlap = omega._calc_cb_site_overlap(
        cb_ids=cb_ids,
        sub_sites=sub_sites,
        float_type=np.float64,
    )
    out_direct = omega._calc_tmp_E_sum(
        cb_ids=cb_ids,
        sub_sites=sub_sites,
        sub_branches=sub_branches,
        float_type=np.float64,
    )
    out_cached = omega._calc_tmp_E_sum(
        cb_ids=cb_ids,
        sub_sites=sub_sites,
        sub_branches=sub_branches,
        float_type=np.float64,
        cb_site_overlap=site_overlap,
    )
    np.testing.assert_allclose(out_cached, out_direct, atol=1e-12)


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


def test_can_use_cython_tmp_E_sum_rejects_non_float64_inputs():
    cb_ids = np.array([[0, 1]], dtype=np.int64)
    sub_sites = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    sub_branches = np.array([0.5, 0.6], dtype=np.float64)
    assert omega._can_use_cython_tmp_E_sum(cb_ids, sub_sites, sub_branches) is False


def test_can_use_cython_expected_state_rejects_large_state_space():
    parent = np.full((2, 20), 1.0 / 20.0, dtype=np.float64)
    trans = np.eye(20, dtype=np.float64)
    assert omega._can_use_cython_expected_state(parent, trans) is False


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


def test_expected_state_transition_cache_retains_only_latest_branch_length(monkeypatch):
    state = np.zeros((4, 2, 2), dtype=np.float64)
    state[0, :, 0] = 1.0
    state_e = np.zeros_like(state)
    calls = []

    def _fake_expm(matrix):
        calls.append(matrix.copy())
        return np.eye(2, dtype=np.float64)

    monkeypatch.setattr(omega, "expm", _fake_expm)
    omega._project_expected_state_chunk(
        branch_jobs=[(1, 0, 0.5), (2, 0, 1.0), (3, 0, 0.5)],
        state=state,
        stateE=state_e,
        unique_site_rates=np.array([0.5, 1.0], dtype=np.float64),
        rate_site_indices=[np.array([0]), np.array([1])],
        inst=np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float64),
        float_tol=1e-12,
    )

    # The first branch length was evicted when the second was processed, so
    # returning to it recomputes its two rate-specific matrices.
    assert len(calls) == 6


def test_get_exp_state_respects_parallel_threshold(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1):1,C:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    num_node = max(labels.values()) + 1
    rng = np.random.default_rng(11)
    state = rng.random((num_node, 8, 4), dtype=np.float64)
    state /= state.sum(axis=2, keepdims=True)
    g = {
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
        "iqtree_rate_values": np.ones(8, dtype=np.float64),
        "float_type": np.float64,
        "float_tol": 1e-12,
        "threads": 4,
    }
    for node in tr.traverse():
        if ete.is_root(node):
            continue
        ete.set_prop(node, "Ndist", 0.1)
    invoked = {"parallel": False}
    orig_run_starmap = parallel.run_starmap

    def _wrapped_run_starmap(*args, **kwargs):
        invoked["parallel"] = True
        return orig_run_starmap(*args, **kwargs)

    monkeypatch.setattr(parallel, "run_starmap", _wrapped_run_starmap)
    out = omega.get_exp_state(g=g, mode="pep")
    assert invoked["parallel"] is False
    assert np.isfinite(out).all()


def test_resolve_expected_state_n_jobs_uses_param_default_thresholds():
    n_jobs, estimated_work = omega._resolve_expected_state_n_jobs(
        num_branch_jobs=139,
        num_site=956,
        num_state=20,
        g={"threads": 3},
    )

    assert estimated_work == 139 * 956 * 20 * 20
    assert n_jobs == 3


def test_calibrate_dsc_skips_substitution_class_without_finite_pairs():
    combinatorial_substitutions = [
        "any2any",
        "any2spe",
        "any2dif",
        "dif2any",
        "dif2spe",
        "dif2dif",
        "spe2any",
        "spe2spe",
        "spe2dif",
    ]
    row = {"branch_id_1": 0}
    for sub in combinatorial_substitutions:
        row["dNC" + sub] = np.nan
        row["dSC" + sub] = np.nan
        row["omegaC" + sub] = np.nan
    cb = pd.DataFrame([row])

    out = omega.calibrate_dsc(cb=cb.copy())

    for sub in combinatorial_substitutions:
        assert "dSC" + sub in out.columns
        assert "omegaC" + sub in out.columns
        assert "dSC" + sub + "_nocalib" not in out.columns
        assert "omegaC" + sub + "_nocalib" not in out.columns


def test_get_omega_applies_dirichlet_pseudocount_to_any2spe_columns():
    cb = pd.DataFrame(
        {
            "OCNany2spe": [0.0],
            "ECNany2spe": [0.0],
            "OCSany2spe": [1.0],
            "ECSany2spe": [0.0],
        }
    )
    g = {
        "float_tol": 1e-12,
        "output_stat": "any2spe",
        "pseudocount_alpha": 0.5,
        "pseudocount_mode": "symmetric",
        "pseudocount_target": "both",
        "pseudocount_report": False,
    }
    out = omega.get_omega(cb=cb.copy(), g=g)
    assert np.isfinite(out.loc[0, "omegaCany2spe"])
    assert out.loc[0, "omegaCany2spe"] == pytest.approx(1.0 / 3.0)
    assert out.loc[0, "omegaCany2spe_raw"] == pytest.approx(0.0)
    assert "omegaCany2spe_smoothed" in out.columns
    assert "logomegaCany2spe_smoothed" in out.columns


def test_get_omega_keeps_legacy_behavior_when_pseudocount_disabled():
    cb = pd.DataFrame(
        {
            "OCNany2spe": [2.0],
            "ECNany2spe": [4.0],
            "OCSany2spe": [1.0],
            "ECSany2spe": [2.0],
        }
    )
    g = {
        "float_tol": 1e-12,
        "output_stat": "any2spe",
        "pseudocount_alpha": 0.0,
        "pseudocount_mode": "none",
        "pseudocount_target": "both",
        "pseudocount_report": False,
    }
    out = omega.get_omega(cb=cb.copy(), g=g)
    assert out.loc[0, "dNCany2spe"] == pytest.approx(0.5)
    assert out.loc[0, "dSCany2spe"] == pytest.approx(0.5)
    assert out.loc[0, "omegaCany2spe"] == pytest.approx(1.0)
    assert "omegaCany2spe_raw" not in out.columns


def test_get_omega_accepts_auto_alpha_and_estimates_finite_smoothed_rates():
    cb = pd.DataFrame(
        {
            "OCNany2spe": [0.0, 1.0, 0.0, 3.0],
            "ECNany2spe": [0.0, 1.0, 2.0, 0.0],
            "OCSany2spe": [0.0, 0.0, 1.0, 2.0],
            "ECSany2spe": [0.0, 1.0, 1.0, 0.0],
        }
    )
    g = {
        "float_tol": 1e-12,
        "output_stat": "any2spe",
        "pseudocount_alpha": "auto",
        "pseudocount_mode": "symmetric",
        "pseudocount_target": "both",
        "pseudocount_report": False,
    }
    out = omega.get_omega(cb=cb.copy(), g=g)
    assert np.isfinite(out["dNCany2spe"].to_numpy()).all()
    assert np.isfinite(out["dSCany2spe"].to_numpy()).all()
    assert np.isfinite(out["omegaCany2spe"].to_numpy()).all()


def test_get_omega_accepts_prevalidated_auto_alpha_configuration():
    cb = pd.DataFrame(
        {
            "OCNany2spe": [0.0, 1.0, 0.0, 3.0],
            "ECNany2spe": [0.0, 1.0, 2.0, 0.0],
            "OCSany2spe": [0.0, 0.0, 1.0, 2.0],
            "ECSany2spe": [0.0, 1.0, 1.0, 0.0],
        }
    )
    g = {
        "float_tol": 1e-12,
        "output_stat": "any2spe",
        "pseudocount_alpha": 0.0,
        "pseudocount_alpha_auto": True,
        "pseudocount_mode": "symmetric",
        "pseudocount_target": "both",
        "pseudocount_report": False,
    }
    out = omega.get_omega(cb=cb.copy(), g=g)
    assert "omegaCany2spe_raw" in out.columns
    assert np.isfinite(out["omegaCany2spe"].to_numpy()).all()
    assert not np.allclose(
        out["omegaCany2spe"].to_numpy(dtype=np.float64),
        out["omegaCany2spe_raw"].to_numpy(dtype=np.float64),
        equal_nan=True,
    )


def test_get_omega_empirical_mode_returns_finite_values():
    cb = pd.DataFrame(
        {
            "OCNany2any": [10.0],
            "ECNany2any": [8.0],
            "OCSany2any": [8.0],
            "ECSany2any": [7.0],
            "OCNany2spe": [0.0],
            "ECNany2spe": [3.0],
            "OCSany2spe": [0.0],
            "ECSany2spe": [2.0],
        }
    )
    g = {
        "float_tol": 1e-12,
        "output_stat": "any2any,any2spe",
        "pseudocount_alpha": 0.5,
        "pseudocount_mode": "empirical",
        "pseudocount_target": "both",
        "pseudocount_report": False,
    }
    out = omega.get_omega(cb=cb.copy(), g=g)
    assert np.isfinite(out.loc[0, "omegaCany2any"])
    assert np.isfinite(out.loc[0, "omegaCany2spe"])
