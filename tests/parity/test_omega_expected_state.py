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
