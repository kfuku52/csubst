import numpy as np
import pytest

from csubst import omega
from csubst import tree
from csubst import ete


def test_prepare_epistasis_supports_s_channel_and_applies_only_to_ocs():
    ON_tensor = np.zeros(shape=(2, 3, 1, 1, 1), dtype=np.float64)
    OS_tensor = np.zeros(shape=(2, 3, 1, 1, 1), dtype=np.float64)
    OS_tensor[0, :, 0, 0, 0] = np.array([2.0, 1.0, 0.0], dtype=np.float64)
    OS_tensor[1, :, 0, 0, 0] = np.array([0.0, 1.0, 3.0], dtype=np.float64)
    g = {
        "float_tol": 1e-12,
        "is_site_nonmissing": np.ones(shape=(2, 3), dtype=bool),
        "_asrv_branch_ids": np.array([0, 1], dtype=np.int64),
        "epistasis_apply_to": "S",
        "epistasis_beta_auto": False,
        "epistasis_beta_value": 1.0,
        "epistasis_clip_auto": False,
        "epistasis_clip_value": 3.0,
        "asrv_dirichlet_alpha": 1.0,
        "epistasis_site_degree_internal": np.array([-1.0, 0.0, 1.0], dtype=np.float64),
    }
    g = omega.prepare_epistasis(g=g, ON_tensor=ON_tensor, OS_tensor=OS_tensor)
    assert g["epistasis_enabled"] is True
    assert "S" in g["_epistasis_state"]
    assert "N" not in g["_epistasis_state"]
    sub_sites = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.2, 0.3, 0.5],
        ],
        dtype=np.float64,
    )
    out_s = omega._apply_epistasis_to_sub_sites(sub_sites=sub_sites, obs_col="OCSany2any", g=g)
    out_n = omega._apply_epistasis_to_sub_sites(sub_sites=sub_sites, obs_col="OCNany2any", g=g)
    assert not np.allclose(out_s, sub_sites)
    np.testing.assert_allclose(out_n, sub_sites)


def test_prepare_epistasis_prints_negative_control_summary_for_s_channel(capsys):
    ON_tensor = np.zeros(shape=(2, 3, 1, 1, 1), dtype=np.float64)
    OS_tensor = np.zeros(shape=(2, 3, 1, 1, 1), dtype=np.float64)
    OS_tensor[0, :, 0, 0, 0] = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    OS_tensor[1, :, 0, 0, 0] = np.array([0.0, 2.0, 1.0], dtype=np.float64)
    g = {
        "float_tol": 1e-12,
        "is_site_nonmissing": np.ones(shape=(2, 3), dtype=bool),
        "_asrv_branch_ids": np.array([0, 1], dtype=np.int64),
        "epistasis_apply_to": "S",
        "epistasis_beta_auto": False,
        "epistasis_beta_value": 1.0,
        "epistasis_clip_auto": False,
        "epistasis_clip_value": 3.0,
        "asrv_dirichlet_alpha": 1.0,
        "epistasis_site_metric": "proximity",
        "epistasis_site_metric_resolved": "proximity",
        "epistasis_site_degree_internal": np.array([-1.0, 0.0, 1.0], dtype=np.float64),
    }
    omega.prepare_epistasis(g=g, ON_tensor=ON_tensor, OS_tensor=OS_tensor)
    captured = capsys.readouterr()
    assert "Epistasis summary: apply_to=S, site_metric=proximity" in captured.out
    assert "Epistasis negative-control summary: apply_to=S" in captured.out


def test_apply_epistasis_beta_to_probs_supports_multifeature_site_profiles():
    base_probs = np.full((2, 3), 1.0 / 3.0, dtype=np.float64)
    mask = np.ones_like(base_probs, dtype=bool)
    branch_context = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    site_features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=np.float64,
    )
    out = omega._apply_epistasis_beta_to_probs(
        base_probs=base_probs,
        branch_context=branch_context,
        degree_z=site_features,
        beta=1.5,
        clip_value=5.0,
        is_site_nonmissing=mask,
        float_tol=1e-12,
    )
    assert out.shape == base_probs.shape
    assert out[0, 0] > out[0, 1]
    assert out[1, 1] > out[1, 0]


def test_prepare_epistasis_branch_depth_partition_records_branchwise_parameters():
    ON_tensor = np.zeros(shape=(4, 3, 1, 1, 1), dtype=np.float64)
    OS_tensor = np.zeros(shape=(4, 3, 1, 1, 1), dtype=np.float64)
    ON_tensor[:, :, 0, 0, 0] = np.array(
        [
            [3.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    g = {
        "float_tol": 1e-12,
        "is_site_nonmissing": np.ones(shape=(4, 3), dtype=bool),
        "_asrv_branch_ids": np.array([0, 1, 2, 3], dtype=np.int64),
        "epistasis_apply_to": "N",
        "epistasis_beta_auto": False,
        "epistasis_beta_value": 0.8,
        "epistasis_clip_auto": False,
        "epistasis_clip_value": 3.0,
        "epistasis_beta_partition": "branch_depth",
        "epistasis_branch_depth_bins": 2,
        "epistasis_joint_auto": False,
        "asrv_dirichlet_alpha": 1.0,
        "epistasis_site_degree_internal": np.array([-1.0, 0.0, 1.0], dtype=np.float64),
    }
    out = omega.prepare_epistasis(g=g, ON_tensor=ON_tensor, OS_tensor=OS_tensor)
    state = out["_epistasis_state"]["N"]
    assert state["beta_diag"]["partition"] == "branch_depth"
    assert len(state["beta_diag"]["bins"]) == 2
    assert state["beta_by_branch"].shape == (4,)
    assert state["clip_by_branch"].shape == (4,)
    np.testing.assert_allclose(state["beta_by_branch"], np.full((4,), 0.8, dtype=np.float64))


@pytest.mark.slow
def test_prepare_epistasis_joint_auto_selects_alpha_and_clip_from_grid():
    ON_tensor = np.zeros(shape=(4, 3, 1, 1, 1), dtype=np.float64)
    OS_tensor = np.zeros(shape=(4, 3, 1, 1, 1), dtype=np.float64)
    ON_tensor[:, :, 0, 0, 0] = np.array(
        [
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    g = {
        "float_tol": 1e-12,
        "is_site_nonmissing": np.ones(shape=(4, 3), dtype=bool),
        "_asrv_branch_ids": np.array([0, 1, 2, 3], dtype=np.int64),
        "epistasis_apply_to": "N",
        "epistasis_beta_auto": True,
        "epistasis_beta_value": np.nan,
        "epistasis_clip_auto": True,
        "epistasis_clip_value": np.nan,
        "epistasis_joint_auto": True,
        "epistasis_joint_alpha_grid": [0.0, 0.5, 1.0],
        "epistasis_joint_clip_grid": [1.5, 2.0, 3.0],
        "epistasis_beta_partition": "global",
        "asrv_dirichlet_alpha": 1.0,
        "epistasis_site_degree_internal": np.array([-1.0, 0.0, 1.0], dtype=np.float64),
    }
    out = omega.prepare_epistasis(g=g, ON_tensor=ON_tensor, OS_tensor=OS_tensor)
    state = out["_epistasis_state"]["N"]
    assert state["alpha"] in {0.0, 0.5, 1.0}
    assert state["clip"] in {1.5, 2.0, 3.0}


def test_get_epistasis_branch_depth_by_id_uses_cumulative_branch_lengths():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:2)X:3,C:4)R;", format=1))
    by_name = {n.name: n for n in tr.traverse()}
    max_id = max(int(ete.get_prop(n, "numerical_label")) for n in tr.traverse())
    out = omega._get_epistasis_branch_depth_by_id(g={"tree": tr}, num_branch=max_id + 1)
    id_A = int(ete.get_prop(by_name["A"], "numerical_label"))
    id_B = int(ete.get_prop(by_name["B"], "numerical_label"))
    id_X = int(ete.get_prop(by_name["X"], "numerical_label"))
    id_C = int(ete.get_prop(by_name["C"], "numerical_label"))
    assert out[id_X] == pytest.approx(3.0, abs=1e-12)
    assert out[id_A] == pytest.approx(4.0, abs=1e-12)
    assert out[id_B] == pytest.approx(5.0, abs=1e-12)
    assert out[id_C] == pytest.approx(4.0, abs=1e-12)


def test_fit_epistasis_beta_cv_returns_zero_when_active_branches_are_insufficient():
    counts = np.array([[2.0, 1.0, 0.0]], dtype=np.float64)
    mask = np.ones(shape=counts.shape, dtype=bool)
    base_probs = np.array([[2.0 / 3.0, 1.0 / 3.0, 0.0]], dtype=np.float64)
    branch_context = np.array([0.1], dtype=np.float64)
    degree_z = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    beta, diag = omega._fit_epistasis_beta_cv(
        counts=counts,
        base_probs=base_probs,
        branch_context=branch_context,
        degree_z=degree_z,
        is_site_nonmissing=mask,
        clip_value=3.0,
        float_tol=1e-12,
    )
    assert beta == pytest.approx(0.0)
    assert diag["active_branch_count"] == 1


def test_fit_epistasis_beta_cv_detects_positive_signal():
    degree_z = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    branch_context = np.array([-1.2, -0.8, -0.4, 0.4, 0.8, 1.2], dtype=np.float64)
    mask = np.ones(shape=(branch_context.shape[0], degree_z.shape[0]), dtype=bool)
    base_probs = np.full(shape=mask.shape, fill_value=(1.0 / 3.0), dtype=np.float64)
    probs = omega._apply_epistasis_beta_to_probs(
        base_probs=base_probs,
        branch_context=branch_context,
        degree_z=degree_z,
        beta=1.2,
        clip_value=5.0,
        is_site_nonmissing=mask,
        float_tol=1e-12,
    )
    counts = np.round(probs * 120.0).astype(np.float64)
    beta, diag = omega._fit_epistasis_beta_cv(
        counts=counts,
        base_probs=base_probs,
        branch_context=branch_context,
        degree_z=degree_z,
        is_site_nonmissing=mask,
        clip_value=5.0,
        float_tol=1e-12,
        dirichlet_alpha=1.0,
    )
    assert beta > 0.0
    assert diag["selection_rule"] == "argmax_mean_cv_loglik"
    assert diag["best_beta"] == pytest.approx(beta)


def test_auto_epistasis_clip_stays_within_bounds():
    branch_context = np.array([0.0, 1.0], dtype=np.float64)
    degree_z = np.array([-2.0, 0.0, 2.0], dtype=np.float64)
    mask = np.ones(shape=(2, 3), dtype=bool)
    clip = omega._auto_epistasis_clip(
        beta=10.0,
        branch_context=branch_context,
        degree_z=degree_z,
        is_site_nonmissing=mask,
    )
    assert clip >= 1.5
    assert clip <= 5.0
