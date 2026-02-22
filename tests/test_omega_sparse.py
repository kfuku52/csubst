import numpy as np
import pandas as pd
import pytest

from csubst import omega
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
