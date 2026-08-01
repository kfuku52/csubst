import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from csubst import main_sites
from csubst import substitution
from csubst import substitution_sparse
from csubst import tree
from csubst import ete


def _rerooted_tree_with_state_less_synthetic_internal():
    iqtree_like = ete.PhyloNode("(A:1,B:1,(C:1,D:1)Y:1)R;", format=1)
    rooted = ete.PhyloNode("(A:1,(B:1,(C:1,D:1)Y:1):1)RR;", format=1)
    return tree.add_numerical_node_labels(tree.transfer_root(tree_to=iqtree_like, tree_from=rooted))


def test_get_substitution_tensor_asis_matches_manual_outer_products():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    state = np.zeros((3, 2, 2), dtype=float)
    state[labels["R"], :, :] = [[1.0, 0.0], [0.5, 0.5]]
    state[labels["A"], :, :] = [[0.0, 1.0], [1.0, 0.0]]
    state[labels["B"], :, :] = [[1.0, 0.0], [0.0, 1.0]]
    g = {"tree": tr, "ml_anc": "yes", "float_tol": 1e-12}
    mmap_file = Path("tmp.csubst.sub_tensor.toy.mmap")
    try:
        out = substitution.get_substitution_tensor(state_tensor=state, mode="asis", g=g, mmap_attr="toy")
        assert isinstance(out, substitution_sparse.SparseSubstitutionTensor)
        out_dense = out.to_dense()
        # Branch A, site 0: parent state 0 -> child state 1 with prob 1.
        np.testing.assert_allclose(out_dense[labels["A"], 0, 0, :, :], [[0.0, 1.0], [0.0, 0.0]], atol=1e-12)
        # Branch A, site 1: parent [0.5, 0.5], child state 0 => only 1->0 survives diag masking.
        np.testing.assert_allclose(out_dense[labels["A"], 1, 0, :, :], [[0.0, 0.0], [0.5, 0.0]], atol=1e-12)
    finally:
        if mmap_file.exists():
            mmap_file.unlink()


def test_apply_min_sub_pp_threshold():
    g = {"min_sub_pp": 0.3, "ml_anc": False}
    sub = np.array([[[[[0.2, 0.4], [0.1, 0.5]]]]], dtype=float)
    out = substitution.apply_min_sub_pp(g, sub)
    np.testing.assert_allclose(out, [[[[[0.0, 0.4], [0.0, 0.5]]]]], atol=1e-12)


def test_apply_min_sub_pp_parses_ml_anc_string_no_as_false():
    g = {"min_sub_pp": 0.3, "ml_anc": "no"}
    sub = np.array([[[[[0.2, 0.4], [0.1, 0.5]]]]], dtype=float)
    out = substitution.apply_min_sub_pp(g, sub)
    np.testing.assert_allclose(out, [[[[[0.0, 0.4], [0.0, 0.5]]]]], atol=1e-12)


def test_get_substitution_tensor_treats_ml_anc_string_no_same_as_false():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    state = np.zeros((3, 1, 2), dtype=float)
    state[labels["R"], 0, :] = [1.0, 0.0]
    state[labels["A"], 0, :] = [0.0, 1.0]
    state[labels["B"], 0, :] = [1.0, 0.0]

    g_bool = {"tree": tr, "ml_anc": False, "float_tol": 1e-12}
    g_str = {"tree": tr, "ml_anc": "no", "float_tol": 1e-12}
    out_bool = substitution.get_substitution_tensor(state_tensor=state, mode="asis", g=g_bool, mmap_attr="toy_bool")
    out_str = substitution.get_substitution_tensor(state_tensor=state, mode="asis", g=g_str, mmap_attr="toy_str")
    dense_bool = out_bool.to_dense()
    np.testing.assert_allclose(dense_bool, out_str.to_dense(), atol=1e-12)
    np.testing.assert_allclose(dense_bool[labels["R"], :, :, :, :], 0.0, atol=1e-12)


def test_get_s_get_cs_and_get_bs_match_manual_values():
    # shape = [branch, site, group, from, to]
    sub = np.zeros((2, 2, 1, 2, 2), dtype=float)
    sub[0, 0, 0, :, :] = [[0.0, 0.2], [0.1, 0.0]]
    sub[1, 0, 0, :, :] = [[0.0, 0.5], [0.2, 0.0]]
    sub[0, 1, 0, :, :] = [[0.0, 0.4], [0.0, 0.0]]
    sub[1, 1, 0, :, :] = [[0.0, 0.1], [0.3, 0.0]]

    s = substitution.get_s(sub, attr="N")
    np.testing.assert_allclose(s["N_sub"].to_numpy(), [1.0, 0.8], atol=1e-12)

    cs = substitution.get_cs(np.array([[0, 1]]), sub, attr="N")
    np.testing.assert_allclose(cs["OCNany2any"].to_numpy(), [0.21, 0.16], atol=1e-12)
    np.testing.assert_allclose(cs["OCNspe2any"].to_numpy(), [0.12, 0.04], atol=1e-12)
    np.testing.assert_allclose(cs["OCNany2spe"].to_numpy(), [0.12, 0.04], atol=1e-12)
    np.testing.assert_allclose(cs["OCNspe2spe"].to_numpy(), [0.12, 0.04], atol=1e-12)

    bs = substitution.get_bs(S_tensor=sub, N_tensor=sub * 2.0)
    # first branch, first site
    row0 = bs.loc[(bs["branch_id"] == 0) & (bs["site"] == 0), :].iloc[0]
    assert pytest.approx(float(row0["S_sub"]), abs=1e-12) == 0.3
    assert pytest.approx(float(row0["N_sub"]), abs=1e-12) == 0.6


def test_get_b_uses_tree_numerical_labels_for_branch_ids(tiny_tree):
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    sub = np.zeros((num_node, 1, 1, 2, 2), dtype=float)
    a_id = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse() if n.name == "A"][0]
    sub[a_id, 0, 0, 0, 1] = 0.5

    out = substitution.get_b(g={"tree": tiny_tree, "num_node": num_node}, sub_tensor=sub, attr="S", sitewise=False)
    expected_ids = sorted([ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()])
    assert out["branch_id"].tolist() == expected_ids
    assert set(out["branch_name"]) == set([n.name for n in tiny_tree.traverse()])


def test_get_b_sitewise_skips_nan_only_sites_without_crashing(tiny_tree):
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    sub = np.zeros((num_node, 1, 1, 2, 2), dtype=float)
    a_id = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse() if n.name == "A"][0]
    sub[a_id, 0, 0, :, :] = np.nan
    out = substitution.get_b(
        g={"tree": tiny_tree, "num_node": num_node, "amino_acid_orders": ["A", "B"]},
        sub_tensor=sub,
        attr="N",
        sitewise=True,
    )
    assert "N_sitewise" in out.columns
    assert out.loc[out["branch_id"] == a_id, "N_sitewise"].iloc[0] == ""


def test_get_b_sitewise_uses_nonsyn_state_orders_when_available(tiny_tree):
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    sub = np.zeros((num_node, 1, 1, 2, 2), dtype=float)
    a_id = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse() if n.name == "A"][0]
    sub[a_id, 0, 0, 0, 1] = 0.9
    out = substitution.get_b(
        g={
            "tree": tiny_tree,
            "num_node": num_node,
            "amino_acid_orders": ["A", "B"],
            "nonsyn_state_orders": ["grpX", "grpY"],
        },
        sub_tensor=sub,
        attr="N",
        sitewise=True,
    )
    assert out.loc[out["branch_id"] == a_id, "N_sitewise"].iloc[0] == "grpX1grpY"


def test_get_substitution_tensor_collapses_state_less_synthetic_parent():
    tr = _rerooted_tree_with_state_less_synthetic_internal()
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse() if n.name}
    synthetic_id = [int(ete.get_prop(n, "numerical_label")) for n in tr.traverse() if (not ete.is_leaf(n)) and (not ete.is_root(n)) and (n.name == "")][0]
    num_node = max(int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()) + 1
    state = np.zeros((num_node, 1, 2), dtype=float)
    state[labels["R"], 0, :] = [1.0, 0.0]
    state[labels["Y"], 0, :] = [1.0, 0.0]
    state[labels["A"], 0, :] = [1.0, 0.0]
    state[labels["B"], 0, :] = [0.0, 1.0]
    state[labels["C"], 0, :] = [1.0, 0.0]
    state[labels["D"], 0, :] = [1.0, 0.0]
    g = {"tree": tr, "ml_anc": False, "float_tol": 1e-12}
    pairs = substitution._collect_sub_tensor_branch_pairs(g=g, state_tensor_anc=state, selected_branch_set=None)
    assert (labels["B"], labels["R"]) in pairs
    assert all(child != synthetic_id for child, _parent in pairs)

    out = substitution.get_substitution_tensor(state_tensor=state, mode="asis", g=g, mmap_attr="synthetic_parent")
    assert pytest.approx(float(np.nansum(out.to_dense()[labels["B"]])), abs=1e-12) == 1.0


def test_get_parent_branch_ids_skips_state_less_synthetic_parent():
    tr = _rerooted_tree_with_state_less_synthetic_internal()
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse() if n.name}
    num_node = max(int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()) + 1
    state_cdn = np.zeros((num_node, 1, 2), dtype=float)
    state_cdn[labels["R"], 0, :] = [1.0, 0.0]
    state_cdn[labels["Y"], 0, :] = [1.0, 0.0]
    state_cdn[labels["A"], 0, :] = [1.0, 0.0]
    state_cdn[labels["B"], 0, :] = [0.0, 1.0]
    state_cdn[labels["C"], 0, :] = [1.0, 0.0]
    state_cdn[labels["D"], 0, :] = [1.0, 0.0]
    out = main_sites.get_parent_branch_ids(
        branch_ids=[labels["B"], labels["Y"]],
        g={"tree": tr, "state_cdn": state_cdn, "float_tol": 1e-12},
    )
    assert out[labels["B"]] == labels["R"]
    assert out[labels["Y"]] == labels["R"]


def test_get_sub_sites_handles_noncontiguous_branch_ids():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,(B:1,C:1)X:1)R;", format=1))
    reassigned = {"A": 11, "B": 29, "C": 41, "X": 73, "R": 5}
    for node in tr.traverse():
        ete.set_prop(node, "numerical_label", reassigned[node.name])
    num_branch = max(reassigned.values()) + 1
    num_site = 3
    state_tensor = np.zeros((num_branch, num_site, 2), dtype=float)
    for node in tr.traverse():
        nl = int(ete.get_prop(node, "numerical_label"))
        state_tensor[nl, :, :] = 1.0
    sS = pd.DataFrame({"S_sub": [1.0, 2.0, 3.0]})
    sN = pd.DataFrame({"N_sub": [1.0, 1.0, 1.0]})
    g = {"tree": tr, "asrv": "pool", "float_type": np.float64}

    out = substitution.get_sub_sites(g=g, sS=sS, sN=sN, state_tensor=state_tensor)

    assert out["is_site_nonmissing"].shape == (num_branch, num_site)
    assert out["sub_sites"]["pool"].shape == (num_branch, num_site)
    assert not out["is_site_nonmissing"][0, :].any()
    for node in tr.traverse():
        nl = int(ete.get_prop(node, "numerical_label"))
        assert out["is_site_nonmissing"][nl, :].all()
        np.testing.assert_allclose(out["sub_sites"]["pool"][nl, :].sum(), 1.0, atol=1e-12)


def test_get_sub_sites_raises_when_tree_branch_id_exceeds_state_tensor_axis():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    for node in tr.traverse():
        ete.set_prop(node, "numerical_label", int(ete.get_prop(node, "numerical_label")) + 10)
    state_tensor = np.zeros((3, 2, 2), dtype=float)
    sS = pd.DataFrame({"S_sub": [1.0, 1.0]})
    sN = pd.DataFrame({"N_sub": [1.0, 1.0]})
    g = {"tree": tr, "asrv": "pool", "float_type": np.float64}
    with pytest.raises(ValueError, match="out of bounds"):
        substitution.get_sub_sites(g=g, sS=sS, sN=sN, state_tensor=state_tensor)


def test_get_sub_sites_sn_applies_dirichlet_pseudocount():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    num_branch = max(int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()) + 1
    state_tensor = np.ones((num_branch, 2, 2), dtype=float)
    sS = pd.DataFrame({"S_sub": [2.0, 0.0]})
    sN = pd.DataFrame({"N_sub": [0.0, 1.0]})
    g = {
        "tree": tr,
        "asrv": "sn",
        "float_type": np.float64,
        "asrv_dirichlet_alpha": 1.0,
    }

    out = substitution.get_sub_sites(g=g, sS=sS, sN=sN, state_tensor=state_tensor)

    for node in tr.traverse():
        nl = int(ete.get_prop(node, "numerical_label"))
        np.testing.assert_allclose(out["sub_sites"]["S"][nl, :], np.array([0.75, 0.25]), atol=1e-12)
        np.testing.assert_allclose(out["sub_sites"]["N"][nl, :], np.array([1.0 / 3.0, 2.0 / 3.0]), atol=1e-12)


def test_get_each_sub_sites_applies_dirichlet_pseudocount():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    num_branch = max(int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()) + 1
    state_tensor = np.ones((num_branch, 2, 2), dtype=float)
    sS = pd.DataFrame({"S_sub": [1.0, 1.0]})
    sN = pd.DataFrame({"N_sub": [1.0, 1.0]})
    g = {
        "tree": tr,
        "asrv": "each",
        "float_type": np.float64,
        "asrv_dirichlet_alpha": 1.0,
    }
    g = substitution.get_sub_sites(g=g, sS=sS, sN=sN, state_tensor=state_tensor)
    sub_sg = np.array([[2.0], [0.0]], dtype=np.float64)  # [site, group]

    out = substitution.get_each_sub_sites(sub_sg=sub_sg, mode="any2any", sg=0, a=0, d=0, g=g)

    for node in tr.traverse():
        nl = int(ete.get_prop(node, "numerical_label"))
        np.testing.assert_allclose(out[nl, :], np.array([0.75, 0.25]), atol=1e-12)


def test_get_each_sub_sites_file_each_uses_rate_hybrid_weight():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    num_branch = max(int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()) + 1
    state_tensor = np.ones((num_branch, 2, 2), dtype=float)
    sS = pd.DataFrame({"S_sub": [1.0, 1.0]})
    sN = pd.DataFrame({"N_sub": [1.0, 1.0]})
    g = {
        "tree": tr,
        "asrv": "file_each",
        "float_type": np.float64,
        "asrv_dirichlet_alpha": 0.0,
        "iqtree_rate_values": np.array([0.5, 2.0], dtype=np.float64),
    }
    g = substitution.get_sub_sites(g=g, sS=sS, sN=sN, state_tensor=state_tensor)
    sub_sg = np.array([[2.0], [1.0]], dtype=np.float64)  # [site, group]

    out = substitution.get_each_sub_sites(sub_sg=sub_sg, mode="any2any", sg=0, a=0, d=0, g=g)

    for node in tr.traverse():
        nl = int(ete.get_prop(node, "numerical_label"))
        np.testing.assert_allclose(out[nl, :], np.array([1.0 / 3.0, 2.0 / 3.0]), atol=1e-12)


def test_normalize_site_weights_by_branch_cython_matches_python_fallback(monkeypatch):
    if not hasattr(substitution.substitution_cy, "normalize_branch_site_weights_double"):
        pytest.skip("Cython branch-site normalization fast path is unavailable")
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    num_branch = max(int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()) + 1
    state_tensor = np.ones((num_branch, 3, 2), dtype=float)
    sS = pd.DataFrame({"S_sub": [1.0, 1.0, 1.0]})
    sN = pd.DataFrame({"N_sub": [1.0, 1.0, 1.0]})
    g = {
        "tree": tr,
        "asrv": "each",
        "float_type": np.float64,
        "asrv_dirichlet_alpha": 0.0,
    }
    g = substitution.get_sub_sites(g=g, sS=sS, sN=sN, state_tensor=state_tensor)
    weights = np.array([0.2, 0.0, 0.8], dtype=np.float64)
    out_fast = substitution._normalize_site_weights_by_branch(
        nonadjusted_sub_sites=weights,
        g=g,
        dirichlet_alpha=0.3,
    )

    monkeypatch.setattr(substitution, "_can_use_cython_site_weight_normalization", lambda *_args, **_kwargs: False)
    out_py = substitution._normalize_site_weights_by_branch(
        nonadjusted_sub_sites=weights,
        g=g,
        dirichlet_alpha=0.3,
    )
    np.testing.assert_allclose(out_fast, out_py, atol=1e-12)


def test_add_dif_column_and_add_dif_stats():
    cb = pd.DataFrame(
        {
            "OCSany2any": [1.0, 1.0],
            "OCSany2spe": [0.4, 1.2],
            "OCSspe2any": [0.5, 0.2],
            "OCSspe2spe": [0.4, 0.2],
            "OCNany2any": [1.0, 1.0],
            "OCNany2spe": [0.5, 0.2],
            "OCNspe2any": [0.5, 0.2],
            "OCNspe2spe": [0.1, 0.3],
        }
    )
    out = substitution.add_dif_column(cb.copy(), "tmp", "OCSany2any", "OCSany2spe", tol=1e-6)
    np.testing.assert_allclose(out["tmp"].to_numpy(), [0.6, np.nan], equal_nan=True)

    out2 = substitution.add_dif_stats(cb.copy(), tol=1e-6, prefix="OC")
    assert "OCSany2dif" in out2.columns
    assert "OCNdif2spe" in out2.columns

    out3 = substitution.add_dif_stats(cb.copy(), tol=1e-6, prefix="OC", output_stats=["dif2dif"])
    assert "OCSany2dif" in out3.columns
    assert "OCSspe2dif" in out3.columns
    assert "OCSdif2dif" in out3.columns
    assert "OCNany2dif" in out3.columns
    assert "OCNspe2dif" in out3.columns
    assert "OCNdif2dif" in out3.columns


def test_get_substitution_tensor_syn_matches_manual_groupwise_products():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    # codon state order: [AAA, AAG, TTT, TTC]
    # synonymous groups: K=[AAA,AAG], F=[TTT,TTC]
    state = np.zeros((3, 1, 4), dtype=float)
    state[labels["R"], 0, :] = [0.6, 0.4, 0.0, 0.0]
    state[labels["A"], 0, :] = [0.1, 0.9, 0.0, 0.0]
    state[labels["B"], 0, :] = [0.8, 0.2, 0.0, 0.0]
    g = {
        "tree": tr,
        "ml_anc": "yes",
        "float_tol": 1e-12,
        "amino_acid_orders": ["K", "F"],
        "synonymous_indices": {"K": [0, 1], "F": [2, 3]},
        "max_synonymous_size": 2,
    }
    mmap_file = Path("tmp.csubst.sub_tensor.toy_syn.mmap")
    try:
        out = substitution.get_substitution_tensor(state_tensor=state, mode="syn", g=g, mmap_attr="toy_syn")
        assert isinstance(out, substitution_sparse.SparseSubstitutionTensor)
        out_dense = out.to_dense()
        # Branch A, group K: diag masked outer product of parent [0.6,0.4] and child [0.1,0.9].
        np.testing.assert_allclose(out_dense[labels["A"], 0, 0, :, :], [[0.0, 0.54], [0.04, 0.0]], atol=1e-12)
        # Group F has no support in this toy example.
        np.testing.assert_allclose(out_dense[labels["A"], 0, 1, :, :], [[0.0, 0.0], [0.0, 0.0]], atol=1e-12)
        # Branch B, group K:
        np.testing.assert_allclose(out_dense[labels["B"], 0, 0, :, :], [[0.0, 0.12], [0.32, 0.0]], atol=1e-12)
    finally:
        if mmap_file.exists():
            mmap_file.unlink()


def test_get_substitution_tensor_requires_g():
    with pytest.raises(ValueError, match="g is required"):
        substitution.get_substitution_tensor(
            state_tensor=np.zeros((1, 1, 1), dtype=float),
            mode="asis",
            g=None,
        )


def test_get_selected_branch_set_accepts_scalar_state_loaded_branch_ids():
    out = substitution._get_selected_branch_set({"state_loaded_branch_ids": np.int64(7)})
    assert out == {7}


def test_get_selected_branch_set_rejects_non_integer_state_loaded_branch_ids():
    with pytest.raises(ValueError, match="integer-like"):
        substitution._get_selected_branch_set({"state_loaded_branch_ids": np.array([1.5])})


def test_collect_sub_tensor_branch_pairs_includes_root_child_branches_when_root_state_exists():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    state = np.zeros((len(labels), 1, 1), dtype=float)
    state[labels["R"], 0, 0] = 1.0
    g = {"tree": tr, "float_tol": 1e-12}
    out = substitution._collect_sub_tensor_branch_pairs(
        g=g,
        state_tensor_anc=state,
        selected_branch_set=None,
    )
    assert out == [(labels["A"], labels["R"]), (labels["B"], labels["R"])]
