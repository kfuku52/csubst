import numpy as np
import pandas as pd
import pytest

from csubst import parser_misc
from csubst import sequence
from csubst import ete
from csubst import tree


def test_map_internal_site_indices_defaults_to_identity_without_mapping():
    g = {}
    out = parser_misc.map_internal_site_indices(site_indices=np.array([0, 2, 4], dtype=np.int64), g=g)
    np.testing.assert_array_equal(out, np.array([0, 2, 4], dtype=np.int64))


def test_map_internal_site_indices_handles_invalid_sites_when_allowed():
    g = {"site_index_alignment": np.array([3, 5], dtype=np.int64)}
    out = parser_misc.map_internal_site_indices(
        site_indices=np.array([0, 1, 2, -1], dtype=np.int64),
        g=g,
        missing_value=-1,
        allow_invalid=True,
    )
    np.testing.assert_array_equal(out, np.array([3, 5, -1, -1], dtype=np.int64))


def test_expand_site_axis_table_to_alignment_without_groups():
    g = {
        "num_input_site": 5,
        "site_index_alignment": np.array([1, 3], dtype=np.int64),
    }
    df = pd.DataFrame(
        {
            "site": np.array([1, 3], dtype=np.int64),
            "N_sub": np.array([0.25, 0.75], dtype=float),
        }
    )
    out = parser_misc.expand_site_axis_table_to_alignment(
        df=df,
        g=g,
        site_col="site",
        group_cols=[],
        site_is_one_based=False,
    )
    assert out.shape[0] == 5
    assert out["site"].tolist() == [0, 1, 2, 3, 4]
    assert out["is_site_retained"].tolist() == ["N", "Y", "N", "Y", "N"]
    assert np.isnan(out.loc[out["site"] == 0, "N_sub"]).all()
    assert out.loc[out["site"] == 1, "N_sub"].iloc[0] == pytest.approx(0.25)
    assert out.loc[out["site"] == 3, "N_sub"].iloc[0] == pytest.approx(0.75)


def test_expand_site_axis_table_to_alignment_with_groups_and_one_based_sites():
    g = {
        "num_input_site": 4,
        "site_index_alignment": np.array([0, 2], dtype=np.int64),
    }
    df = pd.DataFrame(
        {
            "branch_id": np.array([10, 10, 11, 11], dtype=np.int64),
            "codon_site_alignment": np.array([1, 3, 1, 3], dtype=np.int64),
            "N_sub": np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
        }
    )
    out = parser_misc.expand_site_axis_table_to_alignment(
        df=df,
        g=g,
        site_col="codon_site_alignment",
        group_cols=["branch_id"],
        site_is_one_based=True,
    )
    assert out.shape[0] == 8
    for bid in [10, 11]:
        sub = out.loc[out["branch_id"] == bid, :]
        assert sub["codon_site_alignment"].tolist() == [1, 2, 3, 4]
        assert sub["is_site_retained"].tolist() == ["Y", "N", "Y", "N"]
    assert np.isnan(out.loc[(out["branch_id"] == 10) & (out["codon_site_alignment"] == 2), "N_sub"]).all()
    assert out.loc[(out["branch_id"] == 11) & (out["codon_site_alignment"] == 3), "N_sub"].iloc[0] == pytest.approx(4.0)


def test_drop_invariant_tip_sites_drops_and_writes_site_map(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    seq_by_leaf = {
        "A": "AAAAAAAAG",
        "B": "AAAAAAAAG",
        "C": "AAAAAAAAA",
    }
    for leaf in ete.iter_leaves(tr):
        ete.set_prop(leaf, "sequence", seq_by_leaf[leaf.name])
    num_node = len(list(tr.traverse()))
    state_cdn = np.zeros((num_node, 3, 2), dtype=float)
    state_pep = np.zeros((num_node, 3, 2), dtype=float)
    state_nsy = np.zeros((num_node, 3, 2), dtype=float)
    state_nuc = np.zeros((num_node, 9, 4), dtype=float)
    for node_id in range(num_node):
        for site_id in range(3):
            state_cdn[node_id, site_id, :] = [node_id + 1.0, site_id + 1.0]
            state_pep[node_id, site_id, :] = [site_id + 1.0, node_id + 1.0]
            state_nsy[node_id, site_id, :] = [1.0, 0.0]
        state_nuc[node_id, :, :] = 1.0
    g = {
        "tree": tr,
        "num_input_site": 3,
        "write_site_index_map": True,
        "codon_orders": np.array(["AAA", "AAG"], dtype=object),
        "state_nuc": state_nuc,
        "state_cdn": state_cdn,
        "state_pep": state_pep,
        "state_nsy": state_nsy,
        "iqtree_rate_values": np.array([0.1, 0.2, 0.3], dtype=float),
    }
    out = parser_misc.drop_invariant_tip_sites(g)
    np.testing.assert_array_equal(out["site_index_alignment"], np.array([2], dtype=np.int64))
    assert out["num_dropped_tip_invariant_sites"] == 2
    np.testing.assert_array_equal(out["dropped_tip_invariant_site_alignment"], np.array([0, 1], dtype=np.int64))
    assert out["state_cdn"].shape[1] == 1
    assert out["state_pep"].shape[1] == 1
    assert out["state_nsy"].shape[1] == 1
    assert out["state_nuc"].shape[1] == 3
    np.testing.assert_allclose(out["iqtree_rate_values"], np.array([0.3], dtype=float), atol=1e-12)
    map_path = tmp_path / "csubst_site_index_map.tsv"
    assert map_path.exists()
    site_map = pd.read_csv(map_path, sep="\t")
    assert site_map["codon_site_alignment"].tolist() == [0, 1, 2]
    assert site_map["site"].tolist() == [-1, -1, 0]
    assert site_map["is_retained"].tolist() == ["N", "N", "Y"]


def test_drop_invariant_tip_sites_skips_site_map_when_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    for leaf in ete.iter_leaves(tr):
        ete.set_prop(leaf, "sequence", "AAAAAAAAA")
    num_node = len(list(tr.traverse()))
    state_cdn = np.ones((num_node, 3, 2), dtype=float)
    state_pep = np.ones((num_node, 3, 2), dtype=float)
    state_nsy = np.ones((num_node, 3, 2), dtype=float)
    state_nuc = np.ones((num_node, 9, 4), dtype=float)
    g = {
        "tree": tr,
        "num_input_site": 3,
        "codon_orders": np.array(["AAA", "AAG"], dtype=object),
        "state_nuc": state_nuc,
        "state_cdn": state_cdn,
        "state_pep": state_pep,
        "state_nsy": state_nsy,
        "_precomputed_tip_invariant_site_mask": np.array([True, False, True], dtype=bool),
    }
    parser_misc.drop_invariant_tip_sites(g)
    assert not (tmp_path / "csubst_site_index_map.tsv").exists()


def test_drop_invariant_tip_sites_drops_single_nonmissing_tip_site(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    seq_by_leaf = {
        # site0: AAA/AAA/AAA -> tip-invariant
        # site1: AAA/NNN/NNN -> only one unambiguous tip codon, should now be dropped
        # site2: AAG/AAA/AAA -> variable among unambiguous tips, should be retained
        "A": "AAAAAAAAG",
        "B": "AAANNNAAA",
        "C": "AAANNNAAA",
    }
    for leaf in ete.iter_leaves(tr):
        ete.set_prop(leaf, "sequence", seq_by_leaf[leaf.name])
    num_node = len(list(tr.traverse()))
    state_cdn = np.zeros((num_node, 3, 2), dtype=float)
    state_pep = np.zeros((num_node, 3, 2), dtype=float)
    state_nsy = np.zeros((num_node, 3, 2), dtype=float)
    state_nuc = np.zeros((num_node, 9, 4), dtype=float)
    for node_id in range(num_node):
        for site_id in range(3):
            state_cdn[node_id, site_id, :] = [node_id + 1.0, site_id + 1.0]
            state_pep[node_id, site_id, :] = [site_id + 1.0, node_id + 1.0]
            state_nsy[node_id, site_id, :] = [1.0, 0.0]
        state_nuc[node_id, :, :] = 1.0
    g = {
        "tree": tr,
        "num_input_site": 3,
        "write_site_index_map": True,
        "codon_orders": np.array(["AAA", "AAG"], dtype=object),
        "state_nuc": state_nuc,
        "state_cdn": state_cdn,
        "state_pep": state_pep,
        "state_nsy": state_nsy,
        "iqtree_rate_values": np.array([0.1, 0.2, 0.3], dtype=float),
    }
    out = parser_misc.drop_invariant_tip_sites(g)
    np.testing.assert_array_equal(out["site_index_alignment"], np.array([2], dtype=np.int64))
    assert out["num_dropped_tip_invariant_sites"] == 2
    np.testing.assert_array_equal(out["dropped_tip_invariant_site_alignment"], np.array([0, 1], dtype=np.int64))
    site_map = pd.read_csv(tmp_path / "csubst_site_index_map.tsv", sep="\t")
    assert site_map["site"].tolist() == [-1, -1, 0]
    assert site_map["is_retained"].tolist() == ["N", "N", "Y"]


def test_drop_invariant_tip_sites_uses_precomputed_mask_when_available(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    for leaf in ete.iter_leaves(tr):
        ete.set_prop(leaf, "sequence", "AAAAAAAAA")
    num_node = len(list(tr.traverse()))
    state_cdn = np.ones((num_node, 3, 2), dtype=float)
    state_pep = np.ones((num_node, 3, 2), dtype=float)
    state_nsy = np.ones((num_node, 3, 2), dtype=float)
    state_nuc = np.ones((num_node, 9, 4), dtype=float)
    g = {
        "tree": tr,
        "num_input_site": 3,
        "write_site_index_map": True,
        "codon_orders": np.array(["AAA", "AAG"], dtype=object),
        "state_nuc": state_nuc,
        "state_cdn": state_cdn,
        "state_pep": state_pep,
        "state_nsy": state_nsy,
        "_precomputed_tip_invariant_site_mask": np.array([True, False, True], dtype=bool),
    }
    monkeypatch.setattr(
        parser_misc,
        "_get_tip_invariant_site_mask",
        lambda g, site_index_alignment: (_ for _ in ()).throw(AssertionError("unexpected fallback mask call")),
    )
    out = parser_misc.drop_invariant_tip_sites(g)
    np.testing.assert_array_equal(out["site_index_alignment"], np.array([1], dtype=np.int64))
    assert out["state_cdn"].shape[1] == 1
    site_map = pd.read_csv(tmp_path / "csubst_site_index_map.tsv", sep="\t")
    assert site_map["site"].tolist() == [-1, 0, -1]
    assert site_map["is_retained"].tolist() == ["N", "Y", "N"]


def test_drop_invariant_tip_sites_zero_sub_mass_mode_drops_only_zero_mass_sites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    node_by_name = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tr.traverse()}
    root_id = node_by_name["R"]
    a_id = node_by_name["A"]
    b_id = node_by_name["B"]
    num_node = len(list(tr.traverse()))
    state_cdn = np.zeros((num_node, 3, 3), dtype=float)
    # site 0: AAA -> AAA in all branches (zero N/S substitution mass)
    state_cdn[root_id, 0, 0] = 1.0
    state_cdn[a_id, 0, 0] = 1.0
    state_cdn[b_id, 0, 0] = 1.0
    # site 1: AAA -> AAG in branch B (synonymous substitution mass > 0)
    state_cdn[root_id, 1, 0] = 1.0
    state_cdn[a_id, 1, 0] = 1.0
    state_cdn[b_id, 1, 1] = 1.0
    # site 2: AAA -> AAC in branch B (nonsynonymous substitution mass > 0)
    state_cdn[root_id, 2, 0] = 1.0
    state_cdn[a_id, 2, 0] = 1.0
    state_cdn[b_id, 2, 2] = 1.0
    state_nuc = np.ones((num_node, 9, 4), dtype=float)
    g = {
        "tree": tr,
        "num_input_site": 3,
        "write_site_index_map": True,
        "float_tol": 1e-12,
        "drop_invariant_tip_sites_mode": "zero_sub_mass",
        "codon_orders": np.array(["AAA", "AAG", "AAC"], dtype=object),
        "amino_acid_orders": np.array(["K", "N"], dtype=object),
        "synonymous_indices": {"K": [0, 1], "N": [2]},
        "nonsyn_state_orders": np.array(["K", "N"], dtype=object),
        "nonsynonymous_indices": {"K": [0, 1], "N": [2]},
        "state_nuc": state_nuc,
        "state_cdn": state_cdn,
        "iqtree_rate_values": np.array([0.1, 0.2, 0.3], dtype=float),
        "iqtree_categorized_rate_values": np.array([1.0, 2.0, 3.0], dtype=float),
    }
    g["state_pep"] = sequence.cdn2pep_state(state_cdn=state_cdn, g=g)
    g["state_nsy"] = sequence.cdn2nsy_state(state_cdn=state_cdn, g=g)
    out = parser_misc.drop_invariant_tip_sites(g)
    np.testing.assert_array_equal(out["site_index_alignment"], np.array([1, 2], dtype=np.int64))
    assert out["num_dropped_tip_invariant_sites"] == 1
    np.testing.assert_array_equal(out["dropped_tip_invariant_site_alignment"], np.array([0], dtype=np.int64))
    assert out["state_cdn"].shape[1] == 2
    assert out["state_pep"].shape[1] == 2
    assert out["state_nsy"].shape[1] == 2
    assert out["state_nuc"].shape[1] == 6
    np.testing.assert_allclose(out["iqtree_rate_values"], np.array([0.2, 0.3], dtype=float), atol=1e-12)
    np.testing.assert_allclose(
        out["iqtree_categorized_rate_values"], np.array([2.0, 3.0], dtype=float), atol=1e-12
    )
    site_map = pd.read_csv(tmp_path / "csubst_site_index_map.tsv", sep="\t")
    assert site_map["site"].tolist() == [-1, 0, 1]


def test_zero_sub_mass_ignores_state_less_synthetic_parent_after_reroot():
    iqtree_like = ete.PhyloNode("(A:1,B:1,(C:1,D:1)Y:1)R;", format=1)
    rooted = ete.PhyloNode("(A:1,(B:1,(C:1,D:1)Y:1):1)RR;", format=1)
    tr = tree.add_numerical_node_labels(tree.transfer_root(tree_to=iqtree_like, tree_from=rooted))
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tr.traverse() if node.name}
    num_node = max(int(ete.get_prop(node, "numerical_label")) for node in tr.traverse()) + 1
    state_cdn = np.zeros((num_node, 1, 3), dtype=float)
    state_cdn[labels["R"], 0, 0] = 1.0
    state_cdn[labels["Y"], 0, 0] = 1.0
    state_cdn[labels["A"], 0, 0] = 1.0
    state_cdn[labels["B"], 0, 1] = 1.0
    state_cdn[labels["C"], 0, 0] = 1.0
    state_cdn[labels["D"], 0, 0] = 1.0
    state_nsy = np.zeros((num_node, 1, 2), dtype=float)
    state_nsy[:, 0, 0] = state_cdn[:, 0, 0] + state_cdn[:, 0, 1]
    state_nsy[:, 0, 1] = state_cdn[:, 0, 2]
    g = {
        "tree": tr,
        "state_cdn": state_cdn,
        "state_nsy": state_nsy,
        "float_tol": 1e-12,
        "amino_acid_orders": np.array(["K", "N"], dtype=object),
        "synonymous_indices": {"K": [0, 1], "N": [2]},
    }
    out = parser_misc._get_zero_substitution_mass_site_mask(g)
    np.testing.assert_array_equal(out, np.array([False], dtype=bool))


def test_drop_invariant_tip_sites_works_without_state_tensors_for_tip_invariant_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    seq_by_leaf = {
        "A": "AAAAAAAAG",
        "B": "AAAAAAAAG",
        "C": "AAAAAAAAA",
    }
    for leaf in ete.iter_leaves(tr):
        ete.set_prop(leaf, "sequence", seq_by_leaf[leaf.name])
    g = {
        "tree": tr,
        "num_input_site": 3,
        "write_site_index_map": True,
        "codon_orders": np.array(["AAA", "AAG"], dtype=object),
        "iqtree_rate_values": np.array([0.1, 0.2, 0.3], dtype=float),
    }
    out = parser_misc.drop_invariant_tip_sites(g)
    np.testing.assert_array_equal(out["site_index_alignment"], np.array([2], dtype=np.int64))
    assert "state_cdn" not in out
    np.testing.assert_allclose(out["iqtree_rate_values"], np.array([0.3], dtype=float), atol=1e-12)
    site_map = pd.read_csv(tmp_path / "csubst_site_index_map.tsv", sep="\t")
    assert site_map["site"].tolist() == [-1, -1, 0]
    assert site_map["is_retained"].tolist() == ["N", "N", "Y"]
