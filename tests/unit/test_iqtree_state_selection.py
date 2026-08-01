
import numpy as np
import pytest

from csubst import parser_iqtree
from csubst import tree
from csubst import ete


def _make_state_tensor_g(tmp_path, alignment_text):
    alignment_file = tmp_path / "toy.fa"
    state_file = tmp_path / "toy.state.tsv"
    alignment_file.write_text(alignment_text, encoding="utf-8")
    state_file.write_text(
        "Node\tSite\tState\tp_AAA\tp_AAC\tp_AAG\n"
        "R\t1\tAAA\t1.0\t0.0\t0.0\n"
        "R\t2\tAAC\t0.0\t1.0\t0.0\n",
        encoding="utf-8",
    )
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    return {
        "tree": tr,
        "alignment_file": str(alignment_file),
        "path_iqtree_state": str(state_file),
        "num_input_site": 2,
        "num_input_state": 3,
        "input_data_type": "cdn",
        "codon_orders": np.array(["AAA", "AAC", "AAG"]),
        "float_type": np.float64,
        "ml_anc": False,
    }


def test_get_state_tensor_selected_branch_ids_preserve_global_branch_index(tmp_path):
    g = _make_state_tensor_g(
        tmp_path=tmp_path,
        alignment_text=">A\nAAAAAC\n>B\nAAGAAG\n",
    )
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in g["tree"].traverse()}
    full = parser_iqtree.get_state_tensor(g)
    selected = parser_iqtree.get_state_tensor(
        g=g,
        selected_branch_ids=np.array([labels["A"]], dtype=np.int64),
    )
    assert selected.shape == full.shape
    np.testing.assert_allclose(selected[labels["A"], :, :], full[labels["A"], :, :], atol=1e-12)
    assert selected[labels["B"], :, :].sum() == 0


def test_get_state_tensor_selected_branch_ids_match_internal_masking_parity(tmp_path):
    alignment_file = tmp_path / "toy_internal.fa"
    state_file = tmp_path / "toy_internal.state.tsv"
    alignment_file.write_text(
        ">A\nAAA---\n"
        ">B\nAAG---\n"
        ">C\nAAAAAA\n",
        encoding="utf-8",
    )
    state_file.write_text(
        "Node\tSite\tState\tp_AAA\tp_AAC\tp_AAG\n"
        "N1\t1\tAAA\t1.0\t0.0\t0.0\n"
        "N1\t2\tAAC\t0.0\t1.0\t0.0\n"
        "R\t1\tAAA\t1.0\t0.0\t0.0\n"
        "R\t2\tAAC\t0.0\t1.0\t0.0\n",
        encoding="utf-8",
    )
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    g = {
        "tree": tr,
        "alignment_file": str(alignment_file),
        "path_iqtree_state": str(state_file),
        "num_input_site": 2,
        "num_input_state": 3,
        "input_data_type": "cdn",
        "codon_orders": np.array(["AAA", "AAC", "AAG"]),
        "float_type": np.float64,
        "ml_anc": False,
    }
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in g["tree"].traverse()}
    full = parser_iqtree.get_state_tensor(g)
    selected = parser_iqtree.get_state_tensor(
        g=g,
        selected_branch_ids=np.array([labels["N1"]], dtype=np.int64),
    )
    assert selected.shape == full.shape
    np.testing.assert_allclose(selected[labels["N1"], :, :], full[labels["N1"], :, :], atol=1e-12)
    assert selected[labels["C"], :, :].sum() == 0


def test_get_state_tensor_selected_internal_rejects_required_leaf_length_mismatch(tmp_path):
    alignment_file = tmp_path / "toy_internal_badlen.fa"
    state_file = tmp_path / "toy_internal_badlen.state.tsv"
    alignment_file.write_text(
        ">A\nAAAAAC\n"
        ">B\nAAA\n"
        ">C\nAAAAAC\n",
        encoding="utf-8",
    )
    state_file.write_text(
        "Node\tSite\tState\tp_AAA\tp_AAC\tp_AAG\n"
        "N1\t1\tAAA\t1.0\t0.0\t0.0\n"
        "N1\t2\tAAC\t0.0\t1.0\t0.0\n"
        "R\t1\tAAA\t1.0\t0.0\t0.0\n"
        "R\t2\tAAC\t0.0\t1.0\t0.0\n",
        encoding="utf-8",
    )
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    g = {
        "tree": tr,
        "alignment_file": str(alignment_file),
        "path_iqtree_state": str(state_file),
        "num_input_site": 2,
        "num_input_state": 3,
        "input_data_type": "cdn",
        "codon_orders": np.array(["AAA", "AAC", "AAG"]),
        "float_type": np.float64,
        "ml_anc": False,
    }
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse() if n.name}
    with pytest.raises(AssertionError, match="Codon site count did not match alignment size"):
        parser_iqtree.get_state_tensor(
            g=g,
            selected_branch_ids=np.array([labels["N1"]], dtype=np.int64),
        )


def test_get_leaf_nonmissing_sites_handles_noncontiguous_branch_ids():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    reassigned = {"A": 11, "B": 29, "C": 41, "N1": 73, "R": 5}
    for node in tr.traverse():
        ete.set_prop(node, "numerical_label", reassigned[node.name])
        if ete.is_leaf(node):
            ete.set_prop(node, "sequence", "AAAAAC")
    g = {"tree": tr, "num_input_site": 2, "codon_orders": np.array(["AAA", "AAC", "AAG"])}
    required_leaf_ids = {11, 29, 41}
    out = parser_iqtree._get_leaf_nonmissing_sites(g=g, required_leaf_ids=required_leaf_ids)
    assert out.shape == (74, 2)
    assert not out[0, :].any()
    for leaf_id in sorted(required_leaf_ids):
        assert out[leaf_id, :].all()


def test_get_state_tensor_selected_internal_handles_noncontiguous_branch_ids(tmp_path):
    alignment_file = tmp_path / "toy_noncontig.fa"
    state_file = tmp_path / "toy_noncontig.state.tsv"
    alignment_file.write_text(
        ">A\nAAAAAC\n"
        ">B\nAAGAAG\n"
        ">C\nAAAAAC\n",
        encoding="utf-8",
    )
    state_file.write_text(
        "Node\tSite\tState\tp_AAA\tp_AAC\tp_AAG\n"
        "N1\t1\tAAA\t1.0\t0.0\t0.0\n"
        "N1\t2\tAAC\t0.0\t1.0\t0.0\n"
        "R\t1\tAAA\t1.0\t0.0\t0.0\n"
        "R\t2\tAAC\t0.0\t1.0\t0.0\n",
        encoding="utf-8",
    )
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    reassigned = {"A": 11, "B": 29, "C": 41, "N1": 73, "R": 5}
    for node in tr.traverse():
        ete.set_prop(node, "numerical_label", reassigned[node.name])
    g = {
        "tree": tr,
        "alignment_file": str(alignment_file),
        "path_iqtree_state": str(state_file),
        "num_input_site": 2,
        "num_input_state": 3,
        "input_data_type": "cdn",
        "codon_orders": np.array(["AAA", "AAC", "AAG"]),
        "float_type": np.float64,
        "ml_anc": False,
    }
    out = parser_iqtree.get_state_tensor(
        g=g,
        selected_branch_ids=np.array([73], dtype=np.int64),
    )
    assert out.shape == (74, 2, 3)
    np.testing.assert_allclose(out[73, 0, :], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(out[73, 1, :], [0.0, 1.0, 0.0], atol=1e-12)


def test_get_state_tensor_maps_internal_rows_by_site_label_not_file_order(tmp_path):
    alignment_file = tmp_path / "toy_siteorder.fa"
    state_file = tmp_path / "toy_siteorder.state.tsv"
    alignment_file.write_text(
        ">A\nAAAAAC\n"
        ">B\nAAAAAC\n"
        ">C\nAAAAAC\n",
        encoding="utf-8",
    )
    state_file.write_text(
        "Node\tSite\tState\tp_AAA\tp_AAC\tp_AAG\n"
        "N1\t2\tAAC\t0.0\t1.0\t0.0\n"
        "N1\t1\tAAG\t0.0\t0.0\t1.0\n"
        "R\t1\tAAA\t1.0\t0.0\t0.0\n"
        "R\t2\tAAC\t0.0\t1.0\t0.0\n",
        encoding="utf-8",
    )
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    g = {
        "tree": tr,
        "alignment_file": str(alignment_file),
        "path_iqtree_state": str(state_file),
        "num_input_site": 2,
        "num_input_state": 3,
        "input_data_type": "cdn",
        "codon_orders": np.array(["AAA", "AAC", "AAG"]),
        "float_type": np.float64,
        "ml_anc": False,
    }
    out = parser_iqtree.get_state_tensor(g)
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse() if n.name}
    np.testing.assert_allclose(out[labels["N1"], 0, :], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(out[labels["N1"], 1, :], [0.0, 1.0, 0.0], atol=1e-12)


def test_get_selected_branch_context_accepts_scalar_selected_branch_id():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse() if n.name}
    selected_set, selected_internal_ids, required_leaf_ids = parser_iqtree._get_selected_branch_context(
        tree=tr,
        selected_branch_ids=np.int64(labels["N1"]),
    )
    root_id = int(ete.get_prop(ete.get_tree_root(tr), "numerical_label"))
    assert labels["N1"] in selected_set
    assert root_id in selected_set
    assert selected_internal_ids == [labels["N1"]]
    expected_leaf_ids = {
        int(ete.get_prop(node, "numerical_label"))
        for node in tr.traverse()
        if ete.is_leaf(node)
    }
    assert required_leaf_ids == expected_leaf_ids


def test_get_selected_branch_context_rejects_non_integer_selected_branch_id():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    with pytest.raises(ValueError, match="integer-like"):
        parser_iqtree._get_selected_branch_context(
            tree=tr,
            selected_branch_ids=np.array([1.5]),
        )


def test_get_state_tensor_selected_branch_ids_ignore_unknown_ids_in_ml_mode(tmp_path):
    g = _make_state_tensor_g(
        tmp_path=tmp_path,
        alignment_text=">A\nAAAAAC\n>B\nAAGAAG\n",
    )
    g["ml_anc"] = True
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in g["tree"].traverse()}
    selected = parser_iqtree.get_state_tensor(
        g=g,
        selected_branch_ids=np.array([labels["A"], 9999], dtype=np.int64),
    )
    assert selected.shape[0] == len(list(g["tree"].traverse()))
    assert selected[labels["A"], :, :].sum() > 0
    assert selected[labels["B"], :, :].sum() == 0


def test_get_state_tensor_selected_internal_keeps_root_rows(tmp_path):
    alignment_file = tmp_path / "toy_selected_root.fa"
    state_file = tmp_path / "toy_selected_root.state.tsv"
    alignment_file.write_text(
        ">A\nAAAAAC\n"
        ">B\nAAAAAC\n"
        ">C\nAAGAAG\n",
        encoding="utf-8",
    )
    state_file.write_text(
        "Node\tSite\tState\tp_AAA\tp_AAC\tp_AAG\n"
        "N1\t1\tAAC\t0.0\t1.0\t0.0\n"
        "N1\t2\tAAG\t0.0\t0.0\t1.0\n"
        "R\t1\tAAG\t0.0\t0.0\t1.0\n"
        "R\t2\tAAA\t1.0\t0.0\t0.0\n",
        encoding="utf-8",
    )
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse() if n.name}
    g = {
        "tree": tr,
        "alignment_file": str(alignment_file),
        "path_iqtree_state": str(state_file),
        "num_input_site": 2,
        "num_input_state": 3,
        "input_data_type": "cdn",
        "codon_orders": np.array(["AAA", "AAC", "AAG"]),
        "float_type": np.float64,
        "ml_anc": False,
    }
    out = parser_iqtree.get_state_tensor(
        g=g,
        selected_branch_ids=np.array([labels["N1"]], dtype=np.int64),
    )
    np.testing.assert_allclose(out[labels["R"], 0, :], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(out[labels["R"], 1, :], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(out[labels["N1"], 0, :], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(out[labels["N1"], 1, :], [0.0, 0.0, 1.0], atol=1e-12)


def test_get_state_tensor_rejects_nucleotide_input(tmp_path):
    alignment_file = tmp_path / "toy_nuc.fa"
    state_file = tmp_path / "toy_nuc.state.tsv"
    alignment_file.write_text(
        ">A\nAC\n"
        ">B\nGT\n",
        encoding="utf-8",
    )
    state_file.write_text(
        "Node\tSite\tState\tp_A\tp_C\tp_G\tp_T\n",
        encoding="utf-8",
    )
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    g = {
        "tree": tr,
        "alignment_file": str(alignment_file),
        "path_iqtree_state": str(state_file),
        "num_input_site": 2,
        "num_input_state": 4,
        "input_data_type": "nuc",
        "input_state": np.array(["A", "C", "G", "T"]),
        "float_type": np.float64,
        "ml_anc": False,
    }
    with pytest.raises(NotImplementedError, match="Non-codon input is obsolete"):
        parser_iqtree.get_state_tensor(g)
