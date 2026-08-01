import numpy as np
import pytest

from csubst import parser_misc
from csubst import ete
from csubst import tree


def test_annotate_tree_handles_none_root_dist(tmp_path):
    rooted_tree_file = tmp_path / "toy_rooted.nwk"
    iqtree_tree_file = tmp_path / "toy_iqtree.treefile"
    rooted_tree_file.write_text("(A:1,B:1)R;\n", encoding="utf-8")
    iqtree_tree_file.write_text("(A:1,B:1)R;\n", encoding="utf-8")

    g = {
        "iqtree_treefile": str(iqtree_tree_file),
        "rooted_tree": ete.PhyloNode(rooted_tree_file.read_text(encoding="utf-8"), format=1),
    }
    out = parser_misc.annotate_tree(g)
    assert "tree" in out
    assert len(list(out["tree"].traverse())) == 3


def test_annotate_tree_rejects_inconsistent_leaf_sets(tmp_path):
    rooted_tree_file = tmp_path / "toy_rooted.nwk"
    iqtree_tree_file = tmp_path / "toy_iqtree.treefile"
    rooted_tree_file.write_text("(A:1,B:1)R;\n", encoding="utf-8")
    iqtree_tree_file.write_text("(A:1,C:1)R;\n", encoding="utf-8")
    g = {
        "iqtree_treefile": str(iqtree_tree_file),
        "rooted_tree": ete.PhyloNode(rooted_tree_file.read_text(encoding="utf-8"), format=1),
    }
    with pytest.raises(ValueError, match="did not have identical leaves"):
        parser_misc.annotate_tree(g)


def test_resolve_state_loading_enables_selective_mode_with_targeted_cb_only():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    g = {
        "tree": tr,
        "num_node": len(list(tr.traverse())),
        "exhaustive_until": 1,
        "foreground": "dummy.tsv",
        "cb": True,
        "b": False,
        "s": False,
        "bs": False,
        "cs": False,
        "cbs": False,
        "plot_state_aa": False,
        "plot_state_codon": False,
        "fg_clade_permutation": 0,
        "target_ids": {"trait1": np.array([labels["N1"], labels["C"]], dtype=np.int64)},
    }
    out = parser_misc.resolve_state_loading(g)
    assert out["is_state_selective_loading"] is True
    np.testing.assert_array_equal(
        out["state_loaded_branch_ids"],
        np.array(sorted([labels["R"], labels["N1"], labels["C"]]), dtype=np.int64),
    )


def test_get_required_state_branch_ids_accepts_scalar_target_id():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    g = {
        "tree": tr,
        "target_ids": {"trait1": np.int64(labels["N1"])},
    }
    out = parser_misc._get_required_state_branch_ids(g)
    np.testing.assert_array_equal(out, np.array(sorted([labels["R"], labels["N1"]]), dtype=np.int64))


def test_get_required_state_branch_ids_ignores_none_target_ids():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    g = {
        "tree": tr,
        "target_ids": {"trait1": None, "trait2": np.array([labels["N1"]], dtype=np.int64)},
    }
    out = parser_misc._get_required_state_branch_ids(g)
    np.testing.assert_array_equal(out, np.array(sorted([labels["R"], labels["N1"]]), dtype=np.int64))


def test_get_required_state_branch_ids_climbs_past_unnamed_internal_parent():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,(B:1,C:1):1)X:1,D:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse() if n.name}
    target_node = next(n for n in tr.traverse() if n.name == "B")
    unnamed_parent = target_node.up
    assert unnamed_parent is not None
    assert unnamed_parent.name in ("", None)
    g = {
        "tree": tr,
        "target_ids": {"trait1": np.array([labels["B"]], dtype=np.int64)},
    }
    out = parser_misc._get_required_state_branch_ids(g)
    np.testing.assert_array_equal(
        out,
        np.array(
            sorted([labels["R"], labels["X"], labels["B"], int(ete.get_prop(unnamed_parent, "numerical_label"))]),
            dtype=np.int64,
        ),
    )


def test_get_required_state_branch_ids_rejects_non_integer_target_ids():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    g = {
        "tree": tr,
        "target_ids": {"trait1": np.array(["x"])},
    }
    with pytest.raises(ValueError, match="integer-like"):
        parser_misc._get_required_state_branch_ids(g)


def test_get_required_state_branch_ids_rejects_non_integer_float_target_ids():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    g = {
        "tree": tr,
        "target_ids": {"trait1": np.array([1.5])},
    }
    with pytest.raises(ValueError, match="integer-like"):
        parser_misc._get_required_state_branch_ids(g)


def test_resolve_state_loading_disables_selective_mode_when_full_tree_outputs_requested():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    g = {
        "tree": tr,
        "num_node": len(list(tr.traverse())),
        "exhaustive_until": 1,
        "foreground": "dummy.tsv",
        "cb": True,
        "b": True,
        "s": False,
        "bs": False,
        "cs": False,
        "cbs": False,
        "plot_state_aa": False,
        "plot_state_codon": False,
        "fg_clade_permutation": 0,
        "target_ids": {"trait1": np.array([1], dtype=np.int64)},
    }
    out = parser_misc.resolve_state_loading(g)
    assert out["is_state_selective_loading"] is False
    assert out["state_loaded_branch_ids"] is None
