
import numpy as np
import pytest

from csubst import parser_iqtree
from csubst import tree
from csubst import ete


def test_mask_missing_sites_nonbinary_internal_uses_all_child_groups():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1,C:1)N1:1,D:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse() if n.name}
    state = np.zeros((len(list(tr.traverse())), 1, 1), dtype=float)
    # Only one child clade (C) and one sister clade (D) have data at this site.
    state[labels["C"], 0, 0] = 1.0
    state[labels["D"], 0, 0] = 1.0
    state[labels["N1"], 0, 0] = 1.0
    out = parser_iqtree.mask_missing_sites(state_tensor=state, tree=tr)
    assert out[labels["N1"], 0, 0] == pytest.approx(1.0)


def test_run_iqtree_ancestral_nonzero_exit_raises_clear_error_and_cleans_tmp_tree(tmp_path, monkeypatch):
    alignment_file = tmp_path / "toy.fa"
    alignment_file.write_text(">A\nAAA\n>B\nAAA\n", encoding="utf-8")
    rooted_tree = ete.PhyloNode("(A:1,B:1)R;", format=1)
    g = {
        "rooted_tree": rooted_tree,
        "alignment_file": str(alignment_file),
        "iqtree_exe": "iqtree2",
        "iqtree_model": "GY",
        "genetic_code": 1,
        "threads": 1,
    }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(parser_iqtree.tree, "is_consistent_tree_and_aln", lambda g: True)

    def fake_write_tree(tree_obj, outfile, add_numerical_label=False):
        (tmp_path / outfile).write_text("(A:1,B:1)R;\n", encoding="utf-8")

    monkeypatch.setattr(parser_iqtree.tree, "write_tree", fake_write_tree)
    monkeypatch.setattr(parser_iqtree.runtime, "run_subprocess_tee", lambda command: 2)

    with pytest.raises(AssertionError, match="exit code 2"):
        parser_iqtree.run_iqtree_ancestral(g)
    assert not (tmp_path / "tmp.csubst.nwk").exists()


def test_run_iqtree_ancestral_rejects_inconsistent_tree_without_force(tmp_path, monkeypatch):
    alignment_file = tmp_path / "toy.fa"
    alignment_file.write_text(">A\nAAA\n>B\nAAA\n", encoding="utf-8")
    rooted_tree = ete.PhyloNode("(A:1,B:1)R;", format=1)
    g = {
        "rooted_tree": rooted_tree,
        "alignment_file": str(alignment_file),
        "iqtree_exe": "iqtree2",
        "iqtree_model": "GY",
        "genetic_code": 1,
        "threads": 1,
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(parser_iqtree.tree, "is_consistent_tree_and_aln", lambda g: False)

    def fake_write_tree(tree_obj, outfile, add_numerical_label=False):
        (tmp_path / outfile).write_text("(A:1,B:1)R;\n", encoding="utf-8")

    monkeypatch.setattr(parser_iqtree.tree, "write_tree", fake_write_tree)
    with pytest.raises(ValueError, match="not consistent"):
        parser_iqtree.run_iqtree_ancestral(g, force_notree_run=False)
    assert not (tmp_path / "tmp.csubst.nwk").exists()
