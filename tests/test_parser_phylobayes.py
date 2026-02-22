import numpy as np
import pytest

from csubst import parser_phylobayes
from csubst import tree
from csubst import ete
from csubst import genetic_code


def test_get_state_tensor_ignores_unknown_selected_branch_ids_in_ml_mode(monkeypatch, tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    g = {
        "tree": tr,
        "phylobayes_dir": str(tmp_path),
        "num_input_state": 4,
        "num_input_site": 2,
        "float_type": np.float64,
        "ml_anc": True,
    }
    monkeypatch.setattr(parser_phylobayes.os, "listdir", lambda _path: [])
    out = parser_phylobayes.get_state_tensor(g=g, selected_branch_ids=np.array([9999], dtype=np.int64))
    assert out.shape == (len(list(tr.traverse())), 2, 4)
    assert out.sum() == 0


def test_get_state_tensor_rejects_non_integer_selected_branch_ids(monkeypatch, tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    g = {
        "tree": tr,
        "phylobayes_dir": str(tmp_path),
        "num_input_state": 4,
        "num_input_site": 2,
        "float_type": np.float64,
        "ml_anc": False,
    }
    monkeypatch.setattr(parser_phylobayes.os, "listdir", lambda _path: [])
    with pytest.raises(ValueError, match="integer-like"):
        parser_phylobayes.get_state_tensor(g=g, selected_branch_ids=np.array([1.5]))


def test_get_input_information_raises_when_sample_label_file_is_missing(tmp_path):
    g = {
        "phylobayes_dir": str(tmp_path),
        "codon_table": genetic_code.get_codon_table(1),
    }
    with pytest.raises(ValueError, match=r"_sample\.labels"):
        parser_phylobayes.get_input_information(g)


def test_get_input_information_raises_when_state_file_is_missing(tmp_path):
    sample_label_file = tmp_path / "run_sample.labels"
    sample_label_file.write_text("(A:1,B:1)R;", encoding="utf-8")
    g = {
        "phylobayes_dir": str(tmp_path),
        "codon_table": genetic_code.get_codon_table(1),
    }
    with pytest.raises(ValueError, match=r"ancstatepostprob"):
        parser_phylobayes.get_input_information(g)


def test_get_input_information_accepts_directory_without_trailing_separator(tmp_path):
    sample_label_file = tmp_path / "run_sample.labels"
    sample_label_file.write_text("(A:1,B:1)R;", encoding="utf-8")
    state_file = tmp_path / "run.ancstatepostprob"
    state_file.write_text(
        "site\tpos\tA\tC\tG\tT\n"
        "0\t0\t0.1\t0.2\t0.3\t0.4\n",
        encoding="utf-8",
    )
    g = {
        "phylobayes_dir": str(tmp_path),
        "codon_table": genetic_code.get_codon_table(1),
    }
    out = parser_phylobayes.get_input_information(g)
    assert out["num_input_site"] == 1
    assert out["num_input_state"] == 4
    assert out["input_state"] == ["A", "C", "G", "T"]


def test_get_input_information_rejects_unsupported_state_count(tmp_path):
    sample_label_file = tmp_path / "run_sample.labels"
    sample_label_file.write_text("(A:1,B:1)R;", encoding="utf-8")
    state_file = tmp_path / "run.ancstatepostprob"
    state_file.write_text(
        "site\tpos\tS1\tS2\tS3\n"
        "0\t0\t0.2\t0.3\t0.5\n",
        encoding="utf-8",
    )
    g = {
        "phylobayes_dir": str(tmp_path),
        "codon_table": genetic_code.get_codon_table(1),
    }
    with pytest.raises(ValueError, match=r"Unsupported number of input states"):
        parser_phylobayes.get_input_information(g)


def test_get_state_tensor_accepts_directory_without_trailing_separator(tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tr.traverse() if node.name}
    leaf_a_id = labels["A"]
    state_file = tmp_path / "A_A.ancstatepostprob"
    state_file.write_text(
        "site\tpos\tA\tC\tG\tT\n"
        "0\t0\t0.1\t0.2\t0.3\t0.4\n",
        encoding="utf-8",
    )
    g = {
        "tree": tr,
        "phylobayes_dir": str(tmp_path),
        "num_input_state": 4,
        "num_input_site": 1,
        "float_type": np.float64,
        "ml_anc": False,
    }
    out = parser_phylobayes.get_state_tensor(g=g, selected_branch_ids=np.array([leaf_a_id], dtype=np.int64))
    np.testing.assert_allclose(out[leaf_a_id, 0, :], np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64), atol=1e-12)


def test_get_state_tensor_ml_anc_keeps_missing_nodes_zero_without_selection(tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tr.traverse() if node.name}
    leaf_a_id = labels["A"]
    leaf_b_id = labels["B"]
    root_id = labels["R"]
    state_file = tmp_path / "A_A.ancstatepostprob"
    state_file.write_text(
        "site\tpos\tA\tC\tG\tT\n"
        "0\t0\t0.1\t0.2\t0.3\t0.4\n"
        "1\t1\t0.5\t0.2\t0.2\t0.1\n",
        encoding="utf-8",
    )
    g = {
        "tree": tr,
        "phylobayes_dir": str(tmp_path),
        "num_input_state": 4,
        "num_input_site": 2,
        "float_type": np.float64,
        "ml_anc": True,
    }
    out = parser_phylobayes.get_state_tensor(g=g, selected_branch_ids=None)
    assert out.dtype == bool
    assert out[leaf_a_id].sum() == 2
    assert out[leaf_b_id].sum() == 0
    assert out[root_id].sum() == 0


def test_get_state_tensor_handles_noncontiguous_branch_ids(tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    reassigned = {"A": 11, "B": 29, "R": 5}
    for node in tr.traverse():
        ete.set_prop(node, "numerical_label", reassigned[node.name])
    leaf_a_id = reassigned["A"]
    state_file = tmp_path / "A_A.ancstatepostprob"
    state_file.write_text(
        "site\tpos\tA\tC\tG\tT\n"
        "0\t0\t0.1\t0.2\t0.3\t0.4\n",
        encoding="utf-8",
    )
    g = {
        "tree": tr,
        "phylobayes_dir": str(tmp_path),
        "num_input_state": 4,
        "num_input_site": 1,
        "float_type": np.float64,
        "ml_anc": False,
    }
    out = parser_phylobayes.get_state_tensor(g=g, selected_branch_ids=np.array([leaf_a_id], dtype=np.int64))
    assert out.shape == (30, 1, 4)
    np.testing.assert_allclose(out[leaf_a_id, 0, :], np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64), atol=1e-12)


def test_get_state_tensor_rejects_negative_numerical_label(tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    ete.set_prop(next(n for n in tr.traverse() if n.name == "A"), "numerical_label", -1)
    g = {
        "tree": tr,
        "phylobayes_dir": str(tmp_path),
        "num_input_state": 4,
        "num_input_site": 1,
        "float_type": np.float64,
        "ml_anc": False,
    }
    with pytest.raises(ValueError, match="non-negative"):
        parser_phylobayes.get_state_tensor(g=g, selected_branch_ids=np.array([1], dtype=np.int64))
