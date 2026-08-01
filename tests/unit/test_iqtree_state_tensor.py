
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


def test_build_unambiguous_codon_lookup_marks_known_codons():
    lookup = parser_iqtree._build_unambiguous_codon_lookup(np.array(["AAA", "AAC", "AAG"], dtype=object))
    assert lookup.shape == (64,)
    assert lookup[parser_iqtree._encode_unambiguous_codon("AAA")] == 0
    assert lookup[parser_iqtree._encode_unambiguous_codon("AAC")] == 1
    assert lookup[parser_iqtree._encode_unambiguous_codon("AAG")] == 2


def test_fill_leaf_state_matrix_codon_handles_ambiguous_fallback():
    g = {"codon_orders": np.array(["AAA", "AAC", "AAG"], dtype=object)}
    lookup = parser_iqtree._build_unambiguous_codon_lookup(g["codon_orders"])
    state_matrix = np.zeros((3, 3), dtype=np.float64)
    parser_iqtree._fill_leaf_state_matrix_codon(seq="AAAAARAAG", state_matrix=state_matrix, g=g, codon_lookup=lookup)
    np.testing.assert_allclose(state_matrix[0, :], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(state_matrix[1, :], [0.5, 0.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(state_matrix[2, :], [0.0, 0.0, 1.0], atol=1e-12)


def test_fill_leaf_state_matrix_codon_cython_matches_python_fallback(monkeypatch):
    if (parser_iqtree.parser_iqtree_cy is None) or (not hasattr(parser_iqtree.parser_iqtree_cy, "fill_leaf_state_matrix_codon_unambiguous")):
        pytest.skip("Cython parser_iqtree fast path is unavailable")
    g = {"codon_orders": np.array(["AAA", "AAC", "AAG"], dtype=object)}
    lookup = parser_iqtree._build_unambiguous_codon_lookup(g["codon_orders"])
    seq = "AAAAACNNN"

    monkeypatch.setattr(parser_iqtree, "_can_use_cython_leaf_codon_fill", lambda *args, **kwargs: False)
    expected = np.zeros((3, 3), dtype=np.float64)
    parser_iqtree._fill_leaf_state_matrix_codon(seq=seq, state_matrix=expected, g=g, codon_lookup=lookup)

    monkeypatch.setattr(parser_iqtree, "_can_use_cython_leaf_codon_fill", lambda *args, **kwargs: True)
    observed = np.zeros((3, 3), dtype=np.float64)
    parser_iqtree._fill_leaf_state_matrix_codon(seq=seq, state_matrix=observed, g=g, codon_lookup=lookup)
    np.testing.assert_allclose(observed, expected, atol=1e-12)


def test_get_state_tensor_reads_leaf_sequences_via_ete_compat(tmp_path):
    g = _make_state_tensor_g(
        tmp_path=tmp_path,
        alignment_text=">A\nAAAAAC\n>B\nAAGAAG\n",
    )
    out = parser_iqtree.get_state_tensor(g)

    labels = {n.name: ete.get_prop(n, "numerical_label") for n in g["tree"].traverse()}
    np.testing.assert_allclose(out[labels["A"], 0, :], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(out[labels["A"], 1, :], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(out[labels["B"], 0, :], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(out[labels["B"], 1, :], [0.0, 0.0, 1.0], atol=1e-12)


def test_get_state_tensor_reads_root_rows_from_state_file(tmp_path):
    g = _make_state_tensor_g(
        tmp_path=tmp_path,
        alignment_text=">A\nAAAAAC\n>B\nAAAAAC\n",
    )
    state_file = tmp_path / "toy.state.tsv"
    state_file.write_text(
        "Node\tSite\tState\tp_AAA\tp_AAC\tp_AAG\n"
        "R\t1\tAAG\t0.0\t0.0\t1.0\n"
        "R\t2\tAAC\t0.0\t1.0\t0.0\n",
        encoding="utf-8",
    )
    out = parser_iqtree.get_state_tensor(g)
    root_id = int(ete.get_prop(ete.get_tree_root(g["tree"]), "numerical_label"))
    np.testing.assert_allclose(out[root_id, 0, :], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(out[root_id, 1, :], [0.0, 1.0, 0.0], atol=1e-12)


def test_get_state_tensor_allows_iqtree_state_without_root_rows(tmp_path):
    alignment_file = tmp_path / "toy_without_root.fa"
    state_file = tmp_path / "toy_without_root.state.tsv"
    alignment_file.write_text(
        ">A\nAAAAAC\n>B\nAAGAAG\n>C\nAAAAAC\n",
        encoding="utf-8",
    )
    state_file.write_text(
        "Node\tSite\tState\tp_AAA\tp_AAC\tp_AAG\n"
        "N1\t1\tAAA\t1.0\t0.0\t0.0\n"
        "N1\t2\tAAC\t0.0\t1.0\t0.0\n",
        encoding="utf-8",
    )
    tr = tree.add_numerical_node_labels(
        ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1)
    )
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

    labels = {
        node.name: int(ete.get_prop(node, "numerical_label"))
        for node in tr.traverse()
    }
    np.testing.assert_allclose(out[labels["N1"], 0, :], [1.0, 0.0, 0.0])
    assert out[labels["R"], :, :].sum() == 0


def test_get_state_tensor_still_requires_nonroot_internal_rows(tmp_path):
    alignment_file = tmp_path / "toy_missing_internal.fa"
    state_file = tmp_path / "toy_missing_internal.state.tsv"
    alignment_file.write_text(
        ">A\nAAAAAC\n>B\nAAGAAG\n>C\nAAAAAC\n",
        encoding="utf-8",
    )
    state_file.write_text(
        "Node\tSite\tState\tp_AAA\tp_AAC\tp_AAG\n"
        "R\t1\tAAA\t1.0\t0.0\t0.0\n"
        "R\t2\tAAC\t0.0\t1.0\t0.0\n",
        encoding="utf-8",
    )
    tr = tree.add_numerical_node_labels(
        ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1)
    )
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

    with pytest.raises(ValueError, match="N1"):
        parser_iqtree.get_state_tensor(g)


def test_get_state_tensor_allows_unnamed_synthetic_internal_node(tmp_path):
    alignment_file = tmp_path / "toy_unnamed_internal.fa"
    state_file = tmp_path / "toy_unnamed_internal.state.tsv"
    alignment_file.write_text(
        ">A\nAAAAAC\n>B\nAAGAAG\n>C\nAAAAAC\n>D\nAAAAAC\n",
        encoding="utf-8",
    )
    state_file.write_text(
        "Node\tSite\tState\tp_AAA\tp_AAC\tp_AAG\n"
        "N1\t1\tAAA\t1.0\t0.0\t0.0\n"
        "N1\t2\tAAC\t0.0\t1.0\t0.0\n",
        encoding="utf-8",
    )
    tr = tree.add_numerical_node_labels(
        ete.PhyloNode("((A:1,B:1)N1:1,(C:1,D:1):1)R;", format=1)
    )
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

    labels = {
        node.name: int(ete.get_prop(node, "numerical_label"))
        for node in tr.traverse()
        if node.name
    }
    np.testing.assert_allclose(out[labels["N1"], 0, :], [1.0, 0.0, 0.0])


def test_get_state_tensor_streams_state_rows_without_pandas_read_csv(tmp_path, monkeypatch):
    g = _make_state_tensor_g(
        tmp_path=tmp_path,
        alignment_text=">A\nAAAAAC\n>B\nAAAAAC\n",
    )
    state_file = tmp_path / "toy.state.tsv"
    state_file.write_text(
        "Node\tSite\tState\tp_AAA\tp_AAC\tp_AAG\n"
        "R\t2\tAAC\t0.0\t1.0\t0.0\n"
        "R\t1\tAAG\t0.0\t0.0\t1.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        parser_iqtree.pd,
        "read_csv",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pandas state load")),
    )

    out = parser_iqtree.get_state_tensor(g)

    root_id = int(ete.get_prop(ete.get_tree_root(g["tree"]), "numerical_label"))
    np.testing.assert_allclose(out[root_id, 0, :], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(out[root_id, 1, :], [0.0, 1.0, 0.0], atol=1e-12)


def test_get_state_tensor_raises_when_leaf_sequence_missing(tmp_path):
    g = _make_state_tensor_g(
        tmp_path=tmp_path,
        alignment_text=">A\nAAAAAC\n",
    )
    with pytest.raises(AssertionError):
        parser_iqtree.get_state_tensor(g)


def test_get_state_tensor_rejects_leaf_sequence_length_mismatch(tmp_path):
    g = _make_state_tensor_g(
        tmp_path=tmp_path,
        alignment_text=">A\nAAAAAC\n>B\nAAA\n",
    )
    with pytest.raises(AssertionError, match="Codon site count did not match alignment size"):
        parser_iqtree.get_state_tensor(g)


def test_get_state_tensor_rejects_duplicate_node_site_rows(tmp_path):
    g = _make_state_tensor_g(
        tmp_path=tmp_path,
        alignment_text=">A\nAAAAAC\n>B\nAAGAAG\n",
    )
    state_file = tmp_path / "toy.state.tsv"
    state_file.write_text(
        "Node\tSite\tState\tp_AAA\tp_AAC\tp_AAG\n"
        "R\t1\tAAA\t1.0\t0.0\t0.0\n"
        "R\t1\tAAC\t0.0\t1.0\t0.0\n",
        encoding="utf-8",
    )
    g["path_iqtree_state"] = str(state_file)
    with pytest.raises(ValueError, match="Duplicate Node/Site"):
        parser_iqtree.get_state_tensor(g)


def test_get_state_tensor_rejects_non_integer_site_values(tmp_path):
    g = _make_state_tensor_g(
        tmp_path=tmp_path,
        alignment_text=">A\nAAAAAC\n>B\nAAGAAG\n",
    )
    state_file = tmp_path / "toy_noninteger.state.tsv"
    state_file.write_text(
        "Node\tSite\tState\tp_AAA\tp_AAC\tp_AAG\n"
        "R\t1.5\tAAA\t1.0\t0.0\t0.0\n"
        "R\t2.0\tAAC\t0.0\t1.0\t0.0\n",
        encoding="utf-8",
    )
    g["path_iqtree_state"] = str(state_file)
    with pytest.raises(ValueError, match="Non-integer Site"):
        parser_iqtree.get_state_tensor(g)


def test_get_state_tensor_rejects_invalid_probability_rows(tmp_path):
    g = _make_state_tensor_g(
        tmp_path=tmp_path,
        alignment_text=">A\nAAAAAC\n>B\nAAGAAG\n",
    )
    state_file = tmp_path / "toy_invalid_probability.state.tsv"
    state_file.write_text(
        "Node\tSite\tState\tp_AAA\tp_AAC\tp_AAG\n"
        "R\t1\tAAA\t0.8\t0.3\t-0.1\n"
        "R\t2\tAAC\t0.0\t1.0\t0.0\n",
        encoding="utf-8",
    )
    g["path_iqtree_state"] = str(state_file)
    with pytest.raises(ValueError, match="Invalid probability"):
        parser_iqtree.get_state_tensor(g)
