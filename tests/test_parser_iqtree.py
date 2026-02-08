import numpy
import pytest

from csubst import parser_iqtree
from csubst import tree
from csubst import ete


def _get_base_g(tmp_path, iqtree_text, log_text):
    iqtree_file = tmp_path / "sample.iqtree"
    log_file = tmp_path / "sample.log"
    alignment_file = tmp_path / "sample.fa"
    iqtree_file.write_text(iqtree_text)
    log_file.write_text(log_text)
    alignment_file.write_text(
        ">seq1\n"
        "AAAAACAAGAAGAAG\n"
        ">seq2\n"
        "AAAAACAACAAGAAG\n"
    )
    g = {
        "path_iqtree_iqtree": str(iqtree_file),
        "path_iqtree_log": str(log_file),
        "alignment_file": str(alignment_file),
        "codon_orders": numpy.array(["AAA", "AAC", "AAG"]),
        "float_type": numpy.float64,
    }
    return g


def test_parse_iqtree_version_text_for_v2_and_v3():
    v2_txt = "IQ-TREE multicore version 2.3.6 for MacOS Intel 64-bit built Aug  4 2024"
    v3_txt = "IQ-TREE multicore version 3.0.1 for Linux 64-bit built Jan  1 2025"
    assert parser_iqtree._parse_iqtree_version_text(v2_txt) == ("2.3.6", 2)
    assert parser_iqtree._parse_iqtree_version_text(v3_txt) == ("3.0.1", 3)


def test_read_iqtree_detects_v2_and_parses_equilibrium_frequency(tmp_path):
    iqtree_text = """
IQ-TREE multicore version 2.3.6 for MacOS Intel 64-bit built Aug  4 2024
Model of substitution: ECMK07+F+R4
  pi(AAA) = 0.2  pi(AAC) = 0.3
  pi(AAG) = 0.5
"""
    g = _get_base_g(tmp_path=tmp_path, iqtree_text=iqtree_text, log_text="")
    g = parser_iqtree.read_iqtree(g, eq=True)
    assert g["substitution_model"] == "ECMK07+F+R4"
    assert g["iqtree_output_version_major"] == 2
    assert g["iqtree_parser"] == "iqtree2"
    numpy.testing.assert_allclose(g["equilibrium_frequency"], [0.2, 0.3, 0.5], atol=1e-12)


def test_read_iqtree_detects_v3_and_parses_scientific_notation(tmp_path):
    iqtree_text = """
IQ-TREE multicore version 3.0.1 for Linux 64-bit built Jan  1 2025
Model of substitution: ECMK07+F+R4
 pi(AAA)=2.0e-01 pi(AAC)=3.0e-01
 pi(AAG)=5.0e-01
"""
    g = _get_base_g(tmp_path=tmp_path, iqtree_text=iqtree_text, log_text="")
    g = parser_iqtree.read_iqtree(g, eq=True)
    assert g["substitution_model"] == "ECMK07+F+R4"
    assert g["iqtree_output_version_major"] == 3
    assert g["iqtree_parser"] == "iqtree3"
    numpy.testing.assert_allclose(g["equilibrium_frequency"], [0.2, 0.3, 0.5], atol=1e-12)


def test_read_iqtree_v3_mixed_exponents_do_not_get_overridden_by_legacy_pattern(tmp_path):
    iqtree_text = """
IQ-TREE multicore version 3.0.1 for Linux 64-bit built Jan  1 2025
Model of substitution: ECMK07+F+R4
 pi(AAA)=2.0e-01 pi(AAC)=3.0e-02
 pi(AAG)=5.0e-03
"""
    g = _get_base_g(tmp_path=tmp_path, iqtree_text=iqtree_text, log_text="")
    g = parser_iqtree.read_iqtree(g, eq=True)
    expected = numpy.array([2.0e-01, 3.0e-02, 5.0e-03], dtype=float)
    expected /= expected.sum()
    numpy.testing.assert_allclose(g["equilibrium_frequency"], expected, atol=1e-12)


def test_read_iqtree_v3_mixed_spacing_scientific_notation_parses_all_codons_correctly(tmp_path):
    iqtree_text = """
IQ-TREE multicore version 3.0.1 for Linux 64-bit built Jan  1 2025
Model of substitution: ECMK07+F+R4
 pi( AAA ) = 2.000000e-01   pi(AAC)=3.000000E-01
 pi(AAG)=5.000000e-01
"""
    g = _get_base_g(tmp_path=tmp_path, iqtree_text=iqtree_text, log_text="")
    g = parser_iqtree.read_iqtree(g, eq=True)
    numpy.testing.assert_allclose(g["equilibrium_frequency"], [0.2, 0.3, 0.5], atol=1e-12)


def test_read_iqtree_iqtree2_missing_frequency_raises_clear_error(tmp_path):
    iqtree_text = """
IQ-TREE multicore version 2.3.6 for Linux 64-bit built Jan  1 2025
Model of substitution: ECMK07+F+R4
 pi(AAA)=0.2 pi(AAC)=0.3
"""
    g = _get_base_g(tmp_path=tmp_path, iqtree_text=iqtree_text, log_text="")
    with pytest.raises(AssertionError, match="Missing codon"):
        parser_iqtree.read_iqtree(g, eq=True)


def test_read_iqtree_iqtree3_missing_frequency_falls_back_to_alignment_empirical(tmp_path):
    iqtree_text = """
IQ-TREE multicore version 3.0.1 for Linux 64-bit built Jan  1 2025
Model of substitution: ECMK07+F+R4
"""
    g = _get_base_g(tmp_path=tmp_path, iqtree_text=iqtree_text, log_text="")
    g = parser_iqtree.read_iqtree(g, eq=True)
    assert g["iqtree_parser"] == "iqtree3"
    numpy.testing.assert_allclose(g["equilibrium_frequency"], [0.2, 0.3, 0.5], atol=1e-12)


def test_read_log_parses_omega_kappa_and_codon_table_from_v3_style_log(tmp_path):
    iqtree_text = "IQ-TREE multicore version 3.0.1 for Linux 64-bit built Jan  1 2025\n"
    log_text = """
Converting to codon sequences with genetic code 1 ...
Nonsynonymous/synonymous ratio (omega): 1.23e-01
Transition/transversion ratio (kappa): 2.34E+00
"""
    g = _get_base_g(tmp_path=tmp_path, iqtree_text=iqtree_text, log_text=log_text)
    g = parser_iqtree.read_log(g)
    assert g["iqtree_output_version_major"] == 3
    assert g["reconstruction_codon_table"] == 1
    assert g["omega"] == pytest.approx(0.123)
    assert g["kappa"] == pytest.approx(2.34)


def _make_state_tensor_g(tmp_path, alignment_text):
    alignment_file = tmp_path / "toy.fa"
    state_file = tmp_path / "toy.state.tsv"
    alignment_file.write_text(alignment_text, encoding="utf-8")
    state_file.write_text("Node\tSite\tState\tp_AAA\tp_AAC\tp_AAG\n", encoding="utf-8")
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    return {
        "tree": tr,
        "alignment_file": str(alignment_file),
        "path_iqtree_state": str(state_file),
        "num_input_site": 2,
        "num_input_state": 3,
        "input_data_type": "cdn",
        "codon_orders": numpy.array(["AAA", "AAC", "AAG"]),
        "float_type": numpy.float64,
        "ml_anc": False,
    }


def test_get_state_tensor_reads_leaf_sequences_via_ete_compat(tmp_path):
    g = _make_state_tensor_g(
        tmp_path=tmp_path,
        alignment_text=">A\nAAAAAC\n>B\nAAGAAG\n",
    )
    out = parser_iqtree.get_state_tensor(g)

    labels = {n.name: ete.get_prop(n, "numerical_label") for n in g["tree"].traverse()}
    numpy.testing.assert_allclose(out[labels["A"], 0, :], [1.0, 0.0, 0.0], atol=1e-12)
    numpy.testing.assert_allclose(out[labels["A"], 1, :], [0.0, 1.0, 0.0], atol=1e-12)
    numpy.testing.assert_allclose(out[labels["B"], 0, :], [0.0, 0.0, 1.0], atol=1e-12)
    numpy.testing.assert_allclose(out[labels["B"], 1, :], [0.0, 0.0, 1.0], atol=1e-12)


def test_get_state_tensor_raises_when_leaf_sequence_missing(tmp_path):
    g = _make_state_tensor_g(
        tmp_path=tmp_path,
        alignment_text=">A\nAAAAAC\n",
    )
    with pytest.raises(AssertionError):
        parser_iqtree.get_state_tensor(g)


def test_get_state_tensor_selected_branch_ids_preserve_global_branch_index(tmp_path):
    g = _make_state_tensor_g(
        tmp_path=tmp_path,
        alignment_text=">A\nAAAAAC\n>B\nAAGAAG\n",
    )
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in g["tree"].traverse()}
    full = parser_iqtree.get_state_tensor(g)
    selected = parser_iqtree.get_state_tensor(
        g=g,
        selected_branch_ids=numpy.array([labels["A"]], dtype=numpy.int64),
    )
    assert selected.shape == full.shape
    numpy.testing.assert_allclose(selected[labels["A"], :, :], full[labels["A"], :, :], atol=1e-12)
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
        "N1\t2\tAAC\t0.0\t1.0\t0.0\n",
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
        "codon_orders": numpy.array(["AAA", "AAC", "AAG"]),
        "float_type": numpy.float64,
        "ml_anc": False,
    }
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in g["tree"].traverse()}
    full = parser_iqtree.get_state_tensor(g)
    selected = parser_iqtree.get_state_tensor(
        g=g,
        selected_branch_ids=numpy.array([labels["N1"]], dtype=numpy.int64),
    )
    assert selected.shape == full.shape
    numpy.testing.assert_allclose(selected[labels["N1"], :, :], full[labels["N1"], :, :], atol=1e-12)
    assert selected[labels["C"], :, :].sum() == 0
