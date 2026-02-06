import numpy
import pytest

from csubst import parser_iqtree


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
