
import numpy as np
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
        "codon_orders": np.array(["AAA", "AAC", "AAG"]),
        "float_type": np.float64,
    }
    return g


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
    np.testing.assert_allclose(g["equilibrium_frequency"], [0.2, 0.3, 0.5], atol=1e-12)


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
    np.testing.assert_allclose(g["equilibrium_frequency"], [0.2, 0.3, 0.5], atol=1e-12)


def test_read_iqtree_v3_mixed_exponents_do_not_get_overridden_by_legacy_pattern(tmp_path):
    iqtree_text = """
IQ-TREE multicore version 3.0.1 for Linux 64-bit built Jan  1 2025
Model of substitution: ECMK07+F+R4
 pi(AAA)=2.0e-01 pi(AAC)=3.0e-02
 pi(AAG)=5.0e-03
"""
    g = _get_base_g(tmp_path=tmp_path, iqtree_text=iqtree_text, log_text="")
    g = parser_iqtree.read_iqtree(g, eq=True)
    expected = np.array([2.0e-01, 3.0e-02, 5.0e-03], dtype=float)
    expected /= expected.sum()
    np.testing.assert_allclose(g["equilibrium_frequency"], expected, atol=1e-12)


def test_read_iqtree_v3_mixed_spacing_scientific_notation_parses_all_codons_correctly(tmp_path):
    iqtree_text = """
IQ-TREE multicore version 3.0.1 for Linux 64-bit built Jan  1 2025
Model of substitution: ECMK07+F+R4
 pi( AAA ) = 2.000000e-01   pi(AAC)=3.000000E-01
 pi(AAG)=5.000000e-01
"""
    g = _get_base_g(tmp_path=tmp_path, iqtree_text=iqtree_text, log_text="")
    g = parser_iqtree.read_iqtree(g, eq=True)
    np.testing.assert_allclose(g["equilibrium_frequency"], [0.2, 0.3, 0.5], atol=1e-12)


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
    np.testing.assert_allclose(g["equilibrium_frequency"], [0.2, 0.3, 0.5], atol=1e-12)


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


def test_read_state_rejects_nucleotide_input(tmp_path):
    state_file = tmp_path / "toy_nuc.state.tsv"
    state_file.write_text(
        "Node\tSite\tState\tp_A\tp_C\tp_G\tp_T\n",
        encoding="utf-8",
    )
    with pytest.raises(NotImplementedError, match="Nucleotide ancestral-state input is obsolete"):
        parser_iqtree.read_state({"iqtree_state": str(state_file)})


def test_read_state_rejects_protein_input(tmp_path):
    state_file = tmp_path / "toy_pep.state.tsv"
    header = "Node\tSite\tState\t" + "\t".join(["p_AA{}".format(i) for i in range(20)]) + "\n"
    state_file.write_text(header, encoding="utf-8")
    with pytest.raises(NotImplementedError, match="Protein ancestral-state input is obsolete"):
        parser_iqtree.read_state({"iqtree_state": str(state_file)})


def test_read_state_rejects_duplicate_codon_columns(tmp_path):
    state_file = tmp_path / "toy_dup.state.tsv"
    codon_cols = ["p_C{:02d}".format(i) for i in range(20)] + ["p_AAA", "p_p_AAA"]
    header = "Node\tSite\tState\t" + "\t".join(codon_cols) + "\n"
    state_file.write_text(header, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate codon state columns"):
        parser_iqtree.read_state({"iqtree_state": str(state_file)})


def test_read_rate_falls_back_to_rate_column_when_c_rate_is_missing(tmp_path):
    rate_file = tmp_path / "toy.rate"
    rate_file.write_text(
        "Site\tRate\n"
        "1\t0.5\n"
        "2\t1.5\n",
        encoding="utf-8",
    )
    g = {"path_iqtree_rate": str(rate_file), "num_input_site": 2}
    out = parser_iqtree.read_rate(g)
    np.testing.assert_allclose(out, [0.5, 1.5], atol=1e-12)


def test_read_rate_prefers_posterior_mean_rate_and_retains_categorized_rate(tmp_path):
    rate_file = tmp_path / "toy.rate"
    rate_file.write_text(
        "Site\tRate\tC_Rate\n"
        "1\t0.25\t0.10\n"
        "2\t1.75\t2.00\n",
        encoding="utf-8",
    )
    g = {"path_iqtree_rate": str(rate_file), "num_input_site": 2}

    out = parser_iqtree.read_rate(g)

    np.testing.assert_allclose(out, [0.25, 1.75], atol=1e-12)
    np.testing.assert_allclose(g["iqtree_categorized_rate_values"], [0.10, 2.00], atol=1e-12)


def test_read_rate_accepts_whitespace_padded_column_name(tmp_path):
    rate_file = tmp_path / "toy.rate"
    rate_file.write_text(
        "Site\t C_Rate \n"
        "1\t0.25\n",
        encoding="utf-8",
    )
    g = {"path_iqtree_rate": str(rate_file), "num_input_site": 1}
    out = parser_iqtree.read_rate(g)
    np.testing.assert_allclose(out, [0.25], atol=1e-12)


def test_read_rate_accepts_case_insensitive_c_rate_header(tmp_path):
    rate_file = tmp_path / "toy.rate"
    rate_file.write_text(
        "Site\tc_rate\n"
        "1\t0.75\n",
        encoding="utf-8",
    )
    g = {"path_iqtree_rate": str(rate_file), "num_input_site": 1}
    out = parser_iqtree.read_rate(g)
    np.testing.assert_allclose(out, [0.75], atol=1e-12)


def test_read_rate_rejects_missing_rate_columns(tmp_path):
    rate_file = tmp_path / "toy.rate"
    rate_file.write_text(
        "Site\tFoo\n"
        "1\t0.5\n",
        encoding="utf-8",
    )
    g = {"path_iqtree_rate": str(rate_file), "num_input_site": 1}
    with pytest.raises(ValueError, match="C_Rate"):
        parser_iqtree.read_rate(g)


def test_read_rate_uses_num_input_site_when_rate_file_has_no_rows(tmp_path):
    rate_file = tmp_path / "toy.rate"
    rate_file.write_text("Site\tC_Rate\n", encoding="utf-8")
    g = {"path_iqtree_rate": str(rate_file), "num_input_site": 3}
    out = parser_iqtree.read_rate(g)
    np.testing.assert_allclose(out, [1.0, 1.0, 1.0], atol=1e-12)


def test_read_rate_rejects_site_count_mismatch(tmp_path):
    rate_file = tmp_path / "toy.rate"
    rate_file.write_text(
        "Site\tC_Rate\n"
        "1\t0.5\n"
        "2\t1.5\n",
        encoding="utf-8",
    )
    g = {"path_iqtree_rate": str(rate_file), "num_input_site": 3}
    with pytest.raises(ValueError, match="did not match num_input_site"):
        parser_iqtree.read_rate(g)


def test_read_rate_orders_values_by_site_and_rejects_negative_rates(tmp_path):
    rate_file = tmp_path / "unordered.rate"
    rate_file.write_text(
        "Site\tRate\n"
        "2\t1.5\n"
        "1\t0.5\n",
        encoding="utf-8",
    )
    g = {"path_iqtree_rate": str(rate_file), "num_input_site": 2}
    np.testing.assert_allclose(parser_iqtree.read_rate(g), [0.5, 1.5])
    rate_file.write_text("Site\tRate\n1\t-0.5\n2\t1.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="negative"):
        parser_iqtree.read_rate(g)


def test_read_rate_rejects_duplicate_or_out_of_range_site_labels(tmp_path):
    rate_file = tmp_path / "bad-sites.rate"
    rate_file.write_text("Site\tRate\n1\t0.5\n1\t1.5\n", encoding="utf-8")
    g = {"path_iqtree_rate": str(rate_file), "num_input_site": 2}
    with pytest.raises(ValueError, match="each integer"):
        parser_iqtree.read_rate(g)
