import numpy as np
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
        "codon_orders": np.array(["AAA", "AAC", "AAG"]),
        "float_type": np.float64,
    }
    return g


def test_parse_iqtree_version_text_for_v2_and_v3():
    v2_txt = "IQ-TREE multicore version 2.3.6 for MacOS Intel 64-bit built Aug  4 2024"
    v3_txt = "IQ-TREE multicore version 3.0.1 for Linux 64-bit built Jan  1 2025"
    assert parser_iqtree._parse_iqtree_version_text(v2_txt) == ("2.3.6", 2)
    assert parser_iqtree._parse_iqtree_version_text(v3_txt) == ("3.0.1", 3)


def test_detect_iqtree_output_version_handles_non_utf8_iqtree_file(tmp_path):
    iqtree_file = tmp_path / "sample.iqtree"
    iqtree_file.write_bytes(b"\xff\xfeIQ-TREE multicore version 2.3.6 for Linux 64-bit\n")
    g = {"path_iqtree_iqtree": str(iqtree_file), "path_iqtree_log": str(tmp_path / "missing.log")}
    out = parser_iqtree.detect_iqtree_output_version(g)
    assert out["iqtree_output_version"] == "2.3.6"
    assert out["iqtree_output_version_major"] == 2


def test_is_version_at_least_handles_numeric_versions():
    assert parser_iqtree._is_version_at_least("2.0.0", "2.0.0")
    assert parser_iqtree._is_version_at_least("2.3.6", "2.0.0")
    assert parser_iqtree._is_version_at_least("3.0.0", "2.0.0")
    assert not parser_iqtree._is_version_at_least("1.6.12", "2.0.0")


def test_check_iqtree_dependency_rejects_nonzero_exit(monkeypatch):
    class _FakeProc:
        returncode = 1
        stdout = b""
        stderr = b""

    monkeypatch.setattr(parser_iqtree.subprocess, "run", lambda *args, **kwargs: _FakeProc())
    with pytest.raises(AssertionError, match="iqtree PATH cannot be found"):
        parser_iqtree.check_iqtree_dependency({"iqtree_exe": "iqtree"})


def test_check_iqtree_dependency_handles_non_utf8_version_output(monkeypatch):
    class _FakeProc:
        returncode = 0
        stdout = b"\xff\xfeIQ-TREE multicore version 2.3.6\n"
        stderr = b""

    monkeypatch.setattr(parser_iqtree.subprocess, "run", lambda *args, **kwargs: _FakeProc())
    g = {"iqtree_exe": "iqtree"}
    parser_iqtree.check_iqtree_dependency(g)
    assert g["iqtree_version"] == "2.3.6"
    assert g["iqtree_version_major"] == 2


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
        "N1\t1\tAAG\t0.0\t0.0\t1.0\n",
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

    class _FakeProc:
        def __init__(self, returncode):
            self.returncode = returncode
            self.stdout = None
            self.stderr = None

    monkeypatch.setattr(parser_iqtree.tree, "write_tree", fake_write_tree)
    monkeypatch.setattr(parser_iqtree.subprocess, "run", lambda *args, **kwargs: _FakeProc(returncode=2))

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
