import numpy as np
import pytest

from csubst import parser_misc


def test_fill_instantaneous_rate_matrix_diagonal_sets_row_sums_to_zero():
    inst = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=float)
    out = parser_misc.fill_instantaneous_rate_matrix_diagonal(inst)
    np.testing.assert_allclose(out, np.array([[-1.0, 1.0], [2.0, -2.0]]), atol=1e-12)
    np.testing.assert_allclose(out.sum(axis=1), [0.0, 0.0], atol=1e-12)


def test_scale_instantaneous_rate_matrix_matches_manual_scaling():
    inst = np.array([[0.0, 2.0], [1.0, 0.0]], dtype=float)
    eq = np.array([0.4, 0.6], dtype=float)
    out = parser_misc.scale_instantaneous_rate_matrix(inst.copy(), eq)
    expected = np.array([[0.0, 2.0 / 1.4], [1.0 / 1.4, 0.0]])
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_scale_instantaneous_rate_matrix_requires_zero_diagonal():
    inst = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    with pytest.raises(AssertionError, match="Diagonal elements"):
        parser_misc.scale_instantaneous_rate_matrix(inst, np.array([0.5, 0.5]))


def test_scale_instantaneous_rate_matrix_requires_all_diagonal_elements_zero():
    inst = np.array([[0.0, 2.0], [3.0, 1e-3]], dtype=float)
    with pytest.raises(AssertionError, match="Diagonal elements"):
        parser_misc.scale_instantaneous_rate_matrix(inst, np.array([0.5, 0.5]))


def test_scale_instantaneous_rate_matrix_requires_positive_scaling_factor():
    inst = np.zeros((2, 2), dtype=float)
    with pytest.raises(AssertionError, match="scaling factor must be positive"):
        parser_misc.scale_instantaneous_rate_matrix(inst, np.array([0.5, 0.5]))


def test_exchangeability2Q_matches_manual_result():
    ex = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    eq = np.array([0.25, 0.75], dtype=float)
    out = parser_misc.exchangeability2Q(ex, eq)
    expected = np.array([[-2.0, 2.0], [2.0 / 3.0, -2.0 / 3.0]])
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_get_equilibrium_frequency_for_codon_and_amino_acid_modes():
    g = {
        "equilibrium_frequency": np.array([0.2, 0.3, 0.5]),
        "amino_acid_orders": np.array(["K", "N"]),
        "synonymous_indices": {"K": [0, 1], "N": [2]},
        "float_tol": 1e-12,
    }
    eq_cdn = parser_misc.get_equilibrium_frequency(g, "cdn")
    eq_pep = parser_misc.get_equilibrium_frequency(g, "pep")
    np.testing.assert_allclose(eq_cdn, [0.2, 0.3, 0.5], atol=1e-12)
    np.testing.assert_allclose(eq_pep, [0.5, 0.5], atol=1e-12)


def test_get_equilibrium_frequency_rejects_unknown_mode():
    g = {
        "equilibrium_frequency": np.array([0.2, 0.3, 0.5]),
        "amino_acid_orders": np.array(["K", "N"]),
        "synonymous_indices": {"K": [0, 1], "N": [2]},
        "float_tol": 1e-12,
    }
    with pytest.raises(ValueError, match="Unsupported equilibrium-frequency mode"):
        parser_misc.get_equilibrium_frequency(g, "unknown")


def test_get_codon_order_index_reorders_positions():
    order_from = np.array(["AAA", "AAC", "AAG"])
    order_to = np.array(["AAG", "AAA", "AAC"])
    out = parser_misc.get_codon_order_index(order_from, order_to)
    np.testing.assert_array_equal(out, [1, 2, 0])


def test_get_codon_order_index_raises_on_missing_codon():
    order_from = np.array(["AAA", "XXX", "AAC"])
    order_to = np.array(["AAG", "AAA", "AAC"])
    with pytest.raises(ValueError, match="not found in target order|missing"):
        parser_misc.get_codon_order_index(order_from, order_to)


def test_get_codon_order_index_raises_on_duplicate_target_codon():
    order_from = np.array(["AAA", "AAC", "AAG"])
    order_to = np.array(["AAA", "AAA", "AAC"])
    with pytest.raises(ValueError, match="Duplicate codon"):
        parser_misc.get_codon_order_index(order_from, order_to)


def test_get_codon_order_index_raises_on_duplicate_source_codon():
    order_from = np.array(["AAA", "AAA", "AAC"])
    order_to = np.array(["AAA", "AAC", "AAG"])
    with pytest.raises(ValueError, match="source order"):
        parser_misc.get_codon_order_index(order_from, order_to)


def test_get_exchangeability_codon_order_shape_and_no_stops():
    codons = parser_misc.get_exchangeability_codon_order()
    assert codons.shape == (61,)
    assert set(["TAA", "TAG", "TGA"]).isdisjoint(set(codons))


def test_read_exchangeability_eq_freq_rejects_truncated_file(monkeypatch):
    monkeypatch.setattr(parser_misc, "_read_package_text", lambda file: "line0\nline1")
    g = {"codon_orders": parser_misc.get_exchangeability_codon_order()}
    with pytest.raises(AssertionError, match="expected equilibrium frequencies"):
        parser_misc.read_exchangeability_eq_freq(file="dummy", g=g)


def test_get_rate_tensor_for_asis_and_syn_modes():
    inst = np.array(
        [[-1.0, 0.2, 0.8], [0.3, -0.7, 0.4], [0.5, 0.6, -1.1]],
        dtype=float,
    )
    g = {
        "amino_acid_orders": np.array(["K", "N"]),
        "synonymous_indices": {"K": [0, 1], "N": [2]},
        "max_synonymous_size": 2,
        "float_type": np.float64,
    }
    asis = parser_misc.get_rate_tensor(inst, "asis", g)
    syn = parser_misc.get_rate_tensor(inst, "syn", g)
    np.testing.assert_allclose(
        asis,
        np.array([[[0.0, 0.2, 0.8], [0.3, 0.0, 0.4], [0.5, 0.6, 0.0]]]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        syn,
        np.array([[[0.0, 0.2], [0.3, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
        atol=1e-12,
    )


def test_get_rate_tensor_rejects_unknown_mode():
    inst = np.array(
        [[-1.0, 0.2], [0.3, -0.3]],
        dtype=float,
    )
    g = {
        "amino_acid_orders": np.array(["K"]),
        "synonymous_indices": {"K": [0, 1]},
        "max_synonymous_size": 2,
        "float_type": np.float64,
    }
    with pytest.raises(ValueError, match="Unsupported rate-tensor mode"):
        parser_misc.get_rate_tensor(inst, "unknown", g)


def test_cdn2pep_matrix_matches_manual_group_sum():
    inst_cdn = np.array(
        [[-1.0, 0.2, 0.8], [0.3, -0.7, 0.4], [0.5, 0.6, -1.1]],
        dtype=float,
    )
    g = {
        "amino_acid_orders": np.array(["K", "N"]),
        "synonymous_indices": {"K": [0, 1], "N": [2]},
    }
    out = parser_misc.cdn2pep_matrix(inst_cdn, g)
    np.testing.assert_allclose(out, np.array([[-1.2, 1.2], [1.1, -1.1]]), atol=1e-12)


def test_cdn2nsy_matrix_matches_manual_recoded_group_sum():
    inst_cdn = np.array(
        [
            [-1.2, 0.2, 1.0, 0.0],
            [0.1, -0.7, 0.3, 0.3],
            [0.4, 0.2, -0.8, 0.2],
            [0.0, 0.5, 0.1, -0.6],
        ],
        dtype=float,
    )
    g = {
        "nonsyn_state_orders": np.array(["AG", "C"], dtype=object),
        "nonsynonymous_indices": {"AG": [0, 1, 2], "C": [3]},
    }
    out = parser_misc.cdn2nsy_matrix(inst_cdn, g)
    np.testing.assert_allclose(out, np.array([[-0.5, 0.5], [0.6, -0.6]]), atol=1e-12)


def test_initialize_and_report_nonsyn_recode_writes_table(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    amino_acids = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object)
    codon_orders = np.array(["C{:02d}".format(i) for i in range(amino_acids.shape[0])], dtype=object)
    synonymous_indices = {aa: [i] for i, aa in enumerate(amino_acids.tolist())}
    matrix_groups = {aa: [codon_orders[i]] for i, aa in enumerate(amino_acids.tolist())}
    g = {
        "amino_acid_orders": amino_acids,
        "codon_orders": codon_orders,
        "synonymous_indices": synonymous_indices,
        "matrix_groups": matrix_groups,
        "nonsyn_recode": "dayhoff6",
    }
    out = parser_misc._initialize_and_report_nonsyn_recode(g)
    assert out["nonsyn_recode"] == "dayhoff6"
    output_path = tmp_path / "csubst_nonsyn_recoding.tsv"
    assert output_path.exists() is True
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("recode\tstate_id\tstate_label")
    pca_path = tmp_path / "csubst_nonsyn_recoding_pca.png"
    assert pca_path.exists() is False


def test_initialize_and_report_nonsyn_recode_writes_pca_when_enabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    amino_acids = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object)
    codon_orders = np.array(["C{:02d}".format(i) for i in range(amino_acids.shape[0])], dtype=object)
    synonymous_indices = {aa: [i] for i, aa in enumerate(amino_acids.tolist())}
    matrix_groups = {aa: [codon_orders[i]] for i, aa in enumerate(amino_acids.tolist())}
    g = {
        "amino_acid_orders": amino_acids,
        "codon_orders": codon_orders,
        "synonymous_indices": synonymous_indices,
        "matrix_groups": matrix_groups,
        "nonsyn_recode": "dayhoff6",
        "plot_nonsyn_recode_pca": True,
    }
    out = parser_misc._initialize_and_report_nonsyn_recode(g)
    assert out["nonsyn_recode"] == "dayhoff6"
    pca_path = tmp_path / "csubst_nonsyn_recoding_pca.png"
    assert pca_path.exists() is True
    assert pca_path.stat().st_size > 0


def test_initialize_and_report_nonsyn_recode_writes_pca_for_no_when_enabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    amino_acids = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object)
    codon_orders = np.array(["C{:02d}".format(i) for i in range(amino_acids.shape[0])], dtype=object)
    synonymous_indices = {aa: [i] for i, aa in enumerate(amino_acids.tolist())}
    matrix_groups = {aa: [codon_orders[i]] for i, aa in enumerate(amino_acids.tolist())}
    g = {
        "amino_acid_orders": amino_acids,
        "codon_orders": codon_orders,
        "synonymous_indices": synonymous_indices,
        "matrix_groups": matrix_groups,
        "nonsyn_recode": "no",
        "plot_nonsyn_recode_pca": True,
    }
    out = parser_misc._initialize_and_report_nonsyn_recode(g)
    assert out["nonsyn_recode"] == "no"
    table_path = tmp_path / "csubst_nonsyn_recoding.tsv"
    assert table_path.exists() is False
    pca_path = tmp_path / "csubst_nonsyn_recoding_pca.png"
    assert pca_path.exists() is True
    assert pca_path.stat().st_size > 0


def test_get_mechanistic_instantaneous_rate_matrix_matches_manual_example():
    g = {
        "codon_orders": np.array(["AAA", "AAG", "AAC"]),
        "amino_acid_orders": np.array(["K", "N"]),
        "synonymous_indices": {"K": [0, 1], "N": [2]},
        "omega": 2.0,
        "kappa": 3.0,
        "equilibrium_frequency": np.array([0.2, 0.3, 0.5]),
        "float_type": np.float64,
    }
    out = parser_misc.get_mechanistic_instantaneous_rate_matrix(g)
    expected = np.array(
        [
            [-1.397058823529412, 0.661764705882353, 0.735294117647059],
            [0.441176470588235, -1.176470588235294, 0.735294117647059],
            [0.294117647058824, 0.441176470588235, -0.735294117647059],
        ]
    )
    np.testing.assert_allclose(out, expected, atol=1e-12)
    np.testing.assert_allclose(out.sum(axis=1), [0.0, 0.0, 0.0], atol=1e-12)


def test_get_mechanistic_instantaneous_rate_matrix_applies_kappa_only_to_transitions():
    g = {
        "codon_orders": np.array(["AAA", "AAG", "AAC"]),
        "amino_acid_orders": np.array(["K", "N"]),
        "synonymous_indices": {"K": [0, 1], "N": [2]},
        "omega": 1.0,
        "kappa": 5.0,
        "equilibrium_frequency": np.array([1 / 3, 1 / 3, 1 / 3]),
        "float_type": np.float64,
    }
    out = parser_misc.get_mechanistic_instantaneous_rate_matrix(g)
    # AAA->AAG is transition (A<->G), AAA->AAC is transversion (A<->C).
    assert out[0, 1] > out[0, 2]


def test_get_mechanistic_instantaneous_rate_matrix_supports_zero_omega_without_nan():
    g = {
        "codon_orders": np.array(["AAA", "AAG", "AAC"]),
        "amino_acid_orders": np.array(["K", "N"]),
        "synonymous_indices": {"K": [0, 1], "N": [2]},
        "omega": 0.0,
        "kappa": None,
        "equilibrium_frequency": np.array([1 / 3, 1 / 3, 1 / 3]),
        "float_type": np.float64,
    }
    out = parser_misc.get_mechanistic_instantaneous_rate_matrix(g)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out.sum(axis=1), [0.0, 0.0, 0.0], atol=1e-12)
    assert out[0, 2] == pytest.approx(0.0, abs=1e-12)
    assert out[1, 2] == pytest.approx(0.0, abs=1e-12)
    assert out[2, 0] == pytest.approx(0.0, abs=1e-12)
    assert out[2, 1] == pytest.approx(0.0, abs=1e-12)
