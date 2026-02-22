import numpy as np
import pandas as pd
import pytest

from csubst import parser_misc
from csubst import table
from csubst import ete
from csubst import tree


def test_sort_branch_ids_sorts_within_rows_and_by_row():
    df = pd.DataFrame(
        {
            "branch_id_1": [8, 3, 5],
            "branch_id_2": [1, 2, 4],
            "site": [2, 1, 0],
        }
    )
    out = table.sort_branch_ids(df)
    assert list(map(tuple, out[["branch_id_1", "branch_id_2", "site"]].to_numpy())) == [
        (1, 8, 2),
        (2, 3, 1),
        (4, 5, 0),
    ]
    assert out["branch_id_1"].dtype.kind in "iu"
    assert out["branch_id_2"].dtype.kind in "iu"


def test_sort_branch_ids_without_branch_columns_sorts_by_site_only():
    df = pd.DataFrame(
        {
            "site": [3, 1, 2],
            "value": [30, 10, 20],
        }
    )
    out = table.sort_branch_ids(df.copy())
    assert out["site"].tolist() == [1, 2, 3]
    assert out["value"].tolist() == [10, 20, 30]
    assert out["site"].dtype.kind in "iu"


def test_sort_branch_ids_without_branch_or_site_columns_returns_input_order():
    df = pd.DataFrame({"value": [3, 1, 2]})
    out = table.sort_branch_ids(df.copy())
    assert out["value"].tolist() == [3, 1, 2]


def test_sort_branch_ids_rejects_non_integer_like_branch_values():
    df = pd.DataFrame(
        {
            "branch_id_1": [1.5, 2],
            "branch_id_2": [3, 4],
            "site": [1, 2],
        }
    )
    with pytest.raises(ValueError, match="integer-like"):
        table.sort_branch_ids(df.copy())


def test_sort_branch_ids_rejects_non_integer_like_site_values():
    df = pd.DataFrame(
        {
            "branch_id_1": [1, 2],
            "branch_id_2": [3, 4],
            "site": ["1", "2.5"],
        }
    )
    with pytest.raises(ValueError, match="integer-like"):
        table.sort_branch_ids(df.copy())


def test_sort_cb_stats_handles_non_string_column_names_regression():
    # Regression target inspired by issue #74.
    cb_stats = pd.DataFrame(
        {
            999: [1],
            "num_fg": [2],
            "mode": ["branch_and_bound"],
            "arity": [2],
            "elapsed_sec": [3.0],
            "cutoff_stat": ["OCNany2spe,2.0"],
            "fg_enrichment_factor": [1],
            "dSC_calibration": ["N"],
        }
    )
    out = table.sort_cb_stats(cb_stats)
    assert list(out.columns[:7]) == [
        "arity",
        "elapsed_sec",
        "cutoff_stat",
        "fg_enrichment_factor",
        "mode",
        "dSC_calibration",
        "num_fg",
    ]
    assert 999 in out.columns


def test_sort_cb_stats_handles_empty_columns_regression():
    cb_stats = pd.DataFrame()
    out = table.sort_cb_stats(cb_stats)
    assert out.shape == (0, 6)
    assert out.columns.tolist() == [
        "arity",
        "elapsed_sec",
        "cutoff_stat",
        "fg_enrichment_factor",
        "mode",
        "dSC_calibration",
    ]


def test_sort_cb_does_not_treat_unrelated_ocn_prefix_as_convergence_stat():
    cb = pd.DataFrame(
        {
            "OCNfoo": [9.0],
            "branch_id_1": [1],
            "ECNany2spe": [2.0],
            "OCNany2spe": [1.0],
        }
    )
    out = table.sort_cb(cb.copy())
    assert out.columns.tolist() == ["branch_id_1", "OCNany2spe", "ECNany2spe", "OCNfoo"]


def test_set_substitution_dtype_casts_integral_columns_only():
    df = pd.DataFrame(
        {
            "S_sub": [1.0, 2.0],
            "OCNany2any": [1.5, 2.0],
            "OCSany2spe": [3.0, 4.0],
        }
    )
    out = table.set_substitution_dtype(df)
    assert out["S_sub"].dtype.kind in "iu"
    assert out["OCSany2spe"].dtype.kind in "iu"
    assert out["OCNany2any"].dtype.kind == "f"


def test_get_linear_regression_residuals_match_manual_solution():
    cb = pd.DataFrame(
        {
            "OCSany2any": [1.0, 2.0],
            "OCSany2spe": [2.0, 4.0],
            "OCNany2any": [1.0, 2.0],
            "OCNany2spe": [1.0, 1.0],
        }
    )
    out = table.get_linear_regression(cb)
    np.testing.assert_allclose(out["OCS_linreg_residual"].to_numpy(), [0.0, 0.0], atol=1e-12)
    # coef = (1*1 + 2*1) / (1^2 + 2^2) = 3/5 = 0.6
    np.testing.assert_allclose(out["OCN_linreg_residual"].to_numpy(), [0.4, -0.2], atol=1e-12)


def test_get_linear_regression_skips_missing_mode_columns():
    cb = pd.DataFrame(
        {
            "OCSany2any": [1.0, 2.0],
            "OCSany2spe": [2.0, 4.0],
        }
    )
    out = table.get_linear_regression(cb)
    assert "OCS_linreg_residual" in out.columns
    assert "OCN_linreg_residual" not in out.columns


def test_chisq_test_returns_probability_for_nonzero_observation():
    x = pd.Series({"OCSany2spe": 3.0, "OCNany2spe": 2.0})
    out = table.chisq_test(x=x, total_S=10, total_N=20)
    assert 0.0 <= float(out) <= 1.0


def test_get_cutoff_stat_bool_array_parses_compound_expression():
    cb = pd.DataFrame(
        {
            "OCNany2spe": [1.9, 2.0, 2.1],
            "omegaCany2spe": [10.0, 4.9, 5.0],
        }
    )
    out = table.get_cutoff_stat_bool_array(cb, "OCNany2spe,2.0|omegaCany2spe,5.0")
    assert out.tolist() == [False, False, True]


def test_get_cutoff_stat_bool_array_rejects_unknown_column():
    cb = pd.DataFrame({"OCNany2spe": [1.0]})
    with pytest.raises(ValueError, match="was not found"):
        table.get_cutoff_stat_bool_array(cb, "DOES_NOT_EXIST,1.0")


def test_get_cutoff_stat_bool_array_accepts_whitespace_around_tokens():
    cb = pd.DataFrame(
        {
            "OCNany2spe": [1.9, 2.0, 2.1],
            "omegaCany2spe": [10.0, 4.9, 5.0],
        }
    )
    out = table.get_cutoff_stat_bool_array(cb, " OCNany2spe, 2.0 | omegaCany2spe , 5.0 ")
    assert out.tolist() == [False, False, True]


def test_parse_cutoff_stat_rejects_malformed_token():
    with pytest.raises(ValueError, match="Expected"):
        table.parse_cutoff_stat("OCNany2spe|omegaCany2spe,5.0")


def test_parse_cutoff_stat_rejects_invalid_regex():
    with pytest.raises(ValueError, match="Invalid cutoff regex"):
        table.parse_cutoff_stat("OCN[any2spe,2.0")


@pytest.mark.parametrize("value_text", ["nan", "inf", "-inf"])
def test_parse_cutoff_stat_rejects_non_finite_cutoff_value(value_text):
    with pytest.raises(ValueError, match="finite"):
        table.parse_cutoff_stat("OCNany2spe,{}".format(value_text))


def test_parse_cutoff_stat_supports_regex_with_comma_quantifier():
    out = table.parse_cutoff_stat(r"omegaC.{1,2},5.0")
    assert out == [(r"omegaC.{1,2}", 5.0)]


def test_parse_cutoff_stat_supports_regex_with_alternation_pipe():
    out = table.parse_cutoff_stat(r"OCN(any|dif)2spe,2.0|omegaCany2spe,5.0")
    assert out == [(r"OCN(any|dif)2spe", 2.0), ("omegaCany2spe", 5.0)]


def test_get_cutoff_stat_bool_array_supports_alternation_pipe_regex():
    cb = pd.DataFrame(
        {
            "OCNany2spe": [2.0, 1.9, 0.0],
            "OCNdif2spe": [2.1, 2.2, 0.0],
            "omegaCany2spe": [5.1, 5.1, 10.0],
        }
    )
    out = table.get_cutoff_stat_bool_array(cb, r"OCN(any|dif)2spe,2.0|omegaCany2spe,5.0")
    assert out.tolist() == [True, False, False]


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


def test_read_input_submodel_rejects_unsupported_substitution_model(monkeypatch):
    def fake_get_input_information(local_g):
        local_g["substitution_model"] = "UNSUPPORTED+F+R4"
        return local_g

    monkeypatch.setattr(parser_misc.parser_iqtree, "get_input_information", fake_get_input_information)
    g = {
        "infile_type": "iqtree",
        "omegaC_method": "submodel",
    }
    with pytest.raises(ValueError, match="Unsupported substitution model"):
        parser_misc.read_input(g)


def test_read_input_submodel_detects_reverse_signed_rate_sum_mismatch(monkeypatch):
    def fake_get_input_information(local_g):
        local_g.update(
            {
                "substitution_model": "GY+F+R4",
                "omega": 1.0,
                "kappa": 1.0,
                "equilibrium_frequency": np.array([0.5, 0.5], dtype=float),
                "codon_orders": np.array(["AAA", "AAC"]),
                "amino_acid_orders": np.array(["K", "N"]),
                "synonymous_indices": {"K": [0], "N": [1]},
                "matrix_groups": {"K": ["AAA"], "N": ["AAC"]},
            }
        )
        return local_g

    monkeypatch.setattr(parser_misc.parser_iqtree, "get_input_information", fake_get_input_information)
    monkeypatch.setattr(
        parser_misc,
        "get_mechanistic_instantaneous_rate_matrix",
        lambda g: np.array([[-0.75, 0.75], [0.75, -0.75]], dtype=float),
    )
    monkeypatch.setattr(
        parser_misc,
        "cdn2pep_matrix",
        lambda inst_cdn, g: np.array([[-2.0, 2.0], [2.0, -2.0]], dtype=float),
    )

    def fake_get_rate_tensor(inst, mode, g):
        if mode == "syn":
            return np.array([[[1.0, 0.0], [0.0, 0.0]]], dtype=float)
        if mode == "asis":
            return np.array([[[0.0, 0.5], [0.0, 0.0]]], dtype=float)
        raise AssertionError("unexpected mode")

    monkeypatch.setattr(parser_misc, "get_rate_tensor", fake_get_rate_tensor)
    monkeypatch.setattr(parser_misc.np, "savetxt", lambda *args, **kwargs: None)

    g = {
        "infile_type": "iqtree",
        "omegaC_method": "submodel",
        "float_tol": 1e-12,
    }
    with pytest.raises(AssertionError, match="Sum of rates did not match"):
        parser_misc.read_input(g)
