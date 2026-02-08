import numpy
import pandas
import pytest

from csubst import parser_misc
from csubst import table
from csubst import ete


def test_sort_branch_ids_sorts_within_rows_and_by_row():
    df = pandas.DataFrame(
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


def test_sort_cb_stats_handles_non_string_column_names_regression():
    # Regression target inspired by issue #74.
    cb_stats = pandas.DataFrame(
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


def test_set_substitution_dtype_casts_integral_columns_only():
    df = pandas.DataFrame(
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
    cb = pandas.DataFrame(
        {
            "OCSany2any": [1.0, 2.0],
            "OCSany2spe": [2.0, 4.0],
            "OCNany2any": [1.0, 2.0],
            "OCNany2spe": [1.0, 1.0],
        }
    )
    out = table.get_linear_regression(cb)
    numpy.testing.assert_allclose(out["OCS_linreg_residual"].to_numpy(), [0.0, 0.0], atol=1e-12)
    # coef = (1*1 + 2*1) / (1^2 + 2^2) = 3/5 = 0.6
    numpy.testing.assert_allclose(out["OCN_linreg_residual"].to_numpy(), [0.4, -0.2], atol=1e-12)


def test_get_cutoff_stat_bool_array_parses_compound_expression():
    cb = pandas.DataFrame(
        {
            "OCNany2spe": [1.9, 2.0, 2.1],
            "omegaCany2spe": [10.0, 4.9, 5.0],
        }
    )
    out = table.get_cutoff_stat_bool_array(cb, "OCNany2spe,2.0|omegaCany2spe,5.0")
    assert out.tolist() == [False, False, True]


def test_get_cutoff_stat_bool_array_exits_on_unknown_column():
    cb = pandas.DataFrame({"OCNany2spe": [1.0]})
    with pytest.raises(SystemExit):
        table.get_cutoff_stat_bool_array(cb, "DOES_NOT_EXIST,1.0")


def test_fill_instantaneous_rate_matrix_diagonal_sets_row_sums_to_zero():
    inst = numpy.array([[0.0, 1.0], [2.0, 0.0]], dtype=float)
    out = parser_misc.fill_instantaneous_rate_matrix_diagonal(inst)
    numpy.testing.assert_allclose(out, numpy.array([[-1.0, 1.0], [2.0, -2.0]]), atol=1e-12)
    numpy.testing.assert_allclose(out.sum(axis=1), [0.0, 0.0], atol=1e-12)


def test_scale_instantaneous_rate_matrix_matches_manual_scaling():
    inst = numpy.array([[0.0, 2.0], [1.0, 0.0]], dtype=float)
    eq = numpy.array([0.4, 0.6], dtype=float)
    out = parser_misc.scale_instantaneous_rate_matrix(inst.copy(), eq)
    expected = numpy.array([[0.0, 2.0 / 1.4], [1.0 / 1.4, 0.0]])
    numpy.testing.assert_allclose(out, expected, atol=1e-12)


def test_scale_instantaneous_rate_matrix_requires_zero_diagonal():
    inst = numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    with pytest.raises(AssertionError, match="Diagonal elements"):
        parser_misc.scale_instantaneous_rate_matrix(inst, numpy.array([0.5, 0.5]))


def test_exchangeability2Q_matches_manual_result():
    ex = numpy.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    eq = numpy.array([0.25, 0.75], dtype=float)
    out = parser_misc.exchangeability2Q(ex, eq)
    expected = numpy.array([[-2.0, 2.0], [2.0 / 3.0, -2.0 / 3.0]])
    numpy.testing.assert_allclose(out, expected, atol=1e-12)


def test_get_equilibrium_frequency_for_codon_and_amino_acid_modes():
    g = {
        "equilibrium_frequency": numpy.array([0.2, 0.3, 0.5]),
        "amino_acid_orders": numpy.array(["K", "N"]),
        "synonymous_indices": {"K": [0, 1], "N": [2]},
        "float_tol": 1e-12,
    }
    eq_cdn = parser_misc.get_equilibrium_frequency(g, "cdn")
    eq_pep = parser_misc.get_equilibrium_frequency(g, "pep")
    numpy.testing.assert_allclose(eq_cdn, [0.2, 0.3, 0.5], atol=1e-12)
    numpy.testing.assert_allclose(eq_pep, [0.5, 0.5], atol=1e-12)


def test_get_codon_order_index_reorders_positions():
    order_from = numpy.array(["AAA", "AAC", "AAG"])
    order_to = numpy.array(["AAG", "AAA", "AAC"])
    out = parser_misc.get_codon_order_index(order_from, order_to)
    numpy.testing.assert_array_equal(out, [1, 2, 0])


def test_get_exchangeability_codon_order_shape_and_no_stops():
    codons = parser_misc.get_exchangeability_codon_order()
    assert codons.shape == (61,)
    assert set(["TAA", "TAG", "TGA"]).isdisjoint(set(codons))


def test_get_rate_tensor_for_asis_and_syn_modes():
    inst = numpy.array(
        [[-1.0, 0.2, 0.8], [0.3, -0.7, 0.4], [0.5, 0.6, -1.1]],
        dtype=float,
    )
    g = {
        "amino_acid_orders": numpy.array(["K", "N"]),
        "synonymous_indices": {"K": [0, 1], "N": [2]},
        "max_synonymous_size": 2,
        "float_type": numpy.float64,
    }
    asis = parser_misc.get_rate_tensor(inst, "asis", g)
    syn = parser_misc.get_rate_tensor(inst, "syn", g)
    numpy.testing.assert_allclose(
        asis,
        numpy.array([[[0.0, 0.2, 0.8], [0.3, 0.0, 0.4], [0.5, 0.6, 0.0]]]),
        atol=1e-12,
    )
    numpy.testing.assert_allclose(
        syn,
        numpy.array([[[0.0, 0.2], [0.3, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
        atol=1e-12,
    )


def test_cdn2pep_matrix_matches_manual_group_sum():
    inst_cdn = numpy.array(
        [[-1.0, 0.2, 0.8], [0.3, -0.7, 0.4], [0.5, 0.6, -1.1]],
        dtype=float,
    )
    g = {
        "amino_acid_orders": numpy.array(["K", "N"]),
        "synonymous_indices": {"K": [0, 1], "N": [2]},
    }
    out = parser_misc.cdn2pep_matrix(inst_cdn, g)
    numpy.testing.assert_allclose(out, numpy.array([[-1.2, 1.2], [1.1, -1.1]]), atol=1e-12)


def test_get_mechanistic_instantaneous_rate_matrix_matches_manual_example():
    g = {
        "codon_orders": numpy.array(["AAA", "AAG", "AAC"]),
        "amino_acid_orders": numpy.array(["K", "N"]),
        "synonymous_indices": {"K": [0, 1], "N": [2]},
        "omega": 2.0,
        "kappa": 3.0,
        "equilibrium_frequency": numpy.array([0.2, 0.3, 0.5]),
        "float_type": numpy.float64,
    }
    out = parser_misc.get_mechanistic_instantaneous_rate_matrix(g)
    expected = numpy.array(
        [
            [-1.160714285714286, 0.267857142857143, 0.892857142857143],
            [0.178571428571429, -1.071428571428571, 0.892857142857143],
            [0.357142857142857, 0.535714285714286, -0.892857142857143],
        ]
    )
    numpy.testing.assert_allclose(out, expected, atol=1e-12)
    numpy.testing.assert_allclose(out.sum(axis=1), [0.0, 0.0, 0.0], atol=1e-12)


def test_annotate_tree_handles_none_root_dist(tmp_path):
    rooted_tree_file = tmp_path / "toy_rooted.nwk"
    iqtree_tree_file = tmp_path / "toy_iqtree.treefile"
    rooted_tree_file.write_text("(A:1,B:1)R;\n", encoding="utf-8")
    iqtree_tree_file.write_text("(A:1,B:1)R;\n", encoding="utf-8")

    g = {
        "iqtree_treefile": str(iqtree_tree_file),
        "rooted_tree": ete.PhyloNode(str(rooted_tree_file), format=1),
    }
    out = parser_misc.annotate_tree(g)
    assert "tree" in out
    assert len(list(out["tree"].traverse())) == 3
