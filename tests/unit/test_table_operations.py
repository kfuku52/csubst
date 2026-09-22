import numpy as np
import pandas as pd
import pytest

from csubst import parser_misc
from csubst import table


def test_generate_intermediate_files_explicit_paths_do_not_require_manifest(monkeypatch):
    g = {
        "infile_type": "iqtree",
        "iqtree_redo": False,
        "iqtree_model": "MG",
        "iqtree_iqtree": "input.iqtree",
        "iqtree_log": "input.log",
        "iqtree_rate": "input.rate",
        "iqtree_state": "input.state",
        "iqtree_treefile": "input.treefile",
    }
    monkeypatch.setattr(
        parser_misc.parser_iqtree,
        "check_intermediate_files",
        lambda value: (value, True),
    )
    monkeypatch.setattr(
        parser_misc.parser_iqtree,
        "check_iqtree_dependency",
        lambda value: (_ for _ in ()).throw(AssertionError("dependency check")),
    )
    monkeypatch.setattr(
        parser_misc.parser_iqtree,
        "is_iqtree_manifest_compatible",
        lambda value: (_ for _ in ()).throw(AssertionError("manifest check")),
    )

    def fake_read_iqtree(value, eq=False):
        assert eq is False
        value["substitution_model"] = "MG"
        return value

    monkeypatch.setattr(parser_misc.parser_iqtree, "read_iqtree", fake_read_iqtree)
    monkeypatch.setattr(
        parser_misc.parser_iqtree,
        "run_iqtree_ancestral",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("IQ-TREE rerun")),
    )

    out = parser_misc.generate_intermediate_files(g)

    assert out["substitution_model"] is None


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


def test_sort_branch_ids_normalizes_strings_before_numeric_sorting():
    df = pd.DataFrame({
        "branch_id_1": ["10", "2", "2"],
        "branch_id_2": [2, 10, 3],
        "site": ["10", "2", "1"],
        "value": [100, 200, 300],
    })
    out = table.sort_branch_ids(df)
    assert out[["branch_id_1", "branch_id_2", "site"]].values.tolist() == [
        [2, 3, 1], [2, 10, 2], [2, 10, 10],
    ]
    assert out["value"].tolist() == [300, 200, 100]


@pytest.mark.parametrize("value", ["9007199254740993", "9223372036854775807.0"])
def test_integer_like_strings_preserve_exact_values(value):
    out = table.sort_branch_ids(pd.DataFrame({"branch_id_1": [value]}))
    assert out["branch_id_1"].iloc[0] == int(value.split(".")[0])


@pytest.mark.parametrize("values", [
    np.array([2**63], dtype=np.uint64),
    np.array([float(2**63)]),
    np.array(["9223372036854775808"]),
])
def test_sort_branch_ids_rejects_int64_overflow(values):
    with pytest.raises(ValueError, match="int64"):
        table.sort_branch_ids(pd.DataFrame({"branch_id_1": values}))


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


@pytest.mark.parametrize("values", [[1.0, np.nan], [np.nan, np.nan], [1.0, np.inf], [float(2**63), 0.0]])
def test_set_substitution_dtype_preserves_nonrepresentable_values(values):
    df = pd.DataFrame({"S_sub": values})
    out = table.set_substitution_dtype(df.copy())
    pd.testing.assert_frame_equal(out, df)


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


def test_parse_cutoff_stat_supports_regex_with_comma_quantifier():
    out = table.parse_cutoff_stat(r"omegaC.{1,2},5.0")
    assert out == [(r"omegaC.{1,2}", 5.0)]


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
