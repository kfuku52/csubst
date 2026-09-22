import numpy as np
import pandas as pd
import pytest

from csubst import main_sites
from csubst import ete


def test_get_state_rejects_missing_leaf_sequence():
    node = ete.PhyloNode("A;", format=1)
    with pytest.raises(AssertionError, match="Leaf sequence not found"):
        main_sites.get_state(node=node, g={})


def test_get_state_rejects_non_codon_length_sequence():
    node = ete.PhyloNode("A;", format=1)
    ete.set_prop(node, "sequence", "AT")
    with pytest.raises(AssertionError, match="multiple of 3"):
        main_sites.get_state(node=node, g={})


def test_get_gapsite_rate_matches_manual_count():
    state_tensor = np.array(
        [
            [[1, 0], [0, 0], [1, 0]],
            [[0, 1], [0, 0], [0, 1]],
            [[1, 0], [1, 0], [0, 0]],
        ],
        dtype=float,
    )
    # site-wise gap rates: [0/3, 2/3, 1/3]
    out = main_sites.get_gapsite_rate(state_tensor)
    np.testing.assert_allclose(out, [0.0, 2.0 / 3.0, 1.0 / 3.0], atol=1e-12)


def test_get_gapsite_rate_returns_zero_when_branch_axis_is_empty():
    state_tensor = np.zeros((0, 3, 2), dtype=float)
    out = main_sites.get_gapsite_rate(state_tensor)
    np.testing.assert_allclose(out, [0.0, 0.0, 0.0], atol=1e-12)


def test_extend_site_index_edge_fills_missing_edges():
    sites = pd.Series([2, 3, 7], dtype=int)
    out = main_sites.extend_site_index_edge(sites, num_extend=2)
    # Gap between 3 and 7 is filled by 5 and 6.
    assert out.tolist() == [2, 3, 5, 6, 7]


def test_initialize_site_df_columns_are_correct():
    df = main_sites.initialize_site_df(4)
    assert df["codon_site_alignment"].tolist() == [0, 1, 2, 3]
    assert df["nuc_site_alignment"].tolist() == [1, 4, 7, 10]


def test_remap_codon_site_columns_to_alignment_uses_site_index_mapping():
    df = pd.DataFrame(
        {
            "codon_site_alignment": [0, 1],
            "codon_site_geneA": [1, -1],
            "other": [10, 20],
        }
    )
    g = {
        "state_cdn": np.zeros((1, 2, 2), dtype=float),
        "site_index_alignment": np.array([5, 9], dtype=np.int64),
    }
    out = main_sites.remap_codon_site_columns_to_alignment(df=df, g=g)
    assert out["codon_site_alignment"].tolist() == [5, 9]
    assert out["codon_site_geneA"].tolist() == [9, -1]
    assert out["other"].tolist() == [10, 20]


def test_get_leaf_state_letter_and_gap_checks_use_alignment_site_mapping():
    g = {
        "state_pep": np.array([[[0.0, 1.0], [0.0, 0.0]]], dtype=float),
        "state_cdn": np.zeros((1, 2, 2), dtype=float),
        "amino_acid_orders": np.array(["A", "V"], dtype=object),
        "site_index_alignment": np.array([5, 9], dtype=np.int64),
    }
    assert main_sites.get_leaf_state_letter(g=g, leaf_id=0, codon_site_alignment=6) == "V"
    assert main_sites.get_leaf_state_letter(g=g, leaf_id=0, codon_site_alignment=10) == ""
    assert main_sites._is_branch_site_gap(g=g, branch_id=0, codon_site_alignment=6) is False
    assert main_sites._is_branch_site_gap(g=g, branch_id=0, codon_site_alignment=10) is True


def test_combinatorial2single_columns_removes_combination_columns():
    df = pd.DataFrame(
        {
            "OCSany2any": [1],
            "OCSany2spe": [2],
            "OCNspe2dif": [3],
            "kept": [4],
        }
    )
    out = main_sites.combinatorial2single_columns(df)
    assert list(out.columns) == ["kept"]


@pytest.mark.parametrize("mode_name,expected", [("intersection", True), ("lineage", False), ("set", False)])
def test_mode_helpers_only_enable_optional_outputs_in_intersection(mode_name, expected):
    g = {"mode": mode_name}
    assert main_sites.should_plot_state(g) == expected
    assert main_sites.should_save_pymol_views(g) == expected


def test_get_yvalues_for_supported_modes():
    df = pd.DataFrame(
        {
            "S_sub": [0.0, 0.5],
            "N_sub": [1.0, 2.0],
            "S_sub_1": [0.0, 0.3],
            "S_sub_2": [0.0, 0.4],
            "N_sub_1": [0.7, 0.1],
            "N_sub_2": [0.3, 0.2],
            "OCNany2spe": [0.2, 0.4],
            "OCSany2spe": [0.1, 0.2],
        }
    )
    np.testing.assert_allclose(main_sites.get_yvalues(df, "_sub", "S"), [0.0, 2.5], atol=1e-12)
    np.testing.assert_allclose(main_sites.get_yvalues(df, "_sub", "N"), [1.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(main_sites.get_yvalues(df, "_sub_", "S"), [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(main_sites.get_yvalues(df, "any2spe", "S"), [0.3, 0.6], atol=1e-12)
    np.testing.assert_allclose(main_sites.get_yvalues(df, "any2spe", "N"), [0.2, 0.4], atol=1e-12)


def test_get_yvalues_for_lineage_branch_specific_rows():
    df = pd.DataFrame(
        {
            "S_sub_13": [0.0, 0.6],
            "N_sub_13": [0.2, 0.3],
            "S_sub_12": [0.2, 0.0],
            "N_sub_12": [0.4, 0.5],
        }
    )
    np.testing.assert_allclose(main_sites.get_yvalues(df, "_sub_branch_13", "N"), [0.2, 0.3], atol=1e-12)
    np.testing.assert_allclose(main_sites.get_yvalues(df, "_sub_branch_13", "S"), [0.0, 0.9], atol=1e-12)
    np.testing.assert_allclose(main_sites.get_yvalues(df, "_sub_branch_12", "N"), [0.4, 0.5], atol=1e-12)
    np.testing.assert_allclose(main_sites.get_yvalues(df, "_sub_branch_12", "S"), [0.6, 0.0], atol=1e-12)


def test_get_set_expression_display_branch_ids_accepts_scalar_branch_id():
    g = {"mode_expression": "117|48", "branch_ids": np.int64(117)}
    out = main_sites._get_set_expression_display_branch_ids(g)
    assert out.tolist() == [117]


def test_get_yvalues_set_expression_prefers_probability_column():
    df = pd.DataFrame({"N_set_expr_prob": [0.0, 1.7, 2.0], "N_set_expr": [False, True, True]})
    np.testing.assert_allclose(main_sites.get_yvalues(df, "_set_expr", "N"), [0.0, 1.7, 2.0], atol=1e-12)
    np.testing.assert_allclose(main_sites.get_yvalues(df, "_set_expr", "S"), [0.0, 0.0, 0.0], atol=1e-12)


def test_get_yvalues_set_other_uses_other_prob_columns():
    df = pd.DataFrame({"N_set_other": [False, True], "N_set_other_prob": [0.2, 0.4], "S_set_other_prob": [0.1, 0.0]})
    np.testing.assert_allclose(main_sites.get_yvalues(df, "_set_other", "N"), [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(main_sites.get_yvalues(df, "_set_other", "S"), [0.0, 0.0], atol=1e-12)


def test_get_set_expression_channel_labels_spe():
    prob = np.array(
        [
            [0.0, 0.8, 0.2],
            [0.7, 0.1, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    out = main_sites.get_set_expression_channel_labels(
        prob_matrix=prob,
        set_stat_type="spe",
        state_orders=np.array(["A", "V", "T"]),
    )
    assert out.tolist() == ["X→V", "X→A", ""]


def test_get_set_heatmap_column_labels_uses_set_expr_channel_index():
    df = pd.DataFrame(
        {
            "codon_site_alignment": [2, 3, 4],
            "N_set_expr_channel_index": [0, -1, 1],
        }
    )
    display_meta = [{"site": 2}, {"site": None}, {"site": 3}, {"site": 4}]
    g = {"mode": "set", "set_stat_type": "spe", "amino_acid_orders": np.array(["A", "V"])}
    out = main_sites.get_set_heatmap_column_labels(df=df, display_meta=display_meta, g=g)
    assert out == {2: "X→A", 4: "X→V"}
