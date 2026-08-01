import numpy as np
import pandas as pd
import pytest

from csubst import main_sites
from csubst import tree
from csubst import ete


def test_get_state_orders():
    g = {"amino_acid_orders": np.array(["A", "B"]), "matrix_groups": {"grp": ["AA", "AB"]}}
    orders_nsy, keys_nsy = main_sites.get_state_orders(g, "nsy")
    assert keys_nsy == ["nsy"]
    assert list(orders_nsy["nsy"]) == ["A", "B"]
    orders_syn, keys_syn = main_sites.get_state_orders(g, "syn")
    assert keys_syn == ["grp"]
    assert orders_syn["grp"] == ["AA", "AB"]


def test_add_gapline_empty_df_is_noop():
    df = pd.DataFrame({"codon_site_alignment": [], "gap_rate_all": []})
    fig, ax = main_sites.plt.subplots()
    main_sites.add_gapline(df=df, gapcol="gap_rate_all", xcol="codon_site_alignment", yvalue=0.5, lw=1, ax=ax)
    assert len(ax.collections) == 0
    main_sites.plt.close(fig)


def test_classify_tree_site_categories_prefers_larger_signal():
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4],
            "OCNany2spe": [0.6, 0.1, 0.4, 0.8],
            "OCNany2dif": [0.1, 0.7, 0.4, 0.9],
        }
    )
    g = {
        "single_branch_mode": False,
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
    }
    out, min_prob = main_sites.classify_tree_site_categories(df=df, g=g)
    assert pytest.approx(min_prob, abs=1e-12) == 0.5
    assert out["tree_site_category"].tolist() == ["convergent", "divergent", "blank", "divergent"]


def test_get_tree_site_display_sites_respects_max_sites_when_one_and_both_categories_exist():
    tree_site_df = pd.DataFrame(
        {
            "codon_site_alignment": [11, 22],
            "convergent_score": [0.91, 0.10],
            "divergent_score": [0.10, 0.95],
            "tree_site_category": ["convergent", "divergent"],
        }
    )
    g = {"tree_site_plot_max_sites": 1}

    display_meta = main_sites.get_tree_site_display_sites(tree_site_df=tree_site_df, g=g)
    site_rows = [row for row in display_meta if row["site"] is not None]

    assert len(site_rows) == 1
    assert site_rows[0]["site"] == 22
    assert site_rows[0]["category"] == "divergent"
    assert all(row["category"] != "separator" for row in display_meta)


def test_get_tree_site_display_sites_refills_capacity_from_other_category():
    tree_site_df = pd.DataFrame(
        {
            "codon_site_alignment": [10, 20, 21, 22, 23, 24],
            "convergent_score": [0.99, 0.10, 0.09, 0.08, 0.07, 0.06],
            "divergent_score": [0.01, 0.95, 0.94, 0.93, 0.92, 0.91],
            "tree_site_category": ["convergent", "divergent", "divergent", "divergent", "divergent", "divergent"],
        }
    )
    g = {"tree_site_plot_max_sites": 4}

    display_meta = main_sites.get_tree_site_display_sites(tree_site_df=tree_site_df, g=g)
    site_rows = [row for row in display_meta if row["site"] is not None]
    conv_count = sum(row["category"] == "convergent" for row in site_rows)
    div_count = sum(row["category"] == "divergent" for row in site_rows)

    assert len(site_rows) == 4
    assert conv_count == 1
    assert div_count == 3


def test_get_tree_site_display_sites_intersection_sorts_within_each_category():
    tree_site_df = pd.DataFrame(
        {
            "codon_site_alignment": [90, 50, 80, 20],
            "convergent_score": [0.99, 0.98, 0.20, 0.10],
            "divergent_score": [0.01, 0.02, 0.97, 0.96],
            "tree_site_category": ["convergent", "convergent", "divergent", "divergent"],
        }
    )
    g = {"tree_site_plot_max_sites": 4}
    out = main_sites.get_tree_site_display_sites(tree_site_df=tree_site_df, g=g)
    assert out == [
        {"site": 50, "category": "convergent"},
        {"site": 90, "category": "convergent"},
        {"site": None, "category": "separator"},
        {"site": 20, "category": "divergent"},
        {"site": 80, "category": "divergent"},
    ]


def test_get_tree_site_display_sites_respects_max_sites_when_one():
    tree_site_df = pd.DataFrame(
        {
            "codon_site_alignment": [10, 20, 30, 40],
            "convergent_score": [0.8, 0.7, 0.0, 0.0],
            "divergent_score": [0.0, 0.0, 0.9, 0.6],
            "tree_site_category": ["convergent", "convergent", "divergent", "divergent"],
        }
    )
    g = {"tree_site_plot_max_sites": 1}
    out = main_sites.get_tree_site_display_sites(tree_site_df=tree_site_df, g=g)
    plotted = [item for item in out if item["site"] is not None]
    assert len(plotted) == 1
    assert plotted[0]["site"] == 30
    assert plotted[0]["category"] == "divergent"


def test_get_tree_site_display_sites_lineage_includes_sites_above_min_prob():
    tree_site_df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4, 5],
            "convergent_score": [0.0, 0.0, 0.0, 0.0, 0.0],
            "divergent_score": [0.0, 0.0, 0.0, 0.0, 0.0],
            "tree_site_category": ["blank", "blank", "blank", "blank", "blank"],
        }
    )
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4, 5],
            "N_sub_10": [0.00, 0.70, 0.82, 0.00, 0.10],
            "N_sub_11": [0.00, 0.00, 0.30, 0.00, 0.81],
        }
    )
    g = {
        "mode": "lineage",
        "branch_ids": np.array([10, 11], dtype=np.int64),
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "tree_site_plot_max_sites": 10,
        "single_branch_mode": False,
    }
    out = main_sites.get_tree_site_display_sites(tree_site_df=tree_site_df, g=g, df=df)
    plotted = [item for item in out if item["site"] is not None]
    assert [item["site"] for item in plotted] == [3, 5]
    assert set(item["category"] for item in plotted) == {"lineage"}


def test_get_tree_site_display_sites_lineage_includes_sites_equal_to_min_prob():
    tree_site_df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3],
            "convergent_score": [0.0, 0.0, 0.0],
            "divergent_score": [0.0, 0.0, 0.0],
            "tree_site_category": ["blank", "blank", "blank"],
        }
    )
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3],
            "N_sub_10": [0.80, 0.79, 0.10],
            "N_sub_11": [0.00, 0.00, 0.85],
        }
    )
    g = {
        "mode": "lineage",
        "branch_ids": np.array([10, 11], dtype=np.int64),
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "tree_site_plot_max_sites": 10,
        "single_branch_mode": False,
    }
    out = main_sites.get_tree_site_display_sites(tree_site_df=tree_site_df, g=g, df=df)
    plotted = [item for item in out if item["site"] is not None]
    assert [item["site"] for item in plotted] == [1, 3]
    assert set(item["category"] for item in plotted) == {"lineage"}


def test_get_tree_site_display_sites_lineage_respects_max_sites_by_prob():
    tree_site_df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4, 5],
            "convergent_score": [0.0, 0.0, 0.0, 0.0, 0.0],
            "divergent_score": [0.0, 0.0, 0.0, 0.0, 0.0],
            "tree_site_category": ["blank", "blank", "blank", "blank", "blank"],
        }
    )
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4, 5],
            "N_sub_10": [0.00, 0.70, 0.81, 0.00, 0.10],
            "N_sub_11": [0.00, 0.00, 0.30, 0.00, 0.95],
        }
    )
    g = {
        "mode": "lineage",
        "branch_ids": np.array([10, 11], dtype=np.int64),
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "tree_site_plot_max_sites": 1,
        "single_branch_mode": False,
    }
    out = main_sites.get_tree_site_display_sites(tree_site_df=tree_site_df, g=g, df=df)
    plotted = [item for item in out if item["site"] is not None]
    assert [item["site"] for item in plotted] == [5]
    assert set(item["category"] for item in plotted) == {"lineage"}


def test_get_tree_site_display_sites_lineage_returns_empty_when_no_foreground_substitutions():
    tree_site_df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3],
            "convergent_score": [0.9, 0.8, 0.7],
            "divergent_score": [0.1, 0.2, 0.3],
            "tree_site_category": ["convergent", "convergent", "divergent"],
        }
    )
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3],
            "N_sub_10": [0.70, 0.60, 0.20],
            "N_sub_11": [0.10, 0.75, 0.30],
        }
    )
    g = {
        "mode": "lineage",
        "branch_ids": np.array([10, 11], dtype=np.int64),
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "tree_site_plot_max_sites": 60,
        "single_branch_mode": False,
    }
    out = main_sites.get_tree_site_display_sites(tree_site_df=tree_site_df, g=g, df=df)
    assert out == []


def test_get_tree_site_display_sites_lineage_accepts_scalar_branch_id():
    tree_site_df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3],
            "convergent_score": [0.0, 0.0, 0.0],
            "divergent_score": [0.0, 0.0, 0.0],
            "tree_site_category": ["blank", "blank", "blank"],
        }
    )
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3],
            "N_sub_10": [0.2, 0.85, 0.1],
        }
    )
    g = {
        "mode": "lineage",
        "branch_ids": np.int64(10),
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "tree_site_plot_max_sites": 10,
        "single_branch_mode": False,
    }
    out = main_sites.get_tree_site_display_sites(tree_site_df=tree_site_df, g=g, df=df)
    plotted = [item for item in out if item["site"] is not None]
    assert [item["site"] for item in plotted] == [2]
    assert set(item["category"] for item in plotted) == {"lineage"}


def test_get_tree_site_display_sites_set_uses_set_expression_columns():
    tree_site_df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4, 5],
            "convergent_score": [0.95, 0.90, 0.85, 0.80, 0.75],
            "divergent_score": [0.10, 0.10, 0.10, 0.10, 0.10],
            "tree_site_category": ["convergent", "convergent", "convergent", "convergent", "convergent"],
        }
    )
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4, 5],
            "N_set_expr": [False, True, False, True, True],
            "N_set_expr_prob": [0.0, 0.95, 0.0, 0.82, 0.91],
        }
    )
    g = {
        "mode": "set",
        "set_stat_type": "any",
        "min_combinat_prob": 0.5,
        "tree_site_plot_max_sites": 2,
        "single_branch_mode": False,
    }
    out = main_sites.get_tree_site_display_sites(tree_site_df=tree_site_df, g=g, df=df)
    plotted = [item for item in out if item["site"] is not None]
    # Top-2 by N_set_expr_prob: sites 2 (0.95) and 5 (0.91).
    assert [item["site"] for item in plotted] == [2, 5]
    assert set(item["category"] for item in plotted) == {"set"}


def test_get_tree_site_overflow_count_set_uses_set_expression_candidates():
    tree_site_df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4, 5],
            "convergent_score": [0.95, 0.90, 0.85, 0.80, 0.75],
            "divergent_score": [0.10, 0.10, 0.10, 0.10, 0.10],
            "tree_site_category": ["convergent", "convergent", "convergent", "convergent", "convergent"],
        }
    )
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4, 5],
            "N_set_expr": [False, True, False, True, True],
            "N_set_expr_prob": [0.0, 0.95, 0.0, 0.82, 0.91],
        }
    )
    g = {
        "mode": "set",
        "set_stat_type": "any",
        "min_combinat_prob": 0.5,
        "tree_site_plot_max_sites": 2,
        "single_branch_mode": False,
    }
    display_meta = [{"site": 2, "category": "set"}, {"site": 5, "category": "set"}]
    overflow = main_sites.get_tree_site_overflow_count(tree_site_df=tree_site_df, display_meta=display_meta, g=g, df=df)
    assert overflow == 1


def test_get_tree_site_overflow_label_y_is_half_row_below_alignment():
    y = main_sites.get_tree_site_overflow_label_y(
        num_alignment_rows=3,
        has_structure_track=False,
        structure_row_y=None,
        gap_rows=0.5,
    )
    assert y == pytest.approx(3.0)


def test_get_tree_site_overflow_label_y_with_structure_is_below_structure_row():
    y = main_sites.get_tree_site_overflow_label_y(
        num_alignment_rows=3,
        has_structure_track=True,
        structure_row_y=3.5,
        gap_rows=0.5,
    )
    assert y == pytest.approx(4.5)


def test_get_lineage_site_branch_ids_lists_all_nonzero_branches():
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4],
            "N_sub_10": [0.80, 0.20, 0.70, 0.00],
            "N_sub_11": [0.00, 0.70, 0.60, 0.00],
        }
    )
    display_meta = [
        {"site": 1, "category": "lineage"},
        {"site": None, "category": "separator"},
        {"site": 2, "category": "lineage"},
        {"site": 3, "category": "lineage"},
        {"site": 4, "category": "lineage"},
    ]
    g = {
        "mode": "lineage",
        "branch_ids": np.array([10, 11], dtype=np.int64),
    }
    out = main_sites.get_lineage_site_branch_ids(
        df=df,
        display_meta=display_meta,
        g=g,
        min_prob=0.5,
    )
    assert out[1] == [10]
    assert out[2] == [11]
    assert out[3] == [10, 11]
    assert 4 not in out


def test_get_lineage_site_branch_ids_includes_values_equal_to_min_prob():
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2],
            "N_sub_10": [0.80, 0.79],
            "N_sub_11": [0.00, 0.80],
        }
    )
    display_meta = [
        {"site": 1, "category": "lineage"},
        {"site": 2, "category": "lineage"},
    ]
    g = {
        "mode": "lineage",
        "branch_ids": np.array([10, 11], dtype=np.int64),
    }
    out = main_sites.get_lineage_site_branch_ids(
        df=df,
        display_meta=display_meta,
        g=g,
        min_prob=0.8,
    )
    assert out[1] == [10]
    assert out[2] == [11]


def test_get_lineage_site_branch_ids_accepts_scalar_branch_id():
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2],
            "N_sub_10": [0.81, 0.10],
        }
    )
    display_meta = [
        {"site": 1, "category": "lineage"},
        {"site": 2, "category": "lineage"},
    ]
    g = {
        "mode": "lineage",
        "branch_ids": np.int64(10),
    }
    out = main_sites.get_lineage_site_branch_ids(
        df=df,
        display_meta=display_meta,
        g=g,
        min_prob=0.8,
    )
    assert out == {1: [10]}


def test_get_tree_plot_coordinates_returns_expected_root_and_leaf_positions(tiny_tree):
    xcoord, ycoord, leaf_order = main_sites.get_tree_plot_coordinates(tiny_tree)
    root = ete.get_tree_root(tiny_tree)
    root_id = ete.get_prop(root, "numerical_label")
    all_ids = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()]
    assert set(xcoord.keys()) == set(all_ids)
    assert set(ycoord.keys()) == set(all_ids)
    assert pytest.approx(float(xcoord[root_id]), abs=1e-12) == 0.0
    leaf_ids = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse() if ete.is_leaf(n)]
    leaf_y = [ycoord[i] for i in leaf_ids]
    assert len(set(leaf_y)) == len(leaf_ids)
    assert set(leaf_order) == set(leaf_ids)


def test_get_highlight_leaf_and_branch_ids_marks_descendant_leaves_for_internal_targets(tiny_tree):
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tiny_tree.traverse()}

    leaf_ids, branch_ids = main_sites.get_highlight_leaf_and_branch_ids(
        tree=tiny_tree,
        branch_ids={labels["X"]},
    )
    assert branch_ids == {labels["X"]}
    assert leaf_ids == {labels["A"], labels["C"]}

    leaf_ids, branch_ids = main_sites.get_highlight_leaf_and_branch_ids(
        tree=tiny_tree,
        branch_ids={labels["A"], labels["X"]},
    )
    assert branch_ids == {labels["A"], labels["X"]}
    assert leaf_ids == {labels["A"], labels["C"]}


def test_get_species_overlap_node_types_classifies_speciation_and_duplication():
    tr = ete.PhyloNode(
        "((Homo_sapiens_gene1:1,Homo_sapiens_gene2:1)Dup:1,(Mus_musculus_gene1:1,Rattus_norvegicus_gene1:1)Spec:1)Root;",
        format=1,
    )
    tr = tree.add_numerical_node_labels(tr)
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tr.traverse()}
    out = main_sites.get_species_overlap_node_types(
        tree=tr,
        species_regex=r"^([^_]+_[^_]+)_",
    )
    assert out[labels["Dup"]] == "duplication"
    assert out[labels["Spec"]] == "speciation"
    assert out[labels["Root"]] == "speciation"


def test_get_species_overlap_node_types_returns_empty_without_regex(tiny_tree):
    out = main_sites.get_species_overlap_node_types(tree=tiny_tree, species_regex="")
    assert out == {}


def test_get_species_overlap_node_types_auto_requires_all_tip_labels():
    tr = ete.PhyloNode(
        "((Homo_sapiens_gene1:1,Homo_sapiens_gene2:1)Dup:1,(BADLABEL:1,Rattus_norvegicus_gene1:1)Spec:1)Root;",
        format=1,
    )
    tr = tree.add_numerical_node_labels(tr)
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tr.traverse()}
    out_auto = main_sites.get_species_overlap_node_types(
        tree=tr,
        species_regex=r"^([^_]+_[^_]+)_",
        require_all_tip_labels=True,
    )
    out_yes = main_sites.get_species_overlap_node_types(
        tree=tr,
        species_regex=r"^([^_]+_[^_]+)_",
        require_all_tip_labels=False,
    )
    assert out_auto == {}
    assert out_yes[labels["Dup"]] == "duplication"
