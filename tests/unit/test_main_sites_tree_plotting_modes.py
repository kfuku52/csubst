import re
import numpy as np
import pandas as pd
import pytest

from csubst import main_sites
from csubst import ete


def test_plot_tree_site_lineage_svg_uses_branch_palette_and_plots_all_threshold_sites(tmp_path, tiny_tree):
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tiny_tree.traverse()}
    branch_ids = [labels["X"], labels["C"]]
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    aa_orders = np.array(["A", "V", "T", "I"])
    state_pep = np.zeros((num_node, 5, aa_orders.shape[0]), dtype=float)
    for leaf_name in ("A", "B", "C"):
        leaf_id = labels[leaf_name]
        state_pep[leaf_id, 0, 0] = 1.0
        state_pep[leaf_id, 1, 1] = 1.0
        state_pep[leaf_id, 2, 2] = 1.0
        state_pep[leaf_id, 3, 3] = 1.0
        state_pep[leaf_id, 4, 0] = 1.0

    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4, 5],
            "OCNany2spe": [0.0, 0.0, 0.0, 0.0, 0.0],
            "OCNany2dif": [0.0, 0.0, 0.0, 0.0, 0.0],
            "N_sub_{}".format(labels["X"]): [0.60, 0.20, 0.85, 0.00, 0.00],
            "N_sub_{}".format(labels["C"]): [0.00, 0.81, 0.70, 0.00, 0.51],
        }
    )
    g = {
        "tree": tiny_tree,
        "mode": "lineage",
        "branch_ids": np.array(branch_ids, dtype=np.int64),
        "single_branch_mode": False,
        "tree_site_plot": True,
        "tree_site_plot_format": "svg",
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "tree_site_plot_max_sites": 10,
        "site_outdir": str(tmp_path),
        "float_format": "%.4f",
        "state_pep": state_pep,
        "amino_acid_orders": aa_orders,
    }
    out_paths = main_sites.plot_tree_site(df=df, g=g)
    svg_path = tmp_path / "csubst_sites.tree_site.svg"
    table_path = tmp_path / "csubst_sites.tree_site.tsv"
    assert str(svg_path) in out_paths
    assert table_path.exists()

    plotted_df = pd.read_csv(table_path, sep="\t")
    plotted_sites = plotted_df.loc[plotted_df["is_plotted"], "codon_site_alignment"].astype(int).tolist()
    assert plotted_sites == [2, 3]

    branch_rgb = main_sites._get_lineage_rgb_by_branch(branch_ids=branch_ids, g=g)
    x_hex = main_sites.matplotlib.colors.to_hex(branch_rgb[labels["X"]]).lower()
    c_hex = main_sites.matplotlib.colors.to_hex(branch_rgb[labels["C"]]).lower()
    svg_text = svg_path.read_text(encoding="utf-8").lower()
    assert x_hex in svg_text
    assert c_hex in svg_text
    assert re.search(r'fill:\s*{}[^>]*>a</text>'.format(re.escape(x_hex)), svg_text) is not None
    assert re.search(r'fill:\s*{}[^>]*>c</text>'.format(re.escape(c_hex)), svg_text) is not None
    assert re.search(r'>a\|0</text>', svg_text) is None
    assert re.search(r'>c\|2</text>', svg_text) is None
    # Terminal branch IDs are rendered near the tree, not appended to tip labels.
    assert re.search(r'>b0</text>', svg_text) is not None
    # Lineage site labels are now site numbers only.
    assert re.search(r'>1:\s*</text>', svg_text) is None
    assert re.search(r'>2:\s*</text>', svg_text) is None
    assert re.search(r'>3:\s*</text>', svg_text) is None
    assert re.search(r'>5:\s*</text>', svg_text) is None
    assert re.search(r'>2</text>', svg_text) is not None
    assert re.search(r'>3</text>', svg_text) is not None
    # Branch-wise substitution probabilities are shown as a heatmap with fixed 0-1 colorbar.
    assert re.search(r'>0\.0</text>', svg_text) is not None
    assert re.search(r'>1\.0</text>', svg_text) is not None
    # Heatmap row labels show branch IDs.
    assert re.search(r'>branch id</text>', svg_text) is not None


def test_plot_tree_site_set_svg_includes_branch_heatmap_panel(tmp_path, tiny_tree):
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tiny_tree.traverse()}
    branch_ids = [labels["X"], labels["C"]]
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    aa_orders = np.array(["A", "V", "T", "I"])
    state_pep = np.zeros((num_node, 5, aa_orders.shape[0]), dtype=float)
    for leaf_name in ("A", "B", "C"):
        leaf_id = labels[leaf_name]
        state_pep[leaf_id, 0, 0] = 1.0
        state_pep[leaf_id, 1, 1] = 1.0
        state_pep[leaf_id, 2, 2] = 1.0
        state_pep[leaf_id, 3, 3] = 1.0
        state_pep[leaf_id, 4, 0] = 1.0

    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4, 5],
            "OCNany2spe": [0.0, 0.0, 0.0, 0.0, 0.0],
            "OCNany2dif": [0.0, 0.0, 0.0, 0.0, 0.0],
            "N_set_expr": [False, True, True, False, False],
            "N_set_expr_prob": [0.0, 0.95, 0.82, 0.0, 0.0],
            "N_sub_{}".format(labels["X"]): [0.10, 0.90, 0.25, 0.00, 0.00],
            "N_sub_{}".format(labels["C"]): [0.00, 0.40, 0.87, 0.00, 0.00],
        }
    )
    g = {
        "tree": tiny_tree,
        "mode": "set",
        "set_stat_type": "any",
        "mode_expression": "{}|{}".format(labels["X"], labels["C"]),
        "branch_ids": np.array(branch_ids, dtype=np.int64),
        "single_branch_mode": False,
        "tree_site_plot": True,
        "tree_site_plot_format": "svg",
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "tree_site_plot_max_sites": 10,
        "site_outdir": str(tmp_path),
        "float_format": "%.4f",
        "state_pep": state_pep,
        "amino_acid_orders": aa_orders,
    }
    out_paths = main_sites.plot_tree_site(df=df, g=g)
    svg_path = tmp_path / "csubst_sites.tree_site.svg"
    table_path = tmp_path / "csubst_sites.tree_site.tsv"
    assert str(svg_path) in out_paths
    assert table_path.exists()

    plotted_df = pd.read_csv(table_path, sep="\t")
    plotted_sites = plotted_df.loc[plotted_df["is_plotted"], "codon_site_alignment"].astype(int).tolist()
    assert plotted_sites == [2, 3]

    svg_text = svg_path.read_text(encoding="utf-8").lower()
    # Heatmap colorbar remains fixed to 0-1.
    assert re.search(r'>0\.0</text>', svg_text) is not None
    assert re.search(r'>1\.0</text>', svg_text) is not None
    # Heatmap row-axis label should be present in set mode too.
    assert re.search(r'>branch id</text>', svg_text) is not None
    assert re.search(r'>b[0-9]+</text>', svg_text) is not None
    # Heatmap metric title text should not be shown.
    assert re.search(r'heatmap metric:', svg_text) is None
    # Set mode title should include operation text.
    assert re.search(
        r'focal branch ids:\s*[0-9]+,[0-9]+\s*;\s*set operation:\s*[0-9|]+\s*\(any,\s*pp\s*(≥|&#8805;|&ge;|>=|&gt;=)\s*0\.50\)',
        svg_text,
    ) is not None


def test_plot_tree_site_set_any2spe_svg_includes_channel_labels(tmp_path, tiny_tree):
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tiny_tree.traverse()}
    branch_ids = [labels["X"], labels["C"]]
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    aa_orders = np.array(["A", "V", "T", "I"])
    state_pep = np.zeros((num_node, 5, aa_orders.shape[0]), dtype=float)
    for leaf_name in ("A", "B", "C"):
        leaf_id = labels[leaf_name]
        state_pep[leaf_id, 0, 0] = 1.0
        state_pep[leaf_id, 1, 1] = 1.0
        state_pep[leaf_id, 2, 2] = 1.0
        state_pep[leaf_id, 3, 3] = 1.0
        state_pep[leaf_id, 4, 0] = 1.0

    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4, 5],
            "OCNany2spe": [0.0, 0.0, 0.0, 0.0, 0.0],
            "OCNany2dif": [0.0, 0.0, 0.0, 0.0, 0.0],
            "N_set_expr": [False, True, True, False, False],
            "N_set_expr_prob": [0.0, 0.95, 0.82, 0.0, 0.0],
            "N_set_expr_channel_index": [-1, 1, 0, -1, -1],
            "N_set_expr_channel_label": ["", "X→V", "X→A", "", ""],
            "N_set_branch_{}".format(labels["X"]) + "_prob": [0.10, 0.90, 0.25, 0.00, 0.00],
            "N_set_branch_{}".format(labels["C"]) + "_prob": [0.00, 0.40, 0.87, 0.00, 0.00],
        }
    )
    g = {
        "tree": tiny_tree,
        "mode": "set",
        "set_stat_type": "spe",
        "mode_expression": "{}|{}".format(labels["X"], labels["C"]),
        "branch_ids": np.array(branch_ids, dtype=np.int64),
        "single_branch_mode": False,
        "tree_site_plot": True,
        "tree_site_plot_format": "svg",
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "tree_site_plot_max_sites": 10,
        "site_outdir": str(tmp_path),
        "float_format": "%.4f",
        "state_pep": state_pep,
        "amino_acid_orders": aa_orders,
    }
    _ = main_sites.plot_tree_site(df=df, g=g)
    svg_path = tmp_path / "csubst_sites.tree_site.svg"
    svg_text = svg_path.read_text(encoding="utf-8").lower()
    assert re.search(r'heatmap metric:', svg_text) is None
    assert "x→v" in svg_text
    assert "x→a" in svg_text
    assert "a→x" not in svg_text


def test_get_lineage_site_heatmap_values_set_prefers_set_branch_probability_columns():
    g = {"mode": "set", "branch_ids": np.array([10, 11], dtype=np.int64)}
    display_meta = [{"site": 2}, {"site": 3}]
    df = pd.DataFrame(
        {
            "codon_site_alignment": [2, 3],
            "N_sub_10": [0.1, 0.2],
            "N_sub_11": [0.3, 0.4],
            "N_set_branch_10_prob": [0.8, 0.7],
            "N_set_branch_11_prob": [0.9, 0.6],
        }
    )
    heat_values, heat_branch_ids = main_sites.get_lineage_site_heatmap_values(df=df, display_meta=display_meta, g=g)
    assert heat_branch_ids == [10, 11]
    np.testing.assert_allclose(heat_values, np.array([[0.8, 0.7], [0.9, 0.6]], dtype=float), atol=1e-12)


def test_get_lineage_site_heatmap_values_uses_gap_mask_instead_of_zero():
    g = {
        "mode": "intersection",
        "branch_ids": np.array([10], dtype=np.int64),
        "state_pep": np.zeros((20, 3, 2), dtype=float),
    }
    # site 1: non-gap, site 2: gap
    g["state_pep"][10, 0, 0] = 1.0
    display_meta = [{"site": 1}, {"site": 2}]
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2],
            "N_sub_10": [0.25, 0.0],
        }
    )
    heat_values, heat_branch_ids = main_sites.get_lineage_site_heatmap_values(
        df=df,
        display_meta=display_meta,
        g=g,
    )
    assert heat_branch_ids == [10]
    assert heat_values.shape == (1, 2)
    assert float(heat_values[0, 0]) == pytest.approx(0.25, abs=1e-12)
    assert np.isnan(heat_values[0, 1])
