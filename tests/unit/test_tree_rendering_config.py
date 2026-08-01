import numpy as np
import pandas as pd
import pytest

from csubst import tree
from csubst import ete


def test_get_tree_figure_size_scales_with_leaf_count():
    fig_width,fig_height = tree._get_tree_figure_size(num_leaves=10, max_label_len=12)
    expected_height = max(
        tree.TREE_FIG_MIN_HEIGHT,
        tree.TREE_FIG_BASE_HEIGHT + (10 * tree.TREE_FIG_HEIGHT_PER_LEAF),
    )
    assert pytest.approx(fig_height, rel=0, abs=1e-12) == expected_height
    assert pytest.approx(fig_width, rel=0, abs=1e-12) == tree.TREE_FIG_WIDTH


def test_get_tree_figure_size_respects_tip_label_spacing_factor():
    _,default_height = tree._get_tree_figure_size(num_leaves=10, max_label_len=12)
    _,double_height = tree._get_tree_figure_size(num_leaves=10, max_label_len=12, tip_label_spacing_factor=2.0)
    expected_double_height = max(
        tree.TREE_FIG_MIN_HEIGHT,
        tree.TREE_FIG_BASE_HEIGHT + (10 * tree.TREE_FIG_HEIGHT_PER_LEAF * 2.0),
    )
    assert pytest.approx(double_height, rel=0, abs=1e-12) == expected_double_height
    assert double_height > default_height


def test_get_tree_figure_size_respects_tree_fig_max_height_override():
    _,fig_height = tree._get_tree_figure_size(
        num_leaves=100000,
        max_label_len=0,
        tree_fig_max_height=500.0,
    )
    assert fig_height == 500.0


def test_get_tree_figure_size_caps_pdf_height():
    fig_width,fig_height = tree._get_tree_figure_size(num_leaves=100000, max_label_len=0)
    assert fig_width == tree.TREE_FIG_WIDTH
    assert fig_height == tree.TREE_FIG_MAX_HEIGHT


def test_resolve_tree_tip_label_spacing_factor_rejects_nonpositive_values():
    with pytest.raises(ValueError, match="positive finite float"):
        tree._resolve_tree_tip_label_spacing_factor(0)


def test_resolve_tree_figure_max_height_rejects_nonpositive_values():
    with pytest.raises(ValueError, match="positive finite float"):
        tree._resolve_tree_figure_max_height(0)


def test_normalize_state_plot_request_rejects_legacy_yes():
    with pytest.raises(ValueError, match="no longer accepts yes/no"):
        tree.normalize_state_plot_request("yes", param_name="--plot_state_aa")


def test_plot_branch_category_writes_pdf_with_matplotlib_backend(tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,(B:1,C:1)X:1)R;", format=1))
    for node in tr.traverse():
        ete.set_prop(node, "color_PLACEHOLDER", "black")
        ete.set_prop(node, "labelcolor_PLACEHOLDER", "black")
    g = {
        "tree": tr,
        "fg_df": pd.DataFrame(columns=["name", "PLACEHOLDER"]),
    }
    outbase = tmp_path / "csubst_branch_id"
    tree.plot_branch_category(g=g, file_base=str(outbase), label="all")
    outfile = tmp_path / "csubst_branch_id.pdf"
    assert outfile.exists()
    assert outfile.stat().st_size > 0


def test_plot_state_tree_zero_sites_is_noop(tmp_path, monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    g = {
        "tree": tr,
        "fg_df": pd.DataFrame(columns=["name", "PLACEHOLDER"]),
    }
    state = np.zeros((3, 0, 2), dtype=float)
    orders = np.array(["A", "B"])
    monkeypatch.chdir(tmp_path)
    tree.plot_state_tree(state=state, orders=orders, mode="aa", g=g)
    assert list(tmp_path.glob("csubst_state_*.pdf")) == []


def test_foreground_stem_vertical_segment_is_not_colored():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(B:1,(A:1,C:1)X:1)R;", format=1))
    by_name = {n.name: n for n in tr.traverse() if n.name}
    root = by_name["R"]
    x_node = by_name["X"]
    a_node = by_name["A"]
    c_node = by_name["C"]
    b_node = by_name["B"]

    ete.set_prop(root, "is_fg_t", False)
    ete.set_prop(x_node, "is_fg_t", True)
    ete.set_prop(a_node, "is_fg_t", True)
    ete.set_prop(c_node, "is_fg_t", True)
    ete.set_prop(b_node, "is_fg_t", False)

    ete.set_prop(root, "color_t", "black")
    ete.set_prop(x_node, "color_t", "red")
    ete.set_prop(a_node, "color_t", "red")
    ete.set_prop(c_node, "color_t", "red")
    ete.set_prop(b_node, "color_t", "black")

    assert tree._is_foreground_stem_branch(x_node, "t")
    assert not tree._is_foreground_stem_branch(a_node, "t")

    v_color_stem, h_color_stem = tree._get_branch_segment_colors(x_node, "t")
    assert v_color_stem == "black"
    assert h_color_stem == "red"

    v_color_desc, h_color_desc = tree._get_branch_segment_colors(a_node, "t")
    assert v_color_desc == "red"
    assert h_color_desc == "red"


def test_highlighted_clade_uses_foreground_stem_branch_coloring():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(B:1,(A:1,C:1)X:1)R;", format=1))
    by_name = {n.name: n for n in tr.traverse() if n.name}
    x_node = by_name["X"]
    a_node = by_name["A"]

    v_color_stem, h_color_stem = tree._get_branch_segment_colors(
        x_node,
        "t",
        highlighted_node_ids={
            int(ete.get_prop(x_node, "numerical_label")),
            int(ete.get_prop(a_node, "numerical_label")),
        },
        highlight_color="orange",
    )
    assert v_color_stem == "black"
    assert h_color_stem == "orange"

    v_color_desc, h_color_desc = tree._get_branch_segment_colors(
        a_node,
        "t",
        highlighted_node_ids={
            int(ete.get_prop(x_node, "numerical_label")),
            int(ete.get_prop(a_node, "numerical_label")),
        },
        highlight_color="orange",
    )
    assert v_color_desc == "orange"
    assert h_color_desc == "orange"
