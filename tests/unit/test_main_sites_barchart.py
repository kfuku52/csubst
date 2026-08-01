import numpy as np
import pandas as pd
import pytest

from csubst import main_sites
from csubst import ete


def test_plot_barchart_set_has_n_plus_three_rows_for_two_branches(tmp_path):
    df = pd.DataFrame(
        {
            "codon_site_alignment": [0, 1, 2],
            "gap_rate_all": [0.0, 0.2, 0.0],
            "gap_rate_target": [0.0, 0.1, 0.0],
            "N_sub": [0.2, 0.3, 0.4],
            "S_sub": [0.1, 0.0, 0.2],
            "N_sub_117": [0.2, 0.1, 0.0],
            "S_sub_117": [0.0, 0.2, 0.1],
            "N_sub_48": [0.0, 0.3, 0.1],
            "S_sub_48": [0.1, 0.0, 0.0],
            "N_set_expr_prob": [0.0, 1.1, 1.6],
            "N_set_expr": [False, True, True],
        }
    )
    g = {
        "mode": "set",
        "set_stat_type": "any",
        "single_branch_mode": False,
        "mode_expression": "117|48",
        "branch_ids": np.array([48, 117], dtype=np.int64),
        "pdb": None,
        "site_outdir": str(tmp_path),
    }
    main_sites.plot_barchart(df=df, g=g)
    fig = main_sites.plt.gcf()
    axes = fig.axes
    assert len(axes) == 5
    assert axes[0].get_ylabel() == "Branch-wise\nsubstitutions\nin the entire tree"
    assert axes[1].get_ylabel() == "Branch-wise\nsubstitutions\nin the targets"
    assert axes[2].get_ylabel() == "Substitutions in\nbranch_id 117"
    assert axes[3].get_ylabel() == "Substitutions in\nbranch_id 48"
    assert axes[4].get_ylabel() == "Substitutions in\n117|48"
    main_sites.plt.close(fig)


def test_plot_barchart_set_with_A_has_extra_A_row(tmp_path):
    df = pd.DataFrame(
        {
            "codon_site_alignment": [0, 1, 2],
            "gap_rate_all": [0.0, 0.2, 0.0],
            "gap_rate_target": [0.0, 0.1, 0.0],
            "N_sub": [0.2, 0.3, 0.4],
            "S_sub": [0.1, 0.0, 0.2],
            "N_sub_117": [0.2, 0.1, 0.0],
            "S_sub_117": [0.0, 0.2, 0.1],
            "N_sub_48": [0.0, 0.3, 0.1],
            "S_sub_48": [0.1, 0.0, 0.0],
            "N_set_other_prob": [0.1, 0.8, 0.2],
            "S_set_other_prob": [0.0, 0.1, 0.0],
            "N_set_expr_prob": [0.0, 0.3, 0.9],
            "N_set_expr": [False, True, True],
        }
    )
    g = {
        "mode": "set",
        "set_stat_type": "any",
        "single_branch_mode": False,
        "mode_expression": "((117|48)-A)",
        "branch_ids": np.array([48, 117], dtype=np.int64),
        "pdb": None,
        "site_outdir": str(tmp_path),
    }
    main_sites.plot_barchart(df=df, g=g)
    fig = main_sites.plt.gcf()
    axes = fig.axes
    assert len(axes) == 6
    assert axes[4].get_ylabel() == "Substitutions in\nA"
    assert axes[5].get_ylabel() == "Substitutions in\n((117|48)-A)"
    ymin, ymax = axes[4].get_ylim()
    assert pytest.approx(ymin, abs=1e-12) == 0.0
    assert pytest.approx(ymax, abs=1e-12) == 1.0
    main_sites.plt.close(fig)


def test_plot_barchart_lineage_branch_rows_use_fixed_unit_y_range(tmp_path):
    df = pd.DataFrame(
        {
            "codon_site_alignment": [0, 1, 2],
            "gap_rate_all": [0.0, 0.2, 0.0],
            "gap_rate_target": [0.0, 0.1, 0.0],
            "N_sub": [0.2, 0.3, 0.4],
            "S_sub": [0.1, 0.0, 0.2],
            "N_sub_13": [0.2, 0.1, 0.0],
            "S_sub_13": [0.0, 0.2, 0.1],
            "N_sub_12": [0.0, 0.3, 0.1],
            "S_sub_12": [0.1, 0.0, 0.0],
            "N_sub_2": [0.4, 0.2, 0.3],
            "S_sub_2": [0.0, 0.1, 0.0],
        }
    )
    g = {
        "mode": "lineage",
        "single_branch_mode": False,
        "branch_ids": np.array([13, 12, 2], dtype=np.int64),
        "pdb": None,
        "site_outdir": str(tmp_path),
    }
    main_sites.plot_barchart(df=df, g=g)
    fig = main_sites.plt.gcf()
    axes = fig.axes
    # 5 data rows (N+2 for lineage) + 1 bottom colorbar axis
    assert len(axes) == 6
    for ax in axes[2:5]:
        ymin, ymax = ax.get_ylim()
        assert pytest.approx(ymin, abs=1e-12) == 0.0
        assert pytest.approx(ymax, abs=1e-12) == 1.0
    assert "Branch distance from ancestor" in axes[5].get_xlabel()
    main_sites.plt.close(fig)


def test_plot_barchart_lineage_colorbar_uses_actual_branch_length_ticks(tmp_path, tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    x_id = labels["X"]
    c_id = labels["C"]
    df = pd.DataFrame(
        {
            "codon_site_alignment": [0, 1, 2],
            "gap_rate_all": [0.0, 0.1, 0.0],
            "gap_rate_target": [0.0, 0.1, 0.0],
            "N_sub": [0.2, 0.3, 0.4],
            "S_sub": [0.1, 0.0, 0.2],
            "N_sub_{}".format(x_id): [0.1, 0.1, 0.0],
            "S_sub_{}".format(x_id): [0.0, 0.0, 0.0],
            "N_sub_{}".format(c_id): [0.0, 0.2, 0.1],
            "S_sub_{}".format(c_id): [0.0, 0.0, 0.0],
        }
    )
    g = {
        "mode": "lineage",
        "single_branch_mode": False,
        "branch_ids": np.array([x_id, c_id], dtype=np.int64),
        "tree": tiny_tree,
        "pdb": None,
        "site_outdir": str(tmp_path),
    }
    main_sites.plot_barchart(df=df, g=g)
    fig = main_sites.plt.gcf()
    axes = fig.axes
    # 4 data rows (N+2 where N=2) + 1 bottom colorbar axis
    assert len(axes) == 5
    cax = axes[4]
    assert "branch-length units" in cax.get_xlabel()
    fig.canvas.draw()
    tick_vals = [float(t.get_text()) for t in cax.get_xticklabels() if t.get_text() != ""]
    # tiny_tree branch lengths are X=3 and C=2, so midpoint distances are 1.5 and 4.0.
    assert pytest.approx(min(tick_vals), abs=1e-6) == 1.5
    assert pytest.approx(max(tick_vals), abs=1e-6) == 4.0
    main_sites.plt.close(fig)


def test_plot_lineage_tree_writes_pdf_and_applies_lineage_branch_colors(tmp_path, tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    branch_ids = np.array([labels["X"], labels["C"]], dtype=np.int64)
    g = {
        "mode": "lineage",
        "tree": tiny_tree,
        "branch_ids": branch_ids,
    }
    outbase = tmp_path / "csubst_sites"
    main_sites.plot_lineage_tree(g=g, outbase=str(outbase))
    outfile = tmp_path / "csubst_sites.tree.pdf"
    assert outfile.exists()
    assert outfile.stat().st_size > 0
    lineage_rgb = main_sites._get_lineage_rgb_by_branch(branch_ids=branch_ids.tolist(), g=g)
    for node in tiny_tree.traverse():
        bid = int(ete.get_prop(node, "numerical_label"))
        color = ete.get_prop(node, "color_PLACEHOLDER")
        if bid in lineage_rgb:
            assert color == lineage_rgb[bid]
        else:
            assert color == "black"


def test_plot_lineage_tree_accepts_scalar_branch_id(tmp_path, tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    target_id = np.int64(labels["X"])
    g = {
        "mode": "lineage",
        "tree": tiny_tree,
        "branch_ids": target_id,
    }
    outbase = tmp_path / "csubst_sites_scalar"
    main_sites.plot_lineage_tree(g=g, outbase=str(outbase))
    outfile = tmp_path / "csubst_sites_scalar.tree.pdf"
    assert outfile.exists()
    assert outfile.stat().st_size > 0
    assert ete.get_prop(next(node for node in tiny_tree.traverse() if int(ete.get_prop(node, "numerical_label")) == int(target_id)), "color_PLACEHOLDER") != "black"
