import numpy
import pandas
import pytest
from pathlib import Path

from csubst import main_site
from csubst import substitution
from csubst import substitution_cy
from csubst import tree
from csubst import ete


@pytest.fixture
def tiny_tree():
    tr = ete.PhyloNode("(B:1,(A:1,C:2)X:3)R;", format=1)
    return tree.add_numerical_node_labels(tr)


def test_get_gapsite_rate_matches_manual_count():
    state_tensor = numpy.array(
        [
            [[1, 0], [0, 0], [1, 0]],
            [[0, 1], [0, 0], [0, 1]],
            [[1, 0], [1, 0], [0, 0]],
        ],
        dtype=float,
    )
    # site-wise gap rates: [0/3, 2/3, 1/3]
    out = main_site.get_gapsite_rate(state_tensor)
    numpy.testing.assert_allclose(out, [0.0, 2.0 / 3.0, 1.0 / 3.0], atol=1e-12)


def test_extend_site_index_edge_fills_missing_edges():
    sites = pandas.Series([2, 3, 7], dtype=int)
    out = main_site.extend_site_index_edge(sites, num_extend=2)
    # Gap between 3 and 7 is filled by 5 and 6.
    assert out.tolist() == [2, 3, 5, 6, 7]


def test_initialize_site_df_columns_are_correct():
    df = main_site.initialize_site_df(4)
    assert df["codon_site_alignment"].tolist() == [0, 1, 2, 3]
    assert df["nuc_site_alignment"].tolist() == [1, 4, 7, 10]


def test_combinatorial2single_columns_removes_combination_columns():
    df = pandas.DataFrame(
        {
            "OCSany2any": [1],
            "OCSany2spe": [2],
            "OCNspe2dif": [3],
            "kept": [4],
        }
    )
    out = main_site.combinatorial2single_columns(df)
    assert list(out.columns) == ["kept"]


@pytest.mark.parametrize("mode_name,expected", [("intersection", True), ("lineage", False), ("set", False)])
def test_mode_helpers_only_enable_optional_outputs_in_intersection(mode_name, expected):
    g = {"mode": mode_name}
    assert main_site.should_plot_state(g) == expected
    assert main_site.should_save_pymol_views(g) == expected


def test_get_yvalues_for_supported_modes():
    df = pandas.DataFrame(
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
    numpy.testing.assert_allclose(main_site.get_yvalues(df, "_sub", "S"), [0.0, 2.5], atol=1e-12)
    numpy.testing.assert_allclose(main_site.get_yvalues(df, "_sub", "N"), [1.0, 2.0], atol=1e-12)
    numpy.testing.assert_allclose(main_site.get_yvalues(df, "_sub_", "S"), [0.0, 1.0], atol=1e-12)
    numpy.testing.assert_allclose(main_site.get_yvalues(df, "any2spe", "S"), [0.3, 0.6], atol=1e-12)
    numpy.testing.assert_allclose(main_site.get_yvalues(df, "any2spe", "N"), [0.2, 0.4], atol=1e-12)


def test_get_yvalues_for_lineage_branch_specific_rows():
    df = pandas.DataFrame(
        {
            "S_sub_13": [0.0, 0.6],
            "N_sub_13": [0.2, 0.3],
            "S_sub_12": [0.2, 0.0],
            "N_sub_12": [0.4, 0.5],
        }
    )
    numpy.testing.assert_allclose(main_site.get_yvalues(df, "_sub_branch_13", "N"), [0.2, 0.3], atol=1e-12)
    numpy.testing.assert_allclose(main_site.get_yvalues(df, "_sub_branch_13", "S"), [0.0, 0.9], atol=1e-12)
    numpy.testing.assert_allclose(main_site.get_yvalues(df, "_sub_branch_12", "N"), [0.4, 0.5], atol=1e-12)
    numpy.testing.assert_allclose(main_site.get_yvalues(df, "_sub_branch_12", "S"), [0.6, 0.0], atol=1e-12)


def test_get_plot_sub_types_and_colors_lineage_has_n_plus_two_rows():
    g = {"mode": "lineage", "single_branch_mode": False, "branch_ids": numpy.array([13, 12, 2], dtype=numpy.int64)}
    sub_types, sn_colors = main_site.get_plot_sub_types_and_colors(g)
    assert list(sub_types.keys()) == ["_sub", "_sub_", "_sub_branch_13", "_sub_branch_12", "_sub_branch_2"]
    assert "entire tree" in sub_types["_sub"]
    assert sub_types["_sub_"] == "Branch-wise\nsubstitutions\nin the targets"
    assert sub_types["_sub_branch_13"] == "Substitutions in\nbranch_id 13"
    assert sub_types["_sub_branch_12"] == "Substitutions in\nbranch_id 12"
    assert sub_types["_sub_branch_2"] == "Substitutions in\nbranch_id 2"
    assert sn_colors["_sub"]["N"] == "black"
    assert sn_colors["_sub_"]["S"] == "gainsboro"
    assert sn_colors["_sub_branch_12"]["S"] == "gainsboro"


def test_get_set_expression_display_branch_ids_preserves_expression_order():
    g = {"mode_expression": "117|48", "branch_ids": numpy.array([48, 117], dtype=numpy.int64)}
    out = main_site._get_set_expression_display_branch_ids(g)
    assert out.tolist() == [117, 48]


def test_get_plot_sub_types_and_colors_set_has_branch_rows_and_expression_row():
    g = {
        "mode": "set",
        "single_branch_mode": False,
        "mode_expression": "117|48",
        "branch_ids": numpy.array([48, 117], dtype=numpy.int64),
    }
    sub_types, sn_colors = main_site.get_plot_sub_types_and_colors(g)
    assert list(sub_types.keys()) == ["_sub", "_sub_", "_sub_branch_117", "_sub_branch_48", "_set_expr"]
    assert sub_types["_sub"] == "Branch-wise\nsubstitutions\nin the entire tree"
    assert sub_types["_sub_"] == "Branch-wise\nsubstitutions\nin the targets"
    assert sub_types["_sub_branch_117"] == "Substitutions in\nbranch_id 117"
    assert sub_types["_sub_branch_48"] == "Substitutions in\nbranch_id 48"
    assert sub_types["_set_expr"] == "Substitutions in\n117|48"
    assert sn_colors["_set_expr"]["N"] == "red"
    assert sn_colors["_set_expr"]["S"] == "gainsboro"


def test_get_plot_sub_types_and_colors_set_with_A_has_A_row():
    g = {
        "mode": "set",
        "single_branch_mode": False,
        "mode_expression": "((117|48)-A)",
        "branch_ids": numpy.array([48, 117], dtype=numpy.int64),
    }
    sub_types, _ = main_site.get_plot_sub_types_and_colors(g)
    assert list(sub_types.keys()) == ["_sub", "_sub_", "_sub_branch_117", "_sub_branch_48", "_set_other", "_set_expr"]
    assert sub_types["_set_other"] == "Substitutions in\nA"


def test_get_yvalues_set_expression_prefers_probability_column():
    df = pandas.DataFrame({"N_set_expr_prob": [0.0, 1.7, 2.0], "N_set_expr": [False, True, True]})
    numpy.testing.assert_allclose(main_site.get_yvalues(df, "_set_expr", "N"), [0.0, 1.7, 2.0], atol=1e-12)
    numpy.testing.assert_allclose(main_site.get_yvalues(df, "_set_expr", "S"), [0.0, 0.0, 0.0], atol=1e-12)


def test_get_yvalues_set_other_uses_other_prob_columns():
    df = pandas.DataFrame({"N_set_other": [False, True], "N_set_other_prob": [0.2, 0.4], "S_set_other_prob": [0.1, 0.0]})
    numpy.testing.assert_allclose(main_site.get_yvalues(df, "_set_other", "N"), [0.0, 1.0], atol=1e-12)
    numpy.testing.assert_allclose(main_site.get_yvalues(df, "_set_other", "S"), [0.0, 0.0], atol=1e-12)


def test_plot_barchart_set_has_n_plus_three_rows_for_two_branches(tmp_path):
    df = pandas.DataFrame(
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
        "single_branch_mode": False,
        "mode_expression": "117|48",
        "branch_ids": numpy.array([48, 117], dtype=numpy.int64),
        "pdb": None,
        "site_outdir": str(tmp_path),
    }
    main_site.plot_barchart(df=df, g=g)
    fig = main_site.matplotlib.pyplot.gcf()
    axes = fig.axes
    assert len(axes) == 5
    assert axes[0].get_ylabel() == "Branch-wise\nsubstitutions\nin the entire tree"
    assert axes[1].get_ylabel() == "Branch-wise\nsubstitutions\nin the targets"
    assert axes[2].get_ylabel() == "Substitutions in\nbranch_id 117"
    assert axes[3].get_ylabel() == "Substitutions in\nbranch_id 48"
    assert axes[4].get_ylabel() == "Substitutions in\n117|48"
    main_site.matplotlib.pyplot.close(fig)


def test_plot_barchart_set_with_A_has_extra_A_row(tmp_path):
    df = pandas.DataFrame(
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
        "single_branch_mode": False,
        "mode_expression": "((117|48)-A)",
        "branch_ids": numpy.array([48, 117], dtype=numpy.int64),
        "pdb": None,
        "site_outdir": str(tmp_path),
    }
    main_site.plot_barchart(df=df, g=g)
    fig = main_site.matplotlib.pyplot.gcf()
    axes = fig.axes
    assert len(axes) == 6
    assert axes[4].get_ylabel() == "Substitutions in\nA"
    assert axes[5].get_ylabel() == "Substitutions in\n((117|48)-A)"
    ymin, ymax = axes[4].get_ylim()
    assert pytest.approx(ymin, abs=1e-12) == 0.0
    assert pytest.approx(ymax, abs=1e-12) == 1.0
    main_site.matplotlib.pyplot.close(fig)


def test_plot_barchart_lineage_branch_rows_use_fixed_unit_y_range(tmp_path):
    df = pandas.DataFrame(
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
        "branch_ids": numpy.array([13, 12, 2], dtype=numpy.int64),
        "pdb": None,
        "site_outdir": str(tmp_path),
    }
    main_site.plot_barchart(df=df, g=g)
    fig = main_site.matplotlib.pyplot.gcf()
    axes = fig.axes
    # 5 data rows (N+2 for lineage) + 1 bottom colorbar axis
    assert len(axes) == 6
    for ax in axes[2:5]:
        ymin, ymax = ax.get_ylim()
        assert pytest.approx(ymin, abs=1e-12) == 0.0
        assert pytest.approx(ymax, abs=1e-12) == 1.0
    assert "Branch distance from ancestor" in axes[5].get_xlabel()
    main_site.matplotlib.pyplot.close(fig)


def test_plot_barchart_lineage_colorbar_uses_actual_branch_length_ticks(tmp_path, tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    x_id = labels["X"]
    c_id = labels["C"]
    df = pandas.DataFrame(
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
        "branch_ids": numpy.array([x_id, c_id], dtype=numpy.int64),
        "tree": tiny_tree,
        "pdb": None,
        "site_outdir": str(tmp_path),
    }
    main_site.plot_barchart(df=df, g=g)
    fig = main_site.matplotlib.pyplot.gcf()
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
    main_site.matplotlib.pyplot.close(fig)


def test_plot_lineage_tree_writes_pdf_and_applies_lineage_branch_colors(tmp_path, tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    branch_ids = numpy.array([labels["X"], labels["C"]], dtype=numpy.int64)
    g = {
        "mode": "lineage",
        "tree": tiny_tree,
        "branch_ids": branch_ids,
    }
    outbase = tmp_path / "csubst_site"
    main_site.plot_lineage_tree(g=g, outbase=str(outbase))
    outfile = tmp_path / "csubst_site.tree.pdf"
    assert outfile.exists()
    assert outfile.stat().st_size > 0
    lineage_rgb = main_site._get_lineage_rgb_by_branch(branch_ids=branch_ids.tolist(), g=g)
    for node in tiny_tree.traverse():
        bid = int(ete.get_prop(node, "numerical_label"))
        color = ete.get_prop(node, "color_PLACEHOLDER")
        if bid in lineage_rgb:
            assert color == lineage_rgb[bid]
        else:
            assert color == "black"


def test_translate_and_write_fasta(tmp_path):
    g = {"matrix_groups": {"K": ["AAA"], "N": ["AAC"]}}
    assert main_site.translate("AAAAAC", g) == "KN"
    out = tmp_path / "toy.fa"
    main_site.write_fasta(str(out), "sample", "KN")
    assert out.read_text(encoding="utf-8") == ">sample\nKN\n"


def test_get_parent_branch_ids(tiny_tree):
    bids = []
    for node in tiny_tree.traverse():
        if node.name in {"A", "C"}:
            bids.append(ete.get_prop(node, "numerical_label"))
    out = main_site.get_parent_branch_ids(numpy.array(bids), {"tree": tiny_tree})
    assert len(out) == 2
    # Both A and C have internal node X as parent.
    x_id = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse() if n.name == "X"][0]
    assert set(out.values()) == {x_id}


def test_resolve_site_jobs_intersection_mode_preserves_branch_set_and_outdir_prefix(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "intersection", "branch_id": "{},{}".format(labels["A"], labels["C"])}
    out = main_site.resolve_site_jobs(g)
    assert len(out["site_jobs"]) == 1
    numpy.testing.assert_array_equal(out["site_jobs"][0]["branch_ids"], [labels["A"], labels["C"]])
    assert out["site_jobs"][0]["site_outdir"].startswith("./csubst_site.branch_id")


def test_resolve_site_jobs_lineage_mode_returns_ancestor_to_descendant_path(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "lineage", "branch_id": "{},{}".format(labels["X"], labels["C"])}
    out = main_site.resolve_site_jobs(g)
    assert len(out["site_jobs"]) == 1
    numpy.testing.assert_array_equal(out["site_jobs"][0]["branch_ids"], [labels["X"], labels["C"]])
    assert out["site_jobs"][0]["site_outdir"] == "./csubst_site.lineage.branch_id{},{}".format(labels["X"], labels["C"])
    assert not out["site_jobs"][0]["single_branch_mode"]


def test_resolve_site_jobs_lineage_mode_rejects_non_ancestor_pairs(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "lineage", "branch_id": "{},{}".format(labels["B"], labels["C"])}
    with pytest.raises(ValueError, match="ancestor"):
        main_site.resolve_site_jobs(g)


@pytest.mark.parametrize("mode_name", ["total", "each", "all", "clade"])
def test_resolve_site_jobs_rejects_removed_modes(tiny_tree, mode_name):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {
        "tree": tiny_tree,
        "mode": mode_name,
        "branch_id": "{},{}".format(labels["A"], labels["C"]),
    }
    with pytest.raises(ValueError, match="intersection,lineage,set"):
        main_site.resolve_site_jobs(g)


def test_resolve_site_jobs_set_mode_extracts_expression_branch_ids(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    root_id = int(ete.get_prop(tiny_tree, "numerical_label"))
    g = {"tree": tiny_tree, "mode": "set,({}|{})-{}".format(labels["A"], labels["C"], root_id), "branch_id": "unused"}
    out = main_site.resolve_site_jobs(g)
    assert len(out["site_jobs"]) == 1
    numpy.testing.assert_array_equal(out["site_jobs"][0]["branch_ids"], sorted([labels["A"], labels["C"]]))
    assert out["mode"] == "set"
    assert out["mode_expression"] == "({}|{})-{}".format(labels["A"], labels["C"], root_id)
    assert out["site_jobs"][0]["site_outdir"].startswith("./csubst_site.set.expr")
    assert "_or_" in out["site_jobs"][0]["site_outdir"]
    assert "_minus_" in out["site_jobs"][0]["site_outdir"]


def test_resolve_site_jobs_set_mode_with_all_other_symbol_in_label(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "set,({}|{})-A".format(labels["A"], labels["C"]), "branch_id": "unused"}
    out = main_site.resolve_site_jobs(g)
    assert len(out["site_jobs"]) == 1
    numpy.testing.assert_array_equal(out["site_jobs"][0]["branch_ids"], sorted([labels["A"], labels["C"]]))
    assert out["site_jobs"][0]["site_outdir"].startswith("./csubst_site.set.expr")
    assert "_all_other" in out["site_jobs"][0]["site_outdir"]


def test_resolve_site_jobs_set_mode_without_branch_id(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "set,{}|{}".format(labels["A"], labels["B"])}
    out = main_site.resolve_site_jobs(g)
    assert len(out["site_jobs"]) == 1
    numpy.testing.assert_array_equal(out["site_jobs"][0]["branch_ids"], sorted([labels["A"], labels["B"]]))


def test_resolve_site_jobs_set_mode_rejects_unknown_branch_ids(tiny_tree):
    g = {"tree": tiny_tree, "mode": "set,999|1", "branch_id": "unused"}
    with pytest.raises(ValueError, match="unknown branch IDs"):
        main_site.resolve_site_jobs(g)


def test_resolve_site_jobs_set_mode_rejects_invalid_expression_syntax(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "set,({}|{}".format(labels["A"], labels["B"])}
    with pytest.raises(ValueError, match="Unbalanced parentheses"):
        main_site.resolve_site_jobs(g)


def test_add_set_mode_columns_evaluates_set_expression():
    df = pandas.DataFrame(
        {
            "N_sub_1": [0.9, 0.2, 0.9, 0.1],
            "N_sub_5": [0.1, 0.9, 0.9, 0.9],
            "N_sub_25": [0.9, 0.9, 0.1, 0.9],
        }
    )
    g = {"mode": "set", "mode_expression": "((1|5)-0)&25", "pymol_min_single_prob": 0.8}
    out = main_site.add_set_mode_columns(df=df.copy(), g=g)
    # (1|5) => [T,T,T,T], minus root(0)=same, intersect 25 => [T,T,F,T]
    assert out["N_set_expr"].tolist() == [True, True, False, True]
    numpy.testing.assert_allclose(out["N_set_expr_prob"].to_numpy(), [1.9, 2.0, 0.0, 1.9], atol=1e-12)


def test_add_set_mode_columns_supports_xor_and_parentheses():
    df = pandas.DataFrame(
        {
            "N_sub_1": [0.9, 0.1, 0.9, 0.1],
            "N_sub_5": [0.1, 0.9, 0.9, 0.1],
            "N_sub_9": [0.1, 0.1, 0.9, 0.9],
        }
    )
    g = {"mode": "set", "mode_expression": "(1^5)&9", "pymol_min_single_prob": 0.8}
    out = main_site.add_set_mode_columns(df=df.copy(), g=g)
    # 1^5 => [T,T,F,F], intersect 9 => [F,F,F,F]
    assert out["N_set_expr"].tolist() == [False, False, False, False]
    numpy.testing.assert_allclose(out["N_set_expr_prob"].to_numpy(), [0.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_add_set_mode_columns_supports_all_other_symbol(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    a_id = labels["A"]
    c_id = labels["C"]
    b_id = labels["B"]
    x_id = labels["X"]
    n_site = 3
    max_id = max([int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()])
    on_tensor = numpy.zeros((max_id + 1, n_site, 1, 1, 1), dtype=float)
    # Site 0: target-only substitution on A branch.
    on_tensor[a_id, 0, 0, 0, 0] = 0.9
    # Site 1: target on A and another-branch substitution on B.
    on_tensor[a_id, 1, 0, 0, 0] = 0.9
    on_tensor[b_id, 1, 0, 0, 0] = 0.9
    # Site 2: target-only substitution on C branch.
    on_tensor[c_id, 2, 0, 0, 0] = 0.9
    # Keep one internal non-target branch explicitly below threshold at all sites.
    on_tensor[x_id, :, 0, 0, 0] = 0.1
    df = pandas.DataFrame(
        {
            "N_sub_{}".format(a_id): [0.9, 0.9, 0.0],
            "N_sub_{}".format(c_id): [0.0, 0.0, 0.9],
        }
    )
    g = {
        "mode": "set",
        "mode_expression": "({}|{})-A".format(a_id, c_id),
        "pymol_min_single_prob": 0.8,
        "tree": tiny_tree,
    }
    out = main_site.add_set_mode_columns(df=df.copy(), g=g, ON_tensor=on_tensor)
    assert out["N_set_expr"].tolist() == [True, False, True]
    numpy.testing.assert_allclose(out["N_set_expr_prob"].to_numpy(), [0.9, 0.0, 0.9], atol=1e-12)
    assert out["N_set_other"].tolist() == [False, True, False]
    numpy.testing.assert_allclose(out["N_set_other_prob"].to_numpy(), [0.1, 1.0, 0.1], atol=1e-12)
    numpy.testing.assert_allclose(out["S_set_other_prob"].to_numpy(), [0.0, 0.0, 0.0], atol=1e-12)
    assert out["N_set_A"].tolist() == [False, True, False]
    numpy.testing.assert_allclose(out["N_set_A_prob"].to_numpy(), [0.1, 1.0, 0.1], atol=1e-12)
    numpy.testing.assert_allclose(out["S_set_A_prob"].to_numpy(), [0.0, 0.0, 0.0], atol=1e-12)


def test_resolve_site_jobs_intersection_fg_reads_cb_file(tmp_path, tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    cb_path = tmp_path / "cb.tsv"
    pandas.DataFrame(
        {
            "branch_id_1": [labels["A"], labels["B"]],
            "branch_id_2": [labels["C"], labels["C"]],
            "is_fg_demo": ["Y", "N"],
        }
    ).to_csv(cb_path, sep="\t", index=False)
    g = {
        "tree": tiny_tree,
        "mode": "intersection",
        "branch_id": "fg",
        "cb_file": str(cb_path),
    }
    out = main_site.resolve_site_jobs(g)
    assert len(out["site_jobs"]) == 1
    numpy.testing.assert_array_equal(out["site_jobs"][0]["branch_ids"], [labels["A"], labels["C"]])


def test_get_state_orders():
    g = {"amino_acid_orders": numpy.array(["A", "B"]), "matrix_groups": {"grp": ["AA", "AB"]}}
    orders_nsy, keys_nsy = main_site.get_state_orders(g, "nsy")
    assert keys_nsy == ["nsy"]
    assert list(orders_nsy["nsy"]) == ["A", "B"]
    orders_syn, keys_syn = main_site.get_state_orders(g, "syn")
    assert keys_syn == ["grp"]
    assert orders_syn["grp"] == ["AA", "AB"]


def test_add_gapline_empty_df_is_noop():
    df = pandas.DataFrame({"codon_site_alignment": [], "gap_rate_all": []})
    fig, ax = main_site.matplotlib.pyplot.subplots()
    main_site.add_gapline(df=df, gapcol="gap_rate_all", xcol="codon_site_alignment", yvalue=0.5, lw=1, ax=ax)
    assert len(ax.collections) == 0
    main_site.matplotlib.pyplot.close(fig)


def test_classify_tree_site_categories_prefers_larger_signal():
    df = pandas.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4],
            "OCNany2spe": [0.6, 0.1, 0.4, 0.8],
            "OCNany2dif": [0.1, 0.7, 0.4, 0.9],
        }
    )
    g = {
        "single_branch_mode": False,
        "tree_site_plot_min_prob": 0.5,
        "pymol_min_combinat_prob": 0.5,
        "pymol_min_single_prob": 0.8,
    }
    out, min_prob = main_site.classify_tree_site_categories(df=df, g=g)
    assert pytest.approx(min_prob, abs=1e-12) == 0.5
    assert out["tree_site_category"].tolist() == ["convergent", "divergent", "blank", "divergent"]


def test_get_tree_site_display_sites_respects_max_sites_when_one_and_both_categories_exist():
    tree_site_df = pandas.DataFrame(
        {
            "codon_site_alignment": [11, 22],
            "convergent_score": [0.91, 0.10],
            "divergent_score": [0.10, 0.95],
            "tree_site_category": ["convergent", "divergent"],
        }
    )
    g = {"tree_site_plot_max_sites": 1}

    display_meta = main_site.get_tree_site_display_sites(tree_site_df=tree_site_df, g=g)
    site_rows = [row for row in display_meta if row["site"] is not None]

    assert len(site_rows) == 1
    assert site_rows[0]["site"] == 22
    assert site_rows[0]["category"] == "divergent"
    assert all(row["category"] != "separator" for row in display_meta)


def test_get_tree_site_display_sites_refills_capacity_from_other_category():
    tree_site_df = pandas.DataFrame(
        {
            "codon_site_alignment": [10, 20, 21, 22, 23, 24],
            "convergent_score": [0.99, 0.10, 0.09, 0.08, 0.07, 0.06],
            "divergent_score": [0.01, 0.95, 0.94, 0.93, 0.92, 0.91],
            "tree_site_category": ["convergent", "divergent", "divergent", "divergent", "divergent", "divergent"],
        }
    )
    g = {"tree_site_plot_max_sites": 4}

    display_meta = main_site.get_tree_site_display_sites(tree_site_df=tree_site_df, g=g)
    site_rows = [row for row in display_meta if row["site"] is not None]
    conv_count = sum(row["category"] == "convergent" for row in site_rows)
    div_count = sum(row["category"] == "divergent" for row in site_rows)

    assert len(site_rows) == 4
    assert conv_count == 1
    assert div_count == 3


def test_get_tree_plot_coordinates_returns_expected_root_and_leaf_positions(tiny_tree):
    xcoord, ycoord, leaf_order = main_site.get_tree_plot_coordinates(tiny_tree)
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


def test_get_highlight_leaf_and_branch_ids_marks_only_explicit_targets(tiny_tree):
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tiny_tree.traverse()}

    leaf_ids, branch_ids = main_site.get_highlight_leaf_and_branch_ids(
        tree=tiny_tree,
        branch_ids={labels["X"]},
    )
    assert branch_ids == {labels["X"]}
    assert leaf_ids == set()

    leaf_ids, branch_ids = main_site.get_highlight_leaf_and_branch_ids(
        tree=tiny_tree,
        branch_ids={labels["A"], labels["X"]},
    )
    assert branch_ids == {labels["A"], labels["X"]}
    assert leaf_ids == {labels["A"]}


def test_plot_tree_site_writes_figure_and_category_table(tmp_path, tiny_tree):
    branch_ids = []
    labels = {}
    for node in tiny_tree.traverse():
        labels[node.name] = ete.get_prop(node, "numerical_label")
        if node.name in {"A", "C"}:
            branch_ids.append(ete.get_prop(node, "numerical_label"))
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    aa_orders = numpy.array(["A", "V", "T", "I"])
    state_pep = numpy.zeros((num_node, 4, aa_orders.shape[0]), dtype=float)
    # Leaf A states: A, V, T, A
    state_pep[labels["A"], 0, 0] = 1.0
    state_pep[labels["A"], 1, 1] = 1.0
    state_pep[labels["A"], 2, 2] = 1.0
    state_pep[labels["A"], 3, 0] = 1.0
    # Leaf B states: A, T, T, A
    state_pep[labels["B"], 0, 0] = 1.0
    state_pep[labels["B"], 1, 2] = 1.0
    state_pep[labels["B"], 2, 2] = 1.0
    state_pep[labels["B"], 3, 0] = 1.0
    # Leaf C states: A, V, I, A
    state_pep[labels["C"], 0, 0] = 1.0
    state_pep[labels["C"], 1, 1] = 1.0
    state_pep[labels["C"], 2, 3] = 1.0
    state_pep[labels["C"], 3, 0] = 1.0
    df = pandas.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4],
            "OCNany2spe": [0.7, 0.1, 0.2, 0.6],
            "OCNany2dif": [0.1, 0.6, 0.1, 0.1],
        }
    )
    g = {
        "tree": tiny_tree,
        "branch_ids": numpy.array(branch_ids, dtype=int),
        "single_branch_mode": False,
        "tree_site_plot": True,
        "tree_site_plot_format": "pdf",
        "tree_site_plot_min_prob": 0.5,
        "pymol_min_combinat_prob": 0.5,
        "pymol_min_single_prob": 0.8,
        "tree_site_plot_max_sites": 60,
        "site_outdir": str(tmp_path),
        "float_format": "%.4f",
        "state_pep": state_pep,
        "amino_acid_orders": aa_orders,
    }
    main_site.plot_tree_site(df=df, g=g)
    fig_path = tmp_path / "csubst_site.tree_site.pdf"
    table_path = tmp_path / "csubst_site.tree_site.tsv"
    assert fig_path.exists()
    assert table_path.exists()
    out_df = pandas.read_csv(table_path, sep="\t")
    assert out_df["tree_site_category"].tolist() == ["convergent", "divergent", "blank", "convergent"]


def test_get_df_ad_add_site_stats_and_target_flag():
    g = {"amino_acid_orders": numpy.array(["A", "B"]), "matrix_groups": {"grp": ["AA", "AB"]}}
    sub_tensor = numpy.zeros((2, 2, 1, 2, 2), dtype=float)
    # A->B occurs at site 0 and site 1 with totals [2,1].
    sub_tensor[0, 0, 0, 0, 1] = 2.0
    sub_tensor[0, 1, 0, 0, 1] = 1.0
    # B->A occurs with totals [3,1].
    sub_tensor[1, 0, 0, 1, 0] = 3.0
    sub_tensor[1, 1, 0, 1, 0] = 1.0

    df_ad = main_site.get_df_ad(sub_tensor=sub_tensor, g=g, mode="nsy")
    assert df_ad[["group", "state_from", "state_to"]].to_records(index=False).tolist() == [
        ("nsy", "A", "B"),
        ("nsy", "B", "A"),
    ]
    numpy.testing.assert_allclose(df_ad["value"].to_numpy(), [3.0, 4.0], atol=1e-12)

    df_ad = main_site.add_site_stats(df_ad, sub_tensor, g, "nsy", method="tau")
    df_ad = main_site.add_site_stats(df_ad, sub_tensor, g, "nsy", method="tsi")
    df_ad = main_site.add_site_stats(df_ad, sub_tensor, g, "nsy", method="rank1")
    df_ad = main_site.add_site_stats(df_ad, sub_tensor, g, "nsy", method="rank2")
    numpy.testing.assert_allclose(df_ad["site_tau"].to_numpy(), [0.5, 2.0 / 3.0], atol=1e-12)
    numpy.testing.assert_allclose(df_ad["site_tsi"].to_numpy(), [2.0 / 3.0, 3.0 / 4.0], atol=1e-12)
    numpy.testing.assert_allclose(df_ad["site_rank1"].to_numpy(), [2.0, 3.0], atol=1e-12)
    numpy.testing.assert_allclose(df_ad["site_rank2"].to_numpy(), [1.0, 1.0], atol=1e-12)

    flagged = main_site.add_has_target_high_combinat_prob_site(df_ad, sub_tensor, g, "nsy")
    assert flagged["has_target_high_combinat_prob_site"].tolist() == [True, True]


def test_add_site_stats_hg_ignores_zero_probabilities():
    g = {"amino_acid_orders": numpy.array(["A", "B"]), "matrix_groups": {"grp": ["AA", "AB"]}}
    sub_tensor = numpy.zeros((2, 2, 1, 2, 2), dtype=float)
    # A->B totals per site = [1, 0], so entropy should be 0, not NaN.
    sub_tensor[0, 0, 0, 0, 1] = 1.0
    df_ad = main_site.get_df_ad(sub_tensor=sub_tensor, g=g, mode="nsy")
    out = main_site.add_site_stats(df_ad, sub_tensor, g, "nsy", method="hg")
    assert pytest.approx(float(out["site_hg"].iloc[0]), abs=1e-12) == 0.0
    assert pandas.isna(out["site_hg"].iloc[1])


def test_add_site_stats_tau_single_site_returns_zero_not_nan():
    g = {"amino_acid_orders": numpy.array(["A", "B"]), "matrix_groups": {"grp": ["AA", "AB"]}}
    sub_tensor = numpy.zeros((2, 1, 1, 2, 2), dtype=float)
    # Single-site profile; tau denominator would otherwise be 0.
    sub_tensor[0, 0, 0, 0, 1] = 1.0
    sub_tensor[1, 0, 0, 0, 1] = 1.0
    df_ad = main_site.get_df_ad(sub_tensor=sub_tensor, g=g, mode="nsy")
    out = main_site.add_site_stats(df_ad, sub_tensor, g, "nsy", method="tau")
    assert pytest.approx(float(out["site_tau"].iloc[0]), abs=1e-12) == 0.0
    assert pandas.isna(out["site_tau"].iloc[1])


def test_get_df_dist_reports_max_distance_for_multi_branch_substitutions(tiny_tree):
    g = {"tree": tiny_tree, "amino_acid_orders": numpy.array(["A", "B"]), "matrix_groups": {"grp": ["AA", "AB"]}}
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    sub_tensor = numpy.zeros((num_node, 1, 1, 2, 2), dtype=float)
    a_id = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse() if n.name == "A"][0]
    c_id = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse() if n.name == "C"][0]
    b_id = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse() if n.name == "B"][0]
    sub_tensor[a_id, 0, 0, 0, 1] = 0.6
    sub_tensor[c_id, 0, 0, 0, 1] = 0.6
    sub_tensor[b_id, 0, 0, 1, 0] = 0.7
    out = main_site.get_df_dist(sub_tensor=sub_tensor, g=g, mode="nsy")
    row_ab = out.loc[(out["state_from"] == "A") & (out["state_to"] == "B"), :]
    row_ba = out.loc[(out["state_from"] == "B") & (out["state_to"] == "A"), :]
    assert pytest.approx(float(row_ab["max_dist_bl"].iloc[0]), rel=0, abs=1e-12) == 1.0
    assert pandas.isna(row_ba["max_dist_bl"].iloc[0])


def test_get_substitution_tensor_asis_matches_manual_outer_products():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    state = numpy.zeros((3, 2, 2), dtype=float)
    state[labels["R"], :, :] = [[1.0, 0.0], [0.5, 0.5]]
    state[labels["A"], :, :] = [[0.0, 1.0], [1.0, 0.0]]
    state[labels["B"], :, :] = [[1.0, 0.0], [0.0, 1.0]]
    g = {"tree": tr, "ml_anc": "yes", "float_tol": 1e-12}
    mmap_file = Path("tmp.csubst.sub_tensor.toy.mmap")
    try:
        out = substitution.get_substitution_tensor(state_tensor=state, mode="asis", g=g, mmap_attr="toy")
        # Branch A, site 0: parent state 0 -> child state 1 with prob 1.
        numpy.testing.assert_allclose(out[labels["A"], 0, 0, :, :], [[0.0, 1.0], [0.0, 0.0]], atol=1e-12)
        # Branch A, site 1: parent [0.5, 0.5], child state 0 => only 1->0 survives diag masking.
        numpy.testing.assert_allclose(out[labels["A"], 1, 0, :, :], [[0.0, 0.0], [0.5, 0.0]], atol=1e-12)
    finally:
        if mmap_file.exists():
            mmap_file.unlink()


def test_apply_min_sub_pp_threshold():
    g = {"min_sub_pp": 0.3, "ml_anc": False}
    sub = numpy.array([[[[[0.2, 0.4], [0.1, 0.5]]]]], dtype=float)
    out = substitution.apply_min_sub_pp(g, sub)
    numpy.testing.assert_allclose(out, [[[[[0.0, 0.4], [0.0, 0.5]]]]], atol=1e-12)


def test_get_s_get_cs_and_get_bs_match_manual_values():
    # shape = [branch, site, group, from, to]
    sub = numpy.zeros((2, 2, 1, 2, 2), dtype=float)
    sub[0, 0, 0, :, :] = [[0.0, 0.2], [0.1, 0.0]]
    sub[1, 0, 0, :, :] = [[0.0, 0.5], [0.2, 0.0]]
    sub[0, 1, 0, :, :] = [[0.0, 0.4], [0.0, 0.0]]
    sub[1, 1, 0, :, :] = [[0.0, 0.1], [0.3, 0.0]]

    s = substitution.get_s(sub, attr="N")
    numpy.testing.assert_allclose(s["N_sub"].to_numpy(), [1.0, 0.8], atol=1e-12)

    cs = substitution.get_cs(numpy.array([[0, 1]]), sub, attr="N")
    numpy.testing.assert_allclose(cs["OCNany2any"].to_numpy(), [0.21, 0.16], atol=1e-12)
    numpy.testing.assert_allclose(cs["OCNspe2any"].to_numpy(), [0.12, 0.04], atol=1e-12)
    numpy.testing.assert_allclose(cs["OCNany2spe"].to_numpy(), [0.12, 0.04], atol=1e-12)
    numpy.testing.assert_allclose(cs["OCNspe2spe"].to_numpy(), [0.12, 0.04], atol=1e-12)

    bs = substitution.get_bs(S_tensor=sub, N_tensor=sub * 2.0)
    # first branch, first site
    row0 = bs.loc[(bs["branch_id"] == 0) & (bs["site"] == 0), :].iloc[0]
    assert pytest.approx(float(row0["S_sub"]), abs=1e-12) == 0.3
    assert pytest.approx(float(row0["N_sub"]), abs=1e-12) == 0.6


def test_get_b_uses_tree_numerical_labels_for_branch_ids(tiny_tree):
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    sub = numpy.zeros((num_node, 1, 1, 2, 2), dtype=float)
    a_id = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse() if n.name == "A"][0]
    sub[a_id, 0, 0, 0, 1] = 0.5

    out = substitution.get_b(g={"tree": tiny_tree, "num_node": num_node}, sub_tensor=sub, attr="S", sitewise=False)
    expected_ids = sorted([ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()])
    assert out["branch_id"].tolist() == expected_ids
    assert set(out["branch_name"]) == set([n.name for n in tiny_tree.traverse()])


def test_add_dif_column_and_add_dif_stats():
    cb = pandas.DataFrame(
        {
            "OCSany2any": [1.0, 1.0],
            "OCSany2spe": [0.4, 1.2],
            "OCSspe2any": [0.5, 0.2],
            "OCSspe2spe": [0.4, 0.2],
            "OCNany2any": [1.0, 1.0],
            "OCNany2spe": [0.5, 0.2],
            "OCNspe2any": [0.5, 0.2],
            "OCNspe2spe": [0.1, 0.3],
        }
    )
    out = substitution.add_dif_column(cb.copy(), "tmp", "OCSany2any", "OCSany2spe", tol=1e-6)
    numpy.testing.assert_allclose(out["tmp"].to_numpy(), [0.6, numpy.nan], equal_nan=True)

    out2 = substitution.add_dif_stats(cb.copy(), tol=1e-6, prefix="OC")
    assert "OCSany2dif" in out2.columns
    assert "OCNdif2spe" in out2.columns


def test_get_substitution_tensor_syn_matches_manual_groupwise_products():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    # codon state order: [AAA, AAG, TTT, TTC]
    # synonymous groups: K=[AAA,AAG], F=[TTT,TTC]
    state = numpy.zeros((3, 1, 4), dtype=float)
    state[labels["R"], 0, :] = [0.6, 0.4, 0.0, 0.0]
    state[labels["A"], 0, :] = [0.1, 0.9, 0.0, 0.0]
    state[labels["B"], 0, :] = [0.8, 0.2, 0.0, 0.0]
    g = {
        "tree": tr,
        "ml_anc": "yes",
        "float_tol": 1e-12,
        "amino_acid_orders": ["K", "F"],
        "synonymous_indices": {"K": [0, 1], "F": [2, 3]},
        "max_synonymous_size": 2,
    }
    mmap_file = Path("tmp.csubst.sub_tensor.toy_syn.mmap")
    try:
        out = substitution.get_substitution_tensor(state_tensor=state, mode="syn", g=g, mmap_attr="toy_syn")
        # Branch A, group K: diag masked outer product of parent [0.6,0.4] and child [0.1,0.9].
        numpy.testing.assert_allclose(out[labels["A"], 0, 0, :, :], [[0.0, 0.54], [0.04, 0.0]], atol=1e-12)
        # Group F has no support in this toy example.
        numpy.testing.assert_allclose(out[labels["A"], 0, 1, :, :], [[0.0, 0.0], [0.0, 0.0]], atol=1e-12)
        # Branch B, group K:
        numpy.testing.assert_allclose(out[labels["B"], 0, 0, :, :], [[0.0, 0.12], [0.32, 0.0]], atol=1e-12)
    finally:
        if mmap_file.exists():
            mmap_file.unlink()


def _toy_sub_tensor():
    # shape = [branch, site, group, from, to]
    sub = numpy.zeros((3, 2, 1, 2, 2), dtype=float)
    sub[0, 0, 0, :, :] = [[0.0, 0.2], [0.1, 0.0]]
    sub[1, 0, 0, :, :] = [[0.0, 0.5], [0.2, 0.0]]
    sub[2, 0, 0, :, :] = [[0.0, 0.4], [0.3, 0.0]]
    sub[0, 1, 0, :, :] = [[0.0, 0.1], [0.0, 0.0]]
    sub[1, 1, 0, :, :] = [[0.0, 0.1], [0.3, 0.0]]
    sub[2, 1, 0, :, :] = [[0.0, 0.2], [0.1, 0.0]]
    return sub


def test_sub_tensor2cb_mmap_chunk_writer_matches_non_mmap(tmp_path):
    sub = _toy_sub_tensor()
    ids = numpy.array([[2, 0], [1, 2]], dtype=numpy.int64)
    expected = substitution.sub_tensor2cb(ids, sub, mmap=False, df_mmap=None, mmap_start=0, float_type=numpy.float64)

    arity = ids.shape[1]
    mmap_path = tmp_path / "cb_writer.mmap"
    mmap_out = numpy.memmap(mmap_path, dtype=numpy.float64, mode="w+", shape=(ids.shape[0] + 1, arity + 4))
    mmap_out[:] = 0.0
    substitution.sub_tensor2cb(ids, sub, mmap=True, df_mmap=mmap_out, mmap_start=1, float_type=numpy.float64)
    observed = numpy.array(mmap_out[1 : 1 + ids.shape[0], :], copy=True)
    del mmap_out

    numpy.testing.assert_allclose(observed, expected, atol=1e-12)


def test_sub_tensor2cbs_mmap_chunk_writer_matches_non_mmap(tmp_path):
    sub = _toy_sub_tensor()
    ids = numpy.array([[2, 0], [1, 2]], dtype=numpy.int64)
    expected = substitution.sub_tensor2cbs(ids, sub, mmap=False, df_mmap=None, mmap_start=0)

    arity = ids.shape[1]
    num_site = sub.shape[1]
    mmap_path = tmp_path / "cbs_writer.mmap"
    mmap_rows = (ids.shape[0] + 1) * num_site
    mmap_out = numpy.memmap(mmap_path, dtype=numpy.float64, mode="w+", shape=(mmap_rows, arity + 5))
    mmap_out[:] = 0.0
    substitution.sub_tensor2cbs(ids, sub, mmap=True, df_mmap=mmap_out, mmap_start=1)
    row_start = num_site
    row_end = row_start + expected.shape[0]
    observed = numpy.array(mmap_out[row_start:row_end, :], copy=True)
    del mmap_out

    numpy.testing.assert_allclose(observed, expected, atol=1e-12)


def test_sub_tensor2cb_cython_fastpath_matches_python_fallback(monkeypatch):
    if not hasattr(substitution_cy, "calc_combinatorial_sub_double_arity2"):
        pytest.skip("Cython dense reducer fast path is unavailable")
    rng = numpy.random.default_rng(7)
    sub = rng.random((5, 4, 2, 3, 3), dtype=numpy.float64)
    ids = numpy.array([[0, 1], [2, 3], [4, 1]], dtype=numpy.int64)

    monkeypatch.setattr(substitution, "_can_use_cython_dense_cb", lambda *args, **kwargs: False)
    expected = substitution.sub_tensor2cb(ids, sub, mmap=False, df_mmap=None, mmap_start=0, float_type=numpy.float64)

    monkeypatch.setattr(substitution, "_can_use_cython_dense_cb", lambda *args, **kwargs: True)
    observed = substitution.sub_tensor2cb(ids, sub, mmap=False, df_mmap=None, mmap_start=0, float_type=numpy.float64)

    numpy.testing.assert_allclose(observed, expected, atol=1e-12)


def test_sub_tensor2cbs_cython_fastpath_matches_python_fallback(monkeypatch):
    if not hasattr(substitution_cy, "calc_combinatorial_sub_by_site_double_arity2"):
        pytest.skip("Cython dense reducer fast path is unavailable")
    rng = numpy.random.default_rng(11)
    sub = rng.random((4, 3, 2, 3, 3), dtype=numpy.float64)
    ids = numpy.array([[0, 1], [2, 3]], dtype=numpy.int64)

    monkeypatch.setattr(substitution, "_can_use_cython_dense_cbs", lambda *args, **kwargs: False)
    expected = substitution.sub_tensor2cbs(ids, sub, mmap=False, df_mmap=None, mmap_start=0)

    monkeypatch.setattr(substitution, "_can_use_cython_dense_cbs", lambda *args, **kwargs: True)
    observed = substitution.sub_tensor2cbs(ids, sub, mmap=False, df_mmap=None, mmap_start=0)

    numpy.testing.assert_allclose(observed, expected, atol=1e-12)


def test_get_cb_matches_sum_of_get_cs_per_combination():
    sub = _toy_sub_tensor()
    ids = numpy.array([[2, 0], [1, 2]], dtype=numpy.int64)
    cb = substitution.get_cb(ids, sub, {"threads": 1, "float_type": numpy.float64}, attr="OCN")

    for combo in ids:
        c1, c2 = sorted(combo.tolist())
        row = cb.loc[(cb["branch_id_1"] == c1) & (cb["branch_id_2"] == c2), :].iloc[0]
        cs = substitution.get_cs(numpy.array([[c1, c2]], dtype=numpy.int64), sub, attr="N")
        assert pytest.approx(float(row["OCNany2any"]), abs=1e-12) == float(cs["OCNany2any"].sum())
        assert pytest.approx(float(row["OCNspe2any"]), abs=1e-12) == float(cs["OCNspe2any"].sum())
        assert pytest.approx(float(row["OCNany2spe"]), abs=1e-12) == float(cs["OCNany2spe"].sum())
        assert pytest.approx(float(row["OCNspe2spe"]), abs=1e-12) == float(cs["OCNspe2spe"].sum())


def test_get_cbs_grouped_sum_matches_get_cb():
    sub = _toy_sub_tensor()
    ids = numpy.array([[2, 0], [1, 2]], dtype=numpy.int64)

    cb = substitution.get_cb(ids, sub, {"threads": 1, "float_type": numpy.float64}, attr="OCN")
    cbs = substitution.get_cbs(ids, sub, attr="N", g={"threads": 1})
    cols = ["OCNany2any", "OCNspe2any", "OCNany2spe", "OCNspe2spe"]
    summed = cbs.groupby(["branch_id_1", "branch_id_2"], as_index=False)[cols].sum()
    merged = cb.merge(summed, on=["branch_id_1", "branch_id_2"], suffixes=("_cb", "_cbs"))

    for col in cols:
        numpy.testing.assert_allclose(
            merged[f"{col}_cb"].to_numpy(),
            merged[f"{col}_cbs"].to_numpy(),
            atol=1e-12,
        )
