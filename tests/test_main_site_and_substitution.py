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


def test_resolve_site_jobs_each_and_all_modes(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}

    g_each = {"tree": tiny_tree, "mode": "each", "branch_id": "{},{}".format(labels["A"], labels["C"])}
    out_each = main_site.resolve_site_jobs(g_each)
    each_jobs = out_each["site_jobs"]
    assert len(each_jobs) == 2
    assert [int(job["branch_ids"][0]) for job in each_jobs] == [labels["A"], labels["C"]]
    assert all(job["single_branch_mode"] for job in each_jobs)

    g_all = {"tree": tiny_tree, "mode": "all", "branch_id": "ignored"}
    out_all = main_site.resolve_site_jobs(g_all)
    all_jobs = out_all["site_jobs"]
    expected_nonroot = sorted(
        [int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse() if not ete.is_root(n)]
    )
    assert [int(job["branch_ids"][0]) for job in all_jobs] == expected_nonroot
    assert all(job["single_branch_mode"] for job in all_jobs)


def test_resolve_site_jobs_all_mode_without_branch_id(tiny_tree):
    g_all = {"tree": tiny_tree, "mode": "all"}
    out_all = main_site.resolve_site_jobs(g_all)
    assert len(out_all["site_jobs"]) == len([n for n in tiny_tree.traverse() if not ete.is_root(n)])


def test_resolve_site_jobs_clade_mode_returns_all_nonroot_clade_branches(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "clade", "branch_id": str(labels["X"])}
    out = main_site.resolve_site_jobs(g)
    got = sorted([int(job["branch_ids"][0]) for job in out["site_jobs"]])
    expected = sorted([labels["X"], labels["A"], labels["C"]])
    assert got == expected


def test_resolve_site_jobs_lineage_mode_returns_ancestor_to_descendant_path(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "lineage", "branch_id": "{},{}".format(labels["X"], labels["C"])}
    out = main_site.resolve_site_jobs(g)
    assert len(out["site_jobs"]) == 1
    numpy.testing.assert_array_equal(out["site_jobs"][0]["branch_ids"], [labels["X"], labels["C"]])
    assert not out["site_jobs"][0]["single_branch_mode"]


def test_resolve_site_jobs_lineage_mode_rejects_non_ancestor_pairs(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "lineage", "branch_id": "{},{}".format(labels["B"], labels["C"])}
    with pytest.raises(ValueError, match="ancestor"):
        main_site.resolve_site_jobs(g)


def test_resolve_site_jobs_total_mode_preserves_branch_set_and_outdir_prefix(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "total", "branch_id": "{},{}".format(labels["A"], labels["C"])}
    out = main_site.resolve_site_jobs(g)
    assert len(out["site_jobs"]) == 1
    numpy.testing.assert_array_equal(out["site_jobs"][0]["branch_ids"], [labels["A"], labels["C"]])
    assert out["site_jobs"][0]["site_outdir"].startswith("./csubst_site.modetotal.branch_id")


def test_resolve_site_jobs_set_mode_extracts_expression_branch_ids(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    root_id = int(ete.get_prop(tiny_tree, "numerical_label"))
    g = {"tree": tiny_tree, "mode": "set,({}|{})-{}".format(labels["A"], labels["C"], root_id), "branch_id": "unused"}
    out = main_site.resolve_site_jobs(g)
    assert len(out["site_jobs"]) == 1
    numpy.testing.assert_array_equal(out["site_jobs"][0]["branch_ids"], sorted([labels["A"], labels["C"]]))
    assert out["mode"] == "set"
    assert out["mode_expression"] == "({}|{})-{}".format(labels["A"], labels["C"], root_id)
    assert out["site_jobs"][0]["site_outdir"].startswith("./csubst_site.modeset.expr")


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


def test_resolve_site_jobs_each_mode_rejects_fg(tiny_tree):
    g = {"tree": tiny_tree, "mode": "each", "branch_id": "fg"}
    with pytest.raises(ValueError, match="does not support"):
        main_site.resolve_site_jobs(g)


def test_get_state_orders():
    g = {"amino_acid_orders": numpy.array(["A", "B"]), "matrix_groups": {"grp": ["AA", "AB"]}}
    orders_nsy, keys_nsy = main_site.get_state_orders(g, "nsy")
    assert keys_nsy == ["nsy"]
    assert list(orders_nsy["nsy"]) == ["A", "B"]
    orders_syn, keys_syn = main_site.get_state_orders(g, "syn")
    assert keys_syn == ["grp"]
    assert orders_syn["grp"] == ["AA", "AB"]

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
