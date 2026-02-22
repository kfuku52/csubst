import builtins
import numpy as np
import pandas as pd
import pytest
import warnings

from csubst import main_simulate
from csubst import tree
from csubst import ete


def test_add_numerical_node_labels_assigns_unique_integers():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(B:1,(A:1,C:1)X:1)R;", format=1))
    labels = [ete.get_prop(n, "numerical_label") for n in tr.traverse()]
    assert sorted(labels) == list(range(len(labels)))


def test_add_numerical_node_labels_keeps_root_as_max_for_64_leaves():
    leaf_names = [f"L{i}" for i in range(64)]
    tree_txt = f"{leaf_names[0]}:1"
    for leaf_name in leaf_names[1:]:
        tree_txt = f"({tree_txt},{leaf_name}:1):1"
    tr = tree.add_numerical_node_labels(ete.PhyloNode(tree_txt + ";", format=1))

    nodes = list(tr.traverse())
    labels = [int(ete.get_prop(n, "numerical_label")) for n in nodes]
    assert sorted(labels) == list(range(len(nodes)))

    root_label = int(ete.get_prop(tr, "numerical_label"))
    nonroot = [n for n in nodes if not ete.is_root(n)]
    nonroot_labels = [int(ete.get_prop(n, "numerical_label")) for n in nonroot]
    assert root_label == len(nodes) - 1
    assert max(nonroot_labels) == len(nonroot) - 1
    assert root_label not in nonroot_labels


def test_is_consistent_tree_checks_leaf_sets():
    t1 = ete.PhyloNode("(A:1,B:1)R;", format=1)
    t2 = ete.PhyloNode("(B:1,A:1)R2;", format=1)
    t3 = ete.PhyloNode("(A:1,C:1)R3;", format=1)
    assert tree.is_consistent_tree(t1, t2)
    assert not tree.is_consistent_tree(t1, t3)


def test_standardize_node_names_removes_suffixes_and_quotes():
    tr = ete.PhyloNode("('A/1':1,'B[abc]':1)'N1[xy]':0;", format=1)
    out = tree.standardize_node_names(tr)
    names = sorted([n.name for n in out.traverse()])
    assert names == ["A", "B", "N1"]


def test_transfer_internal_node_names_copies_labels_by_topology():
    tree_to = ete.PhyloNode("(A:1,(B:1,C:1):1);", format=1)
    tree_from = ete.PhyloNode("(A:2,(B:2,C:2)X:2)R;", format=1)
    out = tree.transfer_internal_node_names(tree_to, tree_from)
    name_by_leafset = {tuple(sorted(ete.get_leaf_names(n))): n.name for n in out.traverse() if not ete.is_leaf(n)}
    assert name_by_leafset[("A", "B", "C")] == "R"
    assert name_by_leafset[("B", "C")] == "X"


def test_transfer_internal_node_names_rejects_different_topologies():
    tree_to = ete.PhyloNode("((A:1,B:1):1,(C:1,D:1):1);", format=1)
    tree_from = ete.PhyloNode("((A:1,C:1):1,(B:1,D:1):1);", format=1)
    with pytest.raises(AssertionError, match="RF distance"):
        tree.transfer_internal_node_names(tree_to, tree_from)


def test_transfer_root_rejects_missing_root_bipartition():
    tree_to = ete.PhyloNode("((A:1,C:1):1,(B:1,D:1):1);", format=1)
    tree_from = ete.PhyloNode("((A:1,B:1):1,(C:1,D:1):1);", format=1)
    with pytest.raises(ValueError, match="No root bipartition"):
        tree.transfer_root(tree_to=tree_to, tree_from=tree_from)


def test_transfer_root_rejects_non_bifurcating_source_root():
    tree_to = ete.PhyloNode("((A:1,B:1):1,C:1);", format=1)
    tree_from = ete.PhyloNode("(A:1,B:1,C:1);", format=1)
    with pytest.raises(ValueError, match="bifurcating"):
        tree.transfer_root(tree_to=tree_to, tree_from=tree_from)


def test_read_treefile_rejects_unrooted_tree(tmp_path):
    tree_file = tmp_path / "unrooted.nwk"
    tree_file.write_text("(A:1,B:1,C:1);", encoding="utf-8")
    with pytest.raises(AssertionError, match="may be unrooted"):
        tree.read_treefile({"rooted_tree_file": str(tree_file)})


def test_is_internal_node_labeled():
    labeled = ete.PhyloNode("(A:1,B:1)R;", format=1)
    unlabeled = ete.PhyloNode("(A:1,(B:1,C:1):1)R;", format=1)
    assert tree.is_internal_node_labeled(labeled)
    assert not tree.is_internal_node_labeled(unlabeled)


def test_is_internal_node_labeled_ignores_leaf_labels():
    tr = ete.PhyloNode("(A:1,B:1)R;", format=1)
    ete.get_leaves(tr)[0].name = ""
    assert tree.is_internal_node_labeled(tr)


def test_plot_branch_category_writes_pdf_with_matplotlib(tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,(B:1,C:1)X:1)R;", format=1))
    for node in tr.traverse():
        ete.set_prop(node, "color_trait", "black")
        ete.set_prop(node, "labelcolor_trait", "black")
    b_node = [n for n in tr.traverse() if n.name == "B"][0]
    ete.set_prop(b_node, "color_trait", "red")
    ete.set_prop(b_node, "labelcolor_trait", "red")
    g = {
        "tree": tr,
        "fg_df": pd.DataFrame({"lineage_id": [1], "trait": ["B"]}),
    }
    out_base = tmp_path / "branch_plot"
    tree.plot_branch_category(g, file_base=str(out_base), label="all")
    out_file = tmp_path / "branch_plot_trait.pdf"
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_state_tree_writes_site_pdfs_with_matplotlib(tmp_path, monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,(B:1,C:1)X:1)R;", format=1))
    for node in tr.traverse():
        ete.set_prop(node, "color_trait", "black")
        ete.set_prop(node, "labelcolor_trait", "black")
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    state = np.zeros((len(labels), 2, 2), dtype=float)
    state[labels["A"], 0, :] = [1.0, 0.0]
    state[labels["A"], 1, :] = [0.0, 1.0]
    state[labels["B"], 0, :] = [0.0, 1.0]
    state[labels["B"], 1, :] = [0.5, 0.5]  # Tie should render as missing state.
    state[labels["C"], 0, :] = [1.0, 0.0]
    state[labels["C"], 1, :] = [1.0, 0.0]
    state[labels["X"], 0, :] = [0.0, 1.0]
    state[labels["X"], 1, :] = [0.0, 0.0]  # Zero max should render as missing state.
    g = {
        "tree": tr,
        "fg_df": pd.DataFrame({"lineage_id": [1], "trait": ["B"]}),
    }
    monkeypatch.chdir(tmp_path)
    tree.plot_state_tree(state=state, orders=np.array(["K", "N"]), mode="aa", g=g)
    out_files = sorted(tmp_path.glob("csubst_state_trait_aa_*.pdf"))
    assert [p.name for p in out_files] == ["csubst_state_trait_aa_1.pdf", "csubst_state_trait_aa_2.pdf"]
    assert all(p.stat().st_size > 0 for p in out_files)


def test_get_num_adjusted_sites_does_not_mutate_parent_state_tensor():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    nodes = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    a_node = [n for n in tr.traverse() if n.name == "A"][0]
    state_cdn = np.zeros((len(nodes), 1, 2), dtype=float)
    state_cdn[nodes["R"], 0, :] = [0.6, 0.4]
    state_cdn[nodes["A"], 0, :] = [0.0, 0.0]
    state_cdn[nodes["B"], 0, :] = [1.0, 0.0]
    g = {
        "state_cdn": state_cdn.copy(),
        "codon_orders": np.array(["AAA", "AAG"]),
        "codon_table": [["K", "AAA"], ["K", "AAG"]],
        "instantaneous_codon_rate_matrix": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
    }
    parent_before = g["state_cdn"][nodes["R"], :, :].copy()
    tree.get_num_adjusted_sites(g=g, node=a_node)
    np.testing.assert_allclose(g["state_cdn"][nodes["R"], :, :], parent_before, atol=1e-12)


def test_get_nice_scale_length_returns_expected_steps():
    assert tree._get_nice_scale_length(0.0) == 1.0
    assert pytest.approx(tree._get_nice_scale_length(1.0), rel=0, abs=1e-12) == 0.2
    assert pytest.approx(tree._get_nice_scale_length(10.0), rel=0, abs=1e-12) == 2.0


def test_get_pyvolve_codon_order_contains_61_sense_codons():
    codons = main_simulate.get_pyvolve_codon_order()
    assert codons.shape == (61,)
    assert set(["TAA", "TAG", "TGA"]).isdisjoint(set(codons))


def test_get_codons_extracts_expected_members():
    codon_table = [["K", "AAA"], ["K", "AAG"], ["N", "AAC"], ["*", "TAA"]]
    out = main_simulate.get_codons("K", codon_table)
    np.testing.assert_array_equal(out, ["AAA", "AAG"])


def test_biased_index_helpers_match_manual_pairs():
    codon_table = [["K", "AAA"], ["K", "AAG"], ["N", "AAC"], ["*", "TAA"]]
    codon_order = np.array(["AAA", "AAG", "AAC"])
    biased_aas = np.array(["K"])
    nsy_idx = main_simulate.get_biased_nonsynonymous_substitution_index(biased_aas, codon_table, codon_order)
    np.testing.assert_array_equal(nsy_idx, [[2, 0], [2, 1]])
    cdn_idx = main_simulate.get_biased_codon_index(biased_aas, codon_table, codon_order)
    np.testing.assert_array_equal(cdn_idx, [[0], [1]])


def test_get_synonymous_codon_substitution_index_manual_count():
    g = {"codon_table": [["K", "AAA"], ["K", "AAG"], ["N", "AAC"]]}
    codon_order = np.array(["AAA", "AAG", "AAC"])
    out = main_simulate.get_synonymous_codon_substitution_index(g, codon_order)
    np.testing.assert_array_equal(out, [[0, 1], [1, 0]])


def test_get_total_Q_sums_requested_entries():
    mat = np.array([[0.0, 1.0], [2.0, 0.0]])
    idx = np.array([[0, 1], [1, 0]])
    assert main_simulate.get_total_Q(mat, idx) == 3.0


def test_rescale_substitution_matrix_preserves_total_and_zero_row_sum():
    mat = np.array([[0.0, 2.0, 1.0], [3.0, 0.0, 4.0], [5.0, 6.0, 0.0]])
    idx = np.array([[0, 1], [1, 2]])
    out = main_simulate.rescale_substitution_matrix(mat, idx, scaling_factor=2.0)
    offdiag_before = mat[~np.eye(mat.shape[0], dtype=bool)].sum()
    offdiag_after = out[~np.eye(out.shape[0], dtype=bool)].sum()
    assert pytest.approx(offdiag_after, rel=0, abs=1e-12) == offdiag_before
    np.testing.assert_allclose(out.sum(axis=1), [0.0, 0.0, 0.0], atol=1e-12)


def test_bias_eq_freq_scales_and_renormalizes():
    eq = np.array([0.2, 0.3, 0.5], dtype=float)
    out = main_simulate.bias_eq_freq(eq, biased_cdn_index=np.array([[0], [1]]), convergence_intensity_factor=2.0)
    # pre-normalization: [0.4, 0.6, 0.5] -> normalized by 1.5
    np.testing.assert_allclose(out, [4 / 15, 6 / 15, 5 / 15], atol=1e-12)


def test_get_biased_amino_acids_rejects_oversized_random_request():
    codon_table = [["K", "AAA"], ["K", "AAG"], ["N", "AAC"]]
    with pytest.raises(ValueError, match="exceeds available amino acids"):
        main_simulate.get_biased_amino_acids("random3", codon_table)


def test_get_biased_amino_acids_rejects_unknown_symbol():
    codon_table = [["K", "AAA"], ["K", "AAG"], ["N", "AAC"]]
    with pytest.raises(ValueError, match="Unknown amino acid"):
        main_simulate.get_biased_amino_acids("Z", codon_table)


def test_apply_percent_biased_sub_preserves_foreground_omega():
    mat = np.array(
        [[-1.0, 0.5, 0.5], [0.5, -1.0, 0.5], [0.5, 0.5, -1.0]],
        dtype=float,
    )
    all_syn = np.array([[0, 1], [1, 0]])
    all_nsy = np.array([[0, 2], [1, 2], [2, 0], [2, 1]])
    codon_table = [["K", "AAA"], ["K", "AAG"], ["N", "AAC"]]
    out = main_simulate.apply_percent_biased_sub(
        mat=mat,
        percent_biased_sub=20,
        target_index=np.array([[2, 0], [2, 1]]),
        biased_aas=np.array(["K"]),
        codon_table=codon_table,
        codon_orders=np.array(["AAA", "AAG", "AAC"]),
        all_nsy_cdn_index=all_nsy,
        all_syn_cdn_index=all_syn,
        foreground_omega=2.0,
    )
    dnds = main_simulate.get_total_Q(out, all_nsy) / main_simulate.get_total_Q(out, all_syn)
    assert pytest.approx(dnds, rel=0, abs=1e-12) == 2.0
    np.testing.assert_allclose(out.sum(axis=1), [0.0, 0.0, 0.0], atol=1e-12)


def test_apply_percent_biased_sub_with_no_targets_rescales_omega_without_nan():
    mat = np.array(
        [[-1.0, 0.5, 0.5], [0.5, -1.0, 0.5], [0.5, 0.5, -1.0]],
        dtype=float,
    )
    all_syn = np.array([[0, 1], [1, 0]])
    all_nsy = np.array([[0, 2], [1, 2], [2, 0], [2, 1]])
    codon_table = [["K", "AAA"], ["K", "AAG"], ["N", "AAC"]]
    out = main_simulate.apply_percent_biased_sub(
        mat=mat,
        percent_biased_sub=20,
        target_index=np.zeros((0, 2), dtype=int),
        biased_aas=np.array([], dtype=str),
        codon_table=codon_table,
        codon_orders=np.array(["AAA", "AAG", "AAC"]),
        all_nsy_cdn_index=all_nsy,
        all_syn_cdn_index=all_syn,
        foreground_omega=2.0,
    )
    dnds = main_simulate.get_total_Q(out, all_nsy) / main_simulate.get_total_Q(out, all_syn)
    assert np.isfinite(out).all()
    assert pytest.approx(dnds, rel=0, abs=1e-12) == 2.0


@pytest.mark.parametrize(
    "g,pattern",
    [
        ({"num_simulated_site": 0, "percent_convergent_site": 10, "percent_biased_sub": 90}, "num_simulated_site"),
        ({"num_simulated_site": 10, "percent_convergent_site": -1, "percent_biased_sub": 90}, "percent_convergent_site"),
        ({"num_simulated_site": 10, "percent_convergent_site": 101, "percent_biased_sub": 90}, "percent_convergent_site"),
        ({"num_simulated_site": 10, "percent_convergent_site": 10, "percent_biased_sub": 100}, "percent_biased_sub"),
    ],
)
def test_validate_simulate_params_rejects_out_of_range_values(g, pattern):
    with pytest.raises(ValueError, match=pattern):
        main_simulate._validate_simulate_params(g)


def test_get_pyvolve_newick_marks_foreground_without_mutating_distances():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:0.1,B:0.2)R;", format=1))
    for node in tr.traverse():
        ete.add_features(node, **{"is_fg_t": False, "foreground_lineage_id_t": 0})
    a_node = [n for n in tr.traverse() if n.name == "A"][0]
    ete.set_prop(a_node, "is_fg_t", True)
    ete.set_prop(a_node, "foreground_lineage_id_t", 1)
    out = main_simulate.get_pyvolve_newick(tr, "t")
    assert "#m1" in out
    assert pytest.approx(a_node.dist, rel=0, abs=1e-12) == 0.1


def test_scale_tree_multiplies_every_branch_length():
    tr = ete.PhyloNode("(A:1.0,B:2.0)R;", format=1)
    out = main_simulate.scale_tree(tr, 3.0)
    dists = sorted([n.dist for n in out.traverse()])
    assert dists == [0.0, 3.0, 6.0]


def test_get_background_Q_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unsupported Q matrix method"):
        main_simulate.get_background_Q({}, Q_method="unknown_method")


def test_concatenate_alignment_concatenates_matching_headers(tmp_path):
    in1 = tmp_path / "a.fa"
    in2 = tmp_path / "b.fa"
    out = tmp_path / "out.fa"
    in1.write_text(">A\nAAA\n>B\nCCC\n", encoding="utf-8")
    in2.write_text(">A\nGGG\n>B\nTTT\n", encoding="utf-8")
    main_simulate.concatenate_alignment(str(in1), str(in2), str(out))
    assert out.read_text(encoding="utf-8") == ">A\nAAAGGG\n>B\nCCCTTT\n"


def test_concatenate_alignment_rejects_mismatched_headers(tmp_path):
    in1 = tmp_path / "a.fa"
    in2 = tmp_path / "b.fa"
    out = tmp_path / "out.fa"
    in1.write_text(">A\nAAA\n", encoding="utf-8")
    in2.write_text(">B\nGGG\n", encoding="utf-8")
    with pytest.raises(ValueError, match="FASTA headers differ"):
        main_simulate.concatenate_alignment(str(in1), str(in2), str(out))


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


def test_main_simulate_plot_uses_foreground_annotation(monkeypatch):
    captured = {"colored": False}

    def fake_read_treefile(local_g):
        tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
        local_g["tree"] = tr
        local_g["rooted_tree"] = tr
        return local_g

    def fake_passthrough(local_g, *args, **kwargs):
        return local_g

    def fake_read_input(local_g):
        local_g["num_input_site"] = 3
        return local_g

    def fake_get_foreground_branch(local_g, simulate=False):
        local_g["fg_df"] = pd.DataFrame({"name": ["A"], "PLACEHOLDER": [1]})
        for node in local_g["tree"].traverse():
            if node.name == "A":
                ete.set_prop(node, "is_fg_PLACEHOLDER", True)
                ete.set_prop(node, "foreground_lineage_id_PLACEHOLDER", 1)
                ete.set_prop(node, "color_PLACEHOLDER", "red")
                ete.set_prop(node, "labelcolor_PLACEHOLDER", "red")
            else:
                ete.set_prop(node, "is_fg_PLACEHOLDER", False)
                ete.set_prop(node, "foreground_lineage_id_PLACEHOLDER", 0)
                ete.set_prop(node, "color_PLACEHOLDER", "black")
                ete.set_prop(node, "labelcolor_PLACEHOLDER", "black")
        return local_g

    def fake_plot_branch_category(local_g, file_base, label="all"):
        colors = [ete.get_prop(n, "color_PLACEHOLDER", "black") for n in local_g["tree"].traverse()]
        captured["colored"] = any(c != "black" for c in colors)
        raise RuntimeError("stop_after_plot")

    monkeypatch.setattr(main_simulate.tree, "read_treefile", fake_read_treefile)
    monkeypatch.setattr(main_simulate.parser_misc, "generate_intermediate_files", fake_passthrough)
    monkeypatch.setattr(main_simulate.parser_misc, "annotate_tree", fake_passthrough)
    monkeypatch.setattr(main_simulate.parser_misc, "read_input", fake_read_input)
    monkeypatch.setattr(main_simulate.foreground, "get_foreground_branch", fake_get_foreground_branch)
    monkeypatch.setattr(main_simulate.tree, "plot_branch_category", fake_plot_branch_category)

    g = {
        "genetic_code": 1,
        "alignment_file": "dummy.fa",
        "rooted_tree_file": "dummy.nwk",
        "foreground": "dummy_fg.txt",
        "fg_format": 1,
        "num_simulated_site": 10,
        "percent_convergent_site": 0,
        "percent_biased_sub": 90,
        "optimized_branch_length": True,
        "tree_scaling_factor": 1.0,
        "foreground_scaling_factor": 1.0,
    }
    with pytest.raises(RuntimeError, match="stop_after_plot"):
        main_simulate.main_simulate(g)
    assert captured["colored"] is True


def test_require_pyvolve_raises_clear_optional_dependency_message(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyvolve":
            raise ModuleNotFoundError("No module named 'pyvolve'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(main_simulate, "_PYVOLVE", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match=r'optional dependency for `csubst simulate`'):
        main_simulate._require_pyvolve()
    with pytest.raises(ModuleNotFoundError, match=r'csubst\[simulate\]'):
        main_simulate._require_pyvolve()


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


def test_rescale_branch_length_adjusted_site_keeps_ndist_when_only_nonsynonymous_substitutions(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:2.0,B:2.0)R;", format=1))
    node_by_name = {n.name: n for n in tr.traverse() if n.name}
    a_node = node_by_name["A"]
    a_nl = int(ete.get_prop(a_node, "numerical_label"))
    num_nodes = len(list(tr.traverse()))
    g = {
        "tree": tr,
        "state_cdn": np.ones((num_nodes, 1, 1), dtype=float),
        "float_tol": 1e-12,
    }
    os_counts = np.zeros(num_nodes, dtype=float)
    on_counts = np.zeros(num_nodes, dtype=float)
    on_counts[a_nl] = 5.0
    os_tensor = object()
    on_tensor = object()

    def fake_get_branch_sub_counts(tensor):
        if tensor is os_tensor:
            return os_counts
        if tensor is on_tensor:
            return on_counts
        raise AssertionError("Unexpected tensor object")

    monkeypatch.setattr(tree.substitution, "get_branch_sub_counts", fake_get_branch_sub_counts)
    monkeypatch.setattr(tree, "get_num_adjusted_sites", lambda local_g, node: (1.0, 1.0))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tree.rescale_branch_length(g, os_tensor, on_tensor, denominator="adjusted_site")
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []
    assert ete.get_prop(a_node, "Sdist") == pytest.approx(0.0, abs=1e-12)
    assert ete.get_prop(a_node, "Ndist") == pytest.approx(a_node.dist, abs=1e-12)
    assert ete.get_prop(a_node, "SNdist") == pytest.approx(a_node.dist, abs=1e-12)


def test_rescale_branch_length_adjusted_site_keeps_nonzero_component_when_other_adjusted_site_is_zero(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:2.0,B:2.0)R;", format=1))
    node_by_name = {n.name: n for n in tr.traverse() if n.name}
    a_node = node_by_name["A"]
    a_nl = int(ete.get_prop(a_node, "numerical_label"))
    num_nodes = len(list(tr.traverse()))
    g = {
        "tree": tr,
        "state_cdn": np.ones((num_nodes, 1, 1), dtype=float),
        "float_tol": 1e-12,
    }
    os_counts = np.zeros(num_nodes, dtype=float)
    on_counts = np.zeros(num_nodes, dtype=float)
    on_counts[a_nl] = 3.0
    os_tensor = object()
    on_tensor = object()

    def fake_get_branch_sub_counts(tensor):
        if tensor is os_tensor:
            return os_counts
        if tensor is on_tensor:
            return on_counts
        raise AssertionError("Unexpected tensor object")

    monkeypatch.setattr(tree.substitution, "get_branch_sub_counts", fake_get_branch_sub_counts)
    # No adjusted synonymous opportunities but valid adjusted nonsynonymous opportunities.
    monkeypatch.setattr(tree, "get_num_adjusted_sites", lambda local_g, node: (0.0, 1.0))

    tree.rescale_branch_length(g, os_tensor, on_tensor, denominator="adjusted_site")
    assert ete.get_prop(a_node, "Sdist") == pytest.approx(0.0, abs=1e-12)
    assert ete.get_prop(a_node, "Ndist") == pytest.approx(a_node.dist, abs=1e-12)
    assert ete.get_prop(a_node, "SNdist") == pytest.approx(a_node.dist, abs=1e-12)
