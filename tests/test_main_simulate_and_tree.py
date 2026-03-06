import builtins
import numpy as np
import pandas as pd
import pytest
import warnings

from csubst import main_simulate
from csubst import runtime
from csubst import tree
from csubst import ete


def _patch_simulation_index_helpers(monkeypatch):
    monkeypatch.setattr(
        main_simulate,
        "get_synonymous_codon_substitution_index",
        lambda local_g, codon_order: np.zeros((0, 2), dtype=np.int64),
    )
    monkeypatch.setattr(
        main_simulate,
        "get_nonsynonymous_codon_substitution_index",
        lambda all_syn_cdn_index: np.zeros((0, 2), dtype=np.int64),
    )


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
    out_file = tmp_path / "csubst_state_trait_aa_all.pdf"
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_state_tree_supports_site_selection_formats(tmp_path, monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,(B:1,C:1)X:1)R;", format=1))
    for node in tr.traverse():
        ete.set_prop(node, "color_trait", "black")
        ete.set_prop(node, "labelcolor_trait", "black")
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    state = np.zeros((len(labels), 3, 2), dtype=float)
    state[:, :, 0] = 1.0
    g = {
        "tree": tr,
        "fg_df": pd.DataFrame({"lineage_id": [1], "trait": ["B"]}),
    }
    monkeypatch.chdir(tmp_path)
    tree.plot_state_tree(state=state, orders=np.array(["K", "N"]), mode="aa", g=g, plot_request="1,3")
    pages_file = tmp_path / "csubst_state_trait_aa_1,3.pdf"
    assert pages_file.exists()
    assert pages_file.stat().st_size > 0
    pages_file.unlink()
    tree.plot_state_tree(state=state, orders=np.array(["K", "N"]), mode="aa", g=g, plot_request="1-3")
    concat_file = tmp_path / "csubst_state_trait_aa_1-3.pdf"
    assert concat_file.exists()
    assert concat_file.stat().st_size > 0


def test_plot_state_tree_hyphen_request_concatenates_site_labels(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    state = np.zeros((len(labels), 3, 3), dtype=float)
    state[:, 0, 0] = 1.0
    state[:, 1, 1] = 1.0
    state[:, 2, 2] = 1.0
    captured = {}

    def fake_render(tree=None, trait_name=None, file_name=None, label='all', state_by_node=None,
                    state_prob_by_node=None, state_orders=None, state_mode=None,
                    pdf_pages=None, figure_title=None):
        captured["file_name"] = str(file_name)
        captured["figure_title"] = figure_title
        captured["state_by_node"] = dict(state_by_node)

    monkeypatch.setattr(tree, "_render_tree_matplotlib", fake_render)

    g = {
        "tree": tr,
        "fg_df": pd.DataFrame({"lineage_id": [1], "trait": ["A"]}),
    }
    tree.plot_state_tree(
        state=state,
        orders=np.array(["AAC", "TCT", "GAC"], dtype=object),
        mode="codon",
        g=g,
        plot_request="1-2-3",
    )
    assert captured["file_name"].endswith("csubst_state_trait_codon_1-2-3.pdf")
    assert captured["figure_title"] == "Sites 1-2-3"
    assert captured["state_by_node"][labels["A"]] == "AACTCTGAC"


def test_plot_state_tree_hyphen_request_keeps_aa_seqlogo_probabilities(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    state = np.zeros((len(labels), 3, 3), dtype=float)
    state[:, 0, :] = [0.7, 0.2, 0.1]
    state[:, 1, :] = [0.1, 0.8, 0.1]
    state[:, 2, :] = [0.2, 0.3, 0.5]
    captured = {}

    def fake_render(tree=None, trait_name=None, file_name=None, label='all', state_by_node=None,
                    state_prob_by_node=None, state_orders=None, state_mode=None,
                    pdf_pages=None, figure_title=None):
        captured["state_mode"] = state_mode
        captured["state_orders"] = tuple(np.asarray(state_orders, dtype=object).tolist()) if state_orders is not None else None
        captured["state_prob_shape"] = np.asarray(state_prob_by_node[labels["A"]]).shape
        captured["state_by_node"] = dict(state_by_node)

    monkeypatch.setattr(tree, "_render_tree_matplotlib", fake_render)

    g = {
        "tree": tr,
        "fg_df": pd.DataFrame({"lineage_id": [1], "trait": ["A"]}),
    }
    tree.plot_state_tree(
        state=state,
        orders=np.array(["A", "C", "D"], dtype=object),
        mode="aa",
        g=g,
        plot_request="1-2-3",
    )
    assert captured["state_mode"] == "aa"
    assert captured["state_orders"] == ("A", "C", "D")
    assert captured["state_prob_shape"] == (3, 3)
    assert captured["state_by_node"][labels["A"]] == "ACD"


class _FakeTreeAxis:
    def __init__(self):
        self.text_calls = []

    def plot(self, *args, **kwargs):
        return None

    def text(self, x, y, txt, **kwargs):
        self.text_calls.append(str(txt))
        return None

    def set_xlim(self, *args, **kwargs):
        return None

    def set_ylim(self, *args, **kwargs):
        return None

    def axis(self, *args, **kwargs):
        return None


class _FakeTreeFigure:
    def savefig(self, *args, **kwargs):
        return None

    def suptitle(self, *args, **kwargs):
        return None


class _FakeTreePyplot:
    def __init__(self):
        self.axis = _FakeTreeAxis()
        self.figure = _FakeTreeFigure()

    def subplots(self, *args, **kwargs):
        return self.figure, self.axis

    def close(self, *args, **kwargs):
        return None


def test_render_tree_matplotlib_hides_missing_root_state_for_codon(monkeypatch, tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    for node in tr.traverse():
        ete.set_prop(node, "color_trait", "black")
        ete.set_prop(node, "labelcolor_trait", "black")
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    fake_plt = _FakeTreePyplot()
    monkeypatch.setattr(tree, "_get_pyplot", lambda: fake_plt)

    tree._render_tree_matplotlib(
        tree=tr,
        trait_name="trait",
        file_name=str(tmp_path / "state_codon.pdf"),
        state_by_node={
            labels["R"]: "---",
            labels["A"]: "AAA",
            labels["B"]: "AAG",
        },
        state_mode="codon",
    )

    assert "---" not in fake_plt.axis.text_calls
    assert "AAA|A" in fake_plt.axis.text_calls
    assert "AAG|B" in fake_plt.axis.text_calls


def test_render_tree_matplotlib_hides_missing_root_state_for_aa(monkeypatch, tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    for node in tr.traverse():
        ete.set_prop(node, "color_trait", "black")
        ete.set_prop(node, "labelcolor_trait", "black")
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    fake_plt = _FakeTreePyplot()
    monkeypatch.setattr(tree, "_get_pyplot", lambda: fake_plt)
    monkeypatch.setattr(tree, "_get_logo_modules", lambda: (None, None, None, None))

    tree._render_tree_matplotlib(
        tree=tr,
        trait_name="trait",
        file_name=str(tmp_path / "state_aa.pdf"),
        state_by_node={
            labels["R"]: "-",
            labels["A"]: "K",
            labels["B"]: "N",
        },
        state_prob_by_node={
            labels["R"]: None,
            labels["A"]: None,
            labels["B"]: None,
        },
        state_orders=np.array(["K", "N"], dtype=object),
        state_mode="aa",
    )

    assert "-" not in fake_plt.axis.text_calls
    assert "K" in fake_plt.axis.text_calls
    assert "N" in fake_plt.axis.text_calls


def test_normalize_state_plot_request_rejects_legacy_yes():
    with pytest.raises(ValueError, match="no longer accepts yes/no"):
        tree.normalize_state_plot_request("yes", param_name="--plot_state_aa")


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


def test_rescale_substitution_matrix_with_eq_freq_normalizes_expected_rate():
    mat = np.array([[-1.0, 1.0], [2.0, -2.0]], dtype=float)
    idx = np.array([[0, 1]], dtype=int)
    eq = np.array([0.9, 0.1], dtype=float)
    out = main_simulate.rescale_substitution_matrix(mat, idx, scaling_factor=2.0, eq_freq=eq)
    np.testing.assert_allclose(out.sum(axis=1), [0.0, 0.0], atol=1e-12)
    expected_rate = float(np.sum(eq * (-np.diag(out))))
    assert expected_rate == pytest.approx(1.0, abs=1e-12)


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


def test_get_biased_amino_acids_random_mode_is_reproducible_with_rng():
    codon_table = [["K", "AAA"], ["K", "AAG"], ["N", "AAC"], ["F", "TTT"], ["F", "TTC"]]
    rng_a = np.random.default_rng(17)
    rng_b = np.random.default_rng(17)
    out_a1 = main_simulate.get_biased_amino_acids("random1", codon_table, rng=rng_a)
    out_a2 = main_simulate.get_biased_amino_acids("random1", codon_table, rng=rng_a)
    out_b1 = main_simulate.get_biased_amino_acids("random1", codon_table, rng=rng_b)
    out_b2 = main_simulate.get_biased_amino_acids("random1", codon_table, rng=rng_b)
    np.testing.assert_array_equal(out_a1, out_b1)
    np.testing.assert_array_equal(out_a2, out_b2)


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


def test_evolve_nonconvergent_partition_forwards_seed_to_pyvolve(monkeypatch):
    call_kwargs = {}

    class DummyModel:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class DummyPartition:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class DummyEvolver:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def __call__(self, **kwargs):
            call_kwargs.update(kwargs)

    class DummyPyvolve:
        Model = DummyModel
        Partition = DummyPartition
        Evolver = DummyEvolver

    monkeypatch.setattr(main_simulate, "_PYVOLVE", DummyPyvolve())
    g = {
        "num_convergent_site": 0,
        "num_simulated_site": 5,
        "background_Q": np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=float),
        "background_tree": object(),
        "simulate_seed_nonconvergent": 101,
    }
    main_simulate.evolve_nonconvergent_partition(g)
    assert call_kwargs["seed"] == 101
    assert call_kwargs["seqfile"] == "tmp.csubst.simulate_nonconvergent.fa"


def test_resolve_simulation_site_rates_no_mode_returns_ones():
    g = {"num_simulated_site": 4, "simulate_asrv": "no"}
    out = main_simulate._resolve_simulation_site_rates(g)
    assert np.array_equal(out, np.ones(4, dtype=float))


def test_resolve_simulation_site_rates_file_mode_wraps_when_needed():
    g = {
        "num_simulated_site": 5,
        "simulate_asrv": "file",
        "iqtree_rate_values": np.array([0.5, 1.0, 2.0], dtype=float),
    }
    out = main_simulate._resolve_simulation_site_rates(g)
    assert np.allclose(out, np.array([0.5, 1.0, 2.0, 0.5, 1.0], dtype=float))


def test_resolve_simulation_background_omega_prefers_explicit_over_iqtree():
    g = {"background_omega": 0.7, "omega": 1.2}
    out = main_simulate._resolve_simulation_background_omega(g)
    assert out == pytest.approx(0.7, abs=1e-12)


def test_resolve_simulation_background_omega_falls_back_to_iqtree_then_default():
    g_iq = {"background_omega": None, "omega": 0.42}
    out_iq = main_simulate._resolve_simulation_background_omega(g_iq)
    assert out_iq == pytest.approx(0.42, abs=1e-12)
    g_default = {"background_omega": None, "omega": None}
    out_default = main_simulate._resolve_simulation_background_omega(g_default)
    assert out_default == pytest.approx(0.2, abs=1e-12)


def test_resolve_simulation_eq_freq_auto_prefers_iqtree_and_reorders(monkeypatch):
    class _DummyReadFrequencies:
        def __init__(self, *_args, **_kwargs):
            pass

        def compute_frequencies(self):
            raise AssertionError("alignment fallback should not be used when IQ-TREE frequencies exist")

    class _DummyPyvolve:
        ReadFrequencies = _DummyReadFrequencies

    monkeypatch.setattr(main_simulate, "_PYVOLVE", _DummyPyvolve())
    g = {
        "simulate_eq_freq": "auto",
        "equilibrium_frequency": np.array([0.2, 0.3, 0.5], dtype=float),
        "codon_orders": np.array(["AAA", "AAC", "AAG"], dtype=object),
        "pyvolve_codon_orders": np.array(["AAG", "AAA", "AAC"], dtype=object),
        "alignment_file": "dummy.fa",
    }
    out = main_simulate._resolve_simulation_eq_freq(g)
    np.testing.assert_allclose(out, np.array([0.5, 0.2, 0.3], dtype=float), atol=1e-12)


def test_resolve_simulation_eq_freq_alignment_mode_uses_alignment_frequencies(monkeypatch):
    class _DummyReadFrequencies:
        def __init__(self, *_args, **_kwargs):
            pass

        def compute_frequencies(self):
            return np.array([0.1, 0.2, 0.7], dtype=float)

    class _DummyPyvolve:
        ReadFrequencies = _DummyReadFrequencies

    monkeypatch.setattr(main_simulate, "_PYVOLVE", _DummyPyvolve())
    g = {
        "simulate_eq_freq": "alignment",
        "pyvolve_codon_orders": np.array(["AAA", "AAC", "AAG"], dtype=object),
        "alignment_file": "dummy.fa",
    }
    out = main_simulate._resolve_simulation_eq_freq(g)
    np.testing.assert_allclose(out, np.array([0.1, 0.2, 0.7], dtype=float), atol=1e-12)


def test_evolve_nonconvergent_partition_uses_sitewise_rates_when_asrv_file(monkeypatch):
    call_kwargs = {}
    model_matrices = []
    partition_sizes = []

    class DummyModel:
        def __init__(self, *args, **kwargs):
            model_matrices.append(np.array(kwargs["parameters"]["matrix"], dtype=float))

    class DummyPartition:
        def __init__(self, *args, **kwargs):
            partition_sizes.append(int(kwargs["size"]))

    class DummyEvolver:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def __call__(self, **kwargs):
            call_kwargs.update(kwargs)

    class DummyPyvolve:
        Model = DummyModel
        Partition = DummyPartition
        Evolver = DummyEvolver

    monkeypatch.setattr(main_simulate, "_PYVOLVE", DummyPyvolve())
    g = {
        "num_convergent_site": 1,
        "num_simulated_site": 4,
        "background_Q": np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=float),
        "background_tree": object(),
        "simulate_asrv": "file",
        "simulate_rate_nonconvergent": np.array([0.5, 2.0, 3.0], dtype=float),
        "simulate_seed_nonconvergent": 2026,
    }
    main_simulate.evolve_nonconvergent_partition(g)
    assert call_kwargs["seed"] == 2026
    assert partition_sizes == [1, 1, 1]
    assert len(model_matrices) == 3
    assert model_matrices[0][0, 1] == pytest.approx(0.5)
    assert model_matrices[1][0, 1] == pytest.approx(2.0)
    assert model_matrices[2][0, 1] == pytest.approx(3.0)
    assert model_matrices[0][0, 0] == pytest.approx(-0.5)
    assert model_matrices[1][0, 0] == pytest.approx(-2.0)
    assert model_matrices[2][0, 0] == pytest.approx(-3.0)


def test_evolve_nonconvergent_partition_reports_correct_site_range(capsys, monkeypatch):
    class DummyModel:
        def __init__(self, *args, **kwargs):
            pass

    class DummyPartition:
        def __init__(self, *args, **kwargs):
            pass

    class DummyEvolver:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, **kwargs):
            return None

    class DummyPyvolve:
        Model = DummyModel
        Partition = DummyPartition
        Evolver = DummyEvolver

    monkeypatch.setattr(main_simulate, "_PYVOLVE", DummyPyvolve())
    g = {
        "num_convergent_site": 3,
        "num_simulated_site": 10,
        "background_Q": np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=float),
        "background_tree": object(),
        "simulate_asrv": "no",
        "simulate_seed_nonconvergent": None,
    }
    main_simulate.evolve_nonconvergent_partition(g)
    out = capsys.readouterr().out
    assert "Codon site 4-10; Non-convergent codons" in out


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


def test_get_background_Q_auto_respects_target_omega_and_expected_rate(monkeypatch):
    exchangeability = np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]],
        dtype=float,
    )

    def _fake_read_exchangeability_matrix(_file, _codon_orders):
        return exchangeability

    monkeypatch.setattr(main_simulate.parser_misc, "read_exchangeability_matrix", _fake_read_exchangeability_matrix)
    g = {
        "substitution_model": "ECMrest+F+R4",
        "pyvolve_codon_orders": np.array(["AAA", "AAG", "AAC"], dtype=object),
        "eq_freq": np.array([0.2, 0.3, 0.5], dtype=float),
        "all_syn_cdn_index": np.array([[0, 1], [1, 0]], dtype=int),
        "all_nsy_cdn_index": np.array([[0, 2], [1, 2], [2, 0], [2, 1]], dtype=int),
        "background_omega": 1.5,
    }
    out = main_simulate.get_background_Q(g, Q_method="auto")
    dnds = main_simulate.get_total_Q(out, g["all_nsy_cdn_index"]) / main_simulate.get_total_Q(out, g["all_syn_cdn_index"])
    assert dnds == pytest.approx(1.5, abs=1e-12)
    expected_rate = float(np.sum(g["eq_freq"] * (-np.diag(out))))
    assert expected_rate == pytest.approx(1.0, abs=1e-12)


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


def test_split_tip_and_ancestor_alignment_separates_sequences(tmp_path):
    in_fa = tmp_path / "all.fa"
    tip_fa = tmp_path / "tips.fa"
    anc_fa = tmp_path / "anc.fa"
    in_fa.write_text(">R\nAAA\n>A\nCCC\n>B\nGGG\n", encoding="utf-8")
    n_tip, n_anc = main_simulate.split_tip_and_ancestor_alignment(
        in_fasta=str(in_fa),
        tip_out=str(tip_fa),
        anc_out=str(anc_fa),
        tip_names=["A", "B"],
    )
    assert (n_tip, n_anc) == (2, 1)
    tip = main_simulate.read_fasta(str(tip_fa))
    anc = main_simulate.read_fasta(str(anc_fa))
    assert list(tip.keys()) == ["A", "B"]
    assert list(anc.keys()) == ["R"]
    assert anc["R"] == "AAA"


def test_write_true_asr_bundle_writes_iqtree_compatible_files(tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    anc = tmp_path / "anc.fa"
    anc.write_text(">R\nAAA\n", encoding="utf-8")
    g = {
        "tree": tr,
        "eq_freq": np.full(shape=(61,), fill_value=(1 / 61), dtype=float),
        "iqtree_model": "ECMK07+F+R4",
        "genetic_code": 1,
        "background_omega": 0.2,
    }
    prefix = str(tmp_path / "sim_true")
    out = main_simulate.write_true_asr_bundle(g=g, anc_fasta=str(anc), prefix=prefix)
    for key in ["state", "treefile", "rate", "iqtree", "log", "anc_fasta"]:
        assert key in out
        assert (tmp_path / ("sim_true" + "." + ("anc.fa" if key == "anc_fasta" else key))).exists()
    state = (tmp_path / "sim_true.state").read_text(encoding="utf-8").splitlines()
    assert state[0].startswith("Node\tSite\tState\tp_")
    assert state[1].startswith("R\t1\tAAA\t")
    iqtree_text = (tmp_path / "sim_true.iqtree").read_text(encoding="utf-8")
    assert "Model of substitution: ECMK07+F+R4" in iqtree_text


def test_write_true_asr_bundle_prefers_substitution_model_and_kappa(tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    anc = tmp_path / "anc.fa"
    anc.write_text(">R\nAAA\n", encoding="utf-8")
    g = {
        "tree": tr,
        "eq_freq": np.full(shape=(61,), fill_value=(1 / 61), dtype=float),
        "iqtree_model": "ECMK07+F+R4",
        "substitution_model": "GY+F3X4+R4",
        "genetic_code": 1,
        "background_omega": 0.2,
        "kappa": 2.75,
    }
    prefix = str(tmp_path / "sim_true_model")
    _ = main_simulate.write_true_asr_bundle(g=g, anc_fasta=str(anc), prefix=prefix)
    iqtree_text = (tmp_path / "sim_true_model.iqtree").read_text(encoding="utf-8")
    assert "Model of substitution: GY+F3X4+R4" in iqtree_text
    log_text = (tmp_path / "sim_true_model.log").read_text(encoding="utf-8")
    assert "Transition/transversion ratio (kappa): 2.750000" in log_text


def test_write_true_asr_bundle_rejects_invalid_kappa(tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    anc = tmp_path / "anc.fa"
    anc.write_text(">R\nAAA\n", encoding="utf-8")
    g = {
        "tree": tr,
        "eq_freq": np.full(shape=(61,), fill_value=(1 / 61), dtype=float),
        "iqtree_model": "ECMK07+F+R4",
        "genetic_code": 1,
        "background_omega": 0.2,
        "kappa": -0.1,
    }
    with pytest.raises(ValueError, match="kappa should be >= 0"):
        main_simulate.write_true_asr_bundle(g=g, anc_fasta=str(anc), prefix=str(tmp_path / "sim_true_kappa_bad"))


def test_write_true_asr_bundle_uses_simulated_site_rates(tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    anc = tmp_path / "anc.fa"
    anc.write_text(">R\nAAAAACAAG\n", encoding="utf-8")  # 3 codon sites
    g = {
        "tree": tr,
        "eq_freq": np.full(shape=(61,), fill_value=(1 / 61), dtype=float),
        "iqtree_model": "ECMK07+F+R4",
        "genetic_code": 1,
        "background_omega": 0.2,
        "simulate_site_rates": np.array([0.5, 2.0, 3.25], dtype=float),
    }
    prefix = str(tmp_path / "sim_true_rates")
    _ = main_simulate.write_true_asr_bundle(g=g, anc_fasta=str(anc), prefix=prefix)
    rate_lines = (tmp_path / "sim_true_rates.rate").read_text(encoding="utf-8").splitlines()
    assert rate_lines[0] == "Site\tC_Rate"
    assert rate_lines[1] == "1\t0.500000"
    assert rate_lines[2] == "2\t2.000000"
    assert rate_lines[3] == "3\t3.250000"


def test_write_true_asr_bundle_rejects_mismatched_site_rate_length(tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    anc = tmp_path / "anc.fa"
    anc.write_text(">R\nAAAAACAAG\n", encoding="utf-8")  # 3 codon sites
    g = {
        "tree": tr,
        "eq_freq": np.full(shape=(61,), fill_value=(1 / 61), dtype=float),
        "iqtree_model": "ECMK07+F+R4",
        "genetic_code": 1,
        "background_omega": 0.2,
        "simulate_site_rates": np.array([0.5, 2.0], dtype=float),
    }
    with pytest.raises(ValueError, match="simulate_site_rates length"):
        main_simulate.write_true_asr_bundle(g=g, anc_fasta=str(anc), prefix=str(tmp_path / "sim_true_bad_rate"))


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
    _patch_simulation_index_helpers(monkeypatch)

    def fake_prepare_input_context(
        local_g,
        include_foreground=False,
        include_marginal=False,
        resolve_state_subset=False,
        prepare_state=False,
        force_notree_run=False,
        ignore_tree_inconsistency=False,
    ):
        tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
        local_g["tree"] = tr
        local_g["rooted_tree"] = tr
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

    monkeypatch.setattr(main_simulate, "_prepare_simulation_input_context", fake_prepare_input_context)
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


def test_main_simulate_assigns_simulation_seeds_when_requested(monkeypatch):
    captured = {"seed_conv": None, "seed_nonconv": None}
    _patch_simulation_index_helpers(monkeypatch)

    def fake_prepare_input_context(
        local_g,
        include_foreground=False,
        include_marginal=False,
        resolve_state_subset=False,
        prepare_state=False,
        force_notree_run=False,
        ignore_tree_inconsistency=False,
    ):
        tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
        local_g["tree"] = tr
        local_g["rooted_tree"] = tr
        local_g["num_input_site"] = 5
        return local_g

    def fake_get_foreground_branch(local_g, simulate=False):
        local_g["fg_df"] = pd.DataFrame({"name": ["A"], "PLACEHOLDER": [1]})
        for node in local_g["tree"].traverse():
            ete.set_prop(node, "is_fg_PLACEHOLDER", node.name == "A")
            ete.set_prop(node, "foreground_lineage_id_PLACEHOLDER", 1 if node.name == "A" else 0)
            ete.set_prop(node, "color_PLACEHOLDER", "red" if node.name == "A" else "black")
            ete.set_prop(node, "labelcolor_PLACEHOLDER", "red" if node.name == "A" else "black")
        return local_g

    def fake_plot_branch_category(local_g, file_base, label="all"):
        captured["seed_conv"] = local_g.get("simulate_seed_convergent", None)
        captured["seed_nonconv"] = local_g.get("simulate_seed_nonconvergent", None)
        raise RuntimeError("stop_after_plot")

    monkeypatch.setattr(main_simulate, "_prepare_simulation_input_context", fake_prepare_input_context)
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
        "simulate_seed": 77,
    }
    with pytest.raises(RuntimeError, match="stop_after_plot"):
        main_simulate.main_simulate(g)
    assert captured["seed_conv"] == 77
    assert captured["seed_nonconv"] == 78


def test_main_simulate_routes_outputs_into_configured_namespace(tmp_path, monkeypatch):
    captured = {"file_base": None}
    _patch_simulation_index_helpers(monkeypatch)

    def fake_prepare_input_context(
        local_g,
        include_foreground=False,
        include_marginal=False,
        resolve_state_subset=False,
        prepare_state=False,
        force_notree_run=False,
        ignore_tree_inconsistency=False,
    ):
        tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
        local_g["tree"] = tr
        local_g["rooted_tree"] = tr
        local_g["num_input_site"] = 3
        return local_g

    def fake_get_foreground_branch(local_g, simulate=False):
        local_g["fg_df"] = pd.DataFrame(columns=["name", "PLACEHOLDER"])
        for node in local_g["tree"].traverse():
            ete.set_prop(node, "is_fg_PLACEHOLDER", False)
            ete.set_prop(node, "foreground_lineage_id_PLACEHOLDER", 0)
            ete.set_prop(node, "color_PLACEHOLDER", "black")
            ete.set_prop(node, "labelcolor_PLACEHOLDER", "black")
        return local_g

    def fake_plot_branch_category(local_g, file_base, label="all"):
        captured["file_base"] = str(file_base)

    def fake_get_pyvolve_tree(tree_obj, foreground_scaling_factor, trait_name):
        return "pyvolve_tree"

    def fake_resolve_background_omega(local_g):
        return 0.2

    def fake_resolve_eq_freq(local_g):
        return np.ones(61, dtype=float) / 61.0

    def fake_get_background_Q(local_g, method):
        return np.zeros((61, 61), dtype=float)

    def fake_resolve_site_rates(local_g):
        return np.ones(int(local_g["num_simulated_site"]), dtype=float)

    def fake_evolve_nonconvergent_partition(local_g):
        path = runtime.temp_path("tmp.csubst.simulate_nonconvergent.fa")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(">A\nAAA\n>B\nAAA\n")

    monkeypatch.setattr(main_simulate, "_prepare_simulation_input_context", fake_prepare_input_context)
    monkeypatch.setattr(main_simulate.foreground, "get_foreground_branch", fake_get_foreground_branch)
    monkeypatch.setattr(main_simulate.tree, "plot_branch_category", fake_plot_branch_category)
    monkeypatch.setattr(main_simulate, "get_pyvolve_tree", fake_get_pyvolve_tree)
    monkeypatch.setattr(main_simulate, "_resolve_simulation_background_omega", fake_resolve_background_omega)
    monkeypatch.setattr(main_simulate, "_resolve_simulation_eq_freq", fake_resolve_eq_freq)
    monkeypatch.setattr(main_simulate, "get_background_Q", fake_get_background_Q)
    monkeypatch.setattr(main_simulate, "_resolve_simulation_site_rates", fake_resolve_site_rates)
    monkeypatch.setattr(main_simulate, "evolve_nonconvergent_partition", fake_evolve_nonconvergent_partition)

    outdir = tmp_path / "simulate_outputs"
    g = {
        "foreground": None,
        "num_simulated_site": 1,
        "percent_convergent_site": 0,
        "percent_biased_sub": 90,
        "optimized_branch_length": True,
        "tree_scaling_factor": 1.0,
        "foreground_scaling_factor": 1.0,
        "export_true_asr": False,
        "outdir": str(outdir),
        "output_prefix": "run1",
    }
    main_simulate.main_simulate(g)

    assert captured["file_base"] == str((outdir / "run1_branch_id").resolve())
    assert (outdir / "run1.fa").exists()


def test_main_simulate_infers_true_asr_prefix_from_output_namespace(tmp_path, monkeypatch):
    captured = {"prefix": None}
    _patch_simulation_index_helpers(monkeypatch)

    def fake_prepare_input_context(
        local_g,
        include_foreground=False,
        include_marginal=False,
        resolve_state_subset=False,
        prepare_state=False,
        force_notree_run=False,
        ignore_tree_inconsistency=False,
    ):
        tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
        local_g["tree"] = tr
        local_g["rooted_tree"] = tr
        local_g["num_input_site"] = 3
        return local_g

    def fake_get_foreground_branch(local_g, simulate=False):
        local_g["fg_df"] = pd.DataFrame(columns=["name", "PLACEHOLDER"])
        for node in local_g["tree"].traverse():
            ete.set_prop(node, "is_fg_PLACEHOLDER", False)
            ete.set_prop(node, "foreground_lineage_id_PLACEHOLDER", 0)
            ete.set_prop(node, "color_PLACEHOLDER", "black")
            ete.set_prop(node, "labelcolor_PLACEHOLDER", "black")
        return local_g

    def fake_get_pyvolve_tree(tree_obj, foreground_scaling_factor, trait_name):
        return "pyvolve_tree"

    def fake_resolve_background_omega(local_g):
        return 0.2

    def fake_resolve_eq_freq(local_g):
        return np.ones(61, dtype=float) / 61.0

    def fake_get_background_Q(local_g, method):
        return np.zeros((61, 61), dtype=float)

    def fake_resolve_site_rates(local_g):
        return np.ones(int(local_g["num_simulated_site"]), dtype=float)

    def fake_evolve_nonconvergent_partition(local_g):
        path = runtime.temp_path("tmp.csubst.simulate_nonconvergent.fa")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(">A\nAAA\n>B\nAAA\n>R\nAAA\n")

    def fake_split_tip_and_ancestor_alignment(in_fasta, tip_out, anc_out, tip_names):
        with open(tip_out, "w", encoding="utf-8") as handle:
            handle.write(">A\nAAA\n>B\nAAA\n")
        with open(anc_out, "w", encoding="utf-8") as handle:
            handle.write(">R\nAAA\n")
        return 2, 1

    def fake_write_true_asr_bundle(g, anc_fasta, prefix):
        captured["prefix"] = str(prefix)
        return {
            "state": str(tmp_path / "state"),
            "treefile": str(tmp_path / "treefile"),
            "rate": str(tmp_path / "rate"),
            "iqtree": str(tmp_path / "iqtree"),
            "log": str(tmp_path / "log"),
            "anc_fasta": str(tmp_path / "anc.fa"),
        }

    monkeypatch.setattr(main_simulate, "_prepare_simulation_input_context", fake_prepare_input_context)
    monkeypatch.setattr(main_simulate.foreground, "get_foreground_branch", fake_get_foreground_branch)
    monkeypatch.setattr(main_simulate.tree, "plot_branch_category", lambda local_g, file_base, label="all": None)
    monkeypatch.setattr(main_simulate, "get_pyvolve_tree", fake_get_pyvolve_tree)
    monkeypatch.setattr(main_simulate, "_resolve_simulation_background_omega", fake_resolve_background_omega)
    monkeypatch.setattr(main_simulate, "_resolve_simulation_eq_freq", fake_resolve_eq_freq)
    monkeypatch.setattr(main_simulate, "get_background_Q", fake_get_background_Q)
    monkeypatch.setattr(main_simulate, "_resolve_simulation_site_rates", fake_resolve_site_rates)
    monkeypatch.setattr(main_simulate, "evolve_nonconvergent_partition", fake_evolve_nonconvergent_partition)
    monkeypatch.setattr(main_simulate, "split_tip_and_ancestor_alignment", fake_split_tip_and_ancestor_alignment)
    monkeypatch.setattr(main_simulate, "write_true_asr_bundle", fake_write_true_asr_bundle)

    outdir = tmp_path / "simulate_outputs"
    g = {
        "foreground": None,
        "num_simulated_site": 1,
        "percent_convergent_site": 0,
        "percent_biased_sub": 90,
        "optimized_branch_length": True,
        "tree_scaling_factor": 1.0,
        "foreground_scaling_factor": 1.0,
        "export_true_asr": True,
        "outdir": str(outdir),
        "output_prefix": "run1",
        "true_asr_prefix": "",
    }
    main_simulate.main_simulate(g)

    assert captured["prefix"] == str((outdir / "run1_true_asr").resolve())


def test_initialize_simulation_output_context_places_custom_frequency_file_in_namespace(tmp_path):
    outdir = tmp_path / "simulate_outputs"
    g = {
        "outdir": str(outdir),
        "output_prefix": "run1",
        "true_asr_prefix": "",
    }
    out = main_simulate._initialize_simulation_output_context(g)
    assert out["simulate_custom_frequency_file"] == str(
        (outdir / "run1_custom_matrix_frequencies.txt").resolve()
    )


def test_require_pyvolve_prefers_vendored_backend(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyvolve":
            raise ModuleNotFoundError("No module named 'pyvolve'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(main_simulate, "_PYVOLVE", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    backend = main_simulate._require_pyvolve()
    assert backend.__name__ == "csubst._vendor.pyvolve"
    assert hasattr(backend, "Model")
    assert main_simulate._require_pyvolve() is backend


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
