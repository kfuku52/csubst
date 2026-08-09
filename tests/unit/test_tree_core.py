import sys
import numpy as np
import pandas as pd
import pytest

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


def test_is_consistent_tree_rejects_different_topology_with_same_leaves():
    t1 = ete.PhyloNode("((A:1,B:1):1,(C:1,D:1):1)R;", format=1)
    t2 = ete.PhyloNode("((A:1,C:1):1,(B:1,D:1):1)R;", format=1)
    assert not tree.is_consistent_tree(t1, t2)


def test_is_consistent_tree_rejects_duplicate_leaf_names():
    t1 = ete.PhyloNode("(A:1,A:1)R;", format=1)
    t2 = ete.PhyloNode("(A:1,B:1)R;", format=1)
    assert not tree.is_consistent_tree(t1, t2)


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


def test_vectorized_node_distances_match_legacy_chunk():
    tr = tree.add_numerical_node_labels(
        ete.PhyloNode("((A:0.1,B:0.2)X:0.3,(C:0.4,D:0.5)Y:0.6)R;", format=1)
    )
    tree_dict = {
        int(ete.get_prop(node, "numerical_label")): node
        for node in tr.traverse()
    }
    labels = sorted(
        label for label, node in tree_dict.items()
        if not ete.is_root(node)
    )
    combinations = np.array(
        [labels[:3], labels[1:4], labels[-3:], [labels[0], labels[0], labels[0]]],
        dtype=np.int64,
    )
    legacy = tree.calc_node_dist_chunk(
        chunk=combinations,
        start=0,
        tree_dict=tree_dict,
        float_type=np.float64,
    )
    node_num, branch_length, matrix_labels = tree._build_node_distance_matrices(
        tree_dict,
        node_labels=combinations,
    )
    matrix_combinations = tree._map_node_labels_to_distance_indices(
        id_combinations=combinations,
        matrix_labels=matrix_labels,
    )
    observed_num, observed_bl = tree._max_pairwise_node_distances(
        id_combinations=matrix_combinations,
        node_num=node_num,
        branch_length=branch_length,
        float_type=np.float64,
    )
    np.testing.assert_array_equal(observed_num, legacy[:, 1])
    np.testing.assert_allclose(observed_bl, legacy[:, 2], rtol=0.0, atol=0.0)
    assert np.issubdtype(observed_num.dtype, np.integer)


def test_node_distance_matrices_only_allocate_requested_labels():
    tr = tree.add_numerical_node_labels(
        ete.PhyloNode("(((A:1,B:1)X:1,C:1)Y:1,(D:1,E:1)Z:1)R;", format=1)
    )
    tree_dict = {
        int(ete.get_prop(node, "numerical_label")): node
        for node in tr.traverse()
    }
    requested = np.array(sorted(tree_dict)[:2], dtype=np.int64)

    node_num, branch_length, matrix_labels = tree._build_node_distance_matrices(
        tree_dict=tree_dict,
        node_labels=requested,
    )

    assert node_num.shape == (2, 2)
    assert branch_length.shape == (2, 2)
    np.testing.assert_array_equal(matrix_labels, requested)


def test_get_node_distance_uses_direct_path_for_small_workload_and_keeps_integer_dtype(monkeypatch):
    tr = tree.add_numerical_node_labels(
        ete.PhyloNode("((A:1,B:1)X:1,(C:1,D:1)Y:1)R;", format=1)
    )
    labels = [
        int(ete.get_prop(node, "numerical_label"))
        for node in tr.traverse()
        if ete.is_leaf(node)
    ]
    cb = pd.DataFrame({"branch_id_1": [labels[0]], "branch_id_2": [labels[-1]]})
    monkeypatch.setattr(
        tree,
        "_build_node_distance_matrices",
        lambda *args, **kwargs: pytest.fail("small workloads should use direct distances"),
    )

    out = tree.get_node_distance(
        tree=tr,
        cb=cb,
        ncpu=8,
        float_type=np.float64,
    )

    assert np.issubdtype(out["dist_node_num"].dtype, np.integer)
    assert out.at[0, "dist_node_num"] > 0


def test_max_pairwise_node_distances_preserves_negative_branch_values():
    node_num = np.array([[0, 1], [1, 0]], dtype=np.int64)
    branch_length = np.array([[0.0, -0.2], [-0.3, 0.0]], dtype=np.float64)

    observed_num, observed_bl = tree._max_pairwise_node_distances(
        id_combinations=np.array([[0, 1]], dtype=np.int64),
        node_num=node_num,
        branch_length=branch_length,
        float_type=np.float64,
    )

    np.testing.assert_array_equal(observed_num, [1])
    np.testing.assert_array_equal(observed_bl, [-0.2])


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


def test_clear_duplicate_internal_node_names_preserves_first_name():
    tr = ete.PhyloNode("((A:1,B:1)X:1,(C:1,D:1)Y:1)Root;", format=1)
    duplicate_node = [n for n in tr.traverse() if (not ete.is_leaf(n)) and (not ete.is_root(n)) and n.name == "X"][0]
    duplicate_node.name = "Root"

    out = tree._clear_duplicate_internal_node_names(tr)

    internal_names = [n.name for n in out.traverse() if not ete.is_leaf(n)]
    assert internal_names.count("Root") == 1
    assert ete.get_tree_root(out).name == "Root"
    assert duplicate_node.name == ""


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
                    pdf_pages=None, figure_title=None, **kwargs):
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
    assert captured["state_by_node"][labels["R"]] == "AACTCTGAC"
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
                    pdf_pages=None, figure_title=None, **kwargs):
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
    assert captured["state_by_node"][labels["R"]] == "ACD"
    assert captured["state_by_node"][labels["A"]] == "ACD"


def test_plot_state_tree_highlight_pattern_passes_tip_and_branch_highlights(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)X:1,C:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse() if n.name}
    state = np.zeros((len(list(tr.traverse())), 2, 2), dtype=float)
    state[labels["A"], 0, :] = [1.0, 0.0]
    state[labels["A"], 1, :] = [0.0, 1.0]
    state[labels["B"], 0, :] = [1.0, 0.0]
    state[labels["B"], 1, :] = [0.0, 1.0]
    state[labels["C"], 0, :] = [0.0, 1.0]
    state[labels["C"], 1, :] = [1.0, 0.0]
    state[labels["X"], 0, :] = [1.0, 0.0]
    state[labels["X"], 1, :] = [0.0, 1.0]
    state[labels["R"], :, 0] = 1.0
    captured = {}

    def fake_render(tree=None, trait_name=None, file_name=None, label='all', state_by_node=None,
                    state_prob_by_node=None, state_orders=None, state_mode=None,
                    pdf_pages=None, figure_title=None, node_type_by_id=None,
                    tip_label_color_by_node_id=None, highlighted_node_ids=None, highlight_color=None, **kwargs):
        captured["tip_label_color_by_node_id"] = dict(tip_label_color_by_node_id or {})
        captured["highlighted_node_ids"] = set(highlighted_node_ids or set())
        captured["highlight_color"] = highlight_color

    monkeypatch.setattr(tree, "_render_tree_matplotlib", fake_render)

    g = {
        "tree": tr,
        "fg_df": pd.DataFrame({"lineage_id": [1], "trait": ["A"]}),
        "plot_state_aa_highlight_pattern": "KN",
        "plot_state_aa_highlight_color": "orange",
    }
    tree.plot_state_tree(
        state=state,
        orders=np.array(["K", "N"], dtype=object),
        mode="aa",
        g=g,
        plot_request="1-2",
    )
    assert captured["tip_label_color_by_node_id"] == {
        labels["A"]: "orange",
        labels["B"]: "orange",
    }
    assert captured["highlighted_node_ids"] == {
        labels["A"],
        labels["B"],
        labels["X"],
    }
    assert captured["highlight_color"] == "orange"


def test_plot_state_tree_pages_request_preserves_root_state(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    state = np.zeros((len(labels), 1, 2), dtype=float)
    state[labels["R"], 0, :] = [0.0, 1.0]
    state[labels["A"], 0, :] = [1.0, 0.0]
    state[labels["B"], 0, :] = [0.0, 1.0]
    captured = {}

    def fake_render(tree=None, trait_name=None, file_name=None, label='all', state_by_node=None,
                    state_prob_by_node=None, state_orders=None, state_mode=None,
                    pdf_pages=None, figure_title=None, **kwargs):
        captured["state_by_node"] = dict(state_by_node)
        captured["figure_title"] = figure_title

    class _FakePdfPages:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def savefig(self, *args, **kwargs):
            return None

    monkeypatch.setattr(tree, "_render_tree_matplotlib", fake_render)
    monkeypatch.setitem(sys.modules, "matplotlib.backends.backend_pdf", type("M", (), {"PdfPages": _FakePdfPages}))

    g = {
        "tree": tr,
        "fg_df": pd.DataFrame({"lineage_id": [1], "trait": ["A"]}),
    }
    tree.plot_state_tree(
        state=state,
        orders=np.array(["AAA", "AAG"], dtype=object),
        mode="codon",
        g=g,
        plot_request="1,1",
    )
    assert captured["figure_title"] == "Site 1"
    assert captured["state_by_node"][labels["R"]] == "AAG"
