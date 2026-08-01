import re
import numpy as np
import pandas as pd

from csubst import main_sites
from csubst import tree
from csubst import ete


def test_plot_tree_site_writes_figure_and_category_table(tmp_path, tiny_tree):
    branch_ids = []
    labels = {}
    for node in tiny_tree.traverse():
        labels[node.name] = ete.get_prop(node, "numerical_label")
        if node.name in {"A", "C"}:
            branch_ids.append(ete.get_prop(node, "numerical_label"))
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    aa_orders = np.array(["A", "V", "T", "I"])
    state_pep = np.zeros((num_node, 4, aa_orders.shape[0]), dtype=float)
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
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4],
            "OCNany2spe": [0.7, 0.1, 0.2, 0.6],
            "OCNany2dif": [0.1, 0.6, 0.1, 0.1],
        }
    )
    g = {
        "tree": tiny_tree,
        "branch_ids": np.array(branch_ids, dtype=int),
        "single_branch_mode": False,
        "tree_site_plot": True,
        "tree_site_plot_format": "pdf",
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "tree_site_plot_max_sites": 60,
        "site_outdir": str(tmp_path),
        "float_format": "%.4f",
        "state_pep": state_pep,
        "amino_acid_orders": aa_orders,
    }
    main_sites.plot_tree_site(df=df, g=g)
    fig_path = tmp_path / "csubst_sites.tree_site.pdf"
    table_path = tmp_path / "csubst_sites.tree_site.tsv"
    assert fig_path.exists()
    assert table_path.exists()
    out_df = pd.read_csv(table_path, sep="\t")
    assert out_df["tree_site_category"].tolist() == ["convergent", "divergent", "blank", "convergent"]


def test_plot_tree_site_honors_output_prefix(tmp_path, tiny_tree):
    labels = {node.name: ete.get_prop(node, "numerical_label") for node in tiny_tree.traverse()}
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    aa_orders = np.array(["A", "V"])
    state_pep = np.zeros((num_node, 1, aa_orders.shape[0]), dtype=float)
    for leaf_name in ("A", "B", "C"):
        state_pep[labels[leaf_name], 0, 0] = 1.0
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1],
            "OCNany2spe": [0.7],
            "OCNany2dif": [0.0],
        }
    )
    g = {
        "tree": tiny_tree,
        "branch_ids": np.array([labels["A"]], dtype=int),
        "single_branch_mode": False,
        "tree_site_plot": True,
        "tree_site_plot_format": "pdf",
        "tree_site_plot_prefix": "csubst_scan",
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.5,
        "tree_site_plot_max_sites": 30,
        "site_outdir": str(tmp_path),
        "float_format": "%.4f",
        "state_pep": state_pep,
        "amino_acid_orders": aa_orders,
    }

    out_paths = main_sites.plot_tree_site(df=df, g=g)

    assert str(tmp_path / "csubst_scan.tree_site.pdf") in out_paths
    assert str(tmp_path / "csubst_scan.tree_site.tsv") in out_paths
    assert not (tmp_path / "csubst_sites.tree_site.pdf").exists()


def test_plot_tree_site_can_skip_category_table(tmp_path, tiny_tree):
    labels = {node.name: ete.get_prop(node, "numerical_label") for node in tiny_tree.traverse()}
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    aa_orders = np.array(["A", "V"])
    state_pep = np.zeros((num_node, 1, aa_orders.shape[0]), dtype=float)
    for leaf_name in ("A", "B", "C"):
        state_pep[labels[leaf_name], 0, 0] = 1.0
    stale_table = tmp_path / "csubst_scan.tree_site.tsv"
    stale_table.write_text("stale\n", encoding="utf-8")
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1],
            "OCNany2spe": [0.7],
            "OCNany2dif": [0.0],
        }
    )
    g = {
        "tree": tiny_tree,
        "branch_ids": np.array([labels["A"]], dtype=int),
        "single_branch_mode": False,
        "tree_site_plot": True,
        "tree_site_plot_format": "pdf",
        "tree_site_plot_prefix": "csubst_scan",
        "tree_site_output_table": False,
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.5,
        "tree_site_plot_max_sites": 30,
        "site_outdir": str(tmp_path),
        "float_format": "%.4f",
        "state_pep": state_pep,
        "amino_acid_orders": aa_orders,
    }

    out_paths = main_sites.plot_tree_site(df=df, g=g)

    assert out_paths == [str(tmp_path / "csubst_scan.tree_site.pdf")]
    assert not stale_table.exists()


def test_plot_tree_site_supports_separate_highlight_branches_and_single_color(tmp_path, tiny_tree):
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tiny_tree.traverse()}
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    aa_orders = np.array(["A", "V"])
    state_pep = np.zeros((num_node, 1, aa_orders.shape[0]), dtype=float)
    for leaf_name in ("A", "B", "C"):
        state_pep[labels[leaf_name], 0, 0] = 1.0
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1],
            "OCNany2spe": [0.0],
            "OCNany2dif": [0.0],
            "N_sub_{}".format(labels["A"]): [0.9],
            "N_sub_{}".format(labels["C"]): [0.8],
        }
    )
    g = {
        "tree": tiny_tree,
        "branch_ids": np.array([labels["A"], labels["C"]], dtype=int),
        "tree_site_highlight_branch_ids": np.array([labels["X"]], dtype=int),
        "tree_site_branch_color_mode": "single",
        "tree_site_branch_color": "#123456",
        "mode": "lineage",
        "single_branch_mode": False,
        "tree_site_plot": True,
        "tree_site_plot_format": "svg",
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.5,
        "tree_site_plot_max_sites": 30,
        "site_outdir": str(tmp_path),
        "float_format": "%.4f",
        "state_pep": state_pep,
        "amino_acid_orders": aa_orders,
    }

    main_sites.plot_tree_site(df=df, g=g)

    svg_text = (tmp_path / "csubst_sites.tree_site.svg").read_text(encoding="utf-8").lower()
    assert "#123456" in svg_text
    assert re.search(r'fill:\s*#123456[^>]*>a</text>', svg_text) is not None
    branch_rgb = main_sites._get_lineage_rgb_by_branch(branch_ids=[labels["A"], labels["C"]], g=g)
    for color in branch_rgb.values():
        assert main_sites.matplotlib.colors.to_hex(color).lower() not in svg_text


def test_plot_tree_site_svg_contains_expected_labels_and_no_legacy_title(tmp_path, tiny_tree):
    branch_ids = []
    labels = {}
    for node in tiny_tree.traverse():
        labels[node.name] = ete.get_prop(node, "numerical_label")
        if node.name in {"A", "C"}:
            branch_ids.append(ete.get_prop(node, "numerical_label"))
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    aa_orders = np.array(["A", "V", "T", "I"])
    state_pep = np.zeros((num_node, 4, aa_orders.shape[0]), dtype=float)
    state_pep[labels["A"], 0, 0] = 1.0
    state_pep[labels["A"], 1, 1] = 1.0
    state_pep[labels["B"], 0, 0] = 1.0
    state_pep[labels["B"], 1, 2] = 1.0
    state_pep[labels["C"], 0, 0] = 1.0
    state_pep[labels["C"], 1, 1] = 1.0
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4],
            "OCNany2spe": [0.7, 0.1, 0.2, 0.6],
            "OCNany2dif": [0.1, 0.6, 0.1, 0.1],
        }
    )
    g = {
        "tree": tiny_tree,
        "branch_ids": np.array(branch_ids, dtype=int),
        "single_branch_mode": False,
        "tree_site_plot": True,
        "tree_site_plot_format": "svg",
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "tree_site_plot_max_sites": 60,
        "site_outdir": str(tmp_path),
        "float_format": "%.4f",
        "state_pep": state_pep,
        "amino_acid_orders": aa_orders,
    }
    out_paths = main_sites.plot_tree_site(df=df, g=g)
    assert str(tmp_path / "csubst_sites.tree_site.svg") in out_paths
    svg_text = (tmp_path / "csubst_sites.tree_site.svg").read_text(encoding="utf-8")
    assert ("Convergence & Divergence" in svg_text) or ("Convergence &amp; Divergence" in svg_text)
    assert re.search(r"N=\s*[0-9,]+(?:&|&amp;)[0-9,]+,\s*PP", svg_text) is not None
    assert ("PP ≥ 0.5" in svg_text) or ("PP &#8805; 0.5" in svg_text)
    assert "Alignment position (aa)" in svg_text
    assert "Tree + site summary" not in svg_text


def test_plot_tree_site_svg_with_species_regex_adds_speciation_duplication_legend(tmp_path):
    tr = ete.PhyloNode(
        "((Homo_sapiens_gene1:1,Homo_sapiens_gene2:1)Dup:1,(Mus_musculus_gene1:1,Rattus_norvegicus_gene1:1)Spec:1)Root;",
        format=1,
    )
    tr = tree.add_numerical_node_labels(tr)
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tr.traverse()}
    branch_ids = np.array([labels["Dup"], labels["Spec"]], dtype=np.int64)
    num_node = max(ete.get_prop(n, "numerical_label") for n in tr.traverse()) + 1
    aa_orders = np.array(["A", "V", "T", "I"])
    state_pep = np.zeros((num_node, 4, aa_orders.shape[0]), dtype=float)
    for leaf_name in ["Homo_sapiens_gene1", "Homo_sapiens_gene2", "Mus_musculus_gene1", "Rattus_norvegicus_gene1"]:
        leaf_id = labels[leaf_name]
        state_pep[leaf_id, 0, 0] = 1.0
        state_pep[leaf_id, 1, 1] = 1.0
        state_pep[leaf_id, 2, 2] = 1.0
        state_pep[leaf_id, 3, 3] = 1.0
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4],
            "OCNany2spe": [0.7, 0.1, 0.2, 0.6],
            "OCNany2dif": [0.1, 0.6, 0.1, 0.1],
            "N_sub_{}".format(labels["Dup"]): [0.2, 0.4, 0.6, 0.8],
            "N_sub_{}".format(labels["Spec"]): [0.9, 0.7, 0.5, 0.3],
        }
    )
    g = {
        "tree": tr,
        "mode": "set",
        "set_stat_type": "any2any",
        "mode_expression": "{}|{}".format(int(labels["Dup"]), int(labels["Spec"])),
        "branch_ids": branch_ids,
        "single_branch_mode": False,
        "tree_site_plot": True,
        "tree_site_plot_format": "svg",
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "tree_site_plot_max_sites": 10,
        "species_regex": r"^([^_]+_[^_]+)_",
        "site_outdir": str(tmp_path),
        "float_format": "%.4f",
        "state_pep": state_pep,
        "amino_acid_orders": aa_orders,
    }
    _ = main_sites.plot_tree_site(df=df, g=g)
    svg_text = (tmp_path / "csubst_sites.tree_site.svg").read_text(encoding="utf-8")
    assert "Speciation node" in svg_text
    assert "Duplication node" in svg_text


def test_plot_tree_site_svg_species_overlap_auto_hides_markers_when_tip_not_parseable(tmp_path):
    tr = ete.PhyloNode(
        "((Homo_sapiens_gene1:1,Homo_sapiens_gene2:1)Dup:1,(BADLABEL:1,Rattus_norvegicus_gene1:1)Spec:1)Root;",
        format=1,
    )
    tr = tree.add_numerical_node_labels(tr)
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tr.traverse()}
    branch_ids = np.array([labels["Dup"], labels["Spec"]], dtype=np.int64)
    num_node = max(ete.get_prop(n, "numerical_label") for n in tr.traverse()) + 1
    aa_orders = np.array(["A", "V", "T", "I"])
    state_pep = np.zeros((num_node, 4, aa_orders.shape[0]), dtype=float)
    for leaf_name in ["Homo_sapiens_gene1", "Homo_sapiens_gene2", "BADLABEL", "Rattus_norvegicus_gene1"]:
        leaf_id = labels[leaf_name]
        state_pep[leaf_id, 0, 0] = 1.0
        state_pep[leaf_id, 1, 1] = 1.0
        state_pep[leaf_id, 2, 2] = 1.0
        state_pep[leaf_id, 3, 3] = 1.0
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4],
            "OCNany2spe": [0.7, 0.1, 0.2, 0.6],
            "OCNany2dif": [0.1, 0.6, 0.1, 0.1],
            "N_sub_{}".format(labels["Dup"]): [0.2, 0.4, 0.6, 0.8],
            "N_sub_{}".format(labels["Spec"]): [0.9, 0.7, 0.5, 0.3],
        }
    )
    g = {
        "tree": tr,
        "mode": "set",
        "set_stat_type": "any",
        "mode_expression": "{}|{}".format(int(labels["Dup"]), int(labels["Spec"])),
        "branch_ids": branch_ids,
        "single_branch_mode": False,
        "tree_site_plot": True,
        "tree_site_plot_format": "svg",
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "tree_site_plot_max_sites": 10,
        "species_regex": r"^([^_]+_[^_]+)_",
        "species_overlap_node_plot": "auto",
        "site_outdir": str(tmp_path),
        "float_format": "%.4f",
        "state_pep": state_pep,
        "amino_acid_orders": aa_orders,
    }
    _ = main_sites.plot_tree_site(df=df, g=g)
    svg_text = (tmp_path / "csubst_sites.tree_site.svg").read_text(encoding="utf-8")
    assert "Speciation node" not in svg_text
    assert "Duplication node" not in svg_text


def test_plot_tree_site_svg_with_pdb_draws_structure_track_via_df_fallback(tmp_path, tiny_tree):
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tiny_tree.traverse()}
    branch_ids = [labels["A"], labels["C"]]
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    aa_orders = np.array(["A", "V", "T", "I"])
    state_pep = np.zeros((num_node, 4, aa_orders.shape[0]), dtype=float)
    for leaf_name in ("A", "B", "C"):
        leaf_id = labels[leaf_name]
        state_pep[leaf_id, 0, 0] = 1.0
        state_pep[leaf_id, 1, 1] = 1.0
        state_pep[leaf_id, 2, 2] = 1.0
        state_pep[leaf_id, 3, 3] = 1.0
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4],
            "OCNany2spe": [0.7, 0.1, 0.2, 0.6],
            "OCNany2dif": [0.1, 0.6, 0.1, 0.1],
            "aa_modelA": ["A", "V", "T", "I"],
            "codon_site_modelA": [10, 11, 12, 13],
        }
    )
    g = {
        "tree": tiny_tree,
        "mode": "intersection",
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
        "pdb": "dummy.pdb",
        "aa_identity_means": {},
        "species_overlap_node_plot": "no",
    }
    _ = main_sites.plot_tree_site(df=df, g=g)
    svg_text = (tmp_path / "csubst_sites.tree_site.svg").read_text(encoding="utf-8")
    assert "modelA" in svg_text
    assert "Structure position (aa)" in svg_text


def test_plot_tree_site_intersection_svg_includes_branch_heatmap_panel(tmp_path, tiny_tree):
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tiny_tree.traverse()}
    branch_ids = [labels["A"], labels["C"]]
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    aa_orders = np.array(["A", "V", "T", "I"])
    state_pep = np.zeros((num_node, 4, aa_orders.shape[0]), dtype=float)
    for leaf_name in ("A", "B", "C"):
        leaf_id = labels[leaf_name]
        state_pep[leaf_id, 0, 0] = 1.0
        state_pep[leaf_id, 1, 1] = 1.0
        state_pep[leaf_id, 2, 2] = 1.0
        state_pep[leaf_id, 3, 3] = 1.0
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4],
            "OCNany2spe": [0.8, 0.1, 0.2, 0.6],
            "OCNany2dif": [0.1, 0.7, 0.1, 0.1],
            "N_sub_{}".format(labels["A"]): [0.2, 0.4, 0.6, 0.8],
            "N_sub_{}".format(labels["C"]): [0.9, 0.7, 0.5, 0.3],
        }
    )
    g = {
        "tree": tiny_tree,
        "mode": "intersection",
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
    svg_text = (tmp_path / "csubst_sites.tree_site.svg").read_text(encoding="utf-8").lower()
    assert re.search(r'>0\.0</text>', svg_text) is not None
    assert re.search(r'>1\.0</text>', svg_text) is not None
    assert re.search(r'>branch id</text>', svg_text) is not None


def test_plot_tree_site_svg_with_pdb_adds_structure_row_and_axis_label(tmp_path, tiny_tree):
    branch_ids = []
    labels = {}
    for node in tiny_tree.traverse():
        labels[node.name] = ete.get_prop(node, "numerical_label")
        if node.name in {"A", "C"}:
            branch_ids.append(ete.get_prop(node, "numerical_label"))
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    aa_orders = np.array(["A", "V", "T", "I"])
    state_pep = np.zeros((num_node, 4, aa_orders.shape[0]), dtype=float)
    state_pep[labels["A"], 0, 0] = 1.0
    state_pep[labels["A"], 1, 1] = 1.0
    state_pep[labels["B"], 0, 0] = 1.0
    state_pep[labels["B"], 1, 2] = 1.0
    state_pep[labels["C"], 0, 0] = 1.0
    state_pep[labels["C"], 1, 1] = 1.0
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4],
            "OCNany2spe": [0.7, 0.1, 0.2, 0.6],
            "OCNany2dif": [0.1, 0.6, 0.1, 0.1],
            "aa_mock_A": ["W", "Y", "", "F"],
            "codon_site_mock_A": [201, 202, 0, 204],
            "codon_site_pdb_mock_A": [101, 102, 0, 104],
        }
    )
    g = {
        "tree": tiny_tree,
        "branch_ids": np.array(branch_ids, dtype=int),
        "single_branch_mode": False,
        "tree_site_plot": True,
        "tree_site_plot_format": "svg",
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "tree_site_plot_max_sites": 60,
        "site_outdir": str(tmp_path),
        "float_format": "%.4f",
        "state_pep": state_pep,
        "amino_acid_orders": aa_orders,
        "pdb": "mock.pdb",
        "highest_identity_chain_name": "mock_A",
    }
    out_paths = main_sites.plot_tree_site(df=df, g=g)
    assert str(tmp_path / "csubst_sites.tree_site.svg") in out_paths
    svg_text = (tmp_path / "csubst_sites.tree_site.svg").read_text(encoding="utf-8")
    svg_text_lower = svg_text.lower()
    assert "Alignment position (aa)" in svg_text
    assert "Structure position (aa)" in svg_text
    assert "Structure|mock_A" not in svg_text
    assert re.search(r">mock_A</text>", svg_text) is not None
    assert svg_text.count("mock_A") == 1
    assert re.search(r">201</text>", svg_text) is not None
    assert re.search(r">202</text>", svg_text) is not None
    assert re.search(r">204</text>", svg_text) is not None
    assert re.search(r">101</text>", svg_text) is None
    assert re.search(r">102</text>", svg_text) is None
    assert re.search(r">104</text>", svg_text) is None
    assert re.search(r">w</text>", svg_text_lower) is not None
    assert re.search(r">y</text>", svg_text_lower) is not None
    assert re.search(r">f</text>", svg_text_lower) is not None


def test_plot_tree_site_svg_shows_overflow_site_count_label(tmp_path, tiny_tree):
    branch_ids = []
    labels = {}
    for node in tiny_tree.traverse():
        labels[node.name] = ete.get_prop(node, "numerical_label")
        if node.name in {"A", "C"}:
            branch_ids.append(ete.get_prop(node, "numerical_label"))
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    aa_orders = np.array(["A", "V", "T", "I"])
    state_pep = np.zeros((num_node, 4, aa_orders.shape[0]), dtype=float)
    state_pep[labels["A"], 0, 0] = 1.0
    state_pep[labels["A"], 1, 1] = 1.0
    state_pep[labels["B"], 0, 0] = 1.0
    state_pep[labels["B"], 1, 2] = 1.0
    state_pep[labels["C"], 0, 0] = 1.0
    state_pep[labels["C"], 1, 1] = 1.0
    df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2, 3, 4],
            "OCNany2spe": [0.7, 0.8, 0.1, 0.6],
            "OCNany2dif": [0.1, 0.1, 0.9, 0.6],
        }
    )
    g = {
        "tree": tiny_tree,
        "branch_ids": np.array(branch_ids, dtype=int),
        "single_branch_mode": False,
        "tree_site_plot": True,
        "tree_site_plot_format": "svg",
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "tree_site_plot_max_sites": 1,
        "site_outdir": str(tmp_path),
        "float_format": "%.4f",
        "state_pep": state_pep,
        "amino_acid_orders": aa_orders,
    }
    _ = main_sites.plot_tree_site(df=df, g=g)
    svg_text = (tmp_path / "csubst_sites.tree_site.svg").read_text(encoding="utf-8").lower()
    assert re.search(r"\+3 sites with pp (≥|&#8805;|&ge;|>=|&gt;=) 0\.50", svg_text) is not None
