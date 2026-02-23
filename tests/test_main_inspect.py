from pathlib import Path

import numpy as np
import pandas as pd

from csubst import ete
from csubst import main_inspect
from csubst import tree


def _set_trait_defaults(tr, trait_name):
    for node in tr.traverse():
        ete.set_prop(node, "is_fg_" + trait_name, False)
        ete.set_prop(node, "is_mf_" + trait_name, False)
        ete.set_prop(node, "is_mg_" + trait_name, False)
        ete.set_prop(node, "foreground_lineage_id_" + trait_name, 0)
        ete.set_prop(node, "color_" + trait_name, "black")
        ete.set_prop(node, "labelcolor_" + trait_name, "black")


def test_write_branch_maps_writes_combined_and_trait_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)X:1,C:1)R;", format=1))
    trait_name = "traitA"
    _set_trait_defaults(tr, trait_name)
    node_by_name = {node.name: node for node in tr.traverse() if node.name is not None}
    a_id = int(ete.get_prop(node_by_name["A"], "numerical_label"))
    x_id = int(ete.get_prop(node_by_name["X"], "numerical_label"))
    ete.set_prop(node_by_name["A"], "is_fg_" + trait_name, True)
    ete.set_prop(node_by_name["A"], "foreground_lineage_id_" + trait_name, 1)
    ete.set_prop(node_by_name["A"], "color_" + trait_name, "firebrick")
    ete.set_prop(node_by_name["A"], "labelcolor_" + trait_name, "firebrick")
    ete.set_prop(node_by_name["X"], "is_mf_" + trait_name, True)
    g = {
        "tree": tr,
        "fg_df": pd.DataFrame(columns=["name", trait_name]),
        "target_ids": {trait_name: np.array([a_id], dtype=np.int64)},
    }
    main_inspect._write_branch_maps(g)
    combined_path = tmp_path / "csubst_branch_map.tsv"
    trait_path = tmp_path / "csubst_branch_map_traitA.tsv"
    assert combined_path.exists()
    assert trait_path.exists()
    combined_df = pd.read_csv(combined_path, sep="\t")
    trait_df = pd.read_csv(trait_path, sep="\t")
    assert "is_fg_traitA" in combined_df.columns
    assert "is_target_branch_traitA" in combined_df.columns
    assert "is_fg" in trait_df.columns
    assert "is_mf" in trait_df.columns
    row_a = trait_df.loc[trait_df["branch_id"] == a_id].iloc[0]
    row_x = trait_df.loc[trait_df["branch_id"] == x_id].iloc[0]
    assert bool(row_a["is_target_branch"])
    assert bool(row_a["is_fg"])
    assert bool(row_x["is_mf"])


def test_write_branch_maps_strips_placeholder_suffix_from_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)X:1,C:1)R;", format=1))
    trait_name = "PLACEHOLDER"
    _set_trait_defaults(tr, trait_name)
    g = {
        "tree": tr,
        "fg_df": pd.DataFrame(columns=["name", trait_name]),
        "target_ids": {trait_name: np.array([], dtype=np.int64)},
    }
    main_inspect._write_branch_maps(g)
    combined_path = tmp_path / "csubst_branch_map.tsv"
    assert combined_path.exists()
    combined_df = pd.read_csv(combined_path, sep="\t")
    assert "is_target_branch" in combined_df.columns
    assert "is_fg" in combined_df.columns
    assert "branch_color" in combined_df.columns
    assert all(["PLACEHOLDER" not in str(col) for col in combined_df.columns.tolist()])
    assert not (tmp_path / "csubst_branch_map_PLACEHOLDER.tsv").exists()


def test_main_inspect_generates_requested_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)X:1,C:1)R;", format=1))
    node_by_name = {node.name: node for node in tr.traverse() if node.name is not None}
    trait_name = "traitA"
    a_id = int(ete.get_prop(node_by_name["A"], "numerical_label"))

    def fake_read_treefile(local_g):
        local_g["tree"] = tr
        return local_g

    def fake_passthrough(local_g):
        return local_g

    def fake_read_input(local_g):
        local_g["amino_acid_orders"] = np.array(["A", "C"], dtype=object)
        local_g["codon_orders"] = np.array(["AAA", "AAC"], dtype=object)
        local_g["nonsyn_state_orders"] = np.array(["A", "C"], dtype=object)
        return local_g

    def fake_get_foreground_branch(local_g):
        local_g["fg_df"] = pd.DataFrame(columns=["name", trait_name])
        local_g["target_ids"] = {trait_name: np.array([a_id], dtype=np.int64)}
        _set_trait_defaults(local_g["tree"], trait_name)
        ete.set_prop(node_by_name["A"], "is_fg_" + trait_name, True)
        ete.set_prop(node_by_name["A"], "foreground_lineage_id_" + trait_name, 1)
        ete.set_prop(node_by_name["A"], "color_" + trait_name, "firebrick")
        ete.set_prop(node_by_name["A"], "labelcolor_" + trait_name, "firebrick")
        Path("csubst_target_branch_" + trait_name + ".txt").write_text(str(a_id) + "\n", encoding="utf-8")
        return local_g

    def fake_prep_state(local_g):
        num_node = max([int(ete.get_prop(node, "numerical_label")) for node in local_g["tree"].traverse()]) + 1
        local_g["state_pep"] = np.zeros((num_node, 1, 2), dtype=float)
        local_g["state_cdn"] = np.zeros((num_node, 1, 2), dtype=float)
        local_g["state_nsy"] = np.zeros((num_node, 1, 2), dtype=float)
        return local_g

    alignment_calls = []
    plot_calls = []

    def fake_write_alignment(outfile, mode, g, leaf_only=False, branch_ids=None):
        Path(outfile).write_text("aln", encoding="utf-8")
        alignment_calls.append((str(outfile), str(mode)))

    def fake_plot_state_dir(output_dir, state, orders, mode, g):
        Path(output_dir).mkdir(exist_ok=True)
        plot_calls.append((str(output_dir), str(mode)))

    def fake_write_tree(tree_obj, outfile="csubst_tree.nwk", add_numerical_label=True):
        Path(outfile).write_text("(A:1,B:1)R;\n", encoding="utf-8")

    def fake_plot_branch_category(local_g, file_base, label="all"):
        for this_trait in local_g["fg_df"].columns[1:].tolist():
            file_name = (file_base + "_" + this_trait + ".pdf").replace("_PLACEHOLDER", "")
            Path(file_name).write_text("pdf", encoding="utf-8")

    monkeypatch.setattr(main_inspect.tree, "read_treefile", fake_read_treefile)
    monkeypatch.setattr(main_inspect.parser_misc, "generate_intermediate_files", fake_passthrough)
    monkeypatch.setattr(main_inspect.parser_misc, "annotate_tree", fake_passthrough)
    monkeypatch.setattr(main_inspect.parser_misc, "read_input", fake_read_input)
    monkeypatch.setattr(main_inspect.foreground, "get_foreground_branch", fake_get_foreground_branch)
    monkeypatch.setattr(main_inspect.foreground, "get_marginal_branch", fake_passthrough)
    monkeypatch.setattr(main_inspect.parser_misc, "resolve_state_loading", fake_passthrough)
    monkeypatch.setattr(main_inspect.parser_misc, "prep_state", fake_prep_state)
    monkeypatch.setattr(main_inspect.sequence, "write_alignment", fake_write_alignment)
    monkeypatch.setattr(main_inspect, "_plot_state_tree_in_directory", fake_plot_state_dir)
    monkeypatch.setattr(main_inspect.tree, "write_tree", fake_write_tree)
    monkeypatch.setattr(main_inspect.tree, "plot_branch_category", fake_plot_branch_category)

    g = {"genetic_code": 1, "plot_state_aa": True, "plot_state_codon": True}
    main_inspect.main_inspect(g)

    expected = [
        "csubst_branch_id_traitA.pdf",
        "csubst_branch_id_leaf_traitA.pdf",
        "csubst_branch_id_nolabel_traitA.pdf",
        "csubst_plot_state_aa",
        "csubst_plot_state_codon",
        "csubst_tree.nwk",
        "csubst_target_branch_traitA.txt",
        "csubst_branch_map.tsv",
        "csubst_branch_map_traitA.tsv",
        "csubst_alignment_codon.fa",
        "csubst_alignment_aa.fa",
    ]
    for path in expected:
        assert Path(path).exists(), path
    assert ("csubst_alignment_aa.fa", "aa") in alignment_calls
    assert ("csubst_plot_state_aa", "aa") in plot_calls


def test_main_inspect_uses_recoded_state_for_state_aa_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)X:1,C:1)R;", format=1))
    node_by_name = {node.name: node for node in tr.traverse() if node.name is not None}
    trait_name = "traitA"
    a_id = int(ete.get_prop(node_by_name["A"], "numerical_label"))

    def fake_read_treefile(local_g):
        local_g["tree"] = tr
        return local_g

    def fake_passthrough(local_g):
        return local_g

    def fake_read_input(local_g):
        local_g["amino_acid_orders"] = np.array(["A", "C"], dtype=object)
        local_g["codon_orders"] = np.array(["AAA", "AAC"], dtype=object)
        local_g["nonsyn_state_orders"] = np.array(["AC", "DE"], dtype=object)
        return local_g

    def fake_get_foreground_branch(local_g):
        local_g["fg_df"] = pd.DataFrame(columns=["name", trait_name])
        local_g["target_ids"] = {trait_name: np.array([a_id], dtype=np.int64)}
        _set_trait_defaults(local_g["tree"], trait_name)
        return local_g

    def fake_prep_state(local_g):
        num_node = max([int(ete.get_prop(node, "numerical_label")) for node in local_g["tree"].traverse()]) + 1
        local_g["state_pep"] = np.zeros((num_node, 1, 2), dtype=float)
        local_g["state_cdn"] = np.zeros((num_node, 1, 2), dtype=float)
        local_g["state_nsy"] = np.zeros((num_node, 1, 2), dtype=float)
        return local_g

    alignment_calls = []
    plot_calls = []

    def fake_write_alignment(outfile, mode, g, leaf_only=False, branch_ids=None):
        Path(outfile).write_text("aln", encoding="utf-8")
        alignment_calls.append((str(outfile), str(mode)))

    def fake_plot_state_dir(output_dir, state, orders, mode, g):
        Path(output_dir).mkdir(exist_ok=True)
        plot_calls.append((str(output_dir), str(mode), tuple(np.asarray(orders, dtype=object).tolist())))

    def fake_write_tree(tree_obj, outfile="csubst_tree.nwk", add_numerical_label=True):
        Path(outfile).write_text("(A:1,B:1)R;\n", encoding="utf-8")

    def fake_plot_branch_category(local_g, file_base, label="all"):
        for this_trait in local_g["fg_df"].columns[1:].tolist():
            file_name = (file_base + "_" + this_trait + ".pdf").replace("_PLACEHOLDER", "")
            Path(file_name).write_text("pdf", encoding="utf-8")

    monkeypatch.setattr(main_inspect.tree, "read_treefile", fake_read_treefile)
    monkeypatch.setattr(main_inspect.parser_misc, "generate_intermediate_files", fake_passthrough)
    monkeypatch.setattr(main_inspect.parser_misc, "annotate_tree", fake_passthrough)
    monkeypatch.setattr(main_inspect.parser_misc, "read_input", fake_read_input)
    monkeypatch.setattr(main_inspect.foreground, "get_foreground_branch", fake_get_foreground_branch)
    monkeypatch.setattr(main_inspect.foreground, "get_marginal_branch", fake_passthrough)
    monkeypatch.setattr(main_inspect.parser_misc, "resolve_state_loading", fake_passthrough)
    monkeypatch.setattr(main_inspect.parser_misc, "prep_state", fake_prep_state)
    monkeypatch.setattr(main_inspect.sequence, "write_alignment", fake_write_alignment)
    monkeypatch.setattr(main_inspect, "_plot_state_tree_in_directory", fake_plot_state_dir)
    monkeypatch.setattr(main_inspect.tree, "write_tree", fake_write_tree)
    monkeypatch.setattr(main_inspect.tree, "plot_branch_category", fake_plot_branch_category)

    g = {"genetic_code": 1, "plot_state_aa": True, "plot_state_codon": False, "nonsyn_recode": "dayhoff6"}
    main_inspect.main_inspect(g)
    assert ("csubst_alignment_aa.fa", "nsy") in alignment_calls
    assert ("csubst_plot_state_aa", "nsy", ("AC", "DE")) in plot_calls
