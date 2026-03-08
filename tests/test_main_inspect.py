from pathlib import Path

import numpy as np
import pandas as pd

from csubst import ete
from csubst import main_inspect
from csubst import runtime
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

    def fake_prepare_input_context(
        local_g,
        include_foreground=False,
        include_marginal=False,
        resolve_state_subset=False,
        prepare_state=False,
    ):
        local_g["tree"] = tr
        local_g["amino_acid_orders"] = np.array(["A", "C"], dtype=object)
        local_g["codon_orders"] = np.array(["AAA", "AAC"], dtype=object)
        local_g["nonsyn_state_orders"] = np.array(["A", "C"], dtype=object)
        local_g["fg_df"] = pd.DataFrame(columns=["name", trait_name])
        local_g["target_ids"] = {trait_name: np.array([a_id], dtype=np.int64)}
        _set_trait_defaults(local_g["tree"], trait_name)
        ete.set_prop(node_by_name["A"], "is_fg_" + trait_name, True)
        ete.set_prop(node_by_name["A"], "foreground_lineage_id_" + trait_name, 1)
        ete.set_prop(node_by_name["A"], "color_" + trait_name, "firebrick")
        ete.set_prop(node_by_name["A"], "labelcolor_" + trait_name, "firebrick")
        Path(runtime.output_path(local_g, "foreground_branch_" + trait_name + ".txt")).write_text(
            str(a_id) + "\n",
            encoding="utf-8",
        )
        return local_g

    def fake_prep_state(local_g, apply_site_filtering=True):
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

    def fake_plot_state_dir(output_dir, state, orders, mode, g, plot_request, plot_request_name):
        Path(output_dir).mkdir(exist_ok=True)
        plot_token = tree.normalize_state_plot_request(plot_request, param_name=plot_request_name)["token"]
        for trait_name_local in g["fg_df"].columns[1:].tolist():
            plot_file = Path(output_dir) / ("csubst_state_" + trait_name_local + "_" + str(mode) + "_" + plot_token + ".pdf")
            plot_file.write_text("pdf", encoding="utf-8")
        plot_calls.append((str(output_dir), str(mode), plot_request_name))

    def fake_write_tree(tree_obj, outfile="csubst_tree.nwk", add_numerical_label=True):
        Path(outfile).write_text("(A:1,B:1)R;\n", encoding="utf-8")

    def fake_plot_branch_category(local_g, file_base, label="all"):
        for this_trait in local_g["fg_df"].columns[1:].tolist():
            file_name = (file_base + "_" + this_trait + ".pdf").replace("_PLACEHOLDER", "")
            Path(file_name).write_text("pdf", encoding="utf-8")

    monkeypatch.setattr(main_inspect.parser_misc, "prepare_input_context", fake_prepare_input_context)
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
        "csubst_state_traitA_aa_all.pdf",
        "csubst_state_traitA_codon_all.pdf",
        "csubst_tree.nwk",
        "csubst_foreground_branch_traitA.txt",
        "csubst_branch_map.tsv",
        "csubst_branch_map_traitA.tsv",
        "csubst_alignment_codon.fa",
        "csubst_alignment_aa.fa",
    ]
    for path in expected:
        assert Path(path).exists(), path
    assert ((str((tmp_path / "csubst_alignment_aa.fa").resolve())), "aa") in alignment_calls
    assert ((str(tmp_path.resolve())), "aa", "--plot_state_aa") in plot_calls


def test_main_inspect_routes_outputs_into_configured_namespace(tmp_path, monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)X:1,C:1)R;", format=1))
    node_by_name = {node.name: node for node in tr.traverse() if node.name is not None}
    trait_name = "traitA"
    a_id = int(ete.get_prop(node_by_name["A"], "numerical_label"))
    outdir = tmp_path / "inspect_outputs"

    def fake_prepare_input_context(
        local_g,
        include_foreground=False,
        include_marginal=False,
        resolve_state_subset=False,
        prepare_state=False,
    ):
        local_g["tree"] = tr
        local_g["amino_acid_orders"] = np.array(["A", "C"], dtype=object)
        local_g["codon_orders"] = np.array(["AAA", "AAC"], dtype=object)
        local_g["nonsyn_state_orders"] = np.array(["A", "C"], dtype=object)
        local_g["fg_df"] = pd.DataFrame(columns=["name", trait_name])
        local_g["target_ids"] = {trait_name: np.array([a_id], dtype=np.int64)}
        _set_trait_defaults(local_g["tree"], trait_name)
        Path(runtime.output_path(local_g, "foreground_branch_" + trait_name + ".txt")).write_text(
            str(a_id) + "\n",
            encoding="utf-8",
        )
        return local_g

    def fake_prep_state(local_g, apply_site_filtering=True):
        num_node = max([int(ete.get_prop(node, "numerical_label")) for node in local_g["tree"].traverse()]) + 1
        local_g["state_pep"] = np.zeros((num_node, 1, 2), dtype=float)
        local_g["state_cdn"] = np.zeros((num_node, 1, 2), dtype=float)
        local_g["state_nsy"] = np.zeros((num_node, 1, 2), dtype=float)
        return local_g

    def fake_write_alignment(outfile, mode, g, leaf_only=False, branch_ids=None):
        Path(outfile).write_text(str(mode), encoding="utf-8")

    def fake_plot_state_dir(output_dir, state, orders, mode, g, plot_request, plot_request_name):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plot_token = tree.normalize_state_plot_request(plot_request, param_name=plot_request_name)["token"]
        for trait_name_local in g["fg_df"].columns[1:].tolist():
            plot_file = Path(output_dir) / ("csubst_state_" + trait_name_local + "_" + str(mode) + "_" + plot_token + ".pdf")
            plot_file.write_text("pdf", encoding="utf-8")

    def fake_write_tree(tree_obj, outfile="csubst_tree.nwk", add_numerical_label=True):
        Path(outfile).write_text("(A:1,B:1)R;\n", encoding="utf-8")

    def fake_plot_branch_category(local_g, file_base, label="all"):
        Path(str(file_base) + "_traitA.pdf").write_text(label, encoding="utf-8")

    monkeypatch.setattr(main_inspect.parser_misc, "prepare_input_context", fake_prepare_input_context)
    monkeypatch.setattr(main_inspect.parser_misc, "prep_state", fake_prep_state)
    monkeypatch.setattr(main_inspect.sequence, "write_alignment", fake_write_alignment)
    monkeypatch.setattr(main_inspect, "_plot_state_tree_in_directory", fake_plot_state_dir)
    monkeypatch.setattr(main_inspect.tree, "write_tree", fake_write_tree)
    monkeypatch.setattr(main_inspect.tree, "plot_branch_category", fake_plot_branch_category)

    g = {
        "genetic_code": 1,
        "plot_state_aa": True,
        "plot_state_codon": False,
        "outdir": str(outdir),
        "output_prefix": "inspect_run",
    }
    main_inspect.main_inspect(g)

    expected_paths = [
        outdir / "inspect_run_alignment_codon.fa",
        outdir / "inspect_run_alignment_aa.fa",
        outdir / "inspect_run_tree.nwk",
        outdir / "inspect_run_branch_id_traitA.pdf",
        outdir / "csubst_state_traitA_aa_all.pdf",
        outdir / "inspect_run_branch_map.tsv",
        outdir / "inspect_run_foreground_branch_traitA.txt",
    ]
    for path in expected_paths:
        assert path.exists(), path


def test_main_inspect_uses_recoded_state_for_state_aa_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)X:1,C:1)R;", format=1))
    node_by_name = {node.name: node for node in tr.traverse() if node.name is not None}
    trait_name = "traitA"
    a_id = int(ete.get_prop(node_by_name["A"], "numerical_label"))

    def fake_prepare_input_context(
        local_g,
        include_foreground=False,
        include_marginal=False,
        resolve_state_subset=False,
        prepare_state=False,
    ):
        local_g["tree"] = tr
        local_g["amino_acid_orders"] = np.array(["A", "C"], dtype=object)
        local_g["codon_orders"] = np.array(["AAA", "AAC"], dtype=object)
        local_g["nonsyn_state_orders"] = np.array(["AC", "DE"], dtype=object)
        local_g["fg_df"] = pd.DataFrame(columns=["name", trait_name])
        local_g["target_ids"] = {trait_name: np.array([a_id], dtype=np.int64)}
        _set_trait_defaults(local_g["tree"], trait_name)
        return local_g

    def fake_prep_state(local_g, apply_site_filtering=True):
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

    def fake_plot_state_dir(output_dir, state, orders, mode, g, plot_request, plot_request_name):
        Path(output_dir).mkdir(exist_ok=True)
        plot_token = tree.normalize_state_plot_request(plot_request, param_name=plot_request_name)["token"]
        for trait_name_local in g["fg_df"].columns[1:].tolist():
            plot_file = Path(output_dir) / ("csubst_state_" + trait_name_local + "_" + str(mode) + "_" + plot_token + ".pdf")
            plot_file.write_text("pdf", encoding="utf-8")
        plot_calls.append(
            (str(output_dir), str(mode), tuple(np.asarray(orders, dtype=object).tolist()), plot_request_name)
        )

    def fake_write_tree(tree_obj, outfile="csubst_tree.nwk", add_numerical_label=True):
        Path(outfile).write_text("(A:1,B:1)R;\n", encoding="utf-8")

    def fake_plot_branch_category(local_g, file_base, label="all"):
        for this_trait in local_g["fg_df"].columns[1:].tolist():
            file_name = (file_base + "_" + this_trait + ".pdf").replace("_PLACEHOLDER", "")
            Path(file_name).write_text("pdf", encoding="utf-8")

    monkeypatch.setattr(main_inspect.parser_misc, "prepare_input_context", fake_prepare_input_context)
    monkeypatch.setattr(main_inspect.parser_misc, "prep_state", fake_prep_state)
    monkeypatch.setattr(main_inspect.sequence, "write_alignment", fake_write_alignment)
    monkeypatch.setattr(main_inspect, "_plot_state_tree_in_directory", fake_plot_state_dir)
    monkeypatch.setattr(main_inspect.tree, "write_tree", fake_write_tree)
    monkeypatch.setattr(main_inspect.tree, "plot_branch_category", fake_plot_branch_category)

    g = {"genetic_code": 1, "plot_state_aa": True, "plot_state_codon": False, "nonsyn_recode": "dayhoff6"}
    main_inspect.main_inspect(g)
    assert ((str((tmp_path / "csubst_alignment_aa.fa").resolve())), "nsy") in alignment_calls
    assert ((str(tmp_path.resolve())), "nsy", ("AC", "DE"), "--plot_state_aa") in plot_calls


def test_main_inspect_writes_unfiltered_alignments_before_site_filtering(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)X:1,C:1)R;", format=1))
    recorded = {
        "prep_flags": [],
        "alignment_sites": {},
        "plot_sites": {},
    }

    def fake_prepare_input_context(
        local_g,
        include_foreground=False,
        include_marginal=False,
        resolve_state_subset=False,
        prepare_state=False,
    ):
        local_g["tree"] = tr
        local_g["amino_acid_orders"] = np.array(["A", "C"], dtype=object)
        local_g["codon_orders"] = np.array(["AAA", "AAC"], dtype=object)
        local_g["nonsyn_state_orders"] = np.array(["A", "C"], dtype=object)
        local_g["fg_df"] = pd.DataFrame(columns=["name"])
        local_g["target_ids"] = {}
        local_g["float_tol"] = 1e-12
        return local_g

    def fake_prep_state(local_g, apply_site_filtering=True):
        recorded["prep_flags"].append(bool(apply_site_filtering))
        num_node = max([int(ete.get_prop(node, "numerical_label")) for node in local_g["tree"].traverse()]) + 1
        local_g["state_pep"] = np.zeros((num_node, 3, 2), dtype=float)
        local_g["state_cdn"] = np.zeros((num_node, 3, 2), dtype=float)
        local_g["state_nsy"] = np.zeros((num_node, 3, 2), dtype=float)
        local_g["state_pep"][:, :, 0] = 1.0
        local_g["state_cdn"][:, :, 0] = 1.0
        local_g["state_nsy"][:, :, 0] = 1.0
        return local_g

    def fake_apply_site_filters(local_g):
        local_g["state_pep"] = local_g["state_pep"][:, 1:2, :]
        local_g["state_cdn"] = local_g["state_cdn"][:, 1:2, :]
        local_g["state_nsy"] = local_g["state_nsy"][:, 1:2, :]
        local_g["site_index_alignment"] = np.array([1], dtype=np.int64)
        return local_g

    def fake_write_alignment(outfile, mode, g, leaf_only=False, branch_ids=None):
        recorded["alignment_sites"][Path(outfile).name] = int(g["state_cdn"].shape[1])
        Path(outfile).write_text(str(mode), encoding="utf-8")

    def fake_plot_state_dir(output_dir, state, orders, mode, g, plot_request, plot_request_name):
        recorded["plot_sites"][str(mode)] = int(state.shape[1])
        Path(output_dir).mkdir(exist_ok=True)

    def fake_write_tree(tree_obj, outfile="csubst_tree.nwk", add_numerical_label=True):
        Path(outfile).write_text("(A:1,B:1)R;\n", encoding="utf-8")

    monkeypatch.setattr(main_inspect.parser_misc, "prepare_input_context", fake_prepare_input_context)
    monkeypatch.setattr(main_inspect.parser_misc, "prep_state", fake_prep_state)
    monkeypatch.setattr(main_inspect.parser_misc, "apply_site_filters", fake_apply_site_filters)
    monkeypatch.setattr(main_inspect.sequence, "write_alignment", fake_write_alignment)
    monkeypatch.setattr(main_inspect, "_plot_state_tree_in_directory", fake_plot_state_dir)
    monkeypatch.setattr(main_inspect.tree, "write_tree", fake_write_tree)
    monkeypatch.setattr(main_inspect.tree, "plot_branch_category", lambda local_g, file_base, label="all": None)

    g = {"genetic_code": 1, "plot_state_aa": True, "plot_state_codon": True}
    main_inspect.main_inspect(g)

    assert recorded["prep_flags"] == [False]
    assert recorded["alignment_sites"]["csubst_alignment_codon.fa"] == 3
    assert recorded["alignment_sites"]["csubst_alignment_aa.fa"] == 3
    assert recorded["plot_sites"]["aa"] == 3
    assert recorded["plot_sites"]["codon"] == 3


def test_main_inspect_download_prostt5_exits_before_input_loading(monkeypatch):
    call_log = []

    def fake_ensure(g):
        local_g = g
        call_log.append(("ensure", str(local_g.get("prostt5_model", ""))))
        return "/tmp/prostt5-cache"

    def fail_if_called(_g):
        raise AssertionError("read_treefile should not be called in --download_prostt5 mode.")

    monkeypatch.setattr(main_inspect.structural_alphabet, "ensure_prostt5_model_files", fake_ensure)
    monkeypatch.setattr(main_inspect.tree, "read_treefile", fail_if_called)
    g = {"download_prostt5": True, "prostt5_model": "Rostlab/ProstT5"}
    main_inspect.main_inspect(g)
    assert call_log == [("ensure", "Rostlab/ProstT5")]


def test_main_inspect_applies_3di_smoke_branch_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)X:1,C:1)R;", format=1))
    node_ids = sorted([int(ete.get_prop(node, "numerical_label")) for node in tr.traverse()])
    root_id = int(ete.get_prop(ete.get_tree_root(tr), "numerical_label"))
    nonroot_ids = [bid for bid in node_ids if bid != root_id]
    expected_nonroot = np.array(nonroot_ids[:2], dtype=np.int64)
    expected_with_root = np.array([root_id] + expected_nonroot.tolist(), dtype=np.int64)

    def fake_prepare_input_context(
        local_g,
        include_foreground=False,
        include_marginal=False,
        resolve_state_subset=False,
        prepare_state=False,
    ):
        local_g["tree"] = tr
        local_g["amino_acid_orders"] = np.array(["A", "C"], dtype=object)
        local_g["codon_orders"] = np.array(["AAA", "AAC"], dtype=object)
        local_g["nonsyn_state_orders"] = np.array(["A", "C"], dtype=object)
        local_g["fg_df"] = pd.DataFrame(columns=["name"])
        local_g["target_ids"] = {}
        return local_g

    observed = {"sa_ids": None, "nsy_branch_ids": None}

    def fake_prep_state(local_g, apply_site_filtering=True):
        observed["sa_ids"] = np.asarray(local_g.get("sa_inference_branch_ids"), dtype=np.int64)
        num_node = max([int(ete.get_prop(node, "numerical_label")) for node in local_g["tree"].traverse()]) + 1
        local_g["state_pep"] = np.zeros((num_node, 1, 2), dtype=float)
        local_g["state_cdn"] = np.zeros((num_node, 1, 2), dtype=float)
        local_g["state_nsy"] = np.zeros((num_node, 1, 2), dtype=float)
        return local_g

    def fake_write_alignment(outfile, mode, g, leaf_only=False, branch_ids=None):
        Path(outfile).write_text("aln", encoding="utf-8")
        if Path(outfile).name == "csubst_alignment_3di.fa":
            observed["nsy_branch_ids"] = np.asarray(branch_ids, dtype=np.int64)

    def fake_write_tree(tree_obj, outfile="csubst_tree.nwk", add_numerical_label=True):
        Path(outfile).write_text("(A:1,B:1)R;\n", encoding="utf-8")

    monkeypatch.setattr(main_inspect.parser_misc, "prepare_input_context", fake_prepare_input_context)
    monkeypatch.setattr(main_inspect.parser_misc, "prep_state", fake_prep_state)
    monkeypatch.setattr(main_inspect.sequence, "write_alignment", fake_write_alignment)
    monkeypatch.setattr(main_inspect.tree, "write_tree", fake_write_tree)
    monkeypatch.setattr(main_inspect.tree, "plot_branch_category", lambda local_g, file_base, label="all": None)

    g = {
        "genetic_code": 1,
        "nonsyn_recode": "3di20",
        "sa_smoke_max_branches": 2,
        "plot_state_aa": False,
        "plot_state_codon": False,
    }
    main_inspect.main_inspect(g)
    np.testing.assert_array_equal(observed["sa_ids"], expected_with_root)
    np.testing.assert_array_equal(observed["nsy_branch_ids"], expected_nonroot)


def test_main_inspect_uses_fast_bypass_for_site_specific_state_plots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_fast(local_g):
        calls.append("fast")
        return local_g

    monkeypatch.setattr(main_inspect, "_run_fast_unfiltered_outputs", fake_fast)
    monkeypatch.setattr(
        main_inspect,
        "_run_standard_unfiltered_outputs",
        lambda _g: (_ for _ in ()).throw(AssertionError("standard path should not run for site-specific requests")),
    )
    monkeypatch.setattr(main_inspect, "_finalize_inspect_outputs", lambda local_g: local_g)

    g = {
        "genetic_code": 1,
        "plot_state_aa": "2-3",
        "plot_state_codon": "no",
    }
    main_inspect.main_inspect(g)
    assert calls == ["fast"]


def test_main_inspect_uses_standard_path_for_all_state_plots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_standard(local_g):
        calls.append("standard")
        return local_g

    monkeypatch.setattr(
        main_inspect,
        "_run_fast_unfiltered_outputs",
        lambda _g: (_ for _ in ()).throw(AssertionError("fast path should not run for all-site requests")),
    )
    monkeypatch.setattr(main_inspect, "_run_standard_unfiltered_outputs", fake_standard)
    monkeypatch.setattr(main_inspect, "_finalize_inspect_outputs", lambda local_g: local_g)

    g = {
        "genetic_code": 1,
        "plot_state_aa": "all",
        "plot_state_codon": "no",
    }
    main_inspect.main_inspect(g)
    assert calls == ["standard"]


def test_run_fast_unfiltered_outputs_streams_selected_sites_and_writes_alignments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    alignment_file = tmp_path / "alignment.fa"
    alignment_file.write_text(
        ">A\nAAAAAGAAA\n>B\nAAAAAGAAA\n>C\nAAGAAAAAG\n",
        encoding="utf-8",
    )
    state_file = tmp_path / "test.state"
    state_file.write_text(
        "\n".join(
            [
                "Node\tSite\tState\tp_AAA\tp_AAG",
                "R\t1\tAAA\t1.0\t0.0",
                "R\t2\tAAG\t0.0\t1.0",
                "R\t3\tAAA\t1.0\t0.0",
                "N1\t1\tAAA\t1.0\t0.0",
                "N1\t2\t?\t0.0\t0.0",
                "N1\t3\tAAG\t0.0\t1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_prepare(local_g):
        local_g["tree"] = tr
        local_g["num_node"] = len(list(tr.traverse()))
        local_g["num_input_site"] = 3
        local_g["num_input_state"] = 2
        local_g["input_state"] = ["AAA", "AAG"]
        local_g["state_probability_columns"] = ["p_AAA", "p_AAG"]
        local_g["codon_orders"] = np.array(["AAA", "AAG"], dtype=object)
        local_g["amino_acid_orders"] = np.array(["K"], dtype=object)
        local_g["matrix_groups"] = {"K": ["AAA", "AAG"]}
        local_g["synonymous_indices"] = {"K": [0, 1]}
        local_g["max_synonymous_size"] = 2
        local_g["nonsyn_recode"] = "no"
        local_g["fg_df"] = pd.DataFrame(columns=["name"])
        local_g["target_ids"] = {}
        local_g["float_type"] = np.float64
        local_g["float_tol"] = 1e-12
        local_g["alignment_file"] = str(alignment_file)
        local_g["iqtree_state"] = str(state_file)
        local_g["state_loaded_branch_ids"] = None
        local_g["outdir"] = str(tmp_path)
        return local_g

    plot_calls = []

    def fake_plot_state_tree_selected_sites(state, orders, mode, g, site_numbers, output_dir=None, plot_request="all", plot_request_name=None):
        plot_calls.append((str(mode), tuple(site_numbers.tolist()), tuple(state.shape), tuple(np.asarray(orders, dtype=object).tolist())))
        return []

    monkeypatch.setattr(main_inspect, "_prepare_fast_inspect_context", fake_prepare)
    monkeypatch.setattr(main_inspect.tree, "plot_state_tree_selected_sites", fake_plot_state_tree_selected_sites)

    g = {
        "genetic_code": 1,
        "plot_state_aa": "2-3",
        "plot_state_codon": "2-3",
    }
    main_inspect._run_fast_unfiltered_outputs(g)

    codon_alignment = (tmp_path / "csubst_alignment_codon.fa").read_text(encoding="utf-8")
    aa_alignment = (tmp_path / "csubst_alignment_aa.fa").read_text(encoding="utf-8")
    assert ">N1\nAAA---AAG\n" in codon_alignment
    assert ">A\nAAAAAGAAA\n" in codon_alignment
    assert ">N1\nK-K\n" in aa_alignment
    assert ("aa", (2, 3), (5, 2, 1), ("K",)) in plot_calls
    assert ("codon", (2, 3), (5, 2, 2), ("AAA", "AAG")) in plot_calls


def test_stream_internal_states_for_fast_inspect_allows_unnamed_internal_node(tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)X:1,C:1)Root;", format=1))
    duplicate_node = [n for n in tr.traverse() if (not ete.is_leaf(n)) and (not ete.is_root(n)) and n.name == "X"][0]
    duplicate_node.name = "Root"
    tr = tree._clear_duplicate_internal_node_names(tr)
    root = ete.get_tree_root(tr)
    root_id = int(ete.get_prop(root, "numerical_label"))
    unnamed_id = int(ete.get_prop(duplicate_node, "numerical_label"))
    state_file = tmp_path / "unnamed_internal.state"
    state_file.write_text(
        "\n".join(
            [
                "Node\tSite\tState\tp_AAA\tp_AAG",
                "Root\t1\tAAA\t1.0\t0.0",
                "Root\t2\tAAG\t0.0\t1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    g = {
        "tree": tr,
        "num_node": len(list(tr.traverse())),
        "num_input_site": 2,
        "num_input_state": 2,
        "state_probability_columns": ["p_AAA", "p_AAG"],
        "float_type": np.float64,
        "float_tol": 1e-12,
        "iqtree_state": str(state_file),
        "ml_anc": False,
    }
    aa_config = {
        "group_matrix": np.eye(2, dtype=np.float64),
    }

    codon_symbol_index, aa_symbol_index, state_cdn_subset = main_inspect._stream_internal_states_for_fast_inspect(
        g=g,
        selected_site_indices=np.array([0, 1], dtype=np.int64),
        aa_config=aa_config,
    )

    np.testing.assert_allclose(state_cdn_subset[root_id, :, :], [[1.0, 0.0], [0.0, 1.0]], atol=1e-12)
    np.testing.assert_allclose(state_cdn_subset[unnamed_id, :, :], 0.0, atol=1e-12)
    np.testing.assert_array_equal(codon_symbol_index[root_id, :], [0, 1])
