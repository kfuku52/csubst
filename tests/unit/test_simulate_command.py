import builtins
import numpy as np
import pandas as pd
import pytest

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
