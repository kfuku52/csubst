import subprocess
import sys
import os
import re
from pathlib import Path

import pytest


def _resolve_log_path(repo_root, args):
    args = tuple(args)
    if "--log_file" in args:
        log_index = args.index("--log_file") + 1
        return Path(args[log_index])
    for token in args:
        if token.startswith("--log_file="):
            return Path(token.split("=", 1)[1])
    if len(args) > 0:
        if args[0] == "benchmark":
            return repo_root / "csubst_benchmark" / "csubst.log"
        if args[0] == "benchmark-plot":
            return repo_root / "csubst_benchmark_plot" / "csubst.log"
        if args[0] == "doctor":
            return repo_root / "csubst_doctor" / "csubst.log"
        if args[0] == "simulate":
            return repo_root / "csubst_simulate" / "csubst.log"
        if args[0] == "scan":
            return repo_root / "csubst_scan" / "csubst.log"
        if args[0] == "search":
            return repo_root / "csubst_search" / "csubst.log"
        if args[0] == "analyze":
            return repo_root / "csubst_search" / "csubst.log"
        if args[0] == "inspect":
            return repo_root / "csubst_inspect" / "csubst.log"
        if args[0] == "sites":
            return repo_root / "csubst_sites" / "csubst.log"
        if args[0] == "site":
            return repo_root / "csubst_sites" / "csubst.log"
    return repo_root / "csubst.log"


def _run_cli(*args):
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(repo_root / "csubst" / "csubst")]
    cmd.extend(args)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + (
        os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else ""
    )
    proc = subprocess.run(
        cmd, cwd=str(repo_root), env=env, capture_output=True, text=True
    )
    log_path = _resolve_log_path(repo_root, args)
    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    return proc, log_text


def test_sites_invalid_max_sites_fails_cleanly_without_matplotlib_side_effects():
    proc, log_text = _run_cli("sites", "--branch_id", "0", "--tree_site_plot_max_sites", "0")
    assert proc.returncode == 2
    assert "--tree_site_plot_max_sites should be >= 1." in log_text
    assert "Traceback" not in log_text
    assert "Matplotlib" not in log_text


def test_simulate_invalid_percent_biased_sub_fails_cleanly():
    proc, log_text = _run_cli("simulate", "--percent_biased_sub", "-1")
    assert proc.returncode == 2
    assert "--percent_biased_sub should be between 0 and <100." in log_text
    assert "Traceback" not in log_text


def test_simulate_invalid_seed_fails_cleanly():
    proc, log_text = _run_cli("simulate", "--simulate_seed", "-2")
    assert proc.returncode == 2
    assert "--simulate_seed should be -1 or >= 0." in log_text
    assert "Traceback" not in log_text


def test_invalid_common_random_seed_fails_cleanly():
    proc, log_text = _run_cli("scan", "--random_seed", "-2")
    assert proc.returncode == 2
    assert "--random_seed should be -1 or >= 0." in log_text
    assert "Traceback" not in log_text


def test_sites_deprecated_probability_options_are_rejected():
    proc, log_text = _run_cli("sites", "--branch_id", "0", "--tree_site_plot_min_prob", "0.5")
    assert proc.returncode == 2
    assert "unrecognized arguments: --tree_site_plot_min_prob 0.5" in log_text


def test_sites_invalid_species_regex_fails_cleanly():
    proc, log_text = _run_cli("sites", "--branch_id", "0", "--species_regex", "(")
    assert proc.returncode == 2
    assert "--species_regex is not a valid regular expression" in log_text
    assert "Traceback" not in log_text


def test_sites_invalid_species_overlap_node_plot_fails_cleanly():
    proc, log_text = _run_cli("sites", "--branch_id", "0", "--species_overlap_node_plot", "maybe")
    assert proc.returncode == 2
    assert "--species_overlap_node_plot should be one of yes, no, auto." in log_text
    assert "Traceback" not in log_text


def test_inspect_help_includes_species_overlap_node_plot_option():
    proc, log_text = _run_cli("inspect", "-h")
    assert proc.returncode == 0
    help_text = (proc.stdout or "") + (proc.stderr or "") + (log_text or "")
    assert "--species_overlap_node_plot" in help_text
    assert "--output_manifest" in help_text
    assert "--combination_count_max_arity" in help_text


def test_inspect_rejects_invalid_combination_count_max_arity():
    proc, log_text = _run_cli("inspect", "--combination_count_max_arity", "0")
    assert proc.returncode == 2
    assert "--combination_count_max_arity should be >= 1" in log_text
    assert "Traceback" not in log_text


def test_sites_help_shows_output_manifest_and_removed_site_output_manifest_is_rejected():
    proc, log_text = _run_cli("sites", "-h")
    assert proc.returncode == 0
    help_text = (proc.stdout or "") + (proc.stderr or "") + (log_text or "")
    assert "--output_manifest" in help_text
    assert "--site_output_manifest" not in help_text
    proc, log_text = _run_cli("sites", "--branch_id", "0", "--site_output_manifest", "no")
    assert proc.returncode == 2
    assert "unrecognized arguments: --site_output_manifest no" in log_text


def test_sites_help_shows_swissmodel_first_in_database_default():
    proc, log_text = _run_cli("sites", "-h")
    assert proc.returncode == 0
    help_text = (proc.stdout or "") + (proc.stderr or "") + (log_text or "")
    assert "default=swissmodel,pdb,alphafold,alphafill" in help_text
    assert "swissmodel: Run online QBLAST search" in help_text


def test_inspect_help_includes_state_highlight_options():
    proc, _ = _run_cli("inspect", "-h")
    assert proc.returncode == 0
    help_text = (proc.stdout or "") + (proc.stderr or "")
    assert "--plot_state_aa_highlight_pattern" in help_text
    assert "--plot_state_aa_highlight_color" in help_text
    assert "--tree_tip_label_spacing" not in help_text
    assert "--tree_fig_max_height" not in help_text

    proc, _ = _run_cli("inspect", "--help-advanced")
    assert proc.returncode == 0
    help_text = (proc.stdout or "") + (proc.stderr or "")
    assert "--tree_tip_label_spacing" in help_text
    assert "--tree_fig_max_height" in help_text


def test_search_state_plot_options_are_rejected_after_move_to_inspect():
    proc, log_text = _run_cli("search", "--plot_state_aa", "yes")
    assert proc.returncode == 2
    assert "unrecognized arguments: --plot_state_aa yes" in log_text


def test_search_rejects_removed_prostt5_batch_size_option():
    proc, log_text = _run_cli("search", "--prostt5_batch_size", "8")
    assert proc.returncode == 2
    assert "unrecognized arguments: --prostt5_batch_size 8" in log_text


def test_search_rejects_removed_pseudocount_strength_option():
    proc, log_text = _run_cli("search", "--pseudocount_strength", "2.0")
    assert proc.returncode == 2
    assert "unrecognized arguments: --pseudocount_strength 2.0" in log_text


@pytest.mark.parametrize("subcommand", ["search", "benchmark"])
def test_epistasis_options_are_advanced_only(subcommand):
    proc, _ = _run_cli(subcommand, "-h")
    assert proc.returncode == 0
    help_text = (proc.stdout or "") + (proc.stderr or "")
    assert "--epistasis_" not in help_text

    proc, _ = _run_cli(subcommand, "--help-advanced")
    assert proc.returncode == 0
    help_text = (proc.stdout or "") + (proc.stderr or "")
    assert "default=swissmodel,pdb,alphafold,alphafill" in help_text
    assert "Supported:" in help_text
    assert "swissmodel" in help_text
    assert "--epistasis_beta" in help_text
    assert "--epistasis_pdb" in help_text


@pytest.mark.parametrize(
    "subcommand",
    ["search", "scan", "sites", "simulate", "benchmark", "doctor", "inspect"],
)
def test_removed_performance_options_are_absent_from_shared_help(subcommand):
    proc, _ = _run_cli(subcommand, "-h")
    assert proc.returncode == 0
    normal_help = (proc.stdout or "") + (proc.stderr or "")
    assert "--threads" in normal_help
    assert "--parallel_" not in normal_help
    assert "--sub_tensor_backend" not in normal_help
    assert "--float_type" not in normal_help
    assert "--expected_state_backend" not in normal_help

    proc, _ = _run_cli(subcommand, "--help-advanced")
    assert proc.returncode == 0
    advanced_help = (proc.stdout or "") + (proc.stderr or "")
    assert "--threads" in advanced_help
    assert "--parallel_" not in advanced_help
    assert "--sub_tensor_backend" not in advanced_help
    assert "--float_type" not in advanced_help
    assert "--expected_state_backend" in advanced_help


@pytest.mark.parametrize(
    "option,value",
    [
        ("--parallel_backend", "threading"),
        ("--sub_tensor_backend", "dense"),
        ("--float_type", "32"),
        ("--omegaC_method", "modelfree"),
    ],
)
def test_removed_performance_options_are_rejected(option, value):
    proc, log_text = _run_cli("search", option, value)
    assert proc.returncode == 2
    assert "unrecognized arguments:" in log_text


def test_removed_infile_type_option_is_rejected():
    proc, log_text = _run_cli("search", "--infile_type", "iqtree")
    assert proc.returncode == 2
    assert "unrecognized arguments: --infile_type iqtree" in log_text


def test_cli_help_lists_primary_commands_and_legacy_aliases():
    proc, log_text = _run_cli("--help")
    assert proc.returncode == 0
    help_text = (proc.stdout or "") + (proc.stderr or "") + (log_text or "")
    assert "benchmark" in help_text
    assert "benchmark-plot" in help_text
    assert "doctor" in help_text
    assert "scan" in help_text
    assert "search (analyze)" in help_text
    assert "sites (site)" in help_text


def test_legacy_aliases_remain_available():
    analyze_proc, _ = _run_cli("analyze", "-h")
    assert analyze_proc.returncode == 0
    site_proc, _ = _run_cli("site", "-h")
    assert site_proc.returncode == 0


def test_inspect_help_is_available():
    proc, _ = _run_cli("inspect", "-h")
    assert proc.returncode == 0


def test_benchmark_help_is_available():
    proc, log_text = _run_cli("benchmark", "-h")
    assert proc.returncode == 0
    help_text = (proc.stdout or "") + (proc.stderr or "") + (log_text or "")
    assert "--benchmark_expectation_methods" in help_text
    assert "--benchmark_keep_going" in help_text
    assert "--output_manifest" in help_text


def test_scan_help_is_available():
    proc, _ = _run_cli("scan", "-h")
    assert proc.returncode == 0
    help_text = (proc.stdout or "") + (proc.stderr or "")
    assert "--scan_match" in help_text
    assert "--scan_unit_mode" in help_text
    assert "lineage|stem|clade" in help_text
    assert "default=clade" in help_text
    assert "--random_seed" in help_text
    assert "any2any" in help_text
    assert re.search(r"--scan_unit(?![A-Za-z0-9_])", help_text) is None
    assert "--scan_rate_exposure" in help_text
    assert "q_weighted" in help_text
    assert "default=q_weighted" in help_text
    assert "--scan_rate_event_mode" in help_text
    assert "--scan_other_scope" in help_text
    assert "all|sister" in help_text
    assert "--scan_sister_stem_only" in help_text
    assert "--scan_pvalue_calibration" in help_text
    assert "full_scan" in help_text
    assert "--scan_n_permutations" in help_text
    assert "--scan_site_plot" in help_text
    assert "--tree_site_plot_format" in help_text
    assert "--tree_site_plot_max_sites" in help_text
    assert "--tree_site_tip_label_spacing" not in help_text
    assert "--tree_site_fig_max_height" not in help_text
    assert "--scan_permutation_mode" not in help_text
    assert "--scan_rate_length" in help_text
    assert "default=n_rescaled" in help_text
    assert "--nonsyn_recode" in help_text
    assert "--scan_report_targets" not in help_text
    assert "--scan_candidate_target" not in help_text
    assert "--fg_clade_permutation" not in help_text
    assert "--fg_exclude_wg" not in help_text
    assert "--mg_parent" not in help_text
    assert "--mg_sister" not in help_text

    proc, _ = _run_cli("scan", "--help-advanced")
    assert proc.returncode == 0
    advanced_help = (proc.stdout or "") + (proc.stderr or "")
    assert "--tree_site_tip_label_spacing" in advanced_help
    assert "--tree_site_fig_max_height" in advanced_help
    assert "--scan_permutation_sample_original" in advanced_help


def test_benchmark_plot_help_is_available():
    proc, log_text = _run_cli("benchmark-plot", "-h")
    assert proc.returncode == 0
    help_text = (proc.stdout or "") + (proc.stderr or "") + (log_text or "")
    assert "--benchmark_dir" in help_text
    assert "--benchmark_plot_metrics" in help_text
    assert "--benchmark_plot_format" in help_text
    assert "--output_manifest" in help_text


def test_benchmark_plot_missing_directory_fails_cleanly():
    proc, log_text = _run_cli(
        "benchmark-plot",
        "--benchmark_dir",
        str(Path(__file__).resolve().parents[1] / "definitely_missing_benchmark_dir"),
        "--benchmark_plot_recursive",
        "no",
    )
    assert proc.returncode == 2
    assert "--benchmark_dir was not found or is not a directory" in log_text
    assert "Traceback" not in log_text


def test_doctor_help_is_available():
    proc, log_text = _run_cli("doctor", "-h")
    assert proc.returncode == 0
    help_text = (proc.stdout or "") + (proc.stderr or "") + (log_text or "")
    assert "--check_iqtree_exe" in help_text
    assert "--doctor_fail_level" in help_text
    assert "--output_manifest" in help_text


def test_inspect_help_includes_nonsyn_recode_pca_option():
    proc, _ = _run_cli("inspect", "-h")
    assert proc.returncode == 0
    help_text = (proc.stdout or "") + (proc.stderr or "")
    assert "--plot_nonsyn_recode_pca" in help_text
    assert "--download_prostt5" not in help_text
    assert "--sa_smoke_max_branches" not in help_text
    assert "--nonsyn_recode" in help_text

    proc, _ = _run_cli("inspect", "--help-advanced")
    assert proc.returncode == 0
    advanced_help = (proc.stdout or "") + (proc.stderr or "")
    assert "--download_prostt5" in advanced_help
    assert "--sa_smoke_max_branches" in advanced_help


def test_inspect_rejects_legacy_yes_state_plot_option():
    proc, log_text = _run_cli("inspect", "--plot_state_aa", "yes")
    assert proc.returncode == 2
    assert "no longer accepts yes/no" in log_text


def test_cli_entrypoint_runs_from_repo_root_without_pythonpath():
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(repo_root / "csubst" / "csubst"), "--help"]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        cmd, cwd=str(repo_root), env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "ModuleNotFoundError" not in combined
    assert "No module named 'csubst'" not in combined
