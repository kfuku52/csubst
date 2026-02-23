import subprocess
import sys
import os
from pathlib import Path


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
    log_path = repo_root / "csubst.log"
    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    return proc, log_text


def test_site_invalid_max_sites_fails_cleanly_without_matplotlib_side_effects():
    proc, log_text = _run_cli("site", "--branch_id", "0", "--tree_site_plot_max_sites", "0")
    assert proc.returncode == 2
    assert "--tree_site_plot_max_sites should be >= 1." in log_text
    assert "Traceback" not in log_text
    assert "Matplotlib" not in log_text


def test_simulate_invalid_percent_biased_sub_fails_cleanly():
    proc, log_text = _run_cli("simulate", "--percent_biased_sub", "-1")
    assert proc.returncode == 2
    assert "--percent_biased_sub should be between 0 and <100." in log_text
    assert "Traceback" not in log_text


def test_site_deprecated_probability_options_are_rejected():
    proc, log_text = _run_cli("site", "--branch_id", "0", "--tree_site_plot_min_prob", "0.5")
    assert proc.returncode == 2
    assert "unrecognized arguments: --tree_site_plot_min_prob 0.5" in log_text


def test_site_invalid_species_regex_fails_cleanly():
    proc, log_text = _run_cli("site", "--branch_id", "0", "--species_regex", "(")
    assert proc.returncode == 2
    assert "--species_regex is not a valid regular expression" in log_text
    assert "Traceback" not in log_text


def test_site_invalid_species_overlap_node_plot_fails_cleanly():
    proc, log_text = _run_cli("site", "--branch_id", "0", "--species_overlap_node_plot", "maybe")
    assert proc.returncode == 2
    assert "--species_overlap_node_plot should be one of yes, no, auto." in log_text
    assert "Traceback" not in log_text


def test_analyze_state_plot_options_are_rejected_after_move_to_inspect():
    proc, log_text = _run_cli("analyze", "--plot_state_aa", "yes")
    assert proc.returncode == 2
    assert "unrecognized arguments: --plot_state_aa yes" in log_text


def test_inspect_help_is_available():
    proc, _ = _run_cli("inspect", "-h")
    assert proc.returncode == 0


def test_inspect_help_includes_nonsyn_recode_pca_option():
    proc, log_text = _run_cli("inspect", "-h")
    assert proc.returncode == 0
    help_text = (proc.stdout or "") + (proc.stderr or "") + (log_text or "")
    assert "--plot_nonsyn_recode_pca" in help_text
    assert "--nonsyn_recode" in help_text


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
