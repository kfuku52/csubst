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
    return subprocess.run(
        cmd, cwd=str(repo_root), env=env, capture_output=True, text=True
    )


def test_site_invalid_max_sites_fails_cleanly_without_matplotlib_side_effects():
    proc = _run_cli("site", "--branch_id", "0", "--tree_site_plot_max_sites", "0")
    assert proc.returncode == 2
    assert "--tree_site_plot_max_sites should be >= 1." in proc.stderr
    assert "Traceback" not in proc.stderr
    assert "Matplotlib" not in proc.stderr


def test_simulate_invalid_percent_biased_sub_fails_cleanly():
    proc = _run_cli("simulate", "--percent_biased_sub", "-1")
    assert proc.returncode == 2
    assert "--percent_biased_sub should be between 0 and <100." in proc.stderr
    assert "Traceback" not in proc.stderr
