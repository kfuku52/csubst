import os
import subprocess
import sys
from pathlib import Path


def _run_csubst(args, cwd):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "csubst" / "csubst"
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    if old_pythonpath:
        env["PYTHONPATH"] = str(repo_root) + os.pathsep + old_pythonpath
    else:
        env["PYTHONPATH"] = str(repo_root)
    cmd = [sys.executable, str(script_path)] + args
    return subprocess.run(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_cli_writes_help_output_to_csubst_log(tmp_path):
    result = _run_csubst(["--help"], cwd=tmp_path)
    assert result.returncode == 0
    log_file = tmp_path / "csubst.log"
    assert log_file.exists()
    log_text = log_file.read_text(encoding="utf-8")
    assert "CSUBST start:" in log_text
    assert "usage:" in log_text.lower()


def test_cli_writes_stderr_output_to_csubst_log(tmp_path):
    result = _run_csubst(["analyze", "--does_not_exist"], cwd=tmp_path)
    assert result.returncode != 0
    log_file = tmp_path / "csubst.log"
    assert log_file.exists()
    log_text = log_file.read_text(encoding="utf-8")
    assert "error:" in log_text.lower()
    assert "--does_not_exist" in log_text
