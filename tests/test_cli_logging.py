import os
import runpy
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
    log_file = tmp_path / "csubst_analyze" / "csubst.log"
    assert log_file.exists()
    log_text = log_file.read_text(encoding="utf-8")
    assert "error:" in log_text.lower()
    assert "--does_not_exist" in log_text


def test_cli_honors_custom_log_file_for_help(tmp_path):
    custom_log = tmp_path / "logs" / "custom-help.log"
    result = _run_csubst(["--log_file", str(custom_log), "--help"], cwd=tmp_path)
    assert result.returncode == 0
    assert custom_log.exists()
    assert not (tmp_path / "csubst.log").exists()


def test_cli_honors_custom_log_file_after_subcommand(tmp_path):
    custom_log = tmp_path / "logs" / "inspect.log"
    result = _run_csubst(["inspect", "--log_file", str(custom_log), "-h"], cwd=tmp_path)
    assert result.returncode == 0
    assert custom_log.exists()
    assert not (tmp_path / "csubst.log").exists()


def test_simulate_help_uses_simulate_default_log_name(tmp_path):
    result = _run_csubst(["simulate", "-h"], cwd=tmp_path)
    assert result.returncode == 0
    log_file = tmp_path / "csubst_simulate" / "csubst.log"
    assert log_file.exists()
    assert not (tmp_path / "csubst.log").exists()


def test_subcommand_output_namespace_defaults_are_command_specific():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "csubst" / "csubst"
    ns = runpy.run_path(str(script_path), run_name="not_main")
    parser = ns["_build_parser"]()

    analyze = parser.parse_args(["analyze"])
    assert analyze.outdir == "csubst_analyze"
    assert analyze.output_prefix == "csubst"

    inspect = parser.parse_args(["inspect"])
    assert inspect.outdir == "csubst_inspect"
    assert inspect.output_prefix == "csubst"

    simulate = parser.parse_args(["simulate"])
    assert simulate.outdir == "csubst_simulate"
    assert simulate.output_prefix == "csubst"
