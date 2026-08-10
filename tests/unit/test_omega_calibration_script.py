import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / ".github" / "scripts"
MODULE_PATH = SCRIPT_DIR / "omega_pvalue_calibration_check.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("omega_pvalue_calibration_check", MODULE_PATH)
omega_calibration = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(omega_calibration)


def test_run_setting_uses_supported_module_entrypoint(tmp_path, monkeypatch):
    captured = {}

    def stop_after_command(**kwargs):
        captured["cmd"] = kwargs["cmd"]
        raise RuntimeError("command captured")

    monkeypatch.setattr(omega_calibration, "run_timed_command", stop_after_command)

    with pytest.raises(RuntimeError, match="command captured"):
        omega_calibration.run_setting(
            repo_root=REPO_ROOT,
            run_root=tmp_path,
            output_stat="any2spe",
            min_sub_pp=0,
            niter=1,
        )

    assert captured["cmd"][:4] == [sys.executable, "-m", "csubst", "analyze"]
