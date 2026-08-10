import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / ".github" / "scripts" / "_safe_workdir.py"
SPEC = importlib.util.spec_from_file_location("csubst_safe_workdir", MODULE_PATH)
safe_workdir = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(safe_workdir)


def test_prepare_owned_workdir_recreates_only_marked_directory(tmp_path):
    run_root = tmp_path / "runs"
    prepared = safe_workdir.prepare_owned_workdir(run_root, repo_root=REPO_ROOT)
    assert prepared == run_root
    assert (run_root / safe_workdir.OWNERSHIP_MARKER).is_file()
    (run_root / "stale.txt").write_text("stale", encoding="utf-8")

    safe_workdir.prepare_owned_workdir(run_root, repo_root=REPO_ROOT)

    assert not (run_root / "stale.txt").exists()
    assert (run_root / safe_workdir.OWNERSHIP_MARKER).is_file()


def test_prepare_owned_workdir_preserves_unowned_directory(tmp_path):
    run_root = tmp_path / "user-data"
    run_root.mkdir()
    valuable_file = run_root / "valuable.txt"
    valuable_file.write_text("keep", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unowned workdir"):
        safe_workdir.prepare_owned_workdir(run_root, repo_root=REPO_ROOT)

    assert valuable_file.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("path", [Path("/"), Path.home(), REPO_ROOT, REPO_ROOT.parent])
def test_prepare_owned_workdir_rejects_broad_or_protected_paths(path):
    with pytest.raises(ValueError, match="unsafe workdir"):
        safe_workdir.prepare_owned_workdir(path, repo_root=REPO_ROOT)
