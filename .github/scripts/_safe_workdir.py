"""Safety helpers for CI scripts that recreate working directories."""

from pathlib import Path
import shutil


OWNERSHIP_MARKER = ".csubst-ci-workdir"


def _contains(protected_path, candidate_parent):
    """Return whether candidate_parent is protected_path or one of its parents."""

    return candidate_parent == protected_path or candidate_parent in protected_path.parents


def prepare_owned_workdir(path, *, repo_root):
    """Recreate a marked CI workdir without deleting an unowned directory.

    A previous directory is removed only if this helper created and marked it.
    Broad paths that contain the repository or the user's home are always rejected.
    """

    run_root = Path(path).expanduser().resolve()
    protected_paths = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path(repo_root).resolve(),
    }
    if any(_contains(protected_path, run_root) for protected_path in protected_paths):
        raise ValueError("Refusing unsafe workdir: {}".format(run_root))

    marker = run_root / OWNERSHIP_MARKER
    if run_root.exists():
        if (not run_root.is_dir()) or marker.is_symlink() or (not marker.is_file()):
            raise RuntimeError(
                "Refusing to remove unowned workdir {} (missing {}).".format(
                    run_root, OWNERSHIP_MARKER
                )
            )
        shutil.rmtree(run_root)

    run_root.mkdir(parents=True)
    (run_root / OWNERSHIP_MARKER).write_text(
        "Owned by csubst CI scripts; safe to recreate.\n", encoding="utf-8"
    )
    return run_root
