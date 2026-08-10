#!/usr/bin/env python3
"""Reject machine-specific paths and oversized generated report artifacts."""

from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
HOME_PATH_PATTERN = re.compile(rb"/(?:Users|home)/[A-Za-z0-9._-]+/")
MAX_REPORT_BYTES = 5 * 1024 * 1024


def tracked_files():
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
    )
    return [REPO_ROOT / path.decode("utf-8") for path in output.split(b"\0") if path]


def main():
    errors = []
    for path in tracked_files():
        if not path.is_file():
            continue
        relative_path = path.relative_to(REPO_ROOT)
        data = path.read_bytes()
        if HOME_PATH_PATTERN.search(data):
            errors.append("{} contains a user-home absolute path".format(relative_path))
        if (
            relative_path.parts[0] == "reports"
            and path.stat().st_size > MAX_REPORT_BYTES
        ):
            errors.append(
                "{} is larger than the 5 MiB report-artifact limit".format(relative_path)
            )
    if errors:
        raise RuntimeError("Repository hygiene check failed:\n- " + "\n- ".join(errors))
    print("Repository hygiene checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
