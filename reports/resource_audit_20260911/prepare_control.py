"""Copy a compiled baseline and backport only the root-transfer repair.

Usage: python prepare_control.py BEFORE AFTER BEFORE_ROOTFIXED
The source trees are untouched. Native extensions must already be compiled
for the same interpreter; this control changes only Python tree handling.
"""

import ast
import difflib
from pathlib import Path
import shutil
import sys


def main():
    before, after, destination = map(Path, sys.argv[1:])
    shutil.copytree(
        before, destination, ignore=shutil.ignore_patterns("build", "__pycache__")
    )
    path = destination / "csubst/tree.py"
    original = path.read_text()
    updated = (after / "csubst/tree.py").read_text()
    functions = {
        node.name: ast.get_source_segment(updated, node)
        for node in ast.parse(updated).body
        if isinstance(node, ast.FunctionDef)
    }
    target = next(
        node
        for node in ast.parse(original).body
        if isinstance(node, ast.FunctionDef) and node.name == "transfer_root"
    )
    lines = original.splitlines(keepends=True)
    replacement = (
        functions["_internal_node_partition_names"]
        + "\n\n\n"
        + functions["transfer_root"]
        + "\n"
    )
    lines[target.lineno - 1 : target.end_lineno] = [replacement]
    repaired = "".join(lines)
    path.write_text(repaired)
    patch = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            repaired.splitlines(keepends=True),
            fromfile="a/csubst/tree.py",
            tofile="b/csubst/tree.py",
            n=0,
        )
    )
    Path(__file__).with_name("root-backport.patch").write_text(patch)


if __name__ == "__main__":
    main()
