"""Check acceptance of a valid +I fit with zero invariant proportion.

Run from the checkout with its test environment and IQ-TREE 2.3.6 on PATH:
  python reports/review_20260911_2ae3a0a/reproduce_invariant.py
Outputs and fits go into a new temporary directory. No source files are edited.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
work = Path(tempfile.mkdtemp(prefix="csubst-review-invariant-"))
rng = np.random.default_rng(123)
codons = [
    a + b + c
    for a in "ACGT"
    for b in "ACGT"
    for c in "ACGT"
    if a + b + c not in ("TAA", "TAG", "TGA")
]
alignment = work / "tips.fa"
topology = work / "tree.nwk"
alignment.write_text(
    "".join(
        ">" + name + "\n" + "".join(rng.choice(codons, 100)) + "\n" for name in "ABCD"
    )
)
topology.write_text("((A:.2,B:.2):.2,(C:.2,D:.2):.2);\n")
fit = work / "fit"
with (work / "fit.stdout").open("w") as log:
    subprocess.run(
        [
            "iqtree",
            "-s",
            str(alignment),
            "-te",
            str(topology),
            "-st",
            "CODON",
            "-m",
            "GY+FQ+I",
            "-asr",
            "-wsr",
            "-nt",
            "1",
            "-seed",
            "5",
            "-v",
            "-pre",
            str(fit),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
    )
results = []
for mode in ("joint", "marginal"):
    command = [
        sys.executable,
        "-m",
        "csubst",
        "search",
        "--alignment_file",
        str(alignment),
        "--rooted_tree_file",
        str(topology),
        "--outdir",
        str(work / mode),
        "--threads",
        "1",
        "--blas_threads",
        "1",
        "--substitution_posterior",
        mode,
    ]
    for suffix in ("state", "treefile", "rate", "iqtree", "log"):
        command += ["--iqtree_" + suffix, str(fit) + "." + suffix]
    with (work / (mode + ".log")).open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    results.append({"mode": mode, "exit": result.returncode})
print(json.dumps({"workdir": str(work), "results": results}, indent=2))

assert all(row["exit"] == 0 for row in results), results
