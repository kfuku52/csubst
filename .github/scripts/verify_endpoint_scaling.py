#!/usr/bin/env python3
"""Compare only locally generated benchmark pickles, never untrusted inputs."""

import json
import runpy
import re
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

p = argparse.ArgumentParser(
    description="Verify raw CB output invariance across endpoint scaling runs."
)
p.add_argument("--reports", type=Path, nargs="+", required=True)
p.add_argument("--result", type=Path, required=True)
args = p.parse_args()
root = Path(__file__).resolve().parents[2]
helper = runpy.run_path(
    str(root / ".github/scripts/benchmark_endpoint_optimization.py")
)
compare = helper["compare_frames"]
source_hashes = helper["hashes"](root)
refs = {}
out = {
    "within_method_parity": [],
    "branch_table_toggle_parity": [],
    "method_branch_set_counts": {},
    "source_hashes_match": True,
}
for file in args.reports:
    data = json.loads(file.read_text())
    assert data["joint_sha256"] == source_hashes, (
        "Measured source no longer matches the current source"
    )
    legacy = next(r for r in data["runs"] if r["mode"] == "marginal")
    baseline = Path(
        legacy["command"][legacy["command"].index("--alignment_file") + 1].replace(
            "${HOME}", str(Path.home())
        )
    ).parents[2]
    assert helper["hashes"](baseline) == data["baseline_sha256"], (
        "Measured baseline source changed"
    )
    for row in data["runs"]:
        if row["warmup"]:
            continue
        directory = Path(
            row["command"][row["command"].index("--outdir") + 1].replace(
                "${HOME}", str(Path.home())
            )
        )
        tag = directory.name
        for f in sorted(directory.glob("csubst_cb_*.unrounded.pkl")):
            if not re.fullmatch(r"csubst_cb_\d+\.tsv\.unrounded\.pkl", f.name):
                continue
            df = pd.read_pickle(f)
            ids = [c for c in df.columns if c.startswith("branch_id_")]
            df = df.sort_values(ids).reset_index(drop=True)
            key = (row["scenario"], row["mode"], f.name)
            if key not in refs:
                refs[key] = (tag, df)
                continue
            reftag, ref = refs[key]
            try:
                details = compare(ref, df)
            except AssertionError as exc:
                raise AssertionError(
                    f"{tag}, {f.name}, reference={reftag}: {exc}"
                ) from exc
            columns = [
                c
                for c in df.columns
                if c.startswith(("OCN", "OCS", "ECN", "ECS")) and not c.endswith("CoD")
            ]
            delta = (df[columns] - ref[columns]).to_numpy()
            details["max_count_absolute_difference"] = float(
                np.max(np.abs(delta[np.isfinite(delta)]), initial=0)
            )
            out["within_method_parity"].append(
                dict(reference=reftag, candidate=tag, table=f.name, **details)
            )
for mode in ("marginal", "joint"):
    name = "csubst_cb_2.tsv.unrounded.pkl"
    reference = refs.get(("pair", mode, name))
    candidate = refs.get(("pair_b", mode, name))
    if reference is None or candidate is None:
        continue
    details = compare(reference[1], candidate[1])
    columns = [
        c
        for c in reference[1]
        if c.startswith(("OCN", "OCS", "ECN", "ECS")) and not c.endswith("CoD")
    ]
    delta = (candidate[1][columns] - reference[1][columns]).to_numpy()
    details["max_count_absolute_difference"] = float(
        np.max(np.abs(delta[np.isfinite(delta)]), initial=0)
    )
    out["branch_table_toggle_parity"].append(
        dict(reference=reference[0], candidate=candidate[0], table=name, **details)
    )
for scenario, mode, name in refs:
    if mode != "joint":
        continue
    a = refs.get((scenario, "marginal", name))
    b = refs[(scenario, "joint", name)]
    if a is None:
        continue
    ids = [c for c in b[1].columns if c.startswith("branch_id_")]
    x = set(map(tuple, a[1][ids].to_numpy()))
    y = set(map(tuple, b[1][ids].to_numpy()))
    out["method_branch_set_counts"][scenario + "/" + name] = dict(
        marginal=len(x),
        joint=len(y),
        shared=len(x & y),
        marginal_only=len(x - y),
        joint_only=len(y - x),
    )
args.result.parent.mkdir(parents=True, exist_ok=True)
args.result.write_text(json.dumps(out, indent=2) + "\n")
print("Verified comparisons:", len(out["within_method_parity"]))
print(json.dumps(out["method_branch_set_counts"], indent=2))
