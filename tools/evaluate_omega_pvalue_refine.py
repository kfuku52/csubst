#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_float_grid(value, name):
    tokens = [token.strip() for token in str(value).split(",") if token.strip() != ""]
    if len(tokens) == 0:
        raise ValueError("{} should contain one or more comma-delimited numbers.".format(name))
    out = []
    for token in tokens:
        numeric = float(token)
        if (not np.isfinite(numeric)) or (numeric <= 0.0) or (numeric >= 1.0):
            raise ValueError("{} values should be in (0, 1).".format(name))
        out.append(float(numeric))
    dedup = []
    for numeric in out:
        if numeric not in dedup:
            dedup.append(numeric)
    return dedup


def _parse_time_file(path):
    real = np.nan
    user = np.nan
    sys = np.nan
    maxrss = np.nan
    peak = np.nan
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("real "):
            real = float(line.split()[1])
        elif line.startswith("user "):
            user = float(line.split()[1])
        elif line.startswith("sys "):
            sys = float(line.split()[1])
        elif "maximum resident set size" in line:
            maxrss = float(line.split()[0])
        elif "peak memory footprint" in line:
            peak = float(line.split()[0])
    return real, user, sys, maxrss, peak


def _extract_refinement_rows(log_path):
    txt = Path(log_path).read_text(encoding="utf-8", errors="replace")
    rows = []
    pattern = re.compile(r"pomegaC\s+(\S+)\s+refinement after stage\s+(\d+): rows\s+([0-9,]+)\s+->\s+([0-9,]+)")
    for match in pattern.finditer(txt):
        sub = match.group(1)
        stage = int(match.group(2))
        rows_before = int(match.group(3).replace(",", ""))
        rows_after = int(match.group(4).replace(",", ""))
        frac = np.nan
        if rows_before > 0:
            frac = float(rows_after) / float(rows_before)
        rows.append(
            {
                "output_stat": sub,
                "stage": stage,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "refine_fraction": frac,
            }
        )
    return pd.DataFrame(rows)


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _summarize_pq(cb_tsv, output_stat):
    df = pd.read_csv(cb_tsv, sep="\t")
    rows = []
    for prefix in ["pomegaC", "qomegaC"]:
        for suffix in ["", "_nocalib"]:
            col = "{}{}{}".format(prefix, output_stat, suffix)
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            if int(finite.sum()) == 0:
                continue
            rows.append(
                {
                    "column": col,
                    "n_finite": int(finite.sum()),
                    "median": float(np.nanmedian(values[finite])),
                    "mean": float(np.nanmean(values[finite])),
                    "frac_le_0.05": float(np.mean(values[finite] <= 0.05)),
                }
            )
    return pd.DataFrame(rows)


def _compare_cb_tsv(base_tsv, other_tsv, output_stat):
    base = pd.read_csv(base_tsv, sep="\t")
    other = pd.read_csv(other_tsv, sep="\t")
    id_cols = [c for c in base.columns if c.startswith("branch_id_")]
    if len(id_cols) > 0:
        base = base.sort_values(id_cols).reset_index(drop=True)
        other = other.sort_values(id_cols).reset_index(drop=True)
    rows = []
    for col in [
        "pomegaC{}".format(output_stat),
        "qomegaC{}".format(output_stat),
        "pomegaC{}_nocalib".format(output_stat),
        "qomegaC{}_nocalib".format(output_stat),
    ]:
        if (col not in base.columns) or (col not in other.columns):
            continue
        x = pd.to_numeric(base[col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(other[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if int(finite.sum()) == 0:
            continue
        d = np.abs(x[finite] - y[finite])
        corr = np.nan
        if int(finite.sum()) >= 2:
            corr = float(np.corrcoef(x[finite], y[finite])[0, 1])
        rows.append(
            {
                "column": col,
                "n_finite_pair": int(finite.sum()),
                "median_base": float(np.nanmedian(x[finite])),
                "median_other": float(np.nanmedian(y[finite])),
                "mean_abs_diff": float(np.mean(d)),
                "p95_abs_diff": float(np.quantile(d, 0.95)),
                "pearson_r": corr,
            }
        )
    return pd.DataFrame(rows)


def _run_timed(cmd, cwd, stdout_path, stderr_path, dry_run=False):
    timed = ["/usr/bin/time", "-l", "-p"] + list(cmd)
    if dry_run:
        return 0, " ".join(timed)
    with Path(stdout_path).open("w", encoding="utf-8") as out, Path(stderr_path).open("w", encoding="utf-8") as err:
        proc = subprocess.run(timed, cwd=str(cwd), stdout=out, stderr=err, check=False)
    return int(proc.returncode), " ".join(timed)


def _copy_dataset(prefix, dst_dir):
    prefix = Path(prefix)
    required = [
        "{}.alignment.fa".format(prefix),
        "{}.tree.nwk".format(prefix),
        "{}.foreground.txt".format(prefix),
    ]
    optional = [
        "{}.alignment.fa.iqtree".format(prefix),
        "{}.alignment.fa.log".format(prefix),
        "{}.alignment.fa.rate".format(prefix),
        "{}.alignment.fa.state".format(prefix),
        "{}.alignment.fa.treefile".format(prefix),
    ]
    for src in required:
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError("Required dataset file was not found: {}".format(src_path))
        shutil.copy2(src_path, Path(dst_dir) / src_path.name)
    for src in optional:
        src_path = Path(src)
        if src_path.exists():
            shutil.copy2(src_path, Path(dst_dir) / src_path.name)


def _mk_bench_root(outdir):
    if outdir is not None:
        root = Path(outdir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path("/tmp") / "csubst_omega_pvalue_refine_{}".format(stamp)
    root.mkdir(parents=True, exist_ok=False)
    return root


def _resolve_repo_root(repo_root):
    if repo_root is not None:
        return Path(repo_root).resolve()
    return Path(__file__).resolve().parents[1]


def _build_jobs(args):
    jobs = []
    baseline_name = "fixed"
    jobs.append(
        {
            "name": baseline_name,
            "schedule": str(int(args.omega_pvalue_niter)),
            "refine_threshold": float(args.baseline_refine_threshold),
            "refine_ci_alpha": float(args.baseline_refine_ci_alpha),
            "is_baseline": True,
        }
    )
    for threshold in _parse_float_grid(args.refine_threshold_grid, "--refine-threshold-grid"):
        for ci_alpha in _parse_float_grid(args.refine_ci_alpha_grid, "--refine-ci-alpha-grid"):
            name = "auto_t{}_a{}".format(str(threshold), str(ci_alpha)).replace(".", "p")
            jobs.append(
                {
                    "name": name,
                    "schedule": "0",
                    "refine_threshold": float(threshold),
                    "refine_ci_alpha": float(ci_alpha),
                    "is_baseline": False,
                }
            )
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark omega p-value staged refinement settings with runtime + peak RAM + p/q drift summaries.",
    )
    parser.add_argument("--repo-root", default=None, help="Repository root path. Defaults to parent directory of this script.")
    parser.add_argument(
        "--dataset-prefix",
        default=None,
        help="Dataset prefix path (e.g., /path/to/PGK for PGK.alignment.fa, PGK.tree.nwk, ...).",
    )
    parser.add_argument("--outdir", default=None, help="Output directory. Default: /tmp/csubst_omega_pvalue_refine_<timestamp>.")
    parser.add_argument("--reps", type=int, default=1, help="Number of replicate runs per configuration.")
    parser.add_argument("--threads", type=int, default=1, help="--threads value for csubst search.")
    parser.add_argument("--output-stat", default="any2spe", help="Output statistic for p/q comparison (default: any2spe).")
    parser.add_argument("--omega-pvalue-niter", type=int, default=1000, help="--omega_pvalue_niter value.")
    parser.add_argument("--rounding", default="stochastic", help="--omega_pvalue_rounding value.")
    parser.add_argument("--refine-threshold-grid", default="0.01,0.05,0.1", help="Grid for --omega_pvalue_refine_threshold.")
    parser.add_argument("--refine-ci-alpha-grid", default="0.01,0.05,0.1", help="Grid for --omega_pvalue_refine_ci_alpha.")
    parser.add_argument(
        "--baseline-refine-threshold",
        type=float,
        default=0.05,
        help="Threshold value logged for fixed baseline run (not used by fixed schedule itself).",
    )
    parser.add_argument(
        "--baseline-refine-ci-alpha",
        type=float,
        default=0.05,
        help="CI alpha value logged for fixed baseline run (not used by fixed schedule itself).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only; do not execute.")
    args = parser.parse_args()

    if args.reps <= 0:
        raise ValueError("--reps should be >= 1.")
    if args.threads <= 0:
        raise ValueError("--threads should be >= 1.")
    if args.omega_pvalue_niter <= 0:
        raise ValueError("--omega-pvalue-niter should be >= 1.")

    repo_root = _resolve_repo_root(args.repo_root)
    dataset_prefix = args.dataset_prefix
    if dataset_prefix is None:
        dataset_prefix = str(repo_root / "csubst" / "dataset" / "PGK")
    dataset_prefix = Path(dataset_prefix).resolve()
    outdir = _mk_bench_root(args.outdir)
    run_root = outdir / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    input_root = outdir / "inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    _copy_dataset(prefix=dataset_prefix, dst_dir=input_root)

    jobs = _build_jobs(args)
    python_cmd = ["python", str(repo_root / "csubst" / "csubst")]
    runtime_rows = []
    pq_rows = []
    refine_rows = []
    command_rows = []
    hash_rows = []
    run_records = []

    for rep in range(1, int(args.reps) + 1):
        for job in jobs:
            tag = "{}_rep{}".format(job["name"], rep)
            wd = run_root / tag
            wd.mkdir(parents=True, exist_ok=True)
            for src in input_root.iterdir():
                dst = wd / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
            cmd = python_cmd + [
                "search",
                "--alignment_file",
                "PGK.alignment.fa",
                "--rooted_tree_file",
                "PGK.tree.nwk",
                "--foreground",
                "PGK.foreground.txt",
                "--omegaC_method",
                "modelfree",
                "--output_stat",
                str(args.output_stat),
                "--calc_omega_pvalue",
                "yes",
                "--omega_pvalue_null_model",
                "hypergeom",
                "--omega_pvalue_rounding",
                str(args.rounding),
                "--omega_pvalue_niter",
                str(int(args.omega_pvalue_niter)),
                "--omega_pvalue_niter_schedule",
                str(job["schedule"]),
                "--omega_pvalue_refine_threshold",
                str(job["refine_threshold"]),
                "--omega_pvalue_refine_ci_alpha",
                str(job["refine_ci_alpha"]),
                "--threads",
                str(int(args.threads)),
                "--branch_dist",
                "no",
                "--b",
                "no",
                "--s",
                "no",
                "--cs",
                "no",
                "--bs",
                "no",
                "--cbs",
                "no",
                "--cb",
                "yes",
            ]
            stdout_path = wd / "stdout.log"
            stderr_path = wd / "stderr.log"
            rc, rendered_cmd = _run_timed(
                cmd=cmd,
                cwd=wd,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                dry_run=bool(args.dry_run),
            )
            command_rows.append({"rep": rep, "config": job["name"], "command": rendered_cmd, "cwd": str(wd)})
            if args.dry_run:
                continue
            if rc != 0:
                raise RuntimeError("Run failed ({}): {}".format(tag, rendered_cmd))
            cb_tsv = wd / "csubst_cb_2.tsv"
            log_path = wd / "csubst.log"
            cb_copy = wd / "csubst_cb_2.{}.tsv".format(tag)
            log_copy = wd / "csubst.{}.log".format(tag)
            shutil.copy2(cb_tsv, cb_copy)
            shutil.copy2(log_path, log_copy)
            real, user, sys, maxrss, peak = _parse_time_file(stderr_path)
            runtime_rows.append(
                {
                    "rep": rep,
                    "config": job["name"],
                    "schedule": str(job["schedule"]),
                    "refine_threshold": float(job["refine_threshold"]),
                    "refine_ci_alpha": float(job["refine_ci_alpha"]),
                    "is_baseline": bool(job["is_baseline"]),
                    "real_sec": real,
                    "user_sec": user,
                    "sys_sec": sys,
                    "maxrss_bytes": maxrss,
                    "peak_mem_bytes": peak,
                    "workdir": str(wd),
                }
            )
            ref_df = _extract_refinement_rows(log_path=log_path)
            if ref_df.shape[0]:
                ref_df.loc[:, "rep"] = rep
                ref_df.loc[:, "config"] = job["name"]
                ref_df.loc[:, "schedule"] = str(job["schedule"])
                ref_df.loc[:, "refine_threshold"] = float(job["refine_threshold"])
                ref_df.loc[:, "refine_ci_alpha"] = float(job["refine_ci_alpha"])
                refine_rows.append(ref_df)
            pq_df = _summarize_pq(cb_tsv=cb_tsv, output_stat=args.output_stat)
            if pq_df.shape[0]:
                pq_df.loc[:, "rep"] = rep
                pq_df.loc[:, "config"] = job["name"]
                pq_df.loc[:, "schedule"] = str(job["schedule"])
                pq_df.loc[:, "refine_threshold"] = float(job["refine_threshold"])
                pq_df.loc[:, "refine_ci_alpha"] = float(job["refine_ci_alpha"])
                pq_rows.append(pq_df)
            hash_rows.append({"rep": rep, "config": job["name"], "sha256_cb_2_tsv": _sha256(cb_tsv)})
            run_records.append({"rep": rep, "config": job["name"], "cb_tsv": str(cb_tsv), "schedule": str(job["schedule"])})

    pd.DataFrame(command_rows).to_csv(outdir / "commands.tsv", sep="\t", index=False)
    if args.dry_run:
        print("Dry-run commands were written to: {}".format(outdir / "commands.tsv"))
        return

    runtime_df = pd.DataFrame(runtime_rows)
    pq_df = pd.concat(pq_rows, ignore_index=True) if len(pq_rows) else pd.DataFrame()
    refine_df = pd.concat(refine_rows, ignore_index=True) if len(refine_rows) else pd.DataFrame()
    hash_df = pd.DataFrame(hash_rows)
    runtime_df.to_csv(outdir / "runtime_peak.tsv", sep="\t", index=False)
    pq_df.to_csv(outdir / "pq_summary.tsv", sep="\t", index=False)
    refine_df.to_csv(outdir / "refine_rows.tsv", sep="\t", index=False)
    hash_df.to_csv(outdir / "cb_hashes.tsv", sep="\t", index=False)

    compare_rows = []
    for rep in range(1, int(args.reps) + 1):
        baseline = [r for r in run_records if (r["rep"] == rep) and (r["config"] == "fixed")]
        if len(baseline) != 1:
            continue
        base_tsv = baseline[0]["cb_tsv"]
        for record in [r for r in run_records if (r["rep"] == rep) and (r["config"] != "fixed")]:
            cmp_df = _compare_cb_tsv(base_tsv=base_tsv, other_tsv=record["cb_tsv"], output_stat=args.output_stat)
            if cmp_df.shape[0] == 0:
                continue
            cmp_df.loc[:, "rep"] = rep
            cmp_df.loc[:, "config"] = record["config"]
            compare_rows.append(cmp_df)
    compare_df = pd.concat(compare_rows, ignore_index=True) if len(compare_rows) else pd.DataFrame()
    compare_df.to_csv(outdir / "compare_to_fixed.tsv", sep="\t", index=False)

    runtime_summary = runtime_df.groupby("config", as_index=False).agg(
        real_sec_mean=("real_sec", "mean"),
        real_sec_median=("real_sec", "median"),
        peak_mem_bytes_mean=("peak_mem_bytes", "mean"),
        maxrss_bytes_mean=("maxrss_bytes", "mean"),
    )
    runtime_summary.to_csv(outdir / "runtime_peak_summary.tsv", sep="\t", index=False)
    if refine_df.shape[0]:
        refine_summary = refine_df.groupby(["config", "output_stat", "stage"], as_index=False).agg(
            rows_before_mean=("rows_before", "mean"),
            rows_after_mean=("rows_after", "mean"),
            refine_fraction_mean=("refine_fraction", "mean"),
        )
        refine_summary.to_csv(outdir / "refine_rows_summary.tsv", sep="\t", index=False)

    print("Output directory: {}".format(outdir))
    print("runtime_peak.tsv rows: {}".format(runtime_df.shape[0]))
    print("pq_summary.tsv rows: {}".format(pq_df.shape[0]))
    print("refine_rows.tsv rows: {}".format(refine_df.shape[0]))
    print("compare_to_fixed.tsv rows: {}".format(compare_df.shape[0]))
    if runtime_summary.shape[0]:
        print(runtime_summary.to_string(index=False))


if __name__ == "__main__":
    main()
