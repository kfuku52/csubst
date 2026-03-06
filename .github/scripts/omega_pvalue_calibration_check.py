#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "PGK"
OUTPUT_STATS = ("any2spe", "any2any")
KINDS = ("nocalib", "calib")
MIN_SUB_PP_MAX_FPR = {
    ("any2spe", "nocalib"): 0.40,
    ("any2spe", "calib"): 0.50,
    ("any2any", "nocalib"): 0.20,
    ("any2any", "calib"): 0.20,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a lightweight omega p-value calibration regression sweep and "
            "capture runtime + peak RAM."
        )
    )
    parser.add_argument(
        "--output",
        default="omega_pvalue_calibration_summary.tsv",
        help="Output TSV for calibration summary metrics.",
    )
    parser.add_argument(
        "--runtime-output",
        default="omega_pvalue_calibration_runtime.tsv",
        help="Output TSV for runtime and peak-RAM metrics.",
    )
    parser.add_argument(
        "--workdir",
        default="/tmp/csubst_omega_pvalue_calibration",
        help="Temporary run directory (will be recreated).",
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=100,
        help="Number of randomization iterations passed to --omega_pvalue_niter.",
    )
    parser.add_argument(
        "--min-sub-pp-levels",
        default="0,0.05",
        help="Comma-separated --min_sub_pp levels.",
    )
    return parser.parse_args()


def parse_elapsed_seconds(elapsed_text):
    elapsed_text = elapsed_text.strip()
    if ":" not in elapsed_text:
        return float(elapsed_text)
    parts = [float(p) for p in elapsed_text.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + part
    return seconds


def parse_time_metrics(stderr_text):
    m_elapsed = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)",
        stderr_text,
    )
    m_rss_kb = re.search(
        r"Maximum resident set size \(kbytes\):\s*([0-9]+)",
        stderr_text,
    )
    if (m_elapsed is not None) and (m_rss_kb is not None):
        elapsed_sec = parse_elapsed_seconds(m_elapsed.group(1))
        max_rss_bytes = int(m_rss_kb.group(1)) * 1024
        return elapsed_sec, max_rss_bytes, max_rss_bytes

    m_elapsed_p = re.search(r"^real\s+([0-9.]+)$", stderr_text, flags=re.MULTILINE)
    if m_elapsed_p is None:
        raise RuntimeError("Failed to parse elapsed time from /usr/bin/time output.")
    elapsed_sec = float(m_elapsed_p.group(1))
    m_peak = re.search(r"^\s*([0-9]+)\s+peak memory footprint$", stderr_text, flags=re.MULTILINE)
    m_rss = re.search(
        r"^\s*([0-9]+)\s+maximum resident set size$",
        stderr_text,
        flags=re.MULTILINE,
    )
    max_rss_bytes = int(m_rss.group(1)) if m_rss is not None else -1
    peak_mem_bytes = int(m_peak.group(1)) if m_peak is not None else max_rss_bytes
    return elapsed_sec, max_rss_bytes, peak_mem_bytes


def run_timed_command(cmd, cwd, label, env=None):
    if sys.platform.startswith("linux"):
        wrapper = ["/usr/bin/time", "-v"]
    elif sys.platform == "darwin":
        wrapper = ["/usr/bin/time", "-l", "-p"]
    else:
        wrapper = ["/usr/bin/time", "-p"]
    proc = subprocess.run(
        wrapper + cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout_path = Path(cwd) / (label + ".stdout.log")
    stderr_path = Path(cwd) / (label + ".stderr.log")
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed ({}): {}\nSee {}\n{}".format(
                proc.returncode,
                " ".join(cmd),
                str(stdout_path),
                str(stderr_path),
            )
        )
    elapsed_sec, max_rss_bytes, peak_mem_bytes = parse_time_metrics(proc.stderr)
    return elapsed_sec, max_rss_bytes, peak_mem_bytes, stdout_path, stderr_path


def ensure_precomputed_iqtree_outputs(repo_root, dataset_name):
    dataset_dir = repo_root / "csubst" / "dataset"
    required = [
        dataset_dir / f"{dataset_name}.alignment.fa",
        dataset_dir / f"{dataset_name}.tree.nwk",
        dataset_dir / f"{dataset_name}.foreground.txt",
        dataset_dir / f"{dataset_name}.alignment.fa.treefile",
        dataset_dir / f"{dataset_name}.alignment.fa.state",
        dataset_dir / f"{dataset_name}.alignment.fa.rate",
        dataset_dir / f"{dataset_name}.alignment.fa.iqtree",
        dataset_dir / f"{dataset_name}.alignment.fa.log",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if len(missing) > 0:
        raise FileNotFoundError(
            "Missing precomputed inputs for {}:\n{}".format(
                dataset_name, "\n".join(missing)
            )
        )
    return required


def read_iqtree_model(iqtree_report_path):
    iqtree_report_path = Path(iqtree_report_path)
    pattern = re.compile(r"^\s*Model of substitution:\s*(.+?)\s*$")
    with iqtree_report_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.match(line)
            if match is not None:
                model = match.group(1).strip()
                if model != "":
                    return model
                break
    raise RuntimeError("Failed to read IQ-TREE model from {}".format(str(iqtree_report_path)))


def parse_min_sub_pp_levels(levels_text):
    out = []
    for token in [s.strip() for s in str(levels_text).split(",") if s.strip()]:
        out.append(float(token))
    if len(out) < 2:
        raise ValueError("--min-sub-pp-levels should provide at least two values.")
    # Keep order for baseline/guard semantics while dropping exact duplicates.
    deduped = []
    seen = set()
    for value in out:
        key = ("{:.12g}".format(value))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _min_sub_pp_tag(value):
    return ("{:.12g}".format(value)).replace(".", "_")


def _p_col(output_stat, kind):
    if kind == "nocalib":
        return "pomegaC{}_nocalib".format(output_stat)
    return "pomegaC{}".format(output_stat)


def run_setting(repo_root, run_root, output_stat, min_sub_pp, niter):
    run_dir = Path(run_root) / "{}_min_sub_pp_{}".format(output_stat, _min_sub_pp_tag(min_sub_pp))
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = repo_root / "csubst" / "dataset"
    python_exe = sys.executable
    csubst_exe = repo_root / "csubst" / "csubst"
    iqtree_model = read_iqtree_model(dataset_dir / "{}.alignment.fa.iqtree".format(DATASET_NAME))

    cmd = [
        python_exe,
        str(csubst_exe),
        "analyze",
        "--outdir",
        ".",
        "--output_prefix",
        "csubst",
        "--alignment_file",
        str(dataset_dir / "{}.alignment.fa".format(DATASET_NAME)),
        "--rooted_tree_file",
        str(dataset_dir / "{}.tree.nwk".format(DATASET_NAME)),
        "--foreground",
        str(dataset_dir / "{}.foreground.txt".format(DATASET_NAME)),
        "--omegaC_method",
        "modelfree",
        "--output_stat",
        output_stat,
        "--calc_omega_pvalue",
        "yes",
        "--min_sub_pp",
        "{:.12g}".format(min_sub_pp),
        "--omega_pvalue_niter",
        str(int(niter)),
        "--omega_pvalue_rounding",
        "round",
        "--calibrate_longtail",
        "yes",
        "--threads",
        "1",
        "--iqtree_treefile",
        str(dataset_dir / "{}.alignment.fa.treefile".format(DATASET_NAME)),
        "--iqtree_state",
        str(dataset_dir / "{}.alignment.fa.state".format(DATASET_NAME)),
        "--iqtree_rate",
        str(dataset_dir / "{}.alignment.fa.rate".format(DATASET_NAME)),
        "--iqtree_iqtree",
        str(dataset_dir / "{}.alignment.fa.iqtree".format(DATASET_NAME)),
        "--iqtree_log",
        str(dataset_dir / "{}.alignment.fa.log".format(DATASET_NAME)),
        "--iqtree_model",
        iqtree_model,
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

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + (
        os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else ""
    )
    elapsed_sec, max_rss_bytes, peak_mem_bytes, stdout_path, stderr_path = run_timed_command(
        cmd=cmd,
        cwd=run_dir,
        label="analyze",
        env=env,
    )

    cb_path = run_dir / "csubst_cb_2.tsv"
    if not cb_path.exists():
        raise FileNotFoundError("Missing analyze output: {}".format(str(cb_path)))
    cb = pd.read_csv(cb_path, sep="\t")

    summary_rows = []
    for kind in KINDS:
        col = _p_col(output_stat=output_stat, kind=kind)
        if col not in cb.columns:
            raise ValueError("Column '{}' missing in {}".format(col, str(cb_path)))
        pvalues = pd.to_numeric(cb[col], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(pvalues)
        frac = float(np.mean(pvalues[finite] <= 0.05)) if finite.any() else np.nan
        mean = float(np.nanmean(pvalues)) if finite.any() else np.nan
        median = float(np.nanmedian(pvalues)) if finite.any() else np.nan
        summary_rows.append(
            {
                "output_stat": output_stat,
                "min_sub_pp": float(min_sub_pp),
                "kind": kind,
                "n": int(finite.sum()),
                "frac_p_le_0_05": frac,
                "mean_p": mean,
                "median_p": median,
                "run_dir": str(run_dir),
                "cb_path": str(cb_path),
            }
        )

    runtime_row = {
        "output_stat": output_stat,
        "min_sub_pp": float(min_sub_pp),
        "real_sec": float(elapsed_sec),
        "maxrss_bytes": int(max_rss_bytes) if max_rss_bytes is not None else -1,
        "peak_mem_bytes": int(peak_mem_bytes) if peak_mem_bytes is not None else -1,
        "run_dir": str(run_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    return summary_rows, runtime_row


def _select_fpr(df, output_stat, kind, min_sub_pp):
    tol = 1e-12
    picked = df.loc[
        (df["output_stat"] == output_stat)
        & (df["kind"] == kind)
        & (np.abs(df["min_sub_pp"].astype(np.float64) - float(min_sub_pp)) <= tol),
        "frac_p_le_0_05",
    ]
    if picked.shape[0] != 1:
        raise RuntimeError(
            "Expected one row for output_stat={}, kind={}, min_sub_pp={} but found {}.".format(
                output_stat, kind, min_sub_pp, picked.shape[0]
            )
        )
    return float(picked.iloc[0])


def validate_summary(summary_df, baseline_min_sub_pp, guarded_min_sub_pp):
    errors = []
    for output_stat in OUTPUT_STATS:
        for kind in KINDS:
            baseline = _select_fpr(
                summary_df,
                output_stat=output_stat,
                kind=kind,
                min_sub_pp=baseline_min_sub_pp,
            )
            guarded = _select_fpr(
                summary_df,
                output_stat=output_stat,
                kind=kind,
                min_sub_pp=guarded_min_sub_pp,
            )
            if (not np.isfinite(baseline)) or (not np.isfinite(guarded)):
                errors.append(
                    "{} {}: non-finite fpr baseline={} guarded={}".format(
                        output_stat, kind, baseline, guarded
                    )
                )
                continue
            if guarded > baseline + 1e-12:
                errors.append(
                    "{} {}: guarded min_sub_pp={} fpr {} should not exceed baseline min_sub_pp={} fpr {}".format(
                        output_stat,
                        kind,
                        guarded_min_sub_pp,
                        guarded,
                        baseline_min_sub_pp,
                        baseline,
                    )
                )
            cap = MIN_SUB_PP_MAX_FPR[(output_stat, kind)]
            if guarded > cap:
                errors.append(
                    "{} {}: guarded min_sub_pp={} fpr {} exceeds cap {}".format(
                        output_stat,
                        kind,
                        guarded_min_sub_pp,
                        guarded,
                        cap,
                    )
                )
    if len(errors) > 0:
        raise RuntimeError("Calibration regression failed:\n- " + "\n- ".join(errors))


def validate_runtime(runtime_df):
    errors = []
    for row in runtime_df.itertuples(index=False):
        if (not np.isfinite(float(row.real_sec))) or (float(row.real_sec) <= 0):
            errors.append(
                "{} min_sub_pp={}: invalid real_sec={}".format(
                    row.output_stat, row.min_sub_pp, row.real_sec
                )
            )
        peak_mem = int(row.peak_mem_bytes)
        if peak_mem <= 0:
            errors.append(
                "{} min_sub_pp={}: invalid peak_mem_bytes={}".format(
                    row.output_stat, row.min_sub_pp, row.peak_mem_bytes
                )
            )
    if len(errors) > 0:
        raise RuntimeError("Runtime/peak-memory regression failed:\n- " + "\n- ".join(errors))


def main():
    args = parse_args()
    if int(args.niter) <= 0:
        raise ValueError("--niter should be > 0.")
    min_sub_pp_levels = parse_min_sub_pp_levels(args.min_sub_pp_levels)
    if len(min_sub_pp_levels) < 2:
        raise ValueError("--min-sub-pp-levels should include baseline and guarded settings.")
    baseline_min_sub_pp = min_sub_pp_levels[0]
    guarded_min_sub_pp = min_sub_pp_levels[1]
    if abs(baseline_min_sub_pp - 0.0) > 1e-12:
        raise ValueError("The first --min-sub-pp-levels value should be 0 (baseline).")
    if abs(guarded_min_sub_pp - 0.05) > 1e-12:
        raise ValueError("The second --min-sub-pp-levels value should be 0.05 (guarded).")

    repo_root = REPO_ROOT
    ensure_precomputed_iqtree_outputs(repo_root=repo_root, dataset_name=DATASET_NAME)

    run_root = Path(args.workdir).resolve()
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    runtime_rows = []
    for output_stat in OUTPUT_STATS:
        for min_sub_pp in min_sub_pp_levels:
            rows, runtime_row = run_setting(
                repo_root=repo_root,
                run_root=run_root,
                output_stat=output_stat,
                min_sub_pp=min_sub_pp,
                niter=int(args.niter),
            )
            summary_rows.extend(rows)
            runtime_rows.append(runtime_row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["output_stat", "kind", "min_sub_pp"]
    ).reset_index(drop=True)
    runtime_df = pd.DataFrame(runtime_rows).sort_values(
        by=["output_stat", "min_sub_pp"]
    ).reset_index(drop=True)

    validate_summary(
        summary_df=summary_df,
        baseline_min_sub_pp=baseline_min_sub_pp,
        guarded_min_sub_pp=guarded_min_sub_pp,
    )
    validate_runtime(runtime_df=runtime_df)

    out_path = Path(args.output).resolve()
    runtime_out_path = Path(args.runtime_output).resolve()
    summary_df.to_csv(out_path, sep="\t", index=False)
    runtime_df.to_csv(runtime_out_path, sep="\t", index=False)

    print(summary_df.to_string(index=False))
    print(runtime_df.to_string(index=False))
    print("Wrote calibration summary: {}".format(str(out_path)))
    print("Wrote runtime summary: {}".format(str(runtime_out_path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
