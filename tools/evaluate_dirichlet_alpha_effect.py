#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_runs(spec):
    runs = dict()
    for token in [t.strip() for t in str(spec).split(",") if t.strip() != ""]:
        if ":" not in token:
            raise ValueError("Each --runs token should be alpha:path.")
        alpha_txt, path_txt = token.split(":", 1)
        alpha = float(alpha_txt)
        run_dir = Path(path_txt).expanduser().resolve()
        if not run_dir.exists():
            raise FileNotFoundError("Run directory not found: {}".format(run_dir))
        runs[alpha] = run_dir
    if len(runs) < 2:
        raise ValueError("At least two runs should be provided.")
    return dict(sorted(runs.items(), key=lambda x: x[0]))


def _parse_time_file(path):
    if not path.exists():
        return np.nan, np.nan
    real = np.nan
    peak = np.nan
    for line in path.read_text().splitlines():
        if line.startswith("real "):
            real = float(line.split()[1])
        if "peak memory footprint" in line:
            peak = float(line.split()[0])
        if np.isnan(peak) and ("maximum resident set size" in line):
            peak = float(line.split()[0])
    return real, peak


def _to_log10_positive(series):
    series = pd.to_numeric(series, errors="coerce")
    mask = np.isfinite(series.values) & (series.values > 0)
    if not mask.any():
        return np.array([], dtype=float), mask
    return np.log10(series.values[mask]), mask


def _detect_omega_cols(df):
    cols = []
    for col in df.columns:
        if not col.startswith("omegaC"):
            continue
        if col.endswith("_nocalib"):
            continue
        cols.append(col)
    return sorted(cols)


def _detect_info_score(df):
    if {"OCNany2spe", "OCSany2spe"}.issubset(df.columns):
        return pd.to_numeric(df["OCNany2spe"], errors="coerce").fillna(0) + pd.to_numeric(
            df["OCSany2spe"], errors="coerce"
        ).fillna(0)
    if "OCNany2spe" in df.columns:
        return pd.to_numeric(df["OCNany2spe"], errors="coerce").fillna(0)
    if "OCSany2spe" in df.columns:
        return pd.to_numeric(df["OCSany2spe"], errors="coerce").fillna(0)
    return pd.Series(np.nan, index=df.index)


def _calc_table_stats(alpha, df, omega_cols):
    rows = []
    for col in omega_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        finite = np.isfinite(series.values)
        positive = finite & (series.values > 0)
        logv = np.log10(series.values[positive]) if positive.any() else np.array([], dtype=float)
        rows.append(
            {
                "alpha": alpha,
                "column": col,
                "rows_total": int(series.shape[0]),
                "finite_n": int(finite.sum()),
                "positive_n": int(positive.sum()),
                "zero_or_negative_n": int((finite & (series.values <= 0)).sum()),
                "inf_n": int(np.isinf(series.values).sum()),
                "q95_log10": float(np.quantile(logv, 0.95)) if logv.size else np.nan,
                "q99_log10": float(np.quantile(logv, 0.99)) if logv.size else np.nan,
                "q999_log10": float(np.quantile(logv, 0.999)) if logv.size else np.nan,
                "median_log10": float(np.median(logv)) if logv.size else np.nan,
                "gt10_n": int((finite & (series.values > 10)).sum()),
                "gt100_n": int((finite & (series.values > 100)).sum()),
            }
        )
    return rows


def _calc_shift_stats(alpha, merged, omega_cols):
    rows = []
    for col in omega_cols:
        base = pd.to_numeric(merged[col + "_base"], errors="coerce").values
        cur = pd.to_numeric(merged[col + "_cur"], errors="coerce").values
        mask = np.isfinite(base) & np.isfinite(cur) & (base > 0) & (cur > 0)
        if not mask.any():
            rows.append(
                {
                    "alpha": alpha,
                    "column": col,
                    "pair_n": 0,
                    "pearson_log10": np.nan,
                    "median_log10_shift": np.nan,
                    "median_abs_log10_shift": np.nan,
                    "q05_log10_shift": np.nan,
                    "q95_log10_shift": np.nan,
                }
            )
            continue
        log_base = np.log10(base[mask])
        log_cur = np.log10(cur[mask])
        log_diff = log_cur - log_base
        pearson = np.corrcoef(log_base, log_cur)[0, 1] if log_base.size > 1 else np.nan
        rows.append(
            {
                "alpha": alpha,
                "column": col,
                "pair_n": int(mask.sum()),
                "pearson_log10": float(pearson) if np.isfinite(pearson) else np.nan,
                "median_log10_shift": float(np.median(log_diff)),
                "median_abs_log10_shift": float(np.median(np.abs(log_diff))),
                "q05_log10_shift": float(np.quantile(log_diff, 0.05)),
                "q95_log10_shift": float(np.quantile(log_diff, 0.95)),
            }
        )
    return rows


def _calc_shift_stats_by_info(alpha, merged, omega_cols):
    labels = ["low<=2", "mid3to10", "high>10"]
    bins = [-np.inf, 2.0, 10.0, np.inf]
    info = pd.to_numeric(merged["info_score"], errors="coerce")
    strata = pd.cut(info, bins=bins, labels=labels, right=True, include_lowest=True)
    rows = []
    for col in omega_cols:
        base = pd.to_numeric(merged[col + "_base"], errors="coerce").values
        cur = pd.to_numeric(merged[col + "_cur"], errors="coerce").values
        valid = np.isfinite(base) & np.isfinite(cur) & (base > 0) & (cur > 0)
        log_diff = np.full(shape=base.shape, fill_value=np.nan, dtype=float)
        log_diff[valid] = np.log10(cur[valid]) - np.log10(base[valid])
        for label in labels:
            idx = (strata == label).values & np.isfinite(log_diff)
            if idx.sum() == 0:
                med = np.nan
                mad = np.nan
            else:
                med = float(np.median(log_diff[idx]))
                mad = float(np.median(np.abs(log_diff[idx])))
            rows.append(
                {
                    "alpha": alpha,
                    "column": col,
                    "info_stratum": label,
                    "pair_n": int(idx.sum()),
                    "median_log10_shift": med,
                    "median_abs_log10_shift": mad,
                }
            )
    return rows


def _plot_tail_quantiles(table_stats, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for col in sorted(table_stats["column"].unique().tolist()):
        dfc = table_stats.loc[table_stats["column"] == col, :].sort_values("alpha")
        axes[0].plot(dfc["alpha"], dfc["q99_log10"], marker="o", label=col)
        axes[1].plot(dfc["alpha"], dfc["gt100_n"], marker="o", label=col)
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("q99(log10 omega)")
    axes[0].set_title("Upper tail by alpha")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("count(omega > 100)")
    axes[1].set_title("Extreme omega count by alpha")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_shift_summary(shift_stats, out_png):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for col in sorted(shift_stats["column"].unique().tolist()):
        dfc = shift_stats.loc[shift_stats["column"] == col, :].sort_values("alpha")
        ax.plot(dfc["alpha"], dfc["median_abs_log10_shift"], marker="o", label=col)
    ax.set_xlabel("alpha")
    ax.set_ylabel("median |log10(alpha) - log10(baseline)|")
    ax.set_title("Deviation from baseline(alpha=0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_runtime(runtime_df, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(runtime_df["alpha"].astype(str), runtime_df["real_sec"])
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("Runtime (sec)")
    axes[0].set_title("Runtime")
    axes[1].bar(runtime_df["alpha"].astype(str), runtime_df["peak_mib"])
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("Peak RAM (MiB)")
    axes[1].set_title("Peak RAM")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    psr = argparse.ArgumentParser(
        description="Evaluate effect of ASRV Dirichlet alpha using precomputed csubst_cb_2.tsv runs."
    )
    psr.add_argument("--runs", required=True, help="Comma-separated alpha:path entries (example: 0.0:/tmp/off,0.5:/tmp/half).")
    psr.add_argument("--baseline-alpha", type=float, default=0.0)
    psr.add_argument("--outdir", required=True)
    args = psr.parse_args()

    runs = _parse_runs(args.runs)
    if args.baseline_alpha not in runs:
        raise ValueError("--baseline-alpha {} was not found in --runs.".format(args.baseline_alpha))

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    run_tables = dict()
    runtime_rows = []
    for alpha, run_dir in runs.items():
        cb_path = run_dir / "csubst_cb_2.tsv"
        if not cb_path.exists():
            raise FileNotFoundError("Missing {}".format(cb_path))
        df = pd.read_csv(cb_path, sep="\t")
        run_tables[alpha] = df
        real, peak = _parse_time_file(run_dir / "run.stderr.log")
        runtime_rows.append(
            {
                "alpha": alpha,
                "run_dir": str(run_dir),
                "real_sec": real,
                "peak_bytes": peak,
                "peak_mib": peak / (1024 * 1024) if np.isfinite(peak) else np.nan,
            }
        )

    baseline = run_tables[args.baseline_alpha]
    id_cols = [c for c in baseline.columns if c.startswith("branch_id_")]
    if len(id_cols) == 0:
        raise ValueError("No branch_id_ columns in baseline table.")
    omega_cols = _detect_omega_cols(baseline)
    if len(omega_cols) == 0:
        raise ValueError("No omegaC columns found in baseline table.")
    for alpha, df in run_tables.items():
        missing_cols = sorted(list(set(omega_cols).difference(set(df.columns))))
        if len(missing_cols):
            raise ValueError("Run alpha={} was missing omega columns: {}".format(alpha, ",".join(missing_cols)))
        missing_ids = sorted(list(set(id_cols).difference(set(df.columns))))
        if len(missing_ids):
            raise ValueError("Run alpha={} was missing id columns: {}".format(alpha, ",".join(missing_ids)))

    baseline_info = baseline.loc[:, id_cols].copy()
    baseline_info.loc[:, "info_score"] = _detect_info_score(baseline)

    table_rows = []
    shift_rows = []
    shift_info_rows = []
    for alpha, df in run_tables.items():
        table_rows.extend(_calc_table_stats(alpha=alpha, df=df, omega_cols=omega_cols))
        if alpha == args.baseline_alpha:
            continue
        merged = baseline.loc[:, id_cols + omega_cols].merge(
            df.loc[:, id_cols + omega_cols],
            on=id_cols,
            suffixes=("_base", "_cur"),
            how="inner",
        )
        merged = merged.merge(baseline_info, on=id_cols, how="left")
        shift_rows.extend(_calc_shift_stats(alpha=alpha, merged=merged, omega_cols=omega_cols))
        shift_info_rows.extend(_calc_shift_stats_by_info(alpha=alpha, merged=merged, omega_cols=omega_cols))

    table_stats = pd.DataFrame(table_rows).sort_values(["column", "alpha"]).reset_index(drop=True)
    shift_stats = pd.DataFrame(shift_rows).sort_values(["column", "alpha"]).reset_index(drop=True)
    shift_by_info = pd.DataFrame(shift_info_rows).sort_values(["column", "info_stratum", "alpha"]).reset_index(drop=True)
    runtime_df = pd.DataFrame(runtime_rows).sort_values("alpha").reset_index(drop=True)

    table_stats.to_csv(outdir / "alpha_table_stats.tsv", sep="\t", index=False)
    shift_stats.to_csv(outdir / "alpha_vs_baseline_shift.tsv", sep="\t", index=False)
    shift_by_info.to_csv(outdir / "alpha_vs_baseline_shift_by_info.tsv", sep="\t", index=False)
    runtime_df.to_csv(outdir / "runtime_peak.tsv", sep="\t", index=False)

    _plot_tail_quantiles(table_stats=table_stats, out_png=outdir / "alpha_tail_summary.png")
    _plot_shift_summary(shift_stats=shift_stats, out_png=outdir / "alpha_shift_summary.png")
    _plot_runtime(runtime_df=runtime_df, out_png=outdir / "runtime_peak.png")

    metadata = {
        "runs": {str(alpha): str(path) for alpha, path in runs.items()},
        "baseline_alpha": args.baseline_alpha,
        "id_cols": id_cols,
        "omega_cols": omega_cols,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
