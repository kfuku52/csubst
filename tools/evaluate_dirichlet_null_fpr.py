#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_runs(spec):
    runs = {}
    for token in [t.strip() for t in str(spec).split(",") if t.strip()]:
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


def _wilson_ci(k, n, z=1.96):
    if n <= 0:
        return np.nan, np.nan
    p = float(k) / float(n)
    den = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / den
    radius = (z / den) * np.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n)))
    return center - radius, center + radius


def _to_indicator(df, ocn_col, omega_col, ocn_cutoff, omega_cutoff):
    ocn = pd.to_numeric(df[ocn_col], errors="coerce").values
    omg = pd.to_numeric(df[omega_col], errors="coerce").values
    valid = np.isfinite(ocn) & np.isfinite(omg)
    hit = np.zeros(df.shape[0], dtype=bool)
    hit[valid] = (ocn[valid] >= float(ocn_cutoff)) & (omg[valid] >= float(omega_cutoff))
    return hit


def _paired_permutation_pvalue(diff, nperm, seed):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=np.int8).reshape(-1)
    n_total = int(diff.shape[0])
    if n_total == 0:
        return np.nan, np.nan, np.nan, 0
    nonzero = diff[diff != 0]
    n_discord = int(nonzero.shape[0])
    observed_delta = diff.mean(dtype=np.float64)
    if n_discord == 0:
        return observed_delta, 1.0, 1.0, 0
    # Under null, sign is exchangeable within each discordant pair.
    # Because discordant diffs are +/-1, the null distribution depends only on n_discord.
    k = rng.binomial(n=n_discord, p=0.5, size=int(nperm))
    delta_perm = (2.0 * k - float(n_discord)) / float(n_total)
    p_less = (np.sum(delta_perm <= observed_delta) + 1.0) / (float(nperm) + 1.0)
    p_two = (np.sum(np.abs(delta_perm) >= abs(observed_delta)) + 1.0) / (float(nperm) + 1.0)
    return observed_delta, float(p_less), float(p_two), n_discord


def _plot_fpr(summary_df, out_png):
    d = summary_df.sort_values("alpha").reset_index(drop=True)
    labels = d["alpha"].astype(str).tolist()
    x = np.arange(d.shape[0], dtype=np.float64)
    y = d["fpr_rate"].astype(float).values
    lo = d["wilson_lo"].astype(float).values
    hi = d["wilson_hi"].astype(float).values
    yerr = np.vstack([np.clip(y - lo, a_min=0, a_max=None), np.clip(hi - y, a_min=0, a_max=None)])

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(x, y, color="#4C78A8", alpha=0.88)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=4, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Dirichlet alpha")
    ax.set_ylabel("Null FPR (hit rate)")
    ax.set_title("Null FPR by Dirichlet alpha")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    psr = argparse.ArgumentParser(
        description="Evaluate null false-positive rate reduction by ASRV Dirichlet alpha."
    )
    psr.add_argument("--runs", required=True, help="Comma-separated alpha:path entries (example: 0.0:/tmp/off,1.0:/tmp/on).")
    psr.add_argument("--baseline-alpha", type=float, default=0.0)
    psr.add_argument("--ocn-col", default="OCNany2spe")
    psr.add_argument("--omega-col", default="omegaCany2spe")
    psr.add_argument("--ocn-cutoff", type=float, default=2.0)
    psr.add_argument("--omega-cutoff", type=float, default=5.0)
    psr.add_argument("--nperm", type=int, default=200000)
    psr.add_argument("--seed", type=int, default=12345)
    psr.add_argument("--outdir", required=True)
    args = psr.parse_args()

    if args.nperm <= 0:
        raise ValueError("--nperm should be > 0.")

    runs = _parse_runs(args.runs)
    if args.baseline_alpha not in runs:
        raise ValueError("--baseline-alpha {} was not found in --runs.".format(args.baseline_alpha))

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    run_tables = {}
    runtime_rows = []
    for alpha, run_dir in runs.items():
        cb_path = run_dir / "csubst_cb_2.tsv"
        if not cb_path.exists():
            raise FileNotFoundError("Missing {}".format(cb_path))
        df = pd.read_csv(cb_path, sep="\t")
        if args.ocn_col not in df.columns:
            raise ValueError("Column '{}' missing in alpha={}.".format(args.ocn_col, alpha))
        if args.omega_col not in df.columns:
            raise ValueError("Column '{}' missing in alpha={}.".format(args.omega_col, alpha))
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

    summary_rows = []
    baseline_hit = _to_indicator(
        df=baseline,
        ocn_col=args.ocn_col,
        omega_col=args.omega_col,
        ocn_cutoff=args.ocn_cutoff,
        omega_cutoff=args.omega_cutoff,
    )
    for alpha, df in run_tables.items():
        hit = _to_indicator(
            df=df,
            ocn_col=args.ocn_col,
            omega_col=args.omega_col,
            ocn_cutoff=args.ocn_cutoff,
            omega_cutoff=args.omega_cutoff,
        )
        n = int(hit.shape[0])
        k = int(hit.sum())
        lo, hi = _wilson_ci(k=k, n=n)
        summary_rows.append(
            {
                "alpha": alpha,
                "run_dir": str(runs[alpha]),
                "n_total": n,
                "n_hit": k,
                "fpr_rate": float(k) / float(n) if n > 0 else np.nan,
                "wilson_lo": lo,
                "wilson_hi": hi,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("alpha").reset_index(drop=True)
    summary_df.to_csv(outdir / "null_fpr_by_alpha.tsv", sep="\t", index=False)

    baseline_id_hit = baseline.loc[:, id_cols].copy()
    baseline_id_hit["hit_base"] = baseline_hit.astype(np.int8)
    paired_rows = []
    for alpha, df in run_tables.items():
        if alpha == args.baseline_alpha:
            continue
        cur = df.loc[:, id_cols].copy()
        cur["hit_cur"] = _to_indicator(
            df=df,
            ocn_col=args.ocn_col,
            omega_col=args.omega_col,
            ocn_cutoff=args.ocn_cutoff,
            omega_cutoff=args.omega_cutoff,
        ).astype(np.int8)
        merged = baseline_id_hit.merge(cur, on=id_cols, how="inner")
        diff = (merged["hit_cur"].values - merged["hit_base"].values).astype(np.int8)
        observed_delta, p_less, p_two, n_discord = _paired_permutation_pvalue(
            diff=diff,
            nperm=args.nperm,
            seed=args.seed + int(round(alpha * 1000.0)),
        )
        n_total = int(diff.shape[0])
        n_base = int((merged["hit_base"] == 1).sum())
        n_cur = int((merged["hit_cur"] == 1).sum())
        n_10 = int(np.sum((merged["hit_base"] == 1) & (merged["hit_cur"] == 0)))
        n_01 = int(np.sum((merged["hit_base"] == 0) & (merged["hit_cur"] == 1)))
        paired_rows.append(
            {
                "alpha": alpha,
                "baseline_alpha": args.baseline_alpha,
                "n_total_pairs": n_total,
                "n_discordant": n_discord,
                "n_base_hit": n_base,
                "n_alpha_hit": n_cur,
                "n_base1_alpha0": n_10,
                "n_base0_alpha1": n_01,
                "observed_delta_rate": observed_delta,
                "pvalue_one_sided_alpha_lt_baseline": p_less,
                "pvalue_two_sided": p_two,
            }
        )
    paired_df = pd.DataFrame(paired_rows).sort_values("alpha").reset_index(drop=True)
    paired_df.to_csv(outdir / "null_fpr_paired_permutation.tsv", sep="\t", index=False)

    runtime_df = pd.DataFrame(runtime_rows).sort_values("alpha").reset_index(drop=True)
    runtime_df.to_csv(outdir / "runtime_peak.tsv", sep="\t", index=False)
    _plot_fpr(summary_df=summary_df, out_png=outdir / "null_fpr_by_alpha.png")

    metadata = {
        "runs": {str(alpha): str(path) for alpha, path in runs.items()},
        "baseline_alpha": args.baseline_alpha,
        "ocn_col": args.ocn_col,
        "omega_col": args.omega_col,
        "ocn_cutoff": args.ocn_cutoff,
        "omega_cutoff": args.omega_cutoff,
        "nperm": args.nperm,
        "seed": args.seed,
        "id_cols": id_cols,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
