#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_time_file(path):
    path = Path(path)
    if not path.exists():
        return np.nan, np.nan
    real = np.nan
    peak = np.nan
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("real "):
            parts = line.split()
            if len(parts) >= 2:
                real = float(parts[1])
        if "peak memory footprint" in line:
            parts = line.split()
            if len(parts) >= 1:
                peak = float(parts[0])
        if np.isnan(peak) and ("maximum resident set size" in line):
            parts = line.split()
            if len(parts) >= 1:
                peak = float(parts[0])
    return real, peak


def _sha256(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _average_ranks(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    n = values.shape[0]
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while (j < n) and (values[order[j]] == values[order[i]]):
            j += 1
        rank = 0.5 * (float(i + 1) + float(j))
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _calc_auc(scores, labels):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    valid = np.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if (n_pos == 0) or (n_neg == 0):
        return np.nan
    ranks = _average_ranks(scores)
    sum_pos = float(ranks[labels].sum())
    auc = (sum_pos - float(n_pos) * float(n_pos + 1) / 2.0) / (float(n_pos) * float(n_neg))
    return float(auc)


def _calc_average_precision(scores, labels):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    valid = np.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    n_pos = int(labels.sum())
    if n_pos == 0:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    sorted_pos = labels[order].astype(np.int64)
    cum_pos = np.cumsum(sorted_pos)
    hit_idx = np.where(sorted_pos == 1)[0]
    precision = cum_pos[hit_idx] / (hit_idx.astype(np.float64) + 1.0)
    return float(np.mean(precision))


def _calc_precision_at_k(scores, labels, k):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    valid = np.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    if scores.size == 0:
        return np.nan
    k = int(k)
    if k <= 0:
        return np.nan
    k = min(k, int(scores.size))
    order = np.argsort(-scores, kind="mergesort")
    return float(labels[order][:k].mean())


def _calc_fg_detection_metrics(df, score_col):
    if "is_fg" not in df.columns:
        raise ValueError("Column 'is_fg' is required in csubst_cb_2.tsv.")
    if score_col not in df.columns:
        raise ValueError("Score column '{}' was not found in csubst_cb_2.tsv.".format(score_col))
    score = pd.to_numeric(df[score_col], errors="coerce").values
    labels = df["is_fg"].astype(str).str.upper().values == "Y"
    valid = np.isfinite(score)
    score = score[valid]
    labels = labels[valid]
    n_total = int(score.shape[0])
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    precision_at_k = _calc_precision_at_k(scores=score, labels=labels, k=max(n_pos, 1))
    return {
        "n_total_valid": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auroc": _calc_auc(scores=score, labels=labels),
        "average_precision": _calc_average_precision(scores=score, labels=labels),
        "precision_at_k": precision_at_k,
    }


def _calc_similarity_to_baseline(df_base, df_cur, score_col, jaccard_threshold):
    id_cols = [c for c in df_base.columns if c.startswith("branch_id_")]
    if len(id_cols) == 0:
        raise ValueError("No branch_id_* columns were found in baseline csubst_cb_2.tsv.")
    if score_col not in df_base.columns:
        raise ValueError("Score column '{}' missing in baseline csubst_cb_2.tsv.".format(score_col))
    if score_col not in df_cur.columns:
        raise ValueError("Score column '{}' missing in current csubst_cb_2.tsv.".format(score_col))

    merged = df_base.loc[:, id_cols + [score_col]].merge(
        df_cur.loc[:, id_cols + [score_col]],
        on=id_cols,
        suffixes=("_base", "_cur"),
        how="inner",
    )
    base = pd.to_numeric(merged[score_col + "_base"], errors="coerce").values
    cur = pd.to_numeric(merged[score_col + "_cur"], errors="coerce").values
    valid_log = np.isfinite(base) & np.isfinite(cur) & (base > 0) & (cur > 0)
    if int(valid_log.sum()) >= 2:
        pearson = float(np.corrcoef(np.log10(base[valid_log]), np.log10(cur[valid_log]))[0, 1])
    else:
        pearson = np.nan

    hit_base = np.isfinite(base) & (base >= float(jaccard_threshold))
    hit_cur = np.isfinite(cur) & (cur >= float(jaccard_threshold))
    inter = int(np.logical_and(hit_base, hit_cur).sum())
    union = int(np.logical_or(hit_base, hit_cur).sum())
    jaccard = (float(inter) / float(union)) if union > 0 else np.nan
    return {
        "n_pair": int(merged.shape[0]),
        "pearson_r_log10": pearson,
        "jaccard": jaccard,
        "intersection": inter,
        "union": union,
    }


def _zscore(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    mean = float(values.mean())
    sd = float(values.std(ddof=0))
    if sd <= 0:
        return np.zeros(shape=values.shape, dtype=float)
    return (values - mean) / sd


def _write_epistasis_degree_table(path, num_site, num_convergent_site, scenario, rng):
    if num_site <= 0:
        raise ValueError("num_site should be > 0.")
    if (num_convergent_site < 0) or (num_convergent_site > num_site):
        raise ValueError("num_convergent_site should satisfy 0 <= value <= num_site.")

    if scenario == "epi_signal":
        degree = np.full(shape=(num_site,), fill_value=-0.5, dtype=float)
        proximity = np.full(shape=(num_site,), fill_value=-0.5, dtype=float)
        if num_convergent_site > 0:
            degree[:num_convergent_site] = 2.5
            proximity[:num_convergent_site] = 2.5
        degree += rng.normal(loc=0.0, scale=0.05, size=num_site)
        proximity += rng.normal(loc=0.0, scale=0.05, size=num_site)
    elif scenario == "null":
        degree = rng.normal(loc=0.0, scale=1.0, size=num_site)
        proximity = rng.normal(loc=0.0, scale=1.0, size=num_site)
    else:
        raise ValueError("Unsupported scenario: {}".format(scenario))

    df = pd.DataFrame(
        {
            "codon_site_alignment": np.arange(1, num_site + 1, dtype=int),
            "epistasis_contact_degree_z": _zscore(degree),
            "epistasis_contact_proximity_z": _zscore(proximity),
        }
    )
    df.to_csv(path, sep="\t", index=False)


def _parse_mode_list(mode_spec):
    modes = [token.strip() for token in str(mode_spec).split(",") if token.strip() != ""]
    if len(modes) == 0:
        raise ValueError("--modes should contain one or more mode names.")
    valid = {"baseline_no_epi", "epi_N_auto", "epi_S_auto", "epi_NS_auto"}
    invalid = sorted([m for m in modes if m not in valid])
    if len(invalid):
        raise ValueError("Unknown mode(s): {}".format(",".join(invalid)))
    dedup = []
    for m in modes:
        if m not in dedup:
            dedup.append(m)
    if "baseline_no_epi" not in dedup:
        dedup = ["baseline_no_epi"] + dedup
    return dedup


def _parse_scenarios(scenario_spec):
    scenarios = [token.strip() for token in str(scenario_spec).split(",") if token.strip() != ""]
    if len(scenarios) == 0:
        raise ValueError("--scenarios should contain one or more scenario names.")
    valid = {"null", "epi_signal"}
    invalid = sorted([s for s in scenarios if s not in valid])
    if len(invalid):
        raise ValueError("Unknown scenario(s): {}".format(",".join(invalid)))
    dedup = []
    for s in scenarios:
        if s not in dedup:
            dedup.append(s)
    return dedup


def _run_timed_command(cmd, cwd, stdout_path, stderr_path, env=None):
    timed_cmd = ["/usr/bin/time", "-l", "-p"] + list(cmd)
    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(
            timed_cmd,
            cwd=str(cwd),
            env=env,
            stdout=out,
            stderr=err,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed (exit={}): {}\nstdout={}\nstderr={}".format(
                proc.returncode,
                shlex.join(timed_cmd),
                stdout_path,
                stderr_path,
            )
        )
    return shlex.join(timed_cmd)


def _resolve_dataset_inputs(dataset_prefix):
    dataset_prefix = Path(dataset_prefix).expanduser().resolve()
    base_name = dataset_prefix.name
    parent = dataset_prefix.parent
    suffixes = {
        "alignment": ".alignment.fa",
        "iqtree": ".alignment.fa.iqtree",
        "rate": ".alignment.fa.rate",
        "state": ".alignment.fa.state",
        "treefile": ".alignment.fa.treefile",
        "tree": ".tree.nwk",
        "foreground": ".foreground.txt",
    }
    paths = {}
    for key, suffix in suffixes.items():
        path = parent / (base_name + suffix)
        if not path.exists():
            raise FileNotFoundError("Dataset input is missing: {}".format(path))
        paths[key] = path
    return paths


def _copy_inputs_to_workdir(inputs, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, src in inputs.items():
        dst = dest_dir / Path(src).name
        shutil.copy2(src, dst)
        out[key] = dst
    return out


def _get_scenario_config(name, args):
    if name == "null":
        return {
            "scenario": name,
            "percent_convergent_site": float(args.null_percent_convergent_site),
            "convergent_amino_acids": str(args.null_convergent_amino_acids),
        }
    if name == "epi_signal":
        return {
            "scenario": name,
            "percent_convergent_site": float(args.signal_percent_convergent_site),
            "convergent_amino_acids": str(args.signal_convergent_amino_acids),
        }
    raise ValueError("Unsupported scenario: {}".format(name))


def _simulate_command(paths, args, scenario_cfg, simulate_seed):
    return [
        str(args.python_exe),
        str(args.csubst_script),
        "simulate",
        "--alignment_file",
        str(paths["alignment"]),
        "--rooted_tree_file",
        str(paths["tree"]),
        "--foreground",
        str(paths["foreground"]),
        "--iqtree_iqtree",
        str(paths["iqtree"]),
        "--iqtree_rate",
        str(paths["rate"]),
        "--iqtree_state",
        str(paths["state"]),
        "--iqtree_treefile",
        str(paths["treefile"]),
        "--threads",
        str(int(args.threads)),
        "--num_simulated_site",
        str(int(args.num_simulated_site)),
        "--percent_convergent_site",
        str(float(scenario_cfg["percent_convergent_site"])),
        "--convergent_amino_acids",
        str(scenario_cfg["convergent_amino_acids"]),
        "--percent_biased_sub",
        str(float(args.percent_biased_sub)),
        "--fg_stem_only",
        "no",
        "--simulate_seed",
        str(int(simulate_seed)),
    ]


def _analyze_command(paths, args, mode, degree_file):
    base = [
        str(args.python_exe),
        str(args.csubst_script),
        "analyze",
        "--alignment_file",
        str(paths["simulate_alignment"]),
        "--rooted_tree_file",
        str(paths["tree"]),
        "--foreground",
        str(paths["foreground"]),
        "--threads",
        str(int(args.threads)),
        "--iqtree_exe",
        str(args.iqtree_exe),
        "--omegaC_method",
        "modelfree",
        "--fg_stem_only",
        "no",
    ]
    if mode == "baseline_no_epi":
        return base
    if mode == "epi_N_auto":
        return base + [
            "--epistasis_apply_to",
            "N",
            "--epistasis_beta",
            "auto",
            "--epistasis_degree_file",
            str(degree_file),
        ]
    if mode == "epi_S_auto":
        return base + [
            "--epistasis_apply_to",
            "S",
            "--epistasis_beta",
            "auto",
            "--epistasis_degree_file",
            str(degree_file),
        ]
    if mode == "epi_NS_auto":
        return base + [
            "--epistasis_apply_to",
            "NS",
            "--epistasis_beta",
            "auto",
            "--epistasis_degree_file",
            str(degree_file),
        ]
    raise ValueError("Unsupported mode: {}".format(mode))


def _git_head(path):
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return ""


def main():
    repo_root = Path(__file__).resolve().parents[1]
    psr = argparse.ArgumentParser(
        description="Run reproducible epistasis simulation evaluation with runtime + peak RAM tracking."
    )
    psr.add_argument(
        "--dataset-prefix",
        default=str(repo_root / "csubst" / "dataset" / "PGK"),
        help="Dataset prefix path (example: /path/to/PGK for PGK.alignment.fa, PGK.tree.nwk, ...).",
    )
    psr.add_argument(
        "--outdir",
        default="/tmp/csubst_epistasis_eval_{}".format(time.strftime("%Y%m%d_%H%M%S")),
        help="Output root directory (recommended under /tmp).",
    )
    psr.add_argument("--allow-non-tmp-outdir", action="store_true", help="Allow outdir outside /tmp.")
    psr.add_argument("--scenarios", default="null,epi_signal", help="Comma-separated scenarios: null,epi_signal")
    psr.add_argument("--modes", default="baseline_no_epi,epi_N_auto,epi_S_auto,epi_NS_auto")
    psr.add_argument("--replicates", type=int, default=3)
    psr.add_argument("--base-seed", type=int, default=20260226)
    psr.add_argument("--threads", type=int, default=1)
    psr.add_argument("--num-simulated-site", type=int, default=180)
    psr.add_argument("--percent-biased-sub", type=float, default=90.0)
    psr.add_argument("--null-percent-convergent-site", type=float, default=0.0)
    psr.add_argument("--signal-percent-convergent-site", type=float, default=30.0)
    psr.add_argument("--null-convergent-amino-acids", default="random0")
    psr.add_argument("--signal-convergent-amino-acids", default="KR")
    psr.add_argument("--score-col", default="omegaCany2spe")
    psr.add_argument("--jaccard-threshold", type=float, default=5.0)
    psr.add_argument("--python-exe", default=sys.executable)
    psr.add_argument("--csubst-script", default=str(repo_root / "csubst" / "csubst"))
    psr.add_argument("--iqtree-exe", default="iqtree")
    args = psr.parse_args()

    if args.replicates <= 0:
        raise ValueError("--replicates should be > 0.")
    if args.num_simulated_site <= 0:
        raise ValueError("--num-simulated-site should be > 0.")
    if args.threads <= 0:
        raise ValueError("--threads should be > 0.")
    if (args.percent_biased_sub < 0) or (args.percent_biased_sub >= 100):
        raise ValueError("--percent-biased-sub should satisfy 0 <= value < 100.")
    if (args.null_percent_convergent_site < 0) or (args.null_percent_convergent_site > 100):
        raise ValueError("--null-percent-convergent-site should satisfy 0 <= value <= 100.")
    if (args.signal_percent_convergent_site < 0) or (args.signal_percent_convergent_site > 100):
        raise ValueError("--signal-percent-convergent-site should satisfy 0 <= value <= 100.")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir_txt = str(outdir)
    is_tmp_path = (
        outdir_txt == "/tmp"
        or outdir_txt.startswith("/tmp/")
        or outdir_txt == "/private/tmp"
        or outdir_txt.startswith("/private/tmp/")
    )
    if (not args.allow_non_tmp_outdir) and (not is_tmp_path):
        raise ValueError("Outdir should be under /tmp unless --allow-non-tmp-outdir is provided.")
    outdir.mkdir(parents=True, exist_ok=True)

    args.python_exe = Path(args.python_exe).expanduser().resolve()
    args.csubst_script = Path(args.csubst_script).expanduser().resolve()
    if not args.python_exe.exists():
        raise FileNotFoundError("Python executable was not found: {}".format(args.python_exe))
    if not args.csubst_script.exists():
        raise FileNotFoundError("csubst script was not found: {}".format(args.csubst_script))

    scenarios = _parse_scenarios(args.scenarios)
    modes = _parse_mode_list(args.modes)
    dataset_inputs = _resolve_dataset_inputs(args.dataset_prefix)

    summary_rows = []
    manifest_rows = []

    for sc_idx, scenario in enumerate(scenarios):
        scenario_cfg = _get_scenario_config(name=scenario, args=args)
        for rep in range(1, int(args.replicates) + 1):
            rep_seed = int(args.base_seed) + sc_idx * 10000 + rep
            run_root = outdir / scenario / ("rep_{:03d}".format(rep))
            sim_dir = run_root / "simulate"
            run_inputs = _copy_inputs_to_workdir(inputs=dataset_inputs, dest_dir=sim_dir)

            sim_stdout = sim_dir / "run.stdout.log"
            sim_stderr = sim_dir / "run.stderr.log"
            simulate_seed = rep_seed * 10 + 1
            simulate_cmd = _simulate_command(
                paths=run_inputs,
                args=args,
                scenario_cfg=scenario_cfg,
                simulate_seed=simulate_seed,
            )
            simulate_cmd_txt = _run_timed_command(
                cmd=simulate_cmd,
                cwd=sim_dir,
                stdout_path=sim_stdout,
                stderr_path=sim_stderr,
                env=os.environ.copy(),
            )
            simulate_alignment = sim_dir / "simulate.fa"
            if not simulate_alignment.exists():
                raise FileNotFoundError("simulate.fa was not generated: {}".format(sim_dir))
            run_inputs["simulate_alignment"] = simulate_alignment

            sim_real, sim_peak = _parse_time_file(sim_stderr)
            sim_peak_mib = sim_peak / (1024.0 * 1024.0) if np.isfinite(sim_peak) else np.nan
            manifest_rows.append(
                {
                    "scenario": scenario,
                    "replicate": rep,
                    "step": "simulate",
                    "mode": "simulate",
                    "workdir": str(sim_dir),
                    "command": simulate_cmd_txt,
                    "stdout_log": str(sim_stdout),
                    "stderr_log": str(sim_stderr),
                    "real_sec": sim_real,
                    "peak_bytes": sim_peak,
                    "peak_mib": sim_peak_mib,
                    "cb2_sha256": "",
                    "cbstats_sha256": "",
                    "simulate_fa_sha256": _sha256(simulate_alignment),
                    "degree_file_sha256": "",
                }
            )

            num_convergent_site = int(
                int(args.num_simulated_site) * float(scenario_cfg["percent_convergent_site"]) / 100.0
            )
            degree_file = sim_dir / "epistasis_degree.tsv"
            rng = np.random.default_rng(rep_seed)
            _write_epistasis_degree_table(
                path=degree_file,
                num_site=int(args.num_simulated_site),
                num_convergent_site=num_convergent_site,
                scenario=scenario,
                rng=rng,
            )
            degree_sha = _sha256(degree_file)

            baseline_df = None
            baseline_metrics = None
            for mode in modes:
                mode_dir = run_root / mode
                mode_dir.mkdir(parents=True, exist_ok=True)
                mode_stdout = mode_dir / "run.stdout.log"
                mode_stderr = mode_dir / "run.stderr.log"
                analyze_cmd = _analyze_command(paths=run_inputs, args=args, mode=mode, degree_file=degree_file)
                analyze_cmd_txt = _run_timed_command(
                    cmd=analyze_cmd,
                    cwd=mode_dir,
                    stdout_path=mode_stdout,
                    stderr_path=mode_stderr,
                    env=os.environ.copy(),
                )

                cb_path = mode_dir / "csubst_cb_2.tsv"
                stats_path = mode_dir / "csubst_cb_stats.tsv"
                if not cb_path.exists():
                    raise FileNotFoundError("Missing {}".format(cb_path))
                df = pd.read_csv(cb_path, sep="\t")

                detect = _calc_fg_detection_metrics(df=df, score_col=args.score_col)
                real, peak = _parse_time_file(mode_stderr)
                peak_mib = peak / (1024.0 * 1024.0) if np.isfinite(peak) else np.nan

                row = {
                    "scenario": scenario,
                    "replicate": rep,
                    "seed": rep_seed,
                    "mode": mode,
                    "num_simulated_site": int(args.num_simulated_site),
                    "num_convergent_site": int(num_convergent_site),
                    "score_col": args.score_col,
                    "n_total_valid": detect["n_total_valid"],
                    "n_pos": detect["n_pos"],
                    "n_neg": detect["n_neg"],
                    "auroc_fg": detect["auroc"],
                    "average_precision_fg": detect["average_precision"],
                    "precision_at_k_fg": detect["precision_at_k"],
                    "runtime_sec": real,
                    "peak_ram_mib": peak_mib,
                    "pearson_r_log10_to_baseline": np.nan,
                    "jaccard_to_baseline": np.nan,
                    "jaccard_intersection": np.nan,
                    "jaccard_union": np.nan,
                    "auroc_delta_vs_baseline": np.nan,
                    "average_precision_delta_vs_baseline": np.nan,
                    "precision_at_k_delta_vs_baseline": np.nan,
                    "cb2_sha256": _sha256(cb_path),
                    "cbstats_sha256": _sha256(stats_path) if stats_path.exists() else "",
                    "workdir": str(mode_dir),
                    "simulate_workdir": str(sim_dir),
                    "degree_file": str(degree_file),
                    "simulate_alignment": str(simulate_alignment),
                }

                if mode == "baseline_no_epi":
                    baseline_df = df
                    baseline_metrics = detect
                elif baseline_df is not None:
                    sim = _calc_similarity_to_baseline(
                        df_base=baseline_df,
                        df_cur=df,
                        score_col=args.score_col,
                        jaccard_threshold=float(args.jaccard_threshold),
                    )
                    row["pearson_r_log10_to_baseline"] = sim["pearson_r_log10"]
                    row["jaccard_to_baseline"] = sim["jaccard"]
                    row["jaccard_intersection"] = sim["intersection"]
                    row["jaccard_union"] = sim["union"]
                    if baseline_metrics is not None:
                        row["auroc_delta_vs_baseline"] = row["auroc_fg"] - baseline_metrics["auroc"]
                        row["average_precision_delta_vs_baseline"] = (
                            row["average_precision_fg"] - baseline_metrics["average_precision"]
                        )
                        row["precision_at_k_delta_vs_baseline"] = (
                            row["precision_at_k_fg"] - baseline_metrics["precision_at_k"]
                        )

                summary_rows.append(row)
                manifest_rows.append(
                    {
                        "scenario": scenario,
                        "replicate": rep,
                        "step": "analyze",
                        "mode": mode,
                        "workdir": str(mode_dir),
                        "command": analyze_cmd_txt,
                        "stdout_log": str(mode_stdout),
                        "stderr_log": str(mode_stderr),
                        "real_sec": real,
                        "peak_bytes": peak,
                        "peak_mib": peak_mib,
                        "cb2_sha256": row["cb2_sha256"],
                        "cbstats_sha256": row["cbstats_sha256"],
                        "simulate_fa_sha256": _sha256(simulate_alignment),
                        "degree_file_sha256": degree_sha,
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["scenario", "replicate", "mode"]).reset_index(drop=True)
    summary_path = outdir / "scenario_mode_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df = manifest_df.sort_values(["scenario", "replicate", "step", "mode"]).reset_index(drop=True)
    manifest_path = outdir / "run_manifest.tsv"
    manifest_df.to_csv(manifest_path, sep="\t", index=False)

    agg = (
        summary_df.groupby(["scenario", "mode"], as_index=False)
        .agg(
            n_rep=("replicate", "count"),
            auroc_fg_mean=("auroc_fg", "mean"),
            auroc_fg_sd=("auroc_fg", "std"),
            average_precision_fg_mean=("average_precision_fg", "mean"),
            average_precision_fg_sd=("average_precision_fg", "std"),
            precision_at_k_fg_mean=("precision_at_k_fg", "mean"),
            precision_at_k_fg_sd=("precision_at_k_fg", "std"),
            runtime_sec_mean=("runtime_sec", "mean"),
            runtime_sec_sd=("runtime_sec", "std"),
            peak_ram_mib_mean=("peak_ram_mib", "mean"),
            peak_ram_mib_sd=("peak_ram_mib", "std"),
            pearson_r_log10_to_baseline_mean=("pearson_r_log10_to_baseline", "mean"),
            jaccard_to_baseline_mean=("jaccard_to_baseline", "mean"),
            auroc_delta_vs_baseline_mean=("auroc_delta_vs_baseline", "mean"),
            average_precision_delta_vs_baseline_mean=("average_precision_delta_vs_baseline", "mean"),
            precision_at_k_delta_vs_baseline_mean=("precision_at_k_delta_vs_baseline", "mean"),
        )
        .sort_values(["scenario", "mode"])\
        .reset_index(drop=True)
    )
    agg_path = outdir / "scenario_mode_aggregate.tsv"
    agg.to_csv(agg_path, sep="\t", index=False)

    metadata = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(repo_root),
        "commit": _git_head(repo_root),
        "dataset_prefix": str(Path(args.dataset_prefix).expanduser().resolve()),
        "outdir": str(outdir),
        "scenarios": scenarios,
        "modes": modes,
        "replicates": int(args.replicates),
        "base_seed": int(args.base_seed),
        "num_simulated_site": int(args.num_simulated_site),
        "percent_biased_sub": float(args.percent_biased_sub),
        "null_percent_convergent_site": float(args.null_percent_convergent_site),
        "signal_percent_convergent_site": float(args.signal_percent_convergent_site),
        "score_col": str(args.score_col),
        "jaccard_threshold": float(args.jaccard_threshold),
        "python_exe": str(args.python_exe),
        "csubst_script": str(args.csubst_script),
        "iqtree_exe": str(args.iqtree_exe),
        "outputs": {
            "summary": str(summary_path),
            "aggregate": str(agg_path),
            "manifest": str(manifest_path),
        },
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("Wrote: {}".format(summary_path), flush=True)
    print("Wrote: {}".format(agg_path), flush=True)
    print("Wrote: {}".format(manifest_path), flush=True)
    print("Wrote: {}".format(outdir / "metadata.json"), flush=True)


if __name__ == "__main__":
    main()
