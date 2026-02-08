#!/usr/bin/env python3

import argparse
import itertools
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas

# Ensure `csubst` is importable when this script is run directly in CI.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from csubst import ete
from csubst import tree


EXPECTED = {
    "PGK": {
        "branch_id_1": 23,
        "branch_id_2": 51,
        "omegaCany2spe": 1.990354,
        "convergent": 5,
        "divergent": 7,
        "blank": 405,
    },
    "PEPC": {
        "branch_id_1": 9,
        "branch_id_2": 108,
        "omegaCany2spe": 0.049655,
        "convergent": 0,
        "divergent": 2,
        "blank": 969,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PGK/PEPC parity checks and capture runtime + peak RAM."
    )
    parser.add_argument(
        "--output",
        default="parity_metrics.tsv",
        help="Output TSV path for collected metrics.",
    )
    parser.add_argument(
        "--workdir",
        default="parity_runs",
        help="Temporary working directory for generated analysis files.",
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
    # GNU time -v (Linux CI expected path)
    m_elapsed = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)",
        stderr_text,
    )
    m_rss = re.search(
        r"Maximum resident set size \(kbytes\):\s*([0-9]+)",
        stderr_text,
    )
    if (m_elapsed is not None) and (m_rss is not None):
        return parse_elapsed_seconds(m_elapsed.group(1)), int(m_rss.group(1))

    # POSIX time -p fallback (elapsed only)
    m_elapsed_p = re.search(r"^real\s+([0-9.]+)$", stderr_text, flags=re.MULTILINE)
    if m_elapsed_p is not None:
        return float(m_elapsed_p.group(1)), -1

    raise RuntimeError("Failed to parse timing metrics from /usr/bin/time output.")


def run_timed_command(cmd, cwd, label, env=None):
    if sys.platform.startswith("linux"):
        wrapper = ["/usr/bin/time", "-v"]
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
    log_path = Path(cwd) / (label + ".log.txt")
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed ({}): {}\nSee {}".format(
                proc.returncode, " ".join(cmd), str(log_path)
            )
        )
    elapsed_sec, max_rss_kb = parse_time_metrics(proc.stderr)
    return elapsed_sec, max_rss_kb


def ensure_precomputed_iqtree_outputs(repo_root, dataset_name):
    dataset_dir = repo_root / "csubst" / "dataset"
    required = [
        dataset_dir / f"{dataset_name}.alignment.fa",
        dataset_dir / f"{dataset_name}.tree.nwk",
        dataset_dir / f"{dataset_name}.alignment.fa.treefile",
        dataset_dir / f"{dataset_name}.alignment.fa.state",
        dataset_dir / f"{dataset_name}.alignment.fa.rate",
        dataset_dir / f"{dataset_name}.alignment.fa.iqtree",
        dataset_dir / f"{dataset_name}.alignment.fa.log",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if len(missing) > 0:
        raise FileNotFoundError(
            "Missing precomputed IQ-TREE outputs for {}:\n{}".format(
                dataset_name, "\n".join(missing)
            )
        )
    return required


def write_pepc_foreground_file(repo_root, run_dir):
    tree_file = repo_root / "csubst" / "dataset" / "PEPC.tree.nwk"
    tr = ete.PhyloNode(str(tree_file), format=1)
    tr = tree.add_numerical_node_labels(tr)
    leaves = list(ete.iter_leaves(tr))
    best = None
    for leaf_a, leaf_b in itertools.combinations(leaves, 2):
        dist = ete.get_distance(leaf_a, leaf_b, topology_only=False)
        key = (dist, leaf_a.name, leaf_b.name)
        if (best is None) or (key > best):
            best = key
    fg_path = Path(run_dir) / "PEPC.parity.foreground.txt"
    with fg_path.open("w", encoding="utf-8") as handle:
        # Foreground file names are regex patterns; anchor exact tip-name matches.
        handle.write("1\t^{}$\n".format(re.escape(best[1])))
        handle.write("2\t^{}$\n".format(re.escape(best[2])))
    return fg_path


def run_dataset(repo_root, run_root, dataset_name):
    run_dir = Path(run_root) / dataset_name
    run_dir.mkdir(parents=True, exist_ok=True)
    python_exe = sys.executable
    csubst_exe = repo_root / "csubst" / "csubst"
    dataset_dir = repo_root / "csubst" / "dataset"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + (
        os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else ""
    )
    ensure_precomputed_iqtree_outputs(repo_root=repo_root, dataset_name=dataset_name)

    if dataset_name == "PGK":
        foreground = dataset_dir / "PGK.foreground.txt"
    elif dataset_name == "PEPC":
        foreground = write_pepc_foreground_file(repo_root=repo_root, run_dir=run_dir)
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset_name))

    analyze_cmd = [
        python_exe,
        str(csubst_exe),
        "analyze",
        "--alignment_file",
        str(dataset_dir / f"{dataset_name}.alignment.fa"),
        "--rooted_tree_file",
        str(dataset_dir / f"{dataset_name}.tree.nwk"),
        "--foreground",
        str(foreground),
        "--exhaustive_until",
        "1",
        "--threads",
        "1",
        "--float_digit",
        "6",
        "--iqtree_treefile",
        str(dataset_dir / f"{dataset_name}.alignment.fa.treefile"),
        "--iqtree_state",
        str(dataset_dir / f"{dataset_name}.alignment.fa.state"),
        "--iqtree_rate",
        str(dataset_dir / f"{dataset_name}.alignment.fa.rate"),
        "--iqtree_iqtree",
        str(dataset_dir / f"{dataset_name}.alignment.fa.iqtree"),
        "--iqtree_log",
        str(dataset_dir / f"{dataset_name}.alignment.fa.log"),
    ]
    analyze_elapsed_sec, analyze_max_rss_kb = run_timed_command(
        cmd=analyze_cmd,
        cwd=run_dir,
        label=f"{dataset_name}.analyze",
        env=env,
    )

    cb_path = run_dir / "csubst_cb_2.tsv"
    if not cb_path.exists():
        raise FileNotFoundError("Missing analyze output: {}".format(str(cb_path)))
    cb = pandas.read_csv(cb_path, sep="\t")
    if cb.shape[0] < 1:
        raise RuntimeError("No rows in {}".format(str(cb_path)))
    cb = cb.sort_values(["branch_id_1", "branch_id_2"]).reset_index(drop=True)
    row = cb.iloc[0]
    branch_id_1 = int(row["branch_id_1"])
    branch_id_2 = int(row["branch_id_2"])
    omega = float(row["omegaCany2spe"])

    branch_id_text = "{},{}".format(branch_id_1, branch_id_2)
    site_cmd = [
        python_exe,
        str(csubst_exe),
        "site",
        "--alignment_file",
        str(dataset_dir / f"{dataset_name}.alignment.fa"),
        "--rooted_tree_file",
        str(dataset_dir / f"{dataset_name}.tree.nwk"),
        "--branch_id",
        branch_id_text,
        "--threads",
        "1",
        "--float_digit",
        "6",
        "--site_state_plot",
        "no",
        "--tree_site_plot",
        "yes",
        "--tree_site_plot_format",
        "pdf",
        "--tree_site_plot_min_prob",
        "0.5",
        "--tree_site_plot_max_sites",
        "60",
        "--iqtree_treefile",
        str(dataset_dir / f"{dataset_name}.alignment.fa.treefile"),
        "--iqtree_state",
        str(dataset_dir / f"{dataset_name}.alignment.fa.state"),
        "--iqtree_rate",
        str(dataset_dir / f"{dataset_name}.alignment.fa.rate"),
        "--iqtree_iqtree",
        str(dataset_dir / f"{dataset_name}.alignment.fa.iqtree"),
        "--iqtree_log",
        str(dataset_dir / f"{dataset_name}.alignment.fa.log"),
    ]
    site_elapsed_sec, site_max_rss_kb = run_timed_command(
        cmd=site_cmd,
        cwd=run_dir,
        label=f"{dataset_name}.site",
        env=env,
    )

    tree_site_path = run_dir / f"csubst_site.branch_id{branch_id_text}" / "csubst_site.tree_site.tsv"
    if not tree_site_path.exists():
        raise FileNotFoundError("Missing site output: {}".format(str(tree_site_path)))
    tree_site = pandas.read_csv(tree_site_path, sep="\t")
    counts = tree_site["tree_site_category"].value_counts().to_dict()
    convergent = int(counts.get("convergent", 0))
    divergent = int(counts.get("divergent", 0))
    blank = int(counts.get("blank", 0))

    return {
        "dataset": dataset_name,
        "branch_id_1": branch_id_1,
        "branch_id_2": branch_id_2,
        "omegaCany2spe": omega,
        "convergent": convergent,
        "divergent": divergent,
        "blank": blank,
        "analyze_elapsed_sec": analyze_elapsed_sec,
        "analyze_max_rss_kb": analyze_max_rss_kb,
        "site_elapsed_sec": site_elapsed_sec,
        "site_max_rss_kb": site_max_rss_kb,
    }


def validate_metrics(rows):
    errors = []
    for row in rows:
        dataset = row["dataset"]
        expected = EXPECTED[dataset]
        for key in ["branch_id_1", "branch_id_2", "convergent", "divergent", "blank"]:
            if int(row[key]) != int(expected[key]):
                errors.append(
                    "{}: expected {}={}, got {}".format(
                        dataset, key, expected[key], row[key]
                    )
                )
        if not math.isfinite(float(row["omegaCany2spe"])):
            errors.append("{}: omegaCany2spe is non-finite".format(dataset))
        elif abs(float(row["omegaCany2spe"]) - float(expected["omegaCany2spe"])) > 1e-6:
            errors.append(
                "{}: expected omegaCany2spe={}, got {}".format(
                    dataset, expected["omegaCany2spe"], row["omegaCany2spe"]
                )
            )
    if len(errors) > 0:
        raise RuntimeError("Parity check failed:\n- " + "\n- ".join(errors))


def main():
    args = parse_args()
    repo_root = REPO_ROOT
    run_root = Path(args.workdir).resolve()
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset_name in ["PGK", "PEPC"]:
        rows.append(
            run_dataset(
                repo_root=repo_root,
                run_root=run_root,
                dataset_name=dataset_name,
            )
        )

    validate_metrics(rows)

    out_df = pandas.DataFrame(rows)
    out_df = out_df.sort_values(by=["dataset"]).reset_index(drop=True)
    out_path = Path(args.output).resolve()
    out_df.to_csv(out_path, sep="\t", index=False)
    print(out_df.to_string(index=False))
    print("Wrote parity metrics: {}".format(str(out_path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
