"""Compare fixed native source snapshots in fresh, sequential CLI processes.

Requires macOS /usr/bin/time -l, the same Python/dependencies for both snapshots,
and separately compiled native extensions in both source roots. One warmup and
three measured repetitions alternate before/after order. IQ-TREE fitting,
network downloads, plots and default 1000-draw scan calibration are excluded.
Outputs and full logs stay in --workdir; --result is compact evidence.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "pgk_longtail_empirical": (
        "PGK",
        "search",
        "matched",
        {
            "expectation_method": "urn",
            "output_stat": "any2spe",
            "omega_pvalue_null_model": "poisson",
            "calibrate_longtail": "yes",
            "longtail_method": "empirical",
        },
    ),
    "pgk_longtail_independent": (
        "PGK",
        "search",
        "matched",
        {
            "expectation_method": "urn",
            "output_stat": "any2spe",
            "omega_pvalue_null_model": "poisson",
            "calibrate_longtail": "yes",
            "longtail_method": "independent_null",
        },
    ),
    "pgk_matched": ("PGK", "search", "matched", {}),
    "pepc_matched": ("PEPC", "search", "matched", {}),
    "pgk_defaults": ("PGK", "search", "defaults", {}),
    "pepc_defaults": ("PEPC", "search", "defaults", {}),
    "pepc4_matched": ("PEPCx4", "search", "matched", {}),
    "pepc4_defaults": ("PEPCx4", "search", "defaults", {}),
    "pepc_full": (
        "PEPC",
        "search",
        "defaults",
        {
            "output_stat": "any2any,spe2any,any2spe,spe2spe",
            "drop_invariant_tip_sites": "no",
            "calibrate_longtail": "no",
        },
    ),
    "pepc_full_matched": (
        "PEPC",
        "search",
        "matched",
        {"output_stat": "any2any,spe2any,any2spe,spe2spe"},
    ),
    "pepc_urn": (
        "PEPC",
        "search",
        "matched",
        {"expectation_method": "urn", "output_stat": "any2spe"},
    ),
    "pepc_arity6": (
        "PEPC",
        "search",
        "defaults",
        {
            "max_arity": "6",
            "calibrate_longtail": "no",
            "drop_invariant_tip_sites": "no",
        },
    ),
    "pgk_scan_matched": ("PGK", "scan", "matched", {}),
    "pepc_scan_matched": ("PEPC", "scan", "matched", {}),
    "pgk_scan_defaults": ("PGK", "scan", "defaults", {}),
    "pepc_scan_defaults": ("PEPC", "scan", "defaults", {}),
    "pgk_scan_calibration20": (
        "PGK",
        "scan",
        "defaults",
        {"scan_pvalue_calibration": "full_scan", "scan_n_permutations": "20"},
    ),
    "pepc_sites": ("PEPC", "sites", "defaults", {}),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def make_long_fixture(data):
    original = data / "PEPC.alignment.fa"
    target = data / "PEPCx4.alignment.fa"
    if target.exists():
        return
    sequences = {}
    name = None
    for line in original.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:]
            sequences[name] = ""
        else:
            sequences[name] += line.strip()
    sites = len(next(iter(sequences.values()))) // 3
    target.write_text(
        "".join(">" + name + "\n" + seq * 4 + "\n" for name, seq in sequences.items())
    )
    for suffix in ("treefile", "iqtree", "log"):
        Path(str(target) + "." + suffix).write_bytes(
            Path(str(original) + "." + suffix).read_bytes()
        )
    for suffix in ("tree.nwk", "foreground.txt"):
        (data / ("PEPCx4." + suffix)).write_bytes(
            (data / ("PEPC." + suffix)).read_bytes()
        )
    for suffix in ("state", "rate"):
        source = Path(str(original) + "." + suffix)
        with Path(str(target) + "." + suffix).open("w") as output:
            for repeat in range(4):
                columns = None
                for line in source.open():
                    if line.startswith("#"):
                        if repeat == 0:
                            output.write(line)
                        continue
                    if columns is None:
                        columns = line.rstrip("\n").split("\t")
                        site_index = columns.index("Site")
                        if repeat == 0:
                            output.write(line)
                        continue
                    fields = line.rstrip("\n").split("\t")
                    fields[site_index] = str(int(fields[site_index]) + repeat * sites)
                    output.write("\t".join(fields) + "\n")


def command_for(case, label, source, data, outdir):
    dataset, action, protocol, extra = CASES[case]
    options = {
        "alignment_file": str(data / (dataset + ".alignment.fa")),
        "rooted_tree_file": str(data / (dataset + ".tree.nwk")),
        "outdir": str(outdir),
        "threads": "1",
        "blas_threads": "1",
        "random_seed": "8",
        "float_digit": "12",
    }
    for suffix in ("treefile", "state", "rate", "iqtree", "log"):
        options["iqtree_" + suffix] = str(data / (dataset + ".alignment.fa." + suffix))
    if action == "search":
        options["max_arity"] = "2"
    elif action == "scan":
        options.update(
            foreground=str(data / (dataset + ".foreground.txt")),
            scan_site_plot="no",
            scan_pvalue_calibration="none",
        )
    else:
        options.update(
            branch_id="0,2",
            tree_site_plot="no",
            site_state_plot="no",
            site_summary_plot="no",
        )
    if protocol == "matched":
        options.update(drop_invariant_tip_sites="no")
        if action == "search":
            options.update(calibrate_longtail="no")
        if label == "after":
            options["substitution_posterior"] = "marginal"
        if action == "scan":
            options.update(
                scan_rate_exposure="q_weighted", scan_rate_length="n_rescaled"
            )
    options.update(extra)
    if label == "before":
        options.pop("longtail_method", None)
    command = [sys.executable, "-m", "csubst", action]
    for key, value in options.items():
        command.extend(["--" + key, value])
    return command


def output_summary(outdir):
    frames = {}
    for path in sorted(outdir.rglob("*.tsv")):
        name = str(path.relative_to(outdir))
        if (
            re.fullmatch(r"csubst_cb_\d+\.tsv", name)
            or name == "csubst_scan.tsv"
            or name.endswith(".csubst_sites.tsv")
            or name.endswith("/csubst.tsv")
        ):
            frames[name] = pd.read_csv(path, sep="\t")
    return frames


def compare(reference, candidate):
    result = {}
    for name in sorted(set(reference) | set(candidate)):
        if name not in reference or name not in candidate:
            result[name] = {"table_present_in_both": False}
            continue
        left, right = reference[name], candidate[name]
        ids = [c for c in left if c.startswith("branch_id_") and c in right]
        if ids:
            left, right = left.sort_values(ids), right.sort_values(ids)
        columns = [
            c
            for c in left
            if c in right
            and (
                c.startswith(("OCN", "OCS", "ECN", "ECS", "omegaC", "dNC", "dSC"))
                or c.startswith(("p_rate", "q_rate", "rate_", "num_", "support_"))
            )
        ]
        columns = [
            c
            for c in columns
            if pd.api.types.is_numeric_dtype(left[c])
            and pd.api.types.is_numeric_dtype(right[c])
        ]
        item = {
            "rows_before": len(left),
            "rows_after": len(right),
            "numeric_columns": columns,
            "same_shape": left.shape == right.shape,
        }
        same_ids = len(left) == len(right) and (
            not ids or np.array_equal(left[ids], right[ids])
        )
        item["same_branch_combinations"] = same_ids
        if same_ids and columns:
            a, b = left[columns].to_numpy(float), right[columns].to_numpy(float)
            finite = np.isfinite(a) & np.isfinite(b)
            item["max_absolute_difference"] = (
                float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
            )
            item["same_nan_mask"] = bool(np.array_equal(np.isnan(a), np.isnan(b)))
            item["equivalent_rtol1e-8_atol1e-9"] = bool(
                np.allclose(a, b, rtol=1e-8, atol=1e-9, equal_nan=True)
            )
        result[name] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baseline-label", default="286ac2b")
    parser.add_argument("--candidate-label", default="2ae3a0a plus zero-invariant fix")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    data = args.baseline_root / "csubst/dataset"
    make_long_fixture(data)
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="0",
        CSUBST_STRICT_EXTENSIONS="1",
    )
    env.pop("CSUBST_DISABLE_EXTENSIONS", None)
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "memory_gib": psutil.virtual_memory().total / 2**30,
        "cpu_count": psutil.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "warmups": 1,
        "measured_repeats": args.repeats,
        "blas_threads": 1,
        "threads": 1,
        "baseline": args.baseline_label,
        "candidate": args.candidate_label,
        "source_sha256": {},
        "input_sha256": {},
    }
    for label, source in [
        ("before", args.baseline_root),
        ("after", args.candidate_root),
    ]:
        metadata["source_sha256"][label] = {
            p.name: sha(p)
            for p in sorted((source / "csubst").glob("*"))
            if p.suffix in (".py", ".pyx", ".so")
        }
    for path in sorted(data.glob("*")):
        if path.is_file():
            metadata["input_sha256"][path.name] = sha(path)
    result = {"environment": metadata, "runs": [], "summary": {}, "comparisons": {}}

    def save():
        args.result.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(result, indent=2)
        text = text.replace(str(Path.home()) + "/", "${HOME}/")
        args.result.write_text(text + "\n")

    for case in args.cases:
        frames = {}
        failed = False
        for repeat in range(args.repeats + 1):
            versions = [("before", args.baseline_root), ("after", args.candidate_root)]
            if repeat % 2:
                versions.reverse()
            for label, source in versions:
                out = args.workdir / f"{case}-{label}-{repeat}"
                command = command_for(case, label, source, data, out)
                start_cpu = psutil.cpu_times()
                start = time.perf_counter()
                with (args.workdir / (out.name + ".log")).open("w") as log:
                    proc = subprocess.run(
                        ["/usr/bin/time", "-l", *command],
                        cwd=source,
                        env=dict(env, PYTHONPATH=str(source)),
                        stdout=log,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                wall = time.perf_counter() - start
                end_cpu = psutil.cpu_times()
                row = {
                    "case": case,
                    "version": label,
                    "repeat": repeat,
                    "warmup": repeat == 0,
                    "exit_code": proc.returncode,
                    "seconds": wall,
                    "command": command,
                    "host_active_cpu_cores": sum(
                        getattr(end_cpu, key) - getattr(start_cpu, key)
                        for key in ("user", "nice", "system")
                    )
                    / wall,
                }
                rss = re.search(r"(\d+)\s+maximum resident set size", proc.stderr)
                if rss:
                    row["peak_rss_mib"] = int(rss[1]) / 2**20
                if proc.returncode:
                    row["error"] = proc.stderr[-2000:]
                    failed = True
                result["runs"].append(row)
                print(
                    case,
                    label,
                    repeat,
                    proc.returncode,
                    round(wall, 3),
                    round(row.get("peak_rss_mib", 0), 1),
                    flush=True,
                )
                if repeat == 0 and not failed:
                    frames[label] = output_summary(out)
                save()
            if failed:
                break
            if repeat == 0:
                result["comparisons"][case] = compare(frames["before"], frames["after"])
        if not failed:
            summary = {}
            for label in ("before", "after"):
                rows = [
                    r
                    for r in result["runs"]
                    if r["case"] == case and r["version"] == label and not r["warmup"]
                ]
                summary[label] = {
                    metric: {
                        "median": statistics.median(r[metric] for r in rows),
                        "min": min(r[metric] for r in rows),
                        "max": max(r[metric] for r in rows),
                    }
                    for metric in ("seconds", "peak_rss_mib")
                }
            summary["ratios_after_over_before"] = {
                metric: summary["after"][metric]["median"]
                / summary["before"][metric]["median"]
                for metric in ("seconds", "peak_rss_mib")
            }
            result["summary"][case] = summary
        save()


if __name__ == "__main__":
    main()
