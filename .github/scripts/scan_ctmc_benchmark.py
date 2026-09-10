#!/usr/bin/env python3
"""Reproducible PEPC scan comparison on the same uniform ECMK07+F ASR fit.

Fit once before timing. Worker processes isolate peak RSS, and one warmup per
configuration is excluded. This compares distinct observation models; event differences are reported.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def portable_record(value):
    """Store repository paths relative to its root, without machine home paths."""
    if isinstance(value, str):
        return value.replace(str(ROOT) + os.sep, "").replace(str(Path.home()) + os.sep, "${HOME}/")
    if isinstance(value, list):
        return [portable_record(item) for item in value]
    if isinstance(value, dict):
        return {portable_record(key): portable_record(item) for key, item in value.items()}
    return value


def write_record(path, record):
    path.write_text(json.dumps(portable_record(record), indent=2) + "\n")


def command(args, mode, outdir):
    data = ROOT / "csubst/dataset"
    result = ["csubst", "scan", "--alignment_file", str(data / "PEPC.alignment.fa"),
              "--rooted_tree_file", str(data / "PEPC.tree.nwk"),
              "--foreground", str(ROOT / "reports/csubst_scan_pepc_20260625/PEPC.foreground.independent.txt"),
              "--iqtree_model", "ECMK07+F", "--iqtree_outdir", str(args.fit_dir),
              "--scan_match", "any2spe", "--scan_rate_exposure", "q_weighted" if mode == "legacy_default" else "endpoint",
              "--scan_rate_length", "n_rescaled" if mode == "legacy_default" else "raw",
              "--scan_pvalue_calibration", ("full_scan" if mode == "legacy_default" else "parametric") if args.calibration == "pipeline" else args.calibration, "--scan_n_permutations", str(args.niter),
              "--scan_permutation_seed", "7001", "--min_clade_bin_count", "2",
              "--scan_site_plot", "no", "--threads", "1", "--blas_threads", "1",
              "--float_digit", "10", "--outdir", str(outdir)]
    result += ["--scan_observation", "marginal" if mode == "legacy_default" else mode]
    return result


def worker(args):
    from csubst import cli, substitution_scan, scan_ctmc
    args.outdir.mkdir(parents=True, exist_ok=True)
    record = {"mode": args.worker, "command": command(args, args.worker, args.outdir)}
    prepare = scan_ctmc.prepare
    def measured_prepare(g):
        start = time.perf_counter()
        result = prepare(g)
        record["ctmc_inference_seconds"] = record.get("ctmc_inference_seconds", 0.) + time.perf_counter() - start
        record["ctmc_inference_calls"] = record.get("ctmc_inference_calls", 0) + 1
        return result
    scan_ctmc.prepare = measured_prepare
    original = substitution_scan.scan_substitutions

    def measured(g, ON_tensor, rate_ON_tensor=None):
        record.update(nodes=int(g["state_cdn"].shape[0]), sites=int(g["state_cdn"].shape[1]))
        start = time.perf_counter()
        result = original(g, ON_tensor, rate_ON_tensor)
        record["scan_seconds"] = time.perf_counter() - start
        return result

    substitution_scan.scan_substitutions = measured
    sys.argv = record["command"]
    start = time.perf_counter()
    cli.main()
    record["cli_seconds"] = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record["peak_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
    write_record(args.outdir / "measurement.json", record)


def compare_tables(tables):
    keys = ["trait", "target_class", "scan_match", "site", "from_state_ids", "to_state_ids"]
    keep = ["target_event_count", "other_event_count", "support_unit_count", "candidate_event_mass_sum"]
    indexed = {mode: table.set_index(keys).sort_index() for mode, table in tables.items()}
    reference = indexed["legacy_default"]
    result = {}
    for mode, frame in indexed.items():
        common = reference.index.intersection(frame.index)
        a = reference.loc[common].p_rate_enrichment_asymptotic.rank()
        b = frame.loc[common].p_rate_enrichment_asymptotic.rank()
        result[mode] = dict(candidates=len(frame), common_candidates=len(common),
                            max_event_difference=float((reference.loc[common, keep] - frame.loc[common, keep]).abs().max().max()),
                            finite_p=int(np.isfinite(frame.p_rate_enrichment_asymptotic).sum()),
                            p_rank_spearman_vs_legacy=float(a.corr(b)),
                            nominal_p_below_005=int((frame.p_rate_enrichment_asymptotic < .05).sum()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--calibration", choices=["none", "full_scan", "pipeline"], default="none")
    parser.add_argument("--niter", type=int, default=20)
    parser.add_argument("--worker", choices=["legacy_default", "joint", "bridge"])
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if not args.fit_dir.exists() or not list(args.fit_dir.glob("*.state")):
        parser.error("--fit-dir must contain the precomputed uniform ECMK07+F ASR fit")
    args.outdir.mkdir(parents=True, exist_ok=True)
    inputs = list((ROOT / "csubst/dataset").glob("PEPC.alignment.fa"))
    inputs += [ROOT / "csubst/dataset/PEPC.tree.nwk", ROOT / "reports/csubst_scan_pepc_20260625/PEPC.foreground.independent.txt"]
    inputs += [p for p in args.fit_dir.iterdir() if p.suffix in (".state", ".treefile", ".rate", ".iqtree", ".log")]
    thread_env = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
    record = dict(platform=platform.platform(), python=platform.python_version(),
                  numpy=np.__version__, pandas=pd.__version__, calibration=args.calibration,
                  niter=args.niter, repeats=args.repeats, thread_env=thread_env,
                  command_cwd="repository root (repository paths are relative)",
                  input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  runs=[], summary={}, comparison={})
    modes = ["legacy_default", "joint", "bridge"]
    tables = {}
    for repeat in range(-1, args.repeats):
        # Rotate order to avoid always measuring one implementation first.
        for mode in modes[repeat % 3:] + modes[:repeat % 3]:
            out = args.outdir / (mode + "_" + ("warmup" if repeat < 0 else str(repeat)))
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", mode,
                   "--fit-dir", str(args.fit_dir), "--outdir", str(out),
                   "--calibration", args.calibration, "--niter", str(args.niter)]
            with (args.outdir / (out.name + ".log")).open("w") as log:
                start = time.perf_counter()
                subprocess.run(cmd, stdout=log, stderr=log, check=True, env=dict(os.environ, **thread_env))
                elapsed = time.perf_counter() - start
            measurement = json.loads((out / "measurement.json").read_text())
            measurement.update(process_seconds=elapsed, repeat=repeat)
            if repeat >= 0:
                record["runs"].append(measurement)
                table = pd.read_csv(out / "csubst_scan.tsv", sep="\t")
                if mode in tables:
                    pd.testing.assert_frame_equal(tables[mode], table, check_exact=True)
                tables[mode] = table
            write_record(args.outdir / "performance.json", record)
    for mode in modes:
        rows = [r for r in record["runs"] if r["mode"] == mode]
        record["summary"][mode] = {
            key: dict(median=float(np.median([r[key] for r in rows])),
                      minimum=float(min(r[key] for r in rows)), maximum=float(max(r[key] for r in rows)))
            for key in ("process_seconds", "cli_seconds", "scan_seconds", "peak_rss_bytes")}
        tables[mode].to_csv(args.outdir / (mode + ".tsv"), sep="\t", index=False)
    record["comparison"] = compare_tables(tables)
    write_record(args.outdir / "performance.json", record)
    print(json.dumps(dict(summary=record["summary"], comparison=record["comparison"]), indent=2))


if __name__ == "__main__":
    main()
