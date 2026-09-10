#!/usr/bin/env python3
"""Benchmark PEPC arity/CPU scaling against marginal inference. Requires psutil and threadpoolctl.

Run sequentially with no other benchmark/test processes. Output JSON contains
commands, source hashes, warmups, every timed run, and summary ranges.
"""

import argparse
import json
import runpy
import sys
import time
import resource
from pathlib import Path


def worker():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--arity", type=int, default=2)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--blas", type=int, default=1)
    p.add_argument("--exhaustive", type=int, default=2)
    p.add_argument("--foreground", action="store_true")
    p.add_argument("--branch-table", action="store_true")
    p.add_argument("--cutoff-stat")
    a = p.parse_args()
    sys.path.insert(0, str(a.root))
    b = runpy.run_path(str(a.root / ".github/scripts/benchmark_endpoints.py"))
    cmd = b["cli_args"]("PEPC", a.mode, 64, a.out)
    for flag, val in [
        ("--max_arity", a.arity),
        ("--threads", a.threads),
        ("--blas_threads", a.blas),
    ]:
        cmd[cmd.index(flag) + 1] = str(val)
    cmd += ["--exhaustive_until", str(a.exhaustive)]
    if a.cutoff_stat is not None:
        cmd += ["--cutoff_stat", a.cutoff_stat]
    if a.branch_table:
        cmd[cmd.index("--b") + 1] = "yes"
    if a.foreground:
        foreground = Path(str(a.out) + ".foreground.tsv")
        names = [
            line.split()[1]
            for line in (a.root / "csubst/dataset/PEPC.foreground.txt")
            .read_text()
            .splitlines()
        ]
        foreground.write_text(
            "name\tbenchmark\n"
            + "".join(f"{name}\t{i + 1}\n" for i, name in enumerate(names))
        )
        cmd += [
            "--foreground",
            str(foreground),
            "--fg_format",
            "2",
            "--cutoff_stat",
            "OCNany2spe,0",
        ]
    from csubst import tsv

    orig = tsv.write_dataframe

    def capture(df, path, **kwargs):
        if Path(path).name.startswith("csubst_cb_"):
            df.to_pickle(str(path) + ".unrounded.pkl")
        return orig(df, path, **kwargs)

    tsv.write_dataframe = capture
    cpu_start = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    sys.argv = cmd
    runpy.run_module("csubst", run_name="__main__")
    result = dict(
        seconds=time.perf_counter() - start,
        peak_parent_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        command=cmd,
    )
    cpu_end = resource.getrusage(resource.RUSAGE_SELF)
    result["cpu_seconds"] = (
        cpu_end.ru_utime + cpu_end.ru_stime - cpu_start.ru_utime - cpu_start.ru_stime
    )
    result["mean_parent_cpu_cores"] = result["cpu_seconds"] / result["seconds"]
    # Inspect the loaded libraries after the measured interval.
    from threadpoolctl import threadpool_info

    result["native_pools"] = threadpool_info()
    Path(str(a.out) + ".json").write_text(json.dumps(result))


def main():
    """Sequential PEPC matrix; no simultaneous benchmark workers."""
    import os
    import subprocess
    import platform
    import psutil
    import numpy as np

    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--baseline-root", type=Path, default=root)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--scenarios",
        nargs="+",
        choices=[
            "pair",
            "pair_b",
            "foreground4",
            "exhaustive3",
            "default4",
            "heuristic6",
            "heuristic6_b",
        ],
        default=["pair", "foreground4", "exhaustive3"],
    )
    p.add_argument("--cpus", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--blas", type=int, default=1)
    a = p.parse_args()
    if a.repeats < 1 or a.blas < 1 or min(a.cpus) < 1:
        p.error("Repeats, CPUs and BLAS threads must be positive.")
    a.workdir.mkdir(parents=True, exist_ok=True)
    definitions = {
        "pair": ["--arity", "2"],
        "pair_b": ["--arity", "2", "--branch-table"],
        "foreground4": ["--arity", "4", "--exhaustive", "1", "--foreground"],
        "exhaustive3": ["--arity", "3", "--exhaustive", "3"],
        "default4": ["--arity", "4"],
        "heuristic6": [
            "--arity",
            "6",
            "--cutoff-stat",
            "OCNany2spe,2.0|omegaCany2spe,5.0",
        ],
        "heuristic6_b": [
            "--arity",
            "6",
            "--cutoff-stat",
            "OCNany2spe,2.0|omegaCany2spe,5.0",
            "--branch-table",
        ],
    }
    helper = runpy.run_path(
        str(root / ".github/scripts/benchmark_endpoint_optimization.py")
    )
    result = {
        "environment": {
            "platform": platform.platform(),
            "cpu_model": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            if sys.platform == "darwin"
            else platform.processor(),
            "memory_gib": psutil.virtual_memory().total / 1024**3,
            "source_commit": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "python": sys.version,
            "logical_cpus": psutil.cpu_count(),
            "repeats": a.repeats,
            "warmups": 1,
            "blas_threads": a.blas,
            "rss_sampling_seconds": 0.05,
            "rss_note": "Sum of live process RSS; shared pages can be counted repeatedly.",
        },
        "baseline_sha256": helper["hashes"](a.baseline_root),
        "joint_sha256": helper["hashes"](root),
        "runs": [],
        "summary": {},
    }

    def background_test_jobs():
        found = []
        for process in psutil.process_iter(["pid", "cmdline"]):
            try:
                command = process.info["cmdline"] or []
                if not command:
                    continue
                binary = Path(command[0]).name
                docker_test = (
                    binary == "docker"
                    and "run" in command
                    and ("pytest" in command or "tools/check.py" in command)
                )
                python_test = binary.startswith("python") and "pytest" in command
                if docker_test or python_test:
                    found.append(process.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return sorted(found)

    def save():
        a.result.parent.mkdir(parents=True, exist_ok=True)
        a.result.write_text(
            json.dumps(result, indent=2).replace(str(Path.home()) + "/", "${HOME}/")
            + "\n"
        )

    for scenario in a.scenarios:
        for cpu in a.cpus:
            configs = [("marginal", a.baseline_root), ("joint", root)]
            for repeat in range(a.repeats + 1):
                for mode, source in configs if repeat % 2 == 0 else configs[::-1]:
                    tag = f"{scenario}-t{cpu}-b{a.blas}-{mode}-r{repeat}"
                    out = a.workdir / tag
                    cmd = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--root",
                        str(source),
                        "--out",
                        str(out),
                        "--mode",
                        mode,
                        "--threads",
                        str(cpu),
                        "--blas",
                        str(a.blas),
                    ] + definitions[scenario]
                    env = dict(
                        os.environ,
                        OPENBLAS_NUM_THREADS=str(a.blas),
                        OMP_NUM_THREADS=str(a.blas),
                        MKL_NUM_THREADS=str(a.blas),
                        PYTHONDONTWRITEBYTECODE="1",
                    )
                    peak = 0
                    max_children = 0
                    other_jobs_before = background_test_jobs()
                    host_start = psutil.cpu_times()
                    wall_start = time.perf_counter()
                    with Path(str(out) + ".log").open("w") as log:
                        proc = subprocess.Popen(
                            cmd,
                            cwd=a.workdir,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                        parent = psutil.Process(proc.pid)
                        while proc.poll() is None:
                            try:
                                children = parent.children(recursive=True)
                                max_children = max(max_children, len(children))
                                rss = 0
                                for child in [parent] + children:
                                    try:
                                        rss += child.memory_info().rss
                                    except psutil.NoSuchProcess:
                                        pass
                                peak = max(peak, rss)
                            except psutil.NoSuchProcess:
                                pass
                            time.sleep(0.05)
                        if proc.returncode:
                            raise RuntimeError(
                                f"{tag}: exit {proc.returncode}; see log"
                            )
                    host_end = psutil.cpu_times()
                    host_wall = time.perf_counter() - wall_start
                    host_active = sum(
                        getattr(host_end, field) - getattr(host_start, field)
                        for field in ("user", "nice", "system")
                    )
                    row = json.loads(Path(str(out) + ".json").read_text())
                    row["host_active_cpu_cores"] = host_active / host_wall
                    row["background_test_jobs_before"] = other_jobs_before
                    row["background_test_jobs_after"] = background_test_jobs()
                    row.update(
                        scenario=scenario,
                        threads=cpu,
                        blas=a.blas,
                        mode=mode,
                        repeat=repeat,
                        warmup=repeat == 0,
                        peak_tree_rss_mib=peak / 1024**2,
                        max_children=max_children,
                        process_wall_seconds=time.perf_counter() - wall_start,
                    )
                    import pandas as pd

                    row["tables"] = {
                        f.name: len(pd.read_pickle(f))
                        for f in sorted(out.glob("csubst_cb_*.unrounded.pkl"))
                    }
                    required_arity = {
                        "pair": 2,
                        "pair_b": 2,
                        "foreground4": 4,
                        "exhaustive3": 3,
                        "default4": 2,
                        "heuristic6": 6,
                        "heuristic6_b": 6,
                    }[scenario]
                    if (
                        row["tables"].get(
                            f"csubst_cb_{required_arity}.tsv.unrounded.pkl", 0
                        )
                        == 0
                    ):
                        raise AssertionError(
                            f"{tag}: requested arity was not actually computed"
                        )
                    if scenario.startswith("heuristic6"):
                        final = pd.read_pickle(out / "csubst_cb_6.tsv.unrounded.pkl")
                        ids = final.filter(regex="^branch_id_").to_numpy().tolist()
                        if ids != [[1, 26, 33, 35, 40, 102]]:
                            raise AssertionError(
                                f"{tag}: arity 6 detection changed: {ids}"
                            )
                        if not (
                            (final["OCNany2spe"] >= 2) & (final["omegaCany2spe"] >= 5)
                        ).all():
                            raise AssertionError(f"{tag}: arity 6 thresholds failed")
                        row["arity6_verified"] = True
                    result["runs"].append(row)
                    save()
                    print(
                        tag,
                        round(row["seconds"], 3),
                        round(row["peak_tree_rss_mib"], 1),
                        row["tables"],
                        flush=True,
                    )
            key = f"{scenario}-t{cpu}-b{a.blas}"
            result["summary"][key] = {}
            for mode, _ in configs:
                selected = [
                    r
                    for r in result["runs"]
                    if r["scenario"] == scenario
                    and r["threads"] == cpu
                    and r["mode"] == mode
                    and not r["warmup"]
                ]
                result["summary"][key][mode] = {
                    metric: {
                        "median": float(np.median([r[metric] for r in selected])),
                        "min": min(r[metric] for r in selected),
                        "max": max(r[metric] for r in selected),
                    }
                    for metric in [
                        "seconds",
                        "process_wall_seconds",
                        "peak_parent_rss_mib",
                        "peak_tree_rss_mib",
                    ]
                }
            save()


if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.argv.remove("--worker")
        worker()
    else:
        main()
