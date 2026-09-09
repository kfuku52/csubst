#!/usr/bin/env python3
"""Reproducible CPU comparison of CSUBST's AA-to-3Di inference backends.

Prepare model resources with `csubst download` before running. Each backend
runs in its own process, with downloads and prediction caches disabled. Model
loading, a 32-residue warmup, and single-sequence parity checks are excluded
from the repeated inference times. No neural inference runs concurrently.
"""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BACKENDS = ("prostt5", "prostt5-cnn", "esm3di-35m")


def _read_inputs(path):
    from csubst import genetic_code, sequence
    if path is not None:
        sequences = sequence.read_fasta(path)
    else:
        sequences = {}
        for family in ("PGK", "PEPC"):
            source = ROOT / "csubst" / "dataset" / (family + ".untrimmed_cds.fa")
            for name, cds in sequence.read_fasta(source).items():
                sequences[family + ":" + name] = sequence.translate_codon_aligned_sequence_to_aa(
                    cds, genetic_code.get_codon_table(1)
                ).replace("-", "")
                # One full-length representative per family keeps the slow
                # encoder-decoder baseline practical. Use --input-fasta for more.
                break
    if not sequences or not any(sequences.values()):
        raise ValueError("Benchmark input must contain nonempty amino-acid sequences.")
    return sequences


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _worker(args):
    import torch
    import transformers
    from csubst import structural_alphabet as sa, structural_prediction as sp

    torch.set_num_threads(args.threads)
    sequences = _read_inputs(args.input_fasta)
    sequences = {key: sa._sanitize_aa_sequence_for_prostt5(seq) for key, seq in sequences.items()}
    backend = args.worker
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fasta = "".join(">{}\n{}\n".format(key, seq) for key, seq in sequences.items())
    (out / "input.fa").write_text(fasta, encoding="utf-8")
    config = dict(sa_backend=backend, sa_batch_size=args.batch_size, prostt5_device="cpu",
                  prostt5_no_download=True, prostt5_cache=False, threads=args.threads,
                  blas_threads=args.threads, resource_cache_dir=args.resource_cache_dir)
    start = time.perf_counter()
    if backend == "prostt5":
        components = sa._load_prostt5_components(config)
        parameter_count = sum(parameter.numel() for parameter in components[2].parameters())
        loader_patch = patch.object(sa, "_load_prostt5_components", return_value=components)
    else:
        predictor = sp.load_encoder_predictor(config)
        parameter_count = sum(parameter.numel() for parameter in predictor.model.parameters())
        if predictor.classifier is not None:
            parameter_count += sum(parameter.numel() for parameter in predictor.classifier.parameters())
        loader_patch = patch.object(sp, "load_encoder_predictor", return_value=predictor)
    load_seconds = time.perf_counter() - start
    result = dict(
        backend=backend, model_key=sa.get_3di_model_cache_key(config), config=config,
        platform=platform.platform(), machine=platform.machine(), python=platform.python_version(),
        torch=torch.__version__, transformers=transformers.__version__,
        torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        dtype="float32", parameter_count=parameter_count, load_seconds=load_seconds,
        input_sha256=hashlib.sha256(fasta.encode()).hexdigest(),
        lengths={key: len(seq) for key, seq in sequences.items()},
        total_residues=sum(map(len, sequences.values())), warmup_residues=32,
        source_sha256={name: hashlib.sha256((ROOT / "csubst" / name).read_bytes()).hexdigest()
                       for name in ("structural_alphabet.py", "structural_prediction.py")},
        runs=[],
    )
    print("{}: model load {:.3f}s; {} residues; {} parameters".format(
        backend, load_seconds, result["total_residues"], parameter_count), flush=True)
    with loader_patch:
        sa.predict_3di({"warmup": next(seq for seq in sequences.values() if seq)[:32]}, config)
        reference = None
        for i in range(args.repeats):
            start = time.perf_counter()
            predicted = sa.predict_3di(sequences, config)
            seconds = time.perf_counter() - start
            if set(predicted) != set(sequences) or any(
                len(predicted[key]) != len(seq) or not set(predicted[key]) <= set("ACDEFGHIKLMNPQRSTVWY")
                for key, seq in sequences.items()
            ):
                raise ValueError("Invalid output lengths/alphabet/identifiers from " + backend)
            if reference is not None and predicted != reference:
                raise ValueError("Predictions changed across repeated runs of " + backend)
            reference = predicted
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
            result["runs"].append(dict(seconds=seconds, predictions=predicted))
            _write_json(out / (backend + ".json"), result)
            print("{}: run {}/{} {:.3f}s".format(backend, i + 1, args.repeats, seconds), flush=True)
        if backend != "prostt5":
            single = sa.predict_3di(sequences, dict(config, sa_batch_size=1))
            result["single_sequence_predictions_identical"] = single == reference
            if single != reference:
                _write_json(out / (backend + ".json"), result)
                raise ValueError("Batch and single-sequence predictions differ for " + backend)
    result["repeated_predictions_identical"] = True
    _write_json(out / (backend + ".json"), result)
    (out / (backend + ".fa")).write_text(
        "".join(">{}\n{}\n".format(key, seq) for key, seq in reference.items()), encoding="utf-8"
    )


def _summarize(out, backends):
    results = {backend: json.loads((out / (backend + ".json")).read_text()) for backend in backends}
    if len({value["input_sha256"] for value in results.values()}) != 1:
        raise ValueError("Cannot compare backends with different inputs.")
    baseline = results.get("prostt5")
    baseline_median = statistics.median(run["seconds"] for run in baseline["runs"]) if baseline else None
    rows = ["backend\tmedian_seconds\tmin_seconds\tmax_seconds\tresidues_per_second\tload_seconds\tpeak_rss_bytes\tspeedup_vs_prostt5"]
    for backend, result in results.items():
        times = [run["seconds"] for run in result["runs"]]
        median = statistics.median(times)
        rows.append("{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{}\t{}".format(
            backend, median, min(times), max(times), result["total_residues"] / median,
            result["load_seconds"], result["peak_rss_bytes"],
            "" if baseline_median is None else "{:.6f}".format(baseline_median / median),
        ))
    (out / "summary.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    agreements = []
    for first, second in itertools.combinations(results, 2):
        a, b = (results[name]["runs"][0]["predictions"] for name in (first, second))
        matches = sum(sum(x == y for x, y in zip(a[key], b[key])) for key in a)
        total = sum(map(len, a.values()))
        agreements.append(dict(first=first, second=second, matching_residues=matches,
                               total_residues=total, agreement=matches / total))
    _write_json(out / "model_agreement.json", agreements)
    print("\n".join(rows))
    print("Model agreement measures output differences, not accuracy against known structures.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fasta", dest="input_fasta", help="AA FASTA; default: bundled full-length PGK and PEPC CDS translations")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resource-cache-dir", default="")
    parser.add_argument("--worker", choices=BACKENDS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1 or args.batch_size < 0:
        parser.error("threads/repeats must be positive and batch-size must be nonnegative")
    if args.worker:
        _worker(args)
        return
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(args.threads)
    env.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1", DISABLE_SAFETENSORS_CONVERSION="1")
    for backend in args.backends:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", backend,
                   "--output-dir", str(out), "--threads", str(args.threads),
                   "--batch-size", str(args.batch_size), "--repeats", str(args.repeats),
                   "--resource-cache-dir", args.resource_cache_dir]
        if args.input_fasta:
            command += ["--input-fasta", str(Path(args.input_fasta).resolve())]
        with (out / (backend + ".log")).open("w") as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError("{} failed; see {}".format(backend, out / (backend + ".log")))
        print("Finished {} (see {} for per-run timings).".format(backend, out / (backend + ".log")), flush=True)
    _summarize(out, args.backends)


if __name__ == "__main__":
    main()
