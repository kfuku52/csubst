#!/usr/bin/env python3
"""Compare 3Di predictions with residue-matched reference labels, offline by default.

This measures residue prediction, not structural omegaC calibration. See
docs/STRUCTURAL_VALIDATION.md for the manifest, masks and scientific limits.
"""

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backend", choices=("esm3di-35m", "prostt5-cnn", "prostt5"))
    mode.add_argument("--predictions", type=Path, help="Evaluate a previously saved NPZ without loading models")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are rejected")
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--resource-cache-dir", default="")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 0:
        parser.error("--batch-size must be >= 0")
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")

    from csubst import structural_validation as sv
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = sv.validate_manifest(manifest)
    if args.predictions:
        records, provenance = sv.load_predictions(args.predictions)
    else:
        from csubst import structural_prediction as sp
        config = dict(sa_backend=args.backend, prostt5_device=args.device,
                      sa_batch_size=args.batch_size, resource_cache_dir=args.resource_cache_dir,
                      prostt5_no_download=not args.allow_download, prostt5_cache=False)
        records = sp.predict_3di_records({row["id"]: row["amino_acids"] for row in rows}, config)
        packages = {}
        for name in ("numpy", "torch", "transformers", "peft"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        provenance = dict(config=config, packages=packages, python=platform.python_version(),
                          platform=platform.platform(), source_sha256={
                              name: hashlib.sha256((ROOT / "csubst" / name).read_bytes()).hexdigest()
                              for name in ("structural_prediction.py", "structural_validation.py", "structural_alphabet.py")})
    report = sv.evaluate_predictions(manifest, records)
    # All input validation and inference finish before publishing results.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sv.save_predictions(args.output_dir / "predictions.npz", records, provenance)
    for filename, value in (("manifest.json", manifest), ("metrics.json", report)):
        (args.output_dir / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved residue-level validation to {}. Structural omegaC remains uncalibrated.".format(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
