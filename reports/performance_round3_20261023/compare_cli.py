"""Reuse the maintained whole-CLI comparator with this round's shared lock.

Run from repository root. Inputs and candidate sources are frozen in fresh
--workdir directories; result commands use placeholders for temporary paths.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from bench_common import benchmark_lock

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = ROOT / "reports/performance_round2_implementation_20261022/benchmark.py"
spec = importlib.util.spec_from_file_location("round2_cli_comparator", HISTORICAL)
comparator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparator)
comparator.benchmark_lock = benchmark_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=ROOT)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=comparator.CASES, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.experiment = "none"
    try:
        comparator.run(args)
    finally:
        if args.result.exists():
            data = args.result.read_text()
            replacements = [(str(args.workdir), "<WORKDIR>"),
                            (str(args.baseline), "<BASELINE>"),
                            (str(ROOT), "<REPO>"),
                            (sys.executable, "<PYTHON>")]
            for source, target in sorted(replacements, key=lambda item: -len(item[0])):
                data = data.replace(source, target)
            json.loads(data)
            args.result.write_text(data)


if __name__ == "__main__":
    main()
