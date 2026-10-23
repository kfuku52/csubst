"""Record repository checks while sharing the round's benchmark lock."""

from pathlib import Path
import runpy

from bench_common import benchmark_lock


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    with benchmark_lock():
        runpy.run_path(str(root / 'reports/performance_implementation_20261022/run_check.py'),
                       run_name='__main__')
