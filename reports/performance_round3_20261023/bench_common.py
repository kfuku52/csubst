"""Shared measurement controls for the third round of ten independent investigations."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


@contextmanager
def benchmark_lock():
    """Keep independent agents' CPU/memory measurements from overlapping."""
    with open('/tmp/csubst-performance-round3-20261023.lock', 'a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def environment():
    import numpy
    import pandas
    import scipy

    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=Path(__file__).resolve().parents[2], text=True).strip()
    return {
        'commit': commit,
        'python': sys.version,
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'numpy': numpy.__version__,
        'scipy': scipy.__version__,
        'pandas': pandas.__version__,
        'native_threads': 1,
    }


def measure(function, *args, **kwargs):
    wall, cpu = time.perf_counter(), time.process_time()
    result = function(*args, **kwargs)
    return result, {'wall_s': time.perf_counter() - wall,
                    'cpu_s': time.process_time() - cpu}


def summarize(samples):
    return {key: {'median': statistics.median(x[key] for x in samples),
                  'min': min(x[key] for x in samples),
                  'max': max(x[key] for x in samples)}
            for key in ('wall_s', 'cpu_s')}
