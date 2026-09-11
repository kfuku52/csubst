"""Fixed-fit follow-up benchmark, including sampled aggregate process-tree RSS.

Uses the prior CLI harness and all-column equivalence checks. Process-tree RSS
is sampled every 20 ms, includes the time wrapper and runtime helpers, and may
double-count shared pages. It complements, not replaces, /usr/bin/time peak RSS.
"""

import importlib.util
import json
from pathlib import Path
import statistics
import sys
import threading

import psutil


source = Path(__file__).resolve().parents[1] / 'memory_optimization_20260911/benchmark.py'
spec = importlib.util.spec_from_file_location('memory_benchmark', source)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
harness = previous.harness
harness.output_summary = previous.output_summary
harness.compare = previous.compare_all
harness.CASES.update({
    'pepc4_scan_defaults': ('PEPCx4', 'scan', 'defaults', {}),
    'pgk_scan_parallel20': ('PGK', 'scan', 'defaults', {
        'scan_pvalue_calibration': 'full_scan', 'scan_n_permutations': '20', 'threads': '2',
    }),
})
original_run = harness.subprocess.run
tree_samples = []


def measured_run(command, *args, **kwargs):
    if command[:2] != ['/usr/bin/time', '-l']:
        return original_run(command, *args, **kwargs)
    parent = psutil.Process()
    stop = threading.Event()
    sample = {'peak_process_tree_rss_mib': 0., 'max_process_count': 0}

    def monitor():
        while not stop.wait(.02):
            rss, count = 0, 0
            for process in parent.children(recursive=True):
                try:
                    rss += process.memory_info().rss
                    count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            sample['peak_process_tree_rss_mib'] = max(sample['peak_process_tree_rss_mib'], rss / 2**20)
            sample['max_process_count'] = max(sample['max_process_count'], count)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        return original_run(command, *args, **kwargs)
    finally:
        stop.set()
        thread.join()
        tree_samples.append(sample)


if __name__ == '__main__':
    result_path = Path(sys.argv[sys.argv.index('--result') + 1])
    harness.subprocess.run = measured_run
    try:
        harness.main()
    finally:
        harness.subprocess.run = original_run
        if result_path.exists():
            result = json.loads(result_path.read_text())
            assert len(result['runs']) == len(tree_samples)
            for row, sample in zip(result['runs'], tree_samples):
                row.update(sample)
            for case, summary in result['summary'].items():
                for version in ('before', 'after'):
                    values = [r['peak_process_tree_rss_mib'] for r in result['runs']
                              if r['case'] == case and r['version'] == version and not r['warmup']]
                    summary[version]['peak_process_tree_rss_mib'] = {
                        'median': statistics.median(values), 'min': min(values), 'max': max(values),
                    }
            result['environment']['threads'] = '1; pgk_scan_parallel20 uses 2'
            result['environment']['process_tree_rss_scope'] = __doc__
            result_path.write_text(json.dumps(result, indent=2) + '\n')
