"""Audit every non-log CLI artifact, including model metadata and run status."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'reports/performance_round3_20261023'))
from bench_common import benchmark_lock  # noqa: E402

spec = importlib.util.spec_from_file_location(
    'cli_audit', ROOT / 'reports/performance_round2_implementation_20261022/benchmark.py')
comparator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparator)


def audit(workdir):
    results = {}
    for case in sorted(p for p in workdir.iterdir() if p.is_dir() and p.name != 'candidate_source'):
        results[case.name] = []
        for before in sorted(case.glob('*_before')):
            after = before.with_name(before.name.replace('_before', '_after'))
            checks = comparator.compare_outputs(before, after)
            files = {str(p.relative_to(before)) for p in before.rglob('*') if p.is_file()}
            other_files = {str(p.relative_to(after)) for p in after.rglob('*') if p.is_file()}
            assert files == other_files
            for name in sorted(files):
                if name in checks or name == 'csubst.log':
                    continue
                left, right = before / name, after / name
                if left.suffix == '.json':
                    first, second = json.loads(left.read_text()), json.loads(right.read_text())
                    excluded = []
                    if left.name == 'csubst_search_run.json':
                        assert first.pop('run_id') and second.pop('run_id')
                        excluded = ['run_id']
                    assert first == second, name
                    checks[name] = {'exact': True, 'excluded_dynamic_fields': excluded}
                else:
                    assert left.read_bytes() == right.read_bytes(), name
                    checks[name] = {'byte_exact': True}
            results[case.name].append(checks)
    return {'scope': 'all output files except csubst.log; table elapsed_sec and search run_id excluded',
            'paired_runs_including_warmups': sum(len(v) for v in results.values()), 'cases': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    with benchmark_lock():
        result = audit(args.workdir)
    args.result.write_text(json.dumps(result, indent=2) + '\n')
    print('Audited', result['paired_runs_including_warmups'], 'before/after pairs.')
