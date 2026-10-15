"""Profile representative paths in sequential fresh processes (diagnostic only)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys

from audit import load_harness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=False)
    harness = load_harness()
    cases = {
        'pepc_scan_all': ('PEPC', 'scan', 'defaults', {'scan_match': 'all'}),
        'pgk_full_scan20': ('PGK', 'scan', 'defaults', {
            'scan_pvalue_calibration': 'full_scan', 'scan_n_permutations': '20'}),
        'pepc_fixed10': ('PEPC', 'scan', 'defaults', {
            'scan_pvalue_calibration': 'candidate_fixed', 'scan_n_permutations': '10'}),
        'pepc_full': ('PEPC', 'search', 'defaults', {
            'output_stat': 'any2any,spe2any,any2spe,spe2spe',
            'drop_invariant_tip_sites': 'no', 'calibrate_longtail': 'no'}),
        'pepc_marginal': ('PEPC', 'search', 'defaults', {'substitution_posterior': 'marginal'}),
        'pgk_urn': ('PGK', 'search', 'defaults', {'expectation_method': 'urn', 'output_stat': 'any2spe'}),
    }
    harness.CASES.update(cases)
    environment = dict(os.environ, VECLIB_MAXIMUM_THREADS='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    result = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  python=sys.version, kind='diagnostic cProfile, not timing comparison', cases={})
    hashes = {}
    for case in cases:
        destination = args.workdir / case
        destination.mkdir()
        command = harness.command_for(case, 'before', Path.cwd(), Path('csubst/dataset'), destination / 'output')
        profile = destination / 'profile.pstats'
        with (destination / 'stdout.log').open('w') as output:
            subprocess.run([sys.executable, '-m', 'cProfile', '-o', str(profile), '-m', 'csubst', *command[3:]],
                           stdout=output, stderr=subprocess.STDOUT, env=environment, check=True)
        stats = pstats.Stats(str(profile))
        rows = []
        for (path, line, name), (_, calls, self_time, cumulative, _) in stats.stats.items():
            try:
                relative = str(Path(path).relative_to(Path.cwd()))
            except ValueError:
                continue
            if relative.startswith('csubst/'):
                rows.append(dict(function='{}:{}:{}'.format(relative, line, name), calls=calls,
                                 self_seconds=self_time, cumulative_seconds=cumulative))
        rows.sort(key=lambda record: record['cumulative_seconds'], reverse=True)
        for value in command:
            if value.startswith('csubst/dataset/'):
                hashes[value] = hashlib.sha256(Path(value).read_bytes()).hexdigest()
        command[0] = 'python'
        command = [value.replace(str(args.workdir), '${WORKDIR}') for value in command]
        result['cases'][case] = dict(command=command, total_seconds=stats.total_tt, functions=rows[:60])
        print('done', case, flush=True)
    result['input_sha256'] = hashes
    args.result.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
