"""Compare additional CLI configurations without making timing claims."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess


source = Path(__file__).resolve().parents[1] / 'memory_optimization_20260911/benchmark.py'
spec = importlib.util.spec_from_file_location('memory_benchmark', source)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
CASES = {
    'pgk_defaults': ('PGK', 'search', 'defaults', {}),
    'pepc_defaults': ('PEPC', 'search', 'defaults', {}),
    'pgk_scan_analytic': ('PGK', 'scan', 'defaults', {'scan_analytic_pvalue': 'endpoint_mixture'}),
    'pgk_scan_parametric3': ('PGK', 'scan', 'defaults', {
        'scan_pvalue_calibration': 'parametric', 'scan_n_permutations': '3',
        'drop_invariant_tip_sites': 'no',
    }),
    'pgk_scan_zero_sub_mass': ('PGK', 'scan', 'defaults', {'drop_invariant_tip_sites': 'zero_sub_mass'}),
    'pgk_sites_threshold': ('PGK', 'sites', 'defaults', {'min_sub_pp': '0.05'}),
}
previous.harness.CASES.update(CASES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    result = {}
    for case in CASES:
        frames, commands = {}, {}
        for label, root in [('before', args.baseline_root), ('after', args.candidate_root)]:
            output = args.workdir / (case + '-' + label)
            command = previous.harness.command_for(case, label, root,
                                                   args.baseline_root / 'csubst/dataset', output)
            environment = dict(os.environ, PYTHONPATH=str(root), OPENBLAS_NUM_THREADS='1',
                               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg',
                               PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', CSUBST_STRICT_EXTENSIONS='1')
            environment.pop('CSUBST_DISABLE_EXTENSIONS', None)
            with (args.workdir / (case + '-' + label + '.log')).open('w') as handle:
                subprocess.run(command, cwd=root, env=environment, stdout=handle,
                               stderr=subprocess.STDOUT, check=True)
            commands[label] = command
            frames[label] = previous.output_summary(output)
        result[case] = dict(commands=commands, comparisons=previous.compare_all(frames['before'], frames['after']))
        args.result.write_text(json.dumps(result, indent=2).replace(str(Path.home()) + '/', '${HOME}/') + '\n')
        print(case, 'all columns equivalent', flush=True)


if __name__ == '__main__':
    main()
