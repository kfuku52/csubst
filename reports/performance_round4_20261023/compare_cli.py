"""Extend the existing comparable CLI harness with a marginal calibration case."""

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'reports/performance_round3_20261023'))
path = ROOT / 'reports/performance_round3_20261023/compare_cli.py'
spec = importlib.util.spec_from_file_location('round4_comparator', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.comparator.CASES['pgk_marginal200'] = ('PGK', 'scan', {
    'substitution_posterior': 'marginal', 'scan_observation': 'marginal',
    'scan_rate_exposure': 'raw_branch_length', 'scan_rate_length': 'raw',
    'scan_pvalue_calibration': 'candidate_fixed', 'scan_n_permutations': '200'})

if __name__ == '__main__':
    module.main()
