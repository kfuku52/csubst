"""Seeded count-null rejection and power regressions; no biological FPR claim."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / '.github' / 'scripts'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location('omega_count_null_check', SCRIPT_DIR / 'omega_count_null_check.py')
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


@pytest.mark.parametrize('setting', [
    'raw', 'symmetric', pytest.param('independent_null', marks=pytest.mark.slow),
])
@pytest.mark.parametrize('regime', ['sparse_S', 'dense', 'enriched'])
def test_independent_count_null_and_power(setting, regime):
    check.validate_case(check.run_case(setting, regime, trials=256, draws=199))


def test_regression_gate_rejects_inflation_and_zero_power():
    with pytest.raises(AssertionError, match='Excess rejection'):
        check.validate_case(dict(regime='dense', excess_rejection_p=1e-9))
    with pytest.raises(AssertionError, match='power'):
        check.validate_case(dict(regime='enriched', rate=0.))
