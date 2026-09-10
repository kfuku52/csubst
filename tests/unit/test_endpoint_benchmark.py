from pathlib import Path
import runpy

import pandas as pd
import pytest

compare_frames = runpy.run_path(str(Path(__file__).resolve().parents[2] / '.github/scripts/benchmark_endpoint_optimization.py'))['compare_frames']


def test_conditioned_ratio_comparison_checks_its_inputs():
    a = pd.DataFrame({'OCNany2any': [.5], 'OCNany2spe': [.5 - 1e-6], 'OCNany2dif': [1e-6]})
    a['OCNCoD'] = a.OCNany2spe / a.OCNany2dif
    b = a.copy()
    b['OCNany2dif'] += 1e-15
    b['OCNCoD'] = b.OCNany2spe / b.OCNany2dif
    assert compare_frames(a, b)['ratio_input_error_propagation']
    b['OCNCoD'] += 1
    with pytest.raises(AssertionError, match='ratio'):
        compare_frames(a, b)
    b = a.copy()
    b['OCNany2spe'] += .01
    with pytest.raises(AssertionError):
        compare_frames(a, b)


def test_conditioned_ratio_comparison_accepts_identical_masked_zero():
    a = pd.DataFrame({'OCNany2any': [0.0], 'ECNany2any': [1.0],
                      'OCSany2any': [1.0], 'ECSany2any': [float('nan')],
                      'dNCany2any': [0.0], 'dSCany2any': [float('nan')],
                      'omegaCany2any': [0.0]})
    assert compare_frames(a, a.copy())['ratio_input_error_propagation']
    changed = a.copy()
    changed['omegaCany2any'] = 1.0
    with pytest.raises(AssertionError, match='omegaCany2any'):
        compare_frames(a, changed)
