"""Independent units, selection and unavailable-output accounting for #46."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / '.github/scripts/omega_pipeline_fpr.py'
SPEC = importlib.util.spec_from_file_location('omega_pipeline_fpr', SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def test_selection_and_missing_p_are_not_dropped_from_dataset_denominator():
    frame = pd.DataFrame({
        'OCNany2spe': [1., 2., 3., 4.], 'omegaCany2spe': [8., 4., 6., np.inf],
        'pomegaCany2spe': [.001, .01, .04, np.nan],
        'qomegaCany2spe': [.004, .02, .08, np.nan],
    })
    result = pipeline.summarize_frame(frame)
    assert result['rows'] == 4 and result['finite_p'] == 3
    assert result['selected'] == 2 and result['selected_finite_p'] == 1
    assert result['any_q05'] and result['selected_p05']
    assert not result['selected_q05']
    assert not result['top_selected_p05']  # The highest score has unavailable P.


def test_binomial_interval_counts_independent_datasets_including_no_candidates():
    result = pipeline.binomial_result([False] * 200)
    assert result['replicates'] == 200 and result['rejections'] == 0
    assert result['interval_95'][1] == pytest.approx(1 - .025 ** (1 / 200))


def test_inference_topology_contains_no_true_lengths_or_parameters(tmp_path):
    assert ':' not in pipeline.TOPOLOGY
    cmd = pipeline.search_command(tmp_path, tmp_path / 'search', pipeline.SETTINGS['hypergeom_pp005'], 3999, 17, 'iqtree3')
    assert cmd[cmd.index('--max_arity') + 1] == '3'
    assert cmd[cmd.index('--cutoff_stat') + 1] == pipeline.CUTOFF
    assert cmd[cmd.index('--iqtree_state') + 1] == str(tmp_path / 'fit.state')
    assert cmd[cmd.index('--omega_pvalue_niter_schedule') + 1] == '3999'


def test_seed_channels_and_replicates_are_distinct():
    seeds = [pipeline.seed_for(4609201, regime, replicate, channel)
             for regime in range(2) for replicate in range(200) for channel in range(3)]
    assert len(set(seeds)) == len(seeds)


def test_independent_audit_catches_q_and_selection_errors():
    if str(SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT.parent))
    from omega_pipeline_fpr_report import audit_table

    frame = pd.DataFrame({'pomegaCany2spe': [.01, .5, np.nan],
                          'qomegaCany2spe': [.02, .5, np.nan]})
    stats = pd.DataFrame({'arity': [2], 'num_qualified_all': [1]})
    audit_table(frame, {'selected': 1}, stats, '2')
    with pytest.raises(AssertionError, match='selection'):
        audit_table(frame, {'selected': 0}, stats, '2')
    frame.loc[0, 'qomegaCany2spe'] = .01
    with pytest.raises(AssertionError, match='BH'):
        audit_table(frame, {'selected': 1}, stats, '2')
