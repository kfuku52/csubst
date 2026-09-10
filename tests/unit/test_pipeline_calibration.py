import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from csubst import pipeline_calibration as pc


def test_selected_statistic_repeats_eligibility_and_branch_selection():
    df = pd.DataFrame({'branch_id_1': [0, 0, 1], 'branch_id_2': [1, 2, 2],
                       'OCSany2spe': [2., 0., 3.], 'omegaCany2spe': [2., np.nan, 8.]})
    selection = {'statistic': 'omegaCany2spe', 'minimum_counts': {'OCSany2spe': 1}}
    assert pc.selected_statistic(df, selection) == 8
    assert pc.selected_statistic(df, dict(selection, branch_ids=[1, 0])) == 2
    assert pc.selected_statistic(df, dict(selection, branch_ids=[2, 0])) == -np.inf
    assert pc.selected_statistic(df, dict(selection, exclude_branch_ids=[2])) == 2
    with pytest.raises(ValueError, match='Undefined statistic'):
        pc.selected_statistic(df, {'statistic': 'omegaCany2spe'})
    with pytest.raises(ValueError, match='absent'):
        pc.selected_statistic(df, dict(selection, branch_ids=[0, 999]))


def test_reference_tail_includes_ties_and_empty_families():
    assert pc.upper_tail_reference_pvalue(1, [1, 1, 1]) == 1
    assert pc.upper_tail_reference_pvalue(np.inf, [1, np.inf, np.inf]) == .75
    assert pc.upper_tail_reference_pvalue(-np.inf, [-np.inf, 0]) == 1


def test_validation_uses_independent_datasets_not_branch_rows():
    records = [dict(role='calibration', truth='null', selected_statistic=i) for i in range(100)]
    records += [dict(role='validation', truth='null', selected_statistic=i) for i in range(10)]
    records += [dict(role='validation', truth='alternative', selected_statistic=200)]
    rows, result = pc.summarize_validation(records)
    assert len(rows) == 11
    assert result['validation']['null']['independent_datasets'] == 10
    assert result['validation']['alternative']['rate'] == 1.
    assert not result['fpr_criterion_met']  # ten null datasets are insufficient


def test_reference_rank_superuniform_under_known_independent_null():
    rng = np.random.default_rng(293)
    # A complete family is simulated per dataset, including method selection.
    reference = rng.normal(size=(4000, 5)).max(axis=1)
    validation = rng.normal(size=(2000, 5)).max(axis=1)
    p = np.array([pc.upper_tail_reference_pvalue(v, reference) for v in validation])
    low, high = pc.binomial_interval(int((p <= .05).sum()), len(p))
    assert low < .05 < high


def test_manifest_rejects_reused_calibration_data(tmp_path):
    script = Path(__file__).resolve().parents[2] / 'tools' / 'evaluate_urn_pipeline.py'
    spec = importlib.util.spec_from_file_location('urn_pipeline_runner', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'a.fa').write_text('>a\nGCT\n')
    (tmp_path / 't.nwk').write_text('(a:1,b:1);')
    manifest = {'schema_version': 1, 'simulator': {'independent_replicates': True},
                'replicates': [dict(id='a', role='calibration', truth='null', alignment='a.fa', tree='t.nwk'),
                               dict(id='b', role='validation', truth='null', alignment='a.fa', tree='t.nwk')]}
    with pytest.raises(ValueError, match='reused'):
        module.validate_manifest(manifest, tmp_path)
