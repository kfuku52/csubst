import copy

import numpy as np
import pandas as pd
import pytest

from csubst import omega, omega_calibration, randomness
from csubst.longtail import QuantileMap


def frame(n=(10., 1., 1.), s=(1., 1., 1.)):
    n, s = np.asarray(n), np.asarray(s)
    return pd.DataFrame(dict(branch_id_1=np.arange(len(n)), branch_id_2=np.arange(len(n)) + 10,
                             dNCany2spe=n, dSCany2spe=s,
                             omegaCany2spe=omega._calc_raw_omega(n, s, 1e-12),
                             ECNany2spe=np.ones(len(n)), ECSany2spe=np.ones(len(n)),
                             OCNany2spe=n, OCSany2spe=s))


def config(method='empirical'):
    return dict(output_stats=['any2spe'], calibrate_longtail=True,
                longtail_method=method, expectation_method='urn',
                calc_omega_pvalue=True, random_seed=13, longtail_null_niter=100,
                omega_pvalue_niter_schedule=[100], float_tol=1e-12)


def test_frozen_map_matches_reference_and_is_subset_invariant():
    rng = np.random.default_rng(42)
    n = rng.lognormal(size=150)
    s = np.round(rng.lognormal(size=150), 1)
    mapping = QuantileMap.fit(n, s)
    expected, _, _ = omega._calibrate_dsc_vector(n, s)
    np.testing.assert_array_equal(mapping.apply(s), expected)
    selected = [42, 1, 72, 3]
    np.testing.assert_array_equal(mapping.apply(s[selected]), expected[selected])
    assert QuantileMap.fit(n[::-1], s[::-1]).reference_id == mapping.reference_id
    with pytest.raises(ValueError):
        mapping.source_values[0] = 10


def test_insufficient_reference_and_nonfinite_targets_are_explicit():
    cb = frame([10., np.inf, np.nan], [1., 1., 1.])
    out = omega_calibration.apply_calibration(cb, config())
    np.testing.assert_array_equal(out.omegaCany2spe, cb.omegaCany2spe)
    assert out.calibration_status_any2spe.tolist() == ['insufficient_reference', 'nonfinite_target', 'nonfinite_target']
    assert out.calibration_n_any2spe.tolist() == [1, 1, 1]
    assert 'omegaCany2spe_nocalib' in out
    np.testing.assert_array_equal(omega._calibrate_dsc_vector([10.], [1.])[0], [1.])
    mapping = QuantileMap.fit([np.nan], [1.])
    assert mapping.status == 'insufficient_reference'
    np.testing.assert_array_equal(mapping.apply([0., np.inf, np.nan]), [0., np.inf, np.nan])


def test_missing_batches_share_full_empirical_reference():
    cb, g = frame(), config()
    full = omega_calibration.apply_calibration(cb, g)
    single = omega_calibration.apply_calibration(cb.iloc[[0]], g, reuse_reference=True)
    pair = omega_calibration.apply_calibration(cb.iloc[[0, 1]], g, reuse_reference=True)
    assert single.omegaCany2spe.iloc[0] == pair.omegaCany2spe.iloc[0] == full.omegaCany2spe.iloc[0]
    with pytest.raises(ValueError, match='Missing frozen'):
        omega_calibration.apply_calibration(cb.iloc[[0]], config(), reuse_reference=True)


def test_equal_denominators_are_not_counted_as_increased():
    out = omega_calibration.apply_calibration(frame([1., 1.], [1., 1.]), config())
    assert not out.calibration_increased_any2spe.any()


def test_missing_empirical_pvalues_are_explicitly_unavailable():
    cb, g = frame(), config()
    cb['pomegaCany2spe'] = [.1, .5, .5]
    omega_calibration.apply_calibration(cb, g)
    out = omega_calibration.apply_calibration(cb.iloc[[0]], g, reuse_reference=True)
    assert np.isnan(out.pomegaCany2spe.iloc[0])
    assert out.pomegaCany2spe_nocalib.iloc[0] == .1
    assert out.calibration_pvalue_status_any2spe.iloc[0] == 'unavailable_full_population_null'


def test_unrequested_category_diagnostics_are_removed():
    from csubst import output_stat
    cb = pd.DataFrame({'omegaCany2spe': [1.], 'calibration_n_any2spe': [10],
                       'omegaCspe2spe': [2.], 'calibration_n_spe2spe': [20]})
    out = output_stat.drop_unrequested_stat_columns(cb, ['any2spe'])
    assert out.columns.tolist() == ['omegaCany2spe', 'calibration_n_any2spe']


def test_cli_default_and_independent_null_options():
    from csubst.cli import _build_parser
    parser = _build_parser(show_advanced=True)
    assert parser.parse_args(['search']).calibrate_longtail is False
    assert parser.parse_args(['search']).longtail_method == 'independent_null'
    args = parser.parse_args(['search', '--calibrate_longtail', 'yes',
                              '--longtail_method', 'independent_null', '--longtail_null_niter', '250',
                              '--longtail_test_block_size', '32'])
    assert args.calibrate_longtail is True and args.longtail_null_niter == 250
    assert args.longtail_test_block_size == 32


def test_frozen_pipeline_rejects_inconsistent_transformation_argument():
    with pytest.raises(ValueError, match='quantile transformation only'):
        omega.calibrate_dsc(frame(), transformation='gamma', g=config())


def test_direct_calibration_call_cannot_bypass_smoothing_guard():
    g = config('independent_null')
    g.pop('calibrate_longtail')
    g.update(pseudocount_mode='symmetric', pseudocount_alpha=1.)
    with pytest.raises(ValueError, match='pseudocounts'):
        omega.calibrate_dsc(frame(), g=g, ON_tensor=object(), OS_tensor=object())


@pytest.mark.parametrize('change,match', [
    ({'expectation_method': 'codon_model'}, 'expectation_method urn'),
    ({'output_stats': ['any2dif']}, 'no dif'),
    ({'pseudocount_mode': 'symmetric', 'pseudocount_alpha': 1}, 'pseudocounts'),
    ({'omega_pvalue_null_model': 'nbinom'}, 'fixed'),
    ({'longtail_null_niter': 1}, '>= 100'),
    ({'longtail_test_block_size': 0}, 'positive'),
])
def test_unsupported_null_config_is_rejected(change, match):
    g = config('independent_null')
    g.update(change)
    with pytest.raises(ValueError, match=match):
        omega_calibration.validate_config(g)


def test_same_observed_and_null_counts_give_one(monkeypatch):
    def draw(cb_ids, sub_tensor, mode, SN, niter, g, obs_count=None):
        return np.full((1, niter), 10. if SN == 'N' else 1.)
    monkeypatch.setattr(omega, '_get_mode_permutation_count_matrix', draw)
    g = config('independent_null')
    cb = omega.calibrate_dsc(frame(), g=g, ON_tensor=object(), OS_tensor=object())
    cb = omega.add_omega_empirical_pvalues(cb, object(), object(), g)
    assert cb.pomegaCany2spe.iloc[0] == 1.
    assert cb.calibration_test_n_any2spe.iloc[0] == 100
    assert cb.calibration_test_undefined_any2spe.iloc[0] == 0


def test_independent_null_rows_streams_blocks_and_schedules(monkeypatch):
    seen = []
    def draw(cb_ids, sub_tensor, mode, SN, niter, g, obs_count=None):
        seen.append((int(cb_ids[0, 0]), SN, g['random_seed'], niter))
        return randomness.generator(g['random_seed'], SN, mode).poisson(2., (1, niter)).astype(float)
    monkeypatch.setattr(omega, '_get_mode_permutation_count_matrix', draw)
    raw = frame([10., 1., 3.])
    def run(cb, schedule, block):
        g = config('independent_null')
        g.update(omega_pvalue_niter_schedule=schedule, longtail_test_block_size=block)
        before = copy.deepcopy(g)
        cb = omega.calibrate_dsc(cb, g=g, ON_tensor=object(), OS_tensor=object())
        cb = omega.add_omega_empirical_pvalues(cb, object(), object(), g)
        assert g.get('_random_stream_counters') == before.get('_random_stream_counters')
        return cb
    full = run(raw, [100], 256)
    subset = run(raw.iloc[[2, 0]], [10, 100], 7)
    for col in ['dSCany2spe', 'omegaCany2spe', 'pomegaCany2spe', 'calibration_reference_any2spe']:
        np.testing.assert_array_equal(full.iloc[[2, 0]][col], subset[col])
    seeds = {seed for row, channel, seed, size in seen if row == 0 and channel == 'N'}
    assert len(seeds) == 2  # separate fit and test seeds, stable on repeat


def test_existing_pq_are_preserved_only_as_uncalibrated():
    cb = frame()
    cb['pomegaCany2spe'] = [.1, .5, .5]
    cb['qomegaCany2spe'] = [.3, .5, .5]
    out = omega_calibration.apply_calibration(cb, config())
    assert 'pomegaCany2spe' not in out and 'qomegaCany2spe' not in out
    np.testing.assert_array_equal(out.pomegaCany2spe_nocalib, cb.pomegaCany2spe)
    with pytest.raises(ValueError, match='already'):
        omega_calibration.apply_calibration(out, config())


def test_omitted_method_uses_independent_null_for_observation_and_test(monkeypatch):
    g = config('independent_null')
    del g['longtail_method']
    monkeypatch.setattr(omega, '_get_mode_permutation_count_matrix',
                        lambda cb_ids, sub_tensor, mode, SN, niter, g, obs_count=None:
                        np.ones((len(cb_ids), niter)))
    cb = omega.calibrate_dsc(frame(), g=g, ON_tensor=object(), OS_tensor=object())
    cb = omega.add_omega_empirical_pvalues(cb, object(), object(), g)
    assert (cb.calibration_method_any2spe == 'independent_null').all()
    assert (cb.calibration_pvalue_status_any2spe == 'fixed_independent_null').all()


def test_omitted_method_validates_independent_null_requirements():
    g = config()
    del g['longtail_method']
    g['expectation_method'] = 'codon_model'
    with pytest.raises(ValueError, match='expectation_method urn'):
        omega_calibration.validate_config(g)
