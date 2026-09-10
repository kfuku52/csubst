import copy
import itertools

import numpy as np
import pandas as pd
import pytest

from csubst import omega, omega_null, output_stat, pseudocount
from csubst.omega_statistics import _calc_permutation_omega_matrix, _calc_omega_empirical_upper_tail_counts_from_perm


def frame():
    cb = pd.DataFrame({'branch_id_1': [0, 1, 2], 'branch_id_2': [3, 4, 5]})
    for ch, shift in [('N', 0.), ('S', 1.)]:
        observed = np.array([[1, 0, 3, 2], [0, 1, 0, 2], [2, 2, 1, 0]], dtype=float)+shift
        expected = np.array([[.5, 1, 2, .2], [.2, 2, .1, 1], [2, .2, 1, .1]])+shift
        for sub in output_stat.ALL_OUTPUT_STATS:
            cb['OC'+ch+sub] = omega_null.project_atoms(observed, sub)
            cb['EC'+ch+sub] = omega_null.project_atoms(expected, sub)
    return cb


def config(**kwargs):
    return dict(dict(calc_omega_pvalue=True, expectation_method='urn',
                     omega_pvalue_null_model='poisson', omega_pvalue_niter_schedule=[11],
                     output_stats=list(output_stat.ALL_OUTPUT_STATS), float_tol=1e-12,
                     pseudocount_mode='symmetric', pseudocount_alpha=1., random_seed=52), **kwargs)


@pytest.mark.parametrize('target', ['observed', 'expected', 'both'])
@pytest.mark.parametrize('mode,alpha', [('symmetric', 1.), ('empirical', 1.), ('symmetric', 'auto'), ('empirical', 'auto')])
def test_same_counts_p_one_all_categories_and_smoothing(monkeypatch, target, mode, alpha):
    cb = frame()
    def draw(self, niter):
        ch = 'N' if np.allclose(self.means[0], [.5, 1, 2, .2]) else 'S'
        atoms = np.stack([cb['OC'+ch+s] for s in output_stat.ATOMIC_OUTPUT_STATS], axis=-1)
        return np.repeat(atoms[:, None, :], niter, axis=1)
    monkeypatch.setattr(omega_null.PoissonAtoms, 'draw', draw)
    g = config(pseudocount_mode=mode, pseudocount_alpha=alpha, pseudocount_target=target)
    cb = omega.get_omega(cb, g)
    out = omega.add_omega_empirical_pvalues(cb, None, None, g)
    for sub in output_stat.ALL_OUTPUT_STATS:
        assert (out['pomegaC'+sub] == 1).all()
        assert (out['pvalue_undefined_'+sub] == 0).all()


@pytest.mark.parametrize('calibration', [None, 'empirical', 'independent_null'])
@pytest.mark.parametrize('target', ['observed', 'expected', 'both'])
def test_fixed_base_regression_identical_counts(monkeypatch, target, calibration):
    sub = 'any2spe'
    cb = pd.DataFrame({'branch_id_1': [0], 'branch_id_2': [1],
                       'OCN'+sub: [1.], 'ECN'+sub: [10.], 'OCS'+sub: [10.], 'ECS'+sub: [10.]})
    def draw(cb_ids, sub_tensor, mode, SN, niter, g, obs_count=None):
        return np.full((len(cb_ids), niter), 1. if SN == 'N' else 10.)
    monkeypatch.setattr(omega, '_get_mode_permutation_count_matrix', draw)
    g = config(output_stats=[sub], pseudocount_target=target, calibrate_longtail=calibration is not None,
               longtail_method=calibration or 'independent_null', longtail_null_niter=100)
    out = omega.get_omega(cb, g)
    if target == 'both':
        assert out['omegaC'+sub].iloc[0] == pytest.approx(2/11)
    if calibration:
        out = omega.calibrate_dsc(out, g=g, ON_tensor=object(), OS_tensor=object())
    out = omega.add_omega_empirical_pvalues(out, None, None, g)
    assert out['pomegaC'+sub].iloc[0] == 1.


def test_atomic_streams_preserve_nesting_covariance_and_repetition_ids():
    means = np.array([[1., 2., 3., 4.], [0., 0., 0., 0.]])
    ids = np.array([[1, 3], [2, 4]])
    whole = omega_null.PoissonAtoms(ids, means, 7, 'N').draw(10000)
    sampler = omega_null.PoissonAtoms(ids[::-1], means[::-1], 7, 'N')
    split = np.concatenate([sampler.draw(31), sampler.draw(9969)], axis=1)[::-1]
    np.testing.assert_array_equal(whole, split)
    aa = omega_null.project_atoms(whole, 'any2any')
    ss = omega_null.project_atoms(whole, 'spe2spe')
    assert (aa >= ss).all()
    assert np.cov(aa[0], ss[0])[0, 1] == pytest.approx(1., abs=.1)
    np.testing.assert_array_equal(omega_null.project_atoms(whole, 'dif2dif'), whole[:, :, 3])
    all_specific = omega_null.PoissonAtoms(ids[:1], np.array([[2., 0., 0., 0.]]), 7, 'N').draw(100)
    for stat in output_stat.ALL_OUTPUT_STATS:
        if 'dif' in stat:
            assert (omega_null.project_atoms(all_specific, stat) == 0).all()


@pytest.mark.parametrize('mode,alpha', [('symmetric', 1.), ('empirical', 1.), ('empirical', 'auto')])
@pytest.mark.parametrize('calibrate', [False, True])
def test_joint_order_blocks_and_schedule_invariance(mode, alpha, calibrate):
    def run(cb, stats, block, schedule):
        g = config(pseudocount_mode=mode, pseudocount_alpha=alpha,
                   output_stats=stats, longtail_test_block_size=block,
                   omega_pvalue_niter_schedule=schedule, calibrate_longtail=calibrate,
                   longtail_method='empirical')
        out = omega.get_omega(cb.copy(), g)
        if calibrate:
            out = omega.calibrate_dsc(out, g=g)
        return omega.add_omega_empirical_pvalues(out, None, None, g)
    stats = list(output_stat.ALL_OUTPUT_STATS)
    a = run(frame(), stats, 11, [11])
    b = run(frame().iloc[::-1], stats[::-1], 3, [2, 11]).iloc[::-1]
    for stat in stats:
        np.testing.assert_array_equal(a['pomegaC'+stat], b['pomegaC'+stat])
        assert (a['pvalue_n_'+stat] == 11).all()


def test_joint_auto_matches_independent_naive_refit(monkeypatch):
    cb = frame()
    rng = np.random.default_rng(8)
    samples = {ch: rng.poisson(1.3, (len(cb), 7, 4)).astype(float) for ch in ['N', 'S']}
    def draw(self, niter):
        ch = 'N' if np.allclose(self.means[0], [.5, 1, 2, .2]) else 'S'
        return samples[ch]
    monkeypatch.setattr(omega_null.PoissonAtoms, 'draw', draw)
    g = config(pseudocount_mode='empirical', pseudocount_alpha='auto', omega_pvalue_niter_schedule=[7])
    observed = omega.get_omega(cb.copy(), g)
    ranks = {s: np.zeros(len(cb)) for s in g['output_stats']}
    for r in range(7):
        pseudo = cb.copy()
        for ch in ['N', 'S']:
            for s in output_stat.ALL_OUTPUT_STATS:
                pseudo['OC'+ch+s] = omega_null.project_atoms(samples[ch][:, r, :], s)
        # Independent orchestration: refit by the normal observed-data entrypoint.
        null = omega.get_omega(pseudo, copy.deepcopy(g))
        for s in g['output_stats']:
            ranks[s] += null['omegaC'+s].to_numpy() >= observed['omegaC'+s].to_numpy()
    out = omega.add_omega_empirical_pvalues(observed, None, None, g)
    for s in g['output_stats']:
        np.testing.assert_allclose(out['pomegaC'+s], (ranks[s]+1)/8)
        assert (out['pvalue_alpha_'+s] == 'refit').all()


def test_small_exact_null_space_and_float_ties():
    counts = np.array(list(itertools.product(range(3), repeat=2)), dtype=float)
    got = _calc_permutation_omega_matrix([2.], [3.], counts[:, 0][None, :], counts[:, 1][None, :], 1e-12, alphas=(1., 1., 1., 1.))
    independent = np.array([((n+1)/3)/((s+1)/4) for n, s in counts])
    np.testing.assert_array_equal(got[0], independent)
    for obs in independent:
        ge, valid = _calc_omega_empirical_upper_tail_counts_from_perm([obs], got)
        assert ge[0] == (independent >= obs).sum()
        assert valid[0] == 9
    ge, _ = _calc_omega_empirical_upper_tail_counts_from_perm([1., np.inf, 1e-30], np.array([[np.nextafter(1., 0.)], [np.inf], [0.]]))
    np.testing.assert_array_equal(ge, [1, 1, 0])


def test_invalid_joint_means_fail_without_clipping():
    with pytest.raises(ValueError, match='incompatible marginals'):
        omega_null.atomic_means(dict(any2any=[1.], spe2any=[1.], any2spe=[1.], spe2spe=[0.]), 1e-12)


@pytest.mark.parametrize('model', ['hypergeom', 'poisson_full', 'nbinom'])
@pytest.mark.parametrize('kwargs', [dict(output_stats=['any2dif']), dict(output_stats=['any2spe'], pseudocount_alpha='auto')])
def test_unsupported_joint_models_fail_early(model, kwargs):
    with pytest.raises(ValueError, match='require --omega_pvalue_null_model poisson'):
        omega_null.validate_config(config(omega_pvalue_null_model=model, **kwargs))


def test_alpha_fit_thinning_is_order_independent():
    obs = np.arange(300.) % 19
    exp = np.arange(300.)/21
    a, _ = pseudocount.estimate_alpha_empirical_bayes([obs], [exp], max_samples=20)
    b, _ = pseudocount.estimate_alpha_empirical_bayes([obs[::-1]], [exp[::-1]], max_samples=20)
    assert a == b


def test_zero_alpha_compatibility_and_nonfinite_statistics():
    n = np.array([[0., 1., np.nan, np.inf], [0., 1e-14, 3., 1.]])
    s = np.array([[0., 0., 1., np.inf], [1., 1., 0., np.nan]])
    baseline = _calc_permutation_omega_matrix([0., 1.], [0., 1.], n, s, 1e-12)
    zero = _calc_permutation_omega_matrix([0., 1.], [0., 1.], n, s, 1e-12, alphas=(0., 0., 0., 0.))
    np.testing.assert_array_equal(baseline, zero)
    assert zero[0, 0] == 0 and zero[1, 1] == 0
    assert np.isinf(zero[0, 1]) and np.isnan(zero[0, 2])
    assert np.isnan(zero[0, 3]) and np.isinf(zero[1, 2])
    tiny = np.finfo(float).tiny
    sm = _calc_permutation_omega_matrix([0.], [0.], [[0.]], [[0.]], 1e-12, alphas=(tiny, tiny, tiny, tiny))
    assert sm[0, 0] == 1.


def test_subset_cannot_refit_population_dependent_statistic():
    cb = frame()
    g = config(pseudocount_mode='empirical')
    g['_pseudocount_context'] = omega._get_pseudocount_context(cb, g, g['output_stats'])
    cb = omega.get_omega(cb, g)
    with pytest.raises(ValueError, match='complete row population'):
        omega.add_omega_empirical_pvalues(cb.iloc[:1], None, None, g)


def test_pvalue_metadata_follows_calibration_and_output_selection():
    from csubst import omega_calibration
    cb = frame()
    g = config(longtail_method='empirical', calibrate_longtail=True)
    cb = omega.get_omega(cb, g)
    cb = omega.add_omega_empirical_pvalues(cb, None, None, g)
    cb = omega_calibration.apply_calibration(cb, g)
    cb = omega.add_omega_empirical_pvalues(cb, None, None, g)
    for s in g['output_stats']:
        assert (cb['pvalue_statistic_'+s] == 'smoothed_calibrated').all()
        assert (cb['pvalue_statistic_'+s+'_nocalib'] == 'smoothed').all()
    selected = output_stat.drop_unrequested_stat_columns(cb, ['any2spe'])
    for c in selected:
        if str(c).startswith(('pvalue_', 'pomegaC', 'qomegaC')):
            assert 'any2spe' in c


def test_joint_null_rejects_stale_derived_expectations():
    cb = frame()
    cb['ECNany2dif'] += 1
    with pytest.raises(ValueError, match='disagree with the joint'):
        omega_null.fitted_atoms(cb, None, 'N', config())


def test_independent_calibration_joint_dif_uses_same_map_in_subsets():
    from csubst import omega_calibration
    g = config(calibrate_longtail=True, longtail_method='independent_null', longtail_null_niter=100)
    observed = omega.get_omega(frame(), g)
    calibrated = omega_calibration.apply_calibration(observed, g, object(), object())
    whole = omega.add_omega_empirical_pvalues(calibrated, object(), object(), g)
    partial_g = dict(g, output_stats=list(reversed(g['output_stats'])),
                     omega_pvalue_niter_schedule=[3, 11], longtail_test_block_size=2)
    partial = omega_calibration.apply_calibration(observed.iloc[[2, 0]], partial_g, object(), object(), reuse_reference=True)
    partial = omega.add_omega_empirical_pvalues(partial, object(), object(), partial_g)
    for s in g['output_stats']:
        for prefix in ['omegaC', 'pomegaC', 'calibration_reference_']:
            np.testing.assert_array_equal(whole.iloc[[2, 0]][prefix+s], partial[prefix+s])
        assert (whole['pvalue_undefined_'+s] == 0).all()


@pytest.mark.parametrize('key', ['output_stats', 'output_stat'])
def test_string_categories_cannot_bypass_joint_null_validation(key):
    g = config(omega_pvalue_null_model='hypergeom')
    g.pop('output_stats')
    g[key] = 'any2dif'
    with pytest.raises(ValueError, match='joint category distribution'):
        omega_null.validate_config(g)
    g[key] = 'any2spe'
    omega_null.validate_config(g)


def test_get_omega_refits_when_counts_or_settings_change():
    g = config(pseudocount_mode='empirical')
    omega.get_omega(frame(), g)
    changed = frame().iloc[:1].copy()
    reused = omega.get_omega(changed.copy(), g)
    fresh = omega.get_omega(changed.copy(), config(pseudocount_mode='empirical'))
    pd.testing.assert_frame_equal(reused, fresh)
    g.update(pseudocount_mode='symmetric', pseudocount_alpha=10.)
    changed_again = omega.get_omega(changed.copy(), g)
    independent = omega.get_omega(changed.copy(), config(pseudocount_mode='symmetric', pseudocount_alpha=10.))
    pd.testing.assert_frame_equal(changed_again, independent)


def test_supplemental_rows_reuse_original_prior_and_do_not_test_subset(monkeypatch):
    g = config(pseudocount_mode='empirical')
    omega.get_omega(frame(), g)
    original_context = g['_pseudocount_context']
    extra = frame().iloc[:1].copy()
    extra['branch_id_1'] = 10
    extra['branch_id_2'] = 12
    frozen = omega.get_omega(extra.copy(), dict(g), context=original_context)
    freshly_fitted = omega.get_omega(extra.copy(), dict(g))
    assert not np.allclose(frozen.omegaCany2spe, freshly_fitted.omegaCany2spe)
    monkeypatch.setattr(omega, 'get_E', lambda cb, g, ON, OS: cb)
    result, g = omega.calc_omega(extra, None, None, g, reuse_pseudocount_context=True)
    np.testing.assert_array_equal(result.omegaCany2spe, frozen.omegaCany2spe)
    assert result.pomegaCany2spe.isna().all() and result.qomegaCany2spe.isna().all()
    assert (result.pvalue_status_any2spe == 'unavailable_full_population_null').all()
    assert g['_pseudocount_context'] is original_context
