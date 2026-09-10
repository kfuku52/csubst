import numpy as np
import pandas as pd
import pytest

from csubst import omega, omega_calibration


@pytest.mark.parametrize('model', ['hypergeom', 'poisson', 'poisson_full', 'nbinom'])
@pytest.mark.parametrize('asrv', ['no', 'sn'])
@pytest.mark.parametrize('sub', ['any2any', 'any2spe', 'spe2any', 'spe2spe'])
def test_real_count_engines_support_frozen_null_maps(model, asrv, sub):
    tensor = np.zeros((3, 8, 1, 2, 2))
    tensor[:, :, 0, 0, 1] = np.array([
        [1., 1., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0., 0., 0., 0.],
        [0., 0., 1., 1., 0., 0., 0., 0.],
    ])
    uniform = np.full(8, 1. / 8.)
    g = dict(calibrate_longtail=True, longtail_method='independent_null',
             longtail_null_niter=100, calc_omega_pvalue=True, random_seed=71,
             expectation_method='urn', output_stats=[sub],
             omega_pvalue_null_model=model, omega_pvalue_nbinom_alpha=.5,
             omega_pvalue_niter_schedule=[100], float_tol=1e-12,
             threads=1, float_type=np.float64, asrv=asrv,
             sub_sites={'no': uniform, 'S': uniform, 'N': uniform})
    cb = pd.DataFrame(dict(branch_id_1=[0, 0], branch_id_2=[1, 2],
                           OCNany2any=[1., 0.], OCSany2any=[1., 0.],
                           ECNany2any=[.5, .5], ECSany2any=[.5, .5],
                           dNCany2any=[2., 0.], dSCany2any=[2., 0.],
                           omegaCany2any=[1., 0.]))
    cb = cb.rename(columns=lambda name: name.replace('any2any', sub))
    for channel in ['N', 'S']:
        g[channel + '_ind_nomissing_gad'] = ([0], [0], [1])
        g[channel + '_ind_nomissing_ga'] = ([0], [0])
        g[channel + '_ind_nomissing_gd'] = ([0], [1])
    omega_calibration._base_seed(g)
    cached_g = omega_calibration._draw_config(cb, g)
    for row in [cb.iloc[[1]], cb.iloc[[0]]]:
        baseline = omega_calibration._draw_rates(row, sub, 'fit', 100, tensor, tensor, g)
        cached = omega_calibration._draw_rates(row, sub, 'fit', 100, tensor, tensor, cached_g)
        for expected, actual in zip(baseline, cached):
            np.testing.assert_array_equal(expected, actual)
    full = omega.calibrate_dsc(cb, g=g, ON_tensor=tensor, OS_tensor=tensor)
    assert not any(key.startswith('_longtail_count_') for key in g)
    assert '_longtail_poisson_means' not in g
    full = omega.add_omega_empirical_pvalues(full, tensor, tensor, g)
    single = omega.calibrate_dsc(cb.iloc[[0]], g=g, ON_tensor=tensor, OS_tensor=tensor, reuse_reference=True)
    single = omega.add_omega_empirical_pvalues(single, tensor, tensor, g)
    for column in ['omegaC' + sub, 'pomegaC' + sub, 'calibration_reference_' + sub]:
        assert full[column].iloc[0] == single[column].iloc[0]
    assert full['pomegaC' + sub].between(1. / 101., 1.).all()
    assert (full['calibration_test_n_' + sub] == 100).all()
    assert full['pomegaC' + sub].iloc[1] == 1.
    omega_calibration.validate_config(g)
