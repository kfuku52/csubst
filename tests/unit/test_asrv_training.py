import json

import numpy as np
import pandas as pd
import pytest

from csubst import asrv, ete, omega, substitution, substitution_sparse, tree


def context(mode='each', training='background', concentration=2.):
    tr = tree.add_numerical_node_labels(ete.PhyloNode('((a:1,b:1):1,c:1)R;', format=1))
    names = {n.name: int(ete.get_prop(n, 'numerical_label')) for n in tr.traverse()}
    g = dict(tree=tr, asrv=mode, asrv_training_branches=training,
             asrv_concentration=concentration, float_type=np.float64, float_tol=1e-12,
             target_ids={'trait': np.array([names['a']])}, asrv_dirichlet_alpha=1.,
             expectation_method='urn', urn_model='wallenius')
    n = len(list(tr.traverse()))
    tensor = np.zeros((n, 4, 1, 2, 2))
    tensor[names['a'], 0, 0, 0, 1] = 8.
    tensor[names['b'], :, 0, 0, 1] = 1.
    for channel in ('N', 'S'):
        g[channel + '_ind_nomissing_gad'] = ([0], [0], [1])
        g[channel + '_ind_nomissing_ga'] = ([0], [0])
        g[channel + '_ind_nomissing_gd'] = ([0], [1])
    return g, tensor, names


def prepare(g, tensor):
    sites = substitution.get_s(tensor, attr='N')
    return substitution.get_sub_sites(
        g, sites.rename(columns={'N_sub': 'S_sub'}), sites,
        np.ones(tensor.shape[:2] + (2,)), OS_tensor=tensor, ON_tensor=tensor,
    )


@pytest.mark.parametrize('mode', ['each', 'file_each', 'sn', 'pool'])
@pytest.mark.parametrize('sparse', [False, True])
def test_training_excludes_foreground_mass_but_retains_evaluation_totals(mode, sparse):
    g, raw, names = context(mode, concentration=None if mode == 'pool' else 2.)
    g['iqtree_rate_values'] = np.array([1., 2., 1., 1.])
    weights = []
    totals = []
    for signal in (8., 80.):
        data = raw.copy()
        data[names['a'], 0, 0, 0, 1] = signal
        tensor = substitution_sparse.SparseSubstitutionTensor.from_dense(data) if sparse else data
        prepare(g, tensor)
        bg, sg, _, col, _ = omega._prepare_substitution_permutation_components(tensor, 'any2any', 'N', g)
        totals.append(bg[names['a'], 0])
        weights.append(omega._resolve_sub_sites(g, sg, 'any2any', 0, 'any2', '2any', col))
    np.testing.assert_array_equal(totals, [8., 80.])
    np.testing.assert_array_equal(weights[0], weights[1])


def test_concentration_is_total_mass_on_each_evaluation_mask():
    g, raw, _ = context()
    prepare(g, raw)
    g['is_site_nonmissing'][0] = [1, 1, 0, 0]
    g['_asrv_nonmissing_site_indices'] = None
    p = substitution._normalize_site_weights_by_branch(np.array([2., 0., 0., 0.]), g)
    np.testing.assert_allclose(p[0], [.75, .25, 0., 0.])
    np.testing.assert_allclose(p[1], [.625, .125, .125, .125])


@pytest.mark.parametrize('scale', [1e308, 4e-310])
def test_file_weights_are_normalized_when_raw_total_or_reciprocal_overflows(scale):
    g, raw, _ = context(mode='file', training='all', concentration=None)
    g['iqtree_rate_values'] = np.ones(raw.shape[1])
    prepare(g, raw)
    g['is_site_nonmissing'][0] = [0, 1, 0, 0]
    g['_asrv_nonmissing_site_indices'] = None
    weights = np.array([scale, scale, 0., scale / 2])
    probabilities = substitution._normalize_site_weights_by_branch(weights, g, dirichlet_alpha=0.)
    np.testing.assert_array_equal(probabilities[0], [0., 1., 0., 0.])
    np.testing.assert_allclose(probabilities[1], [.4, .4, 0., .2], atol=1e-14, rtol=0.)


@pytest.mark.parametrize('scale', [1e308, np.nextafter(0., 1.)])
def test_extreme_total_concentration_retains_its_prior_mass(scale):
    g, raw, _ = context(concentration=scale)
    prepare(g, raw)
    weights = np.array([scale, 0., 0., 0.])
    probabilities = substitution._normalize_site_weights_by_branch(weights, g)
    np.testing.assert_allclose(probabilities, np.tile([.625, .125, .125, .125], (len(raw), 1)))
    asrv.record_weight_diagnostics(g, 'scaled', weights, probabilities, alpha=1.)
    diagnostics = g['_asrv_diagnostics']['scaled']
    assert diagnostics['prior_fraction_min'] == .5
    assert diagnostics['prior_fraction_max'] == .5


def test_diagnostics_zero_mass_and_missing_masks_are_explicit(tmp_path):
    g, raw, _ = context()
    prepare(g, raw * 0)
    g['asrv_report'] = True
    asrv.collect_diagnostics(g, raw * 0, raw * 0)
    path = tmp_path / 'audit.json'
    asrv.write_provenance(g, path)
    result = json.loads(path.read_text())
    assert result['pipeline_calibrated'] is False
    assert result['wallenius_methods_complete'] is True
    assert result['weight_diagnostics'][0]['prior_fraction_max'] == 1.
    assert result['weight_diagnostics'][0]['zero_weight_branches'] == 0


def test_background_requires_foreground_and_nonempty_training():
    g, raw, _ = context()
    g['target_ids'] = {}
    with pytest.raises(ValueError, match='nonempty foreground'):
        prepare(g, raw)
    g['target_ids'] = {'trait': np.arange(raw.shape[0])}
    with pytest.raises(ValueError, match='No ASRV training'):
        prepare(g, raw)


@pytest.mark.parametrize('value', ['1,1', '-1', '1.5', '1,', ''])
def test_invalid_training_id_specs(value):
    with pytest.raises(ValueError, match='asrv_training_branches'):
        asrv.parse_training_branches(value)


def test_unknown_and_root_training_ids():
    g, raw, _ = context(training='999')
    with pytest.raises(ValueError, match='absent'):
        prepare(g, raw)
    g['asrv_training_branches'] = str(ete.get_prop(g['tree'], 'numerical_label'))
    with pytest.raises(ValueError, match='root'):
        prepare(g, raw)


@pytest.mark.parametrize('value', [np.nan, np.inf, -1.])
def test_nonfinite_or_negative_mass_rejected(value):
    g, raw, _ = context()
    prepare(g, raw)
    with pytest.raises(ValueError, match='finite and non-negative'):
        substitution._normalize_site_weights_by_branch(np.array([value, 1., 0., 0.]), g)


def test_legacy_concentration_recovers_per_site_alpha():
    g, raw, _ = context(concentration=None)
    prepare(g, raw)
    c = np.array([2., 0., 0., 0.])
    legacy = substitution._normalize_site_weights_by_branch(c, g, dirichlet_alpha=.5)
    g['asrv_concentration'] = 2.
    total = substitution._normalize_site_weights_by_branch(c, g)
    np.testing.assert_allclose(legacy, total, atol=1e-15)


def test_shared_rate_assignments_are_integrated_before_branch_overlap():
    g, raw, names = context(mode='file', training='all', concentration=None)
    g.update(asrv_posterior_draws=64, _asrv_rate_categories=np.array([0., 1.]),
             _asrv_rate_posterior=np.full((2, 4), .5), random_seed=12)
    raw.fill(0.)
    raw[[names['a'], names['b']], 0, 0, 0, 1] = 1.
    prepare(g, raw)
    a, b = names['a'], names['b']
    cb = pd.DataFrame({'branch_id_1': [a], 'branch_id_2': [b]})
    draws = list(asrv.posterior_site_rate_draws(g))
    # With one draw per branch, overlap is 1/k if k sites share positive
    # rates, or zero when all four sites are in the zero-rate category.
    expected = np.mean([1 / row.sum() if row.any() else 0. for row in draws])
    actual = omega.calc_E_stat(cb, raw, 'any2any', SN='N', g=g)[0]
    assert actual == pytest.approx(expected)
    assert actual > .25  # Plug-in mean weights give 1/4.
