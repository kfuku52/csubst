from itertools import product

import numpy as np
import pytest

from csubst.endpoint import EndpointModel
from csubst.scan_analytic import EndpointEnrichment, adjusted_pvalues, family_size


def test_exact_enumeration_certifies_e_expectation_and_p_superuniformity():
    # Shared hidden root/internal nodes and a rate mixture: neither branches
    # nor their inferred changes are independent. Enumerate EVERY observation.
    model = EndpointModel(np.array([-1, 0, 0, 1, 1, 2, 2]),
        [0, .2, .4, .02, .7, .05, .1], [[-1, 1], [1, -1]], [.5, .5], [0, .3, 2], [.1, .3, .6])
    engine = EndpointEnrichment(model, np.arange(2))
    probabilities, pvalues, evalues = [], [], []
    for pattern in product(range(2), repeat=4):
        tips = {leaf: np.eye(2)[v] for leaf, v in zip(sorted(model.leaves), pattern)}
        logs = engine.log_likelihoods(tips, [3, 5], [0], [1])
        # Independent enumeration over all internal states checks pruning.
        explicit = np.zeros(5)
        for category, weight in enumerate(model.weights):
            for states in product(range(2), repeat=3):
                assignment = dict(enumerate(states)) | dict(zip(sorted(model.leaves), pattern))
                for alt, factor in enumerate([1, 2, 10, 100, None]):
                    likelihood = model.pi[assignment[0]] * weight
                    for child in range(1, 7):
                        trans = model.transition(child, category).copy()
                        if child in (3, 5):
                            if factor is None:
                                if trans[0, 1] > 0:
                                    trans[0] = [0, 1]
                            else:
                                trans[0, 1] *= factor
                                trans /= trans.sum(axis=1, keepdims=True)
                        likelihood *= trans[assignment[model.parents[child]], assignment[child]]
                    explicit[alt] += likelihood
        np.testing.assert_allclose(np.exp(logs), explicit, rtol=2e-13)
        p, log_e = engine.test(tips, [3, 5], [0], [1])
        probabilities.append(explicit[0])
        pvalues.append(p)
        evalues.append(np.exp(log_e))
    probabilities, pvalues, evalues = map(np.array, (probabilities, pvalues, evalues))
    assert probabilities.sum() == pytest.approx(1)
    assert probabilities @ evalues == pytest.approx(1)
    for alpha in np.unique(pvalues):
        assert probabilities[pvalues <= alpha].sum() <= alpha + 1e-12


def test_family_includes_unreported_hypotheses_and_by_dependence_factor():
    assert family_size(100, 20, 2, 1, ['any2spe', 'spe2spe']) == 80000
    p = [.001, .02, .1]
    from scipy.stats import false_discovery_control
    padded = np.r_[p, np.ones(97)]
    for method in ('BH', 'BY'):
        np.testing.assert_allclose(adjusted_pvalues(p, 100, method), false_discovery_control(padded, method=method.lower())[:3])
    assert adjusted_pvalues([], 80000).size == 0
    with pytest.raises(ValueError):
        family_size(1, 20, 1, 1, ['dif2spe'])
    with pytest.raises(ValueError):
        adjusted_pvalues([np.nan], 10)


def test_no_target_missing_data_and_zero_length_are_uninformative():
    model = EndpointModel(np.array([-1, 0, 0]), [0, 0, .1], [[-1, 1], [1, -1]], [.5, .5])
    engine = EndpointEnrichment(model, np.arange(2))
    assert engine.test({1: np.ones(2), 2: np.ones(2)}, [1, 2], [0], [1])[0] == pytest.approx(1)
    assert engine.test({1: np.array([1, 0]), 2: np.array([0, 1])}, [1], [0], [1])[0] == pytest.approx(1)
    with pytest.raises(ValueError, match='non-root'):
        engine.test({1: np.ones(2), 2: np.ones(2)}, [0], [0], [1])


def test_prepare_freezes_family_before_filter_and_annotation_uses_only_tips(tmp_path):
    from csubst import scan_analytic, substitution_scan
    from scan_fixtures import make_scan_context
    g, tensor = make_scan_context()
    report = tmp_path / 'fit.iqtree'
    report.write_text('Model of rate heterogeneity: Uniform\n')
    g.update(scan_analytic_pvalue='endpoint_mixture', substitution_model='GY+FQ',
             path_iqtree_iqtree=str(report), state_cdn=np.repeat(g['state_nsy'], 2, axis=2) / 2,
             instantaneous_codon_rate_matrix=(np.ones((4, 4)) - 4 * np.eye(4)) / 3,
             equilibrium_frequency=np.full(4, .25), nonsynonymous_indices={'A': [0, 1], 'K': [2, 3]})
    engine = scan_analytic.prepare(g)
    assert g['scan_analytic_summary']['family_size'] == 2
    frame, units = substitution_scan.scan_substitutions(g, tensor)
    annotated = scan_analytic.annotate(g, frame, units, engine)
    assert len(annotated) == 1
    assert 0 < annotated.iloc[0]['p_endpoint_enrichment_analytic'] <= 1
    original = annotated['p_endpoint_enrichment_analytic'].copy()
    for node in set(engine.model.order) - engine.model.leaves:
        g['state_cdn'][node] = 0  # Changing ASR cannot change the likelihood test.
        g['state_nsy'][node] = 0
    np.testing.assert_array_equal(scan_analytic.annotate(g, frame, units, engine)['p_endpoint_enrichment_analytic'], original)
    empty = scan_analytic.annotate(g, frame.iloc[:0], units, engine)
    assert 'p_endpoint_enrichment_analytic' in empty
    assert g['scan_analytic_summary']['family_size'] == 2
    g['scan_match'] = 'dif2spe'
    with pytest.raises(ValueError, match='supports'):
        scan_analytic.prepare(g)


def test_option_validation_and_display_use_new_p_only_when_enabled():
    import pandas as pd
    from csubst import scan_analytic, substitution_scan, main_scan
    with pytest.raises(ValueError, match='requires --scan_pvalue_calibration none'):
        scan_analytic.validate_options(dict(scan_analytic_pvalue='endpoint_mixture'))
    g = dict(scan_analytic_pvalue='endpoint_mixture', scan_pvalue_calibration='none',
             scan_site_plot_filter='analytical', scan_site_plot_alpha=.05)
    with pytest.raises(ValueError, match='ml_anc'):
        scan_analytic.validate_options(dict(g, ml_anc=True))
    frame = pd.DataFrame({'p_endpoint_enrichment_analytic': [1e-20, .5],
                          'p_rate_enrichment_asymptotic': [.5, .01],
                          'log_e_endpoint_enrichment': [46.051701859880914, -.3]})
    assert substitution_scan.filter_scan_site_plot_candidates(frame, g).index.tolist() == [0]
    assert substitution_scan.filter_scan_site_plot_candidates(frame, dict(g, scan_analytic_pvalue='none')).index.tolist() == [1]
    serialized = main_scan._prepare_scan_output_table(frame)
    np.testing.assert_array_equal(serialized['p_endpoint_enrichment_analytic'].astype(float), frame['p_endpoint_enrichment_analytic'])
    np.testing.assert_array_equal(serialized['log_e_endpoint_enrichment'].astype(float), frame['log_e_endpoint_enrichment'])


def test_two_rare_endpoints_are_not_limited_by_an_arbitrary_effect_size_cap():
    model = EndpointModel(np.array([-1, 0, 0, 0, 0]), [0, 1e-8, 1e-8, 1e-8, 1e-8],
                          [[-1, 1], [1, -1]], [.5, .5])
    engine = EndpointEnrichment(model, np.arange(2))
    p, log_e = engine.test({1: np.array([0, 1]), 2: np.array([0, 1]),
                            3: np.array([1, 0]), 4: np.array([1, 0])}, [1, 2], [0], [1])
    assert 0 < p < 1e-14
    assert log_e > 30


def test_partial_participation_profile_normalizes_and_has_valid_e_expectation(tmp_path):
    import json
    from csubst.scan_analytic import read_profile, validate_options
    profile = dict(version=1, atoms=[
        dict(multiplier=7., participation=.3, weight=.7),
        dict(multiplier=None, participation=.8, weight=.3)])
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(profile))
    assert read_profile(path) == profile
    validate_options(dict(scan_analytic_pvalue='endpoint_mixture',
                          scan_pvalue_calibration='none', scan_analytic_profile=str(path)))
    model = EndpointModel(np.array([-1, 0, 0]), [0, .1, .7], [[-1, 1], [1, -1]], [.5, .5])
    engine = EndpointEnrichment(model, np.arange(2), profile)
    totals, expectation = np.zeros(3), 0.
    for pattern in product(range(2), repeat=2):
        tips = {leaf: np.eye(2)[v] for leaf, v in zip((1, 2), pattern)}
        logs = engine.log_likelihoods(tips, [1, 2], [0], [1])
        # Independent sum over root and per-branch participation.
        explicit = []
        for atom in [dict(multiplier=1, participation=0)] + profile['atoms']:
            kernels = []
            for child in (1, 2):
                base = model.transition(child, 0)
                tilted = base.copy()
                if atom['multiplier'] is None:
                    tilted[0] = [0, 1]
                else:
                    tilted[0, 1] *= atom['multiplier']
                    tilted /= tilted.sum(axis=1, keepdims=True)
                kernels.append(atom['participation'] * tilted + (1 - atom['participation']) * base)
            explicit.append(sum(.5 * kernels[0][root, pattern[0]] * kernels[1][root, pattern[1]]
                                for root in (0, 1)))
        np.testing.assert_allclose(np.exp(logs), explicit, rtol=1e-13)
        totals += np.exp(logs)
        _, log_e = engine.test(tips, [1, 2], [0], [1])
        expectation += np.exp(logs[0] + log_e)
    np.testing.assert_allclose(totals, 1)
    assert expectation == pytest.approx(1)
    for bad in [
        dict(version=2, atoms=profile['atoms']),
        dict(version=1, atoms=[]),
        dict(version=1, atoms=[dict(multiplier=0, participation=1, weight=1)]),
        dict(version=1, atoms=[dict(multiplier=2, participation=float('nan'), weight=1)]),
        dict(version=1, atoms=[dict(multiplier=2, participation=1, weight=.5)]),
    ]:
        with pytest.raises(ValueError):
            EndpointEnrichment(model, np.arange(2), bad)
    with pytest.raises(ValueError, match='requires'):
        validate_options(dict(scan_analytic_profile=str(path)))
