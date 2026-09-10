import itertools
from pathlib import Path
import runpy
import weakref

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm

from csubst import ete, scan_ctmc, tree


def fixture(length=.7, sites=1):
    tr = tree.add_numerical_node_labels(ete.PhyloNode(f"((A:{length},B:{length})X:{length},C:{length})R;", format=1))
    labels = {x.name: int(ete.get_prop(x, "numerical_label")) for x in tr.traverse()}
    state = np.zeros((len(labels), sites, 2))
    for name, value in [('A', 0), ('B', 1), ('C', 1)]:
        state[labels[name], :, value] = 1
    q = np.array([[-1., 1.], [1., -1.]])
    return tr, labels, state, q


def test_joint_equals_enumeration_and_not_marginal_product():
    tr, ids, states, q = fixture()
    posterior, tensor = scan_ctmc.infer(tr, states, q, [.5, .5], [0, 1])
    p = expm(q * .7)
    joint = np.zeros((2, 2))
    for root, internal in itertools.product(range(2), repeat=2):
        joint[root, internal] = .5*p[root, internal]*p[internal, 0]*p[internal, 1]*p[root, 1]
    joint /= joint.sum()
    np.testing.assert_allclose(posterior[ids['R'], 0], joint.sum(axis=1))
    np.testing.assert_allclose(posterior[ids['X'], 0], joint.sum(axis=0))
    assert tensor[ids['X'], 0, 0, 0, 1] == pytest.approx(joint[0, 1])
    assert abs(joint[0, 1] - joint.sum(axis=1)[0]*joint.sum(axis=0)[1]) > .01


@pytest.mark.parametrize('length', [0., 1e-8, .7, 10., 100.])
def test_bridge_against_independent_quadrature(length):
    tr, ids, states, q = fixture(length)
    if length == 0:
        states[:] = 0  # All missing: stationary unconditioned process.
    post, tensor = scan_ctmc.infer(tr, states, q, [.5, .5], [0, 1], 'bridge')
    if length == 0:
        assert tensor.sum() == 0
        return
    a = post[ids['R'], 0] / (expm(q * length) @ (np.ones(2)*.25))
    # Test independent Frechet computation for arbitrary boundaries.
    left = np.array([[.3, .7]])
    right = np.array([[.9, .1]])
    actual = scan_ctmc.bridge_kernel(q, length, left, right)[0]
    for i, j in [(0, 1), (1, 0)]:
        expected = quad(lambda t: (left[0] @ expm(q*t))[i] * q[i, j] * (expm(q*(length-t)) @ right[0])[j], 0, length)[0]
        assert actual[i, j] == pytest.approx(expected, abs=1e-11)
    assert np.isfinite(a).all()
    if length >= 10:
        assert tensor[ids['X']].sum() > 1


def test_bridge_spectral_matches_direct_frechet(monkeypatch):
    tr, ids, states, q = fixture(.9, 3)
    _, spectral = scan_ctmc.infer(tr, states, q, [.5, .5], [0, 1], 'bridge')
    # Force the stable direct path by a tiny numerical perturbation that flags it.
    original = scan_ctmc._integrals
    monkeypatch.setattr(scan_ctmc, '_integrals', lambda values, length: -original(values, length))
    _, direct = scan_ctmc.infer(tr, states, q, [.5, .5], [0, 1], 'bridge')
    np.testing.assert_allclose(spectral, direct, atol=1e-12)


def test_missing_data_and_group_aggregation():
    tr, ids, states, q = fixture(3.)
    states[:] = 0
    posterior, tensor = scan_ctmc.infer(tr, states, q, [.5, .5], [0, 1], 'bridge')
    np.testing.assert_allclose(posterior, .5)
    for name in ['A', 'B', 'C', 'X']:
        assert tensor[ids[name]].sum() == pytest.approx(3.)
    _, grouped = scan_ctmc.infer(tr, states, q, [.5, .5], [0, 0], 'bridge')
    assert grouped.sum() == 0


def test_parametric_bootstrap_repeats_asr_and_candidate_selection(monkeypatch):
    _toy_scan_context = runpy.run_path(str(Path(__file__).with_name("test_scan_permutations.py")))["_toy_scan_context"]
    from csubst import substitution_scan
    g, _ = _toy_scan_context()
    g.update(scan_rate_exposure='endpoint', scan_observation='joint',
             state_cdn=g['state_nsy'].copy(),
             instantaneous_codon_rate_matrix=np.array([[-1., 1.], [1., -1.]]),
             equilibrium_frequency=np.array([.5, .5]), substitution_model='GY',
             nonsynonymous_indices={'A': [0], 'K': [1]}, iqtree_rate_values=np.ones(1),
             scan_n_permutations=3, scan_pvalue_calibration='parametric', scan_min_event_pp=.1)
    g['scan_tip_emissions'] = g['state_cdn'].copy()
    updated, tensor = scan_ctmc.prepare(g)
    prepare = scan_ctmc.prepare
    replicate_tensors = []
    def check_release(context):
        if replicate_tensors:
            assert replicate_tensors[-1]() is None
        result = prepare(context)
        replicate_tensors.append(weakref.ref(result[1]))
        return result
    monkeypatch.setattr(scan_ctmc, 'prepare', check_release)
    original = substitution_scan._scan_substitutions_core
    seen = []
    def wrapped(g, *args, **kwargs):
        seen.append(g['state_cdn'].copy())
        return original(g, *args, **kwargs)
    monkeypatch.setattr(substitution_scan, '_scan_substitutions_core', wrapped)
    frame, _ = substitution_scan.scan_substitutions(updated, tensor, tensor)
    assert len(seen) == 4  # observed plus each full rerun
    assert not np.array_equal(seen[0], seen[1])
    assert (frame['p_rate_enrichment_empirical_maxT'] >= .25).all()
    assert frame['p_rate_enrichment_empirical'].isna().all()
    assert set(frame['scan_observation_method']) == {'joint_endpoint_posterior'}


def test_bridge_exposure_unconditioned_expected_jumps():
    context = runpy.run_path(str(Path(__file__).with_name("test_scan_endpoint.py")))["context"]
    from csubst import scan_endpoint
    g, meta, groups = context((20., 40.))
    g['scan_observation'] = 'bridge'
    g['state_cdn'][:] = 1/3
    g['state_nsy'][:] = 1/3
    ctx = scan_endpoint.build_context(g, meta, groups)
    expected, _, _ = scan_endpoint.expected_events(ctx, g['state_cdn'], g['state_nsy'], 0, [0, 1, 2], [0, 1, 2])
    np.testing.assert_allclose(expected, [20., 40.])


def test_bridge_threshold_can_exceed_one():
    from csubst import substitution_scan
    assert substitution_scan.scan_event_threshold(dict(scan_observation='bridge', scan_min_event_count=2.)) == 2.
    with pytest.raises(ValueError):
        substitution_scan.scan_event_threshold(dict(scan_min_event_pp=2.))


def test_parametric_simulation_replays_partial_ambiguity_and_missingness():
    tr, ids, _, _ = fixture(.7, 20)
    emissions = np.zeros((len(ids), 20, 4))
    emissions[ids['A'], :, [0, 2]] = .5  # AAR, resolve purine/pyrimidine class only.
    emissions[ids['B'], :, 0] = 1
    q = (np.ones((4, 4)) - np.eye(4)*4) / 3
    g = dict(tree=tr, state_cdn=emissions, instantaneous_codon_rate_matrix=q,
             equilibrium_frequency=np.ones(4)/4, codon_orders=['AAA', 'AAC', 'AAG', 'AAT'])
    partitions = scan_ctmc.validate_parametric_inputs(g)
    simulated = scan_ctmc.simulate_tips(g, np.random.default_rng(109), partitions)['state_cdn']
    assert not simulated[ids['C']].any()
    assert ((simulated[ids['B']] > 0).sum(axis=1) == 1).all()
    allowed = {(0.5, 0., .5, 0.), (0., .5, 0., .5)}
    assert {tuple(row) for row in simulated[ids['A']]} == allowed


def test_asymmetric_bridge_spectral_matches_frechet(monkeypatch):
    tr, ids, states, _ = fixture(1.2, 3)
    q = np.array([[-.25, .25], [1., -1.]])
    _, spectral = scan_ctmc.infer(tr, states, q, [.8, .2], [0, 1], 'bridge')
    original = scan_ctmc._integrals
    monkeypatch.setattr(scan_ctmc, '_integrals', lambda values, length: -original(values, length))
    _, direct = scan_ctmc.infer(tr, states, q, [.8, .2], [0, 1], 'bridge')
    np.testing.assert_allclose(spectral, direct, atol=1e-12)


def test_bridge_plot_does_not_clip_mean_counts():
    import pandas as pd
    from csubst import site_tree_plot
    frame = pd.DataFrame({'codon_site_alignment': [1], 'N_sub_1': [2.5]})
    g = dict(branch_ids=[1], scan_observation='bridge')
    values, _ = site_tree_plot.get_lineage_site_heatmap_values(frame, [{'site': 1}], g)
    assert values[0, 0] == 2.5
    values, _ = site_tree_plot.get_lineage_site_heatmap_values(frame, [{'site': 1}], dict(g, scan_observation='marginal'))
    assert values[0, 0] == 1.


def test_bridge_counts_intermediate_jumps_not_impossible_direct_events():
    tr = tree.add_numerical_node_labels(ete.PhyloNode('(A:1,B:1)R;', format=1))
    ids = {n.name: int(ete.get_prop(n, 'numerical_label')) for n in tr.traverse()}
    states = np.zeros((3, 1, 3))
    states[ids['A'], 0, 0] = 1
    states[ids['B'], 0, 2] = 1
    q = .75 * np.array([[-1., 1., 0.], [1., -2., 1.], [0., 1., -1.]])
    _, joint = scan_ctmc.infer(tr, states, q, np.ones(3)/3, [0, 1, 2], 'joint')
    summaries = {'synonymous_mask': np.array([[False, True, False], [True, False, False], [False, False, False]])}
    _, bridge = scan_ctmc.infer(tr, states, q, np.ones(3)/3, [0, 1, 2], 'bridge', summaries=summaries)
    assert joint[ids['B'], 0, 0, 0, 2] > 0
    assert bridge[ids['B'], 0, 0, 0, 2] == 0
    assert bridge[ids['B'], 0, 0, 0, 1] > 0
    assert bridge[ids['B'], 0, 0, 1, 2] > 0
    assert summaries['synonymous_counts'][ids['B']] > 0


def test_parametric_plot_filter_and_count_based_candidate_priority():
    import pandas as pd
    from csubst import substitution_scan
    frame = pd.DataFrame(dict(target_class=['fg', 'fg'], codon_site_alignment=[1, 1],
                              support_unit_count=[2, 2], support_pp_sum=[np.nan, np.nan],
                              candidate_event_pp_sum=[np.nan, np.nan], support_mass_sum=[3., 4.],
                              candidate_event_mass_sum=[3., 4.], p_rate_enrichment_asymptotic=[.001, .05],
                              p_rate_enrichment_empirical_maxT=[.01, .2]))
    selected = substitution_scan._select_scan_plot_rows(frame)
    assert selected.iloc[0]['support_mass_sum'] == 4.
    filtered = substitution_scan.filter_scan_site_plot_candidates(
        frame, dict(scan_pvalue_calibration='parametric', scan_site_plot_filter='full_scan'))
    assert len(filtered) == 1
    assert filtered.iloc[0]['support_mass_sum'] == 3.


def test_joint_kernel_agrees_with_shared_endpoint_engine():
    from csubst.endpoint import EndpointModel
    tr, ids, states, q = fixture(.4, 3)
    states[ids['A'], 1] = 0  # missing tip is integrated by both engines
    parents = np.full(len(ids), -1, dtype=int)
    lengths = np.zeros(len(ids))
    tips = {}
    for node in tr.traverse():
        index = ids[node.name]
        if node.up is not None:
            parents[index] = ids[node.up.name]
            lengths[index] = node.dist
        if ete.is_leaf(node):
            tips[index] = states[index].copy()
            tips[index][tips[index].sum(axis=1) == 0] = 1
    posterior, tensor = scan_ctmc.infer(tr, states, q, [.5, .5], [0, 1])
    for record in EndpointModel(parents, lengths, q, [.5, .5]).iter_blocks(tips, block_size=2):
        np.testing.assert_allclose(record.node, posterior[record.child, record.start:record.stop], atol=1e-12)
        if record.joint is not None:
            record.joint[:, np.arange(2), np.arange(2)] = 0
            np.testing.assert_allclose(record.joint, tensor[record.child, record.start:record.stop, 0], atol=1e-12)
