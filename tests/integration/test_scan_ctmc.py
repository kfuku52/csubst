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


def test_missing_data_and_group_aggregation():
    tr, ids, states, q = fixture(3.)
    states[:] = 0
    posterior, tensor = scan_ctmc.infer(tr, states, q, [.5, .5], [0, 1], 'bridge')
    np.testing.assert_allclose(posterior, .5)
    for name in ['A', 'B', 'C', 'X']:
        assert tensor[ids[name]].sum() == pytest.approx(3.)
    _, grouped = scan_ctmc.infer(tr, states, q, [.5, .5], [0, 0], 'bridge')
    assert grouped.sum() == 0


def test_bridge_site_blocks_match_dense_with_masks_and_synonymous_counts():
    from csubst import site_storage

    tr, ids, states, q = fixture(.25, 5)
    states[ids['A'], 1] = 0.
    states[ids['B'], 2] = [.25, .75]
    states[ids['C'], 3] = [1., 0.]
    eligible = np.ones(states.shape[:2], dtype=bool)
    eligible[ids['A'], 1] = False
    mask = np.array([[False, False], [True, False]])
    dense_summary = dict(eligible=eligible, synonymous_mask=mask)
    disk_summary = dict(dense_summary)
    post, dense = scan_ctmc.infer(tr, states, q, [.5, .5], [0, 1], 'bridge',
                                  block_size=5, summaries=dense_summary)
    actual_post, disk = scan_ctmc.infer(tr, states, q, [.5, .5], [0, 1], 'bridge',
                                         block_size=2, summaries=disk_summary, disk_backed=True)
    assert isinstance(disk, site_storage.SiteEventTensor)
    np.testing.assert_allclose(actual_post, post, atol=1e-14, rtol=1e-13)
    np.testing.assert_allclose(disk.read_sites(0, 5).swapaxes(0, 1), dense, atol=1e-14, rtol=1e-13)
    np.testing.assert_allclose(disk.branch_site, dense.sum(axis=(2, 3, 4)))
    np.testing.assert_allclose(disk_summary['synonymous_counts'], dense_summary['synonymous_counts'])
    assert not disk.read_site(1)[ids['A']].any()


@pytest.mark.parametrize('disk_backed', [False, True])
def test_parametric_bootstrap_repeats_asr_and_candidate_selection(monkeypatch, disk_backed):
    if disk_backed:
        monkeypatch.setattr(scan_ctmc, '_SCAN_IN_MEMORY_BYTES', 0)
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
    models = []
    def check_release(context, *, endpoint_model):
        if replicate_tensors:
            assert replicate_tensors[-1]() is None
        models.append(endpoint_model)
        result = prepare(context, endpoint_model=endpoint_model)
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
    assert len(models) == 3 and models[0] is not None and all(model is models[0] for model in models)
    assert not np.array_equal(seen[0], seen[1])
    assert (frame['p_rate_enrichment_empirical_maxT'] >= .25).all()
    assert frame['p_rate_enrichment_empirical'].isna().all()
    assert set(frame['scan_observation_method']) == {'joint_endpoint_posterior'}
    assert updated['scan_calibration_diagnostics']['endpoint_model']['rates'] == [1.]
    assert 'rate_categories_and_priors' in updated['scan_calibration_diagnostics']['fixed']


@pytest.mark.parametrize("observation", ["joint", "bridge"])
@pytest.mark.parametrize("unit_mode", ["lineage", "stem", "clade"])
def test_inferred_scan_lineage_support_groups_paralogs_in_each_observation_mode(observation, unit_mode):
    from scan_fixtures import make_scan_context
    from csubst import foreground, substitution_scan
    g, _ = make_scan_context()
    g["fg_df"]["trait"] = 7
    foreground.get_foreground_ids(g, write=False)
    g.update(scan_rate_exposure="endpoint", scan_observation=observation, scan_unit_mode=unit_mode,
             scan_min_support="1", scan_min_event_pp=0.1, scan_min_event_count=0.1,
             state_cdn=g["state_nsy"].copy(),
             instantaneous_codon_rate_matrix=np.array([[-1., 1.], [1., -1.]]),
             equilibrium_frequency=np.array([.5, .5]), substitution_model="GY",
             nonsynonymous_indices={"A": [0], "K": [1]}, iqtree_rate_values=np.ones(1))
    for node in g["tree"].traverse():
        if not ete.is_leaf(node):
            g["state_cdn"][int(ete.get_prop(node, "numerical_label"))] = 0
    updated, tensor = scan_ctmc.prepare(g)
    scan, _ = substitution_scan.scan_substitutions(updated, tensor)
    # A bridge includes reversions along the path even when both observed
    # foreground tips end in K; joint endpoints only report the change to K.
    expected_targets = {"A", "K"} if observation == "bridge" else {"K"}
    assert len(scan) == len(expected_targets)
    assert set(scan["to_state"]) == expected_targets
    assert (scan["support_unit_count"] == (1 if unit_mode == "lineage" else 2)).all()
    assert (scan["lineage_total"] == 1).all()
    assert (scan["support_lineage_count"] == 1).all()
    assert (scan["support_lineage_fraction"] == 1.0).all()
    assert (scan["support_lineage_ids"] == "7").all()


def test_prepare_joint_accepts_unloaded_internal_states_and_shares_identical_observation_alphabets():
    from scan_fixtures import make_scan_context
    g, _ = make_scan_context()
    g.update(scan_rate_exposure='endpoint', scan_observation='joint',
             state_cdn=g['state_nsy'].copy(),
             instantaneous_codon_rate_matrix=np.array([[-1., 1.], [1., -1.]]),
             equilibrium_frequency=np.array([.5, .5]), substitution_model='GY',
             nonsynonymous_indices={'A': [0], 'K': [1]}, iqtree_rate_values=np.ones(1))
    for node in g['tree'].traverse():
        if not ete.is_leaf(node):
            i = int(ete.get_prop(node, 'numerical_label'))
            g['state_cdn'][i] = 0
            g['state_nsy'][i] = 0
    updated, tensor = scan_ctmc.prepare(g)
    assert updated['scan_observed_state_pep'] is updated['scan_observed_state_nsy']
    np.testing.assert_allclose(updated['state_cdn'].sum(axis=2), 1.)
    assert tensor.sum() > 0


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
             equilibrium_frequency=np.ones(4)/4, codon_orders=['AAA', 'AAC', 'AAG', 'AAT'], substitution_model='GY')
    partitions = scan_ctmc.validate_parametric_inputs(g)
    simulated = scan_ctmc.simulate_tips(g, np.random.default_rng(109), partitions)['state_cdn']
    assert not simulated[ids['C']].any()
    assert ((simulated[ids['B']] > 0).sum(axis=1) == 1).all()
    allowed = {(0.5, 0., .5, 0.), (0., .5, 0., .5)}
    assert {tuple(row) for row in simulated[ids['A']]} == allowed


def test_parametric_simulation_never_draws_an_impossible_state_at_cdf_boundary():
    tr, ids, states, q = fixture(0., 3)
    g = dict(tree=tr, state_cdn=states, instantaneous_codon_rate_matrix=q,
             equilibrium_frequency=np.array([.5, .5]), substitution_model='GY')

    class BoundaryRng:
        def choice(self, n, size, p):
            return np.ones(size, dtype=np.int64)

        def random(self, size):
            return np.zeros(size)

    simulated = scan_ctmc.simulate_tips(g, BoundaryRng(), {})['state_cdn']
    np.testing.assert_array_equal(simulated[[ids['A'], ids['B'], ids['C']], :, 0], 0.)
    np.testing.assert_array_equal(simulated[[ids['A'], ids['B'], ids['C']], :, 1], 1.)


def test_quadrature_bridge_contraction_matches_direct_frechet():
    rng = np.random.default_rng(837)
    k = 61
    pi = rng.dirichlet(np.ones(k) * 3)
    rates = rng.random((k, k))
    q = (rates + rates.T) * pi
    np.fill_diagonal(q, 0.)
    np.fill_diagonal(q, -q.sum(axis=1))
    t = .7
    values, v, w = scan_ctmc._bridge_spectral(q, pi)
    kernel = scan_ctmc._bridge_integral_kernel(values, t)
    assert isinstance(kernel, tuple)
    a, b = (rng.dirichlet(np.ones(k), size=3) for _ in range(2))
    evidence = ((a @ expm(q * t)) * b).sum(axis=1)
    actual = scan_ctmc._conditioned_bridge_mass(q, t, a, b, evidence, v, w, kernel, np.arange(k))
    expected = scan_ctmc.bridge_kernel(q, t, a, b) / evidence[:, None, None]
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=1e-12)


def test_short_bridge_retains_rare_return_jumps_in_a_sparse_chain():
    n = 8
    edges = np.arange(n - 1)
    rates = .5 + edges / 8
    q = np.zeros((n, n))
    q[edges, edges + 1] = q[edges + 1, edges] = rates
    np.fill_diagonal(q, -q.sum(axis=1))
    length = 1e-8
    a, b = np.eye(n)[[0]], np.eye(n)[[-1]]
    evidence = expm(q * length)[[0], -1]
    values, v, w = scan_ctmc._bridge_spectral(q, np.full(n, 1/n))
    kernel = scan_ctmc._bridge_integral_kernel(values, length)
    mass = scan_ctmc._conditioned_bridge_mass(q, length, a, b, evidence, v, w, kernel, np.arange(n))
    # Crossing the chain takes n-1 forward jumps. A return on edge i adds
    # its two jump rates and two time factors to the leading path term.
    expected_reverse = (length * rates)**2 / (n * (n + 1))
    np.testing.assert_allclose(mass[0, edges + 1, edges], expected_reverse, rtol=2e-7, atol=0.)
    np.testing.assert_allclose(mass[0, edges, edges + 1], 1., rtol=0., atol=1e-14)


@pytest.mark.parametrize('rates,weights', [((1.,), (1.,)), ((1., 1.), (.5, .5))])
def test_bridge_tree_resolves_short_branches_with_many_required_jumps(rates, weights):
    n = 20
    first, second = 1e-4, 2e-4
    tr = tree.add_numerical_node_labels(ete.PhyloNode(f'(L:{first},R:{second})root;', format=1))
    ids = {node.name: int(ete.get_prop(node, 'numerical_label')) for node in tr.traverse()}
    tips = np.zeros((len(ids), 1, n))
    tips[ids['L'], 0, 0] = tips[ids['R'], 0, -1] = 1.
    edges = np.arange(n - 1)
    q = np.zeros((n, n))
    q[edges, edges + 1] = q[edges + 1, edges] = 1.
    moves = q / 2
    np.fill_diagonal(q, -q.sum(axis=1))
    walk = np.eye(n) + q / 2

    def path_moments(length):
        # Count moves directly in the embedded chain, carrying the first
        # moment along each path. The common exp(-2*length) cancels below.
        state = np.eye(n)[0]
        count = np.zeros(n)
        likelihood, mass = state.copy(), count.copy()
        coefficient = 1.
        for steps in range(1, 81):
            count = count @ walk + state @ moves
            state = state @ walk
            coefficient *= 2 * length / steps
            likelihood += coefficient * state
            mass += coefficient * count
        return likelihood, mass

    left, _ = path_moments(first)
    right, _ = path_moments(second)
    root = left * right[::-1]
    likelihood, mass = path_moments(first + second)
    posterior, events = scan_ctmc.infer(tr, tips, q, np.full(n, 1/n), np.arange(n), 'bridge',
                                       rates=rates, weights=weights)
    np.testing.assert_allclose(posterior[ids['root'], 0], root / root.sum(), rtol=2e-13, atol=0.)
    # Reversibility makes the two-edge path equivalent to one edge whose
    # length is the sum, including the number of state-changing jumps.
    assert events.sum() == pytest.approx(mass[-1] / likelihood[-1], rel=5e-14)


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


@pytest.mark.parametrize('native', [True, False])
def test_grouped_joint_and_synonymous_counts_match_full_codon_events(monkeypatch, native):
    from csubst import endpoint, substitution_sparse
    if not native:
        monkeypatch.setattr(substitution_sparse, 'substitution_sparse_cy', None)
    tr, ids, _, _ = fixture(.6, 3)
    states = np.zeros((len(ids), 3, 4))
    states[ids['A'], :, [0, 1]] = .5
    states[ids['B'], :, 2] = 1
    states[ids['C'], :, 3] = 1
    states[ids['C'], 1] = 0
    q = (np.ones((4, 4)) - np.eye(4) * 4) / 3
    groups = np.array([0, 0, 1, 2])
    mapping = np.eye(3)[groups]
    aa = np.array([0, 1, 1, 2])
    mask = (aa[:, None] == aa[None, :]) & ~np.eye(4, dtype=bool)
    valid = np.ones(states.shape[:2], dtype=bool)
    valid[ids['B'], 2] = False
    summaries = dict(synonymous_mask=mask, eligible=valid)
    rates, weights = [.25, 1.75], [.5, .5]
    post, tensor = scan_ctmc.infer(tr, states, q, np.ones(4) / 4, groups,
                                   block_size=2, summaries=summaries, rates=rates, weights=weights)
    parents = np.full(len(ids), -1, dtype=int)
    lengths = np.zeros(len(ids))
    tips = {}
    for node in tr.traverse():
        i = ids[node.name]
        if node.up is not None:
            parents[i], lengths[i] = ids[node.up.name], node.dist
        if ete.is_leaf(node):
            tips[i] = states[i].copy()
            tips[i][tips[i].sum(axis=1) == 0] = 1
    expected_syn = np.zeros(len(ids))
    model = endpoint.EndpointModel(parents, lengths, q, np.ones(4) / 4, rates, weights)
    for record in model.iter_blocks(tips, block_size=2):
        sl = slice(record.start, record.stop)
        np.testing.assert_allclose(post[record.child, sl], record.node, atol=1e-14)
        if record.joint is None:
            continue
        joint = record.joint * valid[record.child, sl, None, None]
        grouped = mapping.T @ joint @ mapping
        grouped[:, np.arange(3), np.arange(3)] = 0
        np.testing.assert_allclose(tensor[record.child, sl, 0], grouped, atol=1e-14)
        expected_syn[record.child] += (joint * mask).sum()
    np.testing.assert_allclose(summaries['synonymous_counts'], expected_syn, atol=1e-14)


@pytest.mark.parametrize('block_size', [1, 3, 64])
@pytest.mark.parametrize('disk_backed', [False, True])
def test_mixture_joint_and_exposure_match_complete_enumeration(tmp_path, block_size, disk_backed):
    import pandas as pd
    from csubst import scan_endpoint
    tr, ids, states, q = fixture(.7, 3)
    # A zero-rate component has zero likelihood at discordant sites, but survives
    # at a completely missing site. Use varying posterior category weights.
    states[:, 1] = 0
    states[ids['B'], 2] = [1, 0]
    rates = np.array([0., .2, 2.])
    priors = np.array([.15, .35, .5])
    summaries = {'synonymous_mask': np.zeros((2, 2))}
    post, tensor = scan_ctmc.infer(tr, states, q, [.5, .5], [0, 1],
                                   block_size=block_size, summaries=summaries, rates=rates, weights=priors,
                                   disk_backed=disk_backed)
    report = tmp_path / 'mixture.iqtree'
    report.write_text('Category Relative_rate Proportion\n1 0 .15\n2 .2 .35\n3 2 .5\n')
    g = dict(scan_rate_length='raw', scan_observation='joint', substitution_model='GY+R3',
             state_cdn=post, state_nsy=post.copy(), instantaneous_codon_rate_matrix=q,
             equilibrium_frequency=np.array([.5, .5]), path_iqtree_iqtree=str(report),
             scan_category_states=summaries['category_states'])
    meta = pd.DataFrame(dict(branch_id=[ids['X']], parent_id=[ids['R']], raw_length=[.7]))
    ctx = scan_endpoint.build_context(g, meta, [0, 1])
    if disk_backed:
        import pickle
        from csubst import substitution_scan
        worker_g, worker_static, paths = substitution_scan._pack_scan_worker_context(
            g, {'q_context': {'state_cdn': post, 'endpoint_context': ctx}})
        worker_g, worker_static = pickle.loads(pickle.dumps((worker_g, worker_static)))
        _, worker_static = substitution_scan._unpack_scan_worker_context(worker_g, worker_static)
        ctx = worker_static['q_context']['endpoint_context']
        assert paths == []  # Tiny ordinary states; site files are borrowed.
    for site in range(3):
        joint = np.zeros((3, 2, 2))
        for c, rate in enumerate(rates):
            p = expm(q * rate * .7)
            for r, x, a, b, leaf_c in itertools.product(range(2), repeat=5):
                mass = priors[c] * .5 * p[r,x] * p[x,a] * p[x,b] * p[r,leaf_c]
                for name, value in [('A', a), ('B', b), ('C', leaf_c)]:
                    emission = states[ids[name], site]
                    mass *= emission[value] if emission.sum() else 1
                joint[c, r, x] += mass
        joint /= joint.sum()
        value = tensor.read_site(site)[ids['X'], 0, 0, 1] if disk_backed else tensor[ids['X'], site, 0, 0, 1]
        categories = summaries['category_states']
        category_value = categories.read_site(site)[:, ids['R']] if disk_backed else categories[:, ids['R'], site]
        np.testing.assert_allclose(value, joint[:, 0, 1].sum(), atol=1e-14)
        np.testing.assert_allclose(category_value, joint.sum(axis=2), atol=1e-14)
        expected = sum(joint[c, 0].sum() * expm(q * rate * .7)[0, 1] for c, rate in enumerate(rates))
        actual, missing, _ = scan_endpoint.expected_events(ctx, post, post, site, [0], [1])
        assert not missing.any()
        np.testing.assert_allclose(actual, [expected], atol=1e-14)


@pytest.mark.parametrize('disk_backed', [False, True])
def test_bridge_rate_mixture_matches_exact_hidden_states_and_jump_integrals(tmp_path, disk_backed):
    import pandas as pd
    from csubst import scan_endpoint

    tr, ids, states, _ = fixture(.7, 3)
    states[ids['A'], 1] = [0., 1.]
    states[ids['C'], 2] = 0.
    states[ids['B'], 2] = [.3, .7]
    pi = np.array([.3, .7])
    q = np.array([[-1., 1.], [1., -1.]]) / (2 * pi[:, None])
    rates, priors = np.array([0., .5, 2.5]), np.array([.15, .6, .25])
    synonymous_mask = np.array([[False, False], [True, False]])
    summaries = {'synonymous_mask': synonymous_mask}
    posterior, events = scan_ctmc.infer(tr, states, q, pi, [0, 1], mode='bridge', summaries=summaries,
                                        block_size=2, rates=rates, weights=priors, disk_backed=disk_backed)
    report = tmp_path / 'fit.iqtree'
    report.write_text('Category Relative_rate Proportion\n1 0 .15\n2 .5 .6\n3 2.5 .25\n')
    g = dict(tree=tr, substitution_model='GY+FQ+R3', scan_observation='bridge', scan_rate_length='raw',
             state_cdn=posterior, state_nsy=posterior, equilibrium_frequency=pi, instantaneous_codon_rate_matrix=q,
             scan_category_states=summaries['category_states'], path_iqtree_iqtree=str(report))
    branch_meta = pd.DataFrame({'branch_id': [ids['X']], 'parent_id': [ids['R']], 'raw_length': [.7]})
    context = scan_endpoint.build_context(g, branch_meta, np.arange(2))

    def transition(t):
        return pi + np.exp(q.trace() * t) * (np.eye(2) - pi)

    expected_synonymous = 0.
    for site in range(3):
        likelihood = 0.
        root_mass = np.zeros((len(rates), 2))
        jump_mass = np.zeros(2)
        expected_exposure = 0.
        for category, (rate, prior) in enumerate(zip(rates, priors)):
            t = .7 * rate
            p = transition(t)
            jumps = np.array([
                [[q[src, dst] * quad(lambda u: transition(u)[a, src] * transition(t-u)[dst, b], 0., t)[0]
                  for b in range(2)] for a in range(2)]
                for src, dst in ((0, 1), (1, 0))
            ])
            root_opportunity = np.array([q[0, 1] * quad(lambda u: transition(u)[r, 0], 0., t)[0]
                                         for r in range(2)])
            for r, x, a, b, c in itertools.product(range(2), repeat=5):
                partial = prior * pi[r] * p[x, a] * p[x, b] * p[r, c]
                for name, assignment in (('A', a), ('B', b), ('C', c)):
                    emission = states[ids[name], site]
                    partial *= emission[assignment] if emission.any() else 1.
                full = partial * p[r, x]
                likelihood += full
                root_mass[category, r] += full
                jump_mass += partial * jumps[:, r, x]
                expected_exposure += full * root_opportunity[r]
        selected = events.read_site(site)[ids['X'], 0] if disk_backed else events[ids['X'], site, 0]
        np.testing.assert_allclose(selected[[0, 1], [1, 0]], jump_mass / likelihood, atol=2e-14)
        expected_synonymous += jump_mass[1] / likelihood
        categories = summaries['category_states']
        parent = categories.read_site(site)[:, ids['R']] if disk_backed else categories[:, ids['R'], site]
        np.testing.assert_allclose(parent, root_mass / likelihood, atol=2e-14)
        exposure, missing, _ = scan_endpoint.expected_events(context, posterior, posterior, site, [0], [1])
        assert not missing.any()
        np.testing.assert_allclose(exposure, [expected_exposure / likelihood], atol=2e-14)
    assert summaries['synonymous_counts'][ids['X']] == pytest.approx(expected_synonymous, abs=2e-14)
