import copy
from contextlib import ExitStack

import numpy as np
import pytest

from csubst import endpoint_io, ete, omega, parser_misc, substitution, substitution_scan, tree, variant_effect


def toy_context(tmp_path, structural=False):
    tr = tree.add_numerical_node_labels(ete.PhyloNode('((A:1,B:1)U:0,C:1)R;', format=1))
    ids = {node.name: int(ete.get_prop(node, 'numerical_label')) for node in tr.traverse()}
    state = np.zeros((5, 3, 3))
    state[ids['A']] = [[1, 0, 0], [0, 0, 1], [0, 0, 0]]
    state[ids['B']] = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    state[ids['U']] = state[ids['R']] = 1 / 3
    report = tmp_path / 'model.iqtree'
    report.write_text('Model of rate heterogeneity: Uniform\n')
    pep = np.stack([state[:, :, :2].sum(2), state[:, :, 2]], axis=2)
    g = dict(tree=tr, state_cdn=state, state_pep=pep, state_nsy=pep,
             substitution_posterior='joint', endpoint_block_size=2,
             substitution_model='GY+F', path_iqtree_iqtree=str(report),
             instantaneous_codon_rate_matrix=np.ones((3, 3)) / 2 - np.eye(3) * 1.5,
             equilibrium_frequency=np.ones(3) / 3, amino_acid_orders=['X', 'Y'],
             nonsyn_state_orders=['X', 'Y'], synonymous_indices={'X': [0, 1], 'Y': [2]},
             nonsynonymous_indices={'X': [0, 1], 'Y': [2]}, max_synonymous_size=2,
             nonsyn_recode='no', expectation_method='codon_model', output_stats=['any2any', 'spe2any', 'any2spe', 'spe2spe'],
             ml_anc=False, min_sub_pp=0, float_type=np.float64, float_tol=1e-12, threads=1)
    if structural:
        g['nonsyn_recode'] = '3di20'
        g['sa_asr_mode'] = 'direct'
        g['sa_iqtree_model'] = 'GTR'
        states = np.zeros((5, 3, 20))
        states[:, :, :3] = state
        g['state_nsy'] = states
        lengths = np.zeros(5)
        for node in tr.traverse():
            if not ete.is_root(node):
                lengths[int(ete.get_prop(node, 'numerical_label'))] = node.dist
        q = np.zeros((20, 20))
        q[:3, :3] = g['instantaneous_codon_rate_matrix']
        g.update({'3di_q': q, '3di_pi': np.r_[np.ones(3) / 3, np.zeros(17)],
                  '3di_rates': np.ones(3), '3di_branch_lengths': lengths,
                  '3di_orders': np.array(list('ACDEFGHIKLMNPQRSTVWY')),
                  '3di_tip_invariant_mask': np.array([False, True, True])})
    return g, ids


@pytest.mark.parametrize('structural', [False, True])
def test_zero_branch_missing_sites_and_expected_projections(tmp_path, structural):
    g, ids = toy_context(tmp_path, structural)
    endpoint_io.prepare(g)
    for kind, state, mode in [('S', g['state_cdn'], 'syn'), ('N', g['state_nsy'], 'asis')]:
        tensor = substitution.get_substitution_tensor(state, mode=mode, g=g)
        values = tensor.to_dense()
        assert not np.any(values[ids['U']])
        assert not np.any(values[:, 2])  # wholly missing site
        assert not np.any(values[ids['C']])  # entirely unobserved leaf
        reducer = omega._get_fused_expected_sparse_reducer(
            g, 'cdn' if kind == 'S' else 'nsy', endpoint_io.BASE_STATS)
        shape = tensor.shape
        # Materialize independently from the complete pair projection, then
        # require all reduced channels to have the same definition.
        matrix = reducer['projections']['spe2spe']
        expected = substitution.substitution_sparse.SparseSubstitutionTensor(shape, float, matrix=matrix)
        assert not np.any(expected.to_dense()[ids['U']])
        assert not np.any(expected.to_dense()[:, 2])
        assert reducer['total'] == pytest.approx(expected.matrix.sum())
        for stat in endpoint_io.BASE_STATS:
            np.testing.assert_allclose(reducer['projections'][stat].toarray(),
                                       substitution._build_sparse_cb_projection(expected, stat).toarray(), atol=1e-14)
    assert g['state_cdn'][ids['R'], 0].sum() == pytest.approx(1)


def test_3di_changes_do_not_change_codon_S_or_AA(tmp_path):
    first, ids = toy_context(tmp_path, structural=True)
    second = copy.deepcopy(first)
    second['state_nsy'][ids['B'], 0] = 0
    second['state_nsy'][ids['B'], 0, 2] = 1
    endpoint_io.prepare(first)
    endpoint_io.prepare(second)
    for kind in ('S', 'AA'):
        np.testing.assert_array_equal(first['_endpoint_tensors'][kind].to_dense(),
                                      second['_endpoint_tensors'][kind].to_dense())
    for stat in endpoint_io.BASE_STATS:
        np.testing.assert_array_equal(first['_endpoint_reducers']['S']['projections'][stat].toarray(),
                                      second['_endpoint_reducers']['S']['projections'][stat].toarray())
    assert not np.array_equal(first['_endpoint_tensors']['N'].to_dense(), second['_endpoint_tensors']['N'].to_dense())


def test_block_size_and_filtered_rebuild(tmp_path):
    first, _ = toy_context(tmp_path)
    second = copy.deepcopy(first)
    first['endpoint_block_size'] = 1
    second['endpoint_block_size'] = 64
    endpoint_io.prepare(first)
    endpoint_io.prepare(second)
    for kind in ('S', 'N'):
        np.testing.assert_allclose(first['_endpoint_tensors'][kind].to_dense(), second['_endpoint_tensors'][kind].to_dense())
    second.update(drop_invariant_tip_sites=True, drop_invariant_tip_sites_mode='zero_sub_mass',
                  num_input_site=3, site_index_alignment=np.arange(3))
    before = second['_endpoint_tensors']['N'].to_dense()
    parser_misc.drop_invariant_tip_sites(second)
    endpoint_io.prepare(second)
    np.testing.assert_allclose(second['_endpoint_tensors']['N'].to_dense(), before[:, :2], atol=1e-14)


def test_spool_supports_interleaved_branches_and_site_blocks():
    expected = np.arange(30.).reshape(2, 5, 3)
    with ExitStack() as stack:
        builder = endpoint_io._Spool(stack, 2, 15, np.float64)
        for start, stop in [(0, 2), (2, 5)]:
            for branch in (1, 0):
                builder.append(branch, expected[branch, start:stop], start, 5)
        np.testing.assert_array_equal(builder.finish().toarray(), expected.transpose(0, 2, 1).reshape(2, 15))


def test_rate_mixture_is_not_mean_site_rate():
    rates, weights = endpoint_io.read_rate_mixture('Category  Relative_rate  Proportion\n 1 0.1 0.25\n 2 1.3 0.75\n\n')
    np.testing.assert_array_equal(rates, [.1, 1.3])
    np.testing.assert_array_equal(weights, [.25, .75])
    with pytest.raises(ValueError, match='rate-category'):
        endpoint_io.read_rate_mixture('Model of rate heterogeneity: unknown')


def test_scan_and_vesm_use_joint_values_not_separate_outer_products(tmp_path):
    g, ids = toy_context(tmp_path)
    endpoint_io.prepare(g)
    tensor = g['_endpoint_tensors']['N']
    scan_events = substitution_scan.extract_atomic_events(tensor, min_event_pp=1e-12)
    assert ids['U'] not in set(scan_events['branch_id'])
    g['vep_min_event_pp'] = 1e-12
    g['site_index_alignment'] = np.arange(3)
    g['_vep_ancestral_contexts'] = {
        'num_alignment_site': 3,
        'contexts': {i: {'sequence': 'XXX', 'presence': np.ones(3, dtype=bool),
                          'alignment_to_ungapped': np.arange(3)} for i in ids.values()},
    }
    events = variant_effect.extract_atomic_aa_events(g, [ids['U'], ids['A']])
    assert ids['U'] not in set(events['branch_id'])
    assert not events.empty
    dense = tensor.to_dense()
    order = {name: i for i, name in enumerate(g['amino_acid_orders'])}
    for row in events.itertuples():
        assert row.event_pp == pytest.approx(
            dense[row.branch_id, row.site, 0, order[row.from_aa], order[row.to_aa]])


def test_legacy_expected_helpers_cannot_silently_use_marginals_in_joint_mode():
    with pytest.raises(ValueError, match='both state axes'):
        omega.get_exp_state({'substitution_posterior': 'joint'}, 'cdn')
    with pytest.raises(ValueError, match='sparse reducer'):
        omega._get_fused_expected_sparse_substitution_tensor({'substitution_posterior': 'joint'}, 'cdn')


@pytest.mark.parametrize('model_name', ['MG+F3X4', 'GY+C20', 'ECMK07+ASC', 'GY+H4', 'GY+unknown'])
def test_unverified_model_modifiers_fail_closed(tmp_path, model_name):
    g, _ = toy_context(tmp_path)
    g['substitution_model'] = model_name
    with pytest.raises(ValueError, match='verified likelihood'):
        endpoint_io.prepare(g)
    assert '_endpoint_tensors' not in g


def test_boolean_state_storage_cannot_turn_probabilities_into_true_values(tmp_path):
    g, _ = toy_context(tmp_path)
    g['state_cdn'] = g['state_cdn'].astype(bool)
    with pytest.raises(ValueError, match='floating-point'):
        endpoint_io.prepare(g)


@pytest.mark.parametrize('extra', [{'ml_anc': True}, {'endpoint_block_size': 0},
                                  {'nonsyn_recode': '3di20', 'sa_asr_mode': 'translate'}])
def test_unsupported_options_are_explicit(extra):
    with pytest.raises(ValueError):
        endpoint_io.validate_options(dict(substitution_posterior='joint', **extra))


@pytest.mark.parametrize('structural', [False, True])
@pytest.mark.parametrize('sitewise', [False, True])
@pytest.mark.parametrize('block_size', [1, 2, 64])
def test_projected_search_matches_full_events(tmp_path, structural, sitewise, block_size):
    import pandas as pd
    from csubst.substitution_sparse import ProjectedSubstitutionTensor
    full, ids = toy_context(tmp_path, structural)
    full.update(output_stats=['any2any', 'spe2any', 'any2spe'], endpoint_block_size=block_size,
                num_node=5, b=sitewise)
    compact = copy.deepcopy(full)
    compact['subcommand'] = 'search'
    endpoint_io.prepare(full)
    endpoint_io.prepare(compact)
    pairs = np.array([[ids['A'], ids['B']], [ids['A'], ids['U']]], dtype=np.int64)
    for kind in ['N', 'S']:
        a, b = full['_endpoint_tensors'][kind], compact['_endpoint_tensors'][kind]
        assert isinstance(b, ProjectedSubstitutionTensor)
        np.testing.assert_allclose(substitution.get_branch_sub_counts(a), substitution.get_branch_sub_counts(b), atol=1e-14)
        np.testing.assert_allclose(substitution.get_site_sub_counts(a), substitution.get_site_sub_counts(b), atol=1e-14)
        assert substitution.get_total_substitution(a) == pytest.approx(substitution.get_total_substitution(b))
        for stat in compact['output_stats']:
            np.testing.assert_allclose(a.project(stat).toarray(), b.project(stat).toarray(), atol=1e-14)
            np.testing.assert_allclose(full['_endpoint_reducers'][kind]['projections'][stat].toarray(),
                                       compact['_endpoint_reducers'][kind]['projections'][stat].toarray(), atol=1e-14)
        pd.testing.assert_frame_equal(
            substitution.get_cb(pairs, a, full, 'OC' + kind, compact['output_stats']),
            substitution.get_cb(pairs, b, compact, 'OC' + kind, compact['output_stats']),
            check_exact=False, atol=1e-14, rtol=1e-12)
        pd.testing.assert_frame_equal(
            substitution.get_b(full, a, kind, sitewise and kind == 'N'),
            substitution.get_b(compact, b, kind, sitewise and kind == 'N'))
        if sitewise:
            for branch in range(5):
                for threshold in [0, 0.05, 0.5]:
                    aa = substitution._get_sparse_branch_sitewise_max_indices(a, branch, threshold)
                    bb = substitution._get_sparse_branch_sitewise_max_indices(b, branch, threshold)
                    for av, bv in zip(aa, bb):
                        np.testing.assert_array_equal(av, bv)
    pd.testing.assert_frame_equal(
        substitution.get_bs(full['_endpoint_tensors']['S'], full['_endpoint_tensors']['N']),
        substitution.get_bs(compact['_endpoint_tensors']['S'], compact['_endpoint_tensors']['N']))


@pytest.mark.parametrize('option,value', [('subcommand', 'sites'), ('min_sub_pp', .1),
                                        ('cs', True), ('cbs', True), ('asrv_report', True),
                                        ('calc_omega_pvalue', True), ('epistasis_requested', True),
                                        ('expectation_method', 'urn'), ('output_stats', ['spe2spe'])])
def test_full_event_consumers_keep_full_representation(tmp_path, option, value):
    from csubst.substitution_sparse import ProjectedSubstitutionTensor
    g, _ = toy_context(tmp_path)
    g.update(subcommand='search', output_stats=['any2any', 'any2spe'])
    g[option] = value
    endpoint_io.prepare(g)
    assert not isinstance(g['_endpoint_tensors']['N'], ProjectedSubstitutionTensor)


@pytest.mark.native
def test_native_syn_projection_matches_numpy(tmp_path, monkeypatch):
    cy = endpoint_io.substitution_sparse.substitution_sparse_cy
    if cy is None or not hasattr(cy, 'project_endpoint_syn_double'):
        pytest.skip('Endpoint Cython projection is not built.')
    g, _ = toy_context(tmp_path)
    rng = np.random.default_rng(180)
    left, right = rng.random((11, 3)), rng.random((11, 3))
    transition = rng.random((3, 3))
    for stats in [{'any2any'}, {'any2any', 'any2spe'}, {'any2any', 'spe2any', 'any2spe'}]:
        native = endpoint_io._projection_transform(g, ['S'], {'S': None}, stats)(left, right, transition)
        with monkeypatch.context() as patch:
            patch.setattr(endpoint_io.substitution_sparse, 'substitution_sparse_cy', None)
            python = endpoint_io._projection_transform(g, ['S'], {'S': None}, stats)(left, right, transition)
        for key in native:
            np.testing.assert_allclose(native[key], python[key], atol=1e-14, rtol=1e-14)
    with pytest.raises(ValueError):
        cy.project_endpoint_syn_double(left, right, transition, np.array([[0, 0, 1, -1, 2]]), 2, 2, True, True)


def test_direct_projections_mixture_and_recoding(tmp_path):
    g, _ = toy_context(tmp_path)
    report = tmp_path / 'model.iqtree'
    report.write_text('Category Relative_rate Proportion\n1 0 0.1\n2 0.5 0.3\n3 2.0 0.6\n')
    g.update(output_stats=['any2any', 'any2spe', 'spe2any'])
    compact = copy.deepcopy(g)
    compact['subcommand'] = 'search'
    endpoint_io.prepare(g)
    endpoint_io.prepare(compact)
    for kind in ['S', 'N']:
        for stat in compact['output_stats']:
            np.testing.assert_allclose(g['_endpoint_tensors'][kind].project(stat).toarray(),
                                       compact['_endpoint_tensors'][kind].project(stat).toarray(), atol=1e-14)
            np.testing.assert_allclose(g['_endpoint_reducers'][kind]['projections'][stat].toarray(),
                                       compact['_endpoint_reducers'][kind]['projections'][stat].toarray(), atol=1e-14)


@pytest.mark.parametrize('structural', [False, True])
@pytest.mark.parametrize('sitewise', [False, True])
@pytest.mark.parametrize('block_size', [1, 2, 64])
def test_streamed_pairs_match_retained_projections(tmp_path, structural, sitewise, block_size):
    import pandas as pd
    from csubst.substitution_sparse import PairwiseSubstitutionSummary
    retained, ids = toy_context(tmp_path, structural)
    retained.update(subcommand='search', max_arity=3, num_node=5, b=sitewise,
                    output_stats=['any2any', 'spe2any', 'any2spe'], endpoint_block_size=block_size)
    streamed = copy.deepcopy(retained)
    streamed['max_arity'] = 2
    endpoint_io.prepare(retained)
    endpoint_io.prepare(streamed)
    pairs = np.array([[a, b] for a in range(5) for b in range(5)], dtype=np.int64)
    for kind in ['S', 'N']:
        a, b = retained['_endpoint_tensors'][kind], streamed['_endpoint_tensors'][kind]
        assert isinstance(b, PairwiseSubstitutionSummary)
        assert b.projections == {}
        np.testing.assert_allclose(a.branch_site, b.branch_site, atol=1e-14)
        for stat in streamed['output_stats']:
            projection = a.project(stat)
            np.testing.assert_allclose(b.pairwise[stat], (projection @ projection.T).toarray(), atol=1e-14)
            expected = retained['_endpoint_reducers'][kind]['projections'][stat]
            np.testing.assert_allclose(streamed['_endpoint_reducers'][kind]['pairwise'][stat],
                                       (expected @ expected.T).toarray(), atol=1e-14)
        pd.testing.assert_frame_equal(
            substitution.get_cb(pairs, a, retained, 'OC' + kind, retained['output_stats']),
            substitution.get_cb(pairs, b, streamed, 'OC' + kind, streamed['output_stats']),
            check_exact=False, atol=1e-14, rtol=1e-12)
        pd.testing.assert_frame_equal(substitution.get_b(retained, a, kind, sitewise and kind == 'N'),
                                      substitution.get_b(streamed, b, kind, sitewise and kind == 'N'))
        expected = endpoint_io.expected_reducer(streamed, 'cdn' if kind == 'S' else 'nsy', ['any2spe'])
        assert set(expected['pairwise']) == {'any2spe'}


def test_endpoint_expected_cache_releases_last_reference(tmp_path):
    import weakref
    g, _ = toy_context(tmp_path)
    endpoint_io.prepare(g)
    ref = weakref.ref(g['_endpoint_reducers']['N']['projections']['any2any'])
    g['EN_reducer'] = endpoint_io.expected_reducer(g, 'nsy', ['any2any'])
    endpoint_io.release_expected(g, 'N')
    assert 'N' in g['_endpoint_reducers']  # No lifecycle permission yet.
    g['_release_state_after_expected_reducer'] = True
    g['fg_clade_permutation'] = 1
    endpoint_io.release_expected(g, 'N')
    assert 'N' in g['_endpoint_reducers']
    g['fg_clade_permutation'] = 0
    endpoint_io.release_expected(g, 'N')
    assert 'N' not in g['_endpoint_reducers']
    assert ref() is not None
    del g['EN_reducer']
    assert ref() is None


@pytest.mark.parametrize('key,value', [('max_arity', 3), ('site_filter_report', True), ('fg_clade_permutation', 1)])
def test_streaming_retains_projections_when_later_consumers_need_them(tmp_path, key, value):
    from csubst.substitution_sparse import PairwiseSubstitutionSummary
    g, _ = toy_context(tmp_path)
    g.update(subcommand='search', max_arity=2, output_stats=['any2any', 'any2spe'])
    g[key] = value
    endpoint_io.prepare(g)
    assert not isinstance(g['_endpoint_tensors']['N'], PairwiseSubstitutionSummary)


def test_streamed_pair_workspace_bound_accounts_for_quadratic_branch_axis(tmp_path):
    g, _ = toy_context(tmp_path)
    kinds = ['S', 'N']
    stats = ['any2any', 'any2spe']
    shapes = {'S': (10000, 3, 2, 2, 2), 'N': (10000, 3, 1, 2, 2)}
    assert endpoint_io._pairwise_storage_bound(10000, 3, shapes, kinds, stats, 64, g) > 64 * 1024**2
    assert endpoint_io._pairwise_storage_bound(5, 3, shapes, kinds, stats, 64, g) < 64 * 1024**2
