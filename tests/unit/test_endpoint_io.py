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


def test_iqtree_invar_zero_without_category_table_is_unit_rate():
    # IQ-TREE 2.3.6 real-output layout for GY+FQ+I with p_inv fitted to zero.
    report = ('Model of substitution: GY+FQ+I\n\n'
              'Model of rate heterogeneity: Invar\n'
              'Proportion of invariable sites: 0.000000\n\n'
              'USER TREE\n---------\n')
    rates, weights = endpoint_io.read_rate_mixture(report)
    np.testing.assert_array_equal(rates, [1.])
    np.testing.assert_array_equal(weights, [1.])


@pytest.mark.parametrize('model,proportion', [
    ('Invar', '.2'), ('Invar', '0.000001'), ('Invar', '-.1'),
    ('Invar', 'nan'), ('Invar', 'inf'), ('Invar', 'unknown'),
    ('Invar+Gamma with 4 categories', '0.000000'),
])
def test_invar_without_table_cannot_hide_other_rate_models(model, proportion):
    report = ('Model of rate heterogeneity: ' + model + '\n'
              'Proportion of invariable sites: ' + proportion + '\n')
    with pytest.raises(ValueError, match='rate-category table'):
        endpoint_io.read_rate_mixture(report)


def test_invar_category_table_retains_positive_invariant_component():
    report = ('Model of rate heterogeneity: Invar\n'
              'Proportion of invariable sites: 0.499942\n\n'
              ' Category  Relative_rate  Proportion\n'
              '  0         0              0.499942\n'
              '  1         1.999766       0.500058\n')
    rates, weights = endpoint_io.read_rate_mixture(report)
    np.testing.assert_array_equal(rates, [0., 1.999766])
    np.testing.assert_allclose(weights, [.499942, .500058])


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


@pytest.mark.parametrize('model_name', ['MG+FO', 'GY+C20', 'ECMK07+ASC', 'GY+H4', 'GY+unknown'])
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
@pytest.mark.parametrize('block_size', [1, 64])
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


@pytest.mark.parametrize("branch_table", [False, True])
def test_direct_projections_mixture_and_recoding(tmp_path, branch_table):
    g, _ = toy_context(tmp_path)
    report = tmp_path / 'model.iqtree'
    report.write_text('Category Relative_rate Proportion\n1 0 0.1\n2 0.5 0.3\n3 2.0 0.6\n')
    g.update(output_stats=['any2any', 'any2spe', 'spe2any'], b=branch_table)
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

        if branch_table:
            for branch in range(5):
                for threshold in [0, 0.05, 0.5]:
                    expected = substitution._get_sparse_branch_sitewise_max_indices(
                        g['_endpoint_tensors'][kind], branch, threshold)
                    actual = substitution._get_sparse_branch_sitewise_max_indices(
                        compact['_endpoint_tensors'][kind], branch, threshold)
                    for a, b in zip(expected, actual):
                        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('structural', [False, True])
@pytest.mark.parametrize('sitewise', [False, True])
@pytest.mark.parametrize('block_size', [1, 64])
@pytest.mark.parametrize('full_stats', [False, True])
def test_streamed_pairs_match_retained_projections(tmp_path, structural, sitewise, block_size, full_stats):
    import pandas as pd
    from csubst.substitution_sparse import PairwiseSubstitutionSummary
    retained, ids = toy_context(tmp_path, structural)
    retained.update(subcommand='search', max_arity=3, num_node=5, b=sitewise,
                    output_stats=['any2any', 'spe2any', 'any2spe'], endpoint_block_size=block_size)
    if full_stats:
        retained['output_stats'].append('spe2spe')
    streamed = copy.deepcopy(retained)
    streamed['max_arity'] = 2
    endpoint_io.prepare(retained)
    endpoint_io.prepare(streamed)
    pairs = np.array([[a, b] for a in range(5) for b in range(5)], dtype=np.int64)
    for kind in ['S', 'N']:
        a, b = retained['_endpoint_tensors'][kind], streamed['_endpoint_tensors'][kind]
        assert isinstance(b, PairwiseSubstitutionSummary)
        assert b.projections == {}
        for branch in range(5):
            np.testing.assert_allclose(substitution.get_branch_site_sub_counts(a, branch),
                                       b.branch_site[branch], atol=1e-14)
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


@pytest.mark.parametrize('structural', [False, True])
def test_selected_sites_branches_keep_global_totals(tmp_path, structural):
    import pandas as pd
    from csubst import main_sites
    full, ids = toy_context(tmp_path, structural)
    full['subcommand'] = 'sites'
    selected = copy.deepcopy(full)
    branches = np.array([ids['A'], ids['B']])
    selected['_endpoint_retained_branches'] = branches.tolist()
    endpoint_io.prepare(full)
    endpoint_io.prepare(selected)
    for kind in ['S', 'N']:
        a, b = full['_endpoint_tensors'][kind], selected['_endpoint_tensors'][kind]
        np.testing.assert_allclose(substitution.get_site_sub_counts(a), substitution.get_site_sub_counts(b), atol=1e-14)
        np.testing.assert_allclose(substitution.get_branch_sub_counts(a), substitution.get_branch_sub_counts(b), atol=1e-14)
        np.testing.assert_allclose(substitution.get_branches_sub_tensor(a, branches),
                                   substitution.get_branches_sub_tensor(b, branches), atol=1e-14)
        before = main_sites.add_cs_info(main_sites.initialize_site_df(3), branches, a, kind)
        after = main_sites.add_cs_info(main_sites.initialize_site_df(3), branches, b, kind)
        pd.testing.assert_frame_equal(before, after, check_exact=False, rtol=1e-12, atol=1e-14)
        assert b.retained_branches == frozenset(branches)
        assert b.matrix.getrow(ids['U']).nnz == 0
    old = selected['_endpoint_tensors']['N']
    selected['_endpoint_retained_branches'] = [ids['C']]
    endpoint_io.prepare(selected)
    assert selected['_endpoint_tensors']['N'] is not old


@pytest.mark.parametrize('key,value', [('max_arity', 3), ('site_filter_report', True), ('fg_clade_permutation', 1)])
def test_streaming_retains_projections_when_later_consumers_need_them(tmp_path, key, value):
    from csubst.substitution_sparse import PairwiseSubstitutionSummary
    g, _ = toy_context(tmp_path)
    g.update(subcommand='search', max_arity=2, output_stats=['any2any', 'any2spe'])
    g[key] = value
    endpoint_io.prepare(g)
    assert not isinstance(g['_endpoint_tensors']['N'], PairwiseSubstitutionSummary)


def test_streamed_full_statistics_accept_empty_site_axis(tmp_path):
    g, _ = toy_context(tmp_path)
    g.update(subcommand='search', max_arity=2)
    for key in ['state_cdn', 'state_pep', 'state_nsy']:
        g[key] = g[key][:, :0]
    endpoint_io.prepare(g)
    for kind in ['S', 'N']:
        for values in g['_endpoint_tensors'][kind].pairwise.values():
            np.testing.assert_array_equal(values, np.zeros((5, 5)))


@pytest.mark.native
@pytest.mark.parametrize('size', [1, 64, 129])
@pytest.mark.parametrize('strided', [False, True])
def test_native_classified_events_match_full_mapping(tmp_path, monkeypatch, size, strided):
    cy = endpoint_io.substitution_sparse.substitution_sparse_cy
    if cy is None or not hasattr(cy, 'project_endpoint_events_double'):
        pytest.skip('Endpoint event extension is not built.')
    g, _ = toy_context(tmp_path)
    rng = np.random.default_rng(315)
    if strided:
        left, right = rng.random((size * 2, 6))[::2, ::2], rng.random((size * 2, 6))[::2, ::2]
        transition = rng.random((6, 6))[::2, ::2]
    else:
        left, right = rng.random((size, 3)), rng.random((size, 3))
        transition = rng.random((3, 3))
    mappings = {'S': None, 'N': endpoint_io._mapping(g, 'N', 3)}
    actual = endpoint_io._event_transform(g, ['S', 'N'], mappings)(left, right, transition)
    with monkeypatch.context() as patch:
        patch.setattr(endpoint_io.substitution_sparse, 'substitution_sparse_cy', None)
        expected = endpoint_io._event_transform(g, ['S', 'N'], mappings)(left, right, transition)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-14, atol=1e-15)
    for pairs in (np.array([[0, -1, 0]]), np.array([[100, 0, 0]]), np.zeros((1, 2), dtype=np.int64)):
        with pytest.raises(ValueError):
            cy.project_endpoint_events_double(left, right, transition, pairs, 2, 2)


@pytest.mark.parametrize('structural', [False, True])
@pytest.mark.parametrize('recoded', [False, True])
def test_full_classified_stream_matches_python_events(tmp_path, monkeypatch, structural, recoded):
    g, _ = toy_context(tmp_path, structural)
    g.update(subcommand='search', b=True, num_node=5)
    if recoded and not structural:
        g.update(nonsyn_recode='dayhoff6', nonsyn_state_orders=['A', 'B'],
                 nonsynonymous_indices={'A': [0], 'B': [1, 2]})
        g['state_nsy'] = np.stack([g['state_cdn'][:, :, 0], g['state_cdn'][:, :, 1:].sum(2)], axis=2)
    (tmp_path / 'model.iqtree').write_text('Category Relative_rate Proportion\n1 0 0.1\n2 0.5 0.3\n3 2.0 0.6\n')
    reference = copy.deepcopy(g)
    with monkeypatch.context() as patch:
        patch.setattr(endpoint_io.substitution_sparse, 'substitution_sparse_cy', None)
        endpoint_io.prepare(reference)
    endpoint_io.prepare(g)
    for kind, tensor in g['_endpoint_tensors'].items():
        np.testing.assert_allclose(tensor.to_dense(), reference['_endpoint_tensors'][kind].to_dense(),
                                   atol=1e-14, rtol=1e-13)
    for kind, reducer in g['_endpoint_reducers'].items():
        for stat, projection in reducer['projections'].items():
            np.testing.assert_allclose(projection.toarray(),
                                       reference['_endpoint_reducers'][kind]['projections'][stat].toarray(),
                                       atol=1e-14, rtol=1e-13)


@pytest.mark.parametrize('cache_bytes', [0, 144, 32 * 1024**2])
def test_cached_prediction_projection_matches_full_contraction(tmp_path, cache_bytes):
    g, _ = toy_context(tmp_path)
    rng = np.random.default_rng(3141)
    left, right = rng.random((9, 3)), np.ones((9, 3))
    mapping = {'S': None, 'N': endpoint_io._mapping(g, 'N', 3)}
    stats = {'any2any', 'any2spe', 'spe2any'}
    full = endpoint_io._projection_transform(g, ['S', 'N'], mapping, stats)
    cached = endpoint_io._projection_transform(g, ['S', 'N'], mapping, stats,
                                               predictive=True, cache_bytes=cache_bytes)
    matrices = [rng.random((3, 3)), np.eye(3), rng.random((3, 3))]
    for matrix in matrices + matrices[::-1] + matrices:
        expected, actual = full(left, right, matrix), cached(left, right, matrix)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize('block_size', [1, 64])
@pytest.mark.parametrize('mixture', [False, True])
def test_scan_and_search_share_all_event_probabilities_and_missing_exposure(tmp_path, block_size, mixture):
    from csubst import scan_ctmc, scan_endpoint
    g, ids = toy_context(tmp_path)
    g.update(endpoint_block_size=block_size, scan_observation='joint', scan_rate_length='raw')
    if mixture:
        g['substitution_model'] = 'GY+F+R2'
        (tmp_path/'model.iqtree').write_text('Category Relative_rate Proportion\n1 0.2 0.4\n2 1.5333333333333333 0.6\n')
    scan_g, scan_tensor = scan_ctmc.prepare(copy.deepcopy(g))
    endpoint_io.prepare(g)
    np.testing.assert_allclose(g['_endpoint_tensors']['N'].to_dense(), scan_tensor, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(substitution.get_branch_sub_counts(g['_endpoint_tensors']['S']),
                               scan_g['scan_ctmc_synonymous_counts'], atol=1e-12, rtol=1e-10)
    tree.rescale_branch_length(g, g['_endpoint_tensors']['S'], g['_endpoint_tensors']['N'])
    scan_ctmc.set_branch_length_summaries(scan_g, scan_tensor)
    for first, second in zip(g['tree'].traverse(), scan_g['tree'].traverse()):
        for attr in ('Sdist', 'Ndist', 'SNdist'):
            assert ete.get_prop(first, attr) == pytest.approx(ete.get_prop(second, attr), abs=1e-12)
    mask = g['_endpoint_tensors']['N'].eligible
    np.testing.assert_array_equal(scan_g['event_eligible'], mask)
    assert not mask[ids['C']].any()
    assert not mask[:, 2].any()
    # Missingness excludes output; it does not erase the inferred latent state.
    np.testing.assert_allclose(scan_g['state_cdn'][ids['C']].sum(axis=1), 1)
    context = scan_endpoint.build_context(scan_g, substitution_scan.build_branch_metadata(scan_g), np.array([0,0,1]))
    for site in range(3):
        expected, missing, _ = scan_endpoint.expected_events(context, scan_g['state_cdn'], scan_g['state_nsy'], site, [0], [1])
        np.testing.assert_array_equal(missing, ~mask[context['branch_ids'], site])
        assert not expected[missing].any()


def test_reporting_preserves_true_zero_and_marks_missing_with_coverage(tmp_path):
    import pandas as pd
    from csubst import event_reporting, tsv
    g, ids = toy_context(tmp_path)
    endpoint_io.prepare(g)
    frame = pd.DataFrame({'branch_id':[ids['U'],ids['C'],ids['A']], 'site':[0,0,2], 'N_sub':[0.,0.,0.], 'S_sub':[0.,0.,0.]})
    result = event_reporting.annotate(frame, g)
    assert result['N_sub'].iloc[0] == 0  # true zero-length edge
    assert result['N_sub'].iloc[1:].isna().all()
    assert result['N_eligible_count'].tolist() == [1,0,0]
    assert frame['N_sub'].notna().all()  # numerical input is untouched
    path = tmp_path/'events.tsv'
    tsv.write_dataframe(frame, path, report_context=g)
    assert '\tNA\tNA\t' in path.read_text()
    g['site_index_alignment'] = np.array([2,5,9])
    sites = pd.DataFrame({'codon_site_alignment':[3,6,10,13], f'N_sub_{ids["A"]}':[0.,0.,0.,0.]})
    result = event_reporting.annotate(sites, g)
    assert result[f'N_eligible_{ids["A"]}'].tolist() == [True,True,False,False]


def test_endpoint_cache_rebuilds_after_original_tip_or_model_change(tmp_path):
    g, ids = toy_context(tmp_path)
    endpoint_io.prepare(g)
    previous = g['_endpoint_tensors']['N']
    fingerprint = g['_endpoint_input_fingerprint']
    endpoint_io.prepare(g)
    assert g['_endpoint_tensors']['N'] is previous
    g['state_cdn'][ids['C'], 0] = [1,0,0]
    endpoint_io.prepare(g)
    assert g['_endpoint_input_fingerprint'] != fingerprint
    assert g['_endpoint_tensors']['N'].eligible[ids['C'], 0]
    assert g['_endpoint_tensors']['N'] is not previous
    previous = g['_endpoint_tensors']['N'].to_dense()
    next(ete.iter_leaves(g['tree'])).dist *= .5
    endpoint_io.prepare(g)
    assert not np.allclose(previous, g['_endpoint_tensors']['N'].to_dense())


def test_combination_reporting_requires_joint_observation_support(tmp_path):
    import pandas as pd
    from csubst import event_reporting
    g, ids = toy_context(tmp_path)
    endpoint_io.prepare(g)
    frame = pd.DataFrame({'site':[0,1,2], 'OCNany2any':[0.,0.,0.],
                          'OCSany2any':[0.,0.,0.], 'omegaCany2any':[1.,1.,1.]})
    result = event_reporting.annotate(frame, dict(g, report_combinations=[[ids['A'],ids['C']]]))
    assert result['OCNany2any'].isna().all()
    assert result['N_combination_eligible_count'].tolist() == [0,0,0]
    assert result['omegaCany2any'].isna().all()


def test_bridge_scan_applies_same_mask_without_erasing_latent_states(tmp_path):
    from csubst import scan_ctmc
    g, ids = toy_context(tmp_path)
    g.update(scan_observation='bridge', scan_rate_length='raw')
    updated, tensor = scan_ctmc.prepare(g)
    assert not tensor[ids['C']].any()
    assert not tensor[:,2].any()
    assert updated['scan_ctmc_synonymous_counts'][ids['C']] == 0
    np.testing.assert_allclose(updated['state_cdn'][ids['C']].sum(1), 1)
    assert tensor[ids['A'],0].sum() > 0


def test_combination_branch_subtotals_use_slot_branch_ids(tmp_path):
    import pandas as pd
    from csubst import event_reporting
    g, ids = toy_context(tmp_path)
    endpoint_io.prepare(g)
    frame = pd.DataFrame({'branch_id_1': [ids['C'], ids['U']],
                          'branch_id_2': [ids['A'], ids['C']],
                          'N_sub_1': [0., 0.], 'N_sub_2': [0., 0.]})
    out = event_reporting.annotate(frame, g)
    assert out['N_eligible_count_1'].tolist() == [0, 2]
    assert out['N_eligible_count_2'].tolist() == [2, 0]
    assert np.isnan(out.loc[0, 'N_sub_1'])
    assert out.loc[1, 'N_sub_1'] == 0
    assert out.loc[0, 'N_sub_2'] == 0
    assert np.isnan(out.loc[1, 'N_sub_2'])


@pytest.mark.parametrize('option,value', [('asrv_report', True), ('calc_omega_pvalue', True),
    ('epistasis_requested', True), ('asrv_training_branches', 'foreground'),
    ('site_filter_report', True), ('fg_clade_permutation', 1)])
def test_endpoint_cache_rebuilds_when_consumer_changes(tmp_path, option, value):
    g, _ = toy_context(tmp_path)
    g.update(subcommand='search', max_arity=2, output_stats=['any2any'])
    endpoint_io.prepare(g)
    previous = g['_endpoint_tensors']['N']
    g[option] = value
    endpoint_io.prepare(g)
    assert g['_endpoint_tensors']['N'] is not previous
    assert g['_endpoint_manifest']['codon']['observed_storage'] == (
        'projections' if option in ('site_filter_report', 'fg_clade_permutation') else 'full_events')


def test_3di_expected_table_survives_early_state_release(tmp_path):
    import pandas as pd
    g, ids = toy_context(tmp_path, structural=True)
    released = copy.deepcopy(g)
    released['_release_state_after_expected_reducer'] = True
    cb = pd.DataFrame({'branch_id_1': [ids['A']], 'branch_id_2': [ids['B']]})
    outputs = []
    for config in (g, released):
        n = substitution.get_substitution_tensor(config['state_nsy'], mode='asis', g=config)
        s = substitution.get_substitution_tensor(config['state_cdn'], mode='syn', g=config)
        outputs.append(omega.get_E(cb.copy(), config, n, s))
    pd.testing.assert_frame_equal(outputs[0], outputs[1])
    assert all(released[key] is None for key in ('state_cdn', 'state_pep', 'state_nsy'))


def test_native_3di_scan_uses_structural_lengths_and_independent_synonymous_model(tmp_path):
    from scipy.linalg import expm
    from csubst import scan_ctmc, scan_endpoint
    g, ids = toy_context(tmp_path, structural=True)
    g.update(subcommand='scan', scan_observation='joint', scan_rate_exposure='endpoint',
             scan_rate_length='raw', scan_rate_event_mode='posterior_sum', scan_pvalue_calibration='none')
    # Make structural branch lengths differ materially from the codon fit.
    g['3di_branch_lengths'] *= 0.3
    original = copy.deepcopy(g)
    updated, tensor = scan_ctmc.prepare(g)
    reference = copy.deepcopy(original)
    endpoint_io._build(reference, structural=True)
    np.testing.assert_allclose(tensor, reference['_endpoint_tensors']['N'].to_dense(), atol=1e-13)
    codon_reference = copy.deepcopy(original)
    codon_reference.update(nonsyn_recode='no', state_nsy=codon_reference['state_pep'])
    codon_result, _ = scan_ctmc.prepare(codon_reference)
    np.testing.assert_allclose(updated['scan_ctmc_synonymous_counts'], codon_result['scan_ctmc_synonymous_counts'])
    np.testing.assert_allclose(updated['state_cdn'], codon_result['state_cdn'])
    meta = substitution_scan.build_branch_metadata(updated)
    context = substitution_scan._build_scan_q_context(updated, 'endpoint', meta)
    expected, missing, _ = scan_endpoint.expected_events(
        context['endpoint_context'], updated['state_nsy'], updated['state_nsy'], 0, [0], [1])
    for row, value, absent in zip(meta.itertuples(), expected, missing):
        if absent:
            assert value == 0
        else:
            p = expm(original['3di_q'] * original['3di_branch_lengths'][row.branch_id])
            assert value == pytest.approx(updated['state_nsy'][row.parent_id, 0, 0] * p[0, 1])
    assert next(node.dist for node in original['tree'].traverse() if node.name == 'A') == 1
