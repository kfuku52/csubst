import csv
import json

import numpy as np
import pytest

from csubst import epistasis, ete, omega, tree


def _arrays():
    rng = np.random.default_rng(7)
    counts = rng.poisson(2., size=(24, 5)).astype(float)
    context = rng.normal(size=(24, 2))
    features = rng.normal(size=(5, 2))
    masks = np.ones_like(counts, dtype=bool)
    groups = np.repeat(np.arange(4), 6)
    partitions = np.zeros(24, dtype=int)
    return counts, context, features, masks, groups, partitions


def _config(**overrides):
    return dict({'epistasis_beta_auto': True, 'epistasis_clip_auto': False,
                 'epistasis_clip_value': 3., 'asrv_dirichlet_alpha': 1.}, **overrides)


@pytest.mark.parametrize('joint', [False, True])
@pytest.mark.parametrize('depth_bins', [False, True])
def test_held_out_response_cannot_change_its_predictions_or_parameters(joint, depth_bins):
    arrays = list(_arrays())
    if depth_bins:
        arrays[-1] = np.tile([0, 1], 12)
    config = _config(epistasis_joint_auto=joint, epistasis_clip_auto=joint,
                     epistasis_joint_alpha_grid=[0., 1.], epistasis_joint_clip_grid=[1.5, 3.])
    first = epistasis.crossfit(*arrays, config)
    test = arrays[-2] == 0
    arrays[0] = arrays[0].copy()
    arrays[0][test] = 0
    arrays[0][test, 0] = 1000
    changed = epistasis.crossfit(*arrays, config)
    for key in ('probabilities', 'null_probabilities', 'beta_by_branch', 'alpha_by_branch', 'clip_by_branch'):
        np.testing.assert_array_equal(first[key][test], changed[key][test])


def test_buffer_responses_never_enter_training_and_are_not_scored():
    arrays = list(_arrays())
    arrays[-2][:2] = -1
    first = epistasis.crossfit(*arrays, _config())
    arrays[0][:2] = 999
    second = epistasis.crossfit(*arrays, _config())
    np.testing.assert_array_equal(first['probabilities'], second['probabilities'])
    buffer = first['beta_diag']['outer_folds'][-1]
    assert buffer['scored'] is False
    assert buffer['outer_log_score'] is None
    assert all(0 not in d['training_branch_ids'] for d in first['beta_diag']['outer_folds'])


def test_row_permutation_preserves_predictions():
    arrays = _arrays()
    order = np.random.default_rng(3).permutation(24)
    permuted = [a[order] if i != 2 else a for i, a in enumerate(arrays)]
    first, second = epistasis.crossfit(*arrays, _config()), epistasis.crossfit(*permuted, _config())
    np.testing.assert_allclose(first['probabilities'][order], second['probabilities'])
    np.testing.assert_array_equal(first['beta_by_branch'][order], second['beta_by_branch'])


def test_zero_context_ties_select_exact_zero_and_remain_normalized():
    arrays = list(_arrays())
    arrays[1][:] = 0
    arrays[3][0, :2] = False
    arrays[3][1] = False
    out = epistasis.crossfit(*arrays, _config())
    np.testing.assert_array_equal(out['beta_by_branch'], 0.)
    np.testing.assert_array_equal(out['probabilities'][~arrays[3]], 0.)
    np.testing.assert_allclose(out['probabilities'].sum(axis=1), arrays[3].any(axis=1))


def test_no_events_select_zero_without_discarding_branches():
    arrays = list(_arrays())
    arrays[0][:] = 0.
    out = epistasis.crossfit(*arrays, _config())
    np.testing.assert_array_equal(out['beta_by_branch'], 0.)
    np.testing.assert_allclose(out['probabilities'], .2)
    assert sum(len(d['prediction_branch_ids']) for d in out['beta_diag']['outer_folds']) == 24


def test_independent_positive_signal_improves_outer_score():
    arrays = list(_arrays())
    base = np.full_like(arrays[0], .2)
    signal = epistasis.predict(base, arrays[1], arrays[2], 1.2, 3., arrays[3])
    arrays[0] = signal * 1000.
    out = epistasis.crossfit(*arrays, _config())
    assert np.all(out['beta_by_branch'] > 0.)
    folds = out['beta_diag']['outer_folds']
    assert sum(d['outer_log_score'] - d['outer_null_log_score'] for d in folds) > 0


def test_predict_matches_independent_multifeature_softmax():
    base = np.array([[.2, .3, .5], [.5, .3, .2]])
    context = np.array([[1., 0.], [0., 1.]])
    features = np.array([[1., 0.], [0., 1.], [-1., -1.]])
    expected = base * np.exp(.8 * context @ features.T / 2)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(epistasis.predict(base, context, features, .8, 3., np.ones_like(base, bool)), expected)
    with pytest.raises(ValueError, match='nonnegative'):
        epistasis.predict(base, context, features, -3., 3., np.ones_like(base, bool))


def test_insufficient_clades_and_depth_bins_fail_explicitly():
    arrays = list(_arrays())
    arrays[-2] %= 2
    with pytest.raises(ValueError, match='at least 3'):
        epistasis.crossfit(*arrays, _config())
    out = epistasis.crossfit(*arrays, _config(epistasis_beta_auto=False, epistasis_beta_value=.5))
    np.testing.assert_array_equal(out['beta_by_branch'], .5)


def _tree():
    return tree.add_numerical_node_labels(ete.PhyloNode('(((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1);'))


def _write_context(path, keys, features=1):
    with path.open('w') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['branch_key'] + ['context_{}'.format(i + 1) for i in range(features)])
        for i, key in keys.items():
            writer.writerow([key] + [float(i % 3 - 1)] * features)


def test_clades_and_context_survive_tree_child_reordering(tmp_path):
    tr = _tree()
    n = len(list(tr.traverse()))
    keys, groups, depths = epistasis.tree_layout(tr, n, 4)
    by_key = {key: (int(groups[i]), float(depths[i])) for i, key in keys.items()}
    for node in tr.traverse():
        node.children.reverse()
    tr = tree.add_numerical_node_labels(tr)
    keys2, groups2, depths2 = epistasis.tree_layout(tr, n, 4)
    assert by_key == {key: (int(groups2[i]), float(depths2[i])) for i, key in keys2.items()}
    assert len(set(groups) - {-1}) == 4
    path = tmp_path / 'context.tsv'
    _write_context(path, keys)
    first, _ = epistasis.load_context(path, keys, n, 1, 'external data')
    second, _ = epistasis.load_context(path, keys2, n, 1, 'external data')
    for i, key in keys.items():
        j = next(j for j in keys2 if keys2[j] == key)
        np.testing.assert_array_equal(first[i], second[j])


@pytest.mark.parametrize('error', ['duplicate', 'missing', 'nonfinite', 'wrong_dimension'])
def test_invalid_context_is_rejected(tmp_path, error):
    path = tmp_path / 'context.tsv'
    keys = {0: '["A"]', 1: '["B"]'}
    _write_context(path, keys)
    lines = path.read_text().splitlines()
    if error == 'duplicate':
        lines.append(lines[-1])
    elif error == 'missing':
        lines.pop()
    elif error == 'nonfinite':
        lines[-1] = lines[-1].rsplit('\t', 1)[0] + '\tnan'
    path.write_text('\n'.join(lines) + '\n')
    with pytest.raises(ValueError):
        epistasis.load_context(path, keys, 2, 2 if error == 'wrong_dimension' else 1, 'external')


@pytest.mark.parametrize('channel', ['N', 'S', 'NS'])
def test_prepare_and_actual_asrv_use_exact_crossfitted_probabilities(tmp_path, channel):
    tr = _tree()
    n = len(list(tr.traverse()))
    keys, _, _ = epistasis.tree_layout(tr, n)
    path = tmp_path / 'context.tsv'
    _write_context(path, keys)
    g = _config(epistasis_requested=True, epistasis_context_file=str(path), epistasis_context_source='independent experiment',
                asrv='sn', expectation_method='urn', tree=tr, epistasis_apply_to=channel,
                float_tol=1e-12, is_site_nonmissing=np.ones((n, 3), bool),
                epistasis_site_feature_matrix_internal=np.array([[-1.], [0.], [1.]]))
    counts = np.random.default_rng(2).poisson(2., (n, 3, 1, 1, 1)).astype(float)
    omega.prepare_epistasis(g, counts, counts)
    original = np.full((n, 3), 1/3)
    g['sub_sites'] = {'N': original, 'S': original}
    for c in ['N', 'S']:
        actual = omega._resolve_sub_sites(g, None, 'any2spe', 0, 0, 0, 'OC' + c + 'any2spe')
        if c in channel:
            assert actual is g['_epistasis_state'][c]['probabilities']
        else:
            assert actual is original
    report = tmp_path / 'report.json'
    epistasis.write_report(g, report)
    payload = json.loads(report.read_text())
    assert payload['pipeline_calibrated'] is False
    assert payload['channels'][channel[0]]['outer_folds']


def test_heldout_zero_events_do_not_change_fold_membership_or_predictions():
    arrays = list(_arrays())
    first = epistasis.crossfit(*arrays, _config())
    arrays[0][arrays[-2] == 0] = 0.
    second = epistasis.crossfit(*arrays, _config())
    np.testing.assert_array_equal(first['probabilities'][:6], second['probabilities'][:6])
    np.testing.assert_array_equal(first['beta_by_branch'][:6], second['beta_by_branch'][:6])


def test_all_undefined_inner_scores_fail_instead_of_arbitrary_selection():
    counts = np.eye(3)
    with pytest.raises(ValueError, match='All inner'):
        epistasis.crossfit(counts, np.ones((3, 1)), np.arange(3.)[:, None],
                          np.ones_like(counts, bool), np.arange(3), np.zeros(3, int),
                          _config(asrv_dirichlet_alpha=0.))


def test_large_clip_ignores_masked_extreme_and_zero_base_sites():
    base = np.array([[.3, .7, 0.]])
    actual = epistasis.predict(base, np.ones((1, 1)), np.array([[0.], [1.], [1000.]]),
                               1., 1000., np.array([[True, True, False]]))
    expected = np.array([[.3, .7 * np.e, 0.]])
    expected /= expected.sum()
    np.testing.assert_allclose(actual, expected)
