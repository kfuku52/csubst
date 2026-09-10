"""Numerical routing and real IQ-TREE context round trips (no predictor download)."""
import copy
import shutil

import pandas as pd
import numpy as np
import pytest
from scipy.linalg import expm

from csubst import ete, expectation_3di, omega, parser_misc, structural_alphabet, substitution, tree


def _context():
    tr = tree.add_numerical_node_labels(ete.PhyloNode('(A:0.2,B:0.4)R;', format=1))
    state = np.random.default_rng(19).dirichlet(np.ones(20), size=(3, 4))
    q = np.full((20, 20), 1 / 19)
    np.fill_diagonal(q, -1)
    lengths = np.zeros(3)
    for node in tr.traverse():
        idx = int(ete.get_prop(node, 'numerical_label'))
        lengths[idx] = 0 if ete.is_root(node) else node.dist
        ete.set_prop(node, 'Ndist', 70)  # Deliberately unrelated codon exposures.
        ete.set_prop(node, 'SNdist', .3)
    return dict(tree=tr, rooted_tree=tr, state_nsy=state, nonsyn_recode='3di20',
                expectation_method='codon_model', float_type=np.float64, float_tol=1e-12,
                threads=1, ml_anc=False, min_sub_pp=0, expected_state_backend='eigen',
                iqtree_rate_values=np.array([.1, .2, 2., 8.]),
                **{'3di_q': q, '3di_pi': np.full(20, .05), '3di_rates': np.ones(4),
                   '3di_branch_lengths': lengths, '3di_orders': np.array(list('ACDEFGHIKLMNPQRSTVWY')),
                   '3di_tip_invariant_mask': np.zeros(4, dtype=bool)})


@pytest.mark.parametrize('backend', ['eigen', 'expm'])
def test_native_N_uses_its_own_Q_rates_lengths_and_sparse_reducers(backend):
    g = _context()
    g['expected_state_backend'] = backend
    state = g['state_nsy']
    expected = omega.get_exp_state(g, 'nsy')
    for node in g['tree'].traverse():
        if ete.is_root(node):
            continue
        child = int(ete.get_prop(node, 'numerical_label'))
        parent = int(ete.get_prop(node.up, 'numerical_label'))
        np.testing.assert_allclose(expected[child], state[parent] @ expm(g['3di_q'] * node.dist), atol=1e-12)
    materialized = substitution.get_substitution_tensor(expected, state, mode='asis', g=g)
    if backend == 'eigen':
        fused = omega._get_fused_expected_sparse_substitution_tensor(g, 'nsy')
        np.testing.assert_allclose(fused.to_dense(), materialized.to_dense(), atol=1e-12)
    reducer = omega._get_fused_expected_sparse_reducer(
        g, 'nsy', ['any2any', 'spe2any', 'any2spe', 'spe2spe'])
    assert reducer['total'] == pytest.approx(substitution.get_total_substitution(materialized))
    for stat, projection in reducer['projections'].items():
        np.testing.assert_allclose(projection.toarray(),
                                   substitution._build_sparse_cb_projection(materialized, stat).toarray(), atol=1e-12)


def test_native_context_cache_round_trip_rejects_missing_or_invalid_Q(tmp_path):
    g = _context()
    g['sa_state_cache_file'] = str(tmp_path / 'cache.npz')
    shape = (3, 4, 61)
    parser_misc._write_3di_state_cache(g, None, shape, g['state_nsy'], g['3di_orders'])
    fresh = {k: v for k, v in g.items() if k not in expectation_3di.CONTEXT_KEYS}
    states, _, error = parser_misc._try_load_3di_state_cache(fresh, None, shape)
    assert error is None
    np.testing.assert_array_equal(states, g['state_nsy'])
    for key in expectation_3di.CONTEXT_KEYS:
        np.testing.assert_array_equal(fresh[key], g[key])
    with np.load(g['sa_state_cache_file']) as cache:
        fields = {key: cache[key] for key in cache.files}
    fields['3di_q'][0, 1] = -1
    np.savez_compressed(g['sa_state_cache_file'], **fields)
    assert parser_misc._try_load_3di_state_cache(fresh, None, shape)[0] is None
    fields.pop('3di_q')
    np.savez_compressed(g['sa_state_cache_file'], **fields)
    assert parser_misc._try_load_3di_state_cache(fresh, None, shape)[0] is None


def test_S_codon_exposure_is_identical_when_3di_observations_change():
    g = _context()
    g['state_pep'] = np.array([[[1., 0.]] * 4, [[0., 1.]] * 4, [[.7, .3]] * 4])
    g['state_cdn'] = g['state_pep'].copy()
    g['instantaneous_codon_rate_matrix'] = np.array([[-.4, .4], [.6, -.6]])
    aa = substitution.get_substitution_tensor(g['state_pep'], mode='asis', g=g)
    os_tensor = aa
    normal = copy.deepcopy(g)
    normal['nonsyn_recode'] = 'no'
    tree.rescale_branch_length(normal, os_tensor, aa)
    unrelated_3di = substitution.get_substitution_tensor(g['state_nsy'], mode='asis', g=g)
    tree.rescale_branch_length(g, os_tensor, unrelated_3di)
    for actual, baseline in zip(g['tree'].traverse(), normal['tree'].traverse()):
        assert ete.get_prop(actual, 'SNdist') == pytest.approx(ete.get_prop(baseline, 'SNdist'))
    np.testing.assert_allclose(omega.get_exp_state(g, 'cdn'), omega.get_exp_state(normal, 'cdn'))


@pytest.mark.parametrize('alphabet', ['ACDE', 'CVY', 'ACDEFGHIKLMNPQRSTVWY'])
def test_real_iqtree_direct_retains_model_and_invariant_sites(tmp_path, monkeypatch, alphabet):
    iqtree = shutil.which('iqtree2') or shutil.which('iqtree')
    if iqtree is None:
        pytest.skip('IQ-TREE is not installed')
    monkeypatch.chdir(tmp_path)
    tr = tree.add_numerical_node_labels(ete.PhyloNode('((A:.1,B:.1):.1,(C:.1,D:.1):.1)R;', format=1))
    rng = np.random.default_rng(31)
    base = np.tile(list(alphabet), 12)
    tips = {}
    for name in 'ABCD':
        seq = base.copy()
        indices = rng.choice(len(seq), len(seq) // 4, replace=False)
        seq[indices] = rng.choice(list(alphabet), len(indices))
        tips[name] = ''.join(seq)
    g = dict(nonsyn_recode='3di20', expectation_method='codon_model', sa_asr_mode='direct',
             rooted_tree=tr, tree=tr, iqtree_exe=iqtree, threads=1, float_type=np.float64,
             drop_invariant_tip_sites=True, drop_invariant_tip_sites_mode='tip_invariant')
    monkeypatch.setattr(structural_alphabet, 'build_tip_3di_alignment_from_full_cds', lambda **kw: tips)
    paths = {}
    original_runner = structural_alphabet._run_iqtree_direct_3di
    def capture_paths(**kwargs):
        paths.update(original_runner(**kwargs))
        return paths
    monkeypatch.setattr(structural_alphabet, '_run_iqtree_direct_3di', capture_paths)
    states, orders, _ = structural_alphabet.build_3di_state_direct(g)
    raw = ete.PhyloNode(paths['treefile'], format=1)
    fitted = tree.add_numerical_node_labels(tree.transfer_root(raw, tr))
    rows = pd.read_csv(paths['state'], sep='\t', comment='#')
    columns = [c for c in rows if c.startswith('p_')]
    order_indices = ['0123456789ABCDEFGHIJ'.index(c[2:]) for c in columns]
    for node in fitted.traverse():
        if ete.is_leaf(node) or ete.is_root(node):
            continue
        idx = int(ete.get_prop(node, 'numerical_label'))
        source = rows.loc[rows.Node == node.name].sort_values('Site')[columns].to_numpy()
        np.testing.assert_allclose(states[idx][:, order_indices], source, atol=1e-12)
        # Independently condition at this vertex using both sides of every edge.
        def message(current, previous):
            current_id = int(ete.get_prop(current, 'numerical_label'))
            if ete.is_leaf(current):
                return states[current_id]
            value = np.ones((len(base), 20))
            neighbors = list(ete.get_children(current))
            if current.up is not None:
                neighbors.append(current.up)
            for neighbor in neighbors:
                if neighbor is previous:
                    continue
                distance = neighbor.dist if neighbor.up is current else current.dist
                value *= message(neighbor, current) @ expm(g['3di_q'] * distance).T
                value /= value.max(axis=1, keepdims=True)
            return value
        independent = message(node, None) * g['3di_pi']
        independent /= independent.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(states[idx], independent, atol=2e-4, rtol=1e-4)

    expectation_3di.validate_context(g, len(list(tr.traverse())), len(base))
    assert states.shape == (len(list(tr.traverse())), len(base), 20)
    assert '_precomputed_tip_invariant_site_mask' not in g
    observed = np.array([list(orders).index(s) for s in alphabet])
    assert np.all(g['3di_pi'][observed] > 0)
    absent = np.flatnonzero(g['3di_pi'] == 0)
    np.testing.assert_allclose(g['3di_q'][absent], 0)
    assert np.all(g['3di_branch_lengths'] >= 0)
    assert np.array_equal(g['3di_orders'], orders)


def test_consistent_state_label_permutation_preserves_N_counts():
    g = _context()
    # Unequal exchangeabilities make this sensitive to matrix/state misalignment.
    ex = np.random.default_rng(99).uniform(.1, 3, (20, 20))
    q = (ex + ex.T) / 2
    np.fill_diagonal(q, 0)
    np.fill_diagonal(q, -q.sum(axis=1))
    q /= -np.diag(q).mean()
    g['3di_q'] = q
    permutation = np.random.default_rng(37).permutation(20)
    permuted = dict(g, state_nsy=g['state_nsy'][:, :, permutation])
    permuted['3di_q'] = q[np.ix_(permutation, permutation)]
    original = omega._get_fused_expected_sparse_reducer(g, 'nsy', ['any2any', 'any2spe'])
    relabeled = omega._get_fused_expected_sparse_reducer(permuted, 'nsy', ['any2any', 'any2spe'])
    assert relabeled['total'] == pytest.approx(original['total'], abs=1e-12)
    for stat in original['projections']:
        a = original['projections'][stat].toarray()
        b = relabeled['projections'][stat].toarray()
        # The sum of products across target-state columns is label invariant.
        np.testing.assert_allclose(a @ a.T, b @ b.T, atol=1e-12)


def test_scan_does_not_require_a_model_expectation_context():
    g = dict(nonsyn_recode='3di20', expectation_method='codon_model',
             sa_asr_mode='translate', subcommand='scan')
    assert not expectation_3di.required(g)
    expectation_3di.validate_options(g)


def test_root_posterior_matches_explicit_likelihood_with_missing_tip():
    g = _context()
    state = np.zeros((3, 4, 20))
    leaves = list(ete.iter_leaves(g['tree']))
    a, b = [int(ete.get_prop(n, 'numerical_label')) for n in leaves]
    state[a, 0, 0] = state[b, 0, 1] = 1
    state[a, 1, 0] = 1  # One missing tip; two wholly missing sites.
    actual = expectation_3di.root_posterior(g, g['tree'], state)
    pa = expm(g['3di_q'] * leaves[0].dist)
    pb = expm(g['3di_q'] * leaves[1].dist)
    expected0 = g['3di_pi'] * pa[:, 0] * pb[:, 1]
    expected1 = g['3di_pi'] * pa[:, 0]
    np.testing.assert_allclose(actual[0], expected0 / expected0.sum(), atol=1e-12)
    np.testing.assert_allclose(actual[1], expected1 / expected1.sum(), atol=1e-12)
    np.testing.assert_array_equal(actual[2:], 0)


def test_native_filter_uses_3di_mask_after_cache_reload(tmp_path):
    g = _context()
    g.update(sa_state_cache_file=str(tmp_path / 'filtered.npz'),
             state_cdn=np.ones((3, 4, 1)), num_input_site=4,
             drop_invariant_tip_sites_mode='tip_invariant')
    # Codon tips have no sequence here: fallback would keep all sites.
    g['3di_tip_invariant_mask'] = np.array([True, False, True, False])
    parser_misc._write_3di_state_cache(g, None, (3, 4, 1), g['state_nsy'], g['3di_orders'])
    fresh = {k: v for k, v in g.items() if k not in expectation_3di.CONTEXT_KEYS}
    assert parser_misc._try_load_3di_state_cache(fresh, None, (3, 4, 1))[2] is None
    out = parser_misc.drop_invariant_tip_sites(fresh)
    np.testing.assert_array_equal(out['site_index_alignment'], [1, 3])
    assert out['state_nsy'].shape[1] == 2
    expectation_3di.validate_context(out, 3, 2)
