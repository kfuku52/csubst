from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from csubst import empirical_urn, ete, tree
from csubst.substitution_sparse import SparseSubstitutionTensor


def test_cv_site_profile_uses_only_observed_cells():
    counts = np.array([[0., 3., 1.], [0., 3., 0.], [0., 0., 1.]])
    eligible = np.array([[True, True, True], [True, True, False], [True, False, True]])
    rate, weight = empirical_urn.fit_site_rates(counts, eligible)
    np.testing.assert_allclose(rate, [0., 2.25, .75])
    assert weight == 0.


def test_cv_site_profile_uses_uniform_when_branch_patterns_disagree():
    rate, weight = empirical_urn.fit_site_rates(np.eye(3), np.ones((3, 3), dtype=bool))
    np.testing.assert_array_equal(rate, np.ones(3))
    assert weight == 1.


def _tensor(events):
    matrix = csr_matrix(events.transpose(0, 2, 3, 4, 1).reshape(len(events), -1))
    result = SparseSubstitutionTensor(events.shape, events.dtype, matrix=matrix)
    result.eligible = np.ones(events.shape[:2], dtype=bool)
    return result


def _example():
    tr = tree.add_numerical_node_labels(ete.PhyloNode(
        '(((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1);', format=1,
    ))
    ids = {node.name: int(ete.get_prop(node, 'numerical_label')) for node in ete.iter_leaves(tr)}
    size = len(tuple(tr.traverse()))
    state = np.broadcast_to([.05, .45, .4, .1], (size, 4, 4))
    n = np.zeros((size, 4, 1, 2, 2))
    s = np.zeros((size, 4, 2, 2, 2))
    n[:, 0, 0, 0, 1] = .3
    n[:, 1, 0, 1, 0] = .2
    s[:, 0, 0, 0, 1] = .4
    s[:, 2, 0, 1, 0] = .3
    g = {'tree': tr, 'state_cdn': state, 'amino_acid_orders': ['A', 'B'],
         'synonymous_indices': {'A': [0, 1], 'B': [2, 3]}}
    return g, ids, n, s


def test_pair_uses_its_fold_outside_training_even_if_other_pairs_are_requested():
    g, ids, n, s = _example()
    g['asrv'] = 'empirical'
    stats = ('any2any', 'any2spe')
    first = np.array([[ids['A'], ids['E']]])
    only_pair = empirical_urn.expected_counts(g, first, _tensor(n), _tensor(s), stats)
    # Keep the branch-wide N+S count fixed while swapping the N destination
    # on an evaluation branch. It cannot affect this pair's training fit.
    n[ids['A'], 0, 0, 1, 0] = n[ids['A'], 0, 0, 0, 1]
    n[ids['A'], 0, 0, 0, 1] = 0.
    batch = np.concatenate((first, [[ids['B'], ids['F']]]))
    with_more_pairs = empirical_urn.expected_counts(g, batch, _tensor(n), _tensor(s), stats)
    for key, counts in only_pair.items():
        assert np.isfinite(counts).all()
        np.testing.assert_allclose(with_more_pairs[key][0], counts[0])
    for kind in ('ECN', 'ECS'):
        assert (with_more_pairs[kind + 'any2any'] >= with_more_pairs[kind + 'any2spe']).all()


def test_all_branch_fit_includes_evaluation_branches_and_preserves_masks():
    g, ids, n, s = _example()
    g['asrv'] = 'empirical_all'
    trio = np.array([[ids['A'], ids['E'], ids['F']]])
    pair = trio[:, :2]
    stats = ('any2any', 'any2spe')

    def counts(combos):
        return empirical_urn.expected_counts(g, combos, _tensor(n), _tensor(s), stats)

    initial = counts(pair)
    assert set(pair[0]).issubset(g['_asrv_training_ids'])
    n[ids['A'], 0, 0, 1, 0] = n[ids['A'], 0, 0, 0, 1]
    n[ids['A'], 0, 0, 0, 1] = 0.
    changed = counts(pair)
    assert not np.isclose(changed['ECNany2spe'], initial['ECNany2spe']).item()
    # Add an invalid root branch in a second row. It must not affect the first.
    root = int(ete.get_prop(g['tree'], 'numerical_label'))
    batch = np.concatenate((trio, [[ids['A'], ids['E'], root]]))
    for key, vals in counts(batch).items():
        assert np.isnan(vals[1])
        assert 0 < vals[0] < changed[key][0]


def test_all_branch_n_uses_observed_branch_totals_and_pooled_destination_shares():
    g, ids, n, s = _example()
    g['asrv'] = 'empirical_all'
    a, e, f, z = (ids[name] for name in ('A', 'E', 'F', 'G'))
    # Give each branch one expected event (N+S) across four equally weighted
    # sites. E has the opposite N destination to A, and G has only S events.
    total = np.full(len(n), .5)
    total[[a, e, f, z]] = [.2, .8, .3, 0.]
    sites = n.shape[1]
    n.fill(0.)
    s.fill(0.)
    n[:, :, 0, 0, 1] = total[:, None] / sites
    n[e, :, 0, 0, 1] = 0.
    n[e, :, 0, 1, 0] = total[e] / sites
    s[:, :, 0, 0, 1] = (1. - total[:, None]) / sites
    nonsyn, syn = _tensor(n), _tensor(s)
    g['_empirical_urn_fit'] = empirical_urn.fit_all_branches(g, nonsyn, syn)
    # Equal parental amino-acid frequencies make the destination shares
    # equal to the pooled observed proportions on these eligible branches.
    dest = n[g['_asrv_training_ids']].sum(axis=(0, 1, 2, 3))
    share = dest / dest.sum()
    for combos in (np.array([[a, e], [e, z]]), np.array([[a, e, f], [a, e, z]])):
        k = combos.shape[1]
        expected = total[combos].prod(axis=1) / sites**(k - 1)
        counts = empirical_urn.expected_counts(g, combos, nonsyn, syn, ('any2any', 'any2spe'))
        np.testing.assert_allclose(counts['ECNany2any'], expected)
        np.testing.assert_allclose(counts['ECNany2spe'], expected * (share**k).sum())
        assert counts['ECSany2spe'][1] > 0.

    g['asrv'] = 'empirical'
    legacy = empirical_urn.expected_counts(g, np.array([[e, z]]), nonsyn, syn, ('any2any',))
    assert legacy['ECNany2any'][0] > 0.


def test_s_pool_preserves_branch_mass_and_accounts_for_masked_parental_opportunities():
    groups = [np.array([2, 0]), np.array([3, 1]), np.array([4])]
    state = np.eye(5)[[[0, 3, 2], [1, 0, 3]]]
    eligible = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
    counts = np.zeros((4, 3, 2))
    counts[0, 0, 0] = 6.  # destination 2
    counts[1, 1, 1] = 4.  # destination 1
    counts[3, 0, 1] = 100.  # outside training
    data = SimpleNamespace(groups=groups, parents=[0, 1, 0, 1], eligible=eligible, syn_totals=counts)
    train = np.arange(3)
    rate = np.array([1., 2., 3.])
    draws = empirical_urn._pool_s_destinations(state, data, train, rate)
    # Rates per opportunity are 6 / (2 * 1) and 4 / (2 * 2 + 2 * 3).
    # Branch 0 allocates 6 draws in the ratio 3 : 0.8 across two families.
    expected = np.zeros_like(counts)
    expected[0, 0, 0], expected[0, 1, 1] = 90 / 19, 24 / 19
    expected[1, 1, 1] = 4.
    np.testing.assert_allclose(draws, expected)
    counts[train] = 0.
    np.testing.assert_array_equal(empirical_urn._pool_s_destinations(state, data, train, rate), 0.)


def test_only_all_branch_s_scores_are_invariant_to_swapping_observed_destinations():
    g, ids, n, s = _example()
    a, c = ids['A'], ids['C']
    s[c, 0, 0, 0, 1] = 0.
    s[c, 0, 1, 1, 0] = .4
    stats = ('any2spe', 'any2any')

    def score(mode, branches):
        g['asrv'] = mode
        return empirical_urn.expected_counts(g, np.array([branches]), _tensor(n), _tensor(s), stats)

    pair = [a, ids['E']]
    cases = [('empirical', pair), ('empirical_all', pair), ('empirical_all', [*pair, ids['F']])]
    before = [score(*case) for case in cases]
    s[[a, c], 0] = s[[c, a], 0]
    after = [score(*case) for case in cases]
    assert not np.allclose(before[0]['ECSany2spe'], after[0]['ECSany2spe'])
    for initial, changed in zip(before[1:], after[1:]):
        for key in initial:
            np.testing.assert_allclose(initial[key], changed[key], rtol=1e-13)


def test_same_destination_and_any_destination_triple_have_distinct_products(monkeypatch):
    probs = np.array([[[.2, .3], [.1, .0]], [[.4, .1], [.1, .2]], [[.7, .0], [.1, .3]]])
    monkeypatch.setattr(empirical_urn.omega, 'wallenius_inclusion_profiles', lambda w, draws, g: w)
    out = {stat: np.zeros(1) for stat in ('any2any', 'any2spe')}
    empirical_urn._add_group_overlaps(out, np.array([[0, 1, 2]]), probs, np.ones((3, 2)), {})
    np.testing.assert_allclose(out['any2spe'], [.2 * .4 * .7 + .1 ** 3])
    np.testing.assert_allclose(out['any2any'], [.5 * .5 * .7 + .1 * .3 * .4])


def test_regional_n_preserves_s_and_accounts_for_removed_alignment_sites():
    import pytest
    g, ids, n, s = _example()
    g.update(asrv='empirical_all', site_index_alignment=np.array([0, 2, 5, 7]), num_input_site=9)
    pairs = np.array([[ids['A'], ids['E']]])
    ns = n.sum(axis=(2, 3, 4))
    groups = empirical_urn._regional_n_counts(dict(g, urn_n_regions='2-2,3-6'), ns)
    np.testing.assert_array_equal(groups[0][0], [0, 3])
    np.testing.assert_array_equal(groups[1][0], [1, 2])
    np.testing.assert_allclose(groups[0][1] + groups[1][1], ns.sum(axis=1))
    with pytest.raises(ValueError, match='overlap'):
        empirical_urn.parse_n_regions('1-4,4-9')
    with pytest.raises(ValueError, match='beyond'):
        empirical_urn._regional_n_counts(dict(g, urn_n_regions='9-10'), ns)

    def estimate(**options):
        return empirical_urn.expected_counts(dict(g, **options), pairs, _tensor(n), _tensor(s), ['any2spe'])

    whole, one, split = estimate(), estimate(urn_n_regions='1-9'), estimate(urn_n_regions='1-1')
    for key in whole:
        np.testing.assert_allclose(whole[key], one[key], rtol=1e-12)
    np.testing.assert_array_equal(whole['ECSany2spe'], split['ECSany2spe'])
    assert not np.allclose(whole['ECNany2spe'], split['ECNany2spe'])


def test_regions_require_the_joint_empirical_all_urn():
    import pytest
    from csubst import asrv
    base = dict(urn_n_regions='1-10', asrv='auto', expectation_method='urn')
    assert asrv.effective_mode(base) == 'empirical_all'
    for options in [dict(expectation_method='codon_model'), dict(asrv='empirical')]:
        with pytest.raises(ValueError, match='empirical_all'):
            asrv.validate_options(dict(base, **options))
    with pytest.raises(ValueError, match='Wallenius'):
        asrv.effective_mode(dict(base, urn_model='fisher'))
