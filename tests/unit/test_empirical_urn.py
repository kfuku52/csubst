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


def test_pair_uses_its_fold_outside_training_even_if_other_pairs_are_requested():
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
