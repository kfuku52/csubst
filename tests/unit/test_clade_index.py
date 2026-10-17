import pickle

import numpy as np
import pytest

from csubst import ete, foreground, scan_permutation, tree


@pytest.mark.parametrize('text', [
    '((A:1,B:1)X:1,(C:1,D:1)Y:1)R;',
    '(((A:1):1,B:1):1,C:1,D:1)R;',
])
def test_lazy_clades_preserve_breadth_order_leaf_order_and_noncontiguous_ids(text):
    tr = tree.add_numerical_node_labels(ete.PhyloNode(text, format=1))
    for index, node in enumerate(tr.traverse()):
        ete.set_prop(node, 'numerical_label', index * 7 + 3)
    cache = foreground._build_trait_clade_permutation_cache({'tree': tr}, 'trait')
    for node in tr.traverse():
        if ete.is_root(node):
            continue
        bid = int(ete.get_prop(node, 'numerical_label'))
        index = cache['branch_id_to_index'][bid]
        ids = [int(ete.get_prop(child, 'numerical_label')) for child in node.traverse()]
        assert cache['descendant_branch_ids_by_bid'][bid].tolist() == ids
        assert cache['descendant_branch_ids_by_index'][index].tolist() == ids
        assert cache['descendant_indices_by_index'][index].tolist() == [cache['branch_id_to_index'][i] for i in ids]
        assert cache['leaf_names_by_index'][index] == tuple(ete.get_leaf_names(node))
    restored = pickle.loads(pickle.dumps(cache))
    assert restored['descendant_indices_by_index'].index._cache_bytes == 0
    for index in range(len(cache['branch_ids'])):
        np.testing.assert_array_equal(cache['descendant_indices_by_index'][index],
                                      restored['descendant_indices_by_index'][index])


@pytest.mark.parametrize('enumerate_space', [True, False])
def test_interval_sampler_has_identical_assignments_and_rng_draws(monkeypatch, enumerate_space):
    if not enumerate_space:
        monkeypatch.setattr(scan_permutation, 'ENUMERATION_PROPOSAL_LIMIT', 0)
    tr = tree.add_numerical_node_labels(ete.PhyloNode('((A:1,B:1)X:1,(C:1,D:1)Y:1)R;', format=1))
    cache = foreground._build_trait_clade_permutation_cache({'tree': tr}, 'trait')
    descendants = cache['descendant_indices_by_index']
    observed = ((cache['branch_id_to_index'][int(ete.get_prop(node, 'numerical_label'))]
                 for node in ete.get_children(tr)),)
    observed = (tuple(observed[0]),)
    args = dict(bin_array=np.zeros(len(descendants), int), eligible=np.ones(len(descendants), bool), observed=observed)
    lazy = scan_permutation.build_plan(**args, descendants=descendants)
    materialized = scan_permutation.build_plan(**args, descendants=[set(v) for v in descendants])
    assert lazy.configurations == materialized.configurations
    a, b = np.random.default_rng(71), np.random.default_rng(71)
    for _ in range(100):
        assert scan_permutation.sample_configuration(lazy, a) == scan_permutation.sample_configuration(materialized, b)


def test_skewed_clade_storage_is_linear_and_serialization_excludes_listing_cache():
    text = 'L0:1'
    for index in range(1, 512):
        text = '(' + text + ',L{}:1):1'.format(index)
    tr = tree.add_numerical_node_labels(ete.PhyloNode(text + ';', format=1))
    cache = foreground._build_trait_clade_permutation_cache({'tree': tr}, 'trait')
    owner = cache['descendant_indices_by_index'].index
    arrays = [value for value in vars(owner).values() if isinstance(value, np.ndarray)]
    assert sum(value.nbytes for value in arrays) < len(owner.branch_ids) * 64
    before = len(pickle.dumps(cache))
    for descendants in (cache['descendant_indices_by_index'], cache['descendant_branch_ids_by_index']):
        for _ in descendants:
            pass
    assert owner._cache_bytes <= 8 * 1024**2
    assert len(pickle.dumps(cache)) == before
