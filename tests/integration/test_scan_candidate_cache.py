import numpy as np
import pandas as pd
import pytest

from csubst import site_storage, substitution, substitution_scan as scan


@pytest.mark.parametrize('storage', ['dense', 'sparse', 'disk'])
def test_fixed_candidate_vectors_preserve_frame_rates_and_ownership(storage):
    dense = np.zeros((4, 2, 1, 2, 2))
    dense[1, 0, 0, 0, 1], dense[3, 0, 0, 0, 1] = .75, .25
    tensor = dense
    if storage == 'sparse':
        tensor = substitution.dense_to_sparse_sub_tensor(dense)
    elif storage == 'disk':
        tensor = site_storage.SiteEventTensor(dense.shape)
        tensor.write_block(0, np.moveaxis(dense, 1, 0))
        tensor.seal()
    context = {}
    frame = scan._cached_posterior_events(context, tensor, 0, [0], [1], 1e-12)
    vector = scan._cached_posterior_events(context, tensor, 0, [0], [1], 1e-12, branch_values=True)
    assert isinstance(frame, pd.DataFrame)
    assert scan._cached_posterior_events(context, tensor, 0, [0], [1], 1e-12, branch_values=True) is vector
    meta = pd.DataFrame({'branch_id': [3, 1, -1, 10**12], 'parent_id': [0] * 4,
                         'raw_length': [.5] * 4, 'sn_rescaled_length': [.5] * 4,
                         'n_rescaled_length': [.5] * 4})
    options = (meta, np.ones((4, 2, 2)), 0, [0], [1], [1], 'raw', 'raw_branch_length')
    np.testing.assert_equal(scan._rate_summary(frame, *options), scan._rate_summary(vector, *options))
    with pytest.raises(ValueError, match='read-only'):
        vector.values[1] = 0
    selected = vector.select(np.array([1, 3]))
    selected[:] = 0
    np.testing.assert_array_equal(vector.values, [0., .75, 0., .25])
    assert context['_event_cache'].nbytes <= context['_event_cache'].max_bytes


def test_unretainable_vectors_and_single_pass_calls_keep_dataframe_path():
    tensor = np.zeros((100, 1, 1, 2, 2))
    context = {'_event_cache': scan._CandidateCache(max_bytes=300)}
    frame = scan._cached_posterior_events(context, tensor, 0, [0], [1], 1e-12, branch_values=True)
    assert isinstance(frame, pd.DataFrame)
    assert context['_event_cache'].nbytes <= 300
    disabled = {'_cache_candidates': False}
    for state in (None, disabled):
        result = scan._cached_posterior_events(state, tensor, 0, [0], [1], 1e-12, branch_values=True)
        pd.testing.assert_frame_equal(result, frame)
    assert '_event_cache' not in disabled


@pytest.mark.parametrize('ids', [[3, 1, 1, 0], [3, -1, 10**12, 1], [], [-1, 10**12]])
def test_prepared_event_selection_preserves_order_missing_ids_and_vector_size_changes(ids):
    meta = pd.DataFrame({'branch_id': ids})
    data, _, _ = scan._ScanRateMetadata(meta).prepare([1], None)
    for size in (4, 2, 4):
        frame = pd.DataFrame({'branch_id': np.arange(size), 'event_pp': np.arange(size) / 4})
        vector = scan._PosteriorBranchValues(frame, size)
        actual = data.select_events(vector)
        expected = np.array([bid / 4 if 0 <= bid < size else 0. for bid in ids])
        np.testing.assert_array_equal(actual, expected, strict=True)
        actual[:] = -1
        np.testing.assert_array_equal(data.select_events(vector), expected, strict=True)
        assert not vector.values.flags.writeable


def test_numeric_candidate_cache_eviction_and_fit_keys():
    tensor = np.zeros((4, 3, 1, 2, 2))
    tensor[1, :, 0, 0, 1] = [.1, .2, .3]
    context = {'_event_cache': scan._CandidateCache(max_items=2)}
    records = [scan._cached_posterior_events(context, tensor, site, [0], [1], 1e-12,
                                           branch_values=True) for site in range(3)]
    assert len(context['_event_cache'].entries) == 2
    refreshed = scan._cached_posterior_events(context, tensor, 0, [0], [1], 1e-12, branch_values=True)
    assert refreshed is not records[0]
    replacement = tensor.copy()
    replacement[1, 0, 0, 0, 1] = .9
    changed = scan._cached_posterior_events(context, replacement, 0, [0], [1], 1e-12, branch_values=True)
    assert changed.values[1] == .9 and refreshed.values[1] == .1
