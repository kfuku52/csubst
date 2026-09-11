import copy
import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from csubst import site_storage, substitution, substitution_scan


@pytest.mark.parametrize('block_size', [1, 3, 9])
@pytest.mark.parametrize('threshold', [0, .3, .9])
def test_stored_events_match_dense_scan_and_threshold_views(block_size, threshold):
    values = np.random.default_rng(12).random((5, 7, 1, 4, 4))
    values[:, :, 0, np.arange(4), np.arange(4)] = 0
    values[0] = 0
    values[:, 2] = 0
    stored = site_storage.SiteEventTensor(values.shape)
    for start in range(0, 7, block_size):
        stored.write_block(start, np.moveaxis(values[:, start:start + block_size], 1, 0))
    stored.seal()
    called = substitution.apply_min_sub_pp(dict(min_sub_pp=threshold, ml_anc=False), stored)
    dense = values.copy()
    dense[dense < threshold] = 0
    np.testing.assert_array_equal(stored.read_site(1), values[:, 1])
    np.testing.assert_allclose(substitution.get_branch_sub_counts(called), dense.sum(axis=(1, 2, 3, 4)))
    np.testing.assert_allclose(substitution.get_site_sub_counts(called), dense.sum(axis=(0, 2, 3, 4)))
    pd.testing.assert_frame_equal(substitution_scan.extract_atomic_events(called),
                                  substitution_scan.extract_atomic_events(dense))
    for site in range(7):
        for source, dest in [([0, 1, 2, 3], [2]), ([0], [1, 2, 3]), ([1, 2], [2, 3])]:
            expected = substitution_scan.extract_candidate_posterior_events(dense, site, source, dest)
            actual = substitution_scan.extract_candidate_posterior_events(called, site, source, dest)
            pd.testing.assert_frame_equal(actual, expected)
    # Parent ownership must survive sharing and threshold views, and worker
    # serialization must never embed the complete array's payload.
    copied = copy.deepcopy(called)
    payload = pickle.dumps(called)
    borrowed = pickle.loads(payload)
    np.testing.assert_array_equal(borrowed.read_site(1), dense[:, 1])
    filename = Path(stored.filename)
    del stored, called, copied
    gc.collect()
    assert not filename.exists()


def test_site_storage_rejects_partial_and_truncated_data():
    stored = site_storage.SiteArray((2, 3, 7, 4), float, site_axis=2)
    with pytest.raises(ValueError, match='unfinished'):
        stored.read_site(0)
    with pytest.raises(ValueError, match='Incomplete'):
        stored.seal()
    data = np.arange(2 * 3 * 7 * 4).reshape(2, 3, 7, 4).astype(float)
    stored.write_block(0, np.moveaxis(data, 2, 0))
    stored.seal()
    np.testing.assert_array_equal(stored.read_site(5), data[:, :, 5])
    with pytest.raises(ValueError, match='read-only'):
        stored.write_block(0, np.moveaxis(data, 2, 0))
    with pytest.raises(IndexError):
        stored.read_site(7)
    with open(stored.filename, 'r+b') as handle:
        handle.truncate(10)
    with pytest.raises(ValueError, match='Truncated'):
        stored.read_site(0)
