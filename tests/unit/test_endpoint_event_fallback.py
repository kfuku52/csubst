"""Independent full-pair references for the one-to-one event fallback."""

import numpy as np
import pytest

from csubst import endpoint_io


@pytest.mark.parametrize('sites,dtypes,permutation,strided', [
    (7, (np.float64,) * 4, [0, 1, 2], False),
    (7, (np.float64,) * 4, [2, 0, 1], True),
    (7, (np.float32, np.float64, np.float32, np.float32), [1, 2, 0], True),
    (7, (np.float32, np.float32, np.float32, np.complex128), [2, 1, 0], False),
    (7, (np.int16,) * 4, [1, 0, 2], True),
    (0, (np.float32,) * 4, [0, 1, 2], False),
])
def test_one_to_one_fallback_matches_full_pair_contraction(
        monkeypatch, sites, dtypes, permutation, strided):
    rng = np.random.default_rng(704)
    left = (rng.random((sites * 2, 3)) * 4).astype(dtypes[0])
    right = (rng.random((sites * 2, 3)) * 4).astype(dtypes[1])
    left, right = (left[::2], right[::2]) if strided else (left[:sites], right[:sites])
    transition = (rng.random((3, 3)) * 4).astype(dtypes[2])
    mapping = np.eye(3, dtype=dtypes[3])[:, permutation]
    for values in (left, right, transition, mapping):
        values.flags.writeable = False
    originals = [values.copy() for values in (left, right, transition, mapping)]
    raw = left[:, :, None] * transition * right[:, None, :]
    expected = np.einsum('sij,ia,jb->sab', raw, mapping, mapping, optimize=True)
    diagonal = np.arange(3)
    expected[:, diagonal, diagonal] = 0

    monkeypatch.setattr(endpoint_io.substitution_sparse, 'substitution_sparse_cy', None)
    actual = endpoint_io._event_transform({}, ['N'], {'N': mapping})(left, right, transition)

    assert actual['N', 'events'].dtype == expected.dtype
    np.testing.assert_array_equal(actual['N', 'events'][:, 0], expected)
    for values, original in zip((left, right, transition, mapping), originals):
        np.testing.assert_array_equal(values, original)


def test_fractional_mapping_retains_full_pair_fallback(monkeypatch):
    left = np.array([[.1, .2, .7], [.4, .5, .1]])
    right = np.array([[.3, .2, .5], [.6, .1, .3]])
    transition = np.arange(9, dtype=float).reshape(3, 3) / 9
    mapping = np.array([[1., 0.], [.5, .5], [0., 1.]])
    raw = left[:, :, None] * transition * right[:, None, :]
    expected = np.einsum('sij,ia,jb->sab', raw, mapping, mapping, optimize=True)
    expected[:, np.arange(2), np.arange(2)] = 0

    monkeypatch.setattr(endpoint_io.substitution_sparse, 'substitution_sparse_cy', None)
    actual = endpoint_io._event_transform({}, ['N'], {'N': mapping})(left, right, transition)

    np.testing.assert_array_equal(actual['N', 'events'][:, 0], expected)
