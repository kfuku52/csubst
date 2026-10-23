"""Conservation annotations on an alignment large enough for site blocks."""

import math

import numpy as np
import pytest

from csubst import substitution_scan


def test_conservation_preserves_missing_sites_ties_and_partial_last_block():
    state = np.zeros((128, 1025, 20), dtype=np.float64)
    state[:64, 1::3, 0] = 1
    state[64:, 1::3, 1] = 1
    state[:, 2::3, 1] = 1
    state[::4, 2::3, 1] = 0
    state.flags.writeable = False
    orders = np.array(list('ACDEFGHIKLMNPQRSTVWY'))

    result = substitution_scan._calc_conservation(state, orders, np.arange(128), 1e-12)

    assert np.isnan(result['conservation'][::3]).all()
    assert np.isnan(result['entropy'][::3]).all()
    np.testing.assert_array_equal(result['major_state'][::3], '')
    np.testing.assert_array_equal(result['valid_tip_count'][::3], 0)
    np.testing.assert_array_equal(result['conservation'][1::3], .5)
    assert result['entropy'][1::3] == pytest.approx(math.log(2) / math.log(20))
    np.testing.assert_array_equal(result['major_state'][1::3], 'A')
    np.testing.assert_array_equal(result['valid_tip_count'][1::3], 128)
    np.testing.assert_array_equal(result['conservation'][2::3], 1.)
    np.testing.assert_array_equal(result['entropy'][2::3], 0.)
    np.testing.assert_array_equal(result['major_state'][2::3], 'C')
    np.testing.assert_array_equal(result['valid_tip_count'][2::3], 96)
