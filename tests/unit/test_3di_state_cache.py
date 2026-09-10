import numpy as np
import pytest

from csubst import parser_misc, structural_alphabet


def _cache(tmp_path):
    source = tmp_path / 'input.fa'
    source.write_text('>A\nATGATG\n')
    g = {'alignment_file': str(source), 'float_type': np.float64,
         'sa_state_cache_file': str(tmp_path / 'states.npz'),
         '_precomputed_tip_invariant_site_mask': np.array([True, False])}
    shape = (2, 2, 61)
    states = np.zeros((2, 2, 20))
    states[0, :, 0] = 1  # The second branch is legitimately unloaded/missing.
    parser_misc._write_3di_state_cache(
        g, None, shape, states, structural_alphabet.get_3di_state_orders())
    return g, shape


def _rewrite(g, mutate):
    with np.load(g['sa_state_cache_file'], allow_pickle=False) as archive:
        fields = {key: archive[key] for key in archive.files}
    mutate(fields)
    np.savez_compressed(g['sa_state_cache_file'], **fields)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.1, 1.1, 0.5])
def test_3di_cache_rejects_invalid_probabilities(tmp_path, value):
    g, shape = _cache(tmp_path)
    _rewrite(g, lambda fields: fields['state_nsy'].__setitem__((0, 0, 0), value))
    states, orders, error = parser_misc._try_load_3di_state_cache(g, None, shape)
    assert states is None and orders is None
    assert 'probabilit' in error


def test_3di_cache_rejects_wrong_state_order(tmp_path):
    g, shape = _cache(tmp_path)
    _rewrite(g, lambda fields: fields.__setitem__('state_orders', fields['state_orders'][::-1]))
    states, orders, error = parser_misc._try_load_3di_state_cache(g, None, shape)
    assert states is None and orders is None
    assert 'state order' in error


def test_3di_cache_accepts_missing_rows_and_iqtree_rounding(tmp_path):
    g, shape = _cache(tmp_path)
    _rewrite(g, lambda fields: fields['state_nsy'].__setitem__((0, 0, 0), 0.99999))
    states, orders, error = parser_misc._try_load_3di_state_cache(g, None, shape)
    assert error is None
    assert states[0, 0, 0] == pytest.approx(0.99999)
    np.testing.assert_array_equal(states[1], 0)
    np.testing.assert_array_equal(orders, structural_alphabet.get_3di_state_orders())


def test_3di_default_does_not_reuse_legacy_prostt5_cache(tmp_path):
    g, shape = _cache(tmp_path)
    # Rewrite metadata using the old explicit predictor, preserving valid tensors.
    states = np.zeros((2, 2, 20))
    states[:, :, 0] = 1
    parser_misc._write_3di_state_cache(dict(g, sa_backend='prostt5'), None, shape, states,
                                     structural_alphabet.get_3di_state_orders())
    cached, _, error = parser_misc._try_load_3di_state_cache(g, None, shape)
    assert cached is None and error == 'cache metadata mismatch.'
    assert parser_misc._try_load_3di_state_cache(dict(g, sa_backend='prostt5'), None, shape)[2] is None


def test_direct_urn_cache_restores_full_site_selection_mask(tmp_path):
    g, shape = _cache(tmp_path)
    fresh = {key: value for key, value in g.items() if not key.startswith('_precomputed')}
    assert parser_misc._try_load_3di_state_cache(fresh, None, shape)[2] is None
    np.testing.assert_array_equal(fresh['_precomputed_tip_invariant_site_mask'], [True, False])
    _rewrite(g, lambda fields: fields.pop('tip_invariant_mask'))
    assert parser_misc._try_load_3di_state_cache(fresh, None, shape)[0] is None
