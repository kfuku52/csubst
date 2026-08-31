import numpy as np
import pandas as pd
import pytest

from csubst import substitution


pytestmark = [pytest.mark.requires_cython, pytest.mark.native]


@pytest.mark.parametrize('source_kind', ['array', 'memmap', 'pandas'])
@pytest.mark.parametrize('alpha', [0.0, 1.0])
def test_native_normalization_accepts_readonly_inputs(tmp_path, source_kind, alpha):
    accelerator = substitution.substitution_cy
    if accelerator is None:
        pytest.skip('Compiled substitution extension is unavailable')
    weights = np.array([2.0, 0.0, 1.0])
    if source_kind == 'memmap':
        path = tmp_path / 'weights.bin'
        weights.tofile(path)
        weights = np.memmap(path, mode='r', dtype=np.float64, shape=(3,))
    elif source_kind == 'pandas':
        weights = pd.Series(weights).to_numpy()
    weights.flags.writeable = False
    mask = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=np.uint8)
    branches = np.arange(3, dtype=np.int64)
    mask.flags.writeable = False
    branches.flags.writeable = False
    observed = accelerator.normalize_branch_site_weights_double(weights, mask, branches, alpha)
    expected = ([[1., 0., 0.], [0., 0., 1.], [0., 0., 0.]] if alpha == 0
                else [[.75, .25, 0.], [0., 1/3, 2/3], [0., 0., 0.]])
    np.testing.assert_allclose(observed, expected, atol=1e-12)
    np.testing.assert_array_equal(weights, [2., 0., 1.])
    assert not weights.flags.writeable
