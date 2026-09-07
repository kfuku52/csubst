import numpy as np
import pytest

from csubst import omega_statistics


@pytest.mark.parametrize("function", [
    omega_statistics._calc_raw_rate, omega_statistics._calc_raw_omega,
])
@pytest.mark.parametrize("numerator,denominator,expected", [
    (2.0, 4.0, 0.5),
    (0.0, [0.0, 1.0], [0.0, 0.0]),
    ([0.0, 2.0], [[0.0, 2.0], [1.0, 4.0]], [[0.0, 1.0], [0.0, 0.5]]),
])
def test_raw_statistics_support_broadcast_arraylike_inputs(function, numerator, denominator, expected):
    actual = function(numerator, denominator, 1e-12)
    np.testing.assert_array_equal(actual, expected)
