import numpy as np
import pytest

from csubst import randomness


@pytest.mark.parametrize("value,expected", [(None, 1), (0, 0), (7, 7), ("9", 9), (-1, -1)])
def test_normalize_seed_accepts_documented_values(value, expected):
    assert randomness.normalize_seed(value) == expected


@pytest.mark.parametrize("value", [-2, 1.0, 1.5, "1.5", "", True, np.bool_(False)])
def test_normalize_seed_rejects_non_integer_or_out_of_range_values(value):
    with pytest.raises(ValueError, match="random_seed"):
        randomness.normalize_seed(value)


def test_derived_generators_are_reproducible_and_component_isolated():
    first = randomness.generator(23, "omega", "N").integers(0, 2**31, size=16)
    second = randomness.generator(23, "omega", "N").integers(0, 2**31, size=16)
    other_component = randomness.generator(23, "omega", "S").integers(0, 2**31, size=16)
    other_seed = randomness.generator(24, "omega", "N").integers(0, 2**31, size=16)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other_component)
    assert not np.array_equal(first, other_seed)


def test_next_generator_advances_namespaced_stream_without_cross_namespace_interference():
    g = {"random_seed": 11}
    a0 = randomness.next_generator(g, "a").integers(0, 2**31, size=8)
    b0 = randomness.next_generator(g, "b").integers(0, 2**31, size=8)
    a1 = randomness.next_generator(g, "a").integers(0, 2**31, size=8)

    reference = {"random_seed": 11}
    expected_a0 = randomness.next_generator(reference, "a").integers(0, 2**31, size=8)
    expected_a1 = randomness.next_generator(reference, "a").integers(0, 2**31, size=8)

    np.testing.assert_array_equal(a0, expected_a0)
    np.testing.assert_array_equal(a1, expected_a1)
    assert not np.array_equal(a0, b0)
