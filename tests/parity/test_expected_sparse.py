import numpy as np

from csubst import expected_sparse


def test_csr_row_builder_matches_dense_with_empty_rows():
    builder = expected_sparse.CSRRowBuilder(num_row=5, num_column=9, initial_capacity=1)
    builder.append(1, np.array([2, 7]), np.array([1.5, -2.0]))
    builder.append(3, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
    builder.append(4, np.array([0]), np.array([3.0]))

    observed = builder.finalize().toarray()
    expected = np.zeros((5, 9), dtype=np.float64)
    expected[1, [2, 7]] = [1.5, -2.0]
    expected[4, 0] = 3.0
    np.testing.assert_allclose(observed, expected, atol=0.0)


def test_csr_row_builder_randomized_parity():
    rng = np.random.default_rng(20260716)
    for _ in range(25):
        num_row = int(rng.integers(1, 40))
        num_column = int(rng.integers(1, 200))
        expected = rng.normal(size=(num_row, num_column))
        expected[rng.random(expected.shape) > rng.uniform(0.01, 0.3)] = 0.0
        builder = expected_sparse.CSRRowBuilder(num_row, num_column, initial_capacity=2)
        for row_id in range(num_row):
            indices = np.flatnonzero(expected[row_id] != 0).astype(np.int32)
            builder.append(row_id, indices, expected[row_id, indices])
        observed = builder.finalize().toarray()
        np.testing.assert_allclose(observed, expected, atol=0.0)
