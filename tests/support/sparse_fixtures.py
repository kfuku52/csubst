import numpy as np

from csubst import substitution


def toy_dense_tensor():
    """Build a small dense substitution tensor with two groups."""
    sub = np.zeros((3, 4, 2, 3, 3), dtype=np.float64)
    sub[0, 0, 0, 0, 1] = 0.2
    sub[1, 0, 0, 0, 1] = 0.3
    sub[2, 1, 0, 2, 1] = 1.1
    sub[0, 3, 1, 1, 2] = 0.8
    sub[1, 2, 1, 1, 2] = 0.6
    sub[2, 3, 1, 0, 0] = 0.5
    return sub


def toy_reducer_tensor():
    """Build the compact tensor used by sparse reducer tests."""
    sub = np.zeros((3, 2, 1, 2, 2), dtype=np.float64)
    sub[0, 0, 0, :, :] = [[0.0, 0.2], [0.1, 0.0]]
    sub[1, 0, 0, :, :] = [[0.0, 0.5], [0.2, 0.0]]
    sub[2, 0, 0, :, :] = [[0.0, 0.4], [0.3, 0.0]]
    sub[0, 1, 0, :, :] = [[0.0, 0.1], [0.0, 0.0]]
    sub[1, 1, 0, :, :] = [[0.0, 0.1], [0.3, 0.0]]
    sub[2, 1, 0, :, :] = [[0.0, 0.2], [0.1, 0.0]]
    return sub


def large_sparse_reducer_tensor(num_branch=40, num_site=12):
    """Build a deterministic moderately sparse tensor for Gram-path tests."""
    rng = np.random.default_rng(7)
    sub = np.zeros((num_branch, num_site, 1, 2, 2), dtype=np.float64)
    vals01 = rng.random((num_branch, num_site), dtype=np.float64)
    vals10 = rng.random((num_branch, num_site), dtype=np.float64)
    vals01[vals01 < 0.6] = 0
    vals10[vals10 < 0.6] = 0
    sub[:, :, 0, 0, 1] = vals01
    sub[:, :, 0, 1, 0] = vals10
    return substitution.dense_to_sparse_sub_tensor(sub, tol=0)
