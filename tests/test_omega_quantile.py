import numpy as np
import pandas as pd
import pytest

from csubst import omega


def test_weighted_sample_without_replacement_masks_excludes_zero_probability_sites():
    p = np.array([0.7, 0.3, 0.0, 0.0], dtype=np.float64)
    masks = omega._weighted_sample_without_replacement_masks(p=p, size=2, niter=128)

    assert masks.shape == (128, 4)
    assert masks.dtype == bool
    assert masks[:, 2].sum() == 0
    assert masks[:, 3].sum() == 0
    np.testing.assert_array_equal(masks.sum(axis=1), np.full((128,), 2))


def test_weighted_sample_without_replacement_masks_raises_when_size_exceeds_positive_sites():
    p = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    with pytest.raises(ValueError):
        omega._weighted_sample_without_replacement_masks(p=p, size=2, niter=8)


def test_get_permutations_fast_returns_expected_shape_and_bounds():
    cb_ids = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
    sub_branches = np.array([2, 3, 5], dtype=np.int64)
    p = np.array([0.2, 0.1, 0.25, 0.15, 0.1, 0.1, 0.05, 0.05], dtype=np.float64)

    out = omega._get_permutations_fast(
        cb_ids=cb_ids,
        sub_branches=sub_branches,
        p=p / p.sum(),
        niter=256,
    )

    assert out.shape == (3, 256)
    assert out.dtype == np.int32
    assert out.min() >= 0
    assert out[0, :].max() <= min(sub_branches[0], sub_branches[1])
    assert out[1, :].max() <= min(sub_branches[1], sub_branches[2])
    assert out[2, :].max() <= min(sub_branches[0], sub_branches[2])


def test_resolve_quantile_parallel_plan_falls_back_for_small_workload():
    n_jobs, chunk_factor = omega._resolve_quantile_parallel_plan(
        cb_rows=1,
        num_categories=200,
        quantile_niter=1000,
        requested_n_jobs=4,
        requested_chunk_factor=1,
    )
    assert n_jobs == 1
    assert chunk_factor == 1


def test_resolve_quantile_parallel_plan_keeps_parallel_for_large_workload():
    n_jobs, chunk_factor = omega._resolve_quantile_parallel_plan(
        cb_rows=1000,
        num_categories=300,
        quantile_niter=1000,
        requested_n_jobs=4,
        requested_chunk_factor=1,
    )
    assert n_jobs == 4
    assert chunk_factor == 4


def test_get_cod_skips_when_required_columns_missing():
    cb = pd.DataFrame(
        {
            "OCNany2spe": [1.0, 2.0],
            "OCSany2spe": [2.0, 3.0],
        }
    )
    out = omega.get_CoD(cb.copy(), g={"float_tol": 1e-12})
    assert "OCNCoD" not in out.columns
    assert "OCSCoD" not in out.columns
