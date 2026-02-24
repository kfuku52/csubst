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


def test_get_permutations_fast_caps_oversized_branch_counts_to_positive_sites():
    cb_ids = np.array([[0, 1]], dtype=np.int64)
    sub_branches = np.array([5, 2], dtype=np.int64)
    p = np.array([0.0, 0.6, 0.4], dtype=np.float64)

    out = omega._get_permutations_fast(
        cb_ids=cb_ids,
        sub_branches=sub_branches,
        p=p,
        niter=128,
    )

    assert out.shape == (1, 128)
    assert out.dtype == np.int32
    assert out.min() >= 0
    assert out.max() <= 2


def test_get_permutations_fast_uses_branch_specific_probabilities_when_given_matrix():
    cb_ids = np.array([[1, 2], [2, 3]], dtype=np.int64)
    sub_branches = np.array([0, 2, 2, 2], dtype=np.int64)
    p_by_branch = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],  # branch 0 has no informative site weights
            [0.4, 0.3, 0.2, 0.1],
            [0.4, 0.3, 0.2, 0.1],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float64,
    )

    np.random.seed(0)
    out_branch_specific = omega._get_permutations_fast(
        cb_ids=cb_ids,
        sub_branches=sub_branches,
        p=p_by_branch,
        niter=64,
    )
    np.random.seed(0)
    out_branch0_only = omega._get_permutations_fast(
        cb_ids=cb_ids,
        sub_branches=sub_branches,
        p=p_by_branch[0, :],
        niter=64,
    )

    assert out_branch_specific.shape == (2, 64)
    assert out_branch_specific.dtype == np.int32
    assert out_branch_specific.max() <= 2
    assert out_branch_specific.sum() > 0
    assert out_branch0_only.sum() == 0


def test_get_permutations_fast_rejects_branch_probability_row_mismatch():
    cb_ids = np.array([[0, 1]], dtype=np.int64)
    sub_branches = np.array([1, 2], dtype=np.int64)
    p = np.ones((3, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="number of rows"):
        omega._get_permutations_fast(cb_ids=cb_ids, sub_branches=sub_branches, p=p, niter=16)


def test_get_permutations_fast_rejects_negative_branch_ids():
    cb_ids = np.array([[0, -1]], dtype=np.int64)
    sub_branches = np.array([2, 3], dtype=np.int64)
    p = np.array([0.5, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="non-negative"):
        omega._get_permutations_fast(cb_ids=cb_ids, sub_branches=sub_branches, p=p, niter=8)


def test_weighted_sample_without_replacement_packed_matches_mask_packbits():
    p = np.array([0.4, 0.25, 0.2, 0.15, 0.0], dtype=np.float64)
    np.random.seed(11)
    packed = omega._weighted_sample_without_replacement_packed(p=p, size=2, niter=64)
    np.random.seed(11)
    masks = omega._weighted_sample_without_replacement_masks(p=p, size=2, niter=64)
    expected = np.packbits(masks, axis=1)
    np.testing.assert_array_equal(packed, expected)


def test_weighted_sample_without_replacement_packed_handles_full_positive_sites():
    p = np.array([0.0, 0.3, 0.2, 0.5], dtype=np.float64)
    packed = omega._weighted_sample_without_replacement_packed(p=p, size=3, niter=8)
    masks = np.unpackbits(packed, axis=1)[:, :p.shape[0]].astype(bool)
    expected = np.array([False, True, True, True], dtype=bool)
    for row in masks:
        np.testing.assert_array_equal(row, expected)


def test_pack_sampled_site_indices_to_uint8_can_use_cython(monkeypatch):
    class DummyOmegaCy:
        def __init__(self):
            self.called = 0

        def pack_sampled_site_indices_uint8(self, sampled_site_indices, num_site):
            self.called += 1
            out = np.zeros((sampled_site_indices.shape[0], (num_site + 7) // 8), dtype=np.uint8)
            for i in range(sampled_site_indices.shape[0]):
                for site in sampled_site_indices[i, :]:
                    out[i, int(site) >> 3] |= np.uint8(1 << (7 - (int(site) & 7)))
            return out

    sampled = np.array([[0, 3, 4], [1, 2, 7]], dtype=np.int64)
    dummy = DummyOmegaCy()
    monkeypatch.setattr(omega, "omega_cy", dummy)
    out_cy = omega._pack_sampled_site_indices_to_uint8(sampled_site_indices=sampled, num_site=8)
    assert dummy.called == 1

    monkeypatch.setattr(omega, "omega_cy", None)
    out_np = omega._pack_sampled_site_indices_to_uint8(sampled_site_indices=sampled, num_site=8)
    np.testing.assert_array_equal(out_cy, out_np)


def test_get_permutations_fast_can_use_cython_packed_shared_counts(monkeypatch):
    class DummyOmegaCy:
        def __init__(self):
            self.called = 0

        def calc_shared_counts_packed_uint8(self, packed_masks, remapped_cb_ids):
            self.called += 1
            arity = remapped_cb_ids.shape[1]
            if arity == 1:
                out = omega._UINT8_POPCOUNT[packed_masks[remapped_cb_ids[:, 0], :, :]].sum(axis=2, dtype=np.int32)
            elif arity == 2:
                out = omega._UINT8_POPCOUNT[np.bitwise_and(
                    packed_masks[remapped_cb_ids[:, 0], :, :],
                    packed_masks[remapped_cb_ids[:, 1], :, :],
                )].sum(axis=2, dtype=np.int32)
            else:
                shared = packed_masks[remapped_cb_ids[:, 0], :, :].copy()
                for col in range(1, arity):
                    shared = np.bitwise_and(shared, packed_masks[remapped_cb_ids[:, col], :, :])
                out = omega._UINT8_POPCOUNT[shared].sum(axis=2, dtype=np.int32)
            return out.astype(np.int32, copy=False)

    cb_ids = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
    sub_branches = np.array([2, 3, 5], dtype=np.int64)
    p = np.array([0.2, 0.1, 0.25, 0.15, 0.1, 0.1, 0.05, 0.05], dtype=np.float64)

    dummy = DummyOmegaCy()
    monkeypatch.setattr(omega, "omega_cy", dummy)
    np.random.seed(7)
    out_cy = omega._get_permutations_fast(
        cb_ids=cb_ids,
        sub_branches=sub_branches,
        p=p / p.sum(),
        niter=128,
    )
    assert dummy.called > 0

    monkeypatch.setattr(omega, "omega_cy", None)
    np.random.seed(7)
    out_np = omega._get_permutations_fast(
        cb_ids=cb_ids,
        sub_branches=sub_branches,
        p=p / p.sum(),
        niter=128,
    )
    np.testing.assert_array_equal(out_cy, out_np)


def test_get_cb_ids_rejects_non_integer_like_branch_ids():
    cb = pd.DataFrame({"branch_id_1": [1.5], "branch_id_2": [2]})
    with pytest.raises(ValueError, match="integer-like"):
        omega._get_cb_ids(cb)


def test_get_cb_ids_accepts_integer_like_strings():
    cb = pd.DataFrame({"branch_id_1": ["1.0", "2"], "branch_id_2": ["3", "4.0"]})
    out = omega._get_cb_ids(cb)
    np.testing.assert_array_equal(out, np.array([[1, 3], [2, 4]], dtype=np.int64))
    assert out.dtype == np.int64


def test_get_cb_ids_rejects_negative_branch_ids():
    cb = pd.DataFrame({"branch_id_1": [-1], "branch_id_2": [2]})
    with pytest.raises(ValueError, match="non-negative"):
        omega._get_cb_ids(cb)


def test_get_cb_ids_requires_branch_id_columns():
    cb = pd.DataFrame({"other": [1], "value": [2]})
    with pytest.raises(ValueError, match="at least one branch_id_"):
        omega._get_cb_ids(cb)


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


def test_resolve_quantile_niter_schedule_prefers_global_schedule():
    schedule = omega._resolve_quantile_niter_schedule(
        g={"quantile_niter_schedule": [100, 1000, 10000]},
        quantile_niter=500,
    )
    assert schedule == [100, 1000, 10000]


def test_resolve_quantile_niter_schedule_rejects_non_increasing_schedule():
    with pytest.raises(ValueError, match="strictly increasing"):
        omega._resolve_quantile_niter_schedule(
            g={"quantile_niter_schedule": [100, 100, 1000]},
            quantile_niter=500,
        )


def test_needs_quantile_refinement_detects_edge_rows():
    mask = omega._needs_quantile_refinement(
        probability_values=np.array([0.99, 0.5, 0.01], dtype=np.float64),
        quantile_niter=100,
        edge_bins=2,
    )
    np.testing.assert_array_equal(mask, np.array([True, False, True], dtype=bool))


def test_calc_e_stat_quantile_refines_only_edge_rows(monkeypatch):
    stage_calls = list()
    staged_outputs = [
        np.array([0.99, 0.50, 0.01], dtype=np.float64),
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0], dtype=np.float64),
    ]

    def fake_calc_quantile_probabilities(
        mode,
        cb_ids,
        obs_values,
        sub_sg,
        sub_bg,
        quantile_niter,
        obs_col,
        num_gad_combinat,
        list_igad,
        g,
        static_sub_sites,
    ):
        call_index = len(stage_calls)
        out = staged_outputs[call_index]
        assert cb_ids.shape[0] == out.shape[0]
        stage_calls.append((cb_ids.shape[0], int(quantile_niter)))
        return out

    monkeypatch.setattr(omega, "_calc_quantile_probabilities", fake_calc_quantile_probabilities)

    cb = pd.DataFrame(
        {
            "branch_id_1": [0, 1, 2],
            "OCNany2any": [3.0, 2.0, 1.0],
        }
    )
    sub_tensor = np.zeros((4, 5, 1, 2, 2), dtype=np.float64)
    g = {
        "float_type": np.float64,
        "threads": 1,
        "asrv": "each",
        "quantile_niter_schedule": [100, 1000, 10000],
        "quantile_refine_edge_bins": 2,
    }

    out = omega.calc_E_stat(
        cb=cb,
        sub_tensor=sub_tensor,
        mode="any2any",
        stat="quantile",
        SN="N",
        g=g,
    )

    np.testing.assert_allclose(out, np.array([0.9999, 0.50, 0.0001], dtype=np.float64))
    assert stage_calls == [(3, 100), (2, 900), (2, 9000)]


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
