import numpy as np
import pytest

from csubst import substitution


try:
    from csubst import substitution_cy
except ImportError:  # pragma: no cover - optional Cython extension
    substitution_cy = None


def _toy_sub_tensor():
    # shape = [branch, site, group, from, to]
    sub = np.zeros((3, 2, 1, 2, 2), dtype=float)
    sub[0, 0, 0, :, :] = [[0.0, 0.2], [0.1, 0.0]]
    sub[1, 0, 0, :, :] = [[0.0, 0.5], [0.2, 0.0]]
    sub[2, 0, 0, :, :] = [[0.0, 0.4], [0.3, 0.0]]
    sub[0, 1, 0, :, :] = [[0.0, 0.1], [0.0, 0.0]]
    sub[1, 1, 0, :, :] = [[0.0, 0.1], [0.3, 0.0]]
    sub[2, 1, 0, :, :] = [[0.0, 0.2], [0.1, 0.0]]
    return sub


def test_sub_tensor2cb_mmap_chunk_writer_matches_non_mmap(tmp_path):
    sub = _toy_sub_tensor()
    ids = np.array([[2, 0], [1, 2]], dtype=np.int64)
    expected = substitution.sub_tensor2cb(ids, sub, mmap=False, df_mmap=None, mmap_start=0, float_type=np.float64)

    arity = ids.shape[1]
    mmap_path = tmp_path / "cb_writer.mmap"
    mmap_out = np.memmap(mmap_path, dtype=np.float64, mode="w+", shape=(ids.shape[0] + 1, arity + 4))
    mmap_out[:] = 0.0
    substitution.sub_tensor2cb(ids, sub, mmap=True, df_mmap=mmap_out, mmap_start=1, float_type=np.float64)
    observed = np.array(mmap_out[1 : 1 + ids.shape[0], :], copy=True)
    del mmap_out

    np.testing.assert_allclose(observed, expected, atol=1e-12)


def test_sub_tensor2cbs_mmap_chunk_writer_matches_non_mmap(tmp_path):
    sub = _toy_sub_tensor()
    ids = np.array([[2, 0], [1, 2]], dtype=np.int64)
    expected = substitution.sub_tensor2cbs(ids, sub, mmap=False, df_mmap=None, mmap_start=0)

    arity = ids.shape[1]
    num_site = sub.shape[1]
    mmap_path = tmp_path / "cbs_writer.mmap"
    mmap_rows = (ids.shape[0] + 1) * num_site
    mmap_out = np.memmap(mmap_path, dtype=np.float64, mode="w+", shape=(mmap_rows, arity + 5))
    mmap_out[:] = 0.0
    substitution.sub_tensor2cbs(ids, sub, mmap=True, df_mmap=mmap_out, mmap_start=1)
    row_start = num_site
    row_end = row_start + expected.shape[0]
    observed = np.array(mmap_out[row_start:row_end, :], copy=True)
    del mmap_out

    np.testing.assert_allclose(observed, expected, atol=1e-12)


def test_sub_tensor2cb_cython_fastpath_matches_python_fallback(monkeypatch):
    if not hasattr(substitution_cy, "calc_combinatorial_sub_double_arity2"):
        pytest.skip("Cython dense reducer fast path is unavailable")
    rng = np.random.default_rng(7)
    sub = rng.random((5, 4, 2, 3, 3), dtype=np.float64)
    ids = np.array([[0, 1], [2, 3], [4, 1]], dtype=np.int64)

    monkeypatch.setattr(substitution, "_can_use_cython_dense_cb", lambda *args, **kwargs: False)
    expected = substitution.sub_tensor2cb(ids, sub, mmap=False, df_mmap=None, mmap_start=0, float_type=np.float64)

    monkeypatch.setattr(substitution, "_can_use_cython_dense_cb", lambda *args, **kwargs: True)
    observed = substitution.sub_tensor2cb(ids, sub, mmap=False, df_mmap=None, mmap_start=0, float_type=np.float64)

    np.testing.assert_allclose(observed, expected, atol=1e-12)


def test_sub_tensor2cbs_cython_fastpath_matches_python_fallback(monkeypatch):
    if not hasattr(substitution_cy, "calc_combinatorial_sub_by_site_double_arity2"):
        pytest.skip("Cython dense reducer fast path is unavailable")
    rng = np.random.default_rng(11)
    sub = rng.random((4, 3, 2, 3, 3), dtype=np.float64)
    ids = np.array([[0, 1], [2, 3]], dtype=np.int64)

    monkeypatch.setattr(substitution, "_can_use_cython_dense_cbs", lambda *args, **kwargs: False)
    expected = substitution.sub_tensor2cbs(ids, sub, mmap=False, df_mmap=None, mmap_start=0)

    monkeypatch.setattr(substitution, "_can_use_cython_dense_cbs", lambda *args, **kwargs: True)
    observed = substitution.sub_tensor2cbs(ids, sub, mmap=False, df_mmap=None, mmap_start=0)

    np.testing.assert_allclose(observed, expected, atol=1e-12)


def test_sub_tensor2cb_cython_failure_warns_and_falls_back(monkeypatch):
    if not hasattr(substitution_cy, "calc_combinatorial_sub_double_arity2"):
        pytest.skip("Cython dense reducer fast path is unavailable")
    rng = np.random.default_rng(19)
    sub = rng.random((4, 3, 2, 3, 3), dtype=np.float64)
    ids = np.array([[0, 1], [2, 3]], dtype=np.int64)

    monkeypatch.setattr(substitution, "_can_use_cython_dense_cb", lambda *args, **kwargs: False)
    expected = substitution.sub_tensor2cb(ids, sub, mmap=False, df_mmap=None, mmap_start=0, float_type=np.float64)

    monkeypatch.setattr(substitution, "_CYTHON_FALLBACK_WARNED", set())
    monkeypatch.setattr(substitution, "_can_use_cython_dense_cb", lambda *args, **kwargs: True)

    def _raise_cython(*args, **kwargs):
        raise RuntimeError("forced-fastpath-failure")

    monkeypatch.setattr(substitution_cy, "calc_combinatorial_sub_double_arity2", _raise_cython)
    with pytest.warns(RuntimeWarning, match='Cython fast path "sub_tensor2cb" failed'):
        observed = substitution.sub_tensor2cb(ids, sub, mmap=False, df_mmap=None, mmap_start=0, float_type=np.float64)
    np.testing.assert_allclose(observed, expected, atol=1e-12)


def test_sub_tensor2cbs_cython_failure_warns_and_falls_back(monkeypatch):
    if not hasattr(substitution_cy, "calc_combinatorial_sub_by_site_double_arity2"):
        pytest.skip("Cython dense reducer fast path is unavailable")
    rng = np.random.default_rng(23)
    sub = rng.random((4, 3, 2, 3, 3), dtype=np.float64)
    ids = np.array([[0, 1], [2, 3]], dtype=np.int64)

    monkeypatch.setattr(substitution, "_can_use_cython_dense_cbs", lambda *args, **kwargs: False)
    expected = substitution.sub_tensor2cbs(ids, sub, mmap=False, df_mmap=None, mmap_start=0)

    monkeypatch.setattr(substitution, "_CYTHON_FALLBACK_WARNED", set())
    monkeypatch.setattr(substitution, "_can_use_cython_dense_cbs", lambda *args, **kwargs: True)

    def _raise_cython(*args, **kwargs):
        raise RuntimeError("forced-fastpath-failure")

    monkeypatch.setattr(substitution_cy, "calc_combinatorial_sub_by_site_double_arity2", _raise_cython)
    with pytest.warns(RuntimeWarning, match='Cython fast path "sub_tensor2cbs" failed'):
        observed = substitution.sub_tensor2cbs(ids, sub, mmap=False, df_mmap=None, mmap_start=0)
    np.testing.assert_allclose(observed, expected, atol=1e-12)


def test_get_cb_matches_sum_of_get_cs_per_combination():
    sub = _toy_sub_tensor()
    ids = np.array([[2, 0], [1, 2]], dtype=np.int64)
    cb = substitution.get_cb(ids, sub, {"threads": 1, "float_type": np.float64}, attr="OCN")

    for combo in ids:
        c1, c2 = sorted(combo.tolist())
        row = cb.loc[(cb["branch_id_1"] == c1) & (cb["branch_id_2"] == c2), :].iloc[0]
        cs = substitution.get_cs(np.array([[c1, c2]], dtype=np.int64), sub, attr="N")
        assert pytest.approx(float(row["OCNany2any"]), abs=1e-12) == float(cs["OCNany2any"].sum())
        assert pytest.approx(float(row["OCNspe2any"]), abs=1e-12) == float(cs["OCNspe2any"].sum())
        assert pytest.approx(float(row["OCNany2spe"]), abs=1e-12) == float(cs["OCNany2spe"].sum())
        assert pytest.approx(float(row["OCNspe2spe"]), abs=1e-12) == float(cs["OCNspe2spe"].sum())


def test_get_cb_selective_base_stats_match_full_columns():
    sub = _toy_sub_tensor()
    ids = np.array([[2, 0], [1, 2]], dtype=np.int64)
    g = {"threads": 1, "float_type": np.float64}
    full = substitution.get_cb(ids, sub, g, attr="OCN")
    subset = substitution.get_cb(ids, sub, g, attr="OCN", selected_base_stats=["any2any", "any2spe"])

    assert subset.columns.tolist() == ["branch_id_1", "branch_id_2", "OCNany2any", "OCNany2spe"]
    merged = full.merge(subset, on=["branch_id_1", "branch_id_2"], suffixes=("_full", "_subset"))
    np.testing.assert_allclose(
        merged["OCNany2any_full"].to_numpy(),
        merged["OCNany2any_subset"].to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        merged["OCNany2spe_full"].to_numpy(),
        merged["OCNany2spe_subset"].to_numpy(),
        atol=1e-12,
    )


def test_get_cbs_grouped_sum_matches_get_cb():
    sub = _toy_sub_tensor()
    ids = np.array([[2, 0], [1, 2]], dtype=np.int64)

    cb = substitution.get_cb(ids, sub, {"threads": 1, "float_type": np.float64}, attr="OCN")
    cbs = substitution.get_cbs(ids, sub, attr="N", g={"threads": 1})
    cols = ["OCNany2any", "OCNspe2any", "OCNany2spe", "OCNspe2spe"]
    summed = cbs.groupby(["branch_id_1", "branch_id_2"], as_index=False)[cols].sum()
    merged = cb.merge(summed, on=["branch_id_1", "branch_id_2"], suffixes=("_cb", "_cbs"))

    for col in cols:
        np.testing.assert_allclose(
            merged[f"{col}_cb"].to_numpy(),
            merged[f"{col}_cbs"].to_numpy(),
            atol=1e-12,
        )
