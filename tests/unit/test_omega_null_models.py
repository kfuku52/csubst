import numpy as np

from csubst import omega


def test_calc_poisson_count_matrix_matches_expected_means():
    cb_ids = np.array([[0, 1], [1, 2]], dtype=np.int64)
    sub_bg = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    sub_sg = np.zeros((2, 1), dtype=np.float64)
    static_sub_sites = np.array(
        [
            [0.6, 0.4],
            [0.5, 0.5],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )
    list_igad = [[0, 0, "any2", "2any"]]
    expected_mean = omega._calc_wallenius_expected_overlap(
        cb_ids=cb_ids,
        sub_sites=static_sub_sites,
        sub_branches=sub_bg[:, 0],
        g={"omega_pvalue_rounding": "round"},
        float_type=np.float64,
    )
    np.random.seed(7)
    out = omega._calc_poisson_count_matrix(
        mode="any2any",
        cb_ids=cb_ids,
        sub_sg=sub_sg,
        sub_bg=sub_bg,
        niter=4000,
        obs_col="OCNany2any",
        num_gad_combinat=1,
        list_igad=list_igad,
        g={"float_tol": 1e-12},
        static_sub_sites=static_sub_sites,
    )
    assert out.shape == (2, 4000)
    assert out.dtype == np.float64
    assert np.all(out >= 0)
    np.testing.assert_allclose(out, np.round(out), atol=0.0)
    np.testing.assert_allclose(out.mean(axis=1), expected_mean, atol=0.12)


def test_calc_poisson_count_matrix_is_reproducible_from_configured_seed():
    kwargs = {
        "mode": "any2any",
        "cb_ids": np.array([[0, 1], [1, 2]], dtype=np.int64),
        "sub_sg": np.zeros((2, 1), dtype=np.float64),
        "sub_bg": np.array([[1.0], [2.0], [3.0]], dtype=np.float64),
        "niter": 256,
        "obs_col": "OCNany2any",
        "num_gad_combinat": 1,
        "list_igad": [[0, 0, "any2", "2any"]],
        "static_sub_sites": np.array(
            [[0.6, 0.4], [0.5, 0.5], [0.2, 0.8]], dtype=np.float64
        ),
    }

    first = omega._calc_poisson_count_matrix(g={"float_tol": 1e-12, "random_seed": 41}, **kwargs)
    second = omega._calc_poisson_count_matrix(g={"float_tol": 1e-12, "random_seed": 41}, **kwargs)
    different = omega._calc_poisson_count_matrix(g={"float_tol": 1e-12, "random_seed": 42}, **kwargs)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_calc_poisson_count_matrix_uses_wallenius_mean_for_skewed_weights():
    cb_ids = np.array([[0, 1]], dtype=np.int64)
    sub_bg = np.array([[80.0], [80.0]], dtype=np.float64)
    sub_sg = np.zeros((100, 1), dtype=np.float64)
    static_sub_sites = np.zeros((2, 100), dtype=np.float64)
    static_sub_sites[:, 0] = 0.9
    static_sub_sites[:, 1:] = 0.1 / 99.0
    list_igad = [[0, 0, "any2", "2any"]]
    expected_mean = omega._calc_wallenius_expected_overlap(
        cb_ids=cb_ids,
        sub_sites=static_sub_sites,
        sub_branches=sub_bg[:, 0],
        g={"omega_pvalue_rounding": "round"},
        float_type=np.float64,
    )
    legacy_mean = omega._calc_tmp_E_sum(
        cb_ids=cb_ids,
        sub_sites=static_sub_sites,
        sub_branches=sub_bg[:, 0],
        float_type=np.float64,
    )
    np.random.seed(17)
    out = omega._calc_poisson_count_matrix(
        mode="any2any",
        cb_ids=cb_ids,
        sub_sg=sub_sg,
        sub_bg=sub_bg,
        niter=8000,
        obs_col="OCNany2any",
        num_gad_combinat=1,
        list_igad=list_igad,
        g={"float_tol": 1e-12},
        static_sub_sites=static_sub_sites,
    )
    assert out.shape == (1, 8000)
    np.testing.assert_allclose(out.mean(axis=1), expected_mean, atol=0.35)
    assert abs(float(out.mean()) - float(legacy_mean[0])) > 1000.0


def test_calc_poisson_full_count_matrix_matches_expected_means():
    cb_ids = np.array([[0, 1], [1, 2]], dtype=np.int64)
    sub_tensor = np.zeros((3, 2, 1, 2, 2), dtype=np.float64)
    sub_tensor[:, :, 0, 0, 1] = np.array(
        [
            [0.4, 0.6],
            [0.3, 0.7],
            [0.8, 0.2],
        ],
        dtype=np.float64,
    )
    list_igad = [[0, 0, "any2", "2any"]]
    site_mass = sub_tensor[:, :, 0, :, :].sum(axis=(2, 3))
    branch_totals = site_mass.sum(axis=1, dtype=np.float64)
    site_probs = np.zeros_like(site_mass, dtype=np.float64)
    nz = (branch_totals > 0)
    site_probs[nz, :] = site_mass[nz, :] / branch_totals[nz, None]
    expected_mean = omega._calc_wallenius_expected_overlap(
        cb_ids=cb_ids,
        sub_sites=site_probs,
        sub_branches=branch_totals,
        g={"omega_pvalue_rounding": "round"},
        float_type=np.float64,
    )
    np.random.seed(11)
    out = omega._calc_poisson_full_count_matrix(
        mode="any2any",
        cb_ids=cb_ids,
        sub_tensor=sub_tensor,
        niter=4000,
        obs_col="OCNany2any",
        num_gad_combinat=1,
        list_igad=list_igad,
        g={"omega_pvalue_rounding": "round"},
    )
    assert out.shape == (2, 4000)
    assert out.dtype == np.float64
    assert np.all(out >= 0)
    np.testing.assert_allclose(out, np.round(out), atol=0.0)
    np.testing.assert_allclose(out.mean(axis=1), expected_mean, atol=0.06)


def test_calc_poisson_full_count_matrix_uses_wallenius_mean_for_skewed_weights():
    cb_ids = np.array([[0, 1]], dtype=np.int64)
    sub_tensor = np.zeros((2, 100, 1, 1, 1), dtype=np.float64)
    p = np.zeros(100, dtype=np.float64)
    p[0] = 0.9
    p[1:] = 0.1 / 99.0
    sub_tensor[:, :, 0, 0, 0] = 80.0 * np.broadcast_to(p.reshape(1, -1), (2, p.shape[0]))
    list_igad = [[0, 0, "any2", "2any"]]
    site_mass = sub_tensor[:, :, 0, :, :].sum(axis=(2, 3))
    branch_totals = site_mass.sum(axis=1, dtype=np.float64)
    site_probs = np.zeros_like(site_mass, dtype=np.float64)
    nz = (branch_totals > 0)
    site_probs[nz, :] = site_mass[nz, :] / branch_totals[nz, None]
    expected_mean = omega._calc_wallenius_expected_overlap(
        cb_ids=cb_ids,
        sub_sites=site_probs,
        sub_branches=branch_totals,
        g={"omega_pvalue_rounding": "round"},
        float_type=np.float64,
    )
    legacy_mean = omega._calc_tmp_E_sum(
        cb_ids=cb_ids,
        sub_sites=site_mass,
        sub_branches=np.ones(shape=(site_mass.shape[0],), dtype=np.float64),
        float_type=np.float64,
    )
    np.random.seed(19)
    out = omega._calc_poisson_full_count_matrix(
        mode="any2any",
        cb_ids=cb_ids,
        sub_tensor=sub_tensor,
        niter=8000,
        obs_col="OCNany2any",
        num_gad_combinat=1,
        list_igad=list_igad,
        g={"omega_pvalue_rounding": "round"},
    )
    assert out.shape == (1, 8000)
    np.testing.assert_allclose(out.mean(axis=1), expected_mean, atol=0.35)
    assert abs(float(out.mean()) - float(legacy_mean[0])) > 1000.0


def test_get_mode_permutation_count_matrix_uses_poisson_model(monkeypatch):
    cb_ids = np.array([[0, 1]], dtype=np.int64)
    expected = np.full((1, 5), 2.0, dtype=np.float64)

    def fake_prepare(sub_tensor, mode, SN, g):
        return np.zeros((2, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64), [[0, 0, "any2", "2any"]], "OCNany2any", 1

    def fake_static(g, sub_sg, mode, obs_col):
        return np.ones((2, 3), dtype=np.float64) / 3.0

    def fake_poisson(mode, cb_ids, sub_sg, sub_bg, niter, obs_col, num_gad_combinat, list_igad, g, static_sub_sites):
        return expected.copy()

    def fake_quantile(*args, **kwargs):
        raise AssertionError("hypergeom path should not be used for poisson null model")

    monkeypatch.setattr(omega, "_prepare_substitution_permutation_components", fake_prepare)
    monkeypatch.setattr(omega, "_get_static_sub_sites_if_available", fake_static)
    monkeypatch.setattr(omega, "_calc_poisson_count_matrix", fake_poisson)
    monkeypatch.setattr(omega, "_calc_hypergeom_count_matrix", fake_quantile)

    out = omega._get_mode_permutation_count_matrix(
        cb_ids=cb_ids,
        sub_tensor=None,
        mode="any2any",
        SN="N",
        niter=5,
        g={"omega_pvalue_null_model": "poisson"},
    )
    np.testing.assert_allclose(out, expected)


def test_get_mode_permutation_count_matrix_uses_poisson_full_model(monkeypatch):
    cb_ids = np.array([[0, 1]], dtype=np.int64)
    expected = np.full((1, 5), 3.0, dtype=np.float64)

    def fake_prepare(sub_tensor, mode, SN, g):
        return np.zeros((2, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64), [[0, 0, "any2", "2any"]], "OCNany2any", 1

    def fake_static(g, sub_sg, mode, obs_col):
        return np.ones((2, 3), dtype=np.float64) / 3.0

    def fake_poisson(*args, **kwargs):
        raise AssertionError("factorized poisson path should not be used for poisson_full null model")

    def fake_poisson_full(mode, cb_ids, sub_tensor, niter, obs_col, num_gad_combinat, list_igad, g):
        return expected.copy()

    def fake_quantile(*args, **kwargs):
        raise AssertionError("hypergeom path should not be used for poisson_full null model")

    monkeypatch.setattr(omega, "_prepare_substitution_permutation_components", fake_prepare)
    monkeypatch.setattr(omega, "_get_static_sub_sites_if_available", fake_static)
    monkeypatch.setattr(omega, "_calc_poisson_count_matrix", fake_poisson)
    monkeypatch.setattr(omega, "_calc_poisson_full_count_matrix", fake_poisson_full)
    monkeypatch.setattr(omega, "_calc_hypergeom_count_matrix", fake_quantile)

    out = omega._get_mode_permutation_count_matrix(
        cb_ids=cb_ids,
        sub_tensor=np.zeros((2, 3, 1, 1, 1), dtype=np.float64),
        mode="any2any",
        SN="N",
        niter=5,
        g={"omega_pvalue_null_model": "poisson_full"},
    )
    np.testing.assert_allclose(out, expected)


def test_get_mode_permutation_count_matrix_uses_nbinom_model(monkeypatch):
    cb_ids = np.array([[0, 1]], dtype=np.int64)
    expected = np.full((1, 5), 4.0, dtype=np.float64)

    def fake_prepare(sub_tensor, mode, SN, g):
        return np.zeros((2, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64), [[0, 0, "any2", "2any"]], "OCNany2any", 1

    def fake_static(g, sub_sg, mode, obs_col):
        return np.ones((2, 3), dtype=np.float64) / 3.0

    def fake_nbinom(mode, cb_ids, sub_sg, sub_bg, niter, obs_col, num_gad_combinat, list_igad, g, static_sub_sites, obs_count):
        assert obs_count is not None
        return expected.copy()

    def fake_poisson(*args, **kwargs):
        raise AssertionError("poisson path should not be used for nbinom null model")

    def fake_quantile(*args, **kwargs):
        raise AssertionError("hypergeom path should not be used for nbinom null model")

    monkeypatch.setattr(omega, "_prepare_substitution_permutation_components", fake_prepare)
    monkeypatch.setattr(omega, "_get_static_sub_sites_if_available", fake_static)
    monkeypatch.setattr(omega, "_calc_nbinom_count_matrix", fake_nbinom)
    monkeypatch.setattr(omega, "_calc_poisson_count_matrix", fake_poisson)
    monkeypatch.setattr(omega, "_calc_hypergeom_count_matrix", fake_quantile)

    out = omega._get_mode_permutation_count_matrix(
        cb_ids=cb_ids,
        sub_tensor=None,
        mode="any2any",
        SN="N",
        niter=5,
        g={"omega_pvalue_null_model": "nbinom"},
        obs_count=np.array([2.0], dtype=np.float64),
    )
    np.testing.assert_allclose(out, expected)


def test_calc_nbinom_count_matrix_supports_fixed_overdispersion():
    cb_ids = np.array([[0, 1], [1, 2]], dtype=np.int64)
    sub_bg = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    sub_sg = np.zeros((2, 1), dtype=np.float64)
    static_sub_sites = np.array(
        [
            [0.6, 0.4],
            [0.5, 0.5],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )
    list_igad = [[0, 0, "any2", "2any"]]
    expected_mean = omega._calc_wallenius_expected_overlap(
        cb_ids=cb_ids,
        sub_sites=static_sub_sites,
        sub_branches=sub_bg[:, 0],
        g={"omega_pvalue_rounding": "round"},
        float_type=np.float64,
    )
    np.random.seed(13)
    out = omega._calc_nbinom_count_matrix(
        mode="any2any",
        cb_ids=cb_ids,
        sub_sg=sub_sg,
        sub_bg=sub_bg,
        niter=4000,
        obs_col="OCNany2any",
        num_gad_combinat=1,
        list_igad=list_igad,
        g={"float_tol": 1e-12, "omega_pvalue_nbinom_alpha": 0.8},
        static_sub_sites=static_sub_sites,
        obs_count=np.array([2.0, 3.0], dtype=np.float64),
    )
    assert out.shape == (2, 4000)
    assert out.dtype == np.float64
    assert np.all(out >= 0)
    np.testing.assert_allclose(out.mean(axis=1), expected_mean, atol=0.18)
    # With alpha>0, variance should exceed mean for overdispersed rows.
    assert np.all(out.var(axis=1) > out.mean(axis=1))
