"""Array-only rate, calibration, and empirical-tail statistics.

No pipeline context, I/O, or accelerator dispatch is owned by this module.
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]


def _calc_raw_rate(
    obs: ArrayLike,
    exp: ArrayLike,
    float_tol: float,
) -> FloatArray:
    obs = np.asarray(obs, dtype=np.float64)
    exp = np.asarray(exp, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = obs / exp
    out[obs < float_tol] = 0
    return out


def _calc_raw_omega(
    dNc: ArrayLike,
    dSc: ArrayLike,
    float_tol: float,
) -> FloatArray:
    dNc = np.asarray(dNc, dtype=np.float64)
    dSc = np.asarray(dSc, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        omega = dNc / dSc
    omega[dNc < float_tol] = 0
    return omega


def _calibrate_dsc_vector(
    dNc_values: ArrayLike,
    dSc_values: ArrayLike,
    transformation: str = 'quantile',
) -> tuple[FloatArray, BoolArray, BoolArray]:
    import scipy.stats as stats

    dNc_values = np.asarray(dNc_values, dtype=np.float64).reshape(-1)
    dSc_values = np.asarray(dSc_values, dtype=np.float64).reshape(-1)
    if dNc_values.shape != dSc_values.shape:
        raise ValueError('dNc_values and dSc_values should have identical shapes.')
    fit_mask = np.isfinite(dNc_values) & np.isfinite(dSc_values)
    calibrated_dSc = np.array(dSc_values, dtype=np.float64, copy=True)
    if not fit_mask.any():
        return calibrated_dSc, fit_mask, np.zeros(shape=dSc_values.shape, dtype=bool)
    dNc_values_wo_na = dNc_values[fit_mask]
    dSc_values_wo_na = dSc_values[fit_mask]
    ranks = stats.rankdata(dSc_values_wo_na)
    quantiles = (ranks - 0.5) / float(ranks.shape[0])
    if transformation == 'gamma':
        alpha, loc, beta = stats.gamma.fit(dNc_values_wo_na)
        calibrated_dSc[fit_mask] = stats.gamma.ppf(q=quantiles, a=alpha, loc=loc, scale=beta)
    elif transformation == 'quantile':
        # ``np.quantile(values, q)`` partitions once for every requested q.
        # Here q has the same length as values, so that otherwise becomes
        # quadratic for exhaustive branch-combination tables. Sorting once
        # and applying NumPy's default linear interpolation is equivalent.
        sorted_dNc = np.sort(dNc_values_wo_na)
        if sorted_dNc.shape[0] == 1:
            calibrated = np.full(quantiles.shape, sorted_dNc[0], dtype=np.float64)
        else:
            virtual_indexes = quantiles * float(sorted_dNc.shape[0] - 1)
            previous_indexes = np.floor(virtual_indexes).astype(np.intp)
            next_indexes = np.ceil(virtual_indexes).astype(np.intp)
            gamma = virtual_indexes - previous_indexes
            previous = sorted_dNc[previous_indexes]
            following = sorted_dNc[next_indexes]
            interval = following - previous
            calibrated = previous + interval * gamma
            # Match NumPy's private _lerp rounding order exactly: interpolate
            # from the upper endpoint when its weight is at least one half.
            # This retains the sort-once complexity without introducing the
            # former one-to-two ULP output difference from np.quantile.
            use_upper = gamma >= 0.5
            calibrated[use_upper] = (
                following[use_upper]
                - interval[use_upper] * (1.0 - gamma[use_upper])
            )
        calibrated_dSc[fit_mask] = calibrated
    else:
        raise ValueError('Unsupported transformation: {}'.format(transformation))
    is_nocalib_higher = (
        np.isfinite(dSc_values) &
        np.isfinite(calibrated_dSc) &
        (dSc_values > calibrated_dSc)
    )
    calibrated_dSc[is_nocalib_higher] = dSc_values[is_nocalib_higher]
    return calibrated_dSc, fit_mask, is_nocalib_higher


def _calibrate_dsc_matrix(
    dNc_matrix: ArrayLike,
    dSc_matrix: ArrayLike,
    transformation: str = 'quantile',
) -> FloatArray:
    dNc_matrix = np.asarray(dNc_matrix, dtype=np.float64)
    dSc_matrix = np.asarray(dSc_matrix, dtype=np.float64)
    if dNc_matrix.shape != dSc_matrix.shape:
        raise ValueError('dNc_matrix and dSc_matrix should have identical shapes.')
    if dNc_matrix.ndim != 2:
        raise ValueError('dNc_matrix and dSc_matrix should be 2D arrays.')
    calibrated = np.array(dSc_matrix, dtype=np.float64, copy=True)
    for col_i in range(dNc_matrix.shape[1]):
        calibrated[:, col_i], _, _ = _calibrate_dsc_vector(
            dNc_values=dNc_matrix[:, col_i],
            dSc_values=dSc_matrix[:, col_i],
            transformation=transformation,
        )
    return calibrated


def _calc_permutation_omega_matrix(
    exp_N: ArrayLike,
    exp_S: ArrayLike,
    perm_count_N: ArrayLike,
    perm_count_S: ArrayLike,
    float_tol: float,
    calibrate_dsc_transformation: str | None = None,
) -> FloatArray:
    exp_N = np.asarray(exp_N, dtype=np.float64).reshape(-1)
    exp_S = np.asarray(exp_S, dtype=np.float64).reshape(-1)
    perm_count_N = np.asarray(perm_count_N, dtype=np.float64)
    perm_count_S = np.asarray(perm_count_S, dtype=np.float64)
    if perm_count_N.shape != perm_count_S.shape:
        raise ValueError('perm_count_N and perm_count_S should have identical shapes.')
    if perm_count_N.ndim != 2:
        raise ValueError('perm_count_N should be a 2D array.')
    if exp_N.shape[0] != perm_count_N.shape[0]:
        txt = 'exp_N rows ({}) and permutation rows ({}) should match.'
        raise ValueError(txt.format(exp_N.shape[0], perm_count_N.shape[0]))
    if exp_S.shape[0] != perm_count_N.shape[0]:
        txt = 'exp_S rows ({}) and permutation rows ({}) should match.'
        raise ValueError(txt.format(exp_S.shape[0], perm_count_N.shape[0]))
    perm_dNc = _calc_raw_rate(obs=perm_count_N, exp=exp_N[:, None], float_tol=float_tol)
    perm_dSc = _calc_raw_rate(obs=perm_count_S, exp=exp_S[:, None], float_tol=float_tol)
    if calibrate_dsc_transformation is not None:
        perm_dSc = _calibrate_dsc_matrix(
            dNc_matrix=perm_dNc,
            dSc_matrix=perm_dSc,
            transformation=calibrate_dsc_transformation,
        )
    return _calc_raw_omega(dNc=perm_dNc, dSc=perm_dSc, float_tol=float_tol)


def _calc_omega_empirical_upper_tail_counts_from_perm(
    obs_omega: ArrayLike,
    perm_omega: ArrayLike,
) -> tuple[IntArray, IntArray]:
    obs_omega = np.asarray(obs_omega, dtype=np.float64).reshape(-1)
    perm_omega = np.asarray(perm_omega, dtype=np.float64)
    if perm_omega.ndim != 2:
        raise ValueError('perm_omega should be a 2D array.')
    if perm_omega.shape[0] != obs_omega.shape[0]:
        txt = 'Permutation rows ({}) and observed rows ({}) should match.'
        raise ValueError(txt.format(perm_omega.shape[0], obs_omega.shape[0]))
    valid_perm = ~np.isnan(perm_omega)
    ge_ranks = (valid_perm & (perm_omega >= obs_omega[:, None])).sum(axis=1, dtype=np.int64)
    valid_niter = valid_perm.sum(axis=1, dtype=np.int64)
    return ge_ranks, valid_niter


def _calc_omega_empirical_upper_tail_pvalues_from_counts(
    obs_omega: ArrayLike,
    exp_S: ArrayLike,
    ge_ranks: ArrayLike,
    valid_niter: ArrayLike,
) -> FloatArray:
    obs_omega = np.asarray(obs_omega, dtype=np.float64).reshape(-1)
    exp_S = np.asarray(exp_S, dtype=np.float64).reshape(-1)
    ge_ranks = np.asarray(ge_ranks, dtype=np.int64).reshape(-1)
    valid_niter = np.asarray(valid_niter, dtype=np.int64).reshape(-1)
    if ge_ranks.shape[0] != obs_omega.shape[0]:
        txt = 'ge_ranks rows ({}) and observed rows ({}) should match.'
        raise ValueError(txt.format(ge_ranks.shape[0], obs_omega.shape[0]))
    if valid_niter.shape[0] != obs_omega.shape[0]:
        txt = 'valid_niter rows ({}) and observed rows ({}) should match.'
        raise ValueError(txt.format(valid_niter.shape[0], obs_omega.shape[0]))
    if exp_S.shape[0] != obs_omega.shape[0]:
        txt = 'exp_S rows ({}) and observed rows ({}) should match.'
        raise ValueError(txt.format(exp_S.shape[0], obs_omega.shape[0]))
    if (ge_ranks < 0).any():
        raise ValueError('ge_ranks should be >= 0.')
    if (valid_niter < 0).any():
        raise ValueError('valid_niter should be >= 0.')
    if (ge_ranks > valid_niter).any():
        raise ValueError('ge_ranks should be <= valid_niter.')
    pvalue = np.full(shape=obs_omega.shape, fill_value=np.nan, dtype=np.float64)
    valid_rows = (~np.isnan(obs_omega)) & (valid_niter > 0)
    valid_rows &= np.isfinite(exp_S)
    pvalue[valid_rows] = (ge_ranks[valid_rows] + 1.0) / (valid_niter[valid_rows] + 1.0)
    return pvalue


def _calc_omega_empirical_upper_tail_pvalues_from_perm(
    obs_omega: ArrayLike,
    exp_S: ArrayLike,
    perm_omega: ArrayLike,
) -> FloatArray:
    ge_ranks, valid_niter = _calc_omega_empirical_upper_tail_counts_from_perm(
        obs_omega=obs_omega,
        perm_omega=perm_omega,
    )
    return _calc_omega_empirical_upper_tail_pvalues_from_counts(
        obs_omega=obs_omega,
        exp_S=exp_S,
        ge_ranks=ge_ranks,
        valid_niter=valid_niter,
    )


def _calc_omega_empirical_upper_tail_counts(
    obs_omega: ArrayLike,
    exp_N: ArrayLike,
    exp_S: ArrayLike,
    perm_count_N: ArrayLike,
    perm_count_S: ArrayLike,
    float_tol: float,
    calibrate_dsc_transformation: str | None = None,
) -> tuple[IntArray, IntArray]:
    obs_omega = np.asarray(obs_omega, dtype=np.float64).reshape(-1)
    exp_N = np.asarray(exp_N, dtype=np.float64).reshape(-1)
    exp_S = np.asarray(exp_S, dtype=np.float64).reshape(-1)
    perm_count_N = np.asarray(perm_count_N, dtype=np.float64)
    perm_count_S = np.asarray(perm_count_S, dtype=np.float64)
    if perm_count_N.shape != perm_count_S.shape:
        raise ValueError('perm_count_N and perm_count_S should have identical shapes.')
    if perm_count_N.ndim != 2:
        raise ValueError('perm_count_N should be a 2D array.')
    if perm_count_N.shape[0] != obs_omega.shape[0]:
        txt = 'Permutation rows ({}) and observed rows ({}) should match.'
        raise ValueError(txt.format(perm_count_N.shape[0], obs_omega.shape[0]))
    if exp_N.shape[0] != obs_omega.shape[0]:
        txt = 'exp_N rows ({}) and observed rows ({}) should match.'
        raise ValueError(txt.format(exp_N.shape[0], obs_omega.shape[0]))
    if exp_S.shape[0] != obs_omega.shape[0]:
        txt = 'exp_S rows ({}) and observed rows ({}) should match.'
        raise ValueError(txt.format(exp_S.shape[0], obs_omega.shape[0]))
    perm_omega = _calc_permutation_omega_matrix(
        exp_N=exp_N,
        exp_S=exp_S,
        perm_count_N=perm_count_N,
        perm_count_S=perm_count_S,
        float_tol=float_tol,
        calibrate_dsc_transformation=calibrate_dsc_transformation,
    )
    return _calc_omega_empirical_upper_tail_counts_from_perm(
        obs_omega=obs_omega,
        perm_omega=perm_omega,
    )


def _needs_omega_pvalue_upper_tail_edge_refinement(
    obs_omega: ArrayLike,
    exp_S: ArrayLike,
    ge_ranks: ArrayLike,
    valid_niter: ArrayLike,
    edge_bins: int,
) -> BoolArray:
    obs_omega = np.asarray(obs_omega, dtype=np.float64).reshape(-1)
    exp_S = np.asarray(exp_S, dtype=np.float64).reshape(-1)
    ge_ranks = np.asarray(ge_ranks, dtype=np.int64).reshape(-1)
    valid_niter = np.asarray(valid_niter, dtype=np.int64).reshape(-1)
    edge_bins = int(edge_bins)
    if obs_omega.shape[0] != exp_S.shape[0]:
        txt = 'obs_omega rows ({}) and exp_S rows ({}) should match.'
        raise ValueError(txt.format(obs_omega.shape[0], exp_S.shape[0]))
    if ge_ranks.shape[0] != obs_omega.shape[0]:
        txt = 'ge_ranks rows ({}) and obs_omega rows ({}) should match.'
        raise ValueError(txt.format(ge_ranks.shape[0], obs_omega.shape[0]))
    if valid_niter.shape[0] != obs_omega.shape[0]:
        txt = 'valid_niter rows ({}) and obs_omega rows ({}) should match.'
        raise ValueError(txt.format(valid_niter.shape[0], obs_omega.shape[0]))
    if (ge_ranks < 0).any():
        raise ValueError('ge_ranks should be >= 0.')
    if (valid_niter < 0).any():
        raise ValueError('valid_niter should be >= 0.')
    if (ge_ranks > valid_niter).any():
        raise ValueError('ge_ranks should be <= valid_niter.')
    if edge_bins < 0:
        raise ValueError('edge_bins should be >= 0.')
    refine = np.zeros(shape=obs_omega.shape, dtype=bool)
    valid_rows = np.isfinite(obs_omega) & np.isfinite(exp_S)
    refine[valid_rows] = (valid_niter[valid_rows] <= 0)
    if edge_bins <= 0:
        return refine
    overlap_rows = valid_rows & (valid_niter > 0)
    refine[overlap_rows] = (ge_ranks[overlap_rows] <= edge_bins)
    return refine


def _calc_omega_empirical_upper_tail_pvalues(
    obs_omega: ArrayLike,
    exp_N: ArrayLike,
    exp_S: ArrayLike,
    perm_count_N: ArrayLike,
    perm_count_S: ArrayLike,
    float_tol: float,
    calibrate_dsc_transformation: str | None = None,
) -> FloatArray:
    ge_ranks, valid_niter = _calc_omega_empirical_upper_tail_counts(
        obs_omega=obs_omega,
        exp_N=exp_N,
        exp_S=exp_S,
        perm_count_N=perm_count_N,
        perm_count_S=perm_count_S,
        float_tol=float_tol,
        calibrate_dsc_transformation=calibrate_dsc_transformation,
    )
    return _calc_omega_empirical_upper_tail_pvalues_from_counts(
        obs_omega=obs_omega,
        exp_S=exp_S,
        ge_ranks=ge_ranks,
        valid_niter=valid_niter,
    )


def _calc_bh_fdr_qvalues(
    pvalues: ArrayLike,
) -> FloatArray:
    pvalues = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    qvalues = np.full(shape=pvalues.shape, fill_value=np.nan, dtype=np.float64)
    is_finite = np.isfinite(pvalues)
    if not is_finite.any():
        return qvalues
    p_finite = np.clip(pvalues[is_finite], a_min=0.0, a_max=1.0)
    m = int(p_finite.shape[0])
    order = np.argsort(p_finite, kind='mergesort')
    ranked = p_finite[order]
    bh = ranked * (float(m) / np.arange(1, m + 1, dtype=np.float64))
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, a_min=0.0, a_max=1.0)
    q_finite = np.empty_like(bh)
    q_finite[order] = bh
    qvalues[is_finite] = q_finite
    return qvalues
