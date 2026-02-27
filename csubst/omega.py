#if __name__ == '__main__':
#    mp.set_start_method('spawn')
#    my_class = MyClass(1)
#    my_class.mp_simple_method()
#    my_class.wait()

import numpy as np
import scipy.stats as stats
from scipy.linalg import expm

import itertools
import os
import re
import sys
import time

from csubst import parallel
from csubst import substitution
from csubst import substitution_sparse
from csubst import table
from csubst import ete
from csubst import output_stat
from csubst import pseudocount
try:
    from csubst import omega_cy
except Exception:  # pragma: no cover - Cython extension is optional
    omega_cy = None

_UINT8_POPCOUNT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None],
    axis=1,
).sum(axis=1).astype(np.uint8)
_EPI_BETA_GRID = np.unique(np.concatenate([
    np.arange(-2.0, 2.001, 0.01, dtype=np.float64),
    np.array([-3.0, -2.5, 2.5, 3.0], dtype=np.float64),
]))
_EPI_AUTO_CLIP_QUANTILE = 0.995
_EPI_AUTO_CLIP_MIN = 1.5
_EPI_AUTO_CLIP_MAX = 5.0
_EPI_CV_FOLDS = 5


def _get_cb_ids(cb):
    bid_columns = cb.columns[cb.columns.str.startswith('branch_id_')]
    if bid_columns.shape[0] == 0:
        raise ValueError('cb should contain at least one branch_id_ column.')
    cb_ids = cb.loc[:, bid_columns].to_numpy(copy=False, dtype=object)
    if cb_ids.size == 0:
        return np.zeros(shape=cb_ids.shape, dtype=np.int64)
    normalized = []
    for value in cb_ids.reshape(-1).tolist():
        if value is None:
            raise ValueError('branch_id columns should be integer-like.')
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('branch_id columns should be integer-like.')
        if isinstance(value, (int, np.integer)):
            if int(value) < 0:
                raise ValueError('branch_id columns should be non-negative integers.')
            normalized.append(int(value))
            continue
        if isinstance(value, (float, np.floating)):
            if (not np.isfinite(value)) or (not float(value).is_integer()):
                raise ValueError('branch_id columns should be integer-like.')
            if int(value) < 0:
                raise ValueError('branch_id columns should be non-negative integers.')
            normalized.append(int(value))
            continue
        value_txt = str(value).strip()
        if (value_txt == '') or (not bool(re.fullmatch(r'[+-]?[0-9]+(?:\.0+)?', value_txt))):
            raise ValueError('branch_id columns should be integer-like.')
        if int(float(value_txt)) < 0:
            raise ValueError('branch_id columns should be non-negative integers.')
        normalized.append(int(float(value_txt)))
    return np.array(normalized, dtype=np.int64).reshape(cb_ids.shape)


def _resolve_requested_output_stats(g):
    if 'output_stats' in g.keys():
        return output_stat.parse_output_stats(g['output_stats'])
    if 'output_stat' in g.keys():
        return output_stat.parse_output_stats(g['output_stat'])
    return list(output_stat.ALL_OUTPUT_STATS)


def _resolve_epistasis_channels(g):
    token = str(g.get('epistasis_apply_to', 'N')).strip().upper()
    if token == 'N':
        return ('N',)
    if token == 'S':
        return ('S',)
    if token == 'NS':
        return ('N', 'S')
    raise ValueError('Unsupported epistasis_apply_to value: {}'.format(token))


def _get_epistasis_weight_matrix_for_obs_col(obs_col, g):
    if not bool(g.get('epistasis_enabled', False)):
        return None
    state = g.get('_epistasis_state', dict())
    if str(obs_col).startswith('OCN'):
        channel_state = state.get('N', dict())
        return channel_state.get('weights', None)
    if str(obs_col).startswith('OCS'):
        channel_state = state.get('S', dict())
        return channel_state.get('weights', None)
    return None


def _get_epistasis_branch_site_counts(sub_tensor, g):
    if 'is_site_nonmissing' in g.keys():
        num_branch = int(g['is_site_nonmissing'].shape[0])
        num_site = int(g['is_site_nonmissing'].shape[1])
    else:
        num_branch = int(sub_tensor.shape[0])
        num_site = int(sub_tensor.shape[1])
    counts = np.zeros(shape=(num_branch, num_site), dtype=np.float64)
    if '_asrv_branch_ids' in g.keys():
        branch_ids = np.asarray(g['_asrv_branch_ids'], dtype=np.int64).reshape(-1)
    else:
        branch_ids = np.arange(num_branch, dtype=np.int64)
    for branch_id in branch_ids.tolist():
        counts[branch_id, :] = substitution.get_branch_site_sub_counts(
            sub_tensor=sub_tensor,
            branch_id=int(branch_id),
        )
    return counts


def _build_epistasis_base_probs(counts, is_site_nonmissing, dirichlet_alpha, float_tol):
    counts = np.asarray(counts, dtype=np.float64)
    is_site_nonmissing = np.asarray(is_site_nonmissing, dtype=bool)
    if counts.shape != is_site_nonmissing.shape:
        txt = 'counts shape ({}) and is_site_nonmissing shape ({}) should match.'
        raise ValueError(txt.format(counts.shape, is_site_nonmissing.shape))
    out = np.zeros(shape=counts.shape, dtype=np.float64)
    dirichlet_alpha = float(dirichlet_alpha)
    for i in range(counts.shape[0]):
        valid = is_site_nonmissing[i, :]
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        row_counts = counts[i, valid]
        row_sum = float(row_counts.sum(dtype=np.float64))
        if dirichlet_alpha > 0:
            denom = row_sum + (dirichlet_alpha * n_valid)
            if denom > float(float_tol):
                out[i, valid] = (row_counts + dirichlet_alpha) / denom
            else:
                out[i, valid] = 1.0 / n_valid
            continue
        if row_sum > float(float_tol):
            out[i, valid] = row_counts / row_sum
        else:
            out[i, valid] = 1.0 / n_valid
    return out


def _coerce_epistasis_site_features(site_features):
    site_features = np.asarray(site_features, dtype=np.float64)
    if site_features.ndim == 1:
        site_features = site_features.reshape(-1, 1)
    if site_features.ndim != 2:
        raise ValueError('site_features should be a 1D or 2D array.')
    return site_features


def _coerce_epistasis_branch_parameter(value, num_branch, name):
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(shape=(num_branch,), fill_value=float(arr), dtype=np.float64)
    arr = arr.reshape(-1).astype(np.float64, copy=False)
    if arr.shape[0] != num_branch:
        txt = '{} length ({}) should match number of branches ({}).'
        raise ValueError(txt.format(name, arr.shape[0], num_branch))
    return arr


def _calc_epistasis_interaction_score(branch_context, site_features):
    branch_context = np.asarray(branch_context, dtype=np.float64)
    site_features = _coerce_epistasis_site_features(site_features=site_features)
    if branch_context.ndim == 1:
        branch_context = branch_context.reshape(-1, 1)
    if branch_context.ndim != 2:
        raise ValueError('branch_context should be a 1D or 2D array.')
    if branch_context.shape[1] != site_features.shape[1]:
        txt = 'branch_context feature dimension ({}) and site_features feature dimension ({}) should match.'
        raise ValueError(txt.format(branch_context.shape[1], site_features.shape[1]))
    score = branch_context @ site_features.T
    if branch_context.shape[1] > 1:
        score = score / float(branch_context.shape[1])
    return score


def _calc_epistasis_branch_context(counts, degree_z, is_site_nonmissing, float_tol):
    counts = np.asarray(counts, dtype=np.float64)
    site_features = _coerce_epistasis_site_features(site_features=degree_z)
    is_site_nonmissing = np.asarray(is_site_nonmissing, dtype=bool)
    if counts.shape != is_site_nonmissing.shape:
        txt = 'counts shape ({}) and is_site_nonmissing shape ({}) should match.'
        raise ValueError(txt.format(counts.shape, is_site_nonmissing.shape))
    if counts.shape[1] != site_features.shape[0]:
        txt = 'counts site axis ({}) and site_features length ({}) should match.'
        raise ValueError(txt.format(counts.shape[1], site_features.shape[0]))
    context = np.zeros(shape=(counts.shape[0], site_features.shape[1]), dtype=np.float64)
    for i in range(counts.shape[0]):
        valid = is_site_nonmissing[i, :]
        if not valid.any():
            continue
        row_counts = counts[i, valid]
        total = float(row_counts.sum(dtype=np.float64))
        if total <= float(float_tol):
            continue
        context[i, :] = (row_counts[:, None] * site_features[valid, :]).sum(axis=0, dtype=np.float64) / total
    if context.shape[1] == 1:
        return context[:, 0]
    return context


def _apply_epistasis_beta_to_probs(base_probs, branch_context, degree_z, beta, clip_value, is_site_nonmissing, float_tol):
    base_probs = np.asarray(base_probs, dtype=np.float64)
    site_features = _coerce_epistasis_site_features(site_features=degree_z)
    branch_context = np.asarray(branch_context, dtype=np.float64)
    is_site_nonmissing = np.asarray(is_site_nonmissing, dtype=bool)
    if base_probs.shape != is_site_nonmissing.shape:
        txt = 'base_probs shape ({}) and is_site_nonmissing shape ({}) should match.'
        raise ValueError(txt.format(base_probs.shape, is_site_nonmissing.shape))
    if base_probs.shape[1] != site_features.shape[0]:
        txt = 'base_probs site axis ({}) and site_features length ({}) should match.'
        raise ValueError(txt.format(base_probs.shape[1], site_features.shape[0]))
    if branch_context.ndim == 1:
        branch_context = branch_context.reshape(-1, 1)
    if branch_context.ndim != 2:
        raise ValueError('branch_context should be a 1D or 2D array.')
    if base_probs.shape[0] != branch_context.shape[0]:
        txt = 'base_probs branch axis ({}) and branch_context length ({}) should match.'
        raise ValueError(txt.format(base_probs.shape[0], branch_context.shape[0]))
    if branch_context.shape[1] != site_features.shape[1]:
        txt = 'branch_context feature dimension ({}) and site_features feature dimension ({}) should match.'
        raise ValueError(txt.format(branch_context.shape[1], site_features.shape[1]))
    beta_by_branch = _coerce_epistasis_branch_parameter(beta, base_probs.shape[0], 'beta')
    if np.all(beta_by_branch <= float(float_tol)):
        return base_probs.copy()
    clip_by_branch = _coerce_epistasis_branch_parameter(clip_value, base_probs.shape[0], 'clip_value')
    score = _calc_epistasis_interaction_score(
        branch_context=branch_context,
        site_features=site_features,
    )
    score = score * beta_by_branch[:, None]
    finite_clip = np.isfinite(clip_by_branch)
    if finite_clip.any():
        clip_abs = np.abs(clip_by_branch)
        score[finite_clip, :] = np.minimum(
            np.maximum(score[finite_clip, :], -clip_abs[finite_clip, None]),
            clip_abs[finite_clip, None],
        )
    weight = np.exp(score)
    numer = base_probs * weight
    numer = np.where(is_site_nonmissing, numer, 0.0)
    denom = numer.sum(axis=1, keepdims=True, dtype=np.float64)
    out = base_probs.copy()
    valid_rows = (denom[:, 0] > float(float_tol))
    if valid_rows.any():
        out[valid_rows, :] = numer[valid_rows, :] / denom[valid_rows, :]
    return out


def _calc_epistasis_branch_loglik(counts, probs, float_tol):
    counts = np.asarray(counts, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    safe_probs = np.clip(probs, a_min=float(float_tol), a_max=1.0)
    return (counts * np.log(safe_probs)).sum(axis=1, dtype=np.float64)


def _fit_epistasis_beta_cv(
    counts,
    base_probs,
    branch_context,
    degree_z,
    is_site_nonmissing,
    clip_value,
    float_tol,
    dirichlet_alpha=1.0,
    beta_grid=None,
):
    counts = np.asarray(counts, dtype=np.float64)
    base_probs = np.asarray(base_probs, dtype=np.float64)
    branch_context = np.asarray(branch_context, dtype=np.float64)
    site_features = _coerce_epistasis_site_features(site_features=degree_z)
    if branch_context.ndim == 1:
        branch_context = branch_context.reshape(-1, 1)
    is_site_nonmissing = np.asarray(is_site_nonmissing, dtype=bool)
    if counts.shape != base_probs.shape:
        txt = 'counts shape ({}) and base_probs shape ({}) should match.'
        raise ValueError(txt.format(counts.shape, base_probs.shape))
    if counts.shape != is_site_nonmissing.shape:
        txt = 'counts shape ({}) and is_site_nonmissing shape ({}) should match.'
        raise ValueError(txt.format(counts.shape, is_site_nonmissing.shape))
    if counts.shape[0] != branch_context.shape[0]:
        txt = 'counts branch axis ({}) and branch_context length ({}) should match.'
        raise ValueError(txt.format(counts.shape[0], branch_context.shape[0]))
    if counts.shape[1] != site_features.shape[0]:
        txt = 'counts site axis ({}) and site_features length ({}) should match.'
        raise ValueError(txt.format(counts.shape[1], site_features.shape[0]))
    if branch_context.shape[1] != site_features.shape[1]:
        txt = 'branch_context feature dimension ({}) and site_features feature dimension ({}) should match.'
        raise ValueError(txt.format(branch_context.shape[1], site_features.shape[1]))
    dirichlet_alpha = float(dirichlet_alpha)
    if dirichlet_alpha < 0:
        raise ValueError('dirichlet_alpha should be >= 0.')
    if beta_grid is None:
        beta_grid = _EPI_BETA_GRID
    beta_grid = np.asarray(beta_grid, dtype=np.float64).reshape(-1)
    if beta_grid.shape[0] == 0:
        raise ValueError('beta_grid should contain one or more values.')

    total_by_branch = counts.sum(axis=1, dtype=np.float64)
    is_active = (total_by_branch > float(float_tol)) & is_site_nonmissing.any(axis=1)
    active_rows = np.where(is_active)[0].astype(np.int64)
    if active_rows.shape[0] < 2:
        fallback_beta = 0.0 if np.isclose(beta_grid, 0.0).any() else float(beta_grid[0])
        return float(fallback_beta), {'active_branch_count': int(active_rows.shape[0]), 'num_folds': 0, 'best_score': np.nan}
    num_folds = min(_EPI_CV_FOLDS, int(active_rows.shape[0]))
    fold_ids = np.arange(active_rows.shape[0], dtype=np.int64) % num_folds
    fold_scores = np.full(shape=(beta_grid.shape[0], num_folds), fill_value=np.nan, dtype=np.float64)
    # Cross-validation uses branch-held-out priors to avoid a trivial beta=0 optimum
    # induced by evaluating each branch against its own empirical distribution.
    for i, beta in enumerate(beta_grid.tolist()):
        for fold in range(num_folds):
            test_rows = active_rows[fold_ids == fold]
            train_rows = active_rows[fold_ids != fold]
            if test_rows.shape[0] == 0:
                continue
            if train_rows.shape[0] == 0:
                fold_base = base_probs[test_rows, :].copy()
            else:
                site_prior = counts[train_rows, :].sum(axis=0, dtype=np.float64)
                site_prior = np.clip(site_prior, a_min=0.0, a_max=None)
                if dirichlet_alpha > 0:
                    site_prior = site_prior + dirichlet_alpha
                fold_base = np.where(
                    is_site_nonmissing[test_rows, :],
                    site_prior[None, :],
                    0.0,
                ).astype(np.float64, copy=False)
                row_sum = fold_base.sum(axis=1, keepdims=True, dtype=np.float64)
                nonzero_rows = (row_sum[:, 0] > float(float_tol))
                if nonzero_rows.any():
                    fold_base[nonzero_rows, :] = fold_base[nonzero_rows, :] / row_sum[nonzero_rows, :]
                zero_rows = ~nonzero_rows
                if zero_rows.any():
                    fill = is_site_nonmissing[test_rows[zero_rows], :].astype(np.float64, copy=False)
                    fill_sum = fill.sum(axis=1, keepdims=True, dtype=np.float64)
                    valid_fill = (fill_sum[:, 0] > 0)
                    if valid_fill.any():
                        fill[valid_fill, :] = fill[valid_fill, :] / fill_sum[valid_fill, :]
                    fold_base[zero_rows, :] = fill
            probs = _apply_epistasis_beta_to_probs(
                base_probs=fold_base,
                branch_context=branch_context[test_rows],
                degree_z=site_features,
                beta=float(beta),
                clip_value=float(clip_value),
                is_site_nonmissing=is_site_nonmissing[test_rows, :],
                float_tol=float_tol,
            )
            ll_by_branch = _calc_epistasis_branch_loglik(
                counts=counts[test_rows, :],
                probs=probs,
                float_tol=float_tol,
            )
            fold_scores[i, fold] = ll_by_branch.sum(dtype=np.float64)
    mean_score = np.nanmean(fold_scores, axis=1)
    best_idx = int(np.nanargmax(mean_score))
    best_mean = float(mean_score[best_idx])
    if num_folds >= 2:
        best_se = float(np.nanstd(fold_scores[best_idx, :], ddof=1) / np.sqrt(num_folds))
    else:
        best_se = 0.0
    selected_idx = best_idx
    selected_beta = float(beta_grid[selected_idx])
    diag = {
        'active_branch_count': int(active_rows.shape[0]),
        'num_folds': int(num_folds),
        'best_score': float(best_mean),
        'best_beta': float(beta_grid[best_idx]),
        'selected_beta': float(selected_beta),
        'one_se': float(best_se),
        'selection_rule': 'argmax_mean_cv_loglik',
    }
    return selected_beta, diag


def _auto_epistasis_clip(beta, branch_context, degree_z, is_site_nonmissing):
    site_features = _coerce_epistasis_site_features(site_features=degree_z)
    branch_context = np.asarray(branch_context, dtype=np.float64)
    if branch_context.ndim == 1:
        branch_context = branch_context.reshape(-1, 1)
    beta_by_branch = _coerce_epistasis_branch_parameter(beta, branch_context.shape[0], 'beta')
    if np.all(beta_by_branch <= 0):
        return float(_EPI_AUTO_CLIP_MIN)
    score = _calc_epistasis_interaction_score(
        branch_context=branch_context,
        site_features=site_features,
    )
    score = np.abs(score * beta_by_branch[:, None])
    mask = np.asarray(is_site_nonmissing, dtype=bool)
    if score.shape != mask.shape:
        txt = 'score shape ({}) and is_site_nonmissing shape ({}) should match.'
        raise ValueError(txt.format(score.shape, mask.shape))
    values = score[mask]
    values = values[np.isfinite(values)]
    if values.shape[0] == 0:
        return float(_EPI_AUTO_CLIP_MIN)
    clip = float(np.quantile(values, _EPI_AUTO_CLIP_QUANTILE))
    clip = min(max(clip, float(_EPI_AUTO_CLIP_MIN)), float(_EPI_AUTO_CLIP_MAX))
    if clip <= 0:
        clip = float(_EPI_AUTO_CLIP_MIN)
    return clip


def _build_epistasis_weight_matrix(branch_context, degree_z, beta, clip_value, is_site_nonmissing, float_tol):
    site_features = _coerce_epistasis_site_features(site_features=degree_z)
    branch_context = np.asarray(branch_context, dtype=np.float64)
    if branch_context.ndim == 1:
        branch_context = branch_context.reshape(-1, 1)
    beta_by_branch = _coerce_epistasis_branch_parameter(beta, branch_context.shape[0], 'beta')
    if np.all(beta_by_branch <= float(float_tol)):
        return np.ones(shape=is_site_nonmissing.shape, dtype=np.float64)
    clip_by_branch = _coerce_epistasis_branch_parameter(clip_value, branch_context.shape[0], 'clip_value')
    score = _calc_epistasis_interaction_score(
        branch_context=branch_context,
        site_features=site_features,
    )
    score = score * beta_by_branch[:, None]
    finite_clip = np.isfinite(clip_by_branch)
    if finite_clip.any():
        clip_abs = np.abs(clip_by_branch)
        score[finite_clip, :] = np.minimum(
            np.maximum(score[finite_clip, :], -clip_abs[finite_clip, None]),
            clip_abs[finite_clip, None],
        )
    weight = np.exp(score)
    weight = np.where(is_site_nonmissing, weight, 1.0)
    return weight


def _apply_epistasis_to_sub_sites(sub_sites, obs_col, g):
    weight = _get_epistasis_weight_matrix_for_obs_col(obs_col=obs_col, g=g)
    if weight is None:
        return sub_sites
    sub_sites_arr = np.asarray(sub_sites)
    if sub_sites_arr.ndim == 1:
        site_weight = np.asarray(weight, dtype=np.float64).mean(axis=0)
        out = np.asarray(sub_sites_arr, dtype=np.float64) * site_weight
        total = float(out.sum(dtype=np.float64))
        if total > float(g['float_tol']):
            out = out / total
        return out.astype(getattr(sub_sites_arr, 'dtype', np.float64), copy=False)
    if sub_sites_arr.ndim != 2:
        return sub_sites
    weight = np.asarray(weight, dtype=np.float64)
    if sub_sites_arr.shape != weight.shape:
        txt = 'Epistasis weight shape ({}) did not match sub_sites shape ({}).'
        raise ValueError(txt.format(weight.shape, sub_sites_arr.shape))
    sub_sites_float = np.asarray(sub_sites_arr, dtype=np.float64)
    numer = sub_sites_float * weight
    denom = numer.sum(axis=1, keepdims=True, dtype=np.float64)
    out = sub_sites_float.copy()
    valid_rows = (denom[:, 0] > float(g['float_tol']))
    if valid_rows.any():
        out[valid_rows, :] = numer[valid_rows, :] / denom[valid_rows, :]
    return out.astype(sub_sites_arr.dtype, copy=False)


def _get_epistasis_branch_depth_by_id(g, num_branch):
    out = np.arange(num_branch, dtype=np.float64)
    tree = g.get('tree', None)
    if tree is None:
        return out
    for node in tree.traverse():
        try:
            branch_id = int(ete.get_prop(node, 'numerical_label'))
        except Exception:
            continue
        if (branch_id < 0) or (branch_id >= num_branch):
            continue
        try:
            depth = 0.0
            cursor = node
            while (cursor is not None) and (not ete.is_root(cursor)):
                depth += 0.0 if (cursor.dist is None) else float(cursor.dist)
                cursor = cursor.up
            out[branch_id] = float(depth)
        except Exception:
            out[branch_id] = np.nan
    finite = np.isfinite(out)
    if not finite.all():
        fill = float(np.nanmedian(out[finite])) if finite.any() else 0.0
        out[~finite] = fill
    return out


def _assign_epistasis_branch_depth_bins(depth_by_branch, active_mask, num_bins):
    depth_by_branch = np.asarray(depth_by_branch, dtype=np.float64).reshape(-1)
    active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    if depth_by_branch.shape[0] != active_mask.shape[0]:
        txt = 'depth_by_branch length ({}) and active_mask length ({}) should match.'
        raise ValueError(txt.format(depth_by_branch.shape[0], active_mask.shape[0]))
    bin_ids = np.full(shape=(depth_by_branch.shape[0],), fill_value=-1, dtype=np.int64)
    active_ids = np.where(active_mask & np.isfinite(depth_by_branch))[0].astype(np.int64)
    if active_ids.shape[0] == 0:
        return bin_ids, list()
    num_bins = max(1, min(int(num_bins), int(active_ids.shape[0])))
    sorted_ids = active_ids[np.argsort(depth_by_branch[active_ids], kind='mergesort')]
    chunks = np.array_split(sorted_ids, num_bins)
    summary = list()
    for i, chunk in enumerate(chunks):
        if chunk.shape[0] == 0:
            continue
        bin_ids[chunk] = int(i)
        summary.append({
            'bin': int(i),
            'count': int(chunk.shape[0]),
            'depth_min': float(depth_by_branch[chunk].min()),
            'depth_max': float(depth_by_branch[chunk].max()),
        })
    return bin_ids, summary


def _fit_epistasis_parameters_for_subset(
    counts,
    is_site_nonmissing,
    branch_context,
    site_features,
    g,
):
    counts = np.asarray(counts, dtype=np.float64)
    is_site_nonmissing = np.asarray(is_site_nonmissing, dtype=bool)
    branch_context = np.asarray(branch_context, dtype=np.float64)
    site_features = _coerce_epistasis_site_features(site_features=site_features)
    if counts.shape != is_site_nonmissing.shape:
        txt = 'counts shape ({}) and is_site_nonmissing shape ({}) should match.'
        raise ValueError(txt.format(counts.shape, is_site_nonmissing.shape))
    if counts.shape[0] == 0:
        return {
            'beta': 0.0,
            'clip': float(g.get('epistasis_clip_value', 3.0)),
            'alpha': float(g.get('asrv_dirichlet_alpha', 1.0)),
            'diag': {'active_branch_count': 0, 'num_folds': 0, 'best_score': np.nan},
        }
    beta_auto = bool(g.get('epistasis_beta_auto', False))
    clip_auto = bool(g.get('epistasis_clip_auto', False))
    joint_auto = bool(g.get('epistasis_joint_auto', False))
    base_alpha = float(g.get('asrv_dirichlet_alpha', 1.0))
    float_tol = float(g['float_tol'])

    if joint_auto:
        alpha_grid = np.asarray(g.get('epistasis_joint_alpha_grid', [0.0, 0.5, 1.0, 2.0]), dtype=np.float64).reshape(-1)
        clip_grid = np.asarray(g.get('epistasis_joint_clip_grid', [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]), dtype=np.float64).reshape(-1)
        if alpha_grid.shape[0] == 0:
            alpha_grid = np.array([base_alpha], dtype=np.float64)
        if clip_grid.shape[0] == 0:
            clip_grid = np.array([float(g.get('epistasis_clip_value', 3.0))], dtype=np.float64)
        best = None
        for alpha in alpha_grid.tolist():
            base_probs = _build_epistasis_base_probs(
                counts=counts,
                is_site_nonmissing=is_site_nonmissing,
                dirichlet_alpha=float(alpha),
                float_tol=float_tol,
            )
            for clip_value in clip_grid.tolist():
                if beta_auto:
                    beta_grid = None
                else:
                    beta_grid = np.array([float(g.get('epistasis_beta_value', 0.0))], dtype=np.float64)
                beta, diag = _fit_epistasis_beta_cv(
                    counts=counts,
                    base_probs=base_probs,
                    branch_context=branch_context,
                    degree_z=site_features,
                    is_site_nonmissing=is_site_nonmissing,
                    clip_value=float(clip_value),
                    float_tol=float_tol,
                    dirichlet_alpha=float(alpha),
                    beta_grid=beta_grid,
                )
                score = float(diag.get('best_score', np.nan))
                cmp_score = score if np.isfinite(score) else -np.inf
                if (best is None) or (cmp_score > best['cmp_score']):
                    best = {
                        'beta': float(beta),
                        'clip': float(clip_value),
                        'alpha': float(alpha),
                        'diag': diag,
                        'cmp_score': float(cmp_score),
                    }
        if best is not None:
            return {
                'beta': best['beta'],
                'clip': best['clip'],
                'alpha': best['alpha'],
                'diag': best['diag'],
            }

    base_probs = _build_epistasis_base_probs(
        counts=counts,
        is_site_nonmissing=is_site_nonmissing,
        dirichlet_alpha=base_alpha,
        float_tol=float_tol,
    )
    if beta_auto:
        clip_for_fit = float(g['epistasis_clip_value']) if (not clip_auto) else float(_EPI_AUTO_CLIP_MAX)
        beta, beta_diag = _fit_epistasis_beta_cv(
            counts=counts,
            base_probs=base_probs,
            branch_context=branch_context,
            degree_z=site_features,
            is_site_nonmissing=is_site_nonmissing,
            clip_value=clip_for_fit,
            float_tol=float_tol,
            dirichlet_alpha=base_alpha,
        )
    else:
        beta = float(g.get('epistasis_beta_value', 0.0))
        beta_diag = {'active_branch_count': int((counts.sum(axis=1) > float_tol).sum()), 'num_folds': 0, 'best_score': np.nan}
    if clip_auto:
        clip_value = _auto_epistasis_clip(
            beta=beta,
            branch_context=branch_context,
            degree_z=site_features,
            is_site_nonmissing=is_site_nonmissing,
        )
        if beta_auto:
            beta, beta_diag = _fit_epistasis_beta_cv(
                counts=counts,
                base_probs=base_probs,
                branch_context=branch_context,
                degree_z=site_features,
                is_site_nonmissing=is_site_nonmissing,
                clip_value=float(clip_value),
                float_tol=float_tol,
                dirichlet_alpha=base_alpha,
            )
            clip_value = _auto_epistasis_clip(
                beta=beta,
                branch_context=branch_context,
                degree_z=site_features,
                is_site_nonmissing=is_site_nonmissing,
            )
    else:
        clip_value = float(g.get('epistasis_clip_value', 3.0))
    return {
        'beta': float(beta),
        'clip': float(clip_value),
        'alpha': float(base_alpha),
        'diag': beta_diag,
    }


def prepare_epistasis(g, ON_tensor, OS_tensor):
    g['epistasis_enabled'] = False
    site_features = g.get('epistasis_site_feature_matrix_internal', None)
    if site_features is None:
        site_features = g.get('epistasis_site_degree_internal', None)
    if site_features is None:
        return g
    site_features = _coerce_epistasis_site_features(site_features=site_features)
    is_site_nonmissing = np.asarray(g.get('is_site_nonmissing', None), dtype=bool)
    if is_site_nonmissing.ndim != 2:
        raise ValueError('is_site_nonmissing should be available before epistasis preparation.')
    if is_site_nonmissing.shape[1] != site_features.shape[0]:
        txt = 'is_site_nonmissing site axis ({}) and epistasis feature length ({}) should match.'
        raise ValueError(txt.format(is_site_nonmissing.shape[1], site_features.shape[0]))
    channels = _resolve_epistasis_channels(g=g)
    resolved_site_metric = str(g.get('epistasis_site_metric_resolved', g.get('epistasis_site_metric', 'auto')))
    resolved_feature_mode = str(g.get('epistasis_feature_mode_resolved', g.get('epistasis_feature_mode', 'single')))
    print(
        'Epistasis summary: apply_to={}, site_metric={}, feature_mode={}, beta_partition={}, joint_auto={}'.format(
            ''.join(channels),
            resolved_site_metric,
            resolved_feature_mode,
            str(g.get('epistasis_beta_partition', 'global')),
            'yes' if bool(g.get('epistasis_joint_auto', False)) else 'no',
        ),
        flush=True,
    )
    if 'S' in channels:
        if channels == ('S',):
            txt = 'Epistasis negative-control summary: apply_to=S (ECS-only reweighting).'
        else:
            txt = 'Epistasis negative-control summary: S channel enabled alongside N (NS mode).'
        print(txt, flush=True)
    state = dict()
    for channel in channels:
        if channel == 'N':
            tensor = ON_tensor
        elif channel == 'S':
            tensor = OS_tensor
        else:
            raise ValueError('Unsupported epistasis channel: {}'.format(channel))
        counts = _get_epistasis_branch_site_counts(sub_tensor=tensor, g=g)
        counts = np.asarray(counts, dtype=np.float64)
        if counts.shape != is_site_nonmissing.shape:
            txt = 'Epistasis counts shape ({}) did not match is_site_nonmissing shape ({}).'
            raise ValueError(txt.format(counts.shape, is_site_nonmissing.shape))
        counts = np.where(is_site_nonmissing, counts, 0.0)
        branch_context = _calc_epistasis_branch_context(
            counts=counts,
            degree_z=site_features,
            is_site_nonmissing=is_site_nonmissing,
            float_tol=float(g['float_tol']),
        )
        if branch_context.ndim == 1:
            branch_context_matrix = branch_context.reshape(-1, 1)
        else:
            branch_context_matrix = branch_context
        active_mask = (counts.sum(axis=1, dtype=np.float64) > float(g['float_tol'])) & is_site_nonmissing.any(axis=1)
        partition_mode = str(g.get('epistasis_beta_partition', 'global')).strip().lower()
        if partition_mode == 'branch_depth':
            depth_by_branch = _get_epistasis_branch_depth_by_id(g=g, num_branch=counts.shape[0])
            bin_ids, bin_summary = _assign_epistasis_branch_depth_bins(
                depth_by_branch=depth_by_branch,
                active_mask=active_mask,
                num_bins=int(g.get('epistasis_branch_depth_bins', 3)),
            )
            beta_by_branch = np.zeros(shape=(counts.shape[0],), dtype=np.float64)
            clip_by_branch = np.full(shape=(counts.shape[0],), fill_value=float(g.get('epistasis_clip_value', 3.0)), dtype=np.float64)
            alpha_by_branch = np.full(shape=(counts.shape[0],), fill_value=float(g.get('asrv_dirichlet_alpha', 1.0)), dtype=np.float64)
            bin_diags = list()
            for item in bin_summary:
                row_mask = (bin_ids == int(item['bin']))
                fit = _fit_epistasis_parameters_for_subset(
                    counts=counts[row_mask, :],
                    is_site_nonmissing=is_site_nonmissing[row_mask, :],
                    branch_context=branch_context_matrix[row_mask, :],
                    site_features=site_features,
                    g=g,
                )
                beta_by_branch[row_mask] = float(fit['beta'])
                clip_by_branch[row_mask] = float(fit['clip'])
                alpha_by_branch[row_mask] = float(fit['alpha'])
                bin_diags.append({
                    'bin': int(item['bin']),
                    'count': int(item['count']),
                    'depth_min': float(item['depth_min']),
                    'depth_max': float(item['depth_max']),
                    'beta': float(fit['beta']),
                    'clip': float(fit['clip']),
                    'alpha': float(fit['alpha']),
                    'diag': fit['diag'],
                })
            beta_by_branch[~active_mask] = 0.0
            if active_mask.any():
                beta = float(beta_by_branch[active_mask].mean(dtype=np.float64))
                clip_value = float(clip_by_branch[active_mask].mean(dtype=np.float64))
                alpha_value = float(alpha_by_branch[active_mask].mean(dtype=np.float64))
            else:
                beta = 0.0
                clip_value = float(g.get('epistasis_clip_value', 3.0))
                alpha_value = float(g.get('asrv_dirichlet_alpha', 1.0))
            beta_diag = {
                'active_branch_count': int(active_mask.sum()),
                'num_folds': 0,
                'best_score': np.nan,
                'partition': 'branch_depth',
                'bins': bin_diags,
            }
        else:
            fit = _fit_epistasis_parameters_for_subset(
                counts=counts,
                is_site_nonmissing=is_site_nonmissing,
                branch_context=branch_context_matrix,
                site_features=site_features,
                g=g,
            )
            beta = float(fit['beta'])
            clip_value = float(fit['clip'])
            alpha_value = float(fit['alpha'])
            beta_diag = dict(fit['diag'])
            beta_diag['partition'] = 'global'
            beta_by_branch = np.full(shape=(counts.shape[0],), fill_value=beta, dtype=np.float64)
            clip_by_branch = np.full(shape=(counts.shape[0],), fill_value=clip_value, dtype=np.float64)
            alpha_by_branch = np.full(shape=(counts.shape[0],), fill_value=alpha_value, dtype=np.float64)
        weights = _build_epistasis_weight_matrix(
            branch_context=branch_context_matrix,
            degree_z=site_features,
            beta=beta_by_branch,
            clip_value=clip_by_branch,
            is_site_nonmissing=is_site_nonmissing,
            float_tol=float(g['float_tol']),
        )
        state[channel] = {
            'beta': float(beta),
            'clip': float(clip_value),
            'alpha': float(alpha_value),
            'weights': weights.astype(np.float64, copy=False),
            'branch_context': branch_context_matrix.astype(np.float64, copy=False),
            'beta_by_branch': beta_by_branch.astype(np.float64, copy=False),
            'clip_by_branch': clip_by_branch.astype(np.float64, copy=False),
            'alpha_by_branch': alpha_by_branch.astype(np.float64, copy=False),
            'beta_diag': beta_diag,
        }
        txt = 'Epistasis [{}]: beta={}, clip={}, alpha={}, active_branches={}'
        print(
            txt.format(
                channel,
                '{:.6g}'.format(float(beta)),
                '{:.6g}'.format(float(clip_value)),
                '{:.6g}'.format(float(alpha_value)),
                int(active_mask.sum()),
            ),
            flush=True,
        )
        if partition_mode == 'branch_depth':
            for item in beta_diag.get('bins', []):
                txt = 'Epistasis [{}] depth-bin {} (n={} depth={:.6g}..{:.6g}): beta={}, clip={}, alpha={}'
                print(
                    txt.format(
                        channel,
                        int(item.get('bin', -1)),
                        int(item.get('count', 0)),
                        float(item.get('depth_min', np.nan)),
                        float(item.get('depth_max', np.nan)),
                        '{:.6g}'.format(float(item.get('beta', np.nan))),
                        '{:.6g}'.format(float(item.get('clip', np.nan))),
                        '{:.6g}'.format(float(item.get('alpha', np.nan))),
                    ),
                    flush=True,
                )
    if len(state) == 0:
        return g
    g['_epistasis_state'] = state
    g['epistasis_enabled'] = True
    return g


def _can_use_cython_expected_state(parent_state_block, transition_prob):
    if omega_cy is None:
        return False
    if not hasattr(omega_cy, 'project_expected_state_block_double'):
        return False
    if not isinstance(parent_state_block, np.ndarray):
        return False
    if not isinstance(transition_prob, np.ndarray):
        return False
    if parent_state_block.dtype != np.float64:
        return False
    if transition_prob.dtype != np.float64:
        return False
    if parent_state_block.ndim != 2:
        return False
    if transition_prob.ndim != 2:
        return False
    if transition_prob.shape[0] != transition_prob.shape[1]:
        return False
    if parent_state_block.shape[1] != transition_prob.shape[0]:
        return False
    # For larger state spaces, NumPy/BLAS matmul is typically faster.
    if parent_state_block.shape[1] > 8:
        return False
    return True


def _can_use_cython_tmp_E_sum(cb_ids, sub_sites, sub_branches):
    if omega_cy is None:
        return False
    if not hasattr(omega_cy, 'calc_tmp_E_sum_double'):
        return False
    if not isinstance(cb_ids, np.ndarray):
        return False
    if not isinstance(sub_sites, np.ndarray):
        return False
    if not isinstance(sub_branches, np.ndarray):
        return False
    if cb_ids.dtype != np.int64:
        return False
    if sub_sites.dtype != np.float64:
        return False
    if sub_branches.dtype != np.float64:
        return False
    if cb_ids.ndim != 2:
        return False
    if sub_sites.ndim != 2:
        return False
    if sub_branches.ndim != 1:
        return False
    if sub_sites.shape[0] != sub_branches.shape[0]:
        return False
    if cb_ids.shape[1] == 0:
        return False
    return True


def _can_use_cython_packed_shared_counts(packed_masks, remapped_cb_ids):
    if omega_cy is None:
        return False
    if not hasattr(omega_cy, 'calc_shared_counts_packed_uint8'):
        return False
    if not isinstance(packed_masks, np.ndarray):
        return False
    if not isinstance(remapped_cb_ids, np.ndarray):
        return False
    if packed_masks.dtype != np.uint8:
        return False
    if remapped_cb_ids.dtype != np.int64:
        return False
    if packed_masks.ndim != 3:
        return False
    if remapped_cb_ids.ndim != 2:
        return False
    if remapped_cb_ids.shape[1] == 0:
        return False
    return True


def _can_use_cython_pack_sampled_indices(sampled_site_indices):
    if omega_cy is None:
        return False
    if not hasattr(omega_cy, 'pack_sampled_site_indices_uint8'):
        return False
    if not isinstance(sampled_site_indices, np.ndarray):
        return False
    if sampled_site_indices.dtype != np.int64:
        return False
    if sampled_site_indices.ndim != 2:
        return False
    return True


def _calc_shared_counts_from_packed_masks(packed_masks, remapped_cb_ids):
    packed_masks = np.asarray(packed_masks, dtype=np.uint8)
    remapped_cb_ids = np.asarray(remapped_cb_ids, dtype=np.int64)
    if packed_masks.ndim != 3:
        raise ValueError('packed_masks should be a 3D array.')
    if remapped_cb_ids.ndim != 2:
        raise ValueError('remapped_cb_ids should be a 2D array.')
    if remapped_cb_ids.shape[1] <= 0:
        raise ValueError('remapped_cb_ids should have at least one column.')
    if remapped_cb_ids.shape[0] == 0:
        return np.zeros((0, packed_masks.shape[1]), dtype=np.int32)
    if (remapped_cb_ids < 0).any():
        raise ValueError('remapped_cb_ids should be non-negative.')
    if remapped_cb_ids.max() >= packed_masks.shape[0]:
        raise ValueError('remapped_cb_ids contain out-of-range branch IDs.')
    if _can_use_cython_packed_shared_counts(packed_masks=packed_masks, remapped_cb_ids=remapped_cb_ids):
        try:
            return omega_cy.calc_shared_counts_packed_uint8(
                packed_masks=packed_masks,
                remapped_cb_ids=remapped_cb_ids,
            )
        except Exception:
            pass
    arity = remapped_cb_ids.shape[1]
    if arity == 1:
        return _UINT8_POPCOUNT[packed_masks[remapped_cb_ids[:, 0], :, :]].sum(axis=2, dtype=np.int32)
    if arity == 2:
        return _UINT8_POPCOUNT[np.bitwise_and(
            packed_masks[remapped_cb_ids[:, 0], :, :],
            packed_masks[remapped_cb_ids[:, 1], :, :],
        )].sum(axis=2, dtype=np.int32)
    shared = packed_masks[remapped_cb_ids[:, 0], :, :].copy()
    for col in range(1, arity):
        shared = np.bitwise_and(shared, packed_masks[remapped_cb_ids[:, col], :, :])
    return _UINT8_POPCOUNT[shared].sum(axis=2, dtype=np.int32)


def _pack_sampled_site_indices_to_uint8(sampled_site_indices, num_site):
    sampled_site_indices = np.asarray(sampled_site_indices, dtype=np.int64)
    if sampled_site_indices.ndim != 2:
        raise ValueError('sampled_site_indices should be a 2D array.')
    niter = int(sampled_site_indices.shape[0])
    size = int(sampled_site_indices.shape[1])
    num_site = int(num_site)
    if num_site < 0:
        raise ValueError('num_site should be >= 0.')
    num_packed_site = (num_site + 7) // 8
    packed = np.zeros((niter, num_packed_site), dtype=np.uint8)
    if (niter == 0) or (size == 0) or (num_site == 0):
        return packed
    if _can_use_cython_pack_sampled_indices(sampled_site_indices=sampled_site_indices):
        try:
            return omega_cy.pack_sampled_site_indices_uint8(
                sampled_site_indices=sampled_site_indices,
                num_site=num_site,
            )
        except Exception:
            pass
    row_indices = np.repeat(np.arange(niter, dtype=np.int64), size)
    flattened_sites = sampled_site_indices.reshape(-1)
    if (flattened_sites < 0).any() or (flattened_sites >= num_site).any():
        raise ValueError('sampled_site_indices contain out-of-range site IDs.')
    byte_indices = flattened_sites >> 3
    bit_offsets = flattened_sites & 7
    bit_values = (1 << (7 - bit_offsets)).astype(np.uint8, copy=False)
    np.bitwise_or.at(packed, (row_indices, byte_indices), bit_values)
    return packed


def _weighted_sample_without_replacement_packed(p, size, niter):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError('p should be a 1D array.')
    if size < 0:
        raise ValueError('size should be >= 0.')
    if niter < 0:
        raise ValueError('niter should be >= 0.')
    num_site = int(p.shape[0])
    num_packed_site = (num_site + 7) // 8
    packed = np.zeros(shape=(niter, num_packed_site), dtype=np.uint8)
    positive_sites = np.flatnonzero(p > 0).astype(np.int64, copy=False)
    num_positive_sites = positive_sites.shape[0]
    if (size == 0) or (num_positive_sites == 0) or (niter == 0):
        return packed
    if size > num_positive_sites:
        txt = 'Sample size ({}) exceeded number of positive-probability sites ({}) in quantile sampling.'
        raise ValueError(txt.format(size, num_positive_sites))
    if size == num_positive_sites:
        row_mask = np.zeros((num_packed_site,), dtype=np.uint8)
        byte_indices = positive_sites >> 3
        bit_offsets = positive_sites & 7
        bit_values = (1 << (7 - bit_offsets)).astype(np.uint8, copy=False)
        np.bitwise_or.at(row_mask, byte_indices, bit_values)
        packed[:, :] = row_mask[None, :]
        return packed
    # Efraimidis-Spirakis weighted sampling without replacement (A-ES).
    positive_weights = p[positive_sites].astype(np.float32, copy=False)
    keys = np.random.random((niter, num_positive_sites)).astype(np.float32, copy=False)
    np.log(keys, out=keys)
    keys /= positive_weights
    kth = num_positive_sites - size
    sampled_local_indices = np.argpartition(keys, kth=kth, axis=1)[:, kth:]
    sampled_site_indices = positive_sites[sampled_local_indices]
    return _pack_sampled_site_indices_to_uint8(sampled_site_indices=sampled_site_indices, num_site=num_site)


def _project_expected_state_block(parent_state_block, transition_prob, float_tol):
    if _can_use_cython_expected_state(parent_state_block, transition_prob):
        return omega_cy.project_expected_state_block_double(
            parent_state_block=parent_state_block,
            transition_prob=transition_prob,
            float_tol=float(float_tol),
        )
    expected_state_block = parent_state_block @ transition_prob
    expected_sum = expected_state_block.sum(axis=1)
    is_over = (expected_sum - 1) > float_tol
    if is_over.any():
        expected_state_block[is_over, :] /= expected_sum[is_over][:, None]
    return expected_state_block


def _resolve_sub_sites(g, sub_sg, mode, sg, a, d, obs_col):
    if g['asrv'] in ['each', 'file_each']:
        sub_sites = substitution.get_each_sub_sites(sub_sg, mode, sg, a, d, g)
        return _apply_epistasis_to_sub_sites(sub_sites=sub_sites, obs_col=obs_col, g=g)
    if (g['asrv']=='sn'):
        if (obs_col.startswith('OCS')):
            sub_sites = g['sub_sites']['S']
            return _apply_epistasis_to_sub_sites(sub_sites=sub_sites, obs_col=obs_col, g=g)
        if (obs_col.startswith('OCN')):
            sub_sites = g['sub_sites']['N']
            return _apply_epistasis_to_sub_sites(sub_sites=sub_sites, obs_col=obs_col, g=g)
    sub_sites = g['sub_sites'][g['asrv']]
    return _apply_epistasis_to_sub_sites(sub_sites=sub_sites, obs_col=obs_col, g=g)


def _get_static_sub_sites_if_available(g, sub_sg, mode, obs_col):
    asrv_mode = str(g.get('asrv', 'no')).strip().lower()
    if asrv_mode in ['each', 'file_each']:
        return None
    return _resolve_sub_sites(g=g, sub_sg=sub_sg, mode=mode, sg=0, a=0, d=0, obs_col=obs_col)


def _calc_cb_site_overlap(cb_ids, sub_sites, float_type):
    cb_ids = np.asarray(cb_ids, dtype=np.int64)
    sub_sites = np.asarray(sub_sites, dtype=float_type)
    if cb_ids.ndim != 2:
        raise ValueError('cb_ids should be a 2D array.')
    if sub_sites.ndim != 2:
        raise ValueError('sub_sites should be a 2D array.')
    if cb_ids.shape[0] == 0:
        return np.zeros(shape=(0,), dtype=float_type)
    if (cb_ids < 0).any():
        raise ValueError('cb_ids should be non-negative.')
    if sub_sites.shape[0] <= cb_ids.max():
        raise ValueError('cb_ids contain out-of-range branch IDs.')
    arity = cb_ids.shape[1]
    if arity <= 0:
        raise ValueError('cb_ids should have at least one column.')
    if arity == 1:
        bids = cb_ids[:, 0]
        return sub_sites[bids, :].sum(axis=1, dtype=float_type)
    if arity == 2:
        bid1 = cb_ids[:, 0]
        bid2 = cb_ids[:, 1]
        site_overlap = sub_sites[bid1, :] * sub_sites[bid2, :]
        return site_overlap.sum(axis=1, dtype=float_type)
    site_overlap = np.ones(shape=(cb_ids.shape[0], sub_sites.shape[1]), dtype=float_type)
    for col in range(arity):
        bids = cb_ids[:, col]
        site_overlap *= sub_sites[bids, :]
    return site_overlap.sum(axis=1, dtype=float_type)


def _calc_cb_branch_factor(cb_ids, sub_branches, float_type):
    cb_ids = np.asarray(cb_ids, dtype=np.int64)
    sub_branches = np.asarray(sub_branches, dtype=float_type)
    if cb_ids.ndim != 2:
        raise ValueError('cb_ids should be a 2D array.')
    if sub_branches.ndim != 1:
        raise ValueError('sub_branches should be a 1D array.')
    if cb_ids.shape[0] == 0:
        return np.zeros(shape=(0,), dtype=float_type)
    if (cb_ids < 0).any():
        raise ValueError('cb_ids should be non-negative.')
    if sub_branches.shape[0] <= cb_ids.max():
        raise ValueError('cb_ids contain out-of-range branch IDs.')
    if cb_ids.shape[1] <= 0:
        raise ValueError('cb_ids should have at least one column.')
    branch_factor = np.ones(shape=(cb_ids.shape[0],), dtype=float_type)
    for col in range(cb_ids.shape[1]):
        branch_factor *= sub_branches[cb_ids[:, col]]
    return branch_factor


def _calc_tmp_E_sum(cb_ids, sub_sites, sub_branches, float_type, cb_site_overlap=None):
    if cb_site_overlap is not None:
        cb_site_overlap = np.asarray(cb_site_overlap, dtype=float_type).reshape(-1)
        if cb_site_overlap.shape[0] != cb_ids.shape[0]:
            txt = 'cb_site_overlap length ({}) did not match number of cb rows ({}).'
            raise ValueError(txt.format(cb_site_overlap.shape[0], cb_ids.shape[0]))
        return cb_site_overlap * _calc_cb_branch_factor(
            cb_ids=cb_ids,
            sub_branches=sub_branches,
            float_type=float_type,
        )
    if _can_use_cython_tmp_E_sum(cb_ids=cb_ids, sub_sites=sub_sites, sub_branches=sub_branches):
        try:
            return omega_cy.calc_tmp_E_sum_double(
                cb_ids=cb_ids,
                sub_sites=sub_sites,
                sub_branches=sub_branches,
            )
        except Exception:
            pass
    if (cb_ids.shape[1] == 1):
        bids = cb_ids[:, 0]
        site_overlap = sub_sites[bids, :].sum(axis=1)
        return site_overlap * sub_branches[bids]
    if (cb_ids.shape[1] == 2):
        bid1 = cb_ids[:, 0]
        bid2 = cb_ids[:, 1]
        site_overlap = (sub_sites[bid1, :] * sub_sites[bid2, :]).sum(axis=1)
        return site_overlap * sub_branches[bid1] * sub_branches[bid2]
    tmp_E = np.ones(shape=(cb_ids.shape[0], sub_sites.shape[1]), dtype=float_type)
    branch_factor = np.ones(shape=(cb_ids.shape[0],), dtype=float_type)
    for col in range(cb_ids.shape[1]):
        bids = cb_ids[:, col]
        tmp_E *= sub_sites[bids, :]
        branch_factor *= sub_branches[bids]
    return tmp_E.sum(axis=1) * branch_factor


def _weighted_sample_without_replacement_masks(p, size, niter):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError('p should be a 1D array.')
    if size < 0:
        raise ValueError('size should be >= 0.')
    positive_sites = np.flatnonzero(p > 0)
    num_positive_sites = positive_sites.shape[0]
    masks = np.zeros(shape=(niter, p.shape[0]), dtype=bool)
    if (size == 0) or (num_positive_sites == 0):
        return masks
    if size > num_positive_sites:
        txt = 'Sample size ({}) exceeded number of positive-probability sites ({}) in quantile sampling.'
        raise ValueError(txt.format(size, num_positive_sites))
    if size == num_positive_sites:
        masks[:, positive_sites] = True
        return masks
    # Efraimidis-Spirakis weighted sampling without replacement (A-ES).
    positive_weights = p[positive_sites].astype(np.float32, copy=False)
    keys = np.random.random((niter, num_positive_sites)).astype(np.float32, copy=False)
    np.log(keys, out=keys)
    keys /= positive_weights
    kth = num_positive_sites - size
    sampled_local_indices = np.argpartition(keys, kth=kth, axis=1)[:, kth:]
    sampled_site_indices = positive_sites[sampled_local_indices]
    row_indices = np.arange(niter)[:, None]
    masks[row_indices, sampled_site_indices] = True
    return masks


def _resolve_omega_pvalue_rounding_mode(g):
    mode = 'round'
    if g is not None:
        mode = str(g.get('omega_pvalue_rounding', 'round')).strip().lower()
    allowed = {'round', 'stochastic', 'floor', 'ceil'}
    if mode not in allowed:
        raise ValueError('omega_pvalue_rounding should be one of round, stochastic, floor, ceil.')
    return mode


def _prepare_permutation_branch_sizes(sub_branches, niter, g):
    sub_branches = np.asarray(sub_branches)
    if sub_branches.ndim != 1:
        raise ValueError('sub_branches should be a 1D array.')
    if np.issubdtype(sub_branches.dtype, np.integer):
        out = sub_branches.astype(np.int64, copy=False)
        if (out < 0).any():
            raise ValueError('sub_branches should be non-negative.')
        return out
    sub_branches = sub_branches.astype(np.float64, copy=False)
    if not np.isfinite(sub_branches).all():
        raise ValueError('sub_branches should be finite.')
    sub_branches = np.clip(sub_branches, a_min=0.0, a_max=None)
    rounding_mode = _resolve_omega_pvalue_rounding_mode(g=g)
    if rounding_mode == 'stochastic':
        niter = int(niter)
        if niter <= 0:
            raise ValueError('niter should be a positive integer.')
        base = np.floor(sub_branches).astype(np.int64, copy=False)
        frac = sub_branches - base
        if (frac > 0).any():
            rand = np.random.random((sub_branches.shape[0], niter))
            inc = (rand < frac[:, None]).astype(np.int64, copy=False)
            return base[:, None] + inc
        return np.repeat(base[:, None], repeats=niter, axis=1)
    if rounding_mode == 'round':
        out = np.rint(sub_branches)
    elif rounding_mode == 'floor':
        out = np.floor(sub_branches)
    elif rounding_mode == 'ceil':
        out = np.ceil(sub_branches)
    else:
        raise ValueError('Unsupported omega_pvalue_rounding: {}'.format(rounding_mode))
    out = out.astype(np.int64, copy=False)
    if (out < 0).any():
        raise ValueError('sub_branches should be non-negative after rounding.')
    return out


def _calc_wallenius_inclusion_probabilities(site_weights, draw_size, float_type=np.float64):
    site_weights = np.asarray(site_weights, dtype=np.float64).reshape(-1)
    if site_weights.ndim != 1:
        raise ValueError('site_weights should be a 1D array.')
    if (site_weights < 0).any():
        raise ValueError('site_weights should be non-negative.')
    draw_size = int(draw_size)
    if draw_size < 0:
        raise ValueError('draw_size should be >= 0.')
    out = np.zeros(shape=site_weights.shape, dtype=np.float64)
    positive_mask = (site_weights > 0)
    positive_weights = site_weights[positive_mask]
    num_positive = int(positive_weights.shape[0])
    if (draw_size == 0) or (num_positive == 0):
        return out.astype(float_type, copy=False)
    if draw_size >= num_positive:
        out[positive_mask] = 1.0
        return out.astype(float_type, copy=False)

    target = float(draw_size)
    lo = 0.0
    hi = 1.0
    for _ in range(128):
        current = (-np.expm1(-hi * positive_weights)).sum(dtype=np.float64)
        if current >= target:
            break
        hi *= 2.0
    for _ in range(96):
        mid = (lo + hi) / 2.0
        current = (-np.expm1(-mid * positive_weights)).sum(dtype=np.float64)
        if current < target:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2.0
    out_positive = -np.expm1(-lam * positive_weights)
    out[positive_mask] = np.clip(out_positive, a_min=0.0, a_max=1.0)
    return out.astype(float_type, copy=False)


def _calc_wallenius_expected_overlap(cb_ids, sub_sites, sub_branches, g, float_type):
    cb_ids = np.asarray(cb_ids, dtype=np.int64)
    if cb_ids.ndim != 2:
        raise ValueError('cb_ids should be a 2D array.')
    if cb_ids.shape[0] == 0:
        return np.zeros(shape=(0,), dtype=float_type)
    if (cb_ids < 0).any():
        raise ValueError('cb_ids should be non-negative.')

    sub_branches = np.asarray(sub_branches, dtype=np.float64).reshape(-1)
    if sub_branches.ndim != 1:
        raise ValueError('sub_branches should be a 1D array.')
    if not np.isfinite(sub_branches).all():
        raise ValueError('sub_branches should be finite.')
    np.clip(sub_branches, a_min=0.0, a_max=None, out=sub_branches)

    sub_sites = np.asarray(sub_sites, dtype=np.float64)
    if sub_sites.ndim == 1:
        sub_sites = np.broadcast_to(sub_sites.reshape(1, -1), (sub_branches.shape[0], sub_sites.shape[0]))
    elif sub_sites.ndim != 2:
        raise ValueError('sub_sites should be a 1D or 2D array.')
    if sub_sites.shape[0] != sub_branches.shape[0]:
        txt = 'sub_sites branch axis ({}) and sub_branches length ({}) should match.'
        raise ValueError(txt.format(sub_sites.shape[0], sub_branches.shape[0]))
    if (sub_sites < 0).any():
        raise ValueError('sub_sites should be non-negative.')
    if sub_sites.shape[0] <= cb_ids.max():
        raise ValueError('cb_ids contain out-of-range branch IDs.')

    rounding_mode = _resolve_omega_pvalue_rounding_mode(g=g)
    inclusion_prob = np.zeros(shape=sub_sites.shape, dtype=np.float64)
    if rounding_mode == 'stochastic':
        base = np.floor(sub_branches).astype(np.int64, copy=False)
        frac = sub_branches - base
        for branch_id in range(sub_sites.shape[0]):
            branch_weights = sub_sites[branch_id, :]
            num_positive = int((branch_weights > 0).sum())
            if num_positive == 0:
                continue
            size_lo = int(np.clip(base[branch_id], a_min=0, a_max=num_positive))
            size_hi = int(np.clip(base[branch_id] + 1, a_min=0, a_max=num_positive))
            prob_lo = _calc_wallenius_inclusion_probabilities(
                site_weights=branch_weights,
                draw_size=size_lo,
                float_type=np.float64,
            )
            if (frac[branch_id] <= 0) or (size_lo == size_hi):
                inclusion_prob[branch_id, :] = prob_lo
            else:
                prob_hi = _calc_wallenius_inclusion_probabilities(
                    site_weights=branch_weights,
                    draw_size=size_hi,
                    float_type=np.float64,
                )
                inclusion_prob[branch_id, :] = (
                    (1.0 - float(frac[branch_id])) * prob_lo +
                    float(frac[branch_id]) * prob_hi
                )
    else:
        rounded_sizes = _prepare_permutation_branch_sizes(
            sub_branches=sub_branches,
            niter=1,
            g=g,
        )
        rounded_sizes = np.asarray(rounded_sizes, dtype=np.int64).reshape(-1)
        for branch_id in range(sub_sites.shape[0]):
            branch_weights = sub_sites[branch_id, :]
            num_positive = int((branch_weights > 0).sum())
            if num_positive == 0:
                continue
            draw_size = int(np.clip(rounded_sizes[branch_id], a_min=0, a_max=num_positive))
            inclusion_prob[branch_id, :] = _calc_wallenius_inclusion_probabilities(
                site_weights=branch_weights,
                draw_size=draw_size,
                float_type=np.float64,
            )
    return _calc_cb_site_overlap(
        cb_ids=cb_ids,
        sub_sites=inclusion_prob,
        float_type=float_type,
    )


def _fill_packed_masks_for_sizes(packed_masks_branch, site_p, size_values):
    size_values = np.asarray(size_values, dtype=np.int64).reshape(-1)
    if packed_masks_branch.shape[0] != size_values.shape[0]:
        txt = 'packed_masks_branch iterations ({}) and size_values ({}) should match.'
        raise ValueError(txt.format(packed_masks_branch.shape[0], size_values.shape[0]))
    if (size_values < 0).any():
        raise ValueError('size_values should be non-negative.')
    site_p = np.asarray(site_p, dtype=np.float64).reshape(-1)
    num_positive_sites = int((site_p > 0).sum())
    if num_positive_sites == 0:
        return None
    positive_sizes = np.unique(size_values[size_values > 0])
    for size in positive_sizes:
        capped_size = int(size)
        if capped_size > num_positive_sites:
            capped_size = num_positive_sites
        if capped_size <= 0:
            continue
        iter_indices = np.where(size_values == size)[0]
        if iter_indices.shape[0] == 0:
            continue
        packed_masks_branch[iter_indices, :] = _weighted_sample_without_replacement_packed(
            p=site_p,
            size=capped_size,
            niter=iter_indices.shape[0],
        )
    return None


def _get_permutations_fast(cb_ids, sub_branches, p, niter):
    niter = int(niter)
    if niter < 0:
        raise ValueError('niter should be >= 0.')
    cb_ids = np.asarray(cb_ids, dtype=np.int64)
    if cb_ids.ndim != 2:
        raise ValueError('cb_ids should be a 2D array.')
    if cb_ids.shape[0] == 0:
        return np.zeros((0, niter), dtype=np.int32)
    if (cb_ids < 0).any():
        raise ValueError('cb_ids should be non-negative.')
    sub_branches = np.asarray(sub_branches)
    if sub_branches.ndim not in [1, 2]:
        raise ValueError('sub_branches should be a 1D or 2D array.')
    is_per_iteration_sizes = (sub_branches.ndim == 2)
    if is_per_iteration_sizes:
        if sub_branches.shape[1] != niter:
            txt = 'When sub_branches is 2D, its number of columns ({}) should match niter ({}).'
            raise ValueError(txt.format(sub_branches.shape[1], niter))
        sub_branches = np.asarray(sub_branches, dtype=np.int64)
    else:
        sub_branches = np.asarray(sub_branches, dtype=np.int64)
    if sub_branches.shape[0] <= cb_ids.max():
        raise ValueError('cb_ids contain out-of-range branch IDs.')
    if (sub_branches < 0).any():
        raise ValueError('sub_branches should be non-negative.')
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        num_site = p.shape[0]
        shared_site_p = p
        branch_site_p = None
    elif p.ndim == 2:
        if p.shape[0] != sub_branches.shape[0]:
            txt = 'When p is 2D, its number of rows ({}) should match sub_branches length ({}).'
            raise ValueError(txt.format(p.shape[0], sub_branches.shape[0]))
        num_site = p.shape[1]
        shared_site_p = None
        branch_site_p = p
    else:
        raise ValueError('p should be a 1D or 2D array.')
    if is_per_iteration_sizes:
        active_branch_ids, inverse_branch_ids = np.unique(cb_ids, return_inverse=True)
        remapped_cb_ids = inverse_branch_ids.reshape(cb_ids.shape)
        active_sub_branches = sub_branches[active_branch_ids, :]
        num_branch = active_sub_branches.shape[0]
        num_packed_site = (num_site + 7) // 8
        packed_masks = np.zeros(shape=(num_branch, niter, num_packed_site), dtype=np.uint8)
        if shared_site_p is not None:
            num_positive_sites = int((shared_site_p > 0).sum())
            if num_positive_sites == 0:
                return np.zeros((cb_ids.shape[0], niter), dtype=np.int32)
            for branch_id in range(num_branch):
                size_values = active_sub_branches[branch_id, :]
                if (size_values > 0).sum() == 0:
                    continue
                _fill_packed_masks_for_sizes(
                    packed_masks_branch=packed_masks[branch_id, :, :],
                    site_p=shared_site_p,
                    size_values=size_values,
                )
        else:
            active_site_p = branch_site_p[active_branch_ids, :]
            first_site_p = active_site_p[0, :]
            is_shared_site_p = np.array_equal(active_site_p, np.broadcast_to(first_site_p, active_site_p.shape))
            if is_shared_site_p:
                num_positive_sites = int((first_site_p > 0).sum())
                if num_positive_sites == 0:
                    return np.zeros((cb_ids.shape[0], niter), dtype=np.int32)
                for branch_id in range(num_branch):
                    size_values = active_sub_branches[branch_id, :]
                    if (size_values > 0).sum() == 0:
                        continue
                    _fill_packed_masks_for_sizes(
                        packed_masks_branch=packed_masks[branch_id, :, :],
                        site_p=first_site_p,
                        size_values=size_values,
                    )
            else:
                for branch_id in range(num_branch):
                    size_values = active_sub_branches[branch_id, :]
                    if (size_values > 0).sum() == 0:
                        continue
                    _fill_packed_masks_for_sizes(
                        packed_masks_branch=packed_masks[branch_id, :, :],
                        site_p=active_site_p[branch_id, :],
                        size_values=size_values,
                    )
        return _calc_shared_counts_from_packed_masks(
            packed_masks=packed_masks,
            remapped_cb_ids=remapped_cb_ids,
        )
    cb_branch_sizes = sub_branches[cb_ids]
    is_active_row = (cb_branch_sizes > 0).all(axis=1)
    if not is_active_row.any():
        return np.zeros((cb_ids.shape[0], niter), dtype=np.int32)
    active_row_indices = np.where(is_active_row)[0]
    active_cb_ids = cb_ids[active_row_indices, :]
    active_branch_ids, inverse_branch_ids = np.unique(active_cb_ids, return_inverse=True)
    remapped_active_cb_ids = inverse_branch_ids.reshape(active_cb_ids.shape)
    active_sub_branches = sub_branches[active_branch_ids]

    num_branch = active_sub_branches.shape[0]
    num_packed_site = (num_site + 7) // 8
    packed_masks = np.zeros(shape=(num_branch, niter, num_packed_site), dtype=np.uint8)
    if shared_site_p is not None:
        num_positive_sites = int((shared_site_p > 0).sum())
        if num_positive_sites == 0:
            return np.zeros((cb_ids.shape[0], niter), dtype=np.int32)
        previous_branch_id_by_size = dict()
        for branch_id in range(num_branch):
            size = int(active_sub_branches[branch_id])
            if size == 0:
                continue
            # Rounding from floating substitution counts can slightly exceed the
            # number of positive-probability sites. Cap to valid bounds.
            if size > num_positive_sites:
                size = num_positive_sites
            if size in previous_branch_id_by_size:
                prev_branch_id = previous_branch_id_by_size[size]
                packed_masks[branch_id, :, :] = packed_masks[prev_branch_id, np.random.permutation(niter), :]
                continue
            previous_branch_id_by_size[size] = branch_id
            packed_masks[branch_id, :, :] = _weighted_sample_without_replacement_packed(
                p=shared_site_p,
                size=size,
                niter=niter,
            )
    else:
        active_site_p = branch_site_p[active_branch_ids, :]
        first_site_p = active_site_p[0, :]
        is_shared_site_p = np.array_equal(active_site_p, np.broadcast_to(first_site_p, active_site_p.shape))
        if is_shared_site_p:
            num_positive_sites = int((first_site_p > 0).sum())
            if num_positive_sites == 0:
                return np.zeros((cb_ids.shape[0], niter), dtype=np.int32)
            previous_branch_id_by_size = dict()
            for branch_id in range(num_branch):
                size = int(active_sub_branches[branch_id])
                if size == 0:
                    continue
                if size > num_positive_sites:
                    size = num_positive_sites
                if size in previous_branch_id_by_size:
                    prev_branch_id = previous_branch_id_by_size[size]
                    packed_masks[branch_id, :, :] = packed_masks[prev_branch_id, np.random.permutation(niter), :]
                    continue
                previous_branch_id_by_size[size] = branch_id
                packed_masks[branch_id, :, :] = _weighted_sample_without_replacement_packed(
                    p=first_site_p,
                    size=size,
                    niter=niter,
                )
        else:
            for branch_id in range(num_branch):
                size = int(active_sub_branches[branch_id])
                if size == 0:
                    continue
                site_p = active_site_p[branch_id, :]
                num_positive_sites = int((site_p > 0).sum())
                if num_positive_sites == 0:
                    continue
                if size > num_positive_sites:
                    size = num_positive_sites
                packed_masks[branch_id, :, :] = _weighted_sample_without_replacement_packed(
                    p=site_p,
                    size=size,
                    niter=niter,
                )

    active_out = _calc_shared_counts_from_packed_masks(
        packed_masks=packed_masks,
        remapped_cb_ids=remapped_active_cb_ids,
    )
    out = np.zeros((cb_ids.shape[0], niter), dtype=np.int32)
    out[active_row_indices, :] = active_out
    return out


def _resolve_quantile_parallel_plan(cb_rows, num_categories, quantile_niter, requested_n_jobs, requested_chunk_factor):
    n_jobs = int(requested_n_jobs)
    chunk_factor = int(requested_chunk_factor)
    if n_jobs <= 1:
        return 1, max(chunk_factor, 1)
    # For small branch-set workloads, thread startup and local-buffer reduction dominate.
    # Fall back to single-thread execution to avoid regressions under --threads > 1.
    workload = int(cb_rows) * int(num_categories) * int(quantile_niter)
    if (cb_rows <= 4) or (workload < 20_000_000):
        return 1, max(chunk_factor, 1)
    return n_jobs, max(chunk_factor, 4)


def _resolve_quantile_niter_schedule(g, quantile_niter):
    if g is not None:
        schedule = g.get('quantile_niter_schedule', None)
        if schedule is not None:
            schedule = [int(v) for v in schedule]
            if len(schedule) == 0:
                raise ValueError('quantile_niter_schedule should contain at least one stage.')
            if min(schedule) <= 0:
                raise ValueError('quantile_niter_schedule should contain positive integers.')
            for prev, curr in zip(schedule, schedule[1:]):
                if curr <= prev:
                    raise ValueError('quantile_niter_schedule should be strictly increasing.')
            return schedule
    niter = int(quantile_niter)
    if niter <= 0:
        raise ValueError('quantile_niter should be a positive integer.')
    return [niter]


def _resolve_quantile_refine_edge_bins(g):
    if g is None:
        return 0
    edge_bins = int(g.get('quantile_refine_edge_bins', 0))
    if edge_bins < 0:
        raise ValueError('quantile_refine_edge_bins should be >= 0.')
    return edge_bins


def _needs_quantile_refinement(probability_values, quantile_niter, edge_bins):
    probability_values = np.asarray(probability_values, dtype=np.float64)
    if probability_values.ndim != 1:
        raise ValueError('probability_values should be a 1D array.')
    if int(edge_bins) <= 0:
        return np.zeros(shape=probability_values.shape, dtype=bool)
    niter = int(quantile_niter)
    if niter <= 0:
        raise ValueError('quantile_niter should be a positive integer.')
    edge = float(edge_bins) / float(niter)
    return (probability_values <= edge) | (probability_values >= (1.0 - edge))


def _calc_quantile_count_matrix(
    mode,
    cb_ids,
    sub_sg,
    sub_bg,
    quantile_niter,
    obs_col,
    num_gad_combinat,
    list_igad,
    g,
    static_sub_sites,
):
    cb_ids = np.asarray(cb_ids, dtype=np.int64)
    quantile_niter = int(quantile_niter)
    if cb_ids.ndim != 2:
        raise ValueError('cb_ids should be a 2D array.')
    if quantile_niter <= 0:
        raise ValueError('quantile_niter should be a positive integer.')
    if cb_ids.shape[0] == 0:
        return np.zeros(shape=(0, quantile_niter), dtype=np.int32)
    requested_n_jobs = parallel.resolve_n_jobs(num_items=len(list_igad), threads=g['threads'])
    chunk_factor = parallel.resolve_chunk_factor(g=g, task='general')
    n_jobs, chunk_factor = _resolve_quantile_parallel_plan(
        cb_rows=cb_ids.shape[0],
        num_categories=len(list_igad),
        quantile_niter=quantile_niter,
        requested_n_jobs=requested_n_jobs,
        requested_chunk_factor=chunk_factor,
    )
    igad_chunks,mmap_start_not_necessary_here = parallel.get_chunks(list_igad, n_jobs, chunk_factor=chunk_factor)
    axis = (cb_ids.shape[0], quantile_niter)
    dfq = np.zeros(shape=axis, dtype=np.int32)
    if n_jobs == 1:
        joblib_calc_quantile(
            mode,
            cb_ids,
            sub_sg,
            sub_bg,
            dfq,
            quantile_niter,
            obs_col,
            num_gad_combinat,
            list_igad,
            g,
            static_sub_sites=static_sub_sites,
        )
        return dfq
    tasks = [
        (
            mode,
            cb_ids,
            sub_sg,
            sub_bg,
            quantile_niter,
            obs_col,
            num_gad_combinat,
            igad_chunk,
            g,
            static_sub_sites,
        )
        for igad_chunk in igad_chunks
    ]
    chunk_dfs = parallel.run_starmap(
        func=_calc_quantile_chunk_local,
        args_iterable=tasks,
        n_jobs=n_jobs,
        backend='threading',
    )
    for dfq_chunk in chunk_dfs:
        dfq += dfq_chunk
    return dfq


def _calc_quantile_probabilities(
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
    cb_ids = np.asarray(cb_ids, dtype=np.int64)
    obs_values = np.asarray(obs_values, dtype=g['float_type']).reshape(-1)
    quantile_niter = int(quantile_niter)
    if cb_ids.ndim != 2:
        raise ValueError('cb_ids should be a 2D array.')
    if obs_values.ndim != 1:
        raise ValueError('obs_values should be a 1D array.')
    if cb_ids.shape[0] != obs_values.shape[0]:
        txt = 'cb_ids rows ({}) and obs_values length ({}) should match.'
        raise ValueError(txt.format(cb_ids.shape[0], obs_values.shape[0]))
    if quantile_niter <= 0:
        raise ValueError('quantile_niter should be a positive integer.')
    if cb_ids.shape[0] == 0:
        return np.zeros(shape=(0,), dtype=g['float_type'])
    dfq = _calc_quantile_count_matrix(
        mode=mode,
        cb_ids=cb_ids,
        sub_sg=sub_sg,
        sub_bg=sub_bg,
        quantile_niter=quantile_niter,
        obs_col=obs_col,
        num_gad_combinat=num_gad_combinat,
        list_igad=list_igad,
        g=g,
        static_sub_sites=static_sub_sites,
    )
    gt_ranks = (dfq < obs_values[:, None]).sum(axis=1, dtype=np.int64)
    ge_ranks = (dfq <= obs_values[:, None]).sum(axis=1, dtype=np.int64)
    probabilities = ((gt_ranks + ge_ranks) / 2) / quantile_niter
    return probabilities.astype(g['float_type'], copy=False)


def _collect_expected_state_branch_jobs(tree, mode, num_node, float_tol):
    jobs = list()
    if mode == 'cdn':
        dist_prop = 'SNdist'
    elif mode in ['pep', 'nsy']:
        dist_prop = 'Ndist'
    else:
        raise ValueError('Unsupported expected-state mode: {}'.format(mode))
    for node in tree.traverse():
        if ete.is_root(node):
            continue
        branch_length = max(float(ete.get_prop(node, dist_prop, 0)), 0.0)
        if branch_length < float_tol:
            continue
        nl = int(ete.get_prop(node, "numerical_label"))
        parent_nl = int(ete.get_prop(node.up, "numerical_label"))
        if (parent_nl < 0) or (parent_nl >= num_node):
            continue
        if (nl < 0) or (nl >= num_node):
            continue
        jobs.append((nl, parent_nl, branch_length))
    return jobs


def _resolve_expected_state_n_jobs(num_branch_jobs, num_site, num_state, g):
    threads = int(g.get('threads', 1))
    if (threads <= 1) or (num_branch_jobs <= 1):
        return 1, 0
    estimated_work = int(num_branch_jobs) * int(num_site) * int(num_state) * int(num_state)
    n_jobs = parallel.resolve_adaptive_n_jobs(
        num_items=estimated_work,
        threads=threads,
        min_items_for_parallel=int(g.get('parallel_min_items_expected_state', 5000000000)),
        min_items_per_job=int(g.get('parallel_min_items_per_job_expected_state', 1000000000)),
    )
    max_jobs_by_branch = parallel.resolve_n_jobs(num_items=num_branch_jobs, threads=threads)
    return min(n_jobs, max_jobs_by_branch), estimated_work


def _project_expected_state_chunk(
    branch_jobs,
    state,
    stateE,
    unique_site_rates,
    rate_site_indices,
    inst,
    float_tol,
):
    if len(branch_jobs) == 0:
        return None
    transition_prob_cache = dict()
    for nl, parent_nl, branch_length in branch_jobs:
        inst_bl = inst * branch_length
        for site_rate, site_indices in zip(unique_site_rates, rate_site_indices):
            if site_indices.shape[0] == 0:
                continue
            cache_key = (branch_length, float(site_rate))
            transition_prob = transition_prob_cache.get(cache_key, None)
            if transition_prob is None:
                inst_bl_site = inst_bl * site_rate
                # Confirmed this implementation (with expm) correctly replicated the example in this instruction (Huelsenbeck, 2012)
                # https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
                transition_prob = expm(inst_bl_site)
                transition_prob_cache[cache_key] = transition_prob
            parent_state_block = state[parent_nl, site_indices, :]
            stateE[nl, site_indices, :] = _project_expected_state_block(
                parent_state_block=parent_state_block,
                transition_prob=transition_prob,
                float_tol=float_tol,
            )
    return None


def calc_E_mean(mode, cb_ids, sub_sg, sub_bg, obs_col, list_igad, g, static_sub_sites=None, cb_site_overlap=None):
    E_b = np.zeros(shape=(cb_ids.shape[0],), dtype=g['float_type'])
    for i,sg,a,d in list_igad:
        if (a==d):
            continue
        if static_sub_sites is None:
            sub_sites = _resolve_sub_sites(g=g, sub_sg=sub_sg, mode=mode, sg=sg, a=a, d=d, obs_col=obs_col)
        else:
            sub_sites = static_sub_sites
        sub_branches = substitution.get_sub_branches(sub_bg, mode, sg, a, d)
        E_b += _calc_wallenius_expected_overlap(
            cb_ids=cb_ids,
            sub_sites=sub_sites,
            sub_branches=sub_branches,
            g=g,
            float_type=g['float_type'],
        )
    return E_b


def joblib_calc_E_mean(
    mode,
    cb_ids,
    sub_sg,
    sub_bg,
    dfEb,
    obs_col,
    num_gad_combinat,
    igad_chunk,
    g,
    static_sub_sites=None,
    cb_site_overlap=None,
):
    iter_start = time.time()
    if (igad_chunk==[]):
        return None # This happens when the number of iteration is smaller than --threads
    i_start = igad_chunk[0][0]
    for i,sg,a,d in igad_chunk:
        if (a==d):
            continue
        if static_sub_sites is None:
            sub_sites = _resolve_sub_sites(g=g, sub_sg=sub_sg, mode=mode, sg=sg, a=a, d=d, obs_col=obs_col)
        else:
            sub_sites = static_sub_sites
        sub_branches = substitution.get_sub_branches(sub_bg, mode, sg, a, d)
        dfEb += _calc_wallenius_expected_overlap(
            cb_ids=cb_ids,
            sub_sites=sub_sites,
            sub_branches=sub_branches,
            g=g,
            float_type=g['float_type'],
        )
    txt = 'E{}: {}-{}th of {} matrix_group/ancestral_state/derived_state combinations. Time elapsed: {:,} [sec]'
    print(txt.format(obs_col, i_start, i, num_gad_combinat, int(time.time()-iter_start)), flush=True)


def joblib_calc_quantile(
    mode,
    cb_ids,
    sub_sg,
    sub_bg,
    dfq,
    quantile_niter,
    obs_col,
    num_gad_combinat,
    igad_chunk,
    g,
    static_sub_sites=None,
):
    for i,sg,a,d in igad_chunk:
        if (a==d):
            continue
        if static_sub_sites is None:
            sub_sites = _resolve_sub_sites(g=g, sub_sg=sub_sg, mode=mode, sg=sg, a=a, d=d, obs_col=obs_col)
        else:
            sub_sites = static_sub_sites
        sub_branches = substitution.get_sub_branches(sub_bg, mode, sg, a, d)
        if np.asarray(sub_sites).ndim == 1:
            p = sub_sites
            if p.sum()==0:
                continue
        else:
            p = sub_sites
            if (p.sum(axis=1) > 0).sum() == 0:
                continue
        pm_start = time.time()
        sub_branches = _prepare_permutation_branch_sizes(
            sub_branches=sub_branches,
            niter=quantile_niter,
            g=g,
        )
        dfq[:,:] += _get_permutations_fast(cb_ids, sub_branches, p, quantile_niter)
        txt = '{}: {}/{} matrix_group/ancestral_state/derived_state combinations. Time elapsed for {:,} permutation: {:,} [sec]'
        print(txt.format(obs_col, i+1, num_gad_combinat, quantile_niter, int(time.time()-pm_start)), flush=True)

def _calc_E_mean_chunk_to_mmap(
    mode,
    cb_ids,
    sub_sg,
    sub_bg,
    mmap_out,
    dtype,
    shape,
    obs_col,
    num_gad_combinat,
    igad_chunk,
    g,
    static_sub_sites,
    cb_site_overlap,
):
    dfEb = np.memmap(filename=mmap_out, dtype=dtype, shape=shape, mode='r+')
    joblib_calc_E_mean(
        mode,
        cb_ids,
        sub_sg,
        sub_bg,
        dfEb,
        obs_col,
        num_gad_combinat,
        igad_chunk,
        g,
        static_sub_sites=static_sub_sites,
        cb_site_overlap=cb_site_overlap,
    )
    dfEb.flush()

def _calc_quantile_chunk_local(
    mode,
    cb_ids,
    sub_sg,
    sub_bg,
    quantile_niter,
    obs_col,
    num_gad_combinat,
    igad_chunk,
    g,
    static_sub_sites,
):
    dfq_local = np.zeros(shape=(cb_ids.shape[0], quantile_niter), dtype=np.int32)
    joblib_calc_quantile(
        mode,
        cb_ids,
        sub_sg,
        sub_bg,
        dfq_local,
        quantile_niter,
        obs_col,
        num_gad_combinat,
        igad_chunk,
        g,
        static_sub_sites=static_sub_sites,
    )
    return dfq_local


def _prepare_substitution_quantile_components(sub_tensor, mode, SN, g):
    supported_modes = {'spe2spe', 'spe2any', 'any2spe', 'any2any'}
    if mode not in supported_modes:
        raise ValueError('Unsupported E-stat mode: {}'.format(mode))
    if isinstance(sub_tensor, substitution_sparse.SparseSubstitutionTensor):
        sub_bg, sub_sg = substitution_sparse.summarize_sparse_sub_tensor(sparse_tensor=sub_tensor, mode=mode)
    if mode=='spe2spe':
        if not isinstance(sub_tensor, substitution_sparse.SparseSubstitutionTensor):
            sub_bg = sub_tensor.sum(axis=1) # branch, matrix_group, ancestral_state, derived_state
            sub_sg = sub_tensor.sum(axis=0) # site, matrix_group, ancestral_state, derived_state
        list_gad = [ [g,a,d] for g,a,d in itertools.zip_longest(*g[SN+'_ind_nomissing_gad']) ]
    elif mode=='spe2any':
        if not isinstance(sub_tensor, substitution_sparse.SparseSubstitutionTensor):
            sub_bg = sub_tensor.sum(axis=(1, 4)) # branch, matrix_group, ancestral_state
            sub_sg = sub_tensor.sum(axis=(0, 4)) # site, matrix_group, ancestral_state
        list_gad = [ [g,a,'2any'] for g,a in itertools.zip_longest(*g[SN+'_ind_nomissing_ga']) ]
    elif mode=='any2spe':
        if not isinstance(sub_tensor, substitution_sparse.SparseSubstitutionTensor):
            sub_bg = sub_tensor.sum(axis=(1, 3)) # branch, matrix_group, derived_state
            sub_sg = sub_tensor.sum(axis=(0, 3)) # site, matrix_group, derived_state
        list_gad = [ [g,'any2',d] for g,d in itertools.zip_longest(*g[SN+'_ind_nomissing_gd']) ]
    elif mode=='any2any':
        if not isinstance(sub_tensor, substitution_sparse.SparseSubstitutionTensor):
            sub_bg = sub_tensor.sum(axis=(1, 3, 4)) # branch, matrix_group
            sub_sg = sub_tensor.sum(axis=(0, 3, 4)) # site, matrix_group
        list_gad = list(itertools.product(np.arange(sub_tensor.shape[2]), ['any2',], ['2any',]))
    num_gad_combinat = len(list_gad)
    list_igad = [ [i,]+list(items) for i,items in zip(range(num_gad_combinat), list_gad) ]
    obs_col = 'OC'+SN+mode
    return sub_bg, sub_sg, list_igad, obs_col, num_gad_combinat


def calc_E_stat(cb, sub_tensor, mode, stat='mean', quantile_niter=1000, SN='', g=None):
    if g is None:
        raise ValueError('g is required.')
    supported_stats = {'mean', 'quantile'}
    if stat not in supported_stats:
        raise ValueError('Unsupported E-stat summary statistic: {}'.format(stat))
    sub_bg, sub_sg, list_igad, obs_col, num_gad_combinat = _prepare_substitution_quantile_components(
        sub_tensor=sub_tensor,
        mode=mode,
        SN=SN,
        g=g,
    )
    txt = 'E{}{}: Total number of substitution categories after NaN removals: {}'
    print(txt.format(SN, mode, num_gad_combinat))
    cb_ids = _get_cb_ids(cb)
    static_sub_sites = _get_static_sub_sites_if_available(g=g, sub_sg=sub_sg, mode=mode, obs_col=obs_col)
    cb_site_overlap = None
    requested_n_jobs = parallel.resolve_n_jobs(num_items=len(list_igad), threads=g['threads'])
    chunk_factor = parallel.resolve_chunk_factor(g=g, task='general')
    n_jobs = requested_n_jobs
    igad_chunks = None
    if stat == 'mean':
        igad_chunks,mmap_start_not_necessary_here = parallel.get_chunks(
            list_igad,
            n_jobs,
            chunk_factor=chunk_factor,
        )
    if stat=='mean':
        if n_jobs == 1:
            E_b = calc_E_mean(
                mode,
                cb_ids,
                sub_sg,
                sub_bg,
                obs_col,
                list_igad,
                g,
                static_sub_sites=static_sub_sites,
                cb_site_overlap=cb_site_overlap,
            )
        else:
            my_dtype = sub_tensor.dtype
            if 'bool' in str(my_dtype): my_dtype = g['float_type']
            mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.dfEb.mmap')
            if os.path.exists(mmap_out): os.unlink(mmap_out)
            axis = (cb.shape[0],)
            dfEb = np.memmap(filename=mmap_out, dtype=my_dtype, shape=axis, mode='w+')
            tasks = [
                (
                    mode,
                    cb_ids,
                    sub_sg,
                    sub_bg,
                    mmap_out,
                    my_dtype,
                    axis,
                    obs_col,
                    num_gad_combinat,
                    igad_chunk,
                    g,
                    static_sub_sites,
                    cb_site_overlap,
                )
                for igad_chunk in igad_chunks
            ]
            parallel.run_starmap(
                func=_calc_E_mean_chunk_to_mmap,
                args_iterable=tasks,
                n_jobs=n_jobs,
                backend='threading',
            )
            dfEb.flush()
            E_b = dfEb
            del dfEb
            if os.path.exists(mmap_out): os.unlink(mmap_out)
    elif stat=='quantile':
        quantile_schedule = _resolve_quantile_niter_schedule(g=g, quantile_niter=quantile_niter)
        edge_bins = _resolve_quantile_refine_edge_bins(g=g)
        obs_values = cb.loc[:,obs_col].values.astype(g['float_type'], copy=False)
        E_b = np.zeros(shape=(cb.shape[0],), dtype=g['float_type'])
        active_rows = np.arange(cb.shape[0], dtype=np.int64)
        active_probs = None
        prev_total_niter = 0
        for stage_index, stage_niter in enumerate(quantile_schedule):
            stage_niter = int(stage_niter)
            incremental_niter = stage_niter - prev_total_niter
            if incremental_niter <= 0:
                txt = 'quantile_niter_schedule should be strictly increasing; got {} after {}.'
                raise ValueError(txt.format(stage_niter, prev_total_niter))
            txt = 'E{}{} quantile stage {}/{}: cumulative_niter={:,}, incremental_niter={:,}, rows={:,}'
            print(
                txt.format(
                    SN,
                    mode,
                    stage_index + 1,
                    len(quantile_schedule),
                    stage_niter,
                    incremental_niter,
                    active_rows.shape[0],
                ),
                flush=True,
            )
            if active_rows.shape[0] == 0:
                break
            incremental_probs = _calc_quantile_probabilities(
                mode,
                cb_ids[active_rows, :],
                obs_values[active_rows],
                sub_sg,
                sub_bg,
                incremental_niter,
                obs_col,
                num_gad_combinat,
                list_igad,
                g,
                static_sub_sites=static_sub_sites,
            )
            if prev_total_niter == 0:
                stage_probs = incremental_probs
            else:
                stage_probs = (
                    (active_probs * prev_total_niter) +
                    (incremental_probs * incremental_niter)
                ) / stage_niter
            E_b[active_rows] = stage_probs
            is_last_stage = (stage_index == (len(quantile_schedule) - 1))
            if is_last_stage:
                break
            refine_mask = _needs_quantile_refinement(
                probability_values=stage_probs,
                quantile_niter=stage_niter,
                edge_bins=edge_bins,
            )
            next_rows = active_rows[refine_mask]
            txt = 'E{}{} quantile refinement after stage {}: rows {} -> {} (edge_bins={})'
            print(
                txt.format(SN, mode, stage_index + 1, active_rows.shape[0], next_rows.shape[0], edge_bins),
                flush=True,
            )
            active_rows = next_rows
            active_probs = stage_probs[refine_mask]
            prev_total_niter = stage_niter
    return E_b

def subroot_E2nan(cb, tree):
    id_cols = cb.columns[cb.columns.str.startswith('branch_id_')]
    E_cols = cb.columns[cb.columns.str.startswith('E')]
    if (E_cols.shape[0]==0):
        return cb
    for node in tree.traverse():
        continue_flag = 1
        if ete.is_root(node):
            continue_flag = 0
        elif ete.is_root(node.up):
            continue_flag = 0
        if continue_flag:
            continue
        for id_col in id_cols:
            is_node = (cb.loc[:,id_col]==ete.get_prop(node, "numerical_label"))
            cb.loc[is_node,E_cols] = np.nan
    return cb

def get_E(cb, g, ON_tensor, OS_tensor):
    requested_output_stats = _resolve_requested_output_stats(g)
    base_stats = output_stat.get_required_base_stats(requested_output_stats)
    if (g['omegaC_method']=='modelfree'):
        ON_gad, ON_ga, ON_gd = substitution.get_group_state_totals(ON_tensor)
        OS_gad, OS_ga, OS_gd = substitution.get_group_state_totals(OS_tensor)
        g['N_ind_nomissing_gad'] = np.where(ON_gad!=0)
        g['N_ind_nomissing_ga'] = np.where(ON_ga!=0)
        g['N_ind_nomissing_gd'] = np.where(ON_gd!=0)
        g['S_ind_nomissing_gad'] = np.where(OS_gad!=0)
        g['S_ind_nomissing_ga'] = np.where(OS_ga!=0)
        g['S_ind_nomissing_gd'] = np.where(OS_gd!=0)
        for st in base_stats:
            cb['ECN'+st] = calc_E_stat(cb, ON_tensor, mode=st, stat='mean', SN='N', g=g)
            cb['ECS'+st] = calc_E_stat(cb, OS_tensor, mode=st, stat='mean', SN='S', g=g)
    if (g['omegaC_method']=='submodel'):
        id_cols = cb.columns[cb.columns.str.startswith('branch_id_')]
        state_nsyE = get_exp_state(g=g, mode='nsy')
        if (g['current_arity']==2):
            g['EN_tensor'] = substitution.get_substitution_tensor(state_nsyE, g['state_nsy'], mode='asis', g=g, mmap_attr='EN')
        txt = 'Number of total empirically expected nonsynonymous substitutions in the tree: {:,.2f}'
        print(txt.format(substitution.get_total_substitution(g['EN_tensor'])))
        print('Preparing the ECN table with up to {:,} process(es).'.format(g['threads']), flush=True)
        cbEN = substitution.get_cb(
            cb.loc[:,id_cols].values,
            g['EN_tensor'],
            g,
            'ECN',
            selected_base_stats=base_stats,
        )
        cb = table.merge_tables(cb, cbEN)
        del state_nsyE,cbEN
        state_cdnE = get_exp_state(g=g, mode='cdn')
        if (g['current_arity'] == 2):
            g['ES_tensor'] = substitution.get_substitution_tensor(state_cdnE, g['state_cdn'], mode='syn', g=g, mmap_attr='ES')
        txt = 'Number of total empirically expected synonymous substitutions in the tree: {:,.2f}'
        print(txt.format(substitution.get_total_substitution(g['ES_tensor'])))
        print('Preparing the ECS table with up to {:,} process(es).'.format(g['threads']), flush=True)
        cbES = substitution.get_cb(
            cb.loc[:,id_cols].values,
            g['ES_tensor'],
            g,
            'ECS',
            selected_base_stats=base_stats,
        )
        cb = table.merge_tables(cb, cbES)
        del state_cdnE,cbES
    if g['calc_quantile']:
        for st in base_stats:
            cb['QCN'+st] = calc_E_stat(cb, ON_tensor, mode=st, stat='quantile', SN='N', g=g)
            cb['QCS'+st] = calc_E_stat(cb, OS_tensor, mode=st, stat='quantile', SN='S', g=g)
    cb = substitution.add_dif_stats(cb, g['float_tol'], prefix='EC', output_stats=requested_output_stats)
    cb = subroot_E2nan(cb, tree=g['tree'])
    return cb

def get_exp_state(g, mode):
    if mode=='cdn':
        state = g['state_cdn'].astype(g['float_type'])
        inst = g['instantaneous_codon_rate_matrix']
    elif mode=='pep':
        state = g['state_pep'].astype(g['float_type'])
        inst = g['instantaneous_aa_rate_matrix']
    elif mode=='nsy':
        state = g['state_nsy'].astype(g['float_type'])
        inst = g['instantaneous_nsy_rate_matrix']
    else:
        raise ValueError('Unsupported expected-state mode: {}'.format(mode))
    stateE = np.zeros_like(state, dtype=g['float_type'])
    rate_values = np.asarray(g['iqtree_rate_values'], dtype=g['float_type'])
    if rate_values.ndim != 1:
        rate_values = rate_values.reshape(-1)
    unique_site_rates, inverse_rate_indices = np.unique(rate_values, return_inverse=True)
    rate_site_indices = [np.where(inverse_rate_indices == i)[0] for i in range(unique_site_rates.shape[0])]
    branch_jobs = _collect_expected_state_branch_jobs(
        tree=g['tree'],
        mode=mode,
        num_node=stateE.shape[0],
        float_tol=float(g['float_tol']),
    )
    n_jobs, estimated_work = _resolve_expected_state_n_jobs(
        num_branch_jobs=len(branch_jobs),
        num_site=state.shape[1],
        num_state=state.shape[2],
        g=g,
    )
    txt = 'Expected-state scheduler (mode={}): branches={}, site_rates={}, estimated_work={}, workers={} (threads={})'
    print(
        txt.format(
            mode,
            len(branch_jobs),
            unique_site_rates.shape[0],
            estimated_work,
            n_jobs,
            int(g.get('threads', 1)),
        ),
        flush=True,
    )
    if n_jobs == 1:
        _project_expected_state_chunk(
            branch_jobs=branch_jobs,
            state=state,
            stateE=stateE,
            unique_site_rates=unique_site_rates,
            rate_site_indices=rate_site_indices,
            inst=inst,
            float_tol=float(g['float_tol']),
        )
    else:
        chunk_factor = parallel.resolve_chunk_factor(g=g, task='general')
        branch_chunks, _ = parallel.get_chunks(input_data=branch_jobs, threads=n_jobs, chunk_factor=chunk_factor)
        tasks = [
            (chunk, state, stateE, unique_site_rates, rate_site_indices, inst, float(g['float_tol']))
            for chunk in branch_chunks
        ]
        parallel.run_starmap(
            func=_project_expected_state_chunk,
            args_iterable=tasks,
            n_jobs=n_jobs,
            backend='threading',
        )
    max_stateE = stateE.sum(axis=(2)).max()
    if (max_stateE - 1) >= g['float_tol']:
        raise AssertionError('Total probability of expected states should not exceed 1. {}'.format(max_stateE))
    return stateE


def _calc_raw_rate(obs, exp, float_tol):
    obs = np.asarray(obs, dtype=np.float64)
    exp = np.asarray(exp, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = obs / exp
    out[obs < float_tol] = 0
    return out


def _calc_raw_omega(dNc, dSc, float_tol):
    dNc = np.asarray(dNc, dtype=np.float64)
    dSc = np.asarray(dSc, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        omega = dNc / dSc
    omega[dNc < float_tol] = 0
    return omega


def _calibrate_dsc_vector(dNc_values, dSc_values, transformation='quantile'):
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
    quantiles = ranks / ranks.max()
    if transformation == 'gamma':
        alpha, loc, beta = stats.gamma.fit(dNc_values_wo_na)
        calibrated_dSc[fit_mask] = stats.gamma.ppf(q=quantiles, a=alpha, loc=loc, scale=beta)
    elif transformation == 'quantile':
        calibrated_dSc[fit_mask] = np.quantile(dNc_values_wo_na, quantiles)
    else:
        raise ValueError('Unsupported transformation: {}'.format(transformation))
    is_nocalib_higher = (
        np.isfinite(dSc_values) &
        np.isfinite(calibrated_dSc) &
        (dSc_values > calibrated_dSc)
    )
    calibrated_dSc[is_nocalib_higher] = dSc_values[is_nocalib_higher]
    return calibrated_dSc, fit_mask, is_nocalib_higher


def _calibrate_dsc_matrix(dNc_matrix, dSc_matrix, transformation='quantile'):
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


def _collect_stat_masses(cb, prefix):
    stat_masses = dict()
    for stat_name in output_stat.ALL_OUTPUT_STATS:
        col_name = prefix + stat_name
        if col_name not in cb.columns:
            continue
        values = cb.loc[:, col_name].to_numpy(dtype=np.float64, copy=False)
        is_finite = np.isfinite(values)
        if not is_finite.any():
            continue
        stat_masses[stat_name] = float(np.clip(values[is_finite], a_min=0.0, a_max=None).sum(dtype=np.float64))
    return stat_masses


def _calc_distribution_entropy(prob):
    prob = np.asarray(prob, dtype=np.float64)
    positive = prob[prob > 0]
    if positive.shape[0] == 0:
        return 0.0
    return float(-(positive * np.log(positive)).sum(dtype=np.float64))


def _calc_zero_fraction(cb, prefix, output_stats, float_tol):
    cols = [prefix + sub for sub in output_stats if (prefix + sub) in cb.columns]
    if len(cols) == 0:
        return np.nan
    values = cb.loc[:, cols].to_numpy(dtype=np.float64, copy=False)
    is_finite = np.isfinite(values)
    if not is_finite.any():
        return np.nan
    return float(((values < float_tol) & is_finite).sum() / is_finite.sum())


def _collect_alpha_fit_pairs(cb, output_stats):
    obs_list = list()
    exp_list = list()
    for sub in output_stats:
        col_obs_n = 'OCN' + sub
        col_exp_n = 'ECN' + sub
        col_obs_s = 'OCS' + sub
        col_exp_s = 'ECS' + sub
        if all([c in cb.columns for c in [col_obs_n, col_exp_n]]):
            obs_list.append(cb.loc[:, col_obs_n].to_numpy(dtype=np.float64, copy=False))
            exp_list.append(cb.loc[:, col_exp_n].to_numpy(dtype=np.float64, copy=False))
        if all([c in cb.columns for c in [col_obs_s, col_exp_s]]):
            obs_list.append(cb.loc[:, col_obs_s].to_numpy(dtype=np.float64, copy=False))
            exp_list.append(cb.loc[:, col_exp_s].to_numpy(dtype=np.float64, copy=False))
    return obs_list, exp_list


def _get_pseudocount_context(cb, g, output_stats):
    config = pseudocount.validate_args(g).copy()
    alpha_fit_diag = dict()
    if config['pseudocount_alpha_auto'] and (config['pseudocount_mode'] != 'none'):
        obs_list, exp_list = _collect_alpha_fit_pairs(cb=cb, output_stats=output_stats)
        estimated_alpha, alpha_fit_diag = pseudocount.estimate_alpha_empirical_bayes(
            obs_list=obs_list,
            exp_list=exp_list,
            float_tol=g['float_tol'],
        )
        config['pseudocount_alpha'] = float(estimated_alpha)
        config['pseudocount_enabled'] = bool((config['pseudocount_alpha'] > 0) and (config['pseudocount_mode'] != 'none'))
        config['pseudocount_add_output_columns'] = bool(config['pseudocount_enabled'] or config['pseudocount_report'])
    K = len(output_stats)
    alpha_obs_N = np.zeros(shape=(K,), dtype=np.float64)
    alpha_exp_N = np.zeros(shape=(K,), dtype=np.float64)
    alpha_obs_S = np.zeros(shape=(K,), dtype=np.float64)
    alpha_exp_S = np.zeros(shape=(K,), dtype=np.float64)
    summary = {
        'pseudocount_mode': config['pseudocount_mode'],
        'pseudocount_alpha': float(config['pseudocount_alpha']),
        'pseudocount_alpha_source': 'auto' if config.get('pseudocount_alpha_auto', False) else 'fixed',
        'pseudocount_target': config['pseudocount_target'],
        'pseudocount_enabled': int(config['pseudocount_enabled']),
        'pseudocount_zero_prop_OCN': _calc_zero_fraction(cb, 'OCN', output_stats, g['float_tol']),
        'pseudocount_zero_prop_ECN': _calc_zero_fraction(cb, 'ECN', output_stats, g['float_tol']),
        'pseudocount_zero_prop_OCS': _calc_zero_fraction(cb, 'OCS', output_stats, g['float_tol']),
        'pseudocount_zero_prop_ECS': _calc_zero_fraction(cb, 'ECS', output_stats, g['float_tol']),
    }
    if len(alpha_fit_diag):
        summary['pseudocount_alpha_fit_pairs'] = int(alpha_fit_diag.get('num_pairs', 0))
        summary['pseudocount_alpha_fit_loglikelihood'] = float(alpha_fit_diag.get('fit_loglikelihood', np.nan))
    if K > 0:
        if config['pseudocount_mode'] == 'empirical':
            obs_N_masses = _collect_stat_masses(cb=cb, prefix='OCN')
            exp_N_masses = _collect_stat_masses(cb=cb, prefix='ECN')
            obs_S_masses = _collect_stat_masses(cb=cb, prefix='OCS')
            exp_S_masses = _collect_stat_masses(cb=cb, prefix='ECS')
            alpha_obs_N, p_atomic_obs_N = pseudocount.compute_empirical_stat_alphas(
                stat_masses=obs_N_masses,
                stats=output_stats,
                alpha=config['pseudocount_alpha'],
            )
            alpha_exp_N, p_atomic_exp_N = pseudocount.compute_empirical_stat_alphas(
                stat_masses=exp_N_masses,
                stats=output_stats,
                alpha=config['pseudocount_alpha'],
            )
            alpha_obs_S, p_atomic_obs_S = pseudocount.compute_empirical_stat_alphas(
                stat_masses=obs_S_masses,
                stats=output_stats,
                alpha=config['pseudocount_alpha'],
            )
            alpha_exp_S, p_atomic_exp_S = pseudocount.compute_empirical_stat_alphas(
                stat_masses=exp_S_masses,
                stats=output_stats,
                alpha=config['pseudocount_alpha'],
            )
            summary['pseudocount_pglobal_entropy_atomic_OCN'] = _calc_distribution_entropy(p_atomic_obs_N)
            summary['pseudocount_pglobal_entropy_atomic_ECN'] = _calc_distribution_entropy(p_atomic_exp_N)
            summary['pseudocount_pglobal_entropy_atomic_OCS'] = _calc_distribution_entropy(p_atomic_obs_S)
            summary['pseudocount_pglobal_entropy_atomic_ECS'] = _calc_distribution_entropy(p_atomic_exp_S)
        else:
            alpha_base = pseudocount.compute_alpha_vector(
                mode=config['pseudocount_mode'],
                alpha=config['pseudocount_alpha'],
                p_global=None,
                K=K,
            )
            alpha_obs_N = alpha_base.copy()
            alpha_exp_N = alpha_base.copy()
            alpha_obs_S = alpha_base.copy()
            alpha_exp_S = alpha_base.copy()
    if config['pseudocount_target'] == 'observed':
        alpha_exp_N[:] = 0
        alpha_exp_S[:] = 0
    elif config['pseudocount_target'] == 'expected':
        alpha_obs_N[:] = 0
        alpha_obs_S[:] = 0
    context = {
        'config': config,
        'output_stats': tuple(output_stats),
        'alpha_obs_N': alpha_obs_N,
        'alpha_exp_N': alpha_exp_N,
        'alpha_obs_S': alpha_obs_S,
        'alpha_exp_S': alpha_exp_S,
        'summary': summary,
    }
    return context


def _print_pseudocount_summary(context):
    summary = context.get('summary', dict())
    txt = 'Pseudocount settings: mode={}, alpha={}, alpha_source={}, target={}, enabled={}'
    print(
        txt.format(
            summary.get('pseudocount_mode', 'none'),
            summary.get('pseudocount_alpha', 0.0),
            summary.get('pseudocount_alpha_source', 'fixed'),
            summary.get('pseudocount_target', 'both'),
            summary.get('pseudocount_enabled', 0),
        ),
        flush=True,
    )
    if summary.get('pseudocount_alpha_source', 'fixed') == 'auto':
        txt = 'Pseudocount alpha auto-fit: pairs={:,}, logL={:.3f}'
        print(
            txt.format(
                int(summary.get('pseudocount_alpha_fit_pairs', 0)),
                float(summary.get('pseudocount_alpha_fit_loglikelihood', np.nan)),
            ),
            flush=True,
        )
    txt = 'Pseudocount sparsity (pre-smoothing): OCN_zero={:.3f}, ECN_zero={:.3f}, OCS_zero={:.3f}, ECS_zero={:.3f}'
    print(
        txt.format(
            float(summary.get('pseudocount_zero_prop_OCN', np.nan)),
            float(summary.get('pseudocount_zero_prop_ECN', np.nan)),
            float(summary.get('pseudocount_zero_prop_OCS', np.nan)),
            float(summary.get('pseudocount_zero_prop_ECS', np.nan)),
        ),
        flush=True,
    )
    if summary.get('pseudocount_mode', 'none') == 'empirical':
        txt = 'Pseudocount empirical atomic p_global entropy: OCN={:.3f}, ECN={:.3f}, OCS={:.3f}, ECS={:.3f}'
        print(
            txt.format(
                float(summary.get('pseudocount_pglobal_entropy_atomic_OCN', np.nan)),
                float(summary.get('pseudocount_pglobal_entropy_atomic_ECN', np.nan)),
                float(summary.get('pseudocount_pglobal_entropy_atomic_OCS', np.nan)),
                float(summary.get('pseudocount_pglobal_entropy_atomic_ECS', np.nan)),
            ),
            flush=True,
        )


def _write_pseudocount_summary_to_cb_stats(g, context):
    if 'df_cb_stats' not in g:
        return
    summary = context.get('summary', dict())
    for key, value in summary.items():
        g['df_cb_stats'].at[0, key] = value


def get_omega(cb, g):
    requested_output_stats = _resolve_requested_output_stats(g)
    context = g.get('_pseudocount_context', None)
    if (context is None) or (tuple(requested_output_stats) != context.get('output_stats', tuple())):
        context = _get_pseudocount_context(cb=cb, g=g, output_stats=requested_output_stats)
    config = context['config']
    if (not config['pseudocount_enabled']) and (not config['pseudocount_add_output_columns']):
        for sub in requested_output_stats:
            col_omega = 'omegaC'+sub
            col_N = 'OCN'+sub
            col_EN = 'ECN'+sub
            col_dNc = 'dNC'+sub
            col_S = 'OCS'+sub
            col_ES = 'ECS'+sub
            col_dSc = 'dSC'+sub
            if all([ col in cb.columns for col in [col_N,col_EN,col_S,col_ES] ]):
                obs_N = cb.loc[:, col_N].to_numpy(dtype=np.float64, copy=False)
                exp_N = cb.loc[:, col_EN].to_numpy(dtype=np.float64, copy=False)
                obs_S = cb.loc[:, col_S].to_numpy(dtype=np.float64, copy=False)
                exp_S = cb.loc[:, col_ES].to_numpy(dtype=np.float64, copy=False)
                dNc = _calc_raw_rate(obs=obs_N, exp=exp_N, float_tol=g['float_tol'])
                dSc = _calc_raw_rate(obs=obs_S, exp=exp_S, float_tol=g['float_tol'])
                omegaC = _calc_raw_omega(dNc=dNc, dSc=dSc, float_tol=g['float_tol'])
                cb.loc[:, col_dNc] = dNc
                cb.loc[:, col_dSc] = dSc
                cb.loc[:, col_omega] = omegaC
        return cb
    for i, sub in enumerate(requested_output_stats):
        col_omega = 'omegaC'+sub
        col_N = 'OCN'+sub
        col_EN = 'ECN'+sub
        col_dNc = 'dNC'+sub
        col_S = 'OCS'+sub
        col_ES = 'ECS'+sub
        col_dSc = 'dSC'+sub
        if all([ col in cb.columns for col in [col_N,col_EN,col_S,col_ES] ]):
            obs_N = cb.loc[:, col_N].to_numpy(dtype=np.float64, copy=False)
            exp_N = cb.loc[:, col_EN].to_numpy(dtype=np.float64, copy=False)
            obs_S = cb.loc[:, col_S].to_numpy(dtype=np.float64, copy=False)
            exp_S = cb.loc[:, col_ES].to_numpy(dtype=np.float64, copy=False)
            raw_dNc = _calc_raw_rate(obs=obs_N, exp=exp_N, float_tol=g['float_tol'])
            raw_dSc = _calc_raw_rate(obs=obs_S, exp=exp_S, float_tol=g['float_tol'])
            raw_omega = _calc_raw_omega(dNc=raw_dNc, dSc=raw_dSc, float_tol=g['float_tol'])
            if config['pseudocount_enabled']:
                sm_dNc = pseudocount.smooth_ratio(
                    O=obs_N,
                    E=exp_N,
                    alpha_obs=context['alpha_obs_N'][i],
                    alpha_exp=context['alpha_exp_N'][i],
                )
                sm_dSc = pseudocount.smooth_ratio(
                    O=obs_S,
                    E=exp_S,
                    alpha_obs=context['alpha_obs_S'][i],
                    alpha_exp=context['alpha_exp_S'][i],
                )
                sm_omega = np.asarray(
                    pseudocount.smooth_ratio(O=sm_dNc, E=sm_dSc, alpha_obs=0, alpha_exp=0),
                    dtype=np.float64,
                )
                # Enforce the same convention as raw ratios: 0/0 -> 0.
                is_zero_over_zero = (
                    np.isfinite(sm_dNc) &
                    np.isfinite(sm_dSc) &
                    (sm_dNc < g['float_tol']) &
                    (sm_dSc < g['float_tol'])
                )
                sm_omega[is_zero_over_zero] = 0
                log_sm_omega = pseudocount.smooth_log_ratio(O=sm_dNc, E=sm_dSc, alpha_obs=0, alpha_exp=0)
            else:
                sm_dNc = raw_dNc.copy()
                sm_dSc = raw_dSc.copy()
                sm_omega = raw_omega.copy()
                with np.errstate(divide='ignore', invalid='ignore'):
                    log_sm_omega = np.log(sm_omega)
            cb.loc[:, col_dNc] = sm_dNc
            cb.loc[:, col_dSc] = sm_dSc
            cb.loc[:, col_omega] = sm_omega
            if config['pseudocount_add_output_columns']:
                cb.loc[:, col_dNc + '_raw'] = raw_dNc
                cb.loc[:, col_dSc + '_raw'] = raw_dSc
                cb.loc[:, col_omega + '_raw'] = raw_omega
                cb.loc[:, col_dNc + '_smoothed'] = sm_dNc
                cb.loc[:, col_dSc + '_smoothed'] = sm_dSc
                cb.loc[:, col_omega + '_smoothed'] = sm_omega
                cb.loc[:, 'log' + col_omega + '_smoothed'] = log_sm_omega
    return cb

def get_CoD(cb, g):
    for NS in ['OCN','OCS']:
        col_spe = NS + 'any2spe'
        col_dif = NS + 'any2dif'
        col_cod = NS + 'CoD'
        if not all([col in cb.columns for col in [col_spe, col_dif]]):
            continue
        spe_values = cb.loc[:, col_spe].to_numpy(dtype=np.float64, copy=False)
        dif_values = cb.loc[:, col_dif].to_numpy(dtype=np.float64, copy=False)
        cb.loc[:, col_cod] = _calc_raw_rate(obs=spe_values, exp=dif_values, float_tol=g['float_tol'])
    return cb


def _calc_dif_count_matrix(any_count, spe_count, tol):
    any_count = np.asarray(any_count, dtype=np.float64)
    spe_count = np.asarray(spe_count, dtype=np.float64)
    out = any_count - spe_count
    is_negative = (out < (-float(tol)))
    is_almost_zero = (~is_negative) & (out < float(tol))
    out[is_negative] = np.nan
    out[is_almost_zero] = 0
    return out


def _compose_permutation_count_matrix(stat, mode_to_count, tol):
    if stat in ['any2any', 'spe2any', 'any2spe', 'spe2spe']:
        return np.asarray(mode_to_count[stat], dtype=np.float64)
    if stat == 'any2dif':
        return _calc_dif_count_matrix(mode_to_count['any2any'], mode_to_count['any2spe'], tol=tol)
    if stat == 'dif2any':
        return _calc_dif_count_matrix(mode_to_count['any2any'], mode_to_count['spe2any'], tol=tol)
    if stat == 'dif2spe':
        return _calc_dif_count_matrix(mode_to_count['any2spe'], mode_to_count['spe2spe'], tol=tol)
    if stat == 'spe2dif':
        return _calc_dif_count_matrix(mode_to_count['spe2any'], mode_to_count['spe2spe'], tol=tol)
    if stat == 'dif2dif':
        any2dif = _calc_dif_count_matrix(mode_to_count['any2any'], mode_to_count['any2spe'], tol=tol)
        spe2dif = _calc_dif_count_matrix(mode_to_count['spe2any'], mode_to_count['spe2spe'], tol=tol)
        return _calc_dif_count_matrix(any2dif, spe2dif, tol=tol)
    raise ValueError('Unsupported output stat for permutation omega p-value: {}'.format(stat))


def _resolve_omega_pvalue_null_model(g):
    model = 'poisson'
    if g is not None:
        model = str(g.get('omega_pvalue_null_model', 'poisson')).strip().lower()
    if model not in ['hypergeom', 'poisson', 'poisson_full', 'nbinom']:
        raise ValueError('omega_pvalue_null_model should be one of hypergeom, poisson, poisson_full, nbinom.')
    return model


def _calc_poisson_count_matrix(
    mode,
    cb_ids,
    sub_sg,
    sub_bg,
    niter,
    obs_col,
    num_gad_combinat,
    list_igad,
    g,
    static_sub_sites,
):
    cb_ids = np.asarray(cb_ids, dtype=np.int64)
    niter = int(niter)
    if cb_ids.ndim != 2:
        raise ValueError('cb_ids should be a 2D array.')
    if niter <= 0:
        raise ValueError('niter should be a positive integer.')
    out = np.zeros(shape=(cb_ids.shape[0], niter), dtype=np.float64)
    if cb_ids.shape[0] == 0:
        return out
    for i, sg, a, d in list_igad:
        if a == d:
            continue
        if static_sub_sites is None:
            sub_sites = _resolve_sub_sites(g=g, sub_sg=sub_sg, mode=mode, sg=sg, a=a, d=d, obs_col=obs_col)
        else:
            sub_sites = static_sub_sites
        sub_branches = substitution.get_sub_branches(sub_bg, mode, sg, a, d)
        mean_count = _calc_wallenius_expected_overlap(
            cb_ids=cb_ids,
            sub_sites=sub_sites,
            sub_branches=sub_branches,
            g=g,
            float_type=np.float64,
        )
        mean_count = np.asarray(mean_count, dtype=np.float64).reshape(-1)
        if mean_count.shape[0] != cb_ids.shape[0]:
            txt = 'mean_count rows ({}) and cb_ids rows ({}) should match.'
            raise ValueError(txt.format(mean_count.shape[0], cb_ids.shape[0]))
        np.clip(mean_count, a_min=0.0, a_max=None, out=mean_count)
        if (mean_count > 0).sum() == 0:
            continue
        pm_start = time.time()
        out += np.random.poisson(
            lam=mean_count[:, None],
            size=(mean_count.shape[0], niter),
        ).astype(np.float64, copy=False)
        txt = '{} (poisson): {}/{} matrix_group/ancestral_state/derived_state combinations. '
        txt += 'Time elapsed for {:,} permutation: {:,} [sec]'
        print(txt.format(obs_col, i + 1, num_gad_combinat, niter, int(time.time() - pm_start)), flush=True)
    return out


def _estimate_nbinom_alpha(obs_count, mean_count, float_tol, max_alpha=10.0):
    obs_count = np.asarray(obs_count, dtype=np.float64).reshape(-1)
    mean_count = np.asarray(mean_count, dtype=np.float64).reshape(-1)
    if obs_count.shape[0] != mean_count.shape[0]:
        txt = 'obs_count rows ({}) and mean_count rows ({}) should match.'
        raise ValueError(txt.format(obs_count.shape[0], mean_count.shape[0]))
    valid = np.isfinite(obs_count) & np.isfinite(mean_count) & (mean_count > float(float_tol))
    if int(valid.sum()) < 20:
        return 0.0, 'auto_insufficient_rows'
    mu = np.clip(mean_count[valid], a_min=float(float_tol), a_max=None)
    resid2 = (obs_count[valid] - mu) ** 2
    alpha_i = (resid2 - mu) / (mu ** 2)
    alpha_i = alpha_i[np.isfinite(alpha_i)]
    if alpha_i.shape[0] == 0:
        return 0.0, 'auto_no_finite_estimates'
    alpha_i = np.clip(alpha_i, a_min=0.0, a_max=float(max_alpha))
    alpha = float(np.median(alpha_i))
    if not np.isfinite(alpha):
        return 0.0, 'auto_non_finite'
    return alpha, 'auto'


def _resolve_nbinom_alpha(g, obs_count, mean_count):
    token = 'auto'
    if g is not None:
        token = g.get('omega_pvalue_nbinom_alpha', 'auto')
    if isinstance(token, str) and (token.strip().lower() == 'auto'):
        alpha, source = _estimate_nbinom_alpha(
            obs_count=obs_count,
            mean_count=mean_count,
            float_tol=float(g.get('float_tol', 1e-12)) if g is not None else 1e-12,
        )
        return float(alpha), source
    alpha = float(token)
    if (not np.isfinite(alpha)) or (alpha < 0):
        raise ValueError('omega_pvalue_nbinom_alpha should be a finite value >= 0 or "auto".')
    return float(alpha), 'fixed'


def _sample_nbinom_count_matrix(mean_count, niter, alpha):
    mean_count = np.asarray(mean_count, dtype=np.float64).reshape(-1)
    if mean_count.ndim != 1:
        raise ValueError('mean_count should be a 1D array.')
    if int(niter) <= 0:
        raise ValueError('niter should be a positive integer.')
    mean_count = np.clip(mean_count, a_min=0.0, a_max=None)
    if mean_count.shape[0] == 0:
        return np.zeros(shape=(0, int(niter)), dtype=np.float64)
    if alpha <= 0:
        return np.random.poisson(
            lam=mean_count[:, None],
            size=(mean_count.shape[0], int(niter)),
        ).astype(np.float64, copy=False)
    gamma_shape = 1.0 / float(alpha)
    gamma_scale = float(alpha) * mean_count[:, None]
    lam = np.random.gamma(
        shape=gamma_shape,
        scale=gamma_scale,
        size=(mean_count.shape[0], int(niter)),
    )
    return np.random.poisson(lam=lam).astype(np.float64, copy=False)


def _calc_nbinom_count_matrix(
    mode,
    cb_ids,
    sub_sg,
    sub_bg,
    niter,
    obs_col,
    num_gad_combinat,
    list_igad,
    g,
    static_sub_sites,
    obs_count,
):
    cb_ids = np.asarray(cb_ids, dtype=np.int64)
    niter = int(niter)
    if cb_ids.ndim != 2:
        raise ValueError('cb_ids should be a 2D array.')
    if niter <= 0:
        raise ValueError('niter should be a positive integer.')
    out = np.zeros(shape=(cb_ids.shape[0], niter), dtype=np.float64)
    if cb_ids.shape[0] == 0:
        return out
    mean_total = np.zeros(shape=(cb_ids.shape[0],), dtype=np.float64)
    for i, sg, a, d in list_igad:
        if a == d:
            continue
        if static_sub_sites is None:
            sub_sites = _resolve_sub_sites(g=g, sub_sg=sub_sg, mode=mode, sg=sg, a=a, d=d, obs_col=obs_col)
        else:
            sub_sites = static_sub_sites
        sub_branches = substitution.get_sub_branches(sub_bg, mode, sg, a, d)
        mean_count = _calc_wallenius_expected_overlap(
            cb_ids=cb_ids,
            sub_sites=sub_sites,
            sub_branches=sub_branches,
            g=g,
            float_type=np.float64,
        )
        mean_count = np.asarray(mean_count, dtype=np.float64).reshape(-1)
        if mean_count.shape[0] != cb_ids.shape[0]:
            txt = 'mean_count rows ({}) and cb_ids rows ({}) should match.'
            raise ValueError(txt.format(mean_count.shape[0], cb_ids.shape[0]))
        np.clip(mean_count, a_min=0.0, a_max=None, out=mean_count)
        mean_total += mean_count
    if (mean_total > 0).sum() == 0:
        return out
    if obs_count is None:
        obs_count = mean_total
    alpha, alpha_source = _resolve_nbinom_alpha(
        g=g,
        obs_count=obs_count,
        mean_count=mean_total,
    )
    pm_start = time.time()
    out = _sample_nbinom_count_matrix(
        mean_count=mean_total,
        niter=niter,
        alpha=alpha,
    )
    txt = '{} (nbinom): categories={}, alpha={:.6g} (source={}), niter={}, elapsed={} sec'
    print(
        txt.format(obs_col, int(num_gad_combinat), float(alpha), alpha_source, int(niter), int(time.time() - pm_start)),
        flush=True,
    )
    return out


def _get_mode_branch_site_mass(sub_tensor, mode, sg, a, d):
    if isinstance(sub_tensor, substitution_sparse.SparseSubstitutionTensor):
        if mode == 'spe2spe':
            mat = sub_tensor.blocks.get((int(sg), int(a), int(d)), None)
            if mat is None:
                return np.zeros(shape=(sub_tensor.num_branch, sub_tensor.num_site), dtype=np.float64)
            return np.asarray(mat.toarray(), dtype=np.float64)
        if mode == 'spe2any':
            return np.asarray(sub_tensor.project_spe2any(int(sg), int(a)).toarray(), dtype=np.float64)
        if mode == 'any2spe':
            return np.asarray(sub_tensor.project_any2spe(int(sg), int(d)).toarray(), dtype=np.float64)
        if mode == 'any2any':
            return np.asarray(sub_tensor.project_any2any(int(sg)).toarray(), dtype=np.float64)
        raise ValueError('Unsupported mode: {}'.format(mode))
    sub_tensor = np.asarray(sub_tensor)
    if mode == 'spe2spe':
        return np.asarray(sub_tensor[:, :, int(sg), int(a), int(d)], dtype=np.float64)
    if mode == 'spe2any':
        return np.asarray(sub_tensor[:, :, int(sg), int(a), :].sum(axis=2), dtype=np.float64)
    if mode == 'any2spe':
        return np.asarray(sub_tensor[:, :, int(sg), :, int(d)].sum(axis=2), dtype=np.float64)
    if mode == 'any2any':
        return np.asarray(sub_tensor[:, :, int(sg), :, :].sum(axis=(2, 3)), dtype=np.float64)
    raise ValueError('Unsupported mode: {}'.format(mode))


def _calc_poisson_full_count_matrix(
    mode,
    cb_ids,
    sub_tensor,
    niter,
    obs_col,
    num_gad_combinat,
    list_igad,
    g,
):
    cb_ids = np.asarray(cb_ids, dtype=np.int64)
    niter = int(niter)
    if cb_ids.ndim != 2:
        raise ValueError('cb_ids should be a 2D array.')
    if niter <= 0:
        raise ValueError('niter should be a positive integer.')
    out = np.zeros(shape=(cb_ids.shape[0], niter), dtype=np.float64)
    if cb_ids.shape[0] == 0:
        return out
    for i, sg, a, d in list_igad:
        if a == d:
            continue
        sub_site_mass = _get_mode_branch_site_mass(
            sub_tensor=sub_tensor,
            mode=mode,
            sg=sg,
            a=a,
            d=d,
        )
        np.clip(sub_site_mass, a_min=0.0, a_max=None, out=sub_site_mass)
        sub_branches = sub_site_mass.sum(axis=1, dtype=np.float64)
        sub_sites = np.zeros_like(sub_site_mass, dtype=np.float64)
        nonzero_branch = (sub_branches > 0)
        if nonzero_branch.any():
            sub_sites[nonzero_branch, :] = (
                sub_site_mass[nonzero_branch, :] /
                sub_branches[nonzero_branch, None]
            )
        mean_count = _calc_wallenius_expected_overlap(
            cb_ids=cb_ids,
            sub_sites=sub_sites,
            sub_branches=sub_branches,
            g=g,
            float_type=np.float64,
        )
        mean_count = np.asarray(mean_count, dtype=np.float64).reshape(-1)
        if mean_count.shape[0] != cb_ids.shape[0]:
            txt = 'mean_count rows ({}) and cb_ids rows ({}) should match.'
            raise ValueError(txt.format(mean_count.shape[0], cb_ids.shape[0]))
        np.clip(mean_count, a_min=0.0, a_max=None, out=mean_count)
        if (mean_count > 0).sum() == 0:
            continue
        pm_start = time.time()
        out += np.random.poisson(
            lam=mean_count[:, None],
            size=(mean_count.shape[0], niter),
        ).astype(np.float64, copy=False)
        txt = '{} (poisson_full): {}/{} matrix_group/ancestral_state/derived_state combinations. '
        txt += 'Time elapsed for {:,} permutation: {:,} [sec]'
        print(txt.format(obs_col, i + 1, num_gad_combinat, niter, int(time.time() - pm_start)), flush=True)
    return out


def _get_mode_permutation_count_matrix(cb_ids, sub_tensor, mode, SN, niter, g, obs_count=None):
    sub_bg, sub_sg, list_igad, obs_col, num_gad_combinat = _prepare_substitution_quantile_components(
        sub_tensor=sub_tensor,
        mode=mode,
        SN=SN,
        g=g,
    )
    static_sub_sites = _get_static_sub_sites_if_available(g=g, sub_sg=sub_sg, mode=mode, obs_col=obs_col)
    null_model = _resolve_omega_pvalue_null_model(g=g)
    txt = 'pomegaC {}{}: {} count matrix (rows={:,}, niter={:,}, categories={:,})'
    model_label = 'randomization'
    if null_model == 'poisson':
        model_label = 'poisson-rate'
    if null_model == 'poisson_full':
        model_label = 'poisson-full-rate'
    if null_model == 'nbinom':
        model_label = 'nbinom-rate'
    print(txt.format(SN, mode, model_label, cb_ids.shape[0], int(niter), int(num_gad_combinat)), flush=True)
    if null_model == 'hypergeom':
        return _calc_quantile_count_matrix(
            mode=mode,
            cb_ids=cb_ids,
            sub_sg=sub_sg,
            sub_bg=sub_bg,
            quantile_niter=int(niter),
            obs_col=obs_col,
            num_gad_combinat=num_gad_combinat,
            list_igad=list_igad,
            g=g,
            static_sub_sites=static_sub_sites,
        )
    if null_model == 'poisson':
        return _calc_poisson_count_matrix(
            mode=mode,
            cb_ids=cb_ids,
            sub_sg=sub_sg,
            sub_bg=sub_bg,
            niter=int(niter),
            obs_col=obs_col,
            num_gad_combinat=num_gad_combinat,
            list_igad=list_igad,
            g=g,
            static_sub_sites=static_sub_sites,
        )
    if null_model == 'poisson_full':
        return _calc_poisson_full_count_matrix(
            mode=mode,
            cb_ids=cb_ids,
            sub_tensor=sub_tensor,
            niter=int(niter),
            obs_col=obs_col,
            num_gad_combinat=num_gad_combinat,
            list_igad=list_igad,
            g=g,
        )
    if null_model == 'nbinom':
        return _calc_nbinom_count_matrix(
            mode=mode,
            cb_ids=cb_ids,
            sub_sg=sub_sg,
            sub_bg=sub_bg,
            niter=int(niter),
            obs_col=obs_col,
            num_gad_combinat=num_gad_combinat,
            list_igad=list_igad,
            g=g,
            static_sub_sites=static_sub_sites,
            obs_count=obs_count,
        )
    raise ValueError('Unsupported omega_pvalue_null_model: {}'.format(null_model))


def _calc_permutation_omega_matrix(
    exp_N,
    exp_S,
    perm_count_N,
    perm_count_S,
    float_tol,
    calibrate_dsc_transformation=None,
):
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


def _calc_omega_empirical_upper_tail_pvalues_from_perm(obs_omega, exp_S, perm_omega):
    obs_omega = np.asarray(obs_omega, dtype=np.float64).reshape(-1)
    exp_S = np.asarray(exp_S, dtype=np.float64).reshape(-1)
    perm_omega = np.asarray(perm_omega, dtype=np.float64)
    if perm_omega.ndim != 2:
        raise ValueError('perm_omega should be a 2D array.')
    if perm_omega.shape[0] != obs_omega.shape[0]:
        txt = 'Permutation rows ({}) and observed rows ({}) should match.'
        raise ValueError(txt.format(perm_omega.shape[0], obs_omega.shape[0]))
    if exp_S.shape[0] != obs_omega.shape[0]:
        txt = 'exp_S rows ({}) and observed rows ({}) should match.'
        raise ValueError(txt.format(exp_S.shape[0], obs_omega.shape[0]))
    valid_perm = ~np.isnan(perm_omega)
    ge_ranks = (valid_perm & (perm_omega >= obs_omega[:, None])).sum(axis=1, dtype=np.int64)
    valid_niter = valid_perm.sum(axis=1, dtype=np.int64)
    pvalue = np.full(shape=obs_omega.shape, fill_value=np.nan, dtype=np.float64)
    valid_rows = (~np.isnan(obs_omega)) & (valid_niter > 0)
    valid_rows &= np.isfinite(exp_S)
    pvalue[valid_rows] = (ge_ranks[valid_rows] + 1.0) / (valid_niter[valid_rows] + 1.0)
    return pvalue


def _calc_omega_empirical_upper_tail_pvalues(
    obs_omega,
    exp_N,
    exp_S,
    perm_count_N,
    perm_count_S,
    float_tol,
    calibrate_dsc_transformation=None,
):
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
    return _calc_omega_empirical_upper_tail_pvalues_from_perm(
        obs_omega=obs_omega,
        exp_S=exp_S,
        perm_omega=perm_omega,
    )


def _calc_bh_fdr_qvalues(pvalues):
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


def _resolve_omega_pvalue_dsc_calibration_transformation(cb, sub, g):
    col_dSc = 'dSC' + sub
    col_noncalibrated_dSc = col_dSc + '_nocalib'
    col_omega = 'omegaC' + sub
    col_noncalibrated_omega = col_omega + '_nocalib'
    has_calibrated_columns = all(
        [col in cb.columns for col in [col_dSc, col_noncalibrated_dSc, col_omega, col_noncalibrated_omega]]
    )
    if not has_calibrated_columns:
        return None
    transformation = str(g.get('calibrate_longtail_transformation', 'quantile')).strip().lower()
    if transformation == '':
        transformation = 'quantile'
    if transformation not in ['quantile', 'gamma']:
        raise ValueError('Unsupported calibrate_longtail_transformation: {}'.format(transformation))
    return transformation


def add_omega_empirical_pvalues(cb, ON_tensor, OS_tensor, g):
    if not bool(g.get('calc_omega_pvalue', False)):
        return cb
    if str(g.get('omegaC_method', '')).strip().lower() != 'modelfree':
        sys.stderr.write('Skipping --calc_omega_pvalue because --omegaC_method is not "modelfree".\n')
        return cb
    niter = int(g.get('omega_pvalue_niter', 1000))
    if niter <= 0:
        raise ValueError('omega_pvalue_niter should be a positive integer.')
    null_model = _resolve_omega_pvalue_null_model(g=g)
    txt = 'omega_C empirical p-value null model: {}'
    print(txt.format(null_model), flush=True)
    cb_ids = _get_cb_ids(cb)
    output_stats = _resolve_requested_output_stats(g)
    for sub in output_stats:
        col_omega = 'omegaC' + sub
        col_exp_N = 'ECN' + sub
        col_exp_S = 'ECS' + sub
        if not all([col in cb.columns for col in [col_omega, col_exp_N, col_exp_S]]):
            continue
        mode_to_count_N = dict()
        mode_to_count_S = dict()
        required_modes = output_stat.get_required_base_stats([sub])
        for mode in required_modes:
            obs_col_N = 'OCN' + mode
            obs_col_S = 'OCS' + mode
            obs_count_N = None
            obs_count_S = None
            if obs_col_N in cb.columns:
                obs_count_N = cb.loc[:, obs_col_N].to_numpy(dtype=np.float64, copy=False)
            if obs_col_S in cb.columns:
                obs_count_S = cb.loc[:, obs_col_S].to_numpy(dtype=np.float64, copy=False)
            mode_to_count_N[mode] = _get_mode_permutation_count_matrix(
                cb_ids=cb_ids,
                sub_tensor=ON_tensor,
                mode=mode,
                SN='N',
                niter=niter,
                g=g,
                obs_count=obs_count_N,
            )
            mode_to_count_S[mode] = _get_mode_permutation_count_matrix(
                cb_ids=cb_ids,
                sub_tensor=OS_tensor,
                mode=mode,
                SN='S',
                niter=niter,
                g=g,
                obs_count=obs_count_S,
            )
        perm_count_N = _compose_permutation_count_matrix(
            stat=sub,
            mode_to_count=mode_to_count_N,
            tol=g['float_tol'],
        )
        perm_count_S = _compose_permutation_count_matrix(
            stat=sub,
            mode_to_count=mode_to_count_S,
            tol=g['float_tol'],
        )
        calibrate_dsc_transformation = _resolve_omega_pvalue_dsc_calibration_transformation(cb=cb, sub=sub, g=g)
        if calibrate_dsc_transformation is not None:
            txt = 'pomegaC {}: applying dSc {} calibration to permutation omega null'
            print(txt.format(sub, calibrate_dsc_transformation), flush=True)
        obs_omega_values = cb.loc[:, col_omega].to_numpy(dtype=np.float64, copy=False)
        exp_N_values = cb.loc[:, col_exp_N].to_numpy(dtype=np.float64, copy=False)
        exp_S_values = cb.loc[:, col_exp_S].to_numpy(dtype=np.float64, copy=False)
        pvalue = _calc_omega_empirical_upper_tail_pvalues(
            obs_omega=obs_omega_values,
            exp_N=exp_N_values,
            exp_S=exp_S_values,
            perm_count_N=perm_count_N,
            perm_count_S=perm_count_S,
            float_tol=g['float_tol'],
            calibrate_dsc_transformation=calibrate_dsc_transformation,
        )
        col_p = 'pomegaC' + sub
        cb.loc[:, col_p] = pvalue
        col_q = 'qomegaC' + sub
        qvalue = _calc_bh_fdr_qvalues(pvalue)
        cb.loc[:, col_q] = qvalue
        finite = np.isfinite(pvalue)
        if finite.any():
            txt = 'Arity = {:,}, cb: median {} = {:.4f} ({:,}/{:,} finite)'
            print(
                txt.format(
                    cb_ids.shape[1],
                    col_p,
                    float(np.nanmedian(pvalue)),
                    int(finite.sum()),
                    int(pvalue.shape[0]),
                ),
                flush=True,
            )
        finite_q = np.isfinite(qvalue)
        if finite_q.any():
            txt = 'Arity = {:,}, cb: median {} = {:.4f} ({:,}/{:,} finite)'
            print(
                txt.format(
                    cb_ids.shape[1],
                    col_q,
                    float(np.nanmedian(qvalue)),
                    int(finite_q.sum()),
                    int(qvalue.shape[0]),
                ),
                flush=True,
            )
    return cb


def print_cb_stats(cb, prefix, output_stats):
    arity = cb.columns.str.startswith('branch_id_').sum()
    hd = 'Arity = {:,}, {}:'.format(arity, prefix)
    for sub in output_stats:
        col_omega = 'omegaC'+sub
        if not col_omega in cb.columns:
            continue
        median_value = cb.loc[:,col_omega].median()
        txt = '{} median {} (non-corrected for dNc vs dSc distribution ranges): {:.3f}'
        print(txt.format(hd, col_omega, median_value), flush=True)

def calc_omega(cb, OS_tensor, ON_tensor, g):
    cb = get_E(cb, g, ON_tensor, OS_tensor)
    output_stats = _resolve_requested_output_stats(g)
    context = _get_pseudocount_context(cb=cb, g=g, output_stats=output_stats)
    g['_pseudocount_context'] = context
    if context['config']['pseudocount_add_output_columns']:
        _print_pseudocount_summary(context=context)
        _write_pseudocount_summary_to_cb_stats(g=g, context=context)
    cb = get_omega(cb, g)
    cb = get_CoD(cb, g)
    cb = add_omega_empirical_pvalues(cb=cb, ON_tensor=ON_tensor, OS_tensor=OS_tensor, g=g)
    print_cb_stats(cb=cb, prefix='cb', output_stats=output_stats)
    return(cb, g)

def calibrate_dsc(cb, transformation='quantile', output_stats=None, float_tol=1e-12):
    prefix='cb'
    arity = cb.columns.str.startswith('branch_id_').sum()
    hd = 'Arity = {:,}, {}:'.format(arity, prefix)
    if output_stats is None:
        output_stats = list(output_stat.ALL_OUTPUT_STATS)
    for sub in output_stats:
        col_dNc = 'dNC'+sub
        col_dSc = 'dSC'+sub
        col_omega = 'omegaC'+sub
        col_pvalue = 'pomegaC'+sub
        col_qvalue = 'qomegaC'+sub
        col_noncalibrated_dSc = 'dSC'+sub+'_nocalib'
        col_noncalibrated_omega = 'omegaC'+sub+'_nocalib'
        if not all([col in cb.columns for col in [col_dNc, col_dSc, col_omega]]):
            continue
        dNc_values = cb.loc[:, col_dNc].to_numpy(dtype=np.float64, copy=False)
        uncorrected_dSc_values = cb.loc[:, col_dSc].to_numpy(dtype=np.float64, copy=False)
        fit_mask = (np.isfinite(dNc_values) & np.isfinite(uncorrected_dSc_values))
        if (~fit_mask).all():
            txt = 'dSc calibration could not be applied: {} (no finite dNc/dSc pairs)\n'
            sys.stderr.write(txt.format(sub))
            continue
        excluded_count = int((~fit_mask).sum())
        if (excluded_count > 0):
            txt = 'dSc calibration could not be applied to {:,}/{:,} branch combinations for {}\n'
            sys.stderr.write(txt.format(excluded_count, cb.shape[0], sub))
        rename_map = {col_dSc: col_noncalibrated_dSc, col_omega: col_noncalibrated_omega}
        if col_pvalue in cb.columns:
            rename_map[col_pvalue] = col_pvalue + '_nocalib'
        if col_qvalue in cb.columns:
            rename_map[col_qvalue] = col_qvalue + '_nocalib'
        cb = cb.rename(columns=rename_map)
        calibrated_dSc_values, _, is_nocalib_higher = _calibrate_dsc_vector(
            dNc_values=dNc_values,
            dSc_values=uncorrected_dSc_values,
            transformation=transformation,
        )
        cb.loc[:, col_dSc] = calibrated_dSc_values
        calibrated_dNc = cb.loc[:,col_dNc].to_numpy(dtype=np.float64, copy=False)
        calibrated_dSc = cb.loc[:, col_dSc].to_numpy(dtype=np.float64, copy=False)
        calibrated_omega = _calc_raw_omega(
            dNc=calibrated_dNc,
            dSc=calibrated_dSc,
            float_tol=float(float_tol),
        )
        noncalibrated_omega_values = cb.loc[:, col_noncalibrated_omega].to_numpy(dtype=np.float64, copy=False)
        calibrated_omega[~fit_mask] = noncalibrated_omega_values[~fit_mask]
        cb.loc[:, col_omega] = calibrated_omega
        median_value = cb.loc[:,col_omega].median()
        corrected_count = int(fit_mask.sum() - is_nocalib_higher[fit_mask].sum())
        txt = '{} median {} ({:,}/{:,} branch combinations were corrected for dNc vs dSc distribution ranges): {:.3f}'
        print(txt.format(hd, col_omega, corrected_count, cb.shape[0], median_value), flush=True)
    return cb
