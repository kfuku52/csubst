import math
import numpy as np

from csubst import output_stat

try:
    from scipy.special import gammaln as _gammaln
except Exception:  # pragma: no cover
    _gammaln = np.vectorize(math.lgamma, otypes=[np.float64])


_PSEUDOCOUNT_MODES = ("none", "symmetric", "empirical")
_PSEUDOCOUNT_TARGETS = ("observed", "expected", "both")


def _get_arg(args, key, default):
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)


def _has_arg(args, key):
    if isinstance(args, dict):
        return key in args
    return hasattr(args, key)


def validate_args(args):
    raw_alpha = _get_arg(args, "pseudocount_alpha", 0.0)
    prevalidated_alpha_auto = bool(_get_arg(args, "pseudocount_alpha_auto", False))
    alpha_auto = False
    if isinstance(raw_alpha, str) and (raw_alpha.strip().lower() == "auto"):
        alpha = 0.0
        alpha_auto = True
    else:
        try:
            alpha = float(raw_alpha)
        except (TypeError, ValueError):
            raise ValueError("--pseudocount_alpha should be a finite number or 'auto'.")
        if not np.isfinite(alpha):
            raise ValueError("--pseudocount_alpha should be a finite number or 'auto'.")
        if alpha < 0:
            raise ValueError("--pseudocount_alpha should be >= 0.")
        # validate_args() may be called on a prevalidated mapping where
        # --pseudocount_alpha auto was already normalized to alpha=0.0.
        if prevalidated_alpha_auto and (alpha == 0.0):
            alpha_auto = True

    mode = str(_get_arg(args, "pseudocount_mode", "none")).strip().lower()
    if mode not in _PSEUDOCOUNT_MODES:
        txt = "--pseudocount_mode should be one of {}."
        raise ValueError(txt.format(", ".join(_PSEUDOCOUNT_MODES)))

    target = str(_get_arg(args, "pseudocount_target", "both")).strip().lower()
    if target not in _PSEUDOCOUNT_TARGETS:
        txt = "--pseudocount_target should be one of {}."
        raise ValueError(txt.format(", ".join(_PSEUDOCOUNT_TARGETS)))

    if _has_arg(args, "pseudocount_strength"):
        raise ValueError("--pseudocount_strength was removed.")

    report = bool(_get_arg(args, "pseudocount_report", False))
    enabled = ((alpha > 0) or alpha_auto) and (mode != "none")

    return {
        "pseudocount_alpha": alpha,
        "pseudocount_alpha_auto": alpha_auto,
        "pseudocount_mode": mode,
        "pseudocount_target": target,
        "pseudocount_report": report,
        "pseudocount_enabled": enabled,
        "pseudocount_add_output_columns": (enabled or report),
    }


def compute_alpha_vector(mode, alpha, p_global=None, K=None):
    mode = str(mode).strip().lower()
    alpha = float(alpha)
    if (not np.isfinite(alpha)) or (alpha < 0):
        raise ValueError("alpha should be finite and >= 0.")
    if mode not in _PSEUDOCOUNT_MODES:
        raise ValueError("Unsupported pseudocount mode: {}".format(mode))

    if K is None:
        if p_global is None:
            raise ValueError("K is required when p_global is not provided.")
        K = int(np.asarray(p_global).size)
    K = int(K)
    if K < 0:
        raise ValueError("K should be >= 0.")
    if K == 0:
        return np.zeros(shape=(0,), dtype=np.float64)
    if (mode == "none") or (alpha == 0):
        return np.zeros(shape=(K,), dtype=np.float64)
    if mode == "symmetric":
        return np.full(shape=(K,), fill_value=alpha, dtype=np.float64)

    if p_global is None:
        normalized = np.full(shape=(K,), fill_value=(1.0 / K), dtype=np.float64)
    else:
        normalized = np.asarray(p_global, dtype=np.float64).reshape(-1)
        if normalized.shape[0] != K:
            txt = "p_global length ({}) and K ({}) did not match."
            raise ValueError(txt.format(normalized.shape[0], K))
        if (~np.isfinite(normalized)).any():
            raise ValueError("p_global should contain only finite numbers.")
        if (normalized < 0).any():
            raise ValueError("p_global should contain only non-negative numbers.")
        total = normalized.sum(dtype=np.float64)
        if total > 0:
            normalized = normalized / total
        else:
            normalized = np.full(shape=(K,), fill_value=(1.0 / K), dtype=np.float64)
    # Keep empirical priors strictly positive to avoid unresolved 0/0 categories.
    normalized = np.clip(normalized, a_min=np.finfo(np.float64).tiny, a_max=None)
    normalized = normalized / normalized.sum(dtype=np.float64)

    return alpha * normalized


def smooth_counts(counts, alpha):
    return np.asarray(counts, dtype=np.float64) + np.asarray(alpha, dtype=np.float64)


def smooth_probs(n_k, alpha_k):
    n_k = np.asarray(n_k, dtype=np.float64)
    alpha_k = np.asarray(alpha_k, dtype=np.float64)
    if n_k.shape != alpha_k.shape:
        txt = "n_k shape ({}) and alpha_k shape ({}) should match."
        raise ValueError(txt.format(n_k.shape, alpha_k.shape))
    num = n_k + alpha_k
    denom = num.sum(dtype=np.float64)
    if denom > 0:
        return num / denom
    if num.size == 0:
        return np.zeros(shape=num.shape, dtype=np.float64)
    return np.full(shape=num.shape, fill_value=(1.0 / num.size), dtype=np.float64)


def _safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    out_shape = np.broadcast(numerator, denominator).shape
    out = np.full(shape=out_shape, fill_value=np.nan, dtype=np.float64)
    numerator_b = np.broadcast_to(numerator, out_shape)
    denominator_b = np.broadcast_to(denominator, out_shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(
            numerator_b,
            denominator_b,
            out=out,
        )
    zero_over_zero = (
        np.isfinite(numerator_b) &
        np.isfinite(denominator_b) &
        (numerator_b == 0) &
        (denominator_b == 0)
    )
    out = np.where(zero_over_zero, 0.0, out)
    if out_shape == ():
        return float(out)
    return out


def smooth_ratio(O, E, alpha_obs, alpha_exp):
    numerator = smooth_counts(O, alpha_obs)
    denominator = smooth_counts(E, alpha_exp)
    return _safe_divide(numerator=numerator, denominator=denominator)


def smooth_log_ratio(O, E, alpha_obs, alpha_exp):
    numerator = smooth_counts(O, alpha_obs)
    denominator = smooth_counts(E, alpha_exp)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(numerator) - np.log(denominator)
    if np.asarray(out).shape == ():
        return float(out)
    return out


def _flatten_valid_pairs(obs, exp):
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    exp = np.asarray(exp, dtype=np.float64).reshape(-1)
    if obs.shape != exp.shape:
        txt = "obs shape ({}) and exp shape ({}) should match."
        raise ValueError(txt.format(obs.shape, exp.shape))
    is_valid = np.isfinite(obs) & np.isfinite(exp) & (obs >= 0) & (exp >= 0)
    return obs[is_valid], exp[is_valid]


def _gamma_poisson_log_marginal(alpha, obs, exp, eps):
    alpha = float(alpha)
    if (not np.isfinite(alpha)) or (alpha <= 0):
        return -np.inf
    obs = np.asarray(obs, dtype=np.float64)
    exp = np.asarray(exp, dtype=np.float64)
    exp_safe = np.maximum(exp, float(eps))
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = _gammaln(obs + alpha) - _gammaln(alpha) - _gammaln(obs + 1.0)
        ll += alpha * (np.log(alpha) - np.log(alpha + exp_safe))
        ll += obs * (np.log(exp_safe) - np.log(alpha + exp_safe))
    ll_sum = float(np.nansum(ll, dtype=np.float64))
    if not np.isfinite(ll_sum):
        return -np.inf
    return ll_sum


def estimate_alpha_empirical_bayes(obs_list, exp_list, float_tol=1e-12, alpha_grid=None, max_samples=200000):
    if alpha_grid is None:
        alpha_grid = np.logspace(-4, 2, 121, dtype=np.float64)
    alpha_grid = np.asarray(alpha_grid, dtype=np.float64).reshape(-1)
    alpha_grid = alpha_grid[np.isfinite(alpha_grid) & (alpha_grid > 0)]
    if alpha_grid.shape[0] == 0:
        raise ValueError("alpha_grid should contain at least one positive finite value.")

    if len(obs_list) != len(exp_list):
        txt = "obs_list length ({}) and exp_list length ({}) should match."
        raise ValueError(txt.format(len(obs_list), len(exp_list)))

    obs_chunks = list()
    exp_chunks = list()
    for obs, exp in zip(obs_list, exp_list):
        obs_i, exp_i = _flatten_valid_pairs(obs=obs, exp=exp)
        if obs_i.shape[0] == 0:
            continue
        obs_chunks.append(obs_i)
        exp_chunks.append(exp_i)
    if len(obs_chunks) == 0:
        return 0.0, {"num_pairs": 0, "fit_loglikelihood": np.nan}

    obs_all = np.concatenate(obs_chunks)
    exp_all = np.concatenate(exp_chunks)

    max_samples = int(max_samples)
    if (max_samples > 0) and (obs_all.shape[0] > max_samples):
        idx = np.linspace(0, obs_all.shape[0] - 1, max_samples, dtype=np.int64)
        obs_all = obs_all[idx]
        exp_all = exp_all[idx]

    eps = max(float(float_tol), np.finfo(np.float64).tiny)
    ll = np.array(
        [_gamma_poisson_log_marginal(alpha=a, obs=obs_all, exp=exp_all, eps=eps) for a in alpha_grid],
        dtype=np.float64,
    )
    best_idx = int(np.nanargmax(ll))
    best_alpha = float(alpha_grid[best_idx])
    best_ll = float(ll[best_idx])

    if alpha_grid.shape[0] >= 3:
        left = alpha_grid[max(best_idx - 1, 0)]
        right = alpha_grid[min(best_idx + 1, alpha_grid.shape[0] - 1)]
        if (right > left) and np.isfinite(left) and np.isfinite(right):
            refine = np.logspace(np.log10(left), np.log10(right), 81, dtype=np.float64)
            ll_refine = np.array(
                [_gamma_poisson_log_marginal(alpha=a, obs=obs_all, exp=exp_all, eps=eps) for a in refine],
                dtype=np.float64,
            )
            ref_idx = int(np.nanargmax(ll_refine))
            if ll_refine[ref_idx] >= best_ll:
                best_alpha = float(refine[ref_idx])
                best_ll = float(ll_refine[ref_idx])

    diagnostics = {
        "num_pairs": int(obs_all.shape[0]),
        "fit_loglikelihood": best_ll,
        "alpha_grid_min": float(alpha_grid.min()),
        "alpha_grid_max": float(alpha_grid.max()),
    }
    return best_alpha, diagnostics


def compute_empirical_stat_alphas(stat_masses, stats, alpha):
    alpha = float(alpha)
    if (not np.isfinite(alpha)) or (alpha < 0):
        raise ValueError("alpha should be finite and >= 0.")
    stats = [str(s).strip().lower() for s in stats]
    if len(stats) == 0:
        return np.zeros(shape=(0,), dtype=np.float64), np.zeros(shape=(4,), dtype=np.float64)
    unknown = sorted(set(stats).difference(set(output_stat.STAT_TO_ATOMIC_WEIGHTS.keys())))
    if len(unknown):
        raise ValueError("Unsupported stat(s): {}.".format(", ".join(unknown)))
    rows = list()
    values = list()
    for stat_name, mass_value in stat_masses.items():
        stat_key = str(stat_name).strip().lower()
        if stat_key not in output_stat.STAT_TO_ATOMIC_WEIGHTS:
            continue
        mass_value = float(mass_value)
        if not np.isfinite(mass_value):
            continue
        rows.append(np.asarray(output_stat.STAT_TO_ATOMIC_WEIGHTS[stat_key], dtype=np.float64))
        values.append(max(mass_value, 0.0))
    if len(rows) == 0:
        p_atomic = np.full(
            shape=(len(output_stat.ATOMIC_OUTPUT_STATS),),
            fill_value=(1.0 / len(output_stat.ATOMIC_OUTPUT_STATS)),
            dtype=np.float64,
        )
    else:
        A = np.vstack(rows).astype(np.float64, copy=False)
        b = np.asarray(values, dtype=np.float64)
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        x = np.clip(x, a_min=0.0, a_max=None)
        total_x = x.sum(dtype=np.float64)
        if total_x > 0:
            p_atomic = x / total_x
        else:
            p_atomic = np.full(
                shape=(len(output_stat.ATOMIC_OUTPUT_STATS),),
                fill_value=(1.0 / len(output_stat.ATOMIC_OUTPUT_STATS)),
                dtype=np.float64,
            )
    # Keep strictly positive to avoid unresolved 0/0 categories for derived stats.
    p_atomic = np.clip(p_atomic, a_min=np.finfo(np.float64).tiny, a_max=None)
    p_atomic = p_atomic / p_atomic.sum(dtype=np.float64)
    alpha_atomic = alpha * p_atomic
    alpha_stats = np.zeros(shape=(len(stats),), dtype=np.float64)
    for i, stat_name in enumerate(stats):
        weights = np.asarray(output_stat.STAT_TO_ATOMIC_WEIGHTS[stat_name], dtype=np.float64)
        alpha_stats[i] = float(weights @ alpha_atomic)
    return alpha_stats, p_atomic
