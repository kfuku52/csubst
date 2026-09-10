"""Long-tail orchestration, separate from count-null generation and smoothing.

The independent-null mode fits a separate map per branch combination. Thus
heterogeneous exposures are not pooled and a displayed subset cannot refit the
map. Sampling is conditional on the existing fitted urn model, not an external
validation of that model. Each row is processed separately to bound memory.
"""

import hashlib

import numpy as np

from csubst import longtail, randomness, omega_null
from csubst.omega_statistics import (
    _calc_bh_fdr_qvalues,
    _count_rates,
    _calc_raw_omega,
    _calc_omega_empirical_upper_tail_counts_from_perm,
    _calc_omega_empirical_upper_tail_pvalues_from_counts,
)


def validate_config(g, calibration_active=None):
    method = str(g.get('longtail_method', 'independent_null'))
    if method not in ('empirical', 'independent_null'):
        raise ValueError('--longtail_method must be empirical or independent_null.')
    if g.get('calibrate_longtail_transformation', 'quantile') != 'quantile':
        raise ValueError('Frozen long-tail references currently support quantile transformation only.')
    niter = int(g.get('longtail_null_niter', 1000))
    if niter < 100:
        raise ValueError('--longtail_null_niter must be >= 100 (a numerical minimum, not a calibration guarantee).')
    if int(g.get('longtail_test_block_size', 256)) < 1:
        raise ValueError('--longtail_test_block_size must be positive.')
    active = g.get('calibrate_longtail', False) if calibration_active is None else calibration_active
    if not active or method != 'independent_null':
        return
    if str(g.get('expectation_method', 'codon_model')) != 'urn':
        raise ValueError('Independent-null long-tail calibration requires --expectation_method urn.')
    stats = omega_null.requested_stats(g)
    if any('dif' in sub for sub in stats) and g.get('omega_pvalue_null_model', 'hypergeom') != 'poisson':
        raise ValueError('Independent-null calibration requires poisson for joint dif categories (other models: no dif).')
    if omega_null.data_dependent(g):
        raise ValueError('Independent-null calibration supports fixed symmetric pseudocounts only; empirical/auto requires empirical calibration or no calibration.')
    if (g.get('omega_pvalue_null_model', 'hypergeom') == 'nbinom'
            and str(g.get('omega_pvalue_nbinom_alpha', 'auto')).lower() == 'auto'):
        raise ValueError('Independent-null calibration with nbinom requires a fixed --omega_pvalue_nbinom_alpha; automatic dispersion estimation is not independent of the tested rows.')


def _base_seed(g):
    if '_longtail_seed' not in g:
        g['_longtail_seed'] = randomness.derive_seed(randomness.configured_seed(g), 'longtail-v1')
    return g['_longtail_seed']


def _draw_rates(row, sub, purpose, niter, ON_tensor, OS_tensor, g):
    # A dedicated seed per row and purpose isolates fitting from testing and
    # from category order, foreground selection, cache history and other RNGs.
    from csubst import omega

    ids = omega._get_cb_ids(row)
    ids = np.sort(ids, axis=1)
    local = dict(g)
    local['random_seed'] = randomness.derive_seed(_base_seed(g), purpose, 'joint' if omega_null.needs_joint(g) else sub, *ids[0].tolist())
    local['_random_stream_counters'] = {}
    local['_omega_nbinom_alpha_cache'] = {}
    counts_by_channel = []
    for channel, tensor in [('N', ON_tensor), ('S', OS_tensor)]:
        if omega_null.needs_joint(g):
            atoms = omega_null.fitted_atoms(row, tensor, channel, g)
            sampler = omega_null.PoissonAtoms(ids, atoms, local['random_seed'], channel)
            counts = omega_null.project_atoms(sampler.draw(niter), sub)
        else:
            counts = omega._get_mode_permutation_count_matrix(
                cb_ids=ids, sub_tensor=tensor, mode=sub, SN=channel, niter=niter,
                g=local, obs_count=row['OC' + channel + sub].to_numpy() if 'OC' + channel + sub in row else None,
            )
        counts_by_channel.append(counts[0])
    context = omega._get_pseudocount_context(row, g, [sub])
    return _count_rates(counts_by_channel[0], float(row['ECN'+sub].iloc[0]),
                        counts_by_channel[1], float(row['ECS'+sub].iloc[0]),
                        g.get('float_tol', 1e-12), omega_null.stat_alphas(context, sub))


def _draw_config(cb, g):
    """Pass-local deterministic inputs; never keep tensor caches on run state."""
    from csubst import omega

    ids = np.sort(omega._get_cb_ids(cb), axis=1)
    return dict(g, _longtail_count_components={}, _longtail_poisson_means={},
                _longtail_count_ids=ids,
                _longtail_count_index={tuple(row): i for i, row in enumerate(ids)})


def _null_map(row, sub, ON_tensor, OS_tensor, g):
    n, s = _draw_rates(row, sub, 'fit', int(g.get('longtail_null_niter', 1000)), ON_tensor, OS_tensor, g)
    return longtail.QuantileMap.fit(n, s)


def _calibrated_rates(mapping, n, s, float_tol):
    calibrated = mapping.apply(s)
    valid = np.isfinite(n) & np.isfinite(s)
    calibrated = np.where(valid, calibrated, s)
    return calibrated, _calc_raw_omega(n, calibrated, float_tol)


def apply_calibration(cb, g, ON_tensor=None, OS_tensor=None, reuse_reference=False):
    validate_config(g, calibration_active=True)
    method = g.get('longtail_method', 'independent_null')
    stats = omega_null.requested_stats(g)
    arity = sum(str(c).startswith('branch_id_') for c in cb.columns)
    context = g.setdefault('_longtail_references', {})
    populations = g.setdefault('_longtail_populations', {})
    out = cb.copy()
    tol = g.get('float_tol', 1e-12)
    for sub in stats:
        ns, ss, ws = 'dNC' + sub, 'dSC' + sub, 'omegaC' + sub
        if not all(col in out for col in (ns, ss, ws)):
            continue
        if ss + '_nocalib' in out:
            raise ValueError('Long-tail calibration has already been applied to ' + sub)
        n, s = out[ns].to_numpy(dtype=float), out[ss].to_numpy(dtype=float)
        inference_metadata = [c for c in out if str(c).startswith('pvalue_') and str(c).endswith('_'+sub)]
        for col in (ss, ws, 'pomegaC' + sub, 'qomegaC' + sub, *inference_metadata):
            if col in out:
                out[col + '_nocalib'] = out[col]
        # Calibrated P/Q values must be recomputed, never retain the raw ones.
        out = out.drop(columns=['pomegaC' + sub, 'qomegaC' + sub], errors='ignore')
        diagnostics = []
        if method == 'empirical':
            key = (arity, sub)
            if not reuse_reference:
                context[key] = longtail.QuantileMap.fit(n, s)
                id_cols = [c for c in cb.columns if str(c).startswith('branch_id_')]
                ids = np.sort(cb[id_cols].to_numpy(dtype=np.int64), axis=1)
                canonical = sorted(tuple(row) for row in ids.tolist())
                populations[key] = hashlib.sha256(repr(canonical).encode()).hexdigest()
            if key not in context:
                raise ValueError('Missing frozen empirical reference for arity {} / {}; cannot fit only the missing foreground rows.'.format(arity, sub))
            mapping = context[key]
            calibrated, values = _calibrated_rates(mapping, n, s, tol)
            diagnostics = [mapping.diagnostics()] * out.shape[0]
        else:
            if ON_tensor is None or OS_tensor is None:
                raise ValueError('Independent-null calibration requires N and S substitution tensors.')
            _base_seed(g)
            draw_g = _draw_config(cb, g)
            print('Fitting independent-null {} references for {:,} combinations ({} draws each).'.format(
                sub, len(out), int(g.get('longtail_null_niter', 1000))), flush=True)
            calibrated, values = s.copy(), out[ws].to_numpy(dtype=float, copy=True)
            for i in range(out.shape[0]):
                mapping = _null_map(out.iloc[[i]], sub, ON_tensor, OS_tensor, draw_g)
                calibrated[i], values[i] = _calibrated_rates(mapping, n[i], s[i], tol)
                diagnostics.append(mapping.diagnostics())
        finite = np.isfinite(n) & np.isfinite(s)
        values[~finite] = out[ws + '_nocalib'].to_numpy()[~finite]
        out[ss], out[ws] = calibrated, values
        out['calibration_method_' + sub] = method
        out['calibration_population_' + sub] = populations.get((arity, sub), '') if method == 'empirical' else 'per_combination'
        for field in ('reference', 'n', 'unique_S', 'status'):
            out['calibration_' + field + '_' + sub] = [d[field] for d in diagnostics]
        out.loc[~finite, 'calibration_status_' + sub] = 'nonfinite_target'
        out['calibration_increased_' + sub] = finite & (calibrated > s)
        out['calibration_seed_' + sub] = str(_base_seed(g)) if method == 'independent_null' else ''
        out['calibration_pvalue_status_' + sub] = 'pending' if g.get('calc_omega_pvalue', False) else 'not_requested'
        if reuse_reference and method == 'empirical' and g.get('calc_omega_pvalue', False):
            # Re-fitting the missing subset would change the test statistic.
            # Full-population joint null draws are not available in this path.
            out['pomegaC' + sub] = np.nan
            out['qomegaC' + sub] = np.nan
            out['calibration_pvalue_status_' + sub] = 'unavailable_full_population_null'
        out = out.copy()
        print('Long-tail {} {}: {}/{} denominators increased; reference is frozen.'.format(method, sub, int((calibrated > s).sum()), len(out)), flush=True)
    return out


def add_independent_null_pvalues(cb, ON_tensor, OS_tensor, g):
    """Same frozen map for observed and test null; fixed final simulation budget.

    Observed and null counts share smoothing before the same frozen map.
    Data-dependent smoothing needs population-level refitting and is rejected.
    """
    from csubst import omega

    validate_config(g, calibration_active=True)
    budget = omega._resolve_omega_pvalue_niter_schedule(g)[-1]
    block_size = int(g.get('longtail_test_block_size', 256))
    tol = g.get('float_tol', 1e-12)
    for sub in omega_null.requested_stats(g):
        col = 'omegaC' + sub
        if col + '_nocalib' not in cb:
            continue
        _base_seed(g)
        draw_g = _draw_config(cb, g)
        ge = np.zeros(len(cb), dtype=np.int64)
        valid = np.zeros(len(cb), dtype=np.int64)
        for i in range(len(cb)):
            row = cb.iloc[[i]]
            mapping = _null_map(row, sub, ON_tensor, OS_tensor, draw_g)
            if mapping.reference_id != row['calibration_reference_' + sub].iloc[0]:
                raise ValueError('Calibration reference changed between observation and null testing.')
            n, s = _draw_rates(row, sub, 'test', budget, ON_tensor, OS_tensor, draw_g)
            for start in range(0, budget, block_size):
                stop = start + block_size
                _, simulated = _calibrated_rates(mapping, n[start:stop], s[start:stop], tol)
                delta, count = _calc_omega_empirical_upper_tail_counts_from_perm(row[col].to_numpy(), simulated[None, :])
                ge[i] += delta[0]
                valid[i] += count[0]
        p = _calc_omega_empirical_upper_tail_pvalues_from_counts(cb[col].to_numpy(), cb['ECS' + sub].to_numpy(), ge, valid)
        smoothing_context = omega._get_pseudocount_context(cb, g, [sub])
        omega_null.add_diagnostics(cb, sub, smoothing_context, valid, budget, 'independent_null', omega_null.needs_joint(g))
        cb['pomegaC' + sub] = p
        cb['qomegaC' + sub] = _calc_bh_fdr_qvalues(p)
        cb['calibration_test_n_' + sub] = valid
        cb['calibration_test_undefined_' + sub] = budget - valid
        cb['calibration_pvalue_status_' + sub] = 'fixed_independent_null'
        cb = cb.copy()
        print('Independent-null calibrated pomegaC {}: {} test draws per row; final scheduled budget, no adaptive row selection.'.format(sub, budget), flush=True)
    return cb
