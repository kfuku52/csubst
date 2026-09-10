"""Joint fitted-count nulls and shared smoothing metadata.

Poisson atoms are disjoint convergence categories, not reconstructed branch
histories. Rows and N/S channels are independent conditional on fitted means.
Other existing marginal samplers do not specify this joint law.
"""
import hashlib

import numpy as np
import pandas as pd

from csubst import output_stat, pseudocount, randomness
from csubst.omega_statistics import (
    _calc_permutation_omega_matrix, _calc_bh_fdr_qvalues,
    _calc_omega_empirical_upper_tail_counts_from_perm,
    _calc_omega_empirical_upper_tail_pvalues_from_counts,
)

ALPHA_KEYS = ('alpha_obs_N', 'alpha_exp_N', 'alpha_obs_S', 'alpha_exp_S')


def requested_stats(g):
    # Use the same aliases, string parsing and defaults as effect-size code.
    from csubst import omega
    return omega._resolve_requested_output_stats(g)


def data_dependent(g):
    cfg = pseudocount.validate_args(g)
    return cfg['pseudocount_enabled'] and (cfg['pseudocount_mode'] == 'empirical' or cfg['pseudocount_alpha_auto'])


def needs_joint(g):
    stats = requested_stats(g)
    return any('dif' in s for s in stats) or data_dependent(g)


def validate_config(g):
    if not g.get('calc_omega_pvalue', False):
        return
    if needs_joint(g) and g.get('omega_pvalue_null_model', 'hypergeom') != 'poisson':
        raise ValueError('dif P values and empirical/auto pseudocount P values require --omega_pvalue_null_model poisson; other marginal nulls do not define a supported joint category distribution.')
    if (data_dependent(g) and g.get('calibrate_longtail', False)
            and g.get('longtail_method', 'independent_null') == 'independent_null'):
        raise ValueError('Independent-null calibration supports fixed symmetric pseudocounts only; empirical/auto requires --longtail_method empirical or --calibrate_longtail no.')


def population_signature(cb):
    columns = [c for c in cb if str(c).startswith('branch_id_')]
    if not columns:
        return None
    ids = np.sort(cb[columns].to_numpy(dtype=np.int64), axis=1)
    canonical = sorted(tuple(row) for row in ids.tolist())
    return hashlib.sha256(repr(canonical).encode()).hexdigest()


def validate_population(cb, g):
    signature = population_signature(cb)
    if data_dependent(g):
        cached = g.get('_pseudocount_context', {}).get('population_signature')
        if cached is not None and cached != signature:
            raise ValueError('Pseudocount P values require the complete row population used to fit prior/alpha; a subset cannot refit that statistic.')
    for col in cb:
        if str(col).startswith('calibration_population_'):
            references = set(cb[col].dropna()) - {'', 'per_combination'}
            if references and references != {signature}:
                raise ValueError('Empirical calibrated P values require the complete calibration row population.')


def stat_alphas(context, stat):
    index = context['output_stats'].index(stat)
    return tuple(context[key][index] for key in ALPHA_KEYS)


def add_diagnostics(cb, stat, context, valid, attempted, calibration=None, joint=False):
    cfg = context['config']
    prefix = 'pvalue_'
    dependent = cfg['pseudocount_enabled'] and (cfg['pseudocount_mode'] == 'empirical' or cfg['pseudocount_alpha_auto'])
    values = {
        'statistic': ('smoothed' if cfg['pseudocount_enabled'] else 'raw') + ('_calibrated' if calibration else ''),
        'n': valid, 'undefined': np.asarray(attempted) - valid,
        'alpha': 'refit' if cfg['pseudocount_alpha_auto'] and cfg['pseudocount_mode'] != 'none' else 'fixed',
        'prior': 'refit' if dependent and cfg['pseudocount_mode'] == 'empirical' else 'fixed',
        'expectation': 'fixed_fitted_count_null',
        'joint': 'independent_poisson_atoms' if joint else 'marginal',
    }
    for name, value in values.items():
        cb[prefix + name + '_' + stat] = value


def atomic_means(base_means, tol):
    """Invert four nested means; incompatible marginals cannot be repaired."""
    aa, sa, ass, ss = (np.asarray(base_means[s], dtype=float) for s in output_stat.BASE_OUTPUT_STATS)
    atoms = np.stack((ss, sa-ss, ass-ss, aa-sa-ass+ss), axis=-1)
    scale = np.maximum(1., np.abs(aa))[..., None]
    if np.any(atoms < -float(tol)*scale) or np.any(np.isinf(atoms)):
        raise ValueError('Fitted category means do not define nonnegative Poisson atoms. The selected ASRV/count model has incompatible marginals; joint dif/empirical/auto P values are unavailable.')
    # Only round-off at subtraction boundaries, never repair a negative draw.
    return np.maximum(atoms, 0.)


def fitted_atoms(cb, tensor, channel, g):
    from csubst import omega
    means = {}
    for sub in output_stat.BASE_OUTPUT_STATS:
        col = 'EC' + channel + sub
        means[sub] = cb[col].to_numpy(dtype=float) if col in cb else omega.calc_E_stat(cb, tensor, sub, SN=channel, g=g)
    atoms = atomic_means(means, g['float_tol'])
    for sub in output_stat.ALL_OUTPUT_STATS:
        col = 'EC'+channel+sub
        if col not in cb:
            continue
        projected = project_atoms(atoms, sub)
        stored = cb[col].to_numpy(dtype=float)
        finite = np.isfinite(projected) & np.isfinite(stored)
        if not np.allclose(projected[finite], stored[finite], rtol=g['float_tol'], atol=g['float_tol']):
            raise ValueError('Stored expected counts for '+col+' disagree with the joint base-category means.')
    return atoms


def project_atoms(atoms, stat):
    weights = np.asarray(output_stat.STAT_TO_ATOMIC_WEIGHTS[stat])
    # Avoid BLAS-dependent reduction ordering, especially for exact ties.
    return (atoms * weights).sum(axis=-1)


class PoissonAtoms:
    """Persistent independent streams per canonical row/channel/atom.

    Splitting repetitions into blocks does not alter any RNG stream. Category
    display order and row order do not enter the seed.
    """
    def __init__(self, ids, means, seed, purpose):
        self.means = np.asarray(means, dtype=float)
        self.rngs = [[randomness.generator(seed, 'omega-atoms-v1', purpose, *sorted(row), atom)
                      for atom in range(4)] for row in np.asarray(ids).tolist()]

    def draw(self, niter):
        out = np.empty((len(self.means), niter, 4), dtype=float)
        for i, means in enumerate(self.means):
            if not np.isfinite(means).all():
                out[i] = np.nan
                continue
            for atom in range(4):
                out[i, :, atom] = self.rngs[i][atom].poisson(means[atom], niter)
        return out


def add_joint_pvalues(cb, ON_tensor, OS_tensor, g):
    from csubst import omega
    if (g.get('longtail_method', 'independent_null') == 'independent_null'
            and any(str(c).startswith('calibration_reference_') for c in cb)):
        from csubst import omega_calibration
        return omega_calibration.add_independent_null_pvalues(cb, ON_tensor, OS_tensor, g)
    fit_stats = omega._resolve_requested_output_stats(g)
    stats = [s for s in fit_stats if 'omegaC'+s in cb]
    budget = omega._resolve_omega_pvalue_niter_schedule(g)[-1]
    block = int(g.get('longtail_test_block_size', 256))
    if block < 1:
        raise ValueError('Null repetition block size must be positive.')
    tol = g['float_tol']
    ids = np.sort(omega._get_cb_ids(cb), axis=1)
    # Canonical order also makes empirical-Bayes subsampling order-independent.
    order = np.lexsort(np.sort(ids, axis=1).T[::-1])
    work = cb.iloc[order].copy()
    ids = ids[order]
    seed = randomness.derive_seed(randomness.configured_seed(g), 'omega-joint-test')
    samplers = {ch: PoissonAtoms(ids, fitted_atoms(work, tensor, ch, g), seed, ch)
                for ch, tensor in [('N', ON_tensor), ('S', OS_tensor)]}
    context = omega._get_pseudocount_context(work, g, fit_stats)
    dependent = data_dependent(g)
    ge = {s: np.zeros(len(work), dtype=np.int64) for s in stats}
    valid = {s: np.zeros(len(work), dtype=np.int64) for s in stats}
    transforms = {s: omega._resolve_omega_pvalue_dsc_calibration_transformation(work, s, g) for s in stats}
    # Preserve precisely the population of columns used for observed priors.
    observed_columns = [(ch, s) for ch in ('N', 'S') for s in output_stat.ALL_OUTPUT_STATS if 'OC'+ch+s in work]
    expected_columns = {p+s: work[p+s].to_numpy(dtype=float) for p in ('ECN', 'ECS') for s in output_stat.ALL_OUTPUT_STATS if p+s in work}
    print('Joint Poisson omega null: {} fixed draws, all {} rows; smoothing {}.'.format(budget, len(work), 'refitted per draw' if dependent else 'fixed'), flush=True)
    for start in range(0, budget, block):
        size = min(block, budget-start)
        draws = {ch: sampler.draw(size) for ch, sampler in samplers.items()}
        counts = {(ch, s): project_atoms(draws[ch], s) for ch in ('N', 'S') for s in set(stats) | {s for _, s in observed_columns}}
        contexts = None
        if dependent:
            contexts = []
            for rep in range(size):
                pseudo = dict(expected_columns)
                for ch, s in observed_columns:
                    pseudo['OC'+ch+s] = counts[ch, s][:, rep]
                contexts.append(pseudocount.fit_stat_context(pseudo, g, fit_stats, tol))
        for s in stats:
            alphas = stat_alphas(context, s)
            if contexts is not None:
                alphas = tuple(np.asarray([stat_alphas(c, s)[i] for c in contexts])[None, :] for i in range(4))
            simulated = _calc_permutation_omega_matrix(
                work['ECN'+s], work['ECS'+s], counts['N', s], counts['S', s], tol,
                calibrate_dsc_transformation=transforms[s], alphas=alphas,
            )
            dg, dv = _calc_omega_empirical_upper_tail_counts_from_perm(work['omegaC'+s], simulated)
            ge[s] += dg
            valid[s] += dv
    inverse = np.argsort(order)
    result = {}
    for s in stats:
        p = _calc_omega_empirical_upper_tail_pvalues_from_counts(work['omegaC'+s], work['ECS'+s], ge[s], valid[s])[inverse]
        result['pomegaC'+s] = p
        result['qomegaC'+s] = _calc_bh_fdr_qvalues(p)
        add_diagnostics(result, s, context, valid[s][inverse], budget, transforms[s], joint=True)
        if 'calibration_reference_'+s in cb:
            result['calibration_pvalue_status_'+s] = 'empirical_full_population'
            result['calibration_test_n_'+s] = valid[s][inverse]
            result['calibration_test_undefined_'+s] = budget-valid[s][inverse]
    return pd.concat([cb.drop(columns=list(result), errors='ignore'), pd.DataFrame(result, index=cb.index)], axis=1)


def add_fixed_pvalues(cb, ON_tensor, OS_tensor, g):
    """Fixed smoothing / empirical calibration with a fixed full-row budget.

    Canonical 128-draw sampling blocks isolate RNGs from the requested schedule
    and transform block size. The last block is drawn in full then truncated.
    """
    from csubst import omega, omega_calibration
    stats = omega._resolve_requested_output_stats(g)
    budget = omega._resolve_omega_pvalue_niter_schedule(g)[-1]
    ids = np.sort(omega._get_cb_ids(cb), axis=1)
    order = np.lexsort(np.sort(ids, axis=1).T[::-1])
    work = cb.iloc[order].copy()
    ids = ids[order]
    inverse = np.argsort(order)
    context = omega._get_pseudocount_context(work, g, stats)
    draw_g = omega_calibration._draw_config(work, g)
    draw_g['_omega_nbinom_alpha_cache'] = {}
    seed = randomness.derive_seed(randomness.configured_seed(g), 'omega-fixed-test')
    print('Omega null uses final scheduled budget ({} draws) and full row population.'.format(budget), flush=True)
    for sub in stats:
        if 'omegaC'+sub not in work:
            continue
        transform = omega._resolve_omega_pvalue_dsc_calibration_transformation(work, sub, g)
        ge, valid = np.zeros(len(work), dtype=np.int64), np.zeros(len(work), dtype=np.int64)
        for start in range(0, budget, 128):
            counts = []
            for channel, tensor in [('N', ON_tensor), ('S', OS_tensor)]:
                local = dict(draw_g, random_seed=randomness.derive_seed(seed, sub, channel, start), _random_stream_counters={})
                col = 'OC'+channel+sub
                counts.append(omega._get_mode_permutation_count_matrix(
                    ids, tensor, sub, channel, 128, local,
                    obs_count=work[col].to_numpy() if col in work else None,
                )[:, :min(128, budget-start)])
            simulated = _calc_permutation_omega_matrix(
                work['ECN'+sub], work['ECS'+sub], counts[0], counts[1], g['float_tol'],
                calibrate_dsc_transformation=transform, alphas=stat_alphas(context, sub),
            )
            dg, dv = _calc_omega_empirical_upper_tail_counts_from_perm(work['omegaC'+sub], simulated)
            ge += dg
            valid += dv
        p = _calc_omega_empirical_upper_tail_pvalues_from_counts(work['omegaC'+sub], work['ECS'+sub], ge, valid)[inverse]
        cb['pomegaC'+sub] = p
        cb['qomegaC'+sub] = _calc_bh_fdr_qvalues(p)
        add_diagnostics(cb, sub, context, valid[inverse], budget, transform)
        if 'calibration_reference_'+sub in cb:
            cb['calibration_pvalue_status_'+sub] = 'empirical_full_population'
            cb['calibration_test_n_'+sub] = valid[inverse]
            cb['calibration_test_undefined_'+sub] = budget-valid[inverse]
    return cb
