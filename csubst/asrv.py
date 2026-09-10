"""Learning-set selection and provenance for empirical site weights.

These weights define a fitted count null, not an independently fitted
phylogenetic model. Evaluation branch totals are never replaced by training
totals. Numerical normalization remains in substitution.py.
"""

import json
from pathlib import Path

import numpy as np

from csubst import ete


def parse_training_branches(value):
    token = str(value).strip().lower()
    if token in ('all', 'background'):
        return token
    parts = token.split(',')
    if not parts or any(not part.strip().isdigit() for part in parts):
        raise ValueError('--asrv_training_branches requires all, background, or comma-separated nonnegative branch IDs.')
    ids = tuple(int(part) for part in parts)
    if len(set(ids)) != len(ids):
        raise ValueError('--asrv_training_branches contains duplicate IDs.')
    return ','.join(str(i) for i in sorted(ids))


def validate_options(g):
    g['asrv_training_branches'] = parse_training_branches(g.get('asrv_training_branches', 'all'))
    concentration = g.get('asrv_concentration', None)
    if concentration is not None:
        concentration = float(concentration)
        if not np.isfinite(concentration) or concentration < 0:
            raise ValueError('--asrv_concentration must be finite and nonnegative.')
        if g.get('asrv', 'each') not in ('sn', 'each', 'file_each'):
            raise ValueError('--asrv_concentration applies only to sn/each/file_each.')
    g['asrv_concentration'] = concentration
    customized = concentration is not None or g['asrv_training_branches'] != 'all' or g.get('asrv_report', False)
    if customized and g.get('expectation_method', 'codon_model') != 'urn':
        raise ValueError('ASRV training/concentration options require --expectation_method urn.')
    if g['asrv_training_branches'] != 'all' and g.get('asrv') in ('no', 'file'):
        raise ValueError('--asrv_training_branches requires empirical ASRV: pool/sn/each/file_each.')
    policy = str(g.get('urn_wallenius_expectation', 'auto')).strip().lower()
    if policy not in ('auto', 'exact'):
        raise ValueError('--urn_wallenius_expectation must be auto or exact.')
    if policy == 'exact' and (g.get('expectation_method') != 'urn' or g.get('urn_model', 'wallenius') != 'wallenius'):
        raise ValueError('--urn_wallenius_expectation exact requires urn with the Wallenius model.')
    g['urn_wallenius_expectation'] = policy


def resolve_training_ids(g, num_branch):
    """Resolve once per input tree; foreground means the union over traits."""
    token = parse_training_branches(g.get('asrv_training_branches', 'all'))
    tree_ids = {int(ete.get_prop(n, 'numerical_label')) for n in g['tree'].traverse()}
    if any(i < 0 or i >= num_branch for i in tree_ids):
        raise ValueError('ASRV tree branch IDs are out of range.')
    if token == 'all':
        ids = sorted(tree_ids)
    elif token == 'background':
        excluded = set()
        for values in g.get('target_ids', {}).values():
            excluded.update(int(i) for i in values)
        if not excluded:
            raise ValueError('Background ASRV training requires a nonempty foreground/target branch set.')
        # The root has no incoming branch and cannot provide training events.
        excluded.add(int(ete.get_prop(g['tree'], 'numerical_label')))
        ids = sorted(tree_ids - excluded)
    else:
        ids = [int(i) for i in token.split(',')]
        if not set(ids).issubset(tree_ids):
            raise ValueError('--asrv_training_branches includes IDs absent from the input tree.')
        if int(ete.get_prop(g['tree'], 'numerical_label')) in ids:
            raise ValueError('The root has no incoming branch and cannot be an ASRV training branch.')
    if not ids:
        raise ValueError('No ASRV training branches remain.')
    g['_asrv_training_ids'] = np.asarray(ids, dtype=np.int64)
    return g['_asrv_training_ids']


def validate_epistasis_compatibility(g):
    customized = g.get('asrv_training_branches', 'all') != 'all' or g.get('asrv_concentration') is not None
    if customized and g.get('epistasis_requested', False):
        raise ValueError('Custom ASRV training/concentration with epistasis is not yet supported: '
                         'structure cross-fitting defines its own clade training sets and uses per-site alpha.')


def training_site_summary(sub_tensor, mode, ids):
    """Sum training rows while preserving the full evaluation tensor."""
    from csubst import substitution_sparse

    if isinstance(sub_tensor, substitution_sparse.SparseSubstitutionTensor):
        # Row slicing preserves sparse storage; only the site/category summary
        # is dense, as in the ordinary all-branch reducer.
        selected = substitution_sparse.SparseSubstitutionTensor(
            shape=(len(ids),) + sub_tensor.shape[1:], dtype=sub_tensor.dtype,
            matrix=sub_tensor.matrix[ids, :],
        )
        return substitution_sparse.summarize_sparse_sub_tensor(selected, mode)[1]
    axes = {'spe2spe': (0,), 'spe2any': (0, 4), 'any2spe': (0, 3), 'any2any': (0, 3, 4)}
    if mode not in axes:
        raise ValueError('Unsupported ASRV category: {}'.format(mode))
    # Avoid copying a potentially large dense branch subset.
    out = None
    for branch_id in ids:
        block = sub_tensor[int(branch_id):int(branch_id) + 1].sum(axis=axes[mode])
        if out is None:
            out = block.copy()
        else:
            out += block
    if out is None:
        raise ValueError('No ASRV training branches remain.')
    return out


def record_weight_diagnostics(g, label, mass, probabilities, alpha):
    """Keep compact category summaries, not branch-by-site arrays."""
    masks = np.asarray(g['is_site_nonmissing'], dtype=bool)
    ids = np.asarray(g['_asrv_branch_ids'], dtype=np.int64)
    masks = masks[ids]
    p = np.asarray(probabilities)[ids]
    sites = masks.sum(axis=1)
    empirical = masks @ np.asarray(mass, dtype=np.float64)
    tau = g.get('asrv_concentration')
    prior = np.where(sites > 0, float(tau), 0.0) if tau is not None else sites * alpha
    total = empirical + prior
    fraction = np.divide(prior, total, out=np.zeros_like(total), where=total > 0)
    square_sum = (p * p).sum(axis=1)
    effective = np.divide(1., square_sum, out=np.zeros_like(square_sum), where=square_sum > 0)
    nonempty = sites > 0
    row = {
        'category': label, 'training_mass': float(np.sum(mass)),
        'evaluation_branches': len(ids), 'empty_mask_branches': int((~nonempty).sum()),
        'zero_empirical_mass_branches': int(((empirical == 0) & nonempty).sum()),
        'zero_weight_branches': int(((square_sum == 0) & nonempty).sum()),
        'prior_fraction_min': float(fraction[nonempty].min()) if nonempty.any() else None,
        'prior_fraction_max': float(fraction[nonempty].max()) if nonempty.any() else None,
        'effective_sites_min': float(effective[nonempty].min()) if nonempty.any() else None,
        'effective_sites_max': float(effective[nonempty].max()) if nonempty.any() else None,
    }
    g.setdefault('_asrv_diagnostics', {})[label] = row


def write_provenance(g, path):
    from csubst import recoding_config

    recode = g.get('nonsyn_recode', 'no')
    payload = {
        'schema_version': 1, 'inference_scope': 'fitted_count_null',
        'pipeline_calibrated': False,
        'asrv': g.get('asrv', 'each'),
        'training_branches': g.get('asrv_training_branches', 'all'),
        'training_branch_ids': np.asarray(g.get('_asrv_training_ids', []), dtype=int).tolist(),
        'asrv_alpha_per_site': None if g.get('asrv_concentration') is not None else g.get('asrv_dirichlet_alpha', 1.0),
        'asrv_total_concentration': g.get('asrv_concentration'),
        'site_rate_file': str(g.get('path_iqtree_rate', g.get('iqtree_rate', ''))),
        'site_rate_independence': 'not_established',
        'urn_model': g.get('urn_model', 'wallenius'),
        'urn_rounding': g.get('omega_pvalue_rounding', 'stochastic'),
        'wallenius_expectation_policy': g.get('urn_wallenius_expectation', 'auto'),
        'wallenius_methods': sorted(g.get('_urn_expectation_methods', {})),
        'null_model': g.get('omega_pvalue_null_model', 'hypergeom'),
        'conditional_pvalues_requested': bool(g.get('calc_omega_pvalue', False)),
        'poisson_full_mean_source': 'observed_branch_site_mass_urn_overlap',
        'nonsyn_recode': recode,
        'recoding_data_adaptive': recode in recoding_config.AUTO_RECODING_SCHEMES,
        'recoding_training': g.get('_nonsyn_recode_training_provenance'),
        'recoding_groups': [str(v) for v in g.get('nonsyn_state_orders', [])],
        'recoding_objective': g.get('nonsyn_recode_auto_score'),
        'recoding_seed': g.get('nonsyn_recode_auto_seed'),
        'recoding_random_starts': g.get('nonsyn_recode_auto_random_starts'),
        'weight_diagnostics': list(g.get('_asrv_diagnostics', {}).values()),
        'weight_diagnostics_requested': bool(g.get('asrv_report', False)),
        'wallenius_methods_complete': bool(g.get('_asrv_diagnostics_complete', False)),
    }
    if g.get('epistasis_enabled', False):
        payload['structure_weighting'] = g['_epistasis_provenance']
        payload['training_branches'] = 'per_prediction_clade_excluded'
        payload['training_branch_ids'] = None
        payload['asrv_alpha_per_site'] = None
        payload['weight_diagnostics'] = []
        payload['weight_diagnostics_source'] = 'epistasis.json'
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    return path


def collect_diagnostics(g, OS_tensor, ON_tensor):
    """Explicit serial audit avoids losing worker-local diagnostics.

    Covers the four base categories, including prerequisites of derived dif
    counts. It does not generate independent dif nulls or transform omega.
    """
    from csubst import omega, substitution

    g['_asrv_diagnostics'] = {}
    for channel, tensor in (('S', OS_tensor), ('N', ON_tensor)):
        for mode in ('any2any', 'spe2any', 'any2spe', 'spe2spe'):
            bg, sg_sites, categories, obs_col, _ = omega._prepare_substitution_permutation_components(
                tensor, mode, channel, g,
            )
            for _, group, ancestral, derived in categories:
                if ancestral == derived:
                    continue
                p = omega._resolve_sub_sites(g, sg_sites, mode, group, ancestral, derived, obs_col)
                asrv_mode = g.get('asrv', 'each')
                if asrv_mode in ('each', 'file_each'):
                    mass = substitution._get_mode_nonadjusted_sub_sites(sg_sites, mode, group, ancestral, derived)
                    if asrv_mode == 'file_each':
                        mass = mass * substitution._get_file_site_rates(g, len(mass))
                    alpha = float(g.get('asrv_dirichlet_alpha', 1.0))
                else:
                    # Recover the unnormalized static training mass saved at
                    # normalization time, rather than treating p as counts.
                    key = channel if asrv_mode == 'sn' else asrv_mode
                    mass = g['_asrv_static_mass'][key]
                    alpha = float(g.get('asrv_dirichlet_alpha', 1.0)) if asrv_mode == 'sn' else 0.
                label = '{}:{}:{}:{}'.format(obs_col, group, ancestral, derived)
                record_weight_diagnostics(g, label, mass, p, alpha)
                if omega._resolve_urn_model(g) == 'wallenius':
                    totals = substitution.get_sub_branches(bg, mode, group, ancestral, derived)
                    # This reducer evaluates inclusion for every branch; the
                    # one-row overlap is discarded, retaining method metadata.
                    omega._calc_wallenius_expected_overlap(
                        np.array([[0]], dtype=np.int64), p, totals, g, np.float64,
                    )
                    if g.get('omega_pvalue_null_model') == 'poisson_full':
                        raw = omega._get_mode_branch_site_mass(tensor, mode, group, ancestral, derived)
                        totals = raw.sum(axis=1)
                        branch_p = np.divide(raw, totals[:, None], out=np.zeros_like(raw), where=totals[:, None] > 0)
                        omega._calc_wallenius_expected_overlap(
                            np.array([[0]], dtype=np.int64), branch_p, totals, g, np.float64,
                        )
    g['_asrv_diagnostics_complete'] = True
