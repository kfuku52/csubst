"""Analytical endpoint-enrichment evidence under a fixed CTMC tree model.

Each alternative exponentially tilts *endpoint* changes on specified branches,
normalizing every transition row. A prespecified mixture likelihood ratio is
an e-value under the simple null. Its reciprocal is a conservative P-value.
Tree pruning integrates shared ancestors and site rate categories. No posterior
mass is treated as an integer observation and no null simulations are used.
"""

import json
import math
import re

import numpy as np
from scipy.special import logsumexp

from csubst import endpoint, endpoint_io


MULTIPLIERS = (2., 10., 100.)
ALTERNATIVES = (*MULTIPLIERS, 'conditional_endpoint')
SUPPORTED_MATCHES = ('any2any', 'any2spe', 'spe2any', 'spe2spe')


def validate_profile(profile):
    """Validate prespecified probabilities; never estimate weights from test data."""
    if not isinstance(profile, dict) or profile.get('version') != 1:
        raise ValueError('Analytical profile requires version 1.')
    atoms = profile.get('atoms')
    if not isinstance(atoms, list) or not atoms:
        raise ValueError('Analytical profile requires nonempty atoms.')
    for atom in atoms:
        if not isinstance(atom, dict) or set(atom) != {'multiplier', 'participation', 'weight'}:
            raise ValueError('Each atom requires multiplier, participation and weight.')
        multiplier, rho, weight = atom['multiplier'], atom['participation'], atom['weight']
        for value in (rho, weight):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Profile probabilities must be finite numbers.')
        if not 0 <= rho <= 1 or weight < 0:
            raise ValueError('Invalid profile probabilities.')
        if multiplier is not None and (isinstance(multiplier, bool)
                or not isinstance(multiplier, (int, float)) or not math.isfinite(multiplier) or multiplier < 1):
            raise ValueError('Multiplier must be finite and >=1, or null for the infinite limit.')
    if not math.isclose(sum(a['weight'] for a in atoms), 1., rel_tol=0, abs_tol=1e-10):
        raise ValueError('Profile weights must sum to one.')
    return profile


def read_profile(path):
    with open(path, encoding='utf-8') as handle:
        return validate_profile(json.load(handle))


def default_profile():
    return dict(version=1, atoms=[dict(multiplier=m, participation=1., weight=.25)
                                for m in (*MULTIPLIERS, None)])


class EndpointEnrichment:
    def __init__(self, model, mapping, profile=None):
        self.profile = validate_profile(default_profile() if profile is None else profile)
        self.atoms = [dict(a) for a in self.profile["atoms"]]
        self.model = model
        self.mapping = np.asarray(mapping)
        if (self.mapping.shape != model.pi.shape or self.mapping.dtype.kind not in 'iu'
                or np.any(self.mapping < 0)):
            raise ValueError('Invalid endpoint-to-scan state mapping.')

    def log_likelihoods(self, tips, branches=(), from_ids=(), to_ids=()):
        """Null then prespecified alternative atoms, with scaled pruning."""
        model = self.model
        if set(tips) != model.leaves:
            raise ValueError('Every leaf needs an observation likelihood.')
        branches = set(map(int, branches))
        if not branches <= set(model.order) - {model.root}:
            raise ValueError('Target branches must be non-root tree nodes.')
        event = (np.isin(self.mapping, from_ids)[:, None]
                 & np.isin(self.mapping, to_ids)[None, :]
                 & (self.mapping[:, None] != self.mapping[None, :]))
        # Infinite-tilt rows are formed without infinities.
        factors = np.array([1.] + [a["multiplier"] or 1. for a in self.atoms])
        participation = np.array([0.] + [a["participation"] for a in self.atoms])
        likelihoods = []
        for category, weight in enumerate(model.weights):
            partial: dict[int, np.ndarray] = {}
            scales: dict[int, np.ndarray] = {}
            for node in reversed(model.order):
                value = np.ones((len(factors), model.pi.size))
                scale = np.zeros(len(factors))
                if node in model.leaves:
                    tip = np.asarray(tips[node], dtype=float)
                    if tip.shape != model.pi.shape or not np.isfinite(tip).all() or np.any(tip < 0):
                        raise ValueError('Invalid leaf observation likelihood.')
                    value *= tip
                for child in model.children[node]:
                    transition = model.transition(child, category)
                    if child in branches:
                        tilted = transition[None, :, :] * np.where(event, factors[:, None, None], 1.)
                        tilted /= tilted.sum(axis=2, keepdims=True)
                        selected = transition * event
                        mass = selected.sum(axis=1, keepdims=True)
                        # If the null gives this row zero event probability
                        # (e.g. zero branch length), every finite tilt is the
                        # original row, so its limit is also the original row.
                        for index, atom in enumerate(self.atoms, 1):
                            if atom["multiplier"] is None:
                                np.divide(selected, mass, out=tilted[index], where=mass > 0)
                        tilted = (participation[:, None, None] * tilted
                                  + (1 - participation[:, None, None]) * transition)
                        value *= np.einsum('aij,aj->ai', tilted, partial.pop(child))
                    else:
                        value *= partial.pop(child) @ transition.T
                    scale += scales.pop(child)
                    normalizer = value.max(axis=1)
                    # A zero likelihood is legitimate for an impossible category.
                    positive = normalizer > 0
                    value[positive] /= normalizer[positive, None]
                    with np.errstate(divide='ignore'):
                        scale += np.log(normalizer)
                partial[node], scales[node] = value, scale
            with np.errstate(divide='ignore'):
                likelihoods.append(np.log(partial[model.root] @ model.pi) + scales[model.root] + math.log(weight))
        return logsumexp(likelihoods, axis=0)

    def test(self, tips, branches, from_ids, to_ids):
        logs = self.log_likelihoods(tips, branches, from_ids, to_ids)
        if not np.isfinite(logs[0]):
            raise ValueError('Observed site has zero likelihood under the declared null.')
        log_e = float(logsumexp(logs[1:], b=[a["weight"] for a in self.atoms]) - logs[0])
        # Keep log evidence even when a very small P underflows in float64.
        return min(1., math.exp(min(0., -log_e))), log_e


def adjusted_pvalues(pvalues, family_size, dependence='BH'):
    """Unreported prespecified hypotheses have P=1, never shrink the family."""
    p = np.asarray(pvalues, dtype=float)
    if (p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or int(family_size) != family_size or family_size < p.size):
        raise ValueError('Invalid analytical P-values or prespecified family size.')
    if dependence not in ('BH', 'BY'):
        raise ValueError('Unknown FDR method.')
    if not p.size:
        return p.copy()
    factor = 1.
    if dependence == 'BY':
        from scipy.special import digamma
        factor = float(digamma(family_size + 1) + np.euler_gamma)
    order = np.argsort(p, kind='stable')
    ranked = p[order] * family_size * factor / np.arange(1, len(p) + 1)
    out = np.empty_like(p)
    out[order] = np.minimum(1., np.minimum.accumulate(ranked[::-1])[::-1])
    return out


def family_size(sites, states, traits, targets, matches):
    if (any(isinstance(v, bool) or int(v) != v for v in (sites, states, traits, targets))
            or sites < 0 or min(states, traits, targets) < 1 or not matches):
        raise ValueError('Invalid prespecified analytical hypothesis dimensions.')
    if set(matches) - set(SUPPORTED_MATCHES):
        raise ValueError('Endpoint analytical P supports any2any, any2spe, spe2any and spe2spe only.')
    counts = {'any2any': 1, 'any2spe': states, 'spe2any': states, 'spe2spe': states * (states - 1)}
    return int(sites * traits * targets * sum(counts[m] for m in set(matches)))


def validate_options(g):
    """Reject unsupported combinations before fitting IQ-TREE or reading data."""
    mode = g.get('scan_analytic_pvalue', 'none')
    if mode == 'none':
        if g.get('scan_analytic_profile'):
            raise ValueError('--scan_analytic_profile requires --scan_analytic_pvalue endpoint_mixture.')
        return False
    if mode != 'endpoint_mixture':
        raise ValueError('--scan_analytic_pvalue must be none or endpoint_mixture.')
    if g.get('scan_pvalue_calibration', 'full_scan') != 'none':
        raise ValueError('Endpoint analytical inference requires --scan_pvalue_calibration none.')
    if str(g.get('ml_anc', False)).strip().lower() in ('true', 'yes', '1', 'y', 'on'):
        raise ValueError('Endpoint analytical scan requires --ml_anc no to retain tip ambiguity.')
    if g.get('nonsyn_recode') == '3di20':
        raise ValueError('Endpoint analytical scan currently requires a codon model.')
    from csubst import substitution_scan
    matches = substitution_scan.normalize_scan_matches(g.get('scan_match', 'any2spe'))
    family_size(0, 2, 1, 1, matches)
    if g.get('scan_analytic_profile'):
        read_profile(g['scan_analytic_profile'])
    return True


def prepare(g):
    """Freeze the null and hypothesis universe before discovery/site filtering."""
    if not validate_options(g):
        return None
    name = str(g.get('substitution_model', ''))
    if not re.fullmatch(r'(?:ECMK07|ECMrest|GY)(?:\+(?:F(?:O|Q|1X4|3X4)?|G\d*|R\d*|I))*', name):
        raise ValueError('Unsupported codon model for endpoint analytical scan.')
    with open(g['path_iqtree_iqtree'], encoding='utf-8') as handle:
        rates, weights = endpoint_io.read_rate_mixture(handle.read())
    parents, lengths = endpoint_io._tree_arrays(g, False)
    model = endpoint.EndpointModel(parents, lengths, g['instantaneous_codon_rate_matrix'],
                                   g.get('equilibrium_frequency', g.get('empirical_eq_freq')), rates, weights)
    mapping = endpoint_io._mapping(g, 'N', model.pi.size).argmax(axis=1)
    from csubst import substitution_scan
    matches = substitution_scan.normalize_scan_matches(g.get('scan_match', 'any2spe'))
    targets = ['fg']
    sites = g['state_cdn'].shape[1]
    total = family_size(sites, len(g['nonsyn_state_orders']), len(g['fg_df'].columns) - 1, len(targets), matches)
    profile = read_profile(g['scan_analytic_profile']) if g.get('scan_analytic_profile') else default_profile()
    g['scan_analytic_summary'] = dict(method='endpoint_mixture_likelihood_ratio_reciprocal',
        p_column='p_endpoint_enrichment_analytic', log_e_column='log_e_endpoint_enrichment',
        family_size=total, family_scope='all_input_sites_states_traits_targets_matches_before_selection',
        alternatives=profile["atoms"], alternative_profile=profile, null='fixed_fitted_codon_CTMC_with_rate_category_mixture',
        validity='superuniform_for_fixed_correct_null; fitted_nuisance_uncertainty_not_calibrated',
        bh_assumption='equivalent_to_eBH_for_reciprocal_e_values; arbitrary_dependence_under_fixed_correct_null',
        by_assumption='arbitrary_dependence_with_valid_marginal_P',
        rates=rates.tolist(), weights=weights.tolist(), input_sites=int(sites),
        null_model=dict(name=name, parents=parents.tolist(), branch_lengths=lengths.tolist(),
                        q=model.q.tolist(), pi=model.pi.tolist(), state_mapping=mapping.tolist()))
    return EndpointEnrichment(model, mapping, profile)


def annotate(g, frame, units, engine):
    if engine is None:
        return frame
    from csubst import substitution_scan
    out = frame.copy()
    import pandas as pd
    # Determine target clades from topology and phenotype only. ASR coverage
    # or support must never select the alternative tested on the same data.
    valid = np.array([n for n in engine.model.order if n != engine.model.root], dtype=int)
    units = substitution_scan.build_scan_units(g, pd.DataFrame({'branch_id': valid}))
    traits = list(g['fg_df'].columns[1:])
    fg = substitution_scan._fg_ids_map_from_units(units, traits, 'fg_branch_ids', g.get('fg_ids', {}))
    rate_fg = substitution_scan._rate_fg_ids_map_from_units(units, fg, traits)
    g['scan_analytic_summary']['target_branches'] = {t: ids.tolist() for t, ids in rate_fg.items()}
    values, logs = [], []
    for _, row in out.iterrows():
        site = int(row['site'])
        tips = {}
        for leaf in engine.model.leaves:
            obs = np.asarray(g['state_cdn'][leaf, site], dtype=float)
            tips[leaf] = obs if obs.sum() else np.ones(engine.model.pi.size)
        branches = substitution_scan._target_branch_ids_from_maps(rate_fg, row['trait'], row['target_class'], valid)
        p, log_e = engine.test(tips, branches,
            [int(v) for v in str(row['from_state_ids']).split(',') if v],
            [int(v) for v in str(row['to_state_ids']).split(',') if v])
        values.append(p)
        logs.append(log_e)
    total = g['scan_analytic_summary']['family_size']
    out['p_endpoint_enrichment_analytic'] = values
    out['log_e_endpoint_enrichment'] = logs
    out['scan_analytic_family_size'] = total
    out['scan_analytic_status'] = 'model_conditional_fitted_null'
    out['q_endpoint_enrichment_analytic_bh'] = adjusted_pvalues(values, total)
    out['q_endpoint_enrichment_analytic_by'] = adjusted_pvalues(values, total, 'BY')
    return out
