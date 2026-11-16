"""Likelihood checks and recovery for the empirical-frequency ECMK07 hybrid."""

from dataclasses import dataclass
import gzip
import re

import numpy as np

from csubst import endpoint, endpoint_io, ete, fitted_model, parser_iqtree, parser_misc, sequence_io, tree


FREQUENCY_FLOOR = '0.0001'
FREE_RATE_MODEL = re.compile(r'ECMK07_GY0K\+F(?:\+R(\d+))?')


def checks_hybrid_omega(g):
    return (g.get('subcommand') in ('search', 'analyze', 'benchmark')
            and FREE_RATE_MODEL.fullmatch(str(g['iqtree_model'])) is not None)


def _fit_ratelaw(checkpoint, category_count):
    if category_count == 1:
        return np.ones(1), np.ones(1)
    block = fitted_model.checkpoint_block(checkpoint, f'RateFree{category_count}')
    arrays = []
    for label in ('rates', 'prop'):
        token = re.search(r'^[ \t]+' + label + r':[ \t]*(.*)$', block, re.MULTILINE)
        if token is None:
            raise ValueError(f'IQ-TREE checkpoint is missing FreeRate {label}.')
        values = np.array([float(value) for value in token[1].split(',')])
        if values.shape != (category_count,) or not np.isfinite(values).all():
            raise ValueError(f'Invalid IQ-TREE checkpoint FreeRate {label}.')
        arrays.append(values)
    rates, weights = arrays
    if not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-8):
        raise ValueError('IQ-TREE checkpoint FreeRate weights must sum to one.')
    keep = weights > 0
    if not keep.any() or (weights < 0).any() or (rates < 0).any():
        raise ValueError('Invalid IQ-TREE checkpoint FreeRate mixture.')
    return rates[keep], weights[keep] / weights[keep].sum()


def _tip_likelihoods(g, fitted_tree):
    records = sequence_io.read_fasta_records(g['alignment_file'])
    sequences = sequence_io.records_to_dict(records, key='id')
    codon_lookup = {codon: i for i, codon in enumerate(g['codon_orders'])}
    lengths = {len(seq) for seq in sequences.values()}
    sites = g['num_input_site']
    if lengths != {sites * 3}:
        raise ValueError('The IQ-TREE likelihood check requires a complete codon alignment.')
    # IQ-TREE treats partially ambiguous codons and stop codons as missing
    # when it converts the nucleotide alignment to codon states.
    alphabet_size = len(codon_lookup)
    symbols = np.vstack([np.eye(alphabet_size), np.ones(alphabet_size)])
    tips = {}
    for node in ete.iter_leaves(fitted_tree):
        seq = sequences.get(node.name)
        if seq is None:
            raise ValueError(f'IQ-TREE likelihood check could not match tip {node.name!r} to the alignment.')
        seq = seq.upper().replace('U', 'T')
        states = [codon_lookup.get(seq[i:i+3], alphabet_size) for i in range(0, len(seq), 3)]
        tips[int(ete.get_prop(node, 'numerical_label'))] = symbols[states]
    return tips


@dataclass(frozen=True)
class OmegaCheck:
    omega: float
    reported: float
    reproduced: float
    probe_gain: float
    tolerance: float

    @property
    def stalled(self):
        return self.probe_gain > self.tolerance


def audit_hybrid_omega(g, *, verify_refit=False):
    """Probe a fit at the IQ-TREE starting value using its exact likelihood model."""
    checkpoint_path = fitted_model.checkpoint_path(g['path_iqtree_state'])
    with gzip.open(checkpoint_path, 'rt') as handle:
        checkpoint = handle.read()
    block = fitted_model.checkpoint_block(checkpoint, 'ModelCodon')
    match = re.search(r'^\s+omega:\s*(\S+)', block, re.MULTILINE)
    if match is None:
        raise ValueError('Missing omega in the IQ-TREE ModelCodon checkpoint.')
    omega = float(match[1])
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError('Invalid omega in the IQ-TREE ModelCodon checkpoint.')
    if omega != 1 and not verify_refit:
        return None
    fit = dict(g, float_type=np.float64)
    parser_iqtree.read_state_header(fit)
    parser_iqtree.read_iqtree(fit)
    parser_iqtree.read_log(fit)
    if fit['substitution_model'] != g['iqtree_model']:
        raise ValueError('IQ-TREE used a different model for the ECMK07 hybrid fit.')
    fit['exchangeability_matrix'] = parser_misc.read_exchangeability_matrix(
        'substitution_matrix/ECMunrest.dat', fit['codon_orders'],
    )
    fitted_model.prepare(fit)
    category_count = FREE_RATE_MODEL.fullmatch(fit['substitution_model'])[1]
    rates, weights = _fit_ratelaw(checkpoint, int(category_count or 1))
    fitted_tree = ete.PhyloNode(g['path_iqtree_treefile'], format=1)
    fitted_tree = tree.add_numerical_node_labels(fitted_tree)
    parents, lengths = endpoint_io._tree_arrays({'tree': fitted_tree}, structural=False)
    tips = _tip_likelihoods(fit, fitted_tree)
    pi = fit['equilibrium_frequency']

    def loglik(q):
        model = endpoint.EndpointModel(parents, lengths, q, pi, rates, weights)
        return sum(float(record.site_log_likelihood.sum()) for record in model.iter_blocks(
            tips, likelihood_only=True,
        ))

    report = parser_iqtree._read_text(g['path_iqtree_iqtree'])
    token = re.search(r'^Log-likelihood of the tree:[ \t]*(\S+)', report, re.MULTILINE)
    if token is None:
        raise ValueError('IQ-TREE likelihood is missing from the fitted model report.')
    reported = float(token[1])
    reproduced = loglik(fit['instantaneous_codon_rate_matrix'])
    tolerance = max(.002, 1e-8 * fit['num_input_site'])
    if not np.isfinite(reported) or abs(reproduced - reported) > tolerance:
        raise ValueError(f'Cannot check IQ-TREE omega convergence: fitted likelihood '
                         f'{reported:.8f} differs from reconstructed likelihood {reproduced:.8f}.')
    nearby = [loglik(parser_misc.get_ecm_omega_rate_matrix(dict(fit, omega=omega * multiplier)))
              for multiplier in (.95, 1 / .95)]
    return OmegaCheck(omega, reported, reproduced, max(nearby) - reproduced, tolerance)


def validate_recovery(original, recovered):
    if recovered.stalled or recovered.reported <= original.reported + original.tolerance:
        raise ValueError(
            'IQ-TREE omega optimization did not recover after applying the codon frequency floor. '
            f'Original omega={original.omega:.8g}, log-likelihood={original.reported:.8f}; '
            f'refit omega={recovered.omega:.8g}, log-likelihood={recovered.reported:.8f}.'
        )
