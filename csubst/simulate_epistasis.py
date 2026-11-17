"""Codon simulation with distance-dependent terminal amino-acid preferences."""

from functools import lru_cache
import re

import numpy as np
import pandas as pd
from scipy.linalg import expm

from csubst import ete, foreground


def configure_focal_branches(g):
    """Use foreground tip matches or a deterministic selection from the input tree."""
    g = foreground.get_foreground_branch(g, simulate=True, write=False)
    trait = g['trait_name']
    tr = g['tree']
    if g['foreground'] is None:
        tips = list(ete.iter_leaves(tr))
        eligible = [node for node in tips if not ete.is_root(node.up)]
        if len(eligible) >= 2:
            tips = eligible
        count = min(len(tips), g['simulate_focal_tips'])
        indices = np.linspace(0, len(tips) - 1, count, dtype=int)
        names = [tips[i].name for i in indices]
    else:
        names = sorted({name for group in g['fg_leaf_names'][trait] for name in group})
    if len(names) < 2:
        raise ValueError('Epistasis scenarios require at least two focal terminal branches.')
    # Use one lineage per selected tip, including when the foreground file
    # groups sister tips. Preference effects apply to terminal edges only.
    g['fg_df'] = pd.DataFrame({'name': [re.escape(n) for n in names], trait: np.arange(1, len(names) + 1)})
    g = foreground.get_foreground_ids(g, write=True)
    focal = set(g['fg_ids'][trait])
    length = g['simulate_focal_branch_length']
    for node in ete.iter_leaves(tr):
        if ete.get_prop(node, 'numerical_label') in focal and length is not None:
            node.dist = length
    return g, focal


def _preference_generators(q, amino, multiplier):
    changing = amino[:, None] != amino
    models = []
    for preferred in range(amino.max() + 1):
        target = changing & (amino == preferred)
        modified = q * np.where(target, multiplier, 1.)
        modified -= np.diag(modified.sum(axis=1))
        if not np.isfinite(modified).all():
            raise ValueError('Epistasis preference multiplier overflows the codon generator.')
        models.append(modified)
    return models


def _transition_cache(q, alternatives):
    @lru_cache(maxsize=256)
    def transition(time, preferred=None):
        if not np.isfinite(time) or time < 0.:
            raise ValueError('Simulation branch length times site rate should be finite and nonnegative.')
        matrix = q if preferred is None else alternatives[preferred]
        probabilities = expm(matrix * time)
        if not np.isfinite(probabilities).all():
            raise ValueError('A simulation transition matrix is nonfinite.')
        return probabilities
    return transition


def _sample_rows(probability, uniform):
    cumulative = np.cumsum(np.maximum(probability, 0.), axis=1)
    cumulative /= cumulative[:, -1, None]
    return (uniform[:, None] >= cumulative).sum(axis=1)


def _draw_sequences(g, focal, mode, transition, amino_count):
    rates = g['simulate_site_rates']
    size = len(rates)
    active = np.arange(size) < g['num_convergent_site']
    seed = int(g.get('simulate_seed', -1))
    rng = g['simulate_rng']
    driver_rng = np.random.default_rng(seed + 100_000 if seed >= 0 else None)
    driver_rate = driver_rng.exponential(1., size)
    states, contexts = {}, {}
    for node in g['tree'].traverse('preorder'):
        node_id = int(ete.get_prop(node, 'numerical_label'))
        if ete.is_root(node):
            states[node_id] = rng.choice(len(g['eq_freq']), size=size, p=g['eq_freq'])
            contexts[node_id] = driver_rng.integers(amino_count, size=size)
            continue
        parent = int(ete.get_prop(node.up, 'numerical_label'))
        duration = float(node.dist or 0.)
        preferred = contexts[parent]
        context_draw = driver_rng.random(size)
        new_context = driver_rng.integers(amino_count, size=size)
        if mode == 'inverse' and duration > 0.:
            survival = np.exp(-driver_rate * (duration / g['simulate_epistasis_distance_scale']))
            contexts[node_id] = np.where(context_draw >= survival, new_context, preferred)
        else:
            contexts[node_id] = preferred
        uniform = rng.random(size)
        draw = np.empty(size, dtype=int)
        changed = active if node_id in focal else np.zeros(size, dtype=bool)
        times = duration * rates * np.where(changed, g['foreground_scaling_factor'], 1.)
        favored = np.where(changed & (mode != 'neutral'), preferred, -1)
        groups, inverse = np.unique(np.column_stack((times, favored)), axis=0, return_inverse=True)
        for i, (time, target) in enumerate(groups):
            keep = inverse == i
            matrix = transition(float(time), int(target) if target >= 0 else None)
            draw[keep] = _sample_rows(matrix[states[parent][keep]], uniform[keep])
        states[node_id] = draw
    codons = np.asarray(g['pyvolve_codon_orders'])
    return {n.name: ''.join(codons[states[int(ete.get_prop(n, 'numerical_label'))]]) for n in g['tree'].traverse()}


def _overlap(joint, left, right, amino_count):
    # The optional leading preference axis shares the same state on both edges.
    by_destination = (left * (joint @ right)).sum(axis=-2).reshape(-1, left.shape[-1]).mean(axis=0)
    return np.asarray([by_destination[:amino_count].sum(), by_destination[amino_count:].sum()])


def _expected_counts(g, focal, mode, transition, oracle):
    distances = [(i, j, ete.get_distance(oracle.nodes[i].up, oracle.nodes[j].up)
                  if oracle.nodes[i].up is not oracle.nodes[j].up else 0.) for i, j in oracle.pairs]
    amino_count = oracle.aa_membership.shape[1]
    rates = g['simulate_site_rates']
    active = np.arange(len(rates)) < g['num_convergent_site']
    groups, numbers = np.unique(np.column_stack((rates, active)), axis=0, return_counts=True)
    focal_mask = [int(ete.get_prop(n, 'numerical_label')) in focal for n in oracle.nodes]
    for (rate, selected), count in zip(groups, numbers):
        if rate == 0.:
            continue
        null, means, conditioned = [], [], {}
        feature_cache = {}
        for index, (node, is_focal) in enumerate(zip(oracle.nodes, focal_mask)):
            is_selected = bool(selected and is_focal)
            time = rate * float(node.dist or 0.) * (g['foreground_scaling_factor'] if is_selected else 1.)
            plain = oracle._event_features(transition(time))
            null.append(plain)
            if is_selected and mode != 'neutral':
                if time not in feature_cache:
                    feature_cache[time] = np.stack([
                        oracle._event_features(transition(time, a)) for a in range(amino_count)
                    ])
                conditioned[index] = feature_cache[time]
                means.append(conditioned[index].mean(axis=0))
            else:
                means.append(plain)
        for i, j, distance in distances:
            # All parent paths use the reversible background Q. Integration
            # over the shared ancestor gives diag(pi) exp(Q * d * rate).
            joint = g['eq_freq'][:, None] * transition(distance * rate)
            baseline = _overlap(joint, null[i], null[j], amino_count)
            signal = baseline
            if i in conditioned or j in conditioned:
                signal = _overlap(joint, means[i], means[j], amino_count)
            if i in conditioned and j in conditioned:
                retain = 1. if mode == 'shared' else 1. / (1. + distance / g['simulate_epistasis_distance_scale'])
                shared = _overlap(joint, conditioned[i], conditioned[j], amino_count)
                signal = retain * shared + (1. - retain) * signal
            oracle.counts[i, j] += count * np.r_[signal, baseline]


def evolve(g, focal, mode, expected=None):
    """Simulate codon endpoints and optionally integrate expected convergence."""
    code = {codon: aa for aa, codon in g['codon_table']}
    amino = np.unique([code[c] for c in g['pyvolve_codon_orders']], return_inverse=True)[1]
    q = g['background_Q']
    stationary_flow = g['eq_freq'][:, None] * q
    if not np.allclose(stationary_flow, stationary_flow.T, rtol=1e-8, atol=1e-12):
        raise ValueError('Epistasis scenarios require a reversible background Q and its equilibrium frequencies.')
    alternatives = _preference_generators(q, amino, g['simulate_epistasis_preference_multiplier'])
    transition = _transition_cache(q, alternatives)
    sequences = _draw_sequences(g, focal, mode, transition, len(alternatives))
    if expected is not None:
        _expected_counts(g, focal, mode, transition, expected)
    return sequences
