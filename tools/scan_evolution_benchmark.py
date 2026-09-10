"""Independent CTMC histories and exact null pruning for analytical scan studies.

The foreground generator changes Q, then samples every waiting time and jump.
No tip is overwritten. The null ASR integrates the latent site rate category.
Small four-codon trees deliberately permit a simple unscaled reference pruner
independent of the production endpoint likelihood implementation.
"""
import contextlib
import os

import numpy as np
import pandas as pd

from csubst import endpoint, endpoint_io, ete, sequence, substitution, substitution_scan, tree
from validate_scan_calibration import CODONS, Q, TOPOLOGIES


def jump_history(initial, q, duration, rng):
    state, elapsed, arrivals = int(initial), 0., 0
    while True:
        rate = -q[state, state]
        if rate == 0:
            return state, arrivals
        elapsed += rng.exponential(1 / rate)
        if elapsed >= duration:
            return state, arrivals
        probabilities = q[state].copy()
        probabilities[state] = 0
        next_state = int(rng.choice(len(q), p=probabilities / rate))
        arrivals += int(state < 2 <= next_state)
        state = next_state


def foreground_q(boost):
    q = Q.copy()
    q[:2, 2:] *= boost
    np.fill_diagonal(q, 0)
    np.fill_diagonal(q, -q.sum(axis=1))
    return q


def simulate(scenario, foreground_count, sites, signal_sites, regime, seed_key):
    rng = np.random.default_rng(np.random.SeedSequence(seed_key))
    tr = tree.add_numerical_node_labels(ete.PhyloNode(TOPOLOGIES[scenario], format=1))
    nodes = list(tr.traverse('preorder'))
    labels = {n.name: int(ete.get_prop(n, 'numerical_label')) for n in nodes}
    fg_names = ['a', 'e'] if foreground_count == 2 else ['a', 'c', 'e', 'g']
    rates = np.array([.2, 1., 2.8]) if scenario == 'uncertain' else np.ones(1)
    categories = rng.integers(len(rates), size=sites)
    if regime == 'training':
        log_boost = rng.uniform(np.log(2), np.log(64), (sites, foreground_count))
        shared = rng.random(sites) < .5
        log_boost[shared] = log_boost[shared, :1]
        boosts = np.exp(log_boost)
    elif regime == 'moderate':
        boosts = np.full((sites, foreground_count), 16.)
    elif regime == 'heterogeneous':
        boosts = np.tile([2., 64.], (sites, foreground_count // 2))
    elif regime == 'null':
        boosts = np.ones((sites, foreground_count))
    else:
        raise ValueError('Unknown generator regime.')
    boosts[signal_sites:] = 1
    states = {}
    recurrent_jumps = np.zeros(sites, dtype=int)
    recurrent_endpoints = np.zeros(sites, dtype=int)
    for node in nodes:
        label = int(ete.get_prop(node, 'numerical_label'))
        if node.up is None:
            states[label] = rng.integers(4, size=sites)
            continue
        parent = states[int(ete.get_prop(node.up, 'numerical_label'))]
        child = np.empty(sites, dtype=int)
        for site in range(sites):
            boost = boosts[site, fg_names.index(node.name)] if node.name in fg_names else 1.
            child[site], jumps = jump_history(parent[site], foreground_q(boost), node.dist * rates[categories[site]], rng)
            if node.name in fg_names:
                recurrent_jumps[site] += int(jumps > 0)
                recurrent_endpoints[site] += int(parent[site] < 2 <= child[site])
        states[label] = child
    observations = {labels[n.name]: np.eye(4)[states[labels[n.name]]] for n in nodes if ete.is_leaf(n)}
    parents, lengths = endpoint_io._tree_arrays({'tree': tr}, False)
    model = endpoint.EndpointModel(parents, lengths, Q, np.full(4, .25), rates, np.full(len(rates), 1 / len(rates)))
    return dict(tree=tr, model=model, observations=observations, fg_names=fg_names,
                fg_ids=[labels[name] for name in fg_names], sites=sites, signal_sites=signal_sites,
                recurrent_jumps=recurrent_jumps >= 2, recurrent_endpoints=recurrent_endpoints >= 2,
                seed_key=list(seed_key), boosts=boosts)


def null_posterior(data):
    tr, model, observations = data['tree'], data['model'], data['observations']
    nodes = list(tr.traverse('preorder'))
    sites = data['sites']
    marginal = np.zeros((len(nodes), sites, 4))
    likelihoods = []
    for rate, weight in zip(model.rates, model.weights):
        cache = {}
        def message(node, previous):
            key = (id(node), id(previous))
            if key in cache:
                return cache[key]
            label = int(ete.get_prop(node, 'numerical_label'))
            value = observations.get(label, np.ones((sites, 4))).copy()
            neighbors = list(ete.get_children(node)) + ([] if node.up is None else [node.up])
            for neighbor in neighbors:
                if neighbor is previous:
                    continue
                edge = node if node.up is neighbor else neighbor
                decay = np.exp(-4 * edge.dist * rate / 3)
                transition = decay * np.eye(4) + (1 - decay) / 4
                value *= message(neighbor, node) @ transition.T
            cache[key] = value
            return value
        for node in nodes:
            label = int(ete.get_prop(node, 'numerical_label'))
            marginal[label] += weight * .25 * message(node, None)
        likelihoods.append(.25 * message(tr, None).sum(axis=1))
    denominator = marginal.sum(axis=2, keepdims=True)
    if np.any(denominator <= 0):
        raise ValueError('Reference pruning underflowed; increase precision before using this fixture.')
    marginal /= denominator
    rate_weights = np.array(likelihoods).T * model.weights
    rate_weights /= rate_weights.sum(axis=1, keepdims=True)
    for node in nodes:
        if ete.is_leaf(node):
            label = int(ete.get_prop(node, 'numerical_label'))
            ete.set_prop(node, 'sequence', ''.join(CODONS[observations[label].argmax(axis=1)]))
    marginal[model.root] = 0  # Inserted-root ASR convention only; engine still integrates this node.
    return marginal, rate_weights @ model.rates


def scan_frame(data, filtered):
    from csubst import parser_misc
    state, rates = null_posterior(data)
    tr = data['tree']
    labels = {n.name: int(ete.get_prop(n, 'numerical_label')) for n in tr.traverse()}
    fg_leaves = {'trait1': [[name] for name in data['fg_names']], 'trait2': [['b'], ['f']]}
    for trait, groups in fg_leaves.items():
        for node in tr.traverse():
            leaves = set(ete.get_leaf_names(node))
            ete.add_features(node, **{'is_fg_' + trait: leaves <= set(sum(groups, []))})
            for i, names in enumerate(groups, 1):
                ete.add_features(node, **{'is_lineage_fg_{}_{}'.format(trait, i): leaves <= set(names)})
    fg = pd.DataFrame({'name': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']})
    for trait, groups in fg_leaves.items():
        fg[trait] = [next((i for i, names in enumerate(groups, 1) if name in names), 0) for name in fg['name']]
    g = dict(tree=tr, state_cdn=state, codon_orders=CODONS, num_input_site=data['sites'],
             amino_acid_orders=['K', 'N'], synonymous_indices={'K': [0, 1], 'N': [2, 3]},
             nonsynonymous_indices={'K': [0, 1], 'N': [2, 3]}, nonsyn_state_orders=np.array(['K', 'N']),
             max_synonymous_size=2, float_type=np.float64, float_tol=1e-12, threads=1,
             iqtree_rate_values=rates, instantaneous_codon_rate_matrix=Q,
             instantaneous_nsy_rate_matrix=np.array([[-2/3, 2/3], [2/3, -2/3]]),
             nonsyn_recode='no', expected_state_backend='expm', min_sub_pp=0,
             fg_df=fg, fg_leaf_names=fg_leaves,
             fg_ids={t: np.array([labels[n[0]] for n in groups]) for t, groups in fg_leaves.items()},
             fg_stem_only=False, scan_unit_mode='clade', scan_min_support='2', scan_min_event_pp=.5,
             scan_rate_length='n_rescaled', scan_rate_exposure='q_weighted', scan_other_scope='all',
             scan_rate_event_mode='posterior_sum', scan_match='any2spe,spe2spe',
             scan_pvalue_calibration='none', scan_n_permutations=0)
    g['state_nsy'] = g['state_pep'] = sequence.cdn2pep_state(state, g)
    with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
        if filtered:
            g['drop_invariant_tip_sites_mode'] = 'tip_invariant'
            if parser_misc.get_site_drop_mask(g, 'tip_invariant', np.arange(data['sites'])).all():
                return pd.DataFrame(columns=substitution_scan.SCAN_OUTPUT_COLUMNS)
            parser_misc.drop_invariant_tip_sites(g)
        on = substitution.get_substitution_tensor(g['state_nsy'], mode='asis', g=g)
        os_tensor = substitution.get_substitution_tensor(g['state_cdn'], mode='syn', g=g)
        tree.rescale_branch_length(g, os_tensor, on)
        frame, _ = substitution_scan.scan_substitutions(g, on)
    return frame
