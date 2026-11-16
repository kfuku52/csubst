"""Replay and decompose the PEPC search expectations under fixed fitted posteriors."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

from csubst import endpoint, endpoint_io, ete, genetic_code, parser_iqtree, parser_misc, tree
from csubst.main_simulate import read_fasta, rescale_substitution_matrix


PAIR = ['branch_id_1', 'branch_id_2']
CHANNELS = ['ECNany2spe', 'ECSany2spe']
SCENARIOS = {'null': 'background', 'biased': 'biased'}
SEEDS = (101, 202, 303)
FACTORS = ('n_exchangeability', 'frequencies', 'lengths', 'site_rates')
CORNERS = tuple(itertools.product((0, 1), repeat=len(FACTORS)))


def load_fit(run, alignment):
    report, = (run / 'iqtree').glob('*.iqtree')
    state = report.with_suffix('.state')
    header = parser_iqtree._read_state_header_columns(state)
    codons = np.array([column.removeprefix('p_') for column in header[3:]])
    config = parser_iqtree.read_iqtree(dict(
        path_iqtree_iqtree=str(report), alignment_file=str(alignment),
        codon_orders=codons, float_type=np.float64,
    ))
    if config['substitution_model'] != 'ECMK07+F+R4':
        raise ValueError('This audit expects ECMK07+F+R4.')
    fit = json.loads((run / 'search/csubst_endpoint_model.json').read_text())['codon']
    rates, weights = endpoint_io.model_rates(config)
    np.testing.assert_array_equal(rates, fit['rates'])
    np.testing.assert_array_equal(weights, fit['weights'])
    return codons, config['equilibrium_frequency'], fit, rates, weights


def load_generators(codons, fitted_pi, fit, bundled_report):
    exchange = parser_misc.read_exchangeability_matrix('substitution_matrix/ECMunrest.dat', codons)
    original = parser_iqtree.read_iqtree(dict(
        path_iqtree_iqtree=str(bundled_report), codon_orders=codons, float_type=np.float64,
    ))
    if original['substitution_model'] != 'ECMK07+F+R4':
        raise ValueError('Expected the original bundled PEPC ECMK07+F+R4 fit.')
    sim_pi = original['equilibrium_frequency']
    sim_pi = sim_pi / sim_pi.sum()
    code = {codon: aa for aa, codon in genetic_code.get_codon_table(1)}
    _, aa = np.unique([code[codon] for codon in codons], return_inverse=True)
    membership = np.eye(aa.max() + 1)[aa]
    nonsyn = aa[:, None] != aa[None, :]
    syn = ~nonsyn & ~np.eye(len(codons), dtype=bool)
    template = parser_misc.exchangeability2Q(exchange, sim_pi)
    # Match simulate's sequential summation in the foreground/background Q setup.
    n_index, s_index = np.argwhere(nonsyn), np.argwhere(syn)
    n_total = sum(template[i, j] for i, j in n_index)
    s_total = sum(template[i, j] for i, j in s_index)
    factor = 0.2 / (n_total / s_total)
    generators = {
        (0, 0): rescale_substitution_matrix(template, n_index, factor, eq_freq=sim_pi),
        (1, 0): template,
        (1, 1): parser_misc.exchangeability2Q(exchange, fitted_pi),
    }
    scaled_exchange = exchange.copy()
    scaled_exchange[nonsyn] *= factor
    generators[0, 1] = parser_misc.exchangeability2Q(scaled_exchange, fitted_pi)
    digest = hashlib.sha256()
    for values in (generators[1, 1], fitted_pi):
        digest.update(np.asarray(values, dtype='<f8').tobytes())
    if digest.hexdigest() != fit['parameter_source']['generator_sha256']:
        raise ValueError('Reconstructed fitted Q and pi differ from the production manifest.')
    flux = {}
    for label, q, pi in [
        ('simulation', generators[0, 0], sim_pi), ('fitted', generators[1, 1], fitted_pi),
    ]:
        flux[f'{label}_stationary_N_fraction'] = ((pi[:, None] * q)[nonsyn]).sum()
        flux[f'{label}_stationary_S_fraction'] = ((pi[:, None] * q)[syn]).sum()
    flux['simulation_n_exchangeability_multiplier'] = factor
    flux['frequency_total_variation'] = np.abs(sim_pi - fitted_pi).sum() / 2
    return generators, membership, nonsyn, syn, flux


def kernel(transition, membership, nonsyn, syn):
    return np.concatenate(((transition * nonsyn) @ membership, transition * syn), axis=1)


def kernel_grid(generators, tree_lengths, model, branches, membership, nonsyn, syn):
    kernels = {}
    for scale, freq, lengths, rate in CORNERS:
        q = generators[scale, freq]
        t = tree_lengths[lengths]
        rates = model.rates if rate else [1.0]
        result = []
        for b in branches:
            branch = []
            for c, value in enumerate(rates):
                if (scale, freq, lengths, rate) == (1, 1, 1, 1):
                    transition = model.transition(b, c)
                else:
                    transition = expm(q * (t[b] * value))
                branch.append(kernel(transition, membership, nonsyn, syn))
            result.append(branch)
        kernels[scale, freq, lengths, rate] = np.asarray(result)
    return kernels


def pair_counts(features, num_aa, pair_index):
    result = []
    for channel in (features[:, :, :num_aa], features[:, :, num_aa:]):
        flattened = np.ascontiguousarray(channel).reshape(len(features), -1)
        result.append((flattened @ flattened.T)[pair_index])
    return np.column_stack(result)


def encode_sequences(sequences, codons):
    lookup = {codon: i for i, codon in enumerate(codons)}
    return {
        name: np.array([lookup[seq[i:i+3]] for i in range(0, len(seq), 3)], dtype=int)
        for name, seq in sequences.items()
    }


def decompose_run(run, bundled_report):
    manifest = json.loads((run / 'search/csubst_search_run.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError(f'Incomplete search in {run}')
    alignment = run / 'sim/sim.fa'
    codons, fitted_pi, fit, rates, weights = load_fit(run, alignment)
    generators, membership, nonsyn, syn, info = load_generators(
        codons, fitted_pi, fit, bundled_report,
    )
    num_aa = membership.shape[1]
    tr = tree.add_numerical_node_labels(ete.PhyloNode(str(run / 'sim/sim_true_asr.treefile'), format=1))
    parents, true_lengths = endpoint_io._tree_arrays({'tree': tr}, structural=False)
    true_lengths[true_lengths <= 1e-8] = 0.0
    fitted_lengths = np.asarray(fit['branch_lengths'])
    model = endpoint.EndpointModel(parents, fitted_lengths, generators[1, 1], fitted_pi, rates, weights)
    names = {int(ete.get_prop(node, 'numerical_label')): node.name for node in tr.traverse()}
    observable = read_fasta(alignment)
    encoded = encode_sequences(observable, codons)
    tips = {leaf: np.eye(len(codons))[encoded[names[leaf]]] for leaf in model.leaves}
    sim = pd.read_csv(run / 'sim/sim_expected_omega.tsv', sep='\t')
    search = pd.read_csv(run / 'search/csubst_cb_2.tsv', sep='\t')
    joined = sim.merge(search[PAIR + CHANNELS], on=PAIR, how='outer', validate='one_to_one', indicator=True)
    if not (joined.pop('_merge') == 'both').all():
        raise ValueError('The simulation and search branch pairs differ.')
    for position in (1, 2):
        if not joined[f'branch_id_{position}'].map(names).eq(joined[f'branch_name_{position}']).all():
            raise ValueError('Branch ID and leaf-signature mapping differ.')
    branches = np.unique(joined[PAIR].to_numpy())
    rows = dict(zip(branches, range(len(branches))))
    pairs = tuple(joined[name].map(rows).to_numpy() for name in PAIR)
    parent_order = parents[branches]
    kernels = kernel_grid(
        generators, (true_lengths, fitted_lengths), model, branches, membership, nonsyn, syn,
    )
    counts = {corner: np.zeros((len(joined), 2)) for corner in CORNERS}
    posterior_rate_sums = np.zeros(2)
    for _, records in itertools.groupby(model.iter_blocks(
        tips, block_size=fit['block_size'], joint=False, category_nodes=True, rate_posterior=True,
    ), key=lambda record: record.start):
        root = next(records)
        posterior = {root.child: root.category_node}
        posterior_rate_sums += np.stack((rates, rates**2)) @ root.rate_posterior.sum(axis=1)
        for record in records:
            posterior[record.child] = record.category_node
        parent = np.stack([posterior[p] for p in parent_order])
        for corner, k in kernels.items():
            if corner[-1]:
                features = np.matmul(parent, k).sum(axis=1)
            else:
                features = parent.sum(axis=1) @ k[:, 0]
            counts[corner] += pair_counts(features, num_aa, pairs)
    full = counts[1, 1, 1, 1]
    reference = joined[CHANNELS].to_numpy()
    # Search suppresses both ECs for any pair containing a child of the root.
    mask = (parents[joined[PAIR].to_numpy()] != model.root).all(axis=1)
    np.testing.assert_array_equal(np.isfinite(reference), np.broadcast_to(mask[:, None], reference.shape))
    np.testing.assert_allclose(full[mask], reference[mask], rtol=1e-9, atol=6e-12)
    info['max_replay_absolute_error'] = np.abs(full[mask] - reference[mask]).max()
    true_states = encode_sequences(read_fasta(run / 'sim/sim_true_asr.anc.fa'), codons)
    known = np.stack([kernels[0, 0, 0, 0][b, 0, true_states[names[p]]]
                      for b, p in enumerate(parent_order)])
    counts['true_parents'] = pair_counts(known, num_aa, pairs)
    counts['sim_exact'] = joined[['muNany2spe_null', 'muSany2spe_null']].to_numpy()
    fg_columns = ['foreground_1', 'foreground_2']
    fg_branches = set(joined.loc[joined[fg_columns[0]].eq(1), PAIR[0]])
    fg_branches.update(joined.loc[joined[fg_columns[1]].eq(1), PAIR[1]])
    info['tree_length_sim'] = true_lengths.sum()
    info['tree_length_fitted'] = fitted_lengths.sum()
    info['foreground_length_sim'] = true_lengths[sorted(fg_branches)].sum()
    info['foreground_length_fitted'] = fitted_lengths[sorted(fg_branches)].sum()
    for i, value in enumerate(posterior_rate_sums / next(iter(tips.values())).shape[0], start=1):
        info[f'posterior_rate_moment_{i}'] = value
    for i, (rate, weight) in enumerate(zip(rates, weights), start=1):
        info[f'category_{i}_rate'] = rate
        info[f'category_{i}_weight'] = weight
    fg_count = joined[fg_columns].sum(axis=1).to_numpy()
    grouped = {}
    for group in range(3):
        selected = fg_count == group
        grouped[group] = {corner: values[selected & mask].sum(axis=0)
                          for corner, values in counts.items()}
        grouped[group]['sim_exact_all'] = counts['sim_exact'][selected].sum(axis=0)
        info[f'pairs_fg{group}'] = selected.sum()
        info[f'ec_pairs_fg{group}'] = (selected & mask).sum()
    return grouped, info


def three_factor_shapley(counts):
    # Average changes on the raw count scale over all six orders of Q, t, r.
    values = {(q, t, r): counts[q, q, t, r] for q, t, r in itertools.product((0, 1), repeat=3)}
    contributions = np.zeros((3, 2))
    for order in itertools.permutations(range(3)):
        current = [0, 0, 0]
        for factor in order:
            before = values[tuple(current)]
            current[factor] = 1
            contributions[factor] += (values[tuple(current)] - before) / 6
    start, stop = values[0, 0, 0], values[1, 1, 1]
    np.testing.assert_allclose(contributions.sum(axis=0), stop - start, rtol=1e-12, atol=1e-10)
    return {
        'sim_exact': counts['sim_exact'], 'true_parents': counts['true_parents'],
        'root_adjacent_exclusion': counts['sim_exact'] - counts['sim_exact_all'],
        'fixed_fitted_parents': start, 'parent_and_pairing_residual': start - counts['sim_exact'],
        'Q': contributions[0], 'branch_lengths': contributions[1], 'site_rates': contributions[2],
        'full_search': stop,
        'Q_n_exchangeability_at_fitted_t_r': (
            counts[1, 0, 1, 1] - counts[0, 0, 1, 1] + counts[1, 1, 1, 1] - counts[0, 1, 1, 1]
        ) / 2,
        'Q_frequencies_at_fitted_t_r': (
            counts[0, 1, 1, 1] - counts[0, 0, 1, 1] + counts[1, 1, 1, 1] - counts[1, 0, 1, 1]
        ) / 2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    corners, diagnostics, effects = [], [], []
    for directory, scenario in SCENARIOS.items():
        for seed in SEEDS:
            print(f'{scenario} {seed}', flush=True)
            by_group, info = decompose_run(
                args.runs / f'{directory}-{seed}', args.runs / 'PEPC.alignment.fa.iqtree',
            )
            diagnostics.append(dict(scenario=scenario, seed=seed, **info))
            for group, counts in by_group.items():
                labels = dict(scenario=scenario, seed=seed, num_fg_branches=group)
                for corner, values in counts.items():
                    label = corner if isinstance(corner, str) else ''.join(map(str, corner))
                    corners.append(dict(**labels, corner=label, **dict(zip(CHANNELS, values))))
                for term, values in three_factor_shapley(counts).items():
                    effects.append(dict(**labels, term=term, **dict(zip(CHANNELS, values))))
    args.output.mkdir(parents=True, exist_ok=True)
    by_seed = pd.DataFrame(effects)
    for filename, frame in (
        ('ec_corners.tsv', pd.DataFrame(corners)), ('ec_fit_diagnostics.tsv', pd.DataFrame(diagnostics)),
        ('ec_effects_by_seed.tsv', by_seed),
        ('ec_effects.tsv', by_seed.drop(columns='seed').groupby(
            ['scenario', 'num_fg_branches', 'term'], sort=False, as_index=False,
        ).sum()),
    ):
        frame.to_csv(args.output / filename, sep='\t', index=False, float_format='%.12g')


if __name__ == '__main__':
    main()
