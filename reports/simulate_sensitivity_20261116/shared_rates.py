"""Replay pair counts and compare products that share the site-rate category."""

import argparse
import gzip
import itertools
from pathlib import Path
import re

import numpy as np
import pandas as pd

from csubst import endpoint, endpoint_io, ete, fitted_model, genetic_code, parser_iqtree, parser_misc, tree
from csubst.main_simulate import read_fasta
from summarize import FIT, PAIR, SIM, read_pairs


def checkpoint_parameters(checkpoint, codons):
    with gzip.open(checkpoint, 'rt') as stream:
        block = re.search(r'^ModelCodon:\n((?:[ \t].*\n)+)', stream.read(), re.MULTILINE)[1]
    omega = float(re.search(r'^\s+omega:\s*(\S+)', block, re.MULTILINE)[1])
    values = np.fromstring(re.search(r'^\s+state_freq:\s*(.*)$', block, re.MULTILINE)[1], sep=',')
    np.testing.assert_allclose(values.sum(), 1.0, rtol=0, atol=1e-8)
    lookup = dict(zip(sorted(codons), values, strict=True))
    pi = np.array([lookup[codon] for codon in codons])
    return omega, pi / pi.sum()


def load_model(rep, meta, frequency_floor=False):
    if frequency_floor:
        report = rep / 'iqtree-restart-freq_floor/fit.iqtree'
    else:
        report, = (rep / 'iqtree-default').glob('*.iqtree')
    alignment = rep / 'sim/sim.fa'
    config = {
        **{f'path_iqtree_{ext}': str(report.with_suffix('.' + ext)) for ext in ('state', 'log', 'iqtree')},
        'codon_table': genetic_code.get_codon_table(1),
        'alignment_file': str(alignment),
        'float_type': np.float64,
    }
    parser_iqtree.read_state_header(config)
    parser_iqtree.read_iqtree(config)
    parser_iqtree.read_log(config)
    codons = config['codon_orders']
    config['exchangeability_matrix'] = parser_misc.read_exchangeability_matrix(
        'substitution_matrix/ECMunrest.dat', codons,
    )
    tr = ete.PhyloNode(str(rep / 'sim/sim_true_asr.treefile'), format=1)
    if frequency_floor:
        omega, pi = checkpoint_parameters(report.with_suffix('.ckp.gz'), codons)
        q = parser_misc.get_ecm_omega_rate_matrix(dict(config, omega=omega, equilibrium_frequency=pi))
        rates, weights = endpoint_io.model_rates(config)
        tr = tree.transfer_root(ete.PhyloNode(str(report.with_suffix('.treefile')), format=1), tr)
        tr = tree.add_numerical_node_labels(tr)
        parents, lengths = endpoint_io._tree_arrays({'tree': tr}, structural=False)
    else:
        fitted_model.prepare(config)
        assert config['fitted_model_provenance']['generator_sha256'] == meta['parameter_source']['generator_sha256']
        tr = tree.add_numerical_node_labels(tr)
        parents, _ = endpoint_io._tree_arrays({'tree': tr}, structural=False)
        lengths, rates, weights = meta['branch_lengths'], meta['rates'], meta['weights']
        q, pi = config['instantaneous_codon_rate_matrix'], config['equilibrium_frequency']
    model = endpoint.EndpointModel(parents, lengths, q, pi, rates, weights)
    names = {int(ete.get_prop(node, 'numerical_label')): node.name for node in tr.traverse()}
    seqs = read_fasta(alignment)
    lookup = {codon: idx for idx, codon in enumerate(codons)}
    eye = np.eye(len(codons))
    tips = {
        node: eye[[lookup[seqs[names[node]][i:i+3]] for i in range(0, len(seqs[names[node]]), 3)]]
        for node in model.leaves
    }
    code = {codon: aa for aa, codon in config['codon_table']}
    aa = np.unique([code[codon] for codon in codons], return_inverse=True)[1]
    signatures = {int(ete.get_prop(node, 'numerical_label')): frozenset(ete.get_leaf_names(node))
                  for node in tr.traverse()}
    return model, tips, aa, signatures


class CategoryFeatures:
    def __init__(self, category_count, aa):
        self.count = category_count
        self.next_index = 0
        self.membership = np.eye(aa.max() + 1)[aa]
        self.nonsyn = aa[:, None] != aa[None, :]
        self.syn = ~self.nonsyn & ~np.eye(len(aa), dtype=bool)

    def __call__(self, left, right, transition, time, weight):
        # iter_blocks visits categories in ascending order for every branch.
        i = self.next_index % self.count
        self.next_index += 1
        n = ((left @ (transition * self.nonsyn)) * right) @ self.membership
        s = (left @ (transition * self.syn)) * right
        return {i: np.concatenate((n, s), axis=1)}


def count_products(projected, weights, index, amino_acids):
    separate = (projected * weights[None, :, :, None]).sum(axis=1)
    # Each projected vector is conditional on the category. Applying its
    # square-root weight on each branch integrates the category exactly once.
    shared = projected * np.sqrt(weights)[None, :, :, None]
    result = []
    for values in (separate, shared):
        counts = []
        for channel in (values[..., :amino_acids], values[..., amino_acids:]):
            flat = np.ascontiguousarray(channel).reshape(len(values), -1)
            counts.append((flat @ flat.T)[index])
        result.append(np.column_stack(counts))
    return np.stack(result)


def evaluate(rep, frequency_floor):
    joined, meta = read_pairs(rep, 'default')
    model, tips, aa, signatures = load_model(rep, meta, frequency_floor)
    if frequency_floor:
        source = ete.PhyloNode(str(rep / 'sim/sim_true_asr.treefile'), format=1)
        original = tree.add_numerical_node_labels(source)
        assert signatures == {int(ete.get_prop(node, 'numerical_label')): frozenset(ete.get_leaf_names(node))
                              for node in original.traverse()}
    category_count = len(model.rates)
    transforms = [CategoryFeatures(category_count, aa) for _ in range(2)]
    branches = sorted(set(joined[PAIR].to_numpy().ravel()))
    column = {branch: i for i, branch in enumerate(branches)}
    index = tuple(joined[key].map(column).to_numpy() for key in PAIR)
    counts = np.zeros((2, len(joined), 4))
    rate_moments = np.zeros(2)
    for _, records in itertools.groupby(
        model.iter_blocks(
            tips, joint=False, predictive=True, transform=transforms[0], predictive_transform=transforms[1],
            transform_rate_context=True, rate_posterior=True,
        ), key=lambda item: item.start,
    ):
        root = next(records)
        weight = root.rate_posterior
        mean = model.rates @ weight
        rate_moments += [(model.rates**2 @ weight).sum(), (mean**2).sum()]
        entries = {item.child: (item.reduced, item.reduced_predictive) for item in records}
        for j in range(2):
            values = np.array([[entries[branch][j][category] for category in range(category_count)]
                               for branch in branches])
            counts[:, :, 2*j:2*j+2] += count_products(values, weight, index, aa.max() + 1)
    replay = counts[0, joined.valid]
    production = joined.loc[joined.valid, FIT].to_numpy()
    if frequency_floor:
        replay_error = np.nan
    else:
        np.testing.assert_allclose(replay, production, rtol=1e-9, atol=6e-12)
        replay_error = np.abs(replay - production).max()
    rows = []
    for group, mask in (
        ('all', joined.valid), ('foreground', joined.valid & joined.foreground.eq(2)),
        ('root_adjacent', ~joined.valid),
    ):
        scenario, seed = rep.name.rsplit('-', 1)
        labels = dict(dataset=rep.parent.name, scenario=scenario, seed=seed, group=group,
                      max_replay_abserror=replay_error,
                      posterior_rate_second_moment=rate_moments[0] / len(next(iter(tips.values()))),
                      posterior_mean_rate_squared=rate_moments[1] / len(next(iter(tips.values()))))
        exact = joined.loc[mask, SIM].sum().to_dict()
        for mode, predicted in zip(('separate', 'shared'), counts):
            rows.append(dict(**labels, mode=mode, pairs=int(mask.sum()), **exact, **dict(zip(
                FIT, predicted[mask].sum(axis=0),
            ))))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('work', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--frequency-floor', action='store_true')
    args = p.parse_args()
    rows = []
    for dataset in ('PGK', 'PEPC'):
        for rep in sorted((args.work / dataset).glob('*-*')):
            if args.frequency_floor:
                if (rep / 'iqtree-restart-freq_floor/fit.iqtree').exists():
                    rows.extend(evaluate(rep, True))
                continue
            if rep.name.rsplit('-', 1)[0] not in ('baseline', 'rate_mixture', 'rate_mixture_bias', 'long_branches'):
                continue
            if (rep / 'default/csubst_endpoint_model.json').exists() and (rep / 'default/csubst_cb_2.tsv').exists():
                rows.extend(evaluate(rep, False))
    pd.DataFrame(rows).to_csv(args.output, sep='\t', index=False, float_format='%.10g')


if __name__ == '__main__':
    main()
