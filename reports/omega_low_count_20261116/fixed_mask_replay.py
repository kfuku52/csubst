"""Hold fitted PEPC model parameters fixed while masking one clade."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
import json
from pathlib import Path

import numpy as np
import pandas as pd

from csubst import endpoint, endpoint_io, ete, fitted_model, sequence, tree
from csubst.endpoint_pair import count_any2spe

from missingness import count_truth, summarize_pairs, tip_mask
from pilot import EXPECTED, KEY, OBS, TRUTH
from replay_posterior import fitted_generator


def separate_counts(model, tips, amino, ids, missing, eligible, block_size=16):
    n = len(model.parents)
    k = len(amino)
    af = np.eye(amino.max() + 1)[amino]
    nsy = amino[:, None] != amino
    syn = ~nsy & ~np.eye(k, dtype=bool)
    a, b = ids.T
    totals = np.zeros((len(ids), 4))
    records = model.iter_blocks(tips, block_size=block_size, predictive=True, tip_missing=missing)
    for start, block in groupby(records, key=lambda record: record.start):
        features = None
        for record in block:
            if features is None:
                size = record.stop - start
                features = [np.zeros((n, size, width)) for width in (af.shape[1], k, af.shape[1], k)]
            child = record.child
            if child != model.root:
                selected = eligible[child, start:record.stop, None]
                for i, raw in enumerate((record.joint, record.predictive)):
                    features[2*i][child] = (raw * nsy).sum(axis=1) @ af * selected
                    features[2*i+1][child] = (raw * syn).sum(axis=1) * selected
        for i, feature in enumerate(features):
            flat = feature.reshape(n, -1)
            totals[:, i] += (flat @ flat.T)[a, b]
    subroot = (model.parents[ids] == model.root).any(axis=1)
    totals[subroot, 2:] = np.nan
    return totals


def compare_case(rep):
    sim = rep / 'sim'
    tr = tree.add_numerical_node_labels(ete.PhyloNode(str(sim / 'sim_true_asr.treefile'), format=1))
    parents, _ = endpoint_io._tree_arrays({'tree': tr}, structural=False)
    cached = rep / 'default'
    metadata = json.loads((cached / 'csubst_endpoint_model.json').read_text())['codon']
    report, = (rep / 'iqtree-default').glob('*.iqtree')
    q, pi, codons, amino = fitted_generator(rep, report, metadata)
    model = endpoint.EndpointModel(parents, metadata['branch_lengths'], q, pi, metadata['rates'], metadata['weights'])
    sequences = sequence.read_fasta(sim / 'sim.fa')
    ancestors = sequence.read_fasta(sim / 'sim_true_asr.anc.fa')
    simulation = pd.read_csv(sim / 'sim_expected_omega.tsv', sep='\t').sort_values(KEY)
    reference = pd.read_csv(cached / 'csubst_cb_2.tsv', sep='\t').sort_values(KEY)
    ids = simulation[KEY].to_numpy()
    np.testing.assert_array_equal(reference[KEY], ids)
    names = {int(ete.get_prop(node, 'numerical_label')): node.name for node in tr.traverse()}
    for i in (1, 2):
        if not simulation[f'branch_id_{i}'].map(names).eq(simulation[f'branch_name_{i}']).all():
            raise ValueError('Simulated and inferred branch labels differ.')
    if pd.read_csv(sim / 'sim_true_asr.rate', sep='\t')['C_Rate'].nunique() != 1:
        raise ValueError('This diagnostic requires a uniform generating rate.')
    sites = len(next(iter(sequences.values()))) // 3
    clade, mask = tip_mask(tr, sequences, sites)
    lookup = {codon: i for i, codon in enumerate(codons)}
    eye = np.eye(len(pi))
    full = np.zeros((len(parents), sites, len(pi)))
    for leaf in model.leaves:
        seq = sequences[names[leaf]]
        full[leaf] = eye[[lookup[seq[i:i+3]] for i in range(0, len(seq), 3)]]
    observed, eligible = fitted_model.observation_masks(tr, full)
    tips = {leaf: full[leaf] for leaf in model.leaves}
    missing = {leaf: ~observed[leaf] for leaf in model.leaves}
    unmasked = separate_counts(model, tips, amino, ids, missing, eligible)
    np.testing.assert_allclose(unmasked, reference[OBS + EXPECTED], rtol=1e-9, atol=6e-12)
    true_unmasked, _ = count_truth(tr, parents, ids, sequences | ancestors, set(), mask)
    np.testing.assert_array_equal(true_unmasked, simulation[TRUTH])

    for leaf in model.leaves:
        if names[leaf] in clade:
            full[leaf, mask] = 0.
    observed, eligible = fitted_model.observation_masks(tr, full)
    missing = {leaf: ~observed[leaf] for leaf in model.leaves}
    factors = separate_counts(model, tips, amino, ids, missing, eligible)
    joint = count_any2spe(model, tips, amino, ids, 4, tip_missing=missing, eligible=eligible)
    truth, support = count_truth(tr, parents, ids, sequences | ancestors, clade, mask)
    np.testing.assert_array_equal(support, (eligible[ids[:, 0]] & eligible[ids[:, 1]]).sum(axis=1))

    rows = []
    scenario, seed = rep.name.rsplit('-', 1)
    for mode, values in [('separate', factors[:, :2]), ('joint', joint)]:
        search = reference[KEY].assign(**dict(zip(OBS + EXPECTED, np.column_stack((values, factors[:, 2:])).T)))
        rows.extend(dict(scenario=scenario, seed=int(seed), condition='clade_block_fixed', mode=mode,
                         masked_tips=len(clade), masked_sites=int(mask.sum()), **stat)
                    for stat in summarize_pairs(search, simulation, parents, truth, support, sites))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    cases = [args.panel / 'PEPC' / f'{scenario}-{seed}'
             for scenario in ('baseline', 'biased') for seed in (101, 202, 303)]
    with ProcessPoolExecutor(args.jobs) as pool:
        results = []
        for rep, rows in zip(cases, pool.map(compare_case, cases)):
            print(f'{rep.name}: {len(rows)} summaries', flush=True)
            results.extend(rows)
    pd.DataFrame(results).to_csv(args.output, sep='\t', index=False, float_format='%.10g')


if __name__ == '__main__':
    main()
