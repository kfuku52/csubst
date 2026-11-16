"""Stress-test branch-pair estimates after masking a codon block in one clade."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from csubst import endpoint_io, ete, genetic_code, sequence, tree

from pilot import EXPECTED, KEY, NULL, OBS, TRUTH


def run_search(folder, alignment, condition, mode, iqtree, seed):
    cmd = [sys.executable, '-m', 'csubst', 'search', '--alignment_file', alignment.name,
           '--rooted_tree_file', 'tree.nwk', '--threads', '1', '--blas_threads', '1',
           '--pair_posterior', mode, '--output_stat', 'any2spe', '--exclude_sister_pair', 'no',
           '--float_digit', '12', '--b', 'no', '--outdir', condition + '-' + mode,
           '--random_seed', str(seed)]
    if condition == 'complete':
        for ext in ('iqtree', 'treefile', 'state', 'log', 'rate'):
            cmd.extend(['--iqtree_' + ext, 'fit.' + ext])
    else:
        cmd.extend(['--iqtree_exe', str(iqtree), '--iqtree_outdir', 'masked_fit'])
    logfile = folder / (condition + '-' + mode + '.stdout')
    with logfile.open('w') as log:
        try:
            subprocess.run(cmd, cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'CSUBST search failed. See {logfile}.') from exc
    output = folder / (condition + '-' + mode)
    record = json.loads((output / 'csubst_search_run.json').read_text())
    if record['status'] != 'complete':
        raise ValueError(f'Incomplete search: {output}.')
    manifest = json.loads((output / 'csubst_endpoint_model.json').read_text())
    method = 'joint_observed_any2spe' if mode == 'joint' else 'factorized_edge_scores'
    if manifest['branch_combinations'] != method:
        raise ValueError(f'Incorrect observed-count estimator: {output}.')
    return pd.read_csv(output / 'csubst_cb_2.tsv', sep='\t'), manifest['codon']


def tip_mask(model_tree, tip_names, sites):
    eligible_clades = [node for node in model_tree.traverse() if not ete.is_leaf(node) and not ete.is_root(node)]
    candidates = [(node, [leaf.name for leaf in ete.iter_leaves(node)]) for node in eligible_clades]
    _, clade = min(candidates, key=lambda element: (abs(len(element[1]) - len(tip_names) / 4),
                                                   int(ete.get_prop(element[0], 'numerical_label'))))
    mask = np.zeros(sites, dtype=bool)
    mask[sites // 3:sites // 3 + 2 * sites // 5] = True
    return set(clade), mask


def count_truth(model_tree, parents, ids, sequences, clade, mask):
    order = [int(ete.get_prop(node, 'numerical_label')) for node in model_tree.traverse('postorder')]
    labels = {int(ete.get_prop(node, 'numerical_label')): node.name for node in model_tree.traverse()}
    names = [labels[i] for i in range(len(labels))]
    codons = np.vstack([np.frombuffer(sequences[name].encode(), dtype='S3') for name in names])
    translation = {codon.encode(): aas for aas, codon in genetic_code.get_codon_table(1)}
    amino = np.vectorize(translation.__getitem__)(codons)
    observed = np.zeros(codons.shape, dtype=bool)
    leaves = {int(ete.get_prop(node, 'numerical_label')) for node in ete.iter_leaves(model_tree)}
    for i in order:
        if i in leaves:
            observed[i] = ~mask if labels[i] in clade else True
        if parents[i] >= 0:
            observed[parents[i]] |= observed[i]
    eligible = observed & observed[parents]
    valid = parents >= 0
    eligible[~valid] = False
    nonsyn = amino != amino[parents]
    syn = ~nonsyn & (codons != codons[parents])
    a, b = ids.T
    both = eligible[a] & eligible[b]
    n = both & nonsyn[a] & nonsyn[b] & (amino[a] == amino[b])
    s = both & syn[a] & syn[b] & (codons[a] == codons[b])
    return np.column_stack((n.sum(axis=1), s.sum(axis=1))), both.sum(axis=1)


def pooled_counts(rows, label):
    observed = rows[OBS].to_numpy()
    true = rows[TRUTH].to_numpy()
    ec = rows[EXPECTED].to_numpy()
    q = rows[NULL].to_numpy()
    n_rmse, s_rmse = np.sqrt(np.mean((observed - true) ** 2, axis=0))
    return dict(
        subset=label, n_pairs=len(rows), exposure=int(rows['eligible_sites'].sum()),
        n_rmse=n_rmse, s_rmse=s_rmse,
        **{f'{name}_{channel}': values[:, i].sum()
           for name, values in [('truth', true), ('observed', observed), ('expected', ec), ('q_null', q)]
           for i, channel in enumerate(('n', 's'))},
    )


def summarize_pairs(search, simulation, parents, counts, eligible_sites, sites):
    ids = simulation[KEY].to_numpy()
    np.testing.assert_array_equal(search[KEY].to_numpy(), ids)
    frame = search.assign(**dict(zip(TRUTH, counts.T)))
    for key in NULL:
        frame[key] = simulation[key].to_numpy() * (eligible_sites / sites)
    frame['eligible_sites'] = eligible_sites
    keep = frame[EXPECTED + OBS].notna().all(axis=1).to_numpy()
    sisters = parents[ids[:, 0]] == parents[ids[:, 1]]
    subsets = {
        'all': np.ones(len(frame), dtype=bool),
        'sisters': sisters,
        'different_parents': ~sisters,
        'affected_eligibility': eligible_sites < sites,
        'two_foreground': (simulation['foreground_1'] & simulation['foreground_2']).to_numpy(bool),
    }
    return [pooled_counts(frame.loc[select & keep], group)
            for group, select in subsets.items() if (select & keep).any()]


def compare_case(job):
    rep, folder, iqtree = job
    folder.mkdir(parents=True)
    sim = rep / 'sim'
    complete = folder / 'complete.fa'
    shutil.copyfile(sim / 'sim.fa', complete)
    shutil.copyfile(sim / 'sim_true_asr.treefile', folder / 'tree.nwk')
    report, = (rep / 'iqtree-default').glob('*.iqtree')
    for ext in ('iqtree', 'treefile', 'state', 'log', 'rate', 'ckp.gz'):
        original = Path(str(report).removesuffix('.iqtree') + '.' + ext)
        shutil.copyfile(original, folder / ('fit.' + ext))
    tr = tree.add_numerical_node_labels(ete.PhyloNode(str(folder / 'tree.nwk'), format=1))
    parents, _ = endpoint_io._tree_arrays({'tree': tr}, structural=False)
    tips = sequence.read_fasta(complete)
    ancestors = sequence.read_fasta(sim / 'sim_true_asr.anc.fa')
    sites = len(next(iter(tips.values()))) // 3
    actual_rates = pd.read_csv(sim / 'sim_true_asr.rate', sep='\t')['C_Rate']
    if actual_rates.nunique() != 1 or len(actual_rates) != sites:
        raise ValueError('This diagnostic requires the uniform generating rate.')
    clade, mask = tip_mask(tr, tips, sites)
    masked = folder / 'masked.fa'
    with masked.open('w') as handle:
        for name, seq in tips.items():
            clean = ''.join('---' if use and name in clade else seq[3*i:3*i+3] for i, use in enumerate(mask))
            handle.write(f'>{name}\n{clean}\n')

    simulation = pd.read_csv(sim / 'sim_expected_omega.tsv', sep='\t').sort_values(KEY)
    ids = simulation[KEY].to_numpy()
    names = {int(ete.get_prop(node, 'numerical_label')): node.name for node in tr.traverse()}
    for i in (1, 2):
        if not simulation[f'branch_id_{i}'].map(names).eq(simulation[f'branch_name_{i}']).all():
            raise ValueError('Simulated and inferred branch labels differ.')
    unmasked_counts, unmasked_sites = count_truth(tr, parents, ids, tips | ancestors, set(), mask)
    np.testing.assert_array_equal(unmasked_counts, simulation[TRUTH].to_numpy())
    np.testing.assert_array_equal(unmasked_sites, sites)
    stats = []
    scenario, seed = rep.name.rsplit('-', 1)
    for label, alignment in [('complete', complete), ('clade_block', masked)]:
        model_reference = None
        branch_mask = set() if label == 'complete' else clade
        observed_counts, eligible_sites = count_truth(tr, parents, ids, tips | ancestors, branch_mask, mask)
        for mode in ('separate', 'joint'):
            search, metadata = run_search(folder, alignment, label, mode, iqtree, seed)
            fitted = {key: metadata[key] for key in ('rates', 'weights', 'branch_lengths', 'parameter_source')}
            if model_reference is None:
                model_reference = fitted
            if fitted != model_reference:
                raise ValueError(f'Fitted model changed in {rep.name}/{label}/{mode}.')
            if mode == 'separate':
                expected = search[EXPECTED].to_numpy()
            else:
                np.testing.assert_array_equal(expected, search[EXPECTED])
            for values in summarize_pairs(search, simulation, parents, observed_counts, eligible_sites, sites):
                stats.append(dict(scenario=scenario, seed=int(seed), condition=label, mode=mode,
                                  masked_tips=len(branch_mask), masked_sites=int(mask.sum()) if branch_mask else 0,
                                  **values))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    parser.add_argument('work', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--iqtree', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=3)
    args = parser.parse_args()
    panel = args.panel.resolve() / 'PEPC'
    work = args.work.resolve()
    cases = [(panel / name, work / name, args.iqtree.resolve()) for name in
             [f'{scenario}-{seed}' for scenario in ('baseline', 'biased') for seed in (101, 202, 303)]]
    with ThreadPoolExecutor(args.jobs) as pool:
        results = []
        for (rep, _, _), stats in zip(cases, pool.map(compare_case, cases)):
            results.extend(stats)
            print(f'{rep.name}: {len(stats)} summaries', flush=True)
    pd.DataFrame(results).to_csv(args.output, sep='\t', index=False, float_format='%.10g')


if __name__ == '__main__':
    main()
