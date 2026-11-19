"""Compare K=4 proposals with exhaustive fitted-model scores on PGK regional simulations."""
import argparse
from pathlib import Path
import re
from unittest.mock import patch

import numpy as np
import pandas as pd
from csubst import cli, ete, substitution

import branch_pair_index as index
import site_intersection_pilot as pilot


def prepare_reference(case, out, isolate_tips=None):
    tree = ete.PhyloNode(str(case / 'sim/csubst_true_asr.treefile'), format=1)
    patterns = [] if isolate_tips is None else [line.split('\t')[1] for line in isolate_tips.read_text().splitlines()]
    tips = ete.get_leaf_names(tree)
    protected = {name for name in tips if any(re.fullmatch(p, name) for p in patterns)}
    clades = [tree]
    while True:
        split = [node for node in clades if not ete.is_leaf(node) and protected.intersection(ete.get_leaf_names(node))]
        if not split and len(clades) >= min(24, len(tips)):
            break
        largest = max(split or clades, key=lambda node: len(ete.get_leaf_names(node)))
        clades.remove(largest)
        clades.extend(ete.get_children(largest))
    foreground = out / 'clades.tsv'
    foreground.write_text(''.join(f'{i}\t^{re.escape(name)}$\n' for i, node in enumerate(clades, 1)
                                  for name in ete.get_leaf_names(node)))
    get_cb = substitution.get_cb
    captured = False

    def capture(ids, tensor, g, attr, selected_base_stats=None):
        nonlocal captured
        if attr == 'OCN' and not captured:
            selected = np.unique(ids)
            reducer = substitution.get_reducer_sub_tensor(tensor, g, label='cb_OCN')
            p = reducer.project('any2spe')[selected].toarray().reshape(len(selected), -1, reducer.num_site)
            np.savez_compressed(out / 'posteriors.npz', ids=selected, p=p)
            captured = True
        return get_cb(ids, tensor, g, attr, selected_base_stats)

    args = ['search', '--alignment_file', case / 'sim/csubst.fa', '--rooted_tree_file',
            case / 'sim/csubst_true_asr.treefile', '--foreground', foreground, '--fg_stem_only', 'yes',
            '--fg_exclude_wg', 'no', '--exclude_sister_pair', 'no', '--exhaustive_until', '1',
            '--max_arity', '4', '--cutoff_stat', 'OCNany2spe,0', '--output_stat', 'any2spe',
            '--branch_dist', 'no', '--b', 'no', '--float_digit', '17', '--min_ocn_site_pp', '.5',
            '--outdir', out / 'reference']
    fit, = (case / 'fit').glob('*.iqtree')
    for ext in ('iqtree', 'treefile', 'log', 'state', 'rate'):
        args.extend(['--iqtree_' + ext, fit.with_suffix('.' + ext)])
    with patch.object(substitution, 'get_cb', capture):
        if cli.main(list(map(str, args))):
            raise RuntimeError('Reference search failed')


def evaluate(out):
    tables = {k: pd.read_csv(out / 'reference' / f'csubst_cb_{k}.tsv', sep='\t') for k in (2, 3, 4)}
    data = np.load(out / 'posteriors.npz')
    ids, p = data['ids'], data['p']
    cumul = p.cumsum(axis=1)
    rng = np.random.default_rng(9109)

    def sampled():
        for _ in range(pilot.ROUNDS):
            value = rng.random((len(ids), 1, p.shape[2]))
            dest = (value >= cumul).sum(axis=1)
            yield np.concatenate([(dest == d).T for d in range(p.shape[1])])

    sampled_keys, *_ = index.rank_witnesses(sampled(), pilot.ROUNDS, rng)
    upper_keys, *_ = index.rank_witnesses([(p.sum(axis=1) >= .5).T], 1, np.random.default_rng(9209))
    ref = tables[4].set_index([f'branch_id_{i+1}' for i in range(4)])
    rows = []
    for minimum, col in [(0, 'OCNany2spe'), (.5, 'cutoff_OCNany2spe')]:
        hits = set(ref.index[(ref[col] >= 2) & (ref.omegaCany2spe >= 5)])
        parent_tables = tables if minimum else {k: tab.drop(columns='cutoff_OCNany2spe') for k, tab in tables.items()}
        for name, local_keys in [('sampled', sampled_keys), ('branch_total_bound', upper_keys)]:
            if not minimum and name == 'branch_total_bound':
                continue
            valid = [tuple(ids[list(key)]) for key in local_keys]
            valid = [key for key in valid if key in ref.index]
            ranked = ref.loc[valid, col].sort_values(ascending=False, kind='stable').iloc[:64]
            rows.append((minimum, name, None, len(ref), len(hits), len(valid), len(hits.intersection(valid)),
                         len(ranked), len(hits.intersection(ranked.index))))
        valid_quartets = set(ref.index)
        for name in ('ocn_bound', 'all_cutoffs'):
            for cap in (pilot.PARENT_CAP, 10_000):
                keys = sorted(pilot.existing(parent_tables, name, parent_cap=cap) & valid_quartets)
                ranked = ref.loc[keys, col].sort_values(ascending=False, kind='stable').iloc[:64]
                rows.append((minimum, name, cap, len(ref), len(hits), len(keys), len(hits.intersection(keys)),
                             len(ranked), len(hits.intersection(ranked.index))))
    return pd.DataFrame(rows, columns=['min_pp','method','parent_cap','eligible_quartets','reference_hits','proposed',
                                       'hits_proposed','evaluated','hits_top64'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_dir', type=Path)
    parser.add_argument('outdir', type=Path)
    parser.add_argument('--isolate_tips', type=Path, help='Two-column simulated lineage file for the enriched panel.')
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    prepare_reference(args.case_dir, args.outdir, args.isolate_tips)
    evaluate(args.outdir).to_csv(args.outdir / 'recovery.tsv', sep='\t', index=False)
