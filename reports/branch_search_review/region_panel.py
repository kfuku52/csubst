"""Blind regional-N sensitivity panel on bundled trees, with paired IQ-TREE fits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from csubst import ete, tree
from csubst.scan_bootstrap import _precise_newick

DATA = Path(__file__).resolve().parents[2] / 'csubst/dataset'
SITES = 520
SEEDS = (5749, 5881, 6151)


def invoke(cwd, label, args):
    with (cwd / (label + '.log')).open('w') as log:
        subprocess.run([sys.executable, '-m', 'csubst', *map(str, args)], cwd=cwd,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def setup(work, dataset):
    inputs = work / dataset
    inputs.mkdir()
    for file in DATA.glob(dataset + '.*'):
        shutil.copy(file, inputs)
    tr = tree.add_numerical_node_labels(ete.PhyloNode(str(inputs / (dataset + '.tree.nwk'))))
    tips = [n for n in ete.iter_leaves(tr) if not ete.is_root(n.up)]
    marked = [tips[i] for i in np.linspace(0, len(tips) - 1, 8, dtype=int)]
    for tip in marked:
        tip.dist = .25
    (inputs / 'tree.nwk').write_text(_precise_newick(tr))
    (inputs / 'fg.tsv').write_text(''.join(f'{i}\t^{re.escape(tip.name)}$\n' for i, tip in enumerate(marked, 1)))


def evaluate(work, iqtree, case):
    dataset, fraction, seed, mode = case
    inputs = work / dataset
    rep = work / f'{dataset}-{fraction}-{seed}-{mode}'
    rep.mkdir()
    fa = inputs / (dataset + '.alignment.fa')
    args = ['simulate', '--alignment_file', fa, '--rooted_tree_file', inputs/'tree.nwk',
            '--foreground', inputs/'fg.tsv', '--num_simulated_site', SITES, '--percent_foreground_sites', fraction,
            '--simulate_seed', seed, '--background_omega', .1, '--foreground_omega', .9,
            '--target_amino_acids', 'random0' if mode == 'relax' else 'random1',
            '--export_expected_omegaC', 'yes', '--outdir', 'sim', '--threads', 1]
    for ext in ('log', 'treefile', 'state', 'rate', 'iqtree'):
        args.extend(['--iqtree_' + ext, str(fa) + '.' + ext])
    invoke(rep, 'simulate', args)
    width = SITES * fraction // 100
    for label, intervals in [('whole', ''), ('exact', f'1-{width}'), ('shifted', f'{1+width//2}-{width+width//2}')]:
        args = ['search', '--alignment_file', 'sim/csubst.fa', '--rooted_tree_file', 'sim/csubst_true_asr.treefile',
                '--iqtree_exe', iqtree, '--expectation_method', 'urn', '--threads', 1, '--branch_dist', 'no',
                '--float_digit', 17, '--b', 'no', '--output_stat', 'any2spe', '--random_seed', seed,
                '--iqtree_outdir', 'fit', '--outdir', label]
        if intervals:
            args.extend(['--urn_n_regions', intervals])
        invoke(rep, label, args)
    return summarize(rep, case)


def summarize(rep, case):
    dataset, fraction, seed, mode = case
    truth = pd.read_table(rep/'sim/csubst_expected_omega.tsv')
    keys = ['branch_id_1', 'branch_id_2']
    reference = None
    rows = []
    for label in ('whole', 'exact', 'shifted'):
        cb = pd.read_table(rep/label/'csubst_cb_2.tsv').merge(truth, how='left', on=keys, validate='one_to_one')
        assert cb.muNany2spe_sim.notna().all()
        observed = cb[keys + ['OCNany2spe', 'OCSany2spe', 'ECSany2spe']]
        if reference is None:
            reference = observed
        else:
            pd.testing.assert_frame_equal(reference, observed, check_exact=True)
        for scope, keep in [('focal', (cb.foreground_1 + cb.foreground_2) == 2),
                            ('other', (cb.foreground_1 + cb.foreground_2) < 2)]:
            d = cb[keep & cb.ECNany2spe.notna() & cb.ECSany2spe.notna()]
            passes = (d.OCNany2spe >= 2) & (d.omegaCany2spe >= 5)
            rows.append(dict(dataset=dataset, fraction=fraction, seed=seed, mode=mode, estimate=label,
                             scope=scope, pairs=len(d), passes=int(passes.sum()), low_s=int((d.OCSany2spe < 1e-9).sum()),
                             infinite=int(np.isinf(d.omegaCany2spe).sum()), expected_N=d.ECNany2spe.sum(),
                             generating_null_N=d.muNany2spe_null.sum(), generating_sim_N=d.muNany2spe_sim.sum(),
                             inferred_N=d.OCNany2spe.sum(), realized_N=d.OCNany2spe_true.sum()))
    print(rep.name, flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('--iqtree', default='iqtree')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir()
    for dataset in ('PGK', 'PEPC'):
        setup(work, dataset)
    cases = product(('PGK', 'PEPC'), (20, 40), SEEDS, ('relax', 'directional'))
    with ThreadPoolExecutor(args.jobs) as pool:
        results = pool.map(lambda case: evaluate(work, args.iqtree, case), cases)
        pd.DataFrame([r for batch in results for r in batch]).to_csv(work/'panel.tsv', sep='\t', index=False)
