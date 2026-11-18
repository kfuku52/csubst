"""Matched PEPC endpoint-rate and directional-preference controls for pairwise search."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import brentq

from csubst import cli, ete, genetic_code, main_simulate as sim, parser_iqtree, simulate_omega, tree

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / 'csubst/dataset'
STEMS = (1, 5, 26, 33, 35, 40, 102, 103, 124)
SEEDS = (5209, 5351, 5483, 5623, 5749, 5881, 6011, 6151)
SITES, DOMAIN, BACKGROUND, LOCAL = 1040, 208, 0.1, 0.9
SCENARIOS = ('baseline', 'global', 'domain', 'directional')
EXT = ('iqtree', 'log', 'treefile', 'rate', 'state')


def calibrate(work, stem_scale):
    """Integrate N endpoint changes over nine non-nested foreground branches."""
    codons = sim.get_pyvolve_codon_order()
    code = genetic_code.get_codon_table(1)
    translation = {codon: aa for aa, codon in code}
    aa = np.array([translation[c] for c in codons])
    nonsyn = aa[:, None] != aa
    report = SOURCE / 'PEPC.alignment.fa.iqtree'
    fit = parser_iqtree.read_iqtree(dict(path_iqtree_iqtree=str(report), codon_orders=codons,
                                         float_type=np.float64))
    pi = fit['equilibrium_frequency']
    pi = pi / pi.sum()
    syn = sim.get_synonymous_codon_substitution_index(dict(codon_table=code), codons)
    ns = sim.get_nonsynonymous_codon_substitution_index(syn)
    g = dict(pyvolve_codon_orders=codons, eq_freq=pi, background_omega=BACKGROUND,
             substitution_model=fit['substitution_model'], all_nsy_cdn_index=ns, all_syn_cdn_index=syn)
    q = sim.get_background_Q(g, 'auto')
    np.testing.assert_allclose(pi @ q, 0, atol=1e-12)
    t = tree.add_numerical_node_labels(ete.PhyloNode(str(SOURCE / 'PEPC.tree.nwk'), format=1))
    nodes = {ete.get_prop(n, 'numerical_label'): n for n in t.traverse()}
    assert all(not any(a == nodes[j] for a in ete.iter_ancestors(nodes[i])) for i in STEMS for j in STEMS)
    if stem_scale != 1:
        from csubst.scan_bootstrap import _precise_newick
        for i in STEMS:
            nodes[i].dist *= stem_scale
        (work / 'generating.tree').write_text(_precise_newick(t))
    lengths = [nodes[i].dist for i in STEMS]
    def n_per_site(matrix):
        return np.array([pi @ (expm(matrix * length) * nonsyn).sum(1) for length in lengths])
    bg = n_per_site(q)
    def profile(omega, target=''):
        route = sim.get_biased_nonsynonymous_substitution_index(target, code, codons)
        matrix = sim.apply_percent_biased_sub(q, 90., route, target, code, codons, ns, syn, omega, pi)
        return n_per_site(matrix)

    domain = (SITES - DOMAIN) * bg + DOMAIN * profile(LOCAL)
    global_omega = brentq(lambda w: SITES * profile(w).sum() - domain.sum(), BACKGROUND, LOCAL)
    global_n = SITES * profile(global_omega)
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        targets = Counter(str(sim.get_biased_amino_acids('random1', code, rng=rng)[0]) for _ in range(DOMAIN))
        def directional(w):
            return (SITES - DOMAIN) * bg + sum(v * profile(w, a) for a, v in targets.items())
        w = brentq(lambda x: directional(x).sum() - domain.sum(), .01, 20)
        for name, omega, expected, fraction in (
            ('baseline', BACKGROUND, SITES * bg, 0), ('global', global_omega, global_n, 100),
            ('domain', LOCAL, domain, 20), ('directional', w, directional(w), 20),
        ):
            if name != 'baseline':
                np.testing.assert_allclose(expected.sum(), domain.sum(), rtol=1e-9)
            rows.append(dict(scenario=name, seed=seed, stem_scale=stem_scale, omega=omega, percent=fraction,
                             **{f'expected_N_{i}': x for i, x in zip(STEMS, expected)}))
    pd.DataFrame(rows).to_csv(work / 'calibration.tsv', sep='\t', index=False)


def invoke(args, cwd, log):
    with (cwd / log).open('w') as out:
        subprocess.run([sys.executable, '-m', 'csubst', *map(str, args)], cwd=cwd, stdout=out,
                       stderr=subprocess.STDOUT, check=True)


def simulation_arguments(work, row):
    fa = SOURCE / 'PEPC.alignment.fa'
    phylogeny = work / 'generating.tree' if row.stem_scale != 1 else SOURCE / 'PEPC.tree.nwk'
    sim_args = ['simulate', '--alignment_file', fa, '--rooted_tree_file', phylogeny,
                '--foreground', SOURCE / 'PEPC.foreground.txt', '--num_simulated_site', SITES,
                '--simulate_seed', row.seed, '--background_omega', BACKGROUND, '--foreground_omega', row.omega,
                '--percent_convergent_site', row.percent, '--convergent_amino_acids',
                'random1' if row.scenario == 'directional' else 'random0', '--percent_biased_sub', 90,
                '--export_expected_omega', 'yes', '--outdir', 'sim', '--output_prefix', 'sim']
    for ext in EXT:
        sim_args += ['--iqtree_' + ext, str(fa) + '.' + ext]
    return sim_args


def run_one(work, row, iqtree):
    rep = work / f'{row.scenario}-{row.seed}'
    rep.mkdir()
    invoke(simulation_arguments(work, row), rep, 'simulate.log')
    args = ['search', '--alignment_file', rep / 'sim/sim.fa', '--rooted_tree_file', rep / 'sim/sim_true_asr.treefile',
            '--random_seed', row.seed, '--threads', 1, '--blas_threads', 1, '--iqtree_exe', iqtree,
            '--iqtree_outdir', rep / 'iqtree', '--outdir', 'codon', '--output_stat', 'any2spe',
            '--b', 'no', '--float_digit', 17]
    invoke(args, rep, 'search.log')
    print(rep.name, flush=True)


class BranchMargins(simulate_omega.ExpectedOmega):
    """Read the actual Pyvolve kernels independently of the rate-matching calculation."""
    def __init__(self, *args):
        super().__init__(*args)
        self.stems = {n.name: int(ete.get_prop(n, 'numerical_label'))
                      for n, fg in zip(self.nodes, self.foreground) if fg}
        assert set(self.stems.values()) == set(STEMS)
        self.margin = Counter()

    def _add_model_group(self, evolver, root_model, models, num_sites):
        by_name = {model.name: model for model in models}
        nonsyn = self.aa[:, None] != self.aa
        def walk(node, prior):
            for child in node.children:
                matrix = by_name[child.model_flag] if len(models) > 1 else root_model
                if len(models) > 1:
                    assert (matrix is not root_model) == (child.name in self.stems)
                transition = evolver._exponentiate_matrix(matrix.matrix, child.branch_length * evolver.scale_tree)
                if child.name in self.stems:
                    value = prior @ (transition * nonsyn).sum(axis=1)
                    self.margin[self.stems[child.name]] += num_sites * value
                walk(child, prior @ transition)
        walk(evolver.full_tree, root_model.params['state_freqs'])

    def write(self, path, sequences=None):
        pd.Series(self.margin, name='expected_N').rename_axis('branch').to_csv(path, sep='\t')


def audit(work):
    import tempfile
    import os
    calibration = pd.read_table(work / 'calibration.tsv')
    for row in calibration[calibration.seed.eq(SEEDS[0])].itertuples():
        original = work / f'{row.scenario}-{row.seed}' / 'sim'
        with tempfile.TemporaryDirectory(dir=work) as temporary:
            before = Path.cwd()
            try:
                os.chdir(temporary)
                with patch.object(simulate_omega, 'ExpectedOmega', BranchMargins):
                    cli.main(list(map(str, simulation_arguments(work, row))))
                for name in ('sim.fa', 'sim_true_asr.anc.fa'):
                    assert sim.read_fasta(Path('sim') / name) == sim.read_fasta(original / name)
                actual = pd.read_table('sim/sim_expected_omega.tsv').set_index('branch').expected_N.loc[list(STEMS)]
                expected = [getattr(row, f'expected_N_{b}') for b in STEMS]
                np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
            finally:
                os.chdir(before)
        print('Verified replay, foreground assignments and per-stem Q expectations:', row.scenario, flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=('run', 'calibrate', 'score', 'summarize', 'audit', 'regional'))
    p.add_argument('work', type=Path)
    p.add_argument('--stem-scale', type=float, choices=(1., 4.), default=1.)
    p.add_argument('--jobs', type=int, default=5)
    p.add_argument('--iqtree', default='iqtree')
    args = p.parse_args()
    work = args.work.resolve()
    if args.mode == 'calibrate':
        calibrate(work, args.stem_scale)
    elif args.mode == 'run':
        rows = pd.read_table(work / 'calibration.tsv').itertuples()
        with ThreadPoolExecutor(args.jobs) as pool:
            list(pool.map(lambda r: run_one(work, r, args.iqtree), rows))
    elif args.mode == 'audit':
        audit(work)
    elif args.mode in ('score', 'regional'):
        from pepc_relaxation_score import run_scores
        run_scores(work, args.jobs, regional=args.mode == 'regional')
    else:
        from pepc_relaxation_score import summarize
        summarize(work)


if __name__ == '__main__':
    main()
