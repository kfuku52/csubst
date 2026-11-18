"""K=2..6 accuracy on the nine PEPC foreground stems, independent of hit selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, product
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from csubst import cli, ete, simulate_omega
from csubst.main_simulate import read_fasta

STEMS = (1, 5, 26, 33, 35, 40, 102, 103, 124)
SCENARIOS = ('baseline', 'rate_mixture', 'biased')
SEEDS = (741, 852, 963)
FITS = ('ECMK07+F', 'ECMK07_GY0K+F')
METHODS = ('codon_model', 'urn')
EXT = ('treefile', 'state', 'rate', 'iqtree', 'log')
COUNTS = ['OCNany2spe', 'OCSany2spe', 'ECNany2spe', 'ECSany2spe']
QCOLS = ['muNany2spe_sim', 'muSany2spe_sim', 'muNany2spe_null', 'muSany2spe_null']
REALIZED = ['OCNany2spe_true', 'OCSany2spe_true']
SCRIPT = Path(__file__).resolve()


class StemOracle(simulate_omega.ExpectedOmega):
    """Integrate the joint event at K independent edges, conditional on shared ancestors."""
    def __init__(self, *args):
        super().__init__(*args)
        self.stems = {node.name: int(ete.get_prop(node, 'numerical_label'))
                      for node, fg in zip(self.nodes, self.foreground) if fg}
        assert sorted(self.stems.values()) == list(STEMS)
        self.moments = {ids: np.zeros(4) for k in range(2, 7) for ids in combinations(STEMS, k)}

    def _add_model_group(self, evolver, root_model, models, num_sites):
        by_name = {m.name: m for m in models}
        aa = self.aa_membership.shape[1]
        cache = {}

        def transition(model, length):
            key = (id(model), length)
            if key not in cache:
                cache[key] = (self.identity if length <= simulate_omega.ZERO else
                              evolver._exponentiate_matrix(model.matrix, length * evolver.scale_tree))
            return cache[key]

        def visit(node):
            result = {}
            for child in node.children:
                m = by_name[child.model_flag] if len(models) > 1 else root_model
                if child.name in self.stems:
                    signal = self._event_features(transition(m, child.branch_length))
                    null = self._event_features(transition(root_model, child.branch_length))
                    branch = {(self.stems[child.name],): np.hstack((signal, null))}
                else:
                    below = visit(child)
                    if not below:
                        continue
                    t = transition(m, child.branch_length)
                    branch = {ids: t @ likelihood for ids, likelihood in below.items()}
                # Different child subtrees are independent given this node's codon.
                both = {tuple(sorted(a + b)): x * y for a, x in result.items()
                        for b, y in branch.items() if len(a) + len(b) <= 6}
                result.update(branch)
                result.update(both)
            return result

        likelihoods = visit(evolver.full_tree)
        pi = root_model.params['state_freqs']
        for ids, count in self.moments.items():
            signal, null = np.split(pi @ likelihoods[ids], 2)
            count += num_sites * np.array([signal[:aa].sum(), signal[aa:].sum(), null[:aa].sum(), null[aa:].sum()])

    def write(self, path, sequences=None):
        def states(name):
            seq = sequences[name]
            return np.array([self.codons[seq[i:i+3]] for i in range(0, len(seq), 3)])
        dest = {}
        for name, b in self.stems.items():
            src, end = states(self.parents[name]), states(name)
            dest[b] = (np.where(self.aa[src] != self.aa[end], self.aa[end], -1),
                       np.where((src != end) & (self.aa[src] == self.aa[end]), end, -1))
        rows = []
        for ids, expected in self.moments.items():
            true = []
            for channel in (0, 1):
                values = np.stack([dest[b][channel] for b in ids])
                true.append(np.count_nonzero((values[0] >= 0) & (values == values[0]).all(0)))
            rows.append([','.join(map(str, ids)), *expected, *true])
        pd.DataFrame(rows, columns=['branches', *QCOLS, *REALIZED]).to_csv(path, sep='\t', index=False)


def check_joint_oracle():
    """Check all K up to six against enumeration of ancestral codon assignments."""
    from scipy.linalg import expm
    o = StemOracle.__new__(StemOracle)
    o.aa = np.array([0, 0, 1])
    o.identity = np.eye(3)
    o.aa_membership = np.eye(2)[o.aa]
    o.synonymous = (o.aa[:, None] == o.aa) & ~np.eye(3, dtype=bool)
    o.stems = {str(i): i for i in range(6)}
    o.moments = {c: np.zeros(4) for k in range(2, 7) for c in combinations(range(6), k)}
    pi = np.array([.2, .3, .5])
    q = np.array([[-.7,.3,.4],[.2,-.6,.4],[.1,.2,-.3]])
    biased = q.copy()
    biased[0] = [-.95, .05, .9]
    root = SimpleNamespace(name='root', matrix=q, params={'state_freqs': pi})
    fg = SimpleNamespace(name='fg', matrix=biased)
    def node(name, length, flag, children=()):
        return SimpleNamespace(name=name, branch_length=length, model_flag=flag, children=children)
    leaves = [node(str(i), .1*(i+1), 'fg' if i % 2 else 'root') for i in range(6)]
    tree = node('r', 0, 'root', [node('l', .2, 'fg', leaves[:3]), node('h', .4, 'root', leaves[3:])])
    o._add_model_group(SimpleNamespace(full_tree=tree, scale_tree=1, _exponentiate_matrix=lambda m,t: expm(t*m)),
                       root, [root, fg], 1)
    trans = [expm(length*m) for length,m in [(.2,biased),(.4,q)]]
    for ci, generator in enumerate(('signal','null')):
        features = [o._event_features(expm(node.branch_length * (
            biased if generator == 'signal' and node.model_flag == 'fg' else q))) for node in leaves]
        for branches, counts in o.moments.items():
            exact = np.zeros(5)
            for r, left, right in product(range(3), repeat=3):
                weight = pi[r] * trans[0][r,left] * trans[1][r,right]
                exact += weight * np.prod([features[b][left if b<3 else right] for b in branches], axis=0)
            np.testing.assert_allclose(counts[ci*2:ci*2+2], [exact[:2].sum(), exact[2:].sum()],
                                       rtol=1e-12, atol=0)


def replay(panel, scenario, seed):
    fit = panel / 'PEPC/inputs/PEPC.alignment.fa'
    rep = panel / f'{scenario}-{seed}/sim'
    argv = ['simulate', '--alignment_file', str(fit), '--rooted_tree_file', str(fit.parent / 'PEPC.tree.nwk'),
            '--foreground', str(fit.parent / 'foreground.txt'), '--num_simulated_site', '520',
            '--simulate_seed', str(seed), '--background_omega', (panel/'background-omega.txt').read_text().strip(),
            '--foreground_omega', (panel/'background-omega.txt').read_text().strip(),
            '--percent_convergent_site', '20' if scenario == 'biased' else '0',
            '--convergent_amino_acids', 'random1' if scenario == 'biased' else 'random0',
            '--percent_biased_sub', '90', '--simulate_asrv', 'file' if scenario == 'rate_mixture' else 'no',
            '--export_expected_omega', 'yes', '--outdir', 'replay', '--output_prefix', 'sim']
    for ext in EXT:
        path = fit.parent/'synthetic.rate' if ext == 'rate' and scenario == 'rate_mixture' else str(fit)+'.'+ext
        argv.extend(['--iqtree_' + ext, str(path)])
    with patch.object(simulate_omega, 'ExpectedOmega', StemOracle):
        cli.main(argv)
    for name in ('sim.fa', 'sim_true_asr.anc.fa'):
        assert read_fasta(str(rep/name)) == read_fasta(str(Path('replay')/name))
    oracle = pd.read_csv('replay/sim_expected_omega.tsv', sep='\t').set_index('branches')
    ref = pd.read_csv(rep/'sim_expected_omega.tsv', sep='\t')
    pair = ref.loc[ref.foreground_1.eq(1) & ref.foreground_2.eq(1)].copy()
    assert len(pair) == 36
    pair.index = pair.branch_id_1.astype(str) + ',' + pair.branch_id_2.astype(str)
    np.testing.assert_allclose(oracle.loc[pair.index, QCOLS], pair[QCOLS], rtol=1e-11, atol=1e-16)
    np.testing.assert_array_equal(oracle.loc[pair.index, REALIZED], pair[REALIZED])


def search(panel, case, base, method, blind):
    fit, = (panel/case/base/'iqtree').glob('*.iqtree')
    sim = panel/case/'sim'
    opts = [] if blind else ['--foreground', str(panel/'PEPC/inputs/foreground.txt'), '--exhaustive_until', '1',
                              '--fg_exclude_wg', 'no', '--cutoff_stat', 'OCNany2spe,0']
    argv = ['search', '--alignment_file', str(sim/'sim.fa'), '--rooted_tree_file', str(sim/'sim_true_asr.treefile'),
            '--expectation_method', method, '--max_arity', '6', '--exclude_sister_pair', 'no', '--b', 'no',
            '--output_stat', 'any2spe', '--float_digit', '17', '--random_seed', case.rsplit('-', 1)[1],
            '--threads', '1', '--blas_threads', '1', '--outdir', 'out', *opts]
    for ext in EXT:
        argv.extend(['--iqtree_' + ext, str(fit.with_suffix('.' + ext))])
    cli.main(argv)


def runner(panel, work, jobs):
    tasks = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            case = f'{scenario}-{seed}'
            tasks.append((work/case, ['oracle', str(panel), scenario, str(seed)]))
            for fit in FITS:
                for method in METHODS:
                    for mode in (('target', 'blind') if fit == FITS[1] else ('target',)):
                        tasks.append((work/case/fit/method/mode, [mode, str(panel), case, fit, method]))
    def call(task):
        wd, args = task
        wd.mkdir(exist_ok=True, parents=True)
        with (wd/'job.log').open('w') as log:
            subprocess.run([sys.executable, str(SCRIPT), *args], cwd=wd, stdout=log, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(call, tasks))


def read_cb(out, k):
    m = json.loads((out/'csubst_search_run.json').read_text())
    assert m['status'] == 'complete'
    name = f'csubst_cb_{k}.tsv'
    if name not in m['outputs']:
        return None
    def number(x):
        return np.nan if x == 'NA' else float(x)
    df = pd.read_csv(out/name, sep='\t', converters={key: number for key in COUNTS})
    ids = df[[f'branch_id_{i+1}' for i in range(k)]]
    df.index = ids.astype(str).agg(','.join, axis=1)
    assert df.index.is_unique
    return df


def summarize(work):
    check_joint_oracle()
    rows, blind = [], []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            rep = work/f'{scenario}-{seed}'
            oracle = pd.read_csv(rep/'replay/sim_expected_omega.tsv', sep='\t').set_index('branches')
            for fit in FITS:
                codon, urn = (rep/fit/m/'target/out' for m in METHODS)
                meta = [json.loads((p/'csubst_endpoint_model.json').read_text())['codon'] for p in (codon, urn)]
                for key in ('parameter_source', 'rates', 'weights', 'branch_lengths'):
                    assert meta[0][key] == meta[1][key]
                prov = json.loads((urn/'csubst_urn_provenance.json').read_text())
                assert prov['asrv'] == 'empirical_all' and len(prov['training_branch_ids']) == 138
                for k in range(2, 7):
                    groups = [','.join(map(str, ids)) for ids in combinations(STEMS, k)]
                    a, b = (read_cb(out, k).sort_index() for out in (codon, urn))
                    assert set(a.index) == set(b.index) == set(groups)
                    np.testing.assert_allclose(a[COUNTS[:2]], b[COUNTS[:2]], rtol=1e-9, atol=2e-16)
                    q = oracle.loc[a.index]
                    assert (q[QCOLS] > 0).all().all()
                    ref = (q[QCOLS[0]]/q[QCOLS[2]])/(q[QCOLS[1]]/q[QCOLS[3]])
                    ok = np.isfinite(a.omegaCany2spe) & np.isfinite(b.omegaCany2spe)
                    positive = ok & (a[COUNTS[:2]] >= 1e-9).all(axis=1) & (b[COUNTS[:2]] >= 1e-9).all(axis=1)
                    for method, cb in zip(METHODS, (a, b)):
                        rows.append(dict(scenario=scenario, seed=seed, fit=fit, method=method, k=k, n=len(q),
                            finite=int(ok.sum()), positive=int(positive.sum()),
                            sse_finite=((np.log1p(cb.omegaCany2spe[ok])-np.log1p(ref[ok]))**2).sum(),
                            true_s_positive=int(q[REALIZED[1]].gt(0).sum()), ocn_ge2=int(cb.OCNany2spe.ge(2).sum()),
                            **cb[COUNTS].sum().to_dict(), **q[QCOLS + REALIZED].sum().to_dict(),
                            abs_err_n=(cb[COUNTS[2]]-q[QCOLS[2]]).abs().sum(),
                            abs_err_s=(cb[COUNTS[3]]-q[QCOLS[3]]).abs().sum()))
                        if fit != FITS[1]:
                            continue
                        scan = read_cb(rep/fit/method/'blind/out', k)
                        passed = set() if scan is None else set(scan.index[(scan.OCNany2spe >= 2) & (scan.omegaCany2spe >= 5)])
                        blind.append(dict(scenario=scenario, seed=seed, method=method, k=k,
                            evaluated=0 if scan is None else len(scan), passed=len(passed),
                            all_stems=len(passed.intersection(groups)), possible_all_stems=len(groups)))
    df = pd.DataFrame(rows)
    df.to_csv(work/'scores.tsv', sep='\t', index=False)
    pd.DataFrame(blind).to_csv(work/'blind.tsv', sep='\t', index=False)
    d = df.drop(columns='seed').groupby(['scenario', 'fit', 'method', 'k']).sum()
    d['ecn_ratio'] = d[COUNTS[2]] / d[QCOLS[2]]
    d['ecs_ratio'] = d[COUNTS[3]] / d[QCOLS[3]]
    d['ecn_wape'] = d.abs_err_n / d[QCOLS[2]]
    d['ecs_wape'] = d.abs_err_s / d[QCOLS[3]]
    with np.errstate(divide='ignore', invalid='ignore'):
        d['q_omega'] = (d[QCOLS[0]]/d[QCOLS[2]])/(d[QCOLS[1]]/d[QCOLS[3]])
        d['omega_pooled'] = (d[COUNTS[0]]/d[COUNTS[2]])/(d[COUNTS[1]]/d[COUNTS[3]])
        d['omega_with_true_observed_means'] = d.q_omega * d.ecs_ratio / d.ecn_ratio
        d['omega_rmse_finite'] = np.sqrt(d.sse_finite / d.finite)
    d.to_csv(work/'pooled.tsv', sep='\t')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.5), sharex=True, sharey=True, layout='constrained')
    for col, (scenario, title) in enumerate(zip(SCENARIOS, ('Unit rate', 'Mixed rates', 'Directional bias'))):
        axes[0,col].set_title(title)
        for row, field in enumerate(('ecn_ratio','ecs_ratio')):
            ax = axes[row,col]
            if not col:
                ax.set_ylabel(('ECN','ECS')[row] + ' / generating null')
            if row:
                ax.set_xlabel('Number of branches (K)')
            ax.axhline(1, ls=':', color='0.4')
            for method, label, color, marker in zip(METHODS, ('Codon Q','All-branch urn'), ('#146487','#be561f'), ('o','s')):
                vals = d.loc[(scenario,FITS[1],method), field]
                ax.plot(vals.index, vals, marker=marker, color=color, label=label)
            ax.set_xticks(range(2,7))
            ax.set_ylim(.20,1.65)
    axes[1,0].legend(loc='lower left', frameon=False, fontsize=8)
    fig.savefig(work/'accuracy.png', dpi=180)
    plt.close(fig)


def real(work, inputs):
    # One warmup and three timed runs per method, alternating order. macOS time -l.
    work.mkdir(parents=True, exist_ok=True)
    fit = inputs/'PEPC.alignment.fa'
    for rep in range(4):
        for method in (METHODS if rep % 2 == 0 else METHODS[::-1]):
            name = f'real-{method}-{rep}'
            args = ['search', '--alignment_file', str(fit), '--rooted_tree_file', str(inputs/'PEPC.tree.nwk'),
                    '--iqtree_model', 'ECMK07+F+R4', '--threads', '1', '--blas_threads', '1',
                    '--random_seed', '8', '--float_digit', '17', '--max_arity', '10', '--b', 'no',
                    '--outdir', name, '--expectation_method', method]
            for ext in EXT:
                args += ['--iqtree_'+ext, str(fit)+'.'+ext]
            with (work/(name+'.log')).open('w') as log:
                subprocess.run(['/usr/bin/time', '-l', sys.executable, '-m', 'csubst', *args],
                               cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)


def real_scores(work):
    rows, timings = [], []
    passes = {}
    for method in METHODS:
        for rep in range(4):
            p = work/f'real-{method}-{rep}'
            log = p.with_suffix('.log').read_text()
            timings.append(dict(method=method, rep=rep, seconds=float(re.search(r'([\d.]+) real', log)[1]),
                                rss_mib=int(re.search(r'(\d+)  maximum resident set size', log)[1])/2**20))
            for k in range(2, 7):
                cb = read_cb(p, k)
                keep = (cb.OCNany2spe >= 2) & (cb.omegaCany2spe >= 5)
                ids = set(cb[keep].index)
                if not rep:
                    passes[method,k] = ids
                    rows.append(dict(method=method, k=k, tested=len(cb), passed=len(ids),
                                     all_stems=sum(set(map(int, key.split(','))) <= set(STEMS) for key in ids)))
                else:
                    ref = read_cb(work/f'real-{method}-0', k)
                    pd.testing.assert_frame_equal(cb[COUNTS+['omegaCany2spe']], ref[COUNTS+['omegaCany2spe']],
                                                  check_exact=True)
    for k in range(2, 7):
        a, b = (read_cb(work/f'real-{method}-0', k) for method in METHODS)
        common = a.index.intersection(b.index)
        np.testing.assert_allclose(a.loc[common,COUNTS[:2]], b.loc[common,COUNTS[:2]], rtol=1e-9, atol=1e-13)
    for row in rows:
        row['intersection'] = len(set.intersection(*(passes[m,row['k']] for m in METHODS)))
    pd.DataFrame(rows).to_csv(work/'real.tsv', sep='\t', index=False)
    pd.DataFrame(timings).to_csv(work/'timing.tsv', sep='\t', index=False)


if __name__ == '__main__':
    if sys.argv[1] == 'real':
        real(Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
        real_scores(Path(sys.argv[2]))
    elif sys.argv[1] == 'real-scores':
        real_scores(Path(sys.argv[2]))
    elif sys.argv[1] == 'score':
        summarize(Path(sys.argv[2]))
    elif sys.argv[1] == 'oracle':
        replay(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
    elif sys.argv[1] in ('target', 'blind'):
        search(Path(sys.argv[2]), *sys.argv[3:], blind=sys.argv[1] == 'blind')
    else:
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument('panel', type=Path)
        p.add_argument('work', type=Path)
        p.add_argument('--jobs', type=int, default=4)
        a = p.parse_args()
        runner(a.panel.resolve(), a.work.resolve(), a.jobs)
        summarize(a.work.resolve())
