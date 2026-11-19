"""Measure recovery of distance-dependent convergence on a PEPC scaffold."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

from csubst import ete, genetic_code, parser_iqtree, parser_misc, tree
from csubst.main_simulate import get_pyvolve_codon_order, write_fasta

from compare_empirical_urn import SEARCH, prepare, run_cli
from pilot import EXPECTED, KEY, OBS
from urn_accuracy import load_search


CONDITIONS = ('neutral', 'shared', 'inverse')
SEEDS = (1013, 2027, 3037, 4051, 5059, 6073, 7091, 8101, 9113,
         10103, 11113, 12101, 13109, 14107, 15107, 16103, 17107, 18119)
SITES = 1200
FOCAL_TIPS = 24
TERMINAL_LENGTH = .08
D0 = .18
PREFERRED_RATE_MULTIPLIER = 8.


def setup(work, reuse=False):
    sources = work / 'PEPC/inputs' if reuse else prepare(work, 'PEPC')
    root = ete.PhyloNode(str(sources / 'PEPC.tree.nwk'), format=1)
    tips = [n for n in ete.iter_leaves(root) if not ete.is_root(n.up)]
    focal = {tips[i].name for i in np.linspace(0, len(tips) - 1, FOCAL_TIPS, dtype=int)}
    for n in tips:
        if n.name in focal:
            n.dist = TERMINAL_LENGTH
    path = work / 'tree.nwk'
    if not reuse:
        ete.write_tree(root, format=1, outfile=str(path))
    source = sources / 'PEPC.alignment.fa.iqtree'
    codons = get_pyvolve_codon_order()
    g = parser_iqtree.read_iqtree(dict(path_iqtree_iqtree=str(source), codon_orders=codons, float_type=np.float64))
    assert g['substitution_model'].split('+')[0] == 'ECMK07'
    pi = g['equilibrium_frequency']
    pi = pi / pi.sum()
    exchange = parser_misc.read_exchangeability_matrix('substitution_matrix/ECMunrest.dat', codons)
    q = parser_misc.exchangeability2Q(exchange, pi)
    code = {codon: aa for aa, codon in genetic_code.get_codon_table(1)}
    aa = np.unique([code[codon] for codon in codons], return_inverse=True)[1]
    np.testing.assert_allclose(pi[:, None] * q, (pi[:, None] * q).T, atol=1e-13)
    return path, focal, codons, pi, q, aa


def kernels(q, aa):
    matrices = []
    for preferred in range(aa.max() + 1):
        forward = (aa[:, None] != aa) & (aa[None, :] == preferred)
        tilted = q * np.where(forward, PREFERRED_RATE_MULTIPLIER, 1.)
        tilted -= np.diag(tilted.sum(axis=1))
        matrices.append(expm(tilted * TERMINAL_LENGTH))
    return np.asarray(matrices)


def feature_kernel(transition, aa):
    nonsyn = (aa[:, None] != aa)
    ntarget = np.eye(aa.max() + 1)[aa]
    syn = (~nonsyn) & ~np.eye(len(aa), dtype=bool)
    return ((transition * nonsyn) @ ntarget, transition * syn)


def convergence_kernel(transitions, aa):
    matrices = [feature_kernel(t, aa) for t in transitions]
    result = []
    for values in zip(*matrices):
        mean = np.mean(values, axis=0)
        result.append((mean @ mean.T, np.mean([f @ f.T for f in values], axis=0)))
    return result


def sample_states(probability, uniform):
    cdf = np.cumsum(np.maximum(probability, 0.), axis=1)
    cdf /= cdf[:, -1, None]
    return (uniform[:, None] >= cdf).sum(axis=1)


def simulate_case(job, setup_data):
    rep, condition, seed = job
    path, focal, codons, pi, q, aa = setup_data
    rep.mkdir()
    tr = tree.add_numerical_node_labels(ete.PhyloNode(str(path), format=1))
    tilted = kernels(q, aa)
    aa_count = len(tilted)
    rng = np.random.default_rng(seed)
    driver_rng = np.random.default_rng(seed + 100_000)
    rate = driver_rng.exponential(1. / D0, SITES)
    favorable = np.arange(SITES) < SITES // 2
    codon_states, preferences = {}, {}
    for node in tr.traverse('preorder'):
        node_id = int(ete.get_prop(node, 'numerical_label'))
        if ete.is_root(node):
            codon_states[node_id] = rng.choice(len(pi), size=SITES, p=pi)
            preferences[node_id] = driver_rng.integers(aa_count, size=SITES)
            continue
        parent_id = int(ete.get_prop(node.up, 'numerical_label'))
        parent = codon_states[parent_id]
        context = preferences[parent_id]
        reset = driver_rng.random(SITES) >= np.exp(-rate * float(node.dist))
        new_context = driver_rng.integers(aa_count, size=SITES)
        preferences[node_id] = np.where(reset & (condition == 'inverse'), new_context, context)
        probability = expm(q * float(node.dist))[parent]
        if node.name in focal and condition != 'neutral':
            probability[favorable] = tilted[context[favorable], parent[favorable]]
        codon_states[node_id] = sample_states(probability, rng.random(SITES))
    tips = {node.name: ''.join(codons[codon_states[int(ete.get_prop(node, 'numerical_label'))]])
            for node in ete.iter_leaves(tr)}
    write_fasta(tips, rep / 'tips.fa')

    null_kernel = convergence_kernel([expm(q * TERMINAL_LENGTH)], aa)
    alternative = convergence_kernel(tilted, aa)
    selected = sorted((n for n in ete.iter_leaves(tr) if n.name in focal),
                      key=lambda n: int(ete.get_prop(n, 'numerical_label')))
    rows = []
    for a, b in combinations(selected, 2):
        ids = [int(ete.get_prop(n, 'numerical_label')) for n in (a, b)]
        parent_ids = [int(ete.get_prop(n.up, 'numerical_label')) for n in (a, b)]
        d = ete.get_distance(a.up, b.up) if a.up is not b.up else 0.
        joint = pi[:, None] * expm(q * d)
        coupling = 1. / (1. + d / D0) if condition == 'inverse' else 1.
        equates = 1. / aa_count + (1. - 1. / aa_count) * coupling
        null_counts, generating_counts = [], []
        for (null, _), (independent, coupled) in zip(null_kernel, alternative):
            expected = SITES * (joint * null).sum()
            shift = null if condition == 'neutral' else (1. - coupling) * independent + coupling * coupled
            generating = (joint * (int(favorable.sum()) * shift + (~favorable).sum() * null)).sum()
            null_counts.append(expected)
            generating_counts.append(generating)
        pa, pb = (codon_states[i] for i in parent_ids)
        ca, cb = (codon_states[i] for i in ids)
        change_n = (aa[pa] != aa[ca]) & (aa[pb] != aa[cb]) & (aa[ca] == aa[cb])
        change_s = (aa[pa] == aa[ca]) & (aa[pb] == aa[cb]) & (pa != ca) & (pb != cb) & (ca == cb)
        observed = (change_n.sum(), change_s.sum())
        actual_match = np.mean(preferences[parent_ids[0]][favorable] == preferences[parent_ids[1]][favorable])
        rows.append([*ids, d, equates, actual_match, *null_counts, *generating_counts, *observed])
    fields = KEY + ['distance', 'expected_context_match', 'realized_context_match',
                    'null_n', 'null_s', 'generating_n', 'generating_s', 'realized_n', 'realized_s']
    pd.DataFrame(rows, columns=fields).to_csv(rep / 'truth.tsv', sep='\t', index=False, float_format='%.14g')


def search_case(job, path, iqtree):
    rep, _, seed = job
    common = ['search', '--alignment_file', rep/'tips.fa', '--rooted_tree_file', path, '--threads', 1,
              '--blas_threads', 1, '--random_seed', seed, *SEARCH]
    run_cli(rep, 'codon.stdout', [*common, '--iqtree_exe', iqtree, '--iqtree_model', 'ECMK07+F',
                                  '--iqtree_outdir', 'iqtree', '--outdir', 'codon'])
    report, = (rep / 'iqtree').glob('*.iqtree')
    fitted = [term for ext in ('iqtree', 'log', 'state', 'rate', 'treefile')
              for term in ('--iqtree_' + ext, report.with_suffix('.' + ext))]
    run_cli(rep, 'urn.stdout', [*common, *fitted, '--expectation_method', 'urn', '--asrv', 'empirical', '--outdir', 'urn'])
    print(f'Completed {rep.name}', flush=True)


def score(job):
    rep, condition, seed = job
    truth = pd.read_csv(rep/'truth.tsv', sep='\t')
    codon, m1 = load_search(rep, 'codon')
    urn, m2 = load_search(rep, 'urn')
    provenance = json.loads((rep / 'urn/csubst_urn_provenance.json').read_text())
    assert provenance['asrv'] == 'empirical'
    assert m1['parameter_source']['generator_sha256'] == m2['parameter_source']['generator_sha256']
    assert m1['branch_lengths'] == m2['branch_lengths']
    np.testing.assert_array_equal(codon[KEY], urn[KEY])
    np.testing.assert_allclose(codon[OBS], urn[OBS], rtol=1e-10, atol=6e-12)
    frames = {'truth': truth}
    for label, tab in [('codon', codon), ('urn', urn)]:
        # Reproduce this report's original secondary axis with either output schema.
        fitted_axis = 'dist_bl_legacy' if 'dist_bl_legacy' in tab else 'dist_bl'
        selected = tab[KEY + OBS + EXPECTED + [fitted_axis]].rename(columns={fitted_axis: 'dist_bl'})
        frame = truth.merge(selected, on=KEY, validate='one_to_one', how='left')
        assert frame[OBS + EXPECTED + ['dist_bl']].notna().all().all() and len(frame) == 276
        frame = frame.rename(columns=dict(zip(OBS + EXPECTED, ('ocn','ocs','ecn','ecs'))))
        frames[label] = frame
    frames['truth']['dist_bl'] = frames['codon']['dist_bl']
    rows = []
    for method, table in frames.items():
        values = (['generating_n', 'generating_s', 'null_n', 'null_s', 'realized_n', 'realized_s']
                  if method == 'truth' else ['ocn','ocs','ecn','ecs'])
        for axis in ('distance', 'dist_bl'):
            bins = pd.qcut(table[axis].rank(method='first'), 4, labels=False)
            for bin_id, part in table.groupby(bins):
                rows.append(dict(condition=condition, seed=seed, method=method, axis=axis, bin=int(bin_id)+1,
                                 n_pairs=len(part), distance_mean=part.distance.mean(), fitted_distance_mean=part.dist_bl.mean(),
                                 mean_context_match=part.realized_context_match.mean(),
                                 expected_context_match=part.expected_context_match.mean(),
                                 **part[values].sum().to_dict()))
    return rows


def plot_curves(summary, path):
    import matplotlib.pyplot as plt

    raw = summary[summary['axis'].eq('distance')]
    pooled = raw.groupby(['condition','method','bin']).sum(numeric_only=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), sharey=True, layout='constrained')
    series = (
        ('truth', 'Generating expectation', '#1c2533', '--', 'generating_n', 'generating_s', 'null_n', 'null_s'),
        ('truth', 'True ancestors', '#747f89', ':', 'realized_n', 'realized_s', 'null_n', 'null_s'),
        ('codon', 'Fitted codon Q', '#166cb3', '-', 'ocn', 'ocs', 'ecn', 'ecs'),
        ('urn', 'Empirical urn', '#af5a22', '-', 'ocn', 'ocs', 'ecn', 'ecs'),
    )
    for ax, condition, title in zip(axes, CONDITIONS, ('ECMK07 only', 'Shared preference', 'Inverse-distance preference')):
        for method, label, color, style, on, os, en, es in series:
            values = pooled.loc[condition, method]
            x = raw[raw.condition.eq(condition) & raw.method.eq(method)].groupby('bin').distance_mean.mean()
            estimate = (values[on] / values[en]) / (values[os] / values[es])
            ax.plot(x, estimate, label=label, color=color, linestyle=style, marker='o', markersize=4, linewidth=1.7)
        ax.set_title(title, fontsize=10, loc='left')
        ax.set_xlabel('True distance between parents')
        ax.yaxis.grid(True, color='#e4e7eb', linewidth=.8)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Pooled $\\omega_C$ (any2spe)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('--iqtree', default='iqtree')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--stage', choices=('all','extend','score'), default='all')
    args = parser.parse_args()
    work = args.work.resolve()
    jobs = [(work / f'{condition}-{seed}', condition, seed) for condition in CONDITIONS for seed in SEEDS]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        if args.stage != 'score':
            reuse = args.stage == 'extend'
            inputs = setup(work, reuse)
            new = [job for job in jobs if not job[0].exists()] if reuse else jobs
            for future in [pool.submit(simulate_case, job, inputs) for job in new]:
                future.result()
            for future in [pool.submit(search_case, job, inputs[0], args.iqtree) for job in new]:
                future.result()
        rows = [row for group in pool.map(score, jobs) for row in group]
    summary = pd.DataFrame(rows)
    summary.to_csv(work / 'summary.tsv', sep='\t', index=False, float_format='%.12g', na_rep='NA')
    plot_curves(summary, work / 'distance_curves.png')


if __name__ == '__main__':
    main()
