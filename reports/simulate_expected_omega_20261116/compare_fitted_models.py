"""Compare fixed and fitted ECMK07 on six PEPC alignments, with two exploratory GY fits."""

import argparse
import gzip
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.special import logsumexp

from csubst import endpoint_io, ete, fitted_model, genetic_code, parser_iqtree, parser_misc
from csubst.main_simulate import read_fasta


PAIR = ['branch_id_1', 'branch_id_2']
COUNTS = ['OCNany2spe', 'OCSany2spe', 'ECNany2spe', 'ECSany2spe']
NULL = ['muNany2spe_null', 'muSany2spe_null']
MODELS = (
    ('ECMK07+F+R4', 'search', 'iqtree'),
    ('GY+F+R4', 'search', 'iqtree'),
    ('ECMK07_GY0K+F+R4', 'hybrid_search', 'hybrid_fit'),
)
GY_RUNS = frozenset(('biased-101', 'biased-202'))


def replay_likelihood(tr, codons, sequences, q, pi, rates, weights):
    """Felsenstein pruning from IQ-TREE's raw tree and full-precision categories."""
    lookup = {codon: index for index, codon in enumerate(codons)}
    eye = np.eye(len(codons))
    tips = {name: eye[[lookup[seq[i:i+3]] for i in range(0, len(seq), 3)]]
            for name, seq in sequences.items()}
    num_sites = len(next(iter(tips.values())))
    site_likelihoods = []
    for rate in rates:
        scale = np.zeros(num_sites)
        active = {}
        transitions = {}
        for node in tr.traverse('postorder'):
            if ete.is_leaf(node):
                active[id(node)] = tips[node.name]
                continue
            likelihood = np.ones((num_sites, len(codons)))
            for child in ete.get_children(node):
                if child.dist not in transitions:
                    transitions[child.dist] = expm(q * (rate * child.dist))
                likelihood *= active.pop(id(child)) @ transitions[child.dist].T
            row_sum = likelihood.sum(axis=1)
            active[id(node)] = likelihood / row_sum[:, None]
            scale += np.log(row_sum)
        site_likelihoods.append(np.log(active[id(tr)] @ pi) + scale)
    return logsumexp(site_likelihoods, axis=0, b=weights[:, None]).sum()


def diagnose_fit(model, report, alignment, endpoint_manifest):
    codon_table = genetic_code.get_codon_table(1)
    paths = {f'path_iqtree_{extension}': str(report.with_suffix('.' + extension))
             for extension in ('state', 'iqtree', 'log')}
    context = dict(paths, codon_table=codon_table, alignment_file=str(alignment), float_type=np.float64)
    parser_iqtree.read_state_header(context)
    parser_iqtree.read_iqtree(context)
    parser_iqtree.read_log(context)
    if context['substitution_model'] != model:
        raise ValueError(f'Unexpected model in {report}')
    codons = context['codon_orders']
    if model.startswith('ECMK07'):
        context['exchangeability_matrix'] = parser_misc.read_exchangeability_matrix(
            'substitution_matrix/ECMunrest.dat', codons,
        )
    if model.startswith('ECMK07+'):
        context['instantaneous_codon_rate_matrix'] = parser_misc.exchangeability2Q(
            context['exchangeability_matrix'], context['equilibrium_frequency'],
        )
    elif model.startswith('GY+'):
        context['instantaneous_codon_rate_matrix'] = parser_misc.get_mechanistic_instantaneous_rate_matrix(context)
    fitted_model.prepare(context)
    provenance = context['fitted_model_provenance']
    if provenance['generator_sha256'] != endpoint_manifest['parameter_source']['generator_sha256']:
        raise ValueError(f'Reconstructed Q and pi differ from {report}')
    q, pi = context['instantaneous_codon_rate_matrix'], context['equilibrium_frequency']
    rates, weights = endpoint_io.model_rates(context)
    checkpoint = report.with_suffix('.ckp.gz')
    if checkpoint.exists():
        with gzip.open(checkpoint, 'rt') as handle:
            mixture = re.search(r'^RateFree4:\n((?:[ \t].*\n)+)', handle.read(), re.MULTILINE)
        if mixture is None:
            raise ValueError(f'Missing precise FreeRate parameters in {checkpoint}')
        rates, weights = (
            np.fromstring(re.search(r'^\s+' + key + r':(.*)$', mixture[1], re.MULTILINE)[1], sep=',')
            for key in ('rates', 'prop')
        )
    tr = ete.PhyloNode(str(report.with_suffix('.treefile')), format=1)
    reproduced = replay_likelihood(tr, codons, read_fasta(alignment), q, pi, rates, weights)
    reported = float(re.search(r'Log-likelihood of the tree:\s*(\S+)', report.read_text())[1])
    diff = reproduced - reported
    if model.startswith('ECMK07_GY0K') and abs(diff) > 0.02:
        raise ValueError(f'Hybrid likelihood mismatch ({diff:.6g}) for {report}')
    code = {codon: aa for aa, codon in codon_table}
    aas = np.array([code[codon] for codon in codons])
    return dict(
        model_omega=context['omega'], kappa=context['kappa'],
        stationary_N_fraction=(pi[:, None] * q)[aas[:, None] != aas].sum(),
        fitted_tree_length=np.sum(endpoint_manifest['branch_lengths']),
        iqtree_loglikelihood=reported, reconstructed_loglikelihood=reproduced,
        likelihood_delta=diff, precise_rate_categories=checkpoint.exists(),
    )


def compare_replicate(original, refits, scenario_tag, seed):
    run = f'{scenario_tag}-{seed}'
    scenario = 'background' if scenario_tag == 'null' else scenario_tag
    original, refits = original / run, refits / run
    alignment = original / 'sim/sim.fa'
    sim = pd.read_csv(original / 'sim/sim_expected_omega.tsv', sep='\t')
    sim['foreground_branches'] = sim.foreground_1 + sim.foreground_2
    counts, fits = [], []
    mask = None
    for model, search_dir, fit_dir in MODELS:
        if model.startswith('GY+') and run not in GY_RUNS:
            continue
        model_dir = original if model.startswith('ECMK07+') else refits
        search_dir = model_dir / search_dir
        manifest = json.loads((search_dir / 'csubst_search_run.json').read_text())
        if manifest['status'] != 'complete':
            raise ValueError(f'Incomplete search in {search_dir}')
        endpoint = json.loads((search_dir / 'csubst_endpoint_model.json').read_text())['codon']
        observed = pd.read_csv(search_dir / 'csubst_cb_2.tsv', sep='\t')
        data = sim.merge(observed[PAIR + COUNTS], on=PAIR, validate='one_to_one', how='outer', indicator=True)
        if not data.pop('_merge').eq('both').all():
            raise ValueError(f'Mismatched branch pairs in {search_dir}')
        reported = data[COUNTS].notna().all(axis=1).to_numpy()
        if mask is None:
            mask = reported
        else:
            np.testing.assert_array_equal(reported, mask)
        report, = (model_dir / fit_dir).glob('*.iqtree')
        labels = dict(scenario=scenario, seed=seed, model=model)
        fits.append(dict(**labels, **diagnose_fit(model, report, alignment, endpoint)))
        for foreground_branches, frame in data.loc[mask].groupby('foreground_branches'):
            totals = frame[COUNTS + NULL].sum().to_dict()
            counts.append(dict(**labels, foreground_branches=foreground_branches, pairs=len(frame), **totals))
    return counts, fits


def pool_counts(by_seed):
    return by_seed.groupby(['scenario', 'model', 'foreground_branches'], as_index=False)[
        ['pairs', *COUNTS, *NULL]
    ].sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('refits', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    counts, fits = [], []
    for scenario in ('null', 'biased'):
        for seed in (101, 202, 303):
            run_counts, run_fits = compare_replicate(args.original, args.refits, scenario, seed)
            counts.extend(run_counts)
            fits.extend(run_fits)
    by_seed = pd.DataFrame(counts)
    pooled = pool_counts(by_seed.loc[~by_seed.model.str.startswith('GY+')])
    matched = (by_seed.scenario + '-' + by_seed.seed.astype(str)).isin(GY_RUNS)
    gy_matched = pool_counts(by_seed.loc[matched])
    for frame in (by_seed, pooled, gy_matched):
        frame['pooled_omegaC'] = (frame.OCNany2spe / frame.ECNany2spe) / (frame.OCSany2spe / frame.ECSany2spe)
        frame['ECN_over_null'] = frame.ECNany2spe / frame.muNany2spe_null
        frame['ECS_over_null'] = frame.ECSany2spe / frame.muSany2spe_null
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in (
        ('codon_model_comparison.tsv', pooled), ('codon_model_comparison_by_seed.tsv', by_seed),
        ('codon_model_gy_matched.tsv', gy_matched),
        ('codon_model_fit_diagnostics.tsv', pd.DataFrame(fits)),
    ):
        table.to_csv(args.output / name, sep='\t', index=False, float_format='%.12g')


if __name__ == '__main__':
    main()
