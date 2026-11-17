"""Compare joint and factorized event-count posteriors against simulated truth."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from csubst import endpoint, endpoint_io, ete, fitted_model, genetic_code, parser_iqtree, parser_misc, sequence, tree
from csubst.main_simulate import get_background_Q, get_nonsynonymous_codon_substitution_index
from csubst.main_simulate import get_synonymous_codon_substitution_index
from csubst._vendor.pyvolve.evolver import ZERO

from endpoint_posterior import count_summary, pair_paths, pair_probabilities
from pilot import EXPECTED, KEY, NULL, OBS, PANEL_REPORT, TRUTH, support_bin


PANEL = {
    'PGK': ('baseline', 'short_alignment', 'long_branches', 'rate_mixture'),
    'PEPC': ('baseline', 'long_branches', 'rate_mixture'),
}
BIAS_PANEL = {
    'PGK': ('biased', 'long_branches_bias', 'rate_mixture_bias'),
    'PEPC': ('biased',),
}
METHODS = ('factorized', 'shared_rate', 'joint_fitted', 'joint_oracle')


def fitted_generator(rep, report, metadata, alignment=None):
    g = {
        'alignment_file': str(rep / 'sim/sim.fa' if alignment is None else alignment), 'float_type': np.float64,
        'codon_table': genetic_code.get_codon_table(1),
        **{f'path_iqtree_{ext}': str(report.with_suffix('.' + ext)) for ext in ('state', 'log', 'iqtree')},
    }
    parser_iqtree.read_state_header(g)
    parser_iqtree.read_iqtree(g)
    parser_iqtree.read_log(g)
    codons = g['codon_orders']
    g['exchangeability_matrix'] = parser_misc.read_exchangeability_matrix(
        'substitution_matrix/ECMunrest.dat', codons,
    )
    fitted_model.prepare(g)
    if metadata['parameter_source']['frequency_source'] == 'verbose_log_or_FQ':
        # The original panel used raw log frequencies before checkpoint
        # support was added. Replay the actual saved generator in that case.
        logs = report.with_suffix('.log').read_text()
        raw = re.findall(r'^Empirical state frequencies:\s*([^\n]+)', logs, re.MULTILINE)[-1]
        lookup = dict(zip(sorted(codons), map(float, raw.split()), strict=True))
        pi = np.array([lookup[c] for c in codons])
        pi /= pi.sum()
        g['equilibrium_frequency'] = pi
        g['instantaneous_codon_rate_matrix'] = parser_misc.get_ecm_omega_rate_matrix(g)
    q, pi = g['instantaneous_codon_rate_matrix'], g['equilibrium_frequency']
    digest = hashlib.sha256()
    for value in (q, pi):
        digest.update(np.asarray(value, dtype='<f8').tobytes())
    if digest.hexdigest() != metadata['parameter_source']['generator_sha256']:
        raise ValueError(f'Fitted generator does not match the saved run in {report}.')
    code = {codon: aa for aa, codon in g['codon_table']}
    amino_acids = np.unique([code[c] for c in codons], return_inverse=True)[1]
    return q, pi, codons, amino_acids


def oracle_generator(rep, codons):
    dataset = rep.parent.name
    bundle = rep.parent / 'inputs' / f'{dataset}.alignment.fa.iqtree'
    original = parser_iqtree.read_iqtree(dict(
        path_iqtree_iqtree=str(bundle), codon_orders=codons, float_type=np.float64,
    ))
    g = dict(codon_table=genetic_code.get_codon_table(1), background_omega=.2,
             eq_freq=original['equilibrium_frequency'], pyvolve_codon_orders=codons,
             substitution_model=original['substitution_model'])
    g['all_syn_cdn_index'] = get_synonymous_codon_substitution_index(g, codons)
    g['all_nsy_cdn_index'] = get_nonsynonymous_codon_substitution_index(g['all_syn_cdn_index'])
    q = get_background_Q(g, 'auto')
    return q, g['eq_freq']


def select_pairs(frame, parents, directional_bias):
    # The set is fixed across replicates and uses only Q exposure and topology.
    candidates = frame.sort_values(KEY).copy()
    paths = [pair_paths(parents, pair) for pair in candidates[KEY].itertuples(index=False, name=None)]
    candidates['split_distance'] = [len(a) + len(b) for _, a, b in paths]
    candidates['s_bin'] = support_bin(candidates[NULL[1]])
    candidates['relationship'] = np.where(candidates.split_distance == 2, 'sisters', 'different_parents')
    candidates['foreground_pair'] = candidates.foreground_1.eq(1) & candidates.foreground_2.eq(1)
    rng = np.random.default_rng(20261116)
    selected = [
        frame.iloc[np.sort(rng.choice(len(frame), size=min(8, len(frame)), replace=False))]
        for _, frame in candidates.groupby(['s_bin', 'relationship'], observed=True, sort=True)
    ]
    if directional_bias:
        selected.append(candidates.loc[candidates.foreground_pair])
    return pd.concat(selected, ignore_index=True).drop_duplicates(KEY)


def load_tips(rep, codons, tr):
    names = {int(ete.get_prop(node, 'numerical_label')): node.name for node in ete.iter_leaves(tr)}
    seqs = sequence.read_fasta(rep / 'sim/sim.fa')
    lookup = {codon: i for i, codon in enumerate(codons)}
    one_hot = np.eye(len(codons))
    return {i: one_hot[[lookup[seqs[name][j:j+3]] for j in range(0, len(seqs[name]), 3)]]
            for i, name in names.items()}


def oracle_posteriors(rep, codons, parents, lengths, tips, amino_acids, pairs, references):
    q, pi = oracle_generator(rep, codons)
    actual_rates = pd.read_csv(rep / 'sim/sim_true_asr.rate', sep='\t')['C_Rate'].to_numpy()
    scores = np.empty((len(pairs), len(actual_rates), 2))
    means = np.zeros((len(pairs), 2))
    for rate in np.unique(actual_rates):
        mask = actual_rates == rate
        model = endpoint.EndpointModel(parents, np.where(lengths <= ZERO, 0., lengths), q, pi, [rate], [1.])
        restricted = {node: states[mask] for node, states in tips.items()}
        scores[:, mask] = pair_probabilities(model, restricted, amino_acids, pairs)[2]
        unobserved = {node: np.ones((1, len(pi))) for node in tips}
        means += mask.sum() * pair_probabilities(model, unobserved, amino_acids, pairs)[2, :, 0]
    np.testing.assert_allclose(means, references, rtol=5e-8, atol=1e-9)
    return scores, float(np.max(np.abs(means - references)))


def analyze(job):
    rep, recovered_fit, directional_bias = job
    if recovered_fit is None:
        search_dir = rep / 'default'
        report_dir = rep / 'iqtree-default'
    else:
        search_dir = recovered_fit / 'csubst_search'
        report_dir = recovered_fit / 'csubst_iqtree'
    run_record = json.loads((search_dir / 'csubst_search_run.json').read_text())
    if run_record['status'] != 'complete':
        raise ValueError(f'Incomplete fit in {search_dir}.')
    metadata = json.loads((search_dir / 'csubst_endpoint_model.json').read_text())['codon']
    report, = report_dir.glob('*.iqtree')
    q, pi, codons, amino_acids = fitted_generator(rep, report, metadata)
    tr = tree.add_numerical_node_labels(ete.PhyloNode(str(rep / 'sim/sim_true_asr.treefile'), format=1))
    parents, _ = endpoint_io._tree_arrays({'tree': tr}, structural=False)
    scenario, seed = rep.name.rsplit('-', 1)
    tips = load_tips(rep, codons, tr)
    sim = pd.read_csv(rep / 'sim/sim_expected_omega.tsv', sep='\t')
    search = pd.read_csv(search_dir / 'csubst_cb_2.tsv', sep='\t')
    columns = [*KEY, *OBS, *EXPECTED, 'omegaCany2spe']
    data = sim.merge(search[columns], on=KEY, how='outer', validate='one_to_one', indicator=True)
    if not data.pop('_merge').eq('both').all():
        raise ValueError(f'Invalid pair mapping in {rep}.')
    names = {int(ete.get_prop(node, 'numerical_label')): node.name for node in tr.traverse()}
    for suffix in (1, 2):
        assert data[f'branch_id_{suffix}'].map(names).eq(data[f'branch_name_{suffix}']).all()
    pairs = select_pairs(data.dropna(subset=EXPECTED), parents, directional_bias)
    ids = pairs[KEY].to_numpy()
    model = endpoint.EndpointModel(parents, metadata['branch_lengths'], q, pi, metadata['rates'], metadata['weights'])
    fitted = pair_probabilities(model, tips, amino_acids, ids)
    production = pairs[OBS].to_numpy()
    np.testing.assert_allclose(fitted[0].sum(axis=1), production, rtol=1e-9, atol=6e-12)
    observed_error = float(np.max(np.abs(fitted[0].sum(axis=1) - production)))

    models = list(fitted)
    prior_error = np.nan
    if not directional_bias:
        if not data['omegaCany2spe_expected'].eq(1.).all():
            raise ValueError(f'Expected homogeneous branch Q in {rep}.')
        original = rep.parent / 'inputs' / f'{rep.parent.name}.tree.nwk'
        simulation_tree = tree.add_numerical_node_labels(ete.PhyloNode(str(original), format=1))
        original_parents, lengths = endpoint_io._tree_arrays({'tree': simulation_tree}, structural=False)
        np.testing.assert_array_equal(original_parents, parents)
        # The saved simulation tree rounds lengths when writing Newick. Use
        # the same multiplication as simulate, before serialization.
        lengths *= 2.5 if scenario == 'long_branches' else 1.
        oracle, prior_error = oracle_posteriors(
            rep, codons, parents, lengths, tips, amino_acids, ids, pairs[NULL].to_numpy(),
        )
        models.append(oracle)
    rng = np.random.default_rng(840000 + int(seed))
    ranks = rng.uniform(size=(len(pairs), 2))
    rows = []
    for method, probabilities in zip(METHODS, models):
        for rank, values, pair in zip(ranks, probabilities, pairs.to_dict('records')):
            truth = tuple(int(pair[key]) for key in TRUTH)
            result = count_summary(values, truth, rank)
            ratio_scale = pair[NULL[1]] / pair[NULL[0]]
            result['ratio_lower'] *= ratio_scale
            result['ratio_upper'] *= ratio_scale
            rows.append(dict(
                dataset=rep.parent.name, scenario=scenario, seed=int(seed), method=method,
                **{key: pair[key] for key in (*KEY, *NULL, *OBS, *TRUTH)},
                relationship=pair['relationship'], s_bin=pair['s_bin'], split_distance=pair['split_distance'],
                foreground_pair=pair['foreground_pair'],
                search_omega=pair['omegaCany2spe'],
                production_replay_max_error=observed_error, oracle_prior_max_error=prior_error,
                **result,
            ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    parser.add_argument('recovered', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--biased', action='store_true')
    args = parser.parse_args()
    repairs = pd.read_csv(PANEL_REPORT / 'omega_recovery_by_seed.tsv', sep='\t')
    repaired = {f'{r.scenario}-{r.seed}' for r in repairs.itertuples(index=False)}
    jobs = []
    panel = BIAS_PANEL if args.biased else PANEL
    for dataset, scenarios in panel.items():
        for scenario in scenarios:
            reps = sorted((args.panel / dataset).glob(f'{scenario}-[0-9]*'))
            if len(reps) != (4 if dataset == 'PGK' else 3):
                raise ValueError(f'Missing replicates for {dataset}/{scenario}.')
            for rep in reps:
                alternate = (args.recovered / rep.name) if dataset == 'PGK' and rep.name in repaired else None
                jobs.append((rep, alternate, args.biased))
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = []
        for (rep, _, _), result in zip(jobs, pool.map(analyze, jobs)):
            rows.extend(result)
            print(f'{rep.parent.name}/{rep.name}: {len(result) // (3 if args.biased else 4)} pairs', flush=True)
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep='\t', index=False, float_format='%.10g', na_rep='NA')


if __name__ == '__main__':
    main()
