"""Score the same independent pairs using reconstructed or known simulated endpoints."""
from concurrent.futures import ThreadPoolExecutor
import json
import re
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from csubst import cli, empirical_urn, endpoint, endpoint_io, ete, main_simulate, omega_statistics
from csubst import param, parser_misc, runtime, tree
from pepc_relaxation import DOMAIN, EXT, SCENARIOS, SEEDS, SITES, STEMS

KEY = ['branch_id_1', 'branch_id_2']
COLS = ['OCNany2spe', 'OCSany2spe', 'ECNany2spe', 'ECSany2spe', 'omegaCany2spe']


def search_context(rep):
    report, = (rep / 'iqtree').glob('*.iqtree')
    assert 'Model of substitution: ECMK07_GY0K+F+R4' in report.read_text()
    args = ['search', '--alignment_file', str(rep / 'sim/sim.fa'),
            '--rooted_tree_file', str(rep / 'sim/sim_true_asr.treefile'),
            '--expectation_method', 'urn', '--outdir', 'diagnostic', '--output_stat', 'any2spe',
            '--b', 'no', '--threads', '1', '--blas_threads', '1', '--float_digit', '17',
            '--random_seed', rep.name.split('-')[1]]
    for ext in EXT:
        args += ['--iqtree_' + ext, str(report.with_suffix('.' + ext))]
    g = param.get_global_parameters(cli._build_parser().parse_args(args))
    g = runtime.ensure_output_layout(g, create_dir=True)
    # Retain the ordinary tip-derived posterior for the known-parent counterfactual.
    g['asrv_posterior_draws'] = 1
    g = parser_misc.prepare_input_context(g, include_foreground=True, include_marginal=True,
                                          resolve_state_subset=True, prepare_state=True)
    endpoint_io.prepare(g)
    metadata = json.loads((rep / 'codon/csubst_endpoint_model.json').read_text())
    assert g['asrv'] == 'empirical_all' and metadata['substitution_posterior'] == 'joint'
    for k in ('weights', 'rates', 'branch_lengths', 'parameter_source'):
        assert metadata['codon'][k] == g['_endpoint_manifest']['codon'][k]
    assert len(metadata['codon']['rates']) == 4
    return g, args


def codes_from_fasta(g, rep):
    truth = tree.add_numerical_node_labels(ete.PhyloNode(str(rep / 'sim/sim_true_asr.treefile'), format=1))
    sequences = main_simulate.read_fasta(rep / 'sim/sim.fa') | main_simulate.read_fasta(rep / 'sim/sim_true_asr.anc.fa')
    def index(n):
        return int(ete.get_prop(n, 'numerical_label'))

    by_id = {index(n): n for n in g['tree'].traverse()}
    names = sorted(truth.traverse(), key=index)
    assert set(sequences) == {n.name for n in names}
    assert all(set(ete.get_leaf_names(n)) == set(ete.get_leaf_names(by_id[index(n)])) for n in names)
    codons = np.asarray(g['codon_orders'], dtype='S3')
    order = np.argsort(codons)
    obs = np.stack([np.frombuffer(sequences[n.name].encode(), dtype='S3') for n in names])
    codes = order[np.searchsorted(codons[order], obs)]
    np.testing.assert_array_equal(codons[codes], obs)
    return codes


def true_tensors(g, codes):
    parents, _ = endpoint_io._tree_arrays(g, False)
    groups = [g['synonymous_indices'][aa] for aa in g['amino_acid_orders']]
    aa = np.empty(len(g['codon_orders']), int)
    local = aa.copy()
    for i, group in enumerate(groups):
        aa[group] = i
        local[group] = np.arange(len(group))
    parent = codes[np.where(parents < 0, np.arange(len(parents)), parents)]
    n = aa[parent] != aa[codes]
    s = ~n & (parent != codes)
    b, j = np.nonzero(n)
    nr = np.zeros((len(codes), len(groups), len(groups)))
    np.add.at(nr, (b, aa[parent[b, j]], aa[codes[b, j]]), 1)
    b, j = np.nonzero(s)
    sd = np.zeros((len(codes), len(groups), g['max_synonymous_size']))
    np.add.at(sd, (b, aa[codes[b, j]], local[codes[b, j]]), 1)
    eligible = g['_endpoint_tensors']['N'].eligible
    assert eligible[parents >= 0].all()
    def pack(mask, totals):
        return SimpleNamespace(eligible=eligible, branch_site=mask.astype(float), event_totals=totals)

    return aa, pack(n, nr), pack(s, sd)


def true_predictive(g, pairs, codes, aa):
    parents, lengths = endpoint_io._tree_arrays(g, False)
    model = endpoint.EndpointModel(parents, lengths, g['instantaneous_codon_rate_matrix'],
                                   g['equilibrium_frequency'], *endpoint_io.model_rates(g))
    weights = g['_asrv_rate_posterior']
    np.testing.assert_allclose(weights.sum(axis=0), 1, atol=1e-12)
    cross = aa[:, None] != aa
    same = ~cross & ~np.eye(len(aa), dtype=bool)
    membership = np.eye(aa.max() + 1)[aa]
    profiles = [np.zeros((*codes.shape, w)) for w in (membership.shape[1], len(aa))]
    for b in np.unique(pairs):
        src = codes[parents[b]]
        for k, weight in enumerate(weights):
            transition = model.transition(b, k)
            pn = (transition * cross) @ membership
            ps = transition * same
            for profile, values in zip(profiles, (pn, ps)):
                profile[b] += weight[:, None] * values[src]
    a, b = pairs.T
    result = []
    for profile in profiles:
        flat = profile.reshape(len(codes), -1)
        result.append((flat @ flat.T)[a, b])
    return result


def urns(g, pairs, n, s):
    data, rates = empirical_urn.fit_all_branches(g, n, s)
    assert len(g['_asrv_training_ids']) == 138 and data.in_scope[pairs].all()
    result = {}
    for method, setting in [('branch_fixed', 'empirical_all'), ('predicted_N', 'predicted_N')]:
        out = {key: np.zeros(len(pairs)) for key in COLS[2:4]}
        empirical_urn._score(dict(g, asrv=setting), pairs, np.ones(len(pairs), bool), data, rates, ['any2spe'], out)
        result[method] = list(out.values())
    np.testing.assert_array_equal(result['branch_fixed'][1], result['predicted_N'][1])
    return result


def read_search(path):
    manifest = json.loads((path / 'csubst_search_run.json').read_text())
    assert manifest['status'] == 'complete' and 'csubst_cb_2.tsv' in manifest['outputs']
    raw = pd.read_table(path / 'csubst_cb_2.tsv', usecols=[*KEY, *COLS],
                        converters={key: lambda v: np.nan if v == 'NA' else float(v) for key in COLS})
    indexed = raw.set_index(KEY).sort_index()
    assert indexed.index.is_unique
    return indexed


def record(scenario, seed, origin, groups, truth, observed, expected, ftol):
    rows = []
    on, osyn = observed
    labels = ('no_stem', 'one_stem', 'two_stems')
    for method, (en, es) in expected.items():
        assert np.isfinite(en).all() and np.isfinite(es).all()
        w = omega_statistics._calc_raw_omega(omega_statistics._calc_raw_rate(on, en, ftol),
                                             omega_statistics._calc_raw_rate(osyn, es, ftol), ftol)
        if method == 'codon_model' and origin == 'inferred':
            np.testing.assert_allclose(w, truth['search_omega'], rtol=2e-8, atol=1e-13)
        hits = (on >= 2) & (w >= 5)
        for num, label in enumerate(labels):
            use = groups == num
            hits_here = use & hits
            rows.append(dict(scenario=scenario, seed=seed, endpoints=origin, method=method, group=label,
                             pairs=int(use.sum()), passes=int(hits_here.sum()), ocn_ge2=int((use & (on >= 2)).sum()),
                             passes_finite=int((hits_here & np.isfinite(w)).sum()),
                             passes_ocs1=int((hits_here & (osyn >= 1)).sum()),
                             observed_N=on[use].sum(), expected_N=en[use].sum(), generated_N=truth['muNany2spe_sim'][use].sum(),
                             observed_S=osyn[use].sum(), expected_S=es[use].sum(), generated_S=truth['muSany2spe_sim'][use].sum()))
    return rows


def score(rep):
    scenario, seed = rep.name.split('-')
    g, args = search_context(rep)
    cb = read_search(rep / 'codon').dropna(subset=COLS[2:4])
    pairs = cb.index.to_frame(index=False).to_numpy(int)
    assert len(pairs) == 8308
    oracle = pd.read_table(rep / 'sim/sim_expected_omega.tsv').set_index(KEY)
    assert oracle.index.is_unique
    q = oracle.loc[cb.index]
    np.testing.assert_array_equal(np.isin(pairs, STEMS), q[['foreground_1', 'foreground_2']])
    np.testing.assert_array_equal(np.loadtxt(rep / 'sim/sim_foreground_branch.txt', dtype=int), STEMS)
    group = np.isin(pairs, STEMS).sum(axis=1)
    assert (group == 2).sum() == 36
    truth = dict(q)
    truth['search_omega'] = cb[COLS[4]]
    inferred = g['_endpoint_tensors']
    inf_expected = urns(g, pairs, inferred['N'], inferred['S'])
    en, es = inf_expected['branch_fixed']
    if int(seed) == SEEDS[0]:
        # Independent executable check of both the observed and expected fitted counts.
        with (rep / 'urn-cli.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'csubst', *args, '--outdir', 'urn-cli'], check=True,
                           stdout=log, stderr=subprocess.STDOUT)
        check = read_search(rep / 'urn-cli').loc[cb.index]
        np.testing.assert_allclose(check[COLS[:2]], cb[COLS[:2]], rtol=1e-10, atol=1e-13)
        np.testing.assert_allclose(check[COLS[2:4]], np.column_stack((en, es)), rtol=1e-10, atol=1e-13)
    inf_expected['codon_model'] = cb[COLS[2:4]].to_numpy().T
    rows = record(scenario, seed, 'inferred', group, truth, cb[COLS[:2]].to_numpy().T, inf_expected, g['float_tol'])
    codes = codes_from_fasta(g, rep)
    aa, n, s = true_tensors(g, codes)
    assert codes.shape[1] == SITES
    # Realized counts independently from codons, including tied origins and separate S destinations.
    realized = []
    for sites, dest in ((n.branch_site, aa[codes]), (s.branch_site, codes)):
        out = np.empty(len(pairs))
        for i in range(0, len(pairs), 256):
            a, b = pairs[i:i + 256].T
            out[i:i + len(a)] = ((sites[a] * sites[b]) * (dest[a] == dest[b])).sum(axis=1)
        realized.append(out)
    np.testing.assert_array_equal(np.array(realized).T, q[['OCNany2spe_true', 'OCSany2spe_true']])
    true_expected = dict(codon_model=true_predictive(g, pairs, codes, aa))
    g['state_cdn'] = np.eye(len(aa))[codes]
    true_expected.update(urns(g, pairs, n, s))
    rows.extend(record(scenario, seed, 'true', group, truth, realized, true_expected, g['float_tol']))
    branch_rows = []
    for origin, tensor in [('true', n), ('inferred', inferred['N'])]:
        for b in STEMS:
            v = tensor.branch_site[b]
            branch_rows.append(dict(scenario=scenario, seed=seed, endpoints=origin, branch=b,
                                    N=v.sum(), domain_N=v[:DOMAIN].sum()))
    pd.DataFrame(rows).to_csv(rep / 'scores.tsv', sep='\t', index=False)
    pd.DataFrame(branch_rows).to_csv(rep / 'branches.tsv', sep='\t', index=False)


def regional_score(rep):
    """Condition N on a fixed site partition; keep the pooled routes and entire S fit."""
    scenario, seed = rep.name.split('-')
    g, _ = search_context(rep)
    cb = read_search(rep / 'codon')
    cb = cb[cb.index.to_frame(index=False).isin(STEMS).all(axis=1).to_numpy()]
    pairs = cb.index.to_frame(index=False).to_numpy(int)
    assert len(pairs) == 36
    q = pd.read_table(rep / 'sim/sim_expected_omega.tsv').set_index(KEY).loc[cb.index]
    control = pd.read_table(rep / 'scores.tsv').query("group == 'two_stems' and method == 'branch_fixed'").set_index('endpoints')
    gbase = g['_endpoint_tensors']
    codes = codes_from_fasta(g, rep)
    aa, nt, st = true_tensors(g, codes)
    rows = []
    for origin, n, s, obs in (
        ('inferred', gbase['N'], gbase['S'], cb[COLS[:2]].to_numpy().T),
        ('true', nt, st, q[['OCNany2spe_true', 'OCSany2spe_true']].to_numpy().T),
    ):
        if origin == 'true':
            g['state_cdn'] = np.eye(len(aa))[codes]
        data, rates = empirical_urn.fit_all_branches(g, n, s)
        assert len(g['_asrv_training_ids']) == 138
        ids, inverse = np.unique(pairs, return_inverse=True)
        local = inverse.reshape(pairs.shape)
        full = {key: np.zeros(len(pairs)) for key in COLS[2:4]}
        empirical_urn._score(g, pairs, np.ones(len(pairs), bool), data, rates, ['any2spe'], full)
        by_region = {'any2spe': np.zeros(len(pairs))}
        site_counts, _ = empirical_urn._event_totals(n, routes=True)
        for region in (slice(0, DOMAIN), slice(DOMAIN, SITES)):
            site = data.eligible[ids, region] * rates.n_rate[region]
            weights = (data.aa_parent[ids, region] @ rates.routes) * site[:, :, None]
            totals = site_counts[ids, region].sum(axis=1)
            draws = empirical_urn._allocate_branch_totals(weights.sum(axis=1), totals, 'N')
            np.testing.assert_allclose(draws.sum(axis=1), totals, atol=1e-12)
            empirical_urn._add_group_overlaps(by_region, local, weights, draws, g)
        expected = dict(branch_fixed=list(full.values()), region_fixed=[by_region['any2spe'], full[COLS[3]]])
        results = record(scenario, seed, origin, np.full(len(pairs), 2), q, obs, expected, g['float_tol'])
        for row in results:
            if row['group'] != 'two_stems':
                continue
            if row['method'] == 'branch_fixed':
                ref = control.loc[origin]
                for key in ('passes', 'expected_N', 'expected_S', 'passes_finite', 'passes_ocs1'):
                    np.testing.assert_allclose(row[key], ref[key], rtol=1e-10, atol=1e-12)
            else:
                rows.append(row)
    pd.DataFrame(rows).to_csv(rep / 'regional.tsv', sep='\t', index=False)


def run_scores(work, jobs, regional=False):
    reps = [work / f'{scenario}-{seed}' for seed in SEEDS for scenario in SCENARIOS]
    def run(rep):
        mode = 'regional' if regional else 'score'
        with (rep / (mode + '.log')).open('w') as log:
            subprocess.run([sys.executable, __file__, str(rep), mode], cwd=rep, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(rep.name, flush=True)
    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(run, reps))


def expected_true_n_gate(ref, scenario):
    """Exact expected number of eligible pairs under the nondirectional generators."""
    if scenario == 'directional':
        # Target-specific site probabilities cannot be recovered from the aggregate oracle.
        return np.nan
    mu = ref.muNany2spe_sim.to_numpy()
    if scenario == 'domain':
        background = ref.muNany2spe_null.to_numpy() / SITES
        by_rate = [(SITES - DOMAIN, background), (DOMAIN, (mu - (SITES - DOMAIN) * background) / DOMAIN)]
    else:
        by_rate = [(SITES, mu / SITES)]
    assert all(np.all((p >= 0) & (p < 1)) for _, p in by_rate)
    # P(0) = product of (1-p), P(1) = P(0) * sum of event odds over sites.
    log_zero = sum(n * np.log1p(-p) for n, p in by_rate)
    one_over_zero = sum(n * p / (1 - p) for n, p in by_rate)
    return float((-np.expm1(log_zero + np.log1p(one_over_zero))).sum())


def summarize(work):
    reps = [work / f'{scenario}-{seed}' for seed in SEEDS for scenario in SCENARIOS]
    calibration = pd.read_table(work / 'calibration.tsv')
    scale, = calibration.stem_scale.unique()
    scores = pd.concat([pd.read_table(rep / 'scores.tsv') for rep in reps])
    scores.insert(0, 'stem_scale', scale)
    assert len(scores) == len(reps) * 18 and not scores.duplicated(
        ['scenario', 'seed', 'group', 'endpoints', 'method']).any()
    regional = pd.concat([pd.read_table(rep / 'regional.tsv') for rep in reps])
    regional.insert(0, 'stem_scale', scale)
    assert len(regional) == 2 * len(reps) and set(regional.group) == {'two_stems'}
    scores = pd.concat([scores, regional], ignore_index=True)
    scores['split'] = np.where(scores.seed.isin(SEEDS[:4]), 'first_four', 'last_four')
    scores.to_csv(work / 'scores.tsv', sep='\t', index=False)
    keys = ['stem_scale', 'scenario', 'group', 'endpoints', 'method']
    summary = scores.drop(columns=['seed', 'split']).groupby(keys).sum()
    summary['pass_fraction'] = summary.passes / summary.pairs
    summary['seeds_with_hits'] = scores.assign(hit=scores.passes > 0).groupby(keys).hit.sum()
    summary['min_passes'] = scores.groupby(keys).passes.min()
    summary['max_passes'] = scores.groupby(keys).passes.max()
    summary.to_csv(work / 'summary.tsv', sep='\t')

    gates = []
    for rep in reps:
        scenario, seed = rep.name.split('-')
        log = (rep / 'sim/sim.log').read_text()
        targets = re.findall(r'Codon site \d+; Biased amino acids = ([A-Z]*);', log)
        assert len(targets) == {'global': SITES, 'domain': DOMAIN, 'directional': DOMAIN, 'baseline': 0}[scenario]
        if scenario == 'directional':
            from csubst import genetic_code
            rng = np.random.default_rng(int(seed))
            code = genetic_code.get_codon_table(1)
            assert targets == [str(main_simulate.get_biased_amino_acids('random1', code, rng)[0]) for _ in targets]
            # The two domain scenarios share the entire untreated block exactly.
            for file in ('sim.fa', 'sim_true_asr.anc.fa'):
                seq = main_simulate.read_fasta(rep / 'sim' / file)
                other = main_simulate.read_fasta(work / f'domain-{seed}/sim' / file)
                assert seq.keys() == other.keys()
                assert all(s[3 * DOMAIN:] == other[k][3 * DOMAIN:] for k, s in seq.items())
        else:
            assert not any(targets)
        table = read_search(rep / 'codon').dropna(subset=COLS[2:4])
        ref = pd.read_table(rep / 'sim/sim_expected_omega.tsv').set_index(KEY).loc[table.index]
        select = ref.foreground_1.eq(1) & ref.foreground_2.eq(1)
        true_n = ref.loc[select, 'OCNany2spe_true'].to_numpy()
        inferred_n = table.loc[select, COLS[0]].to_numpy()
        true, inferred = true_n >= 2, inferred_n >= 2
        gates.append(dict(stem_scale=scale, scenario=scenario, seed=int(seed),
                          Q_expected_passing_N=expected_true_n_gate(ref[select], scenario),
                          both=sum(true & inferred), true_only=sum(true & ~inferred),
                          inferred_only=sum(~true & inferred), neither=sum(~true & ~inferred),
                          missed_true_exactly_two=sum((true_n == 2) & ~inferred),
                          missed_inferred_ge1p999=sum(true & ~inferred & (inferred_n >= 1.999))))
    gates = pd.DataFrame(gates)
    gates.to_csv(work / 'gates.tsv', sep='\t', index=False)

    branches = pd.concat([pd.read_table(rep / 'branches.tsv') for rep in reps])
    branches = branches.pivot(index=['scenario', 'seed', 'branch'], columns='endpoints', values=['N', 'domain_N'])
    branches.columns = ['_'.join(cols) for cols in branches.columns]
    grid = calibration.melt(id_vars=['stem_scale', 'scenario', 'seed', 'omega', 'percent'],
                            value_vars=[f'expected_N_{b}' for b in STEMS], value_name='Q_expected_N')
    grid['branch'] = grid.pop('variable').str.removeprefix('expected_N_').astype(int)
    grid = grid.merge(branches.reset_index(), on=['scenario', 'seed', 'branch'], validate='one_to_one')
    assert len(grid) == len(reps) * len(STEMS)
    grid.to_csv(work / 'branches.tsv', sep='\t', index=False)


if __name__ == '__main__':
    handler = regional_score if sys.argv[2:] == ['regional'] else score
    handler(Path(sys.argv[1]).resolve())
