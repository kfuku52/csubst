#!/usr/bin/env python3
"""Independent four-codon CTMC/pruning validation of scan discovery calibration.

The simulator and exact node-marginal pruning are independent of CSUBST's ASR
parser, score and substitution implementation. Every dataset reruns posterior
reconstruction, optional site filtering, branch rescaling, candidate discovery,
and the two trait x two match families. Parameters are KNOWN, not refitted;
this validates the reference/selection machinery, not fitted-bootstrap FWER.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csubst import ete, parser_misc, pipeline_calibration, scan_statistics, sequence, substitution, substitution_scan, tree  # noqa: E402


CODONS = np.array(['AAA', 'AAG', 'AAC', 'AAT'])
Q = (np.ones((4, 4)) - 4*np.eye(4))/3
TOPOLOGIES = {
    'sparse': '(((a:.05,b:.05)X:.05,(c:.05,d:.05)Y:.05)U:.05,((e:.05,f:.05)Z:.05,(g:.05,h:.05)W:.05)V:.05)R;',
    'unequal': '(((a:.1,b:.8)X:.2,(c:.6,d:.02)Y:.3)U:.1,((e:.6,f:.05)Z:.3,(g:.1,h:.7)W:.1)V:.4)R;',
    'uncertain': '(((a:.5,b:.5)X:.1,(c:.5,d:.5)Y:.1)U:.1,((e:.5,f:.5)Z:.1,(g:.5,h:.5)W:.1)V:.1)R;',
}


def _transition(length, rates):
    decay = np.exp(-4*length*rates/3)
    return decay[:, None, None]*np.eye(4) + (1-decay[:, None, None])/4


def posterior_dataset(scenario, sites, rng, signal_sites=0):
    tr = tree.add_numerical_node_labels(ete.PhyloNode(TOPOLOGIES[scenario], format=1))
    nodes = list(tr.traverse('preorder'))
    index = {id(n): int(ete.get_prop(n, 'numerical_label')) for n in nodes}
    rates = rng.choice([.2, 1., 2.8], sites) if scenario == 'uncertain' else np.ones(sites)
    transitions = {id(n): _transition(n.dist, rates) for n in nodes if n.up is not None}
    hidden = {}
    observations = {}
    signal = np.arange(signal_sites)
    for node in nodes:
        if node.up is None:
            hidden[id(node)] = rng.integers(4, size=sites)
        else:
            p = transitions[id(node)][np.arange(sites), hidden[id(node.up)]]
            hidden[id(node)] = (rng.random(sites)[:, None] > p.cumsum(axis=1)[:, :3]).sum(axis=1)
        if ete.is_leaf(node):
            observed = hidden[id(node)].copy()
            if node.name in ('a', 'e'):
                observed[signal] = 2  # sensitivity control; other sites remain null.
            observations[id(node)] = np.eye(4)[observed]
    cache = {}
    def message(node, previous):
        key = (id(node), id(previous))
        if key in cache:
            return cache[key]
        value = observations.get(id(node), np.ones((sites, 4))).copy()
        neighbors = list(ete.get_children(node)) + ([] if node.up is None else [node.up])
        for neighbor in neighbors:
            if neighbor is previous:
                continue
            edge = node if node.up is neighbor else neighbor
            value *= np.einsum('sij,sj->si', transitions[id(edge)], message(neighbor, node))
            value /= value.max(axis=1, keepdims=True)
        cache[key] = value
        return value
    state = np.zeros((len(nodes), sites, 4))
    for node in nodes:
        marginal = message(node, None)
        state[index[id(node)]] = marginal / marginal.sum(axis=1, keepdims=True)
        if ete.is_leaf(node):
            obs = observations[id(node)].argmax(axis=1)
            ete.set_prop(node, 'sequence', ''.join(CODONS[obs]))
    state[index[id(tr)]] = 0  # production codon ASR has no row for an inserted root.
    return tr, state, rates


def one_dataset(arguments):
    scenario, sites, seed, filtered, signal_sites = arguments
    with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
        tr, state, rates = posterior_dataset(scenario, sites, np.random.default_rng(seed), signal_sites)
        labels = {n.name: int(ete.get_prop(n, 'numerical_label')) for n in tr.traverse()}
        fg_leaves = {'trait1': [['a'], ['e']], 'trait2': [['c'], ['g']]}
        for trait, groups in fg_leaves.items():
            for node in tr.traverse():
                leaves = set(ete.get_leaf_names(node))
                ete.add_features(node, **{'is_fg_' + trait: bool(leaves <= set(sum(groups, [])))})
                for i, names in enumerate(groups, 1):
                    ete.add_features(node, **{'is_lineage_fg_{}_{}'.format(trait, i): leaves <= set(names)})
        g = dict(tree=tr, state_cdn=state, codon_orders=CODONS, num_input_site=sites,
                 amino_acid_orders=['K','N'], synonymous_indices={'K':[0,1], 'N':[2,3]},
                 nonsynonymous_indices={'K':[0,1], 'N':[2,3]}, nonsyn_state_orders=np.array(['K','N']),
                 max_synonymous_size=2, float_type=np.float64, float_tol=1e-12, threads=1,
                 iqtree_rate_values=rates, instantaneous_codon_rate_matrix=Q,
                 instantaneous_nsy_rate_matrix=np.array([[-2/3, 2/3], [2/3, -2/3]]),
                 nonsyn_recode='no', expected_state_backend='expm', min_sub_pp=0,
                 fg_df=pd.DataFrame({'name':['a','e','c','g'], 'trait1':[1,2,0,0], 'trait2':[0,0,1,2]}),
                 fg_leaf_names=fg_leaves,
                 fg_ids={t: np.array([labels[n[0]] for n in groups]) for t,groups in fg_leaves.items()},
                 fg_stem_only=False, scan_unit_mode='clade', scan_min_support='2', scan_min_event_pp=.5,
                 scan_rate_length='n_rescaled', scan_rate_exposure='q_weighted', scan_other_scope='all',
                 scan_rate_event_mode='posterior_sum', scan_match='any2spe,spe2spe',
                 scan_pvalue_calibration='none', scan_n_permutations=0)
        g['state_pep'] = sequence.cdn2pep_state(state, g)
        g['state_nsy'] = g['state_pep']
        if filtered:
            g['drop_invariant_tip_sites_mode'] = 'tip_invariant'
            mask = parser_misc.get_site_drop_mask(g, 'tip_invariant', np.arange(sites))
            if mask.all():
                return dict(score=None, null_score=None, candidates=0, diagnostic_p_reject=False, diagnostic_bh_reject=False)
            parser_misc.drop_invariant_tip_sites(g)
        on = substitution.get_substitution_tensor(g['state_nsy'], mode='asis', g=g)
        os_tensor = substitution.get_substitution_tensor(g['state_cdn'], mode='syn', g=g)
        tree.rescale_branch_length(g, os_tensor, on)
        frame, _ = substitution_scan.scan_substitutions(g, on)
    maximum = scan_statistics.maximum_score(frame)
    # In partial alternatives, only uninjected sites count as false detections.
    null = frame.loc[pd.to_numeric(frame['codon_site_alignment']) > signal_sites]
    null_maximum = scan_statistics.maximum_score(null)
    return dict(score=None if maximum == -np.inf else maximum,
                null_score=None if null_maximum == -np.inf else null_maximum,
                candidates=len(frame),
                diagnostic_p_reject=bool((null['p_rate_enrichment_asymptotic'] <= .05).any()),
                diagnostic_bh_reject=bool((null['q_rate_enrichment_asymptotic_by_trait_match'] <= .05).any()))


def evaluate(reference, validation, level):
    null = [-np.inf if r['score'] is None else r['score'] for r in reference]
    pvalues = [scan_statistics.empirical_pvalue(-np.inf if r['null_score'] is None else r['null_score'], null) for r in validation]
    hits = sum(p <= level for p in pvalues)
    return {'datasets': len(validation), 'rejections': hits, 'rate': hits/len(validation),
            'ci95': list(pipeline_calibration.binomial_interval(hits, len(validation))),
            'empty_families': sum(r['score'] is None for r in validation),
            'diagnostic_p_any_rejections': sum(r['diagnostic_p_reject'] for r in validation),
            'diagnostic_trait_match_bh_any_rejections': sum(r['diagnostic_bh_reject'] for r in validation)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--reference', type=int, default=9999)
    parser.add_argument('--validation', type=int, default=5000)
    parser.add_argument('--partial-alternatives', type=int, default=200)
    parser.add_argument('--sites', type=int, default=12)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=51052026)
    parser.add_argument('--scenarios', default='sparse,unequal,uncertain')
    parser.add_argument('--level', type=float, default=.05)
    parser.add_argument('--fpr-limit', type=float, default=.06)
    args = parser.parse_args()
    scenarios = args.scenarios.split(',')
    if min(args.reference, args.validation, args.sites, args.workers) < 1 or args.partial_alternatives < 0:
        raise ValueError('Invalid replicate/site/worker counts.')
    if not 0 < args.level <= args.fpr_limit < 1 or any(s not in TOPOLOGIES for s in scenarios):
        raise ValueError('Invalid scenario or error-level specification.')
    args.outdir.mkdir(parents=True, exist_ok=False)
    summary = {'scope': 'known_parameter_four_codon_null_with_exact_pruning_and_repeated_discovery',
               'not_validated': 'IQ-TREE parameter refitting, 3Di, universal/strong FWER',
               'reference': args.reference, 'seed': args.seed, 'level': args.level, 'fpr_limit': args.fpr_limit,
               'sites': args.sites, 'seed_scheme': 'separate_scenario_filter_role_streams_v2',
               'interval_scope': 'binomial_validation_interval_conditional_on_fixed_reference_sample',
               'family': 'two_traits_two_matches_all_sites', 'results': []}
    for scenario_id, scenario in enumerate(scenarios):
        for filtered in (False, True):
            # Growing one role never moves a validation dataset into the
            # reference or changes the other role's seeds.
            seeds = np.concatenate([
                np.random.SeedSequence([args.seed, scenario_id, int(filtered), role]).generate_state(count)
                for role, count in enumerate((args.reference, args.validation, args.partial_alternatives))
            ])
            tasks = [(scenario, args.sites, int(seed), filtered,
                      min(2, args.sites) if i >= args.reference+args.validation else 0) for i,seed in enumerate(seeds)]
            print('{} / filter={}: {} datasets'.format(scenario, filtered, len(tasks)), flush=True)
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                records = []
                for i, record in enumerate(pool.map(one_dataset, tasks, chunksize=8)):
                    records.append(dict(record, seed=int(seeds[i])))
                    if (i+1) % 1000 == 0:
                        print('  completed {}/{}'.format(i+1, len(tasks)), flush=True)
            reference = records[:args.reference]
            null_validation = records[args.reference:args.reference+args.validation]
            result = {'scenario': scenario, 'filter_tip_invariant': filtered,
                      'null': evaluate(reference, null_validation, args.level)}
            if args.partial_alternatives:
                result['partial_alternative_false_detections'] = evaluate(reference, records[args.reference+args.validation:], args.level)
            result['null_fpr_criterion_met'] = result['null']['ci95'][1] <= args.fpr_limit
            summary['results'].append(result)
            (args.outdir / '{}_filter{}.json'.format(scenario, int(filtered))).write_text(json.dumps(records) + '\n')
            (args.outdir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
