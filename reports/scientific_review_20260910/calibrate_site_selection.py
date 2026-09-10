"""Known-parameter, four-codon null pilot; not a calibration of production P-values.

Exact pruning posteriors are recomputed from each pseudo-alignment. The current
CSUBST filter, codon rescaling and model expectation kernels run in every draw.
The tested branch pair is fixed before simulation; no search selection is tested.
"""
import argparse
import contextlib
import copy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from csubst import ete, omega, parser_misc, sequence, substitution, tree  # noqa: E402


CODONS = np.array(['AAA', 'AAG', 'AAC', 'AAT'])
MODES = ('no', 'tip_invariant', 'zero_sub_mass')


def make_model():
    tr = tree.add_numerical_node_labels(ete.PhyloNode('((A:.3,B:.3)X:.3,(C:.3,D:.3)Y:.3)R;', format=1))
    q = (np.ones((4, 4)) - 4 * np.eye(4)) / 3
    return tr, q, expm(q * .3)


def simulate_posterior(tr, transition, nsite, rng, ml):
    nodes = list(tr.traverse())
    index = {node.name: int(ete.get_prop(node, 'numerical_label')) for node in nodes}
    hidden = {}
    for node in nodes:
        if ete.is_root(node):
            hidden[node.name] = rng.integers(4, size=nsite)
        else:
            p = transition[hidden[node.up.name]]
            hidden[node.name] = (rng.random(nsite)[:, None] > p.cumsum(axis=1)[:, :3]).sum(axis=1)
    messages = {}
    def message(node, previous):
        key = (node.name, None if previous is None else previous.name)
        if key in messages:
            return messages[key]
        value = np.eye(4)[hidden[node.name]] if ete.is_leaf(node) else np.ones((nsite, 4))
        neighbors = list(ete.get_children(node)) + ([] if node.up is None else [node.up])
        for neighbor in neighbors:
            if neighbor is previous:
                continue
            value = value * (message(neighbor, node) @ transition.T)
            value /= value.max(axis=1, keepdims=True)
        messages[key] = value
        return value
    state = np.zeros((len(nodes), nsite, 4))
    for node in nodes:
        posterior = message(node, None)
        posterior = posterior / posterior.sum(axis=1, keepdims=True)  # uniform stationary prior
        if ml and not ete.is_leaf(node):
            posterior = np.eye(4)[posterior.argmax(axis=1)]
        state[index[node.name]] = posterior
        if ete.is_leaf(node):
            ete.set_prop(node, 'sequence', ''.join(CODONS[hidden[node.name]]))
    return state, index


def one_draw(nsite, rng, ml):
    tr, q, transition = make_model()
    state, ids = simulate_posterior(tr, transition, nsite, rng, ml)
    g = dict(tree=tr, state_cdn=state, codon_orders=CODONS, num_input_site=nsite,
             amino_acid_orders=['K', 'N'], synonymous_indices={'K': [0, 1], 'N': [2, 3]},
             nonsynonymous_indices={'K': [0, 1], 'N': [2, 3]}, nonsyn_state_orders=np.array(['K', 'N']),
             max_synonymous_size=2, float_type=np.float64, float_tol=1e-12, threads=1,
             iqtree_rate_values=np.ones(nsite), instantaneous_codon_rate_matrix=q,
             instantaneous_nsy_rate_matrix=np.ones((2, 2)) - 2 * np.eye(2),
             nonsyn_recode='no', expected_state_backend='expm', min_sub_pp=0)
    g['state_pep'] = sequence.cdn2pep_state(state, g)
    g['state_nsy'] = g['state_pep']
    values, kept = {}, {}
    pair = [ids['A'], ids['C']]
    for mode in MODES:
        local = copy.deepcopy(g)
        mask = parser_misc.get_site_drop_mask(local, mode, np.arange(nsite))
        kept[mode] = int((~mask).sum())
        if mask.all():
            values[mode] = None
            continue
        if mode != 'no':
            local['drop_invariant_tip_sites_mode'] = mode
            parser_misc.drop_invariant_tip_sites(local)
        on = substitution.get_substitution_tensor(local['state_nsy'], mode='asis', g=local)
        os = substitution.get_substitution_tensor(local['state_cdn'], mode='syn', g=local)
        tree.rescale_branch_length(local, os, on)
        counts = [substitution._get_sparse_site_vectors(tensor, pair)[0].sum() for tensor in (on, os)]
        for state_mode, tensor_mode in [('nsy', 'asis'), ('cdn', 'syn')]:
            expected = omega.get_exp_state(local, state_mode)
            tensor = substitution.get_substitution_tensor(expected, state_tensor_anc=local['state_' + state_mode],
                                                           mode=tensor_mode, g=local)
            counts.append(substitution._get_sparse_site_vectors(tensor, pair)[0].sum())
        on_count, os_count, en_count, es_count = counts
        # Predeclared fixed symmetric alpha=1, target=both, identical in all draws.
        values[mode] = ((on_count + 1) / (en_count + 1)) / ((os_count + 1) / (es_count + 1))
    return values, kept


def upper_tail(reference, value):
    if value is None:
        return 1.0  # Explicit no-test outcome, retained in the FPR denominator.
    null = np.array([-np.inf if v is None else v for v in reference])
    return float((1 + np.count_nonzero(null >= value - 1e-12)) / (len(null) + 1))


def run(reference_count, evaluation_count, nsite, seed):
    streams = np.random.SeedSequence(seed).spawn(4)
    output = {'seed': seed, 'reference_replicates': reference_count, 'evaluation_replicates': evaluation_count,
              'sites': nsite, 'scope': 'known_parameter_four_codon_fixed_pair',
              'statistic': 'any2any omegaC; fixed symmetric alpha=1, both; no long-tail',
              'model_fitting': 'parameters known, exact posterior pruning repeated; no IQ-TREE refit',
              'undefined_policy': 'no-test p=1, included in denominator; null atom at minus infinity',
              'production_pvalue_calibrated': False,
              'binomial_interval_scope': 'conditional_on_shared_reference_sample', 'results': []}
    for mi, ml in enumerate((False, True)):
        refs, evals = [], []
        for target, count, stream in [(refs, reference_count, streams[2 * mi]),
                                      (evals, evaluation_count, streams[2 * mi + 1])]:
            rng = np.random.default_rng(stream)
            for _ in range(count):
                target.append(one_draw(nsite, rng, ml))
        for mode in MODES:
            reference = [value[mode] for value, _ in refs]
            pvalues = [upper_tail(reference, value[mode]) for value, _ in evals]
            reject = int(np.count_nonzero(np.asarray(pvalues) <= .05))
            ci = binomtest(reject, evaluation_count).proportion_ci(confidence_level=.95)
            mismatched = [upper_tail([v['no'] for v, _ in refs], value[mode]) for value, _ in evals]
            output['results'].append(dict(
                ml_anc=ml, criterion=mode, rejected=reject, rejection_rate=reject / evaluation_count,
                binomial_ci95=[ci.low, ci.high],
                reference_undefined=sum(v is None for v in reference),
                evaluation_undefined=sum(v[mode] is None for v, _ in evals),
                mean_retained=float(np.mean([k[mode] for _, k in evals])),
                mismatched_unfiltered_null_rejected=int(np.count_nonzero(np.asarray(mismatched) <= .05)),
            ))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=int, default=199)
    parser.add_argument('--evaluation', type=int, default=200)
    parser.add_argument('--sites', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.reference, args.evaluation, args.sites) < 1:
        parser.error('replicate counts and sites must be positive')
    with open('/dev/null', 'w') as silent, contextlib.redirect_stdout(silent):
        result = run(args.reference, args.evaluation, args.sites, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Wrote ' + str(args.output))


if __name__ == '__main__':
    main()
