#!/usr/bin/env python3
"""Generate independent codon CTMC datasets for the urn pipeline runner.

The null is a symmetric, single-nucleotide-neighbor sense-codon CTMC, with
uniform stationary frequencies and mean substitution rate one. It is
independent of CSUBST's urn and substitution-tensor generators. Alternatives
inject a shared terminal GCT state at selected sites in tips a and e; this
is a synthetic sensitivity control, not an alternative evolutionary model.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from csubst import ete, genetic_code, tree  # noqa: E402 -- select this checkout


TREE = '(((a:.3,b:.3):.2,(c:.3,d:.3):.2):.2,((e:.3,f:.3):.2,(g:.3,h:.3):.2):.2)R;'


def codon_generator():
    codons = sorted(codon for aa, codon in genetic_code.get_codon_table(1) if aa != '*')
    q = np.array([[float(sum(a != b for a, b in zip(c1, c2)) == 1) for c2 in codons] for c1 in codons])
    np.fill_diagonal(q, -q.sum(axis=1))
    q /= -np.diag(q).mean()
    return codons, q


def simulate_alignment(codons, q, num_sites, rng, heterogeneous=False, signal_sites=0, missing=0.,
                       branch_scale=1., rate_spread=1.):
    tr = ete.PhyloNode(TREE, format=1)
    rates = np.array([.2, 1., 3.]) if heterogeneous else np.array([1.])
    if rate_spread != 1.:
        with np.errstate(over='ignore'):
            rates = np.exp(rate_spread * np.log(rates / rates.max()))
    rates /= rates.mean()
    classes = rng.integers(0, len(rates), num_sites)
    states = {id(tr): rng.integers(0, len(codons), num_sites)}
    transitions = {}
    for node in tr.traverse('preorder'):
        if node is tr:
            continue
        source = states[id(node.up)]
        result = np.empty(num_sites, dtype=int)
        for k, rate in enumerate(rates):
            key = (float(node.dist) * branch_scale, float(rate))
            if key not in transitions:
                p = np.clip(expm(q * key[0] * rate), 0, None)
                p /= p.sum(axis=1, keepdims=True)
                cumulative = p.cumsum(axis=1)
                cumulative[:, -1] = 1.
                transitions[key] = cumulative
            mask = classes == k
            draws = rng.random(int(mask.sum()))
            result[mask] = (draws[:, None] > transitions[key][source[mask]]).sum(axis=1)
        states[id(node)] = result
    injected = rng.choice(num_sites, signal_sites, replace=False)
    sequences = {}
    for node in tr.traverse():
        if not ete.is_leaf(node):
            continue
        row = states[id(node)].copy()
        if node.name in ('a', 'e'):
            row[injected] = codons.index('GCT')
        tokens = np.array(codons)[row]
        tokens[rng.random(num_sites) < missing] = '---'
        sequences[node.name] = ''.join(tokens)
    return sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--calibration', type=int, default=1000)
    parser.add_argument('--validation', type=int, default=2000)
    parser.add_argument('--alternatives', type=int, default=200)
    parser.add_argument('--sites', type=int, default=300)
    parser.add_argument('--signal-sites', type=int, default=20)
    parser.add_argument('--missing', type=float, default=0.)
    parser.add_argument('--heterogeneous', action='store_true')
    parser.add_argument('--branch-scale', type=float, default=1.)
    parser.add_argument('--rate-spread', type=float, default=1.,
                        help='Power applied to the simulated rates before normalizing to mean one.')
    parser.add_argument('--compare-freerate-asrv', action='store_true',
                        help='Simulate three rate classes and fit GY+FQ+R3 under file, each and file_each ASRV.')
    parser.add_argument('--seed', type=int, default=12120910)
    args = parser.parse_args()
    heterogeneous = args.heterogeneous or args.compare_freerate_asrv
    if min(args.calibration, args.validation, args.sites) < 1 or args.alternatives < 0:
        raise ValueError('Positive calibration/validation/site counts are required.')
    if not 0 <= args.signal_sites <= args.sites or not 0 <= args.missing < 1:
        raise ValueError('Invalid signal-site or missing-data setting.')
    if not np.isfinite([args.branch_scale, args.rate_spread]).all() or min(args.branch_scale, args.rate_spread) <= 0:
        raise ValueError('Branch scale and rate spread must be positive and finite.')
    args.outdir.mkdir(parents=True, exist_ok=False)
    (args.outdir / 'tree.nwk').write_text(TREE + '\n')
    (args.outdir / 'foreground.tsv').write_text('1\t^a$\n2\t^e$\n')
    codons, q = codon_generator()
    replicates = []
    groups = [('calibration', 'null', args.calibration), ('validation', 'null', args.validation),
              ('validation', 'alternative', args.alternatives)]
    seeds = np.random.SeedSequence(args.seed).spawn(sum(g[2] for g in groups))
    index = 0
    for role, truth, count in groups:
        for _ in range(count):
            name = 'rep{:05d}'.format(index)
            sequences = simulate_alignment(codons, q, args.sites, np.random.default_rng(seeds[index]),
                                           heterogeneous, args.signal_sites if truth == 'alternative' else 0,
                                           args.missing, args.branch_scale, args.rate_spread)
            (args.outdir / (name + '.fa')).write_text(''.join('>{}\n{}\n'.format(k, v) for k, v in sequences.items()))
            replicates.append(dict(id=name, role=role, truth=truth, alignment=name + '.fa',
                                   tree='tree.nwk', foreground='foreground.tsv'))
            index += 1
    manifest = {
        'schema_version': 1,
        'simulator': {'independent_replicates': True, 'generator': 'symmetric_single_step_codon_CTMC',
                      'alternative': 'shared_terminal_GCT_injection', 'seed': args.seed, 'sites': args.sites,
                      'heterogeneous': heterogeneous, 'missing': args.missing,
                      'signal_sites': args.signal_sites, 'branch_scale': args.branch_scale,
                      'rate_spread': args.rate_spread},
        'analysis_options': {'iqtree_model': 'GY+F', 'drop_invariant_tip_sites': False},
        'configurations': [
            {'name': 'each_all', 'options': {'asrv': 'each'}},
            {'name': 'each_background', 'options': {'asrv': 'each', 'asrv_training_branches': 'background',
                                                   'asrv_concentration': 2.}},
            {'name': 'auto_background', 'options': {'asrv': 'each', 'asrv_training_branches': 'background',
                                                   'asrv_concentration': 2., 'nonsyn_recode': 'kgbauto6'}},
        ],
        'selection': {'statistic': 'omegaCany2spe', 'arity': 2,
                      'minimum_counts': {'OCNany2spe': 1., 'OCSany2spe': 1.}},
        'replicates': replicates,
    }
    if args.compare_freerate_asrv:
        manifest['analysis_options'].update(iqtree_model='GY+FQ+R3', urn_model='wallenius')
        manifest['configurations'] = [{'name': mode, 'options': {'asrv': mode}}
                                      for mode in ('file', 'each', 'file_each')]
    labeled = tree.add_numerical_node_labels(ete.PhyloNode(TREE, format=1))
    # The product intentionally leaves root/root-adjacent EC undefined.
    # Declare the testable family from topology, before observing any scores.
    manifest['selection']['exclude_branch_ids'] = [
        int(ete.get_prop(n, 'numerical_label')) for n in labeled.traverse()
        if n is labeled or n.up is labeled
    ]
    (args.outdir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
