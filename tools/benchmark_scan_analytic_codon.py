#!/usr/bin/env python3
"""61-codon analytical-kernel scaling and strong-signal sensitivity benchmark.

Fixed GY-like CTMC with kappa=2, omega=.3, equal codon frequencies. Balanced
16/64/256-tip trees, four fitted rate categories. Timings include one complete
pruning for the null and four alternatives per candidate, but exclude IQ-TREE,
candidate discovery, and model construction. No comparison to an unrelated
Poisson scalar kernel is presented as a speedup.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csubst import endpoint, genetic_code, parser_misc, scan_analytic  # noqa: E402


def model_for(leaves, categories):
    table = [(aa, codon) for aa, codon in genetic_code.get_codon_table(1) if aa != '*']
    codons = [codon for aa, codon in table]
    aas = sorted({aa for aa, codon in table})
    groups = {aa: [i for i, (name, _) in enumerate(table) if name == aa] for aa in aas}
    k = len(codons)
    pi = np.full(k, 1/k)
    q = parser_misc.get_mechanistic_instantaneous_rate_matrix(dict(codon_orders=codons,
        amino_acid_orders=aas, synonymous_indices=groups, equilibrium_frequency=pi,
        kappa=2., omega=.3, float_type=np.float64))
    parents = np.r_[-1, (np.arange(1, leaves*2-1)-1)//2]
    lengths = np.r_[0, np.full(leaves*2-2, .05)]
    rates = [1.] if categories == 1 else [.1, .5, 1., 2.4]
    model = endpoint.EndpointModel(parents, lengths, q, pi, rates, np.full(len(rates), 1/len(rates)))
    engine = scan_analytic.EndpointEnrichment(model, np.array([aas.index(aa) for aa, _ in table]))
    return engine, codons, aas


def run_job(leaves, candidates, categories, seed, profile=None):
    engine, codons, aas = model_for(leaves, categories)
    if profile is not None:
        engine = scan_analytic.EndpointEnrichment(engine.model, engine.mapping, profile)
    model = engine.model
    rng = np.random.default_rng(seed)
    # Simulate full observations under the null, conditional on a sampled rate.
    observations = []
    identity = np.eye(len(model.pi))
    foreground = sorted(model.leaves)[::2][:8]
    for _ in range(candidates):
        rate = rng.choice(len(model.rates), p=model.weights)
        states = {model.root: rng.choice(len(model.pi), p=model.pi)}
        for node in model.order[1:]:
            states[node] = rng.choice(len(model.pi), p=model.transition(node, rate)[states[model.parents[node]]])
        observations.append({node: identity[states[node]] for node in model.leaves})
    engine.test(observations[0], foreground, range(len(aas)), [aas.index('N')])
    start = time.perf_counter()
    results = [engine.test(tips, foreground, range(len(aas)), [aas.index('N')]) for tips in observations]
    seconds = time.perf_counter() - start
    # A controlled strong endpoint pattern, varying independent foregrounds.
    # This is an illustrative pattern, not a calibrated power estimate.
    strong = []
    for count in (2, 4, 8):
        selected = foreground[:count]
        tips = {node: identity[codons.index('AAC' if node in selected else 'AAA')]
                for node in model.leaves}
        p, log_e = engine.test(tips, selected, [aas.index('K')], [aas.index('N')])
        strong.append(dict(foregrounds=count, p=p, log_e=log_e,
                           global_bh_if_single_hit=min(1., p * 10000 * 300 * 400)))
    return dict(leaves=leaves, candidates=candidates, categories=categories, seconds=seconds,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                null_min_p=min(p for p, _ in results), strong_patterns=strong)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidates', type=int, default=100)
    parser.add_argument('--profile', type=Path, default=None)
    args = parser.parse_args()
    profile = scan_analytic.read_profile(args.profile) if args.profile else None
    output = dict(analytical_profile=profile or scan_analytic.default_profile(), python=sys.version, platform=platform.platform(), measurements=[])
    for leaves in (16, 64, 256):
        for repeat in range(4):
            with ProcessPoolExecutor(1) as pool:
                record = pool.submit(run_job, leaves, args.candidates, 4, 51052026, profile).result()
                output['measurements'].append(record)
                print(json.dumps(record), flush=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
