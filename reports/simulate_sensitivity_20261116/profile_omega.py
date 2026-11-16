"""Profile the nonsynonymous multiplier with other fitted inputs fixed."""

import argparse
import gzip
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from csubst import ete, genetic_code, parser_iqtree, parser_misc
from csubst.main_simulate import read_fasta
from run import CONDITIONS
from shared_rates import checkpoint_parameters, load_model
from summarize import read_pairs


def fitted_mixture(checkpoint):
    with gzip.open(checkpoint, 'rt') as stream:
        mixture = re.search(r'^RateFree4:\n((?:[ \t].*\n)+)', stream.read(), re.MULTILINE)[1]
    return tuple(
        np.fromstring(re.search(r'^\s+' + key + r':(.*)$', mixture, re.MULTILINE)[1], sep=',')
        for key in ('rates', 'prop')
    )


def profile(rep, frequency_floor):
    if frequency_floor:
        report = rep / 'iqtree-restart-freq_floor/fit.iqtree'
    else:
        report, = (rep / 'iqtree-default').glob('*.iqtree')
    state = report.with_suffix('.state')
    codons = [key.removeprefix('p_') for key in parser_iqtree._read_state_header_columns(state)[3:]]
    aa_by_codon = {codon: aa for aa, codon in genetic_code.get_codon_table(1)}
    amino_acids = np.array([aa_by_codon[c] for c in codons])
    nonsyn = amino_acids[:, None] != amino_acids
    syn = ~nonsyn & ~np.eye(len(codons), dtype=bool)
    exchange = parser_misc.read_exchangeability_matrix('substitution_matrix/ECMunrest.dat', codons)
    bundled = rep.parent / 'inputs' / f'{rep.parent.name}.alignment.fa.iqtree'
    simulation = parser_iqtree.read_iqtree(dict(
        path_iqtree_iqtree=str(bundled), codon_orders=np.array(codons), float_type=np.float64,
    ))
    template = parser_misc.exchangeability2Q(exchange, simulation['equilibrium_frequency'])
    scenario, seed = rep.name.rsplit('-', 1)
    true_multiplier = CONDITIONS[scenario].omega / (template[nonsyn].sum() / template[syn].sum())
    reported = float(re.search(r'^Log-likelihood of the tree:\s*(\S+)', report.read_text(), re.M)[1])
    checkpoint = report.with_suffix('.ckp.gz')
    if frequency_floor:
        estimated, pi = checkpoint_parameters(checkpoint, codons)
    else:
        _, metadata = read_pairs(rep, 'default')
        model, _, _, _ = load_model(rep, metadata)
        estimated = metadata['parameter_source']['omega']
        pi = model.pi
    lengths = ete.PhyloNode(str(report.with_suffix('.treefile')), format=1)
    order = list(lengths.traverse('postorder'))
    lookup = {codon: i for i, codon in enumerate(codons)}
    fasta = read_fasta(rep / 'sim/sim.fa')
    identity = np.eye(len(codons))
    tips = {name: identity[[lookup[s[i:i+3]] for i in range(0, len(s), 3)]] for name, s in fasta.items()}
    sites = len(next(iter(tips.values())))
    rates, weights = fitted_mixture(checkpoint)
    supported = pi > 0
    pi = pi[supported]
    tips = {name: obs[:, supported] for name, obs in tips.items()}
    identity = np.eye(len(pi))
    sqrt_pi = np.sqrt(pi)
    balance = sqrt_pi[:, None] / sqrt_pi

    def loglik(omega):
        matrix = exchange.copy()
        matrix[nonsyn] *= omega
        q = parser_misc.exchangeability2Q(matrix[np.ix_(supported, supported)], pi)
        eig, vectors = np.linalg.eigh(q * balance)
        values = []
        for r in rates:
            active, transitions = {}, {}
            log_scale = np.zeros(sites)
            for node in order:
                if ete.is_leaf(node):
                    active[id(node)] = tips[node.name]
                    continue
                likelihood = np.ones((sites, len(pi)))
                for child in ete.get_children(node):
                    if child.dist not in transitions:
                        if r * child.dist == 0:
                            transition = identity
                        else:
                            transition = ((vectors * np.exp(eig * (r * child.dist))) @ vectors.T) / balance
                            np.maximum(transition, 0, out=transition)
                        transitions[child.dist] = transition
                    likelihood *= active.pop(id(child)) @ transitions[child.dist].T
                scale = likelihood.sum(axis=1)
                with np.errstate(divide='ignore'):
                    log_scale += np.log(scale)
                active[id(node)] = np.divide(
                    likelihood, scale[:, None], out=likelihood, where=scale[:, None] > 0,
                )
            with np.errstate(divide='ignore'):
                values.append(np.log(active[id(lengths)] @ pi) + log_scale)
        return logsumexp(values, axis=0, b=weights[:, None]).sum()

    reproduced = loglik(estimated)
    np.testing.assert_allclose(reproduced, reported, rtol=0, atol=0.02)
    fit = minimize_scalar(lambda x: -loglik(np.exp(x)), bounds=(np.log(0.005), np.log(8)), method='bounded')
    if not fit.success:
        raise RuntimeError(fit.message)
    refined = np.exp(fit.x)
    return dict(
        dataset=rep.parent.name, scenario=scenario, seed=int(seed), omega_sim=true_multiplier,
        variant='freq_floor' if frequency_floor else 'default',
        zero_frequency_codons=int((~supported).sum()), min_positive_frequency=pi.min(),
        omega_iqtree=estimated, omega_profile=refined, log_likelihood_reported=reported,
        log_likelihood_replayed=reproduced, log_likelihood_profile=-fit.fun,
        log_likelihood_gain=-fit.fun - reproduced,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('work', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--frequency-floor', action='store_true')
    args = p.parse_args()
    rows = []
    pattern = '*/*/iqtree-restart-freq_floor/fit.iqtree' if args.frequency_floor else '*/*/default/csubst_cb_2.tsv'
    for path in sorted(args.work.glob(pattern)):
        rep = path.parents[1]
        rows.append(profile(rep, args.frequency_floor))
    pd.DataFrame(rows).to_csv(args.output, sep='\t', index=False, float_format='%.10g')


if __name__ == '__main__':
    main()
