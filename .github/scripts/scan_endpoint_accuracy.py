#!/usr/bin/env python3
"""Compare exposure definitions against exact finite-time codon probabilities.

Known-parent multinomial endpoint draws validate the mean, not ASR uncertainty,
candidate selection, P-values or FWER. GY uses omega=.3/kappa=2; all models use
uniform codon frequencies. N-rescaled lengths use the exact mean of observed
non-synonymous endpoint counts per site (the infinite-alignment limit).
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from csubst import genetic_code, parser_misc, scan_endpoint, substitution_scan

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    table = [(aa, c) for aa, c in genetic_code.get_codon_table(1) if aa != "*"]
    codons = np.array(sorted(c for aa, c in table))
    lookup = {c: aa for aa, c in table}
    aa = sorted(set(lookup.values()))
    ids = np.array([aa.index(lookup[c]) for c in codons])
    freq = np.ones(61)/61
    common = dict(codon_orders=codons, amino_acid_orders=aa,
                  synonymous_indices={a: np.flatnonzero(ids == i) for i, a in enumerate(aa)},
                  omega=.3, kappa=2, equilibrium_frequency=freq, float_type=np.float64)
    matrices = {"GY": parser_misc.get_mechanistic_instantaneous_rate_matrix(common)}
    for model, name in [("ECMrest", "ECMrest.dat"), ("ECMK07", "ECMunrest.dat")]:
        matrices[model] = parser_misc.exchangeability2Q(
            parser_misc.read_exchangeability_matrix("substitution_matrix/" + name, codons), freq)
    parent = list(codons).index("TTT")
    source = aa.index("F")
    state = np.zeros((2, 1, 61))
    state[:, 0, parent] = 1
    nsy = state @ np.eye(20)[ids]
    rng = np.random.default_rng(7001)
    rows = []
    for model, q in matrices.items():
        for t in [1e-4, .01, .1, 1., 10., 100.]:
            p = expm(q*t)[parent]
            n_length = p[ids != source].sum()
            meta = pd.DataFrame(dict(branch_id=[1], parent_id=[0], raw_length=[t],
                                     sn_rescaled_length=[t], n_rescaled_length=[n_length]))
            g = dict(state_cdn=state, state_nsy=nsy, instantaneous_codon_rate_matrix=q,
                     equilibrium_frequency=freq, iqtree_rate_values=np.ones(1),
                     substitution_model=model, scan_rate_length="raw")
            ctx = scan_endpoint.build_context(g, meta, ids)
            draws = rng.multinomial(200000, p/p.sum())
            for target in ["K", "L"]:
                dest = aa.index(target)
                exact = p[ids == dest].sum()
                new, _, _ = scan_endpoint.expected_events(ctx, state, nsy, 0, [source], [dest])
                np.testing.assert_allclose(new, [exact], atol=1e-25, rtol=1e-12)
                exposures = {}
                for length_mode, length in [("raw", t), ("n_rescaled", n_length)]:
                    opp = substitution_scan._q_weighted_opportunity(
                        meta, nsy, 0, [source], [dest], None, length_mode,
                        state_cdn=state, codon_q_matrix=q, codon_state_ids=ids)
                    exposures[length_mode] = float(length*opp[0])
                rows.append(dict(model=model, parent="TTT", candidate="F->"+target, t=t,
                                 endpoint_exact=exact, endpoint_implementation=float(new[0]),
                                 legacy_raw=exposures["raw"], legacy_n=exposures["n_rescaled"],
                                 exact_n_length=float(n_length), draws=200000,
                                 observed=int(draws[ids == dest].sum()),
                                 mc_probability=float(draws[ids == dest].sum()/200000)))
    pd.DataFrame(rows).to_csv(args.outdir / "accuracy.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
