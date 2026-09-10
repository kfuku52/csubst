#!/usr/bin/env python3
"""Compare recomputed non-root ASR marginals with a shared PEPC IQ-TREE fit."""
import argparse
import json
from pathlib import Path
import runpy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from csubst import cli, ete, scan_ctmc, sequence  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-dir', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    args.calibration, args.niter = 'none', 0
    helper = runpy.run_path(str(Path(__file__).with_name('scan_ctmc_benchmark.py')))
    original = scan_ctmc.prepare
    record = {}

    def compare(g):
        updated, tensor = original(g)
        nodes = [n for n in g['tree'].traverse() if not ete.is_leaf(n) and not ete.is_root(n)]
        ids = [int(ete.get_prop(n, 'numerical_label')) for n in nodes]
        imported = np.asarray(g['state_cdn'][ids], float)
        mass = imported.sum(axis=2, keepdims=True)
        valid = mass[:, :, 0] > 0
        imported = np.divide(imported, mass, out=np.zeros_like(imported), where=mass > 0)
        errors = np.abs(updated['state_cdn'][ids] - imported)
        errors[~valid] = np.nan
        tv = .5 * errors.sum(axis=2)
        node_index, site, codon = np.unravel_index(np.nanargmax(errors), errors.shape)
        record.update(nonroot_internal_nodes=len(ids), compared_node_sites=int(valid.sum()),
                      max_absolute_probability_difference=float(np.nanmax(errors)),
                      mean_absolute_probability_difference=float(np.nanmean(errors)),
                      mean_total_variation=float(np.nanmean(tv)), max_total_variation=float(np.nanmax(tv)),
                      max_difference_node=str(nodes[node_index].name), max_difference_site=int(site)+1,
                      max_difference_codon=str(g['codon_orders'][codon]),
                      imported_probability=float(imported[node_index, site, codon]),
                      recomputed_probability=float(updated['state_cdn'][ids[node_index], site, codon]),
                      q_mean_rate=float(-g['equilibrium_frequency'] @ g['instantaneous_codon_rate_matrix'].diagonal()))
        compatibility_emissions = np.asarray(g['state_cdn']).copy()
        ambiguous_sites = np.zeros(compatibility_emissions.shape[1], dtype=bool)
        for leaf in ete.iter_leaves(g['tree']):
            bid = int(ete.get_prop(leaf, 'numerical_label'))
            ambiguous = (compatibility_emissions[bid] > 0).sum(axis=1) > 1
            ambiguous_sites |= ambiguous
            compatibility_emissions[bid, ambiguous] = 0
        compatible, _ = scan_ctmc.infer(g['tree'], compatibility_emissions,
                                        g['instantaneous_codon_rate_matrix'], g['equilibrium_frequency'],
                                        np.zeros(g['state_cdn'].shape[2], dtype=int), 'joint')
        compatibility_errors = np.abs(compatible[ids] - imported)
        compatibility_errors[~valid] = np.nan
        record.update(ambiguous_alignment_sites=(np.flatnonzero(ambiguous_sites)+1).tolist(),
                      large_difference_alignment_sites=(np.flatnonzero(np.nanmax(tv, axis=0) > .001)+1).tolist(),
                      unambiguous_sites_max_difference=float(np.nanmax(errors[:, ~ambiguous_sites])),
                      ambiguity_masked_max_difference=float(np.nanmax(compatibility_errors)),
                      ambiguity_masked_mean_difference=float(np.nanmean(compatibility_errors)))
        # ECMK07+F empirical frequencies can also be recovered without report rounding.
        lookup = {str(c): i for i, c in enumerate(g['codon_orders'])}
        counts = np.zeros(len(lookup))
        for seq in sequence.read_fasta(g['alignment_file']).values():
            seq = seq.upper().replace('U', 'T')
            for start in range(0, len(seq), 3):
                index = lookup.get(seq[start:start+3])
                if index is not None:
                    counts[index] += 1
        precise_pi = counts / counts.sum()
        precise_q = g['instantaneous_codon_rate_matrix'].copy()
        np.fill_diagonal(precise_q, 0)
        precise_q *= (precise_pi / g['equilibrium_frequency'])[None, :]
        np.fill_diagonal(precise_q, -precise_q.sum(axis=1))
        precise_q /= -precise_pi @ precise_q.diagonal()
        precise, _ = scan_ctmc.infer(g['tree'], compatibility_emissions, precise_q, precise_pi,
                                    np.zeros(len(counts), dtype=int), 'joint')
        precise_errors = np.abs(precise[ids] - imported)
        precise_errors[~valid] = np.nan
        raw_errors = np.abs(precise[ids] - g['state_cdn'][ids])
        raw_errors[~valid] = np.nan
        record['ambiguity_masked_unrounded_vs_raw_report_max_difference'] = float(np.nanmax(raw_errors))
        record.update(empirical_vs_report_max_frequency_difference=float(np.max(np.abs(precise_pi-g['equilibrium_frequency']))),
                      ambiguity_masked_unrounded_max_difference=float(np.nanmax(precise_errors)),
                      ambiguity_masked_unrounded_mean_difference=float(np.nanmean(precise_errors)))
        return updated, tensor
    scan_ctmc.prepare = compare
    sys.argv = helper['command'](args, 'joint', args.outdir)
    cli.main()
    (args.outdir / 'asr_comparison.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
