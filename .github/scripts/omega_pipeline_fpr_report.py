#!/usr/bin/env python3
"""Audit completed full-pipeline #46 outputs and export compact evidence."""

import argparse
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from omega_pipeline_fpr import aggregate, sha256


def audit_table(frame, metrics, production_stats, arity):
    p = frame['pomegaCany2spe'].to_numpy(float)
    q = frame['qomegaCany2spe'].to_numpy(float)
    finite = np.isfinite(p)
    order = np.argsort(p[finite])
    sorted_p = p[finite][order]
    ranked = sorted_p * len(sorted_p) / np.arange(1, len(sorted_p) + 1)
    expected = np.minimum(1., np.minimum.accumulate(ranked[::-1])[::-1])
    if not np.allclose(q[finite][order], expected, atol=1e-10, rtol=0):
        raise AssertionError('Production Q differs from independent BH calculation')
    if not np.isnan(q[~finite]).all():
        raise AssertionError('Finite Q without finite P')
    selected = production_stats.loc[production_stats.arity == int(arity), 'num_qualified_all']
    if len(selected) != 1 or int(selected.iloc[0]) != metrics['selected']:
        raise AssertionError('Production selection differs from output-based classification')


def collect(workdir, outdir):
    metadata = json.loads((workdir / 'metadata.json').read_text())
    failures = json.loads((workdir / 'failures.json').read_text())
    if failures:
        raise AssertionError('Failed replicates must be resolved and reported')
    with gzip.open(workdir / 'records.json.gz', 'rt') as handle:
        records = json.load(handle)
    expected = {(regime, i) for regime in metadata['scale_by_regime']
                for i in range(metadata['replicates_per_regime'])}
    observed = [(r['regime'], r['replicate']) for r in records]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise AssertionError('Wrong independent-replicate population')
    tables = []
    for r in records:
        directory = workdir / r['regime'] / ('replicate_' + str(r['replicate']).zfill(4))
        if sha256(directory / 'alignment.fa') != r['alignment_sha256']:
            raise AssertionError('Alignment changed after analysis')
        for ext, checksum in r['fit']['hashes'].items():
            if sha256(directory / ('fit.' + ext)) != checksum:
                raise AssertionError('IQ-TREE fitted artifact changed after analysis')
        if set(r['settings']) != set(metadata['settings']):
            raise AssertionError('Missing analysis setting')
        for setting, result in r['settings'].items():
            output = directory / setting
            stats = pd.read_csv(output / 'csubst_cb_stats.tsv', sep='\t')
            for arity, metrics in result['arities'].items():
                path = output / ('csubst_cb_' + arity + '.tsv')
                if sha256(path) != metrics['sha256']:
                    raise AssertionError('CSUBST output changed after analysis')
                frame = pd.read_csv(path, sep='\t')
                audit_table(frame, metrics, stats, arity)
                columns = [c for c in frame if c.startswith('branch_id_')]
                columns += [c for c in frame if c in ('OCNany2spe', 'OCSany2spe',
                            'ECNany2spe', 'ECSany2spe', 'omegaCany2spe',
                            'pomegaCany2spe', 'qomegaCany2spe',
                            'pvalue_n_any2spe', 'pvalue_undefined_any2spe')]
                compact = frame[columns].copy()
                compact.insert(0, 'arity', int(arity))
                compact.insert(0, 'setting', setting)
                compact.insert(0, 'replicate', r['replicate'])
                compact.insert(0, 'regime', r['regime'])
                tables.append(compact)
    # Same arity-dependent family and trial-level rejection logic as the runner;
    # BH and production selection were independently checked above.
    summary = aggregate(records)
    stored = json.loads((workdir / 'summary.json').read_text())
    if summary != stored['rows']:
        raise AssertionError('Saved aggregate differs from underlying replicates')
    outdir.mkdir(parents=True, exist_ok=True)
    for filename in ('metadata.json', 'summary.json', 'records.json.gz', 'failures.json'):
        shutil.copy2(workdir / filename, outdir / filename)
    pd.concat(tables, ignore_index=True).to_csv(outdir / 'tested_rows.tsv.gz', sep='\t',
                                              index=False, compression={'method': 'gzip', 'mtime': 0})
    with gzip.GzipFile(filename=str(outdir / 'simulated_alignments.fasta.gz'), mode='wb', mtime=0) as handle:
        for r in records:
            directory = workdir / r['regime'] / ('replicate_' + str(r['replicate']).zfill(4))
            for line in (directory / 'alignment.fa').read_text().splitlines():
                if line.startswith('>'):
                    line = '>' + r['regime'] + '.' + str(r['replicate']) + '.' + line[1:]
                handle.write((line + '\n').encode())
    generator = {regime: json.loads((workdir / regime / 'replicate_0000/generator.json').read_text())
                 for regime in metadata['scale_by_regime']}
    (outdir / 'generator.json').write_text(json.dumps(generator, indent=2) + '\n')
    audit = dict(independent_alignments=len(records), fitted_models=len(records),
                 search_runs=sum(len(r['settings']) for r in records), output_tables=len(tables),
                 tested_rows=sum(len(t) for t in tables), failures=0,
                 independent_BH_agreement=True, production_selection_agreement=True,
                 fitted_artifact_hashes_verified=True, csubst_output_hashes_verified=True)
    (outdir / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return metadata, records, summary, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    _, _, _, audit = collect(args.workdir, args.outdir)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
