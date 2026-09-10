#!/usr/bin/env python3
"""Rerun ASR, recoding, ASRV and selection on independent simulated datasets.

See docs/URN_CALIBRATION.md for the manifest contract. Inputs are copied into
fresh run directories so IQ-TREE cannot write next to supplied alignments.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from csubst import pipeline_calibration  # noqa: E402 -- select this checkout before importing


def validate_manifest(manifest, base_dir):
    if manifest.get('schema_version') != 1:
        raise ValueError('Expected manifest schema_version 1.')
    if not manifest.get('simulator', {}).get('independent_replicates'):
        raise ValueError('The simulator contract must declare independent_replicates=true.')
    ids = set()
    paths = set()
    for row in manifest['replicates']:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', row['id']) or row['id'] in ids:
            raise ValueError('Replicate IDs must be unique safe directory names.')
        ids.add(row['id'])
        if row['role'] not in ('calibration', 'validation') or row['truth'] not in ('null', 'alternative'):
            raise ValueError('Invalid replicate role/truth.')
        if row['role'] == 'calibration' and row['truth'] != 'null':
            raise ValueError('Calibration replicates must be null.')
        for key in ('alignment', 'tree', 'foreground', 'training_alignment'):
            if row.get(key):
                row[key] = str((base_dir / row[key]).resolve())
                if not Path(row[key]).is_file():
                    raise ValueError('Missing input {}'.format(row[key]))
        if row['alignment'] in paths:
            raise ValueError('A dataset cannot be reused as an independent replicate.')
        paths.add(row['alignment'])
    names = [c['name'] for c in manifest['configurations']]
    if not names or len(set(names)) != len(names) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', n) for n in names):
        raise ValueError('Configuration names must be unique safe directory names.')
    if not any(r['role'] == 'calibration' for r in manifest['replicates']):
        raise ValueError('At least one calibration replicate is required.')
    if not any(r['role'] == 'validation' and r['truth'] == 'null' for r in manifest['replicates']):
        raise ValueError('At least one separate null validation replicate is required.')
    selection = manifest['selection']
    if not selection['statistic'].startswith('omegaC'):
        raise ValueError('Declare an omegaC statistic for pipeline selection.')
    if int(selection['arity']) < 2:
        raise ValueError('Selection arity must be at least two.')
    for key in ('branch_ids', 'exclude_branch_ids'):
        if key in selection:
            values = selection[key]
            if (not isinstance(values, list) or any(type(i) is not int or i < 0 for i in values)
                    or len(set(values)) != len(values)):
                raise ValueError('Selection branch IDs must be unique nonnegative integers.')
    if 'branch_ids' in selection and len(selection['branch_ids']) != int(selection['arity']):
        raise ValueError('Prespecified branch IDs must match the selection arity.')
    for column, value in selection.get('minimum_counts', {}).items():
        if not column.startswith(('OCN', 'OCS')) or not float(value) >= 0:
            raise ValueError('Eligibility must use nonnegative observed-count thresholds.')
    return manifest


def run_configuration(row, config, common, selection, outdir, seed):
    run_dir = outdir / row['id'] / config['name']
    run_dir.mkdir(parents=True, exist_ok=False)
    options = dict(common, **config.get('options', {}))
    reserved = {'alignment_file', 'rooted_tree_file', 'outdir', 'output_prefix', 'iqtree_outdir',
                'iqtree_treefile', 'iqtree_state', 'iqtree_rate', 'iqtree_iqtree', 'foreground',
                'random_seed', 'calc_omega_pvalue', 'max_arity'}
    if reserved.intersection(options):
        raise ValueError('Run-local input/output/seed options cannot be overridden.')
    options.setdefault('expectation_method', 'urn')
    if options['expectation_method'] != 'urn':
        raise ValueError('This validation runner is for urn analyses.')
    options.setdefault('pseudocount_mode', 'none')
    options.setdefault('calibrate_longtail', 'no')
    options.setdefault('output_stat', selection['statistic'].removeprefix('omegaC').removesuffix('_nocalib'))
    options.setdefault('threads', 1)
    options.setdefault('asrv_report', True)
    shutil.copyfile(row['alignment'], run_dir / 'input.fa')
    shutil.copyfile(row['tree'], run_dir / 'input.nwk')
    options.update(alignment_file=str(run_dir / 'input.fa'), rooted_tree_file=str(run_dir / 'input.nwk'),
                   outdir=str(run_dir), output_prefix='csubst', iqtree_outdir=str(run_dir / 'iqtree'),
                   random_seed=seed, calc_omega_pvalue=False, max_arity=int(selection['arity']))
    if row.get('foreground'):
        shutil.copyfile(row['foreground'], run_dir / 'foreground.tsv')
        options['foreground'] = str(run_dir / 'foreground.tsv')
    if row.get('training_alignment'):
        # Only automatic methods consume a separate learning alignment.
        if options.get('nonsyn_recode') in ('srchisq6', 'kgbauto6'):
            options['nonsyn_recode_training_alignment'] = row['training_alignment']
    cmd = [sys.executable, '-m', 'csubst', 'search']
    for key, value in options.items():
        if not re.fullmatch(r'[a-z][a-z0-9_]*', key):
            raise ValueError('Invalid CLI option name: {}'.format(key))
        cmd.extend(['--' + key, ('yes' if value else 'no') if isinstance(value, bool) else str(value)])
    (run_dir / 'command.json').write_text(json.dumps(cmd, indent=2) + '\n')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(REPO_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    with (run_dir / 'process.log').open('w') as log:
        subprocess.run(cmd, cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    table = pd.read_csv(run_dir / 'csubst_cb_{}.tsv'.format(selection['arity']), sep='\t')
    return pipeline_calibration.selected_statistic(table, selection)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--level', type=float, default=.05)
    parser.add_argument('--fpr-limit', type=float, default=.06)
    args = parser.parse_args()
    manifest = validate_manifest(json.loads(args.manifest.read_text()), args.manifest.resolve().parent)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    records = []
    for row in manifest['replicates']:
        seed = int.from_bytes(hashlib.sha256(row['id'].encode()).digest()[:4], 'big') % (2**31 - 1)
        scores = {}
        for config in manifest['configurations']:
            print('{} / {}'.format(row['id'], config['name']), flush=True)
            scores[config['name']] = run_configuration(
                row, config, manifest.get('analysis_options', {}), manifest['selection'], outdir, seed,
            )
        records.append(dict(id=row['id'], role=row['role'], truth=row['truth'],
                            selected_statistic=max(scores.values()), **{'score_' + k: v for k, v in scores.items()}))
        pd.DataFrame(records).to_csv(outdir / 'replicate_scores.tsv', sep='\t', index=False)
    validation, summary = pipeline_calibration.summarize_validation(records, args.level, args.fpr_limit)
    pd.DataFrame(validation).to_csv(outdir / 'validation.tsv', sep='\t', index=False)
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
