"""Audit exact-1 hybrid fits with optimizer restarts from fixed starting values."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import re
import subprocess

import pandas as pd

from run import EXTENSIONS, MODEL, SEARCH, run_cli
from summarize import FIT, SIM, read_pairs


def likelihood(report):
    return float(re.search(r'^Log-likelihood of the tree:\s*(\S+)', report.read_text(), re.M)[1])


def restart(target):
    rep, iqtree, label, start = target
    output = rep / f'iqtree-restart-{label}'
    output.mkdir(exist_ok=True)
    prefix = output / 'fit'
    model_args = ['-m', MODEL + '+R4'] if start is None else [
        '-m', f'ECMK07_GY0K{{{start}}}+F+R4', '-optfromgiven',
    ]
    if label == 'bounded':
        model_args += ['-optalg', '2-BFGS-B,EM']
    if label == 'freq_floor':
        model_args += ['--inc-zero-freq', '--min-freq', '0.0001']
    new_report = prefix.with_suffix('.iqtree')
    if not new_report.exists():
        with (output / 'stdout').open('w') as handle:
            subprocess.run([
                iqtree, '-s', 'sim/sim.fa', '-te', 'sim/sim_true_asr.treefile', '--seqtype', 'CODON1',
                *model_args, '-T', '1', '--ancestral', '--rate', '-v', '--prefix', str(prefix),
                '-seed', rep.name.rsplit('-', 1)[1],
            ], cwd=rep, stdout=handle, stderr=subprocess.STDOUT, check=True)
    search_dir = rep / f'restart-{label}'
    if not search_dir.exists():
        paths = [value for ext in EXTENSIONS for value in (f'--iqtree_{ext}', str(prefix) + '.' + ext)]
        try:
            run_cli(rep, f'restart-{label}.stdout', [
                'search', '--alignment_file', 'sim/sim.fa', '--rooted_tree_file', 'sim/sim_true_asr.treefile',
                '--outdir', f'restart-{label}', '--random_seed', rep.name.rsplit('-', 1)[1], *paths, *SEARCH,
            ])
        except subprocess.CalledProcessError:
            if label != 'freq_floor':
                raise
    sim, meta = read_pairs(rep, 'default')
    fitted_path = search_dir / 'csubst_cb_2.tsv'
    counts = sim.loc[sim.valid, SIM + FIT].sum().rename(index={c: c + '_default' for c in FIT}).to_dict()
    manifest = search_dir / 'csubst_search_run.json'
    if manifest.exists() and json.loads(manifest.read_text())['status'] == 'complete':
        fitted = pd.read_csv(fitted_path, sep='\t')
        data = sim.merge(fitted[['branch_id_1', 'branch_id_2', *FIT]], on=['branch_id_1', 'branch_id_2'],
                         validate='one_to_one', suffixes=('_default', '_restart'))
        assert len(data) == len(sim) == len(fitted)
        counts.update(data.loc[data.valid, [c + '_restart' for c in FIT]].sum().to_dict())
        status = 'complete'
    else:
        status = (search_dir / 'csubst.log').read_text().splitlines()[-1].strip()
        if label != 'freq_floor' or status != 'Precise log/report fitted frequency mismatch.':
            raise RuntimeError(f'Search did not complete in {search_dir}: {status}')
    old_report, = (rep / 'iqtree-default').glob('*.iqtree')
    with gzip.open(output / 'fit.ckp.gz', 'rt') as stream:
        omega = float(re.search(r'^\s+omega:\s*(\S+)', stream.read(), re.M)[1])
    scenario, seed = rep.name.rsplit('-', 1)
    return dict(
        dataset=rep.parent.name, scenario=scenario, seed=int(seed), method=label, start=start,
        iqtree_version=new_report.read_text().splitlines()[0].split()[1], search_status=status,
        omega_default=meta['parameter_source']['omega'], omega_restart=omega,
        log_likelihood_default=likelihood(old_report), log_likelihood_restart=likelihood(new_report), **counts,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('work', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--iqtree', default='iqtree')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--reps', nargs='+')
    p.add_argument('--methods', nargs='+', choices=('low', 'high', 'bounded', 'standard', 'freq_floor'),
                   default=('low', 'high', 'bounded'))
    args = p.parse_args()
    reps = []
    for manifest in sorted(args.work.glob('*/*/default/csubst_endpoint_model.json')):
        rep = manifest.parents[1]
        if args.reps:
            if f'{rep.parent.name}/{rep.name}' not in args.reps:
                continue
        else:
            value = json.loads(manifest.read_text())['codon']['parameter_source']
            if value['model'] != MODEL + '+R4' or value['omega'] != 1:
                continue
        reps.append(rep)
    jobs = []
    for rep in reps:
        for label in args.methods:
            start = {'low': 0.25, 'high': 2.0}.get(label)
            jobs.append((rep.resolve(), args.iqtree, label, start))
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(restart, jobs))
    data = pd.DataFrame(results)
    if not data.empty:
        data.to_csv(args.output, sep='\t', index=False, na_rep='NA', float_format='%.10g')


if __name__ == '__main__':
    main()
