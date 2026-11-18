"""Simulate and refit a panel of conditions using bundled tree templates."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
import shutil
import subprocess
import sys

from csubst import ete
from csubst.main_simulate import read_fasta


REPORT = Path(__file__).resolve().parent
REPO = REPORT.parents[1]
EXTENSIONS = ('treefile', 'state', 'rate', 'iqtree', 'log')
PGK_SEEDS = (101, 202, 303, 404)
PEPC_SEEDS = (101, 202, 303)
MODEL = 'ECMK07_GY0K+F'
SEARCH = (
    '--max_arity', '2', '--exhaustive_until', '2', '--exclude_sister_pair', 'no',
    '--output_stat', 'any2spe', '--float_digit', '12', '--b', 'no',
)


@dataclass(frozen=True)
class Condition:
    sites: int = 520
    length: float = 1.0
    asrv: bool = False
    omega: float = 0.2
    foreground_omega: float = 0.2
    percent: int = 0
    target: str = 'random0'
    foreground_length: float = 1.0


BASE = Condition()
BIAS = replace(BASE, percent=20, target='random1')
CONDITIONS = {
    'baseline': BASE,
    'biased': BIAS,
    'short_alignment': replace(BASE, sites=120),
    'short_alignment_bias': replace(BIAS, sites=120),
    'long_alignment': replace(BASE, sites=1400),
    'short_branches': replace(BASE, length=0.3),
    'short_branches_bias': replace(BIAS, length=0.3),
    'long_branches': replace(BASE, length=2.5),
    'long_branches_bias': replace(BIAS, length=2.5),
    'rate_mixture': replace(BASE, asrv=True),
    'rate_mixture_bias': replace(BIAS, asrv=True),
    'strong_constraint': replace(BASE, omega=0.05),
    'weak_constraint': replace(BASE, omega=0.6),
    'foreground_acceleration': replace(BASE, percent=20, foreground_length=3.0),
    'foreground_relaxation': replace(BASE, percent=20, foreground_omega=0.8),
}
PEPC_CONDITIONS = ('baseline', 'biased', 'long_branches', 'rate_mixture')
UNIFORM_REFIT = ('baseline', 'short_alignment', 'long_branches', 'rate_mixture')


def run_cli(cwd, log_name, args):
    log = cwd / log_name
    with log.open('w') as handle:
        subprocess.run(
            [sys.executable, '-m', 'csubst', *map(str, args)],
            cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=True,
        )


def prepare(work, dataset):
    sources = work / dataset / 'inputs'
    sources.mkdir(parents=True)
    files = REPO / 'csubst' / 'dataset'
    for filename in (f'{dataset}.tree.nwk', f'{dataset}.alignment.fa',
                     *(f'{dataset}.alignment.fa.{ext}' for ext in EXTENSIONS)):
        shutil.copyfile(files / filename, sources / filename)
    foreground = sources / 'foreground.txt'
    if dataset == 'PEPC':
        shutil.copyfile(files / 'PEPC.foreground.txt', foreground)
    else:
        tree = ete.PhyloNode(str(sources / 'PGK.tree.nwk'), format=1)
        leaves = [node for node in tree.traverse('preorder') if ete.is_leaf(node)]
        indices = [(2 * i + 1) * len(leaves) // 16 for i in range(8)]
        foreground.write_text(''.join(f'{i}\t{leaves[j].name}\n' for i, j in enumerate(indices, 1)))
    num_sites = len(next(iter(read_fasta(sources / f'{dataset}.alignment.fa').values()))) // 3
    # Equal weights and an exact mean of one within every complete four-site cycle.
    site_rates = (0, 0.2, 0.8, 3)
    (sources / 'synthetic.rate').write_text(
        'Site\tRate\n' + ''.join(f'{i+1}\t{site_rates[i % 4]}\n' for i in range(num_sites))
    )
    return sources


def simulate(cwd, dataset, sources, condition, seed):
    original = sources / f'{dataset}.alignment.fa'
    rate = sources / 'synthetic.rate' if condition.asrv else Path(str(original) + '.rate')
    args = ['simulate', '--alignment_file', original, '--rooted_tree_file',
            sources / f'{dataset}.tree.nwk', '--foreground', sources / 'foreground.txt']
    for ext in EXTENSIONS:
        args += ['--iqtree_' + ext, rate if ext == 'rate' else str(original) + '.' + ext]
    run_cli(cwd, 'simulate.stdout', [
        *args, '--num_simulated_site', condition.sites, '--simulate_seed', seed,
        '--background_omega', condition.omega, '--foreground_omega', condition.foreground_omega,
        '--percent_foreground_sites', condition.percent, '--target_amino_acids', condition.target,
        '--target_bias_strength', 90, '--tree_branch_length_scale', condition.length,
        '--foreground_branch_length_scale', condition.foreground_length, '--simulate_asrv',
        'file' if condition.asrv else 'no', '--export_expected_omegaC', 'yes',
        '--outdir', 'sim', '--output_prefix', 'sim',
    ])


def search(cwd, model, seed, iqtree, variant):
    run_cli(cwd, f'{variant}.stdout', [
        'search', '--alignment_file', 'sim/sim.fa', '--rooted_tree_file', 'sim/sim_true_asr.treefile',
        '--iqtree_exe', iqtree, '--iqtree_model', model, '--iqtree_outdir', f'iqtree-{variant}',
        '--random_seed', seed, '--outdir', variant, *SEARCH,
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('--iqtree', default='iqtree')
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    inputs = {dataset: prepare(work, dataset) for dataset in ('PGK', 'PEPC')}
    panel = [
        (dataset, scenario, seed)
        for dataset, scenarios, seeds in (
            ('PGK', CONDITIONS, PGK_SEEDS),
            ('PEPC', PEPC_CONDITIONS, PEPC_SEEDS),
        )
        for seed in seeds for scenario in scenarios
    ]
    errors = []
    with ThreadPoolExecutor(args.jobs) as workers:
        refits = {}
        for dataset, scenario, seed in panel:
            cwd = work / dataset / f'{scenario}-{seed}'
            cwd.mkdir()
            task = workers.submit(simulate, cwd, dataset, inputs[dataset], CONDITIONS[scenario], seed)
            refits[task] = dataset, scenario, seed, cwd
        for task in as_completed(refits):
            dataset, scenario, seed, cwd = refits[task]
            try:
                task.result()
            except subprocess.CalledProcessError:
                errors.append(str(cwd / 'simulate.stdout'))
        fits = {}
        for dataset, scenario, seed, cwd in refits.values():
            if str(cwd / 'simulate.stdout') in errors:
                continue
            variants = [('default', MODEL + '+R4')]
            if dataset == 'PGK' and scenario in UNIFORM_REFIT:
                variants.append(('uniform', MODEL))
            for variant, model in variants:
                task = workers.submit(search, cwd, model, seed, args.iqtree, variant)
                fits[task] = cwd, variant
        for task in as_completed(fits):
            cwd, variant = fits[task]
            try:
                task.result()
            except subprocess.CalledProcessError:
                errors.append(str(cwd / f'{variant}.stdout'))
                print(f'failed {cwd.parent.name}/{cwd.name}/{variant}', flush=True)
            else:
                print(f'finished {cwd.parent.name}/{cwd.name}/{variant}', flush=True)
    if errors:
        raise SystemExit('\n'.join(errors))


if __name__ == '__main__':
    main()
