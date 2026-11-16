"""Measure the current search and exact any2spe kernels on saved fitted inputs."""

import argparse
import json
from pathlib import Path
import platform
import resource
import runpy
import shutil
import subprocess
import sys
import time

import pandas as pd


FITS = {
    'pgk': ('PGK/baseline-101', 'sim', 'iqtree-default'),
    'pepc': ('PEPC/rate_mixture-101', 'sim', 'iqtree-default'),
    'pepc1000': ('null-101', '.', 'hybrid_fit'),
}
EXTENSIONS = ('iqtree', 'log', 'rate', 'state', 'treefile', 'ckp.gz')


def prepare_inputs(work, panel, pepc_fit):
    for name, (rep, seq_folder, fit_folder) in FITS.items():
        source = (pepc_fit if name == 'pepc1000' else panel) / rep
        target = work / name / 'inputs'
        if target.is_dir():
            continue
        target.mkdir(parents=True)
        report, = (source / fit_folder).glob('*.iqtree')
        for src, dest in [('sim.fa', 'sim.fa'), ('sim_true_asr.treefile', 'tree.nwk')]:
            shutil.copyfile(source / seq_folder / src, target / dest)
        prefix = str(report).removesuffix('.iqtree')
        for item in EXTENSIONS:
            shutil.copyfile(prefix + '.' + item, target / ('fit.' + item))


def search_worker(spec):
    argv = [
        'csubst', 'search', '--alignment_file', 'inputs/sim.fa', '--rooted_tree_file', 'inputs/tree.nwk',
        '--max_arity', '2', '--threads', '1', '--blas_threads', '1', '--float_digit', '12',
        '--random_seed', '101', '--outdir', spec['outdir'], '--b', spec['b'],
        '--output_stat', spec['stat'],
    ]
    for ext in EXTENSIONS[:-1]:
        argv += ['--iqtree_' + ext, 'inputs/fit.' + ext]
    sys.argv = argv
    runpy.run_module('csubst', run_name='__main__')


def worker(spec):
    start = time.perf_counter()
    if spec['mode'] == 'search':
        search_worker(spec)
    else:
        from joint_pair_benchmark_kernel import benchmark
        benchmark(spec)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        rss *= 1024
    Path(spec['measure']).write_text(json.dumps({
        'elapsed': time.perf_counter() - start, 'peak_rss_mib': rss / 1024**2,
    }))


def run_one(work, dataset, mode, repeat, block_size=4):
    case = work / dataset
    tag = f'{mode}-{repeat}'
    spec = {'mode': mode, 'outdir': tag, 'measure': tag + '.json', 'block_size': block_size,
            'stat': 'any2any,any2dif,any2spe' if dataset.endswith('-default') else 'any2spe',
            'b': 'yes' if dataset.endswith('-default') else 'no'}
    with (case / (tag + '.log')).open('w') as log:
        started = time.perf_counter()
        subprocess.run([sys.executable, __file__, '--worker', json.dumps(spec)], cwd=case,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        wall = time.perf_counter() - started
    record = json.loads((case / spec['measure']).read_text())
    return dict(dataset=dataset, mode=mode, repeat=repeat, wall_s=wall, **record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('work', type=Path)
    p.add_argument('panel', type=Path)
    p.add_argument('pepc_fit', type=Path)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--block-size', type=int, default=4)
    p.add_argument('--datasets', nargs='+', choices=(*FITS, 'pepc-default', 'pepc1000-default'))
    p.add_argument('--mode', choices=('baseline', 'compare', 'released', 'targeted', 'optimized', 'composed'),
                   default='baseline')
    args = p.parse_args()
    work = args.work.resolve()
    prepare_inputs(work, args.panel.resolve(), args.pepc_fit.resolve())
    for dataset in ('pepc', 'pepc1000'):
        dest = work / (dataset + '-default') / 'inputs'
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(work / dataset / 'inputs', target_is_directory=True)
    modes = {
        'baseline': ('search',), 'compare': ('search', 'exact'), 'released': ('search', 'released'),
        'targeted': ('search', 'targeted'), 'optimized': ('search', 'optimized'),
        'composed': ('search', 'composed'),
    }
    modes = modes[args.mode]
    measurements = []
    datasets = args.datasets or ((*FITS, 'pepc-default', 'pepc1000-default') if args.mode == 'baseline' else FITS)
    for dataset in datasets:
        for repeat in range(args.repeats + 1):
            order = modes if repeat % 2 else tuple(reversed(modes))
            if repeat == 0:
                order = modes
            for mode in order:
                row = run_one(work, dataset, mode, repeat, block_size=args.block_size)
                measurements.append(row)
                print('{} {} {} {:.3f} s {:.1f} MiB'.format(
                    dataset, mode, repeat, row['wall_s'], row['peak_rss_mib'],
                ), flush=True)
    pd.DataFrame(measurements).to_csv(work / (args.mode + '.tsv'), sep='\t', index=False, float_format='%.9g')
    (work / 'environment.json').write_text(json.dumps({
        'platform': platform.platform(), 'python': platform.python_version(),
    }) + '\n')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(json.loads(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == '--verify':
        from joint_pair_benchmark_kernel import check_replay
        check_replay(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == '--compare-tables':
        from joint_pair_benchmark_kernel import check_saved_totals
        check_saved_totals(sys.argv[2], sys.argv[3])
    else:
        main()
