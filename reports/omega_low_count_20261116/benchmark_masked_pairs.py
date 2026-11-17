"""Alternate two observed-pair kernels through search on one fitted PEPC replicate."""

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import resource
import runpy
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from pilot import EXPECTED, KEY, OBS


def worker(args):
    implementation, alignment, mode = args
    if implementation != 'current':
        name = 'csubst.endpoint_pair'
        spec = importlib.util.spec_from_file_location(name, implementation)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        importlib.import_module('csubst').endpoint_pair = module
    outdir = mode
    argv = ['csubst', 'search', '--alignment_file', alignment, '--rooted_tree_file', 'tree.nwk',
            '--b', 'no', '--float_digit', '12', '--output_stat', 'any2spe',
            '--pair_posterior', 'joint', '--exclude_sister_pair', 'no', '--outdir', outdir,
            '--threads', '1', '--blas_threads', '1']
    for ext in ('iqtree', 'state', 'rate', 'treefile', 'log'):
        argv += ['--iqtree_' + ext, 'fit.' + ext]
    sys.argv = argv
    runpy.run_module('csubst', run_name='__main__')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss *= 1 if sys.platform == 'darwin' else 1024
    Path(mode + '.json').write_text(json.dumps({'peak_rss_mib': rss / 2**20}))


def run(root, alignment, implementation, mode):
    start = perf_counter()
    with (root / (mode + '.log')).open('w') as log:
        subprocess.run([sys.executable, __file__, '--worker', str(implementation), alignment, mode],
                       cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    wall = perf_counter() - start
    metrics = json.loads((root / (mode + '.json')).read_text())
    return dict(wall_s=wall, **metrics)


def verify(root, prev, current):
    before, after = (pd.read_csv(root / mode / 'csubst_cb_2.tsv', sep='\t').sort_values(KEY)
                     for mode in (prev, current))
    np.testing.assert_array_equal(before[KEY], after[KEY])
    np.testing.assert_array_equal(before[EXPECTED], after[EXPECTED])
    counts_and_rates = OBS + ['dNCany2spe', 'dSCany2spe', 'omegaCany2spe']
    np.testing.assert_allclose(before[counts_and_rates], after[counts_and_rates], rtol=1e-9, atol=6e-12)
    return len(before), float(np.abs(before[OBS] - after[OBS]).to_numpy().max())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case', type=Path)
    p.add_argument('work', type=Path)
    p.add_argument('reference', type=Path)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--conditions', choices=['complete', 'masked'], nargs='+', default=['complete', 'masked'])
    args = p.parse_args()
    case, work, reference = args.case.resolve(), args.work.resolve(), args.reference.resolve()
    results = []
    for condition in args.conditions:
        root = work / condition
        root.mkdir(parents=True)
        names = [case / 'fit.iqtree'] if condition == 'complete' else list((case / 'masked_fit').glob('*.iqtree'))
        report, = names
        alignment = condition + '.fa'
        for name in (alignment, 'tree.nwk'):
            shutil.copyfile(case / name, root / name)
        for ext in ('iqtree', 'state', 'rate', 'treefile', 'log', 'ckp.gz'):
            prefix = str(report).removesuffix('.iqtree') + '.'
            shutil.copyfile(prefix + ext, root / ('fit.' + ext))
        for repeat in range(args.repeats + 1):
            modes = [('before', reference), ('after', 'current')]
            if repeat % 2 == 0 and repeat:
                modes.reverse()
            rows = []
            for mode, impl in modes:
                metrics = run(root, alignment, impl, f'{mode}-{repeat}')
                rows.append(dict(condition=condition, repeat=repeat, mode=mode, **metrics))
                print(rows[-1], flush=True)
            pairs, error = verify(root, f'before-{repeat}', f'after-{repeat}')
            results.extend(dict(row, pairs=pairs, max_obs_difference=error) for row in rows)
    pd.DataFrame(results).to_csv(work / 'timing.tsv', sep='\t', index=False, float_format='%.12g')


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--worker':
        worker(sys.argv[2:])
    else:
        main()
