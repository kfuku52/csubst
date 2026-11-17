"""Measure fixed-fit search time and peak RSS for both Wallenius urn settings."""

import argparse
import csv
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from time import perf_counter


def compare(work, source, cases, repeats):
    env = dict(os.environ, PYTHONPATH=str(source), MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', OMP_NUM_THREADS='1')
    darwin = platform.system() == 'Darwin'
    time_command = ['/usr/bin/time', '-l' if darwin else '-v']
    rss_pattern = (r'^\s*(\d+)\s+maximum resident set size' if darwin
                   else r'Maximum resident set size \(kbytes\):\s*(\d+)')
    rows = []
    for case, alignment, tree, prefix in cases:
        inputs = work / case / 'inputs'
        inputs.mkdir(parents=True)
        alignment, tree, prefix = [Path(p).resolve() for p in (alignment, tree, prefix)]
        for path, name in [(alignment, 'alignment.fa'), (tree, 'rooted.nwk')]:
            shutil.copyfile(path, inputs / name)
        checkpoint = Path(str(prefix) + '.ckp.gz')
        if checkpoint.is_file():
            shutil.copyfile(checkpoint, inputs / 'fit.ckp.gz')
        common = ['search', '--alignment_file', str(inputs/'alignment.fa'), '--rooted_tree_file', str(inputs/'rooted.nwk'),
                  '--expectation_method', 'urn', '--threads', '1', '--blas_threads', '1', '--b', 'no',
                  '--exclude_sister_pair', 'no', '--float_digit', '12', '--random_seed', '741']
        for ext in ('state', 'iqtree', 'log', 'rate', 'treefile'):
            path = inputs / ('fit.' + ext)
            shutil.copyfile(str(prefix) + '.' + ext, path)
            common.extend(('--iqtree_' + ext, str(path)))
        for rep in range(repeats + 1):
            for mode in (('each', 'empirical') if rep % 2 == 0 else ('empirical', 'each')):
                stem = work / case / f'{mode}-{rep}'
                log = stem.with_suffix('.log')
                start = perf_counter()
                with log.open('w') as handle:
                    subprocess.run([*time_command, sys.executable, '-m', 'csubst', *common,
                                    '--asrv', mode, '--outdir', str(stem)], cwd=work, env=env,
                                   stdout=handle, stderr=subprocess.STDOUT, check=True)
                wall = perf_counter() - start
                match = re.search(rss_pattern, log.read_text(), flags=re.MULTILINE)
                if not match:
                    raise ValueError(f'No peak RSS found in {log}.')
                mib = int(match[1]) / (1024 ** (2 if darwin else 1))
                rows.append(dict(case=case, mode=mode, repetition=rep, wall_seconds=wall, peak_mib=mib))
                print(case, mode, rep, f'{wall:.3f} s', f'{mib:.1f} MiB', flush=True)
    with (work / 'timings.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('--source', type=Path, default=Path.cwd(), help='Root containing the selected csubst package.')
    parser.add_argument('--case', nargs=4, action='append', required=True,
                        metavar=('LABEL','ALIGNMENT','ROOTED_TREE','FIT_PREFIX'))
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(exist_ok=True)
    compare(work, args.source.resolve(), args.case, args.repeats)


if __name__ == '__main__':
    main()
