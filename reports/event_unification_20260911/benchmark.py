"""Compare original search paths restored with benchmark-baseline.patch to current.

Each CLI run is a fresh process, one BLAS thread, default joint, same PGK/PEPC fit.
The reverse patch restores the pre-unification paths exercised by this benchmark;
this avoids comparing against the older git HEAD, which lacks earlier work.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    data = ROOT/'csubst/dataset'
    results = []
    for dataset in ['PGK','PEPC']:
        for repeat in range(4):
            for label, source in ([('before',args.baseline_root),('after',ROOT)] if repeat % 2 == 0 else [('after',ROOT),('before',args.baseline_root)]):
                out = args.workdir/f'{dataset}-{repeat}-{label}'
                command = [sys.executable,'-m','csubst','search','--alignment_file',str(data/(dataset+'.alignment.fa')),
                           '--rooted_tree_file',str(data/(dataset+'.tree.nwk')),'--outdir',str(out),
                           '--threads','1','--blas_threads','1','--max_arity','2','--calibrate_longtail','no','--random_seed','8']
                for suffix in ['state','treefile','rate','iqtree','log']:
                    command += ['--iqtree_'+suffix,str(data/(dataset+'.alignment.fa.'+suffix))]
                start = time.perf_counter()
                with (args.workdir/(out.name+'.log')).open('w') as handle:
                    result = subprocess.run(['/usr/bin/time','-l']+command, cwd=source, stdout=handle, stderr=subprocess.PIPE, text=True,
                                            env=dict(os.environ,PYTHONPATH=str(source),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
                if result.returncode:
                    raise RuntimeError(result.stderr)
                elapsed = time.perf_counter()-start
                rss = next(int(line.split()[0]) for line in result.stderr.splitlines() if 'maximum resident set size' in line)
                results.append(dict(dataset=dataset, repeat=repeat, phase=label, seconds=elapsed, peak_rss_bytes=rss))
            # Compare scientific columns on eligible rows; NA/coverage are intentional schema changes.
            before = pd.read_csv(args.workdir/f'{dataset}-{repeat}-before/csubst_cb_2.tsv',sep='\t')
            after = pd.read_csv(args.workdir/f'{dataset}-{repeat}-after/csubst_cb_2.tsv',sep='\t')
            cols = [c for c in before if c.startswith(('OCS','OCN','ECS','ECN','omegaC')) and c in after]
            for col in cols:
                a,b=before[col].to_numpy(float),after[col].to_numpy(float)
                keep = (after['S_eligible_count'].to_numpy() > 0) & (after['N_eligible_count'].to_numpy() > 0)
                np.testing.assert_allclose(a[keep],b[keep],atol=1e-10,rtol=1e-10)
    summary=[]
    for dataset in ['PGK','PEPC']:
        for phase in ['before','after']:
            rows=[r for r in results if r['dataset']==dataset and r['phase']==phase and r['repeat']>0]
            summary.append(dict(dataset=dataset,phase=phase,median_seconds=statistics.median(r['seconds'] for r in rows),
                                median_peak_rss_bytes=statistics.median(r['peak_rss_bytes'] for r in rows)))
    (args.workdir/'timings.json').write_text(json.dumps(dict(runs=results,summary=summary),indent=2)+'\n')


if __name__=='__main__':
    main()
