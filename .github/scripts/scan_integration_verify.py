#!/usr/bin/env python3
"""Audit matched inputs, repeatability, maximum-score P values and legacy parity."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(args):
    from csubst import scan_statistics
    output = dict(repeatability=[], common_null_alignments=[], pvalue_checks=[], legacy_parity=[])
    frames = {}
    for name in ('baseline','joint','bridge'):
        for calibration in ('none','parametric_bootstrap'):
            reference = None
            for repeat in range(1,4):
                directory = args.benchmark/f'{name}-{calibration}-r{repeat}'
                frame = pd.read_csv(directory/'csubst_scan.tsv',sep='\t')
                stable = frame.drop(columns=['scan_bootstrap_directory'],errors='ignore')
                if reference is None:
                    reference = stable
                else:
                    pd.testing.assert_frame_equal(reference,stable,check_exact=True)
                if calibration == 'parametric_bootstrap':
                    manifest = next(directory.glob('csubst_scan_bootstrap_*/manifest.json'))
                    record = json.loads(manifest.read_text())
                    assert record['failure_count'] == 0 and record['success_count'] == args.niter
                    maxima = [-np.inf if r['maximum_score'] is None else r['maximum_score'] for r in record['replicates']]
                    expected = [scan_statistics.empirical_pvalue(v,maxima,args.niter) for v in frame.score_rate_enrichment]
                    np.testing.assert_allclose(frame.p_rate_enrichment_bootstrap_maxT,expected,rtol=1e-6,atol=0)
                    output['pvalue_checks'].append(dict(mode=name,repeat=repeat,rows=len(frame),verified=True))
                if repeat == 1:
                    frames[name,calibration] = frame
            output['repeatability'].append(dict(mode=name,calibration=calibration,exact=True))
    for replicate in range(args.niter):
        hashes = []
        model_arrays = []
        for name in ('baseline','joint','bridge'):
            directory = next((args.benchmark/f'{name}-parametric_bootstrap-r1').glob('csubst_scan_bootstrap_*'))
            hashes.append(digest(directory/f'rep{replicate:06d}/input.fa'))
            with np.load(directory/'model.npz') as model:
                model_arrays.append((model['q'].copy(),model['pi'].copy(),model['topology'].item()))
        assert len(set(hashes)) == 1
        for arrays in model_arrays[1:]:
            for observed,expected in zip(arrays,model_arrays[0]):
                np.testing.assert_array_equal(observed,expected)
        output['common_null_alignments'].append(dict(replicate=replicate,sha256=hashes[0],models_identical=True))
    run = runpy.run_path(str(ROOT/'.github/scripts/scan_integration_benchmark.py'))['run']
    for calibration in ('none','parametric_bootstrap'):
        directory = args.outdir/('integrated-marginal-'+calibration)
        run(ROOT,'marginal',calibration,args,directory)
        actual = pd.read_csv(directory/'csubst_scan.tsv',sep='\t')
        expected = frames['baseline',calibration]
        columns = [c for c in expected if c != 'scan_bootstrap_directory']
        pd.testing.assert_frame_equal(expected[columns],actual[columns],check_exact=True)
        output['legacy_parity'].append(dict(calibration=calibration,rows=len(actual),columns=len(columns),exact=True))
    (args.outdir/'verification.json').write_text(json.dumps(output,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('benchmark','alignment','fit','outdir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--niter',type=int,default=3)
    args=parser.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)
    verify(args)


if __name__=='__main__':
    main()
