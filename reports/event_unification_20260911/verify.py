"""Compare all reported N events across commands using identical IQ-TREE fits.

Run with --workdir outside the checkout. Requires IQ-TREE on PATH. Search asks
for spe2spe to retain inspectable full sparse events instead of projections.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from csubst import cli, endpoint_io, scan_ctmc, sequence, substitution  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    captured = {}
    original_build, original_scan = endpoint_io._build, scan_ctmc.prepare

    def capture_build(g, *args, **kwargs):
        original_build(g, *args, **kwargs)
        tensor = g['_endpoint_tensors']['N']
        captured.update(events=tensor.to_dense(), eligible=tensor.eligible.copy(),
                        q=g['instantaneous_codon_rate_matrix'].copy(), pi=g['equilibrium_frequency'].copy(),
                        precision=copy.deepcopy(g.get('fitted_model_provenance')))

    def capture_scan(g):
        updated, tensor = original_scan(g)
        captured.update(events=tensor.copy(), eligible=updated['event_eligible'].copy(),
                        q=g['instantaneous_codon_rate_matrix'].copy(), pi=g['equilibrium_frequency'].copy(),
                        precision=copy.deepcopy(g.get('fitted_model_provenance')))
        return updated, tensor

    endpoint_io._build, scan_ctmc.prepare = capture_build, capture_scan
    alignment, topology = work/'tips.fa', work/'tree.nwk'
    sequences = list(sequence.read_fasta(ROOT/'csubst/dataset/PGK.alignment.fa').values())[:4]
    sequences = [s[:900] for s in sequences]
    sequences[3] = '------' + sequences[3][6:]
    sequences = [s[:-3] + '---' for s in sequences]
    alignment.write_text(''.join(f'>Tip{i}\n{s}\n' for i, s in enumerate(sequences)))
    topology.write_text('((Tip0:.1,Tip1:.1):.1,(Tip2:.1,Tip3:.1):.1);\n')
    cases = []
    for model in ['GY+F', 'GY+FQ', 'GY+F+G4', 'MG+F3X4']:
        prefix = work/model.replace('+', '_')
        with (work/(prefix.name+'.fit.log')).open('w') as handle:
            subprocess.run(['iqtree', '-s', str(alignment), '-te', str(topology), '-st', 'CODON',
                            '-m', model, '-asr', '-wsr', '-nt', '1', '-seed', '8', '-v', '-redo', '-pre', str(prefix)],
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
        cases.append((model, alignment, topology, prefix))
    for name in ['PGK', 'PEPC']:
        data = ROOT/'csubst/dataset'
        cases.append((name, data/(name+'.alignment.fa'), data/(name+'.tree.nwk'), data/(name+'.alignment.fa')))
    summary = []
    for name, aln, tr, prefix in cases:
        names = list(sequence.read_fasta(aln))
        foreground = work/(name.replace('+','_')+'.foreground.tsv')
        foreground.write_text('1\t'+names[0]+'\n2\t'+names[2]+'\n')
        baseline = None
        for command in ['search', 'sites', 'scan']:
            out = work/(name.replace('+','_')+'_'+command)
            out.mkdir(exist_ok=True)
            options = [command, '--alignment_file', str(aln), '--rooted_tree_file', str(tr), '--outdir', str(out),
                       '--threads','1', '--blas_threads','1']
            for suffix in ['state','treefile','rate','iqtree','log']:
                options += ['--iqtree_'+suffix, str(prefix)+'.'+suffix]
            if command == 'search':
                options += ['--output_stat','any2any,spe2spe','--calibrate_longtail','no','--max_arity','2']
            elif command == 'sites':
                options += ['--branch_id','0,2','--tree_site_plot','no','--site_state_plot','no','--site_summary_plot','no']
            else:
                options += ['--foreground',str(foreground),'--scan_site_plot','no','--scan_pvalue_calibration','none',
                            '--scan_min_event_pp','1','--scan_min_support','10']
            captured.clear()
            start = time.perf_counter()
            cli._main(options)
            assert captured, (name, command)
            elapsed = time.perf_counter()-start
            if baseline is None:
                baseline = dict(captured)
                error = 0.
            else:
                np.testing.assert_array_equal(baseline['q'], captured['q'])
                np.testing.assert_array_equal(baseline['pi'], captured['pi'])
                np.testing.assert_array_equal(baseline['eligible'], captured['eligible'])
                np.testing.assert_allclose(baseline['events'], captured['events'], atol=1e-12, rtol=1e-10)
                error = float(np.max(np.abs(baseline['events']-captured['events'])))
            summary.append(dict(case=name, command=command, shape=list(captured['events'].shape),
                                eligible=int(captured['eligible'].sum()), max_absolute_error=error,
                                elapsed_seconds=elapsed, precision=captured['precision']))
        baseline = None
    (work/'validation.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    main()
