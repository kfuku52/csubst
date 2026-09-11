"""Verify MG against IQ-TREE and exercise joint defaults using real fits.

Run with --workdir outside the repository. Requires iqtree and the CSUBST test
runtime. This is a correctness check, not a performance benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from csubst import genetic_code, parser_iqtree, parser_misc, scan_bootstrap, sequence  # noqa: E402


def run(command, logfile):
    with logfile.open('w') as handle:
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    seqs = list(sequence.read_fasta(ROOT / 'csubst/dataset/PGK.alignment.fa').values())[:4]
    sequences = {'Tip' + str(i): s[:900] for i, s in enumerate(seqs)}
    alignment = work / 'tips.fa'
    alignment.write_text(''.join('>' + n + '\n' + s + '\n' for n, s in sequences.items()))
    topology = work / 'tree.nwk'
    topology.write_text('((Tip0:0.1,Tip1:0.1):0.1,(Tip2:0.1,Tip3:0.1):0.1);\n')
    foreground = work / 'foreground.tsv'
    foreground.write_text('1\tTip0\n2\tTip2\n')
    codon_map = {c: aa for aa, c in genetic_code.get_codon_table(1) if aa != '*'}
    codons = np.array(sorted(codon_map))
    amino = sorted(set(codon_map.values()))
    synonyms = {aa: [i for i, c in enumerate(codons) if codon_map[c] == aa] for aa in amino}
    result = {'mg_likelihood_checks': [], 'cli_checks': []}
    for model in ['MG+F1X4', 'MG+F3X4', 'MGK+F3X4']:
        prefix = work / model.replace('+', '_')
        run(['iqtree', '-s', str(alignment), '-te', str(topology), '-st', 'CODON', '-m', model,
             '-asr', '-wsr', '-nt', '1', '-seed', '8', '-v', '-redo', '-pre', str(prefix)], prefix.with_suffix('.console'))
        g = dict(codon_orders=codons, alignment_file=str(alignment), float_type=np.float64,
                 path_iqtree_iqtree=str(prefix) + '.iqtree', path_iqtree_log=str(prefix) + '.log',
                 amino_acid_orders=amino, synonymous_indices=synonyms)
        parser_iqtree.read_iqtree(g)
        parser_iqtree.read_log(g)
        q = parser_misc.get_mechanistic_instantaneous_rate_matrix(g)
        fit = dict(q=q, pi=g['equilibrium_frequency'], codons=codons, sites=300,
                   topology=Path(str(prefix) + '.treefile').read_text())
        actual = scan_bootstrap.alignment_loglikelihood(fit, sequences)
        reported = float(re.search(r'Log-likelihood of the tree:\s*([-+\d.eE]+)', Path(str(prefix) + '.iqtree').read_text())[1])
        assert abs(actual - reported) < .001, (model, actual, reported)
        result['mg_likelihood_checks'].append(dict(model=model, reported=reported, reproduced=actual,
                                                   absolute_error=abs(actual-reported)))
        # Both observed estimators use the corrected MG matrix.
        for mode in ['joint', 'marginal']:
            out = work / (prefix.name + '_' + mode)
            command = [sys.executable, '-m', 'csubst', 'search', '--alignment_file', str(alignment),
                       '--rooted_tree_file', str(topology), '--outdir', str(out), '--threads', '1',
                       '--substitution_posterior', mode, '--calibrate_longtail', 'no', '--max_arity', '2']
            for suffix in ['state', 'treefile', 'rate', 'iqtree', 'log']:
                command += ['--iqtree_' + suffix, str(prefix) + '.' + suffix]
            run(command, work / (out.name + '.log'))
            result['cli_checks'].append(dict(command='search', model=model, posterior=mode,
                                             rows=len(pd.read_csv(out / 'csubst_cb_2.tsv', sep='\t'))))
    scan_common = [sys.executable, '-m', 'csubst', 'scan', '--alignment_file', str(alignment),
                   '--rooted_tree_file', str(topology), '--foreground', str(foreground),
                   '--iqtree_outdir', str(work / 'default_fit'), '--scan_site_plot', 'no',
                   '--scan_pvalue_calibration', 'none', '--scan_min_event_pp', '.05', '--scan_min_support', '1']
    frames = {}
    for name, flags in [('default', []), ('explicit_joint', ['--substitution_posterior', 'joint']),
                        ('legacy', ['--substitution_posterior', 'marginal']),
                        ('workers', ['--threads', '2']), ('fixed_bootstrap', ['--scan_pvalue_calibration', 'parametric', '--scan_n_permutations', '3'])]:
        out = work / name
        run(scan_common + ['--outdir', str(out), '--threads', '1'] + flags, work / (name + '.log'))
        frame = pd.read_csv(out / 'csubst_scan.tsv', sep='\t')
        assert len(frame), name
        result['cli_checks'].append(dict(command='scan', case=name, model='ECMK07+F+R4', rows=len(frame),
                                         observation=sorted(frame.scan_observation_method.unique())))
        frames[name] = frame
    pd.testing.assert_frame_equal(frames['default'], frames['explicit_joint'])
    pd.testing.assert_frame_equal(frames['default'], frames['workers'])
    result['default_explicit_and_workers_identical'] = True
    prefix = work / 'MG_F3X4'
    cb = pd.read_csv(work / 'MG_F3X4_joint/csubst_cb_2.tsv', sep='\t')
    pair = ','.join(str(int(cb.iloc[0][c])) for c in ['branch_id_1', 'branch_id_2'])
    for command_name, flags in [
            ('scan', ['--foreground', str(foreground), '--scan_min_support', '1', '--scan_min_event_pp', '.05',
                      '--scan_site_plot', 'no', '--scan_pvalue_calibration', 'none']),
            ('sites', ['--branch_id', pair, '--tree_site_plot', 'no', '--site_state_plot', 'no', '--site_summary_plot', 'no']),
            ('inspect', []),
            ('benchmark', ['--max_arity', '2', '--calibrate_longtail', 'no']),
            ('simulate', ['--num_simulated_site', '10', '--percent_convergent_site', '0'])]:
        out = work / ('mg_' + command_name)
        command = [sys.executable, '-m', 'csubst', command_name, '--alignment_file', str(alignment),
                   '--rooted_tree_file', str(topology), '--outdir', str(out), '--threads', '1',
                   '--iqtree_model', 'MG+F3X4', '--iqtree_outdir', str(work / 'mg_fit')] + flags
        for suffix in ['state', 'treefile', 'rate', 'iqtree', 'log']:
            command += ['--iqtree_' + suffix, str(prefix) + '.' + suffix]
        run(command, work / (out.name + '.log'))
        result['cli_checks'].append(dict(command=command_name, model='MG+F3X4', case='default_options'))
    (work / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
