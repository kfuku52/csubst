"""Run four output-toggle comparisons against the CSUBST on PYTHONPATH.

Run in the GeneGalleon runtime; --output must be a new or disposable directory.
The script records discrepancies; it does not modify CSUBST source.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=False)
inputs = Path(__file__).resolve().parent / 'inputs'
common = [sys.executable, '-m', 'csubst', 'search',
          '--alignment_file', str(inputs / 'input.fa'),
          '--rooted_tree_file', str(inputs / 'tree.nwk'),
          '--foreground', str(inputs / 'foreground.tsv'),
          '--iqtree_model', 'GY+FQ', '--threads', '1',
          '--exhaustive_until', '1', '--s', 'no', '--bs', 'no',
          '--cs', 'no', '--cbs', 'no', '--cb', 'yes']
summary = {}
for mode, b in [('selective', 'no'), ('full', 'yes')]:
    for kind, options in [
        ('training', ['--expectation_method', 'urn', '--asrv', 'sn',
                      '--asrv_training_branches', 'background', '--asrv_report', 'yes']),
        ('sites', ['--site_filter_report', 'yes']),
    ]:
        name = kind + '-' + mode
        target = output / name
        command = common + options + ['--b', b, '--outdir', str(target)]
        with (output / (name + '.log')).open('w') as log:
            subprocess.run(command, cwd=output, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        if kind == 'training':
            provenance = json.loads((target / 'csubst_urn_provenance.json').read_text())
            cb = pd.read_csv(target / 'csubst_cb_2.tsv', sep='\t')
            summary[name] = {
                'S_training_mass': provenance['weight_diagnostics'][0]['training_mass'],
                'cb': cb[['branch_id_1', 'branch_id_2', 'ECNany2any', 'ECSany2any',
                          'omegaCany2spe']].to_dict(orient='records'),
            }
        else:
            sites = pd.read_csv(target / 'csubst_site_filter_sites.tsv', sep='\t')
            summary[name] = {'tip_counts': sorted(sites.num_tips_with_codon_state.unique().tolist())}
text = json.dumps(summary, indent=2)
(output / 'comparison.json').write_text(text + '\n')
print(text)
