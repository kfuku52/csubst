import argparse
import json
import runpy
from pathlib import Path
import pandas as pd
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--workdir', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
work = args.workdir
compare = runpy.run_path(str(root / '.github/scripts/benchmark_endpoint_optimization.py'))['compare_frames']
checks = []
for scenario in ['heuristic6_b', 'heuristic6']:
    ref = work / f'{scenario}-t1-b1-baseline_joint-r0'
    for cpu in [1, 4]:
        for repeat in range(4):
            for mode in ['baseline_joint', 'joint']:
                out = work / f'{scenario}-t{cpu}-b1-{mode}-r{repeat}'
                if not Path(str(out)+'.json').exists():
                    continue
                row = {'run': out.name, 'tables': {}}
                names = [f'csubst_cb_{k}.tsv' for k in range(2,7)]
                if scenario.endswith('_b'):
                    names.append('csubst_b.tsv')
                for name in names:
                    a = pd.read_pickle(ref / (name + '.unrounded.pkl'))
                    b = pd.read_pickle(out / (name + '.unrounded.pkl'))
                    row['tables'][name] = compare(a,b)
                    if name.startswith('csubst_cb'):
                        ids = [c for c in a.columns if c.startswith('branch_id_')]
                        np.testing.assert_array_equal(a[ids].to_numpy(), b[ids].to_numpy())
                        mask_a = (a.OCNany2spe >= 2) & (a.omegaCany2spe >= 5)
                        mask_b = (b.OCNany2spe >= 2) & (b.omegaCany2spe >= 5)
                        np.testing.assert_array_equal(mask_a,mask_b)
                        row['tables'][name]['passed_cutoffs'] = int(mask_a.sum())
                checks.append(row)
assert len(checks) == 32, f'Expected 32 complete runs, found {len(checks)}'
result = {'verified_runs': len(checks), 'checks': checks}
(root / 'reports/endpoint_event_optimization_20260910/validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('Verified',len(checks),'runs, all unrounded CB counts/ratios/IDs/cutoffs and branch tables.')
