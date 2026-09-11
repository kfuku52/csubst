"""Memory regression benchmark using the established fixed-fit CLI harness.

The two sources must use the same scientific options. This runner strengthens
the earlier harness by checking every column of the main result tables,
including branch/site summaries, and stops if numerical equivalence fails.
"""

import importlib.util
from pathlib import Path
import re

import pandas as pd


source = Path(__file__).resolve().parents[1] / 'resource_audit_20260911/benchmark.py'
spec = importlib.util.spec_from_file_location('resource_benchmark', source)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
compare_selected = harness.compare


def output_summary(outdir):
    frames = {}
    for path in sorted(outdir.rglob('*.tsv')):
        name = str(path.relative_to(outdir))
        if (re.fullmatch(r'csubst_(cb_\d+|b|s|bs|cs|cbs|scan|scan_units)\.tsv', name)
                or name.endswith('/csubst.tsv')):
            frames[name] = pd.read_csv(path, sep='\t')
    return frames


def compare_all(reference, candidate):
    assert reference.keys() == candidate.keys(), 'Output table names differ.'
    result = compare_selected(reference, candidate)
    for name, before in reference.items():
        after = candidate[name]
        pd.testing.assert_frame_equal(before, after, check_exact=False, atol=1e-9, rtol=1e-8)
        result[name]['all_columns_equivalent_rtol1e-8_atol1e-9'] = True
        result[name]['all_column_count'] = len(before.columns)
    return result


if __name__ == '__main__':
    harness.output_summary = output_summary
    harness.compare = compare_all
    harness.main()
