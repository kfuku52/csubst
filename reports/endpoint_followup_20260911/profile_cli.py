"""Profile one CLI run; PROFILE_OUTPUT selects the JSON output path.

Use PYTHONPATH to select a frozen source snapshot. Imports are warmed before
profiling. Stage RSS values are entry/exit samples, not per-stage peak memory.
The cProfile run is diagnostic and is separate from performance measurements.
"""

import cProfile
import functools
import importlib
import json
import os
from pathlib import Path
import pstats
import runpy
import time

import numpy as np
import psutil


STAGES = {
    'parser_misc': ['prepare_input_context', 'prep_state', 'apply_site_filters'],
    'parser_iqtree': ['get_state_tensor'],
    'scan_ctmc': ['prepare', 'infer'],
    'substitution_scan': ['scan_substitutions', '_build_scan_static_context',
                          '_scan_substitutions_core', '_calibrate_scan_pvalues'],
    'endpoint_io': ['_build'],
}


def arrays(g):
    if not isinstance(g, dict):
        return {}
    result, identities = {}, {}
    for key, value in g.items():
        if not isinstance(value, np.ndarray) or value.nbytes <= 1024**2:
            continue
        result[key] = {'shape': value.shape, 'mib': value.nbytes / 2**20}
        if id(value) in identities:
            result[key]['shared_with'] = identities[id(value)]
        else:
            identities[id(value)] = key
    return result


def main():
    process = psutil.Process()
    records = []
    output = Path(os.environ['PROFILE_OUTPUT'])

    def wrap(fn, label):
        @functools.wraps(fn)
        def call(*args, **kwargs):
            start = time.perf_counter()
            before = process.memory_info().rss
            result = fn(*args, **kwargs)
            g = result[0] if isinstance(result, tuple) else result
            if not isinstance(g, dict):
                g = args[0] if args else kwargs.get('g')
            records.append(dict(stage=label, seconds=time.perf_counter() - start,
                                rss_before_mib=before / 2**20,
                                rss_after_mib=process.memory_info().rss / 2**20,
                                arrays=arrays(g)))
            return result
        return call

    for module, names in STAGES.items():
        mod = importlib.import_module('csubst.' + module)
        for name in names:
            setattr(mod, name, wrap(getattr(mod, name), module + '.' + name))
    prof = cProfile.Profile()
    prof.enable()
    try:
        runpy.run_module('csubst', run_name='__main__')
    finally:
        prof.disable()
        prof.dump_stats(str(output) + '.prof')
        output.write_text(json.dumps(records, indent=2) + '\n')
        with open(str(output) + '.txt', 'w') as handle:
            pstats.Stats(prof, stream=handle).strip_dirs().sort_stats('cumulative').print_stats(55)


if __name__ == '__main__':
    main()
