"""Count diagnostic array payloads in an actual CLI run; not a timing tool."""

import inspect
import json
import os
from pathlib import Path
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402
from csubst import cli  # noqa: E402

original = np.full
record = {}


def counted(*args, **kwargs):
    result = original(*args, **kwargs)
    value = kwargs.get('fill_value', args[1] if len(args) > 1 else None)
    if isinstance(value, str) and value == 'ok' and result.dtype == np.dtype('U24'):
        caller = inspect.currentframe().f_back
        name = Path(caller.f_code.co_filename).name + ':' + caller.f_code.co_name
        entry = record.setdefault(name, dict(calls=0, allocated_array_bytes=0))
        entry['calls'] += 1
        entry['allocated_array_bytes'] += result.nbytes
    return result


np.full = counted
try:
    code = cli.main(sys.argv[3:])
finally:
    np.full = original
Path(sys.argv[2]).write_text(json.dumps(record, indent=2) + '\n')
raise SystemExit(code)
