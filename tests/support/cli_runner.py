"""Run CLI subprocesses against either the checkout or the installed wheel."""

import os
from pathlib import Path
import subprocess
import sys


def run_csubst(args, cwd):
    env = os.environ.copy()
    if env.get('CSUBST_TEST_INSTALLED') != '1':
        root = Path(__file__).resolve().parents[2]
        env['PYTHONPATH'] = str(root) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    else:
        env.pop('PYTHONPATH', None)
    return subprocess.run(
        [sys.executable, '-m', 'csubst', *args], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=120,
    )
