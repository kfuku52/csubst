#!/usr/bin/env python3
"""Conservatively select the macOS installation/runtime check."""

import json
import os
from pathlib import Path
import re
import subprocess


MACOS_FILES = {
    'pyproject.toml', 'setup.py', 'MANIFEST.in', 'csubst/runtime.py',
    'csubst/cli.py', 'csubst/cli_io.py', 'csubst/_extensions.py',
    'csubst/sequence_io.py', 'tests/unit/test_runtime_threads.py',
}


def needs_macos(paths):
    return any(path in MACOS_FILES or path.endswith(('.pyx', '.c'))
               or path.startswith('.github/') for path in paths)


def changed_paths(event_name, event):
    if event_name == 'pull_request':
        base = event.get('pull_request', {}).get('base', {}).get('sha', '')
    elif event_name == 'push':
        base = event.get('before', '')
    else:
        return None  # Scheduled/manual runs always cover macOS.
    if not isinstance(base, str) or not re.fullmatch('[0-9a-f]{40,64}', base) or set(base) == {'0'}:
        return None
    result = subprocess.run(['git', 'diff', '--name-only', base, 'HEAD'],
                            capture_output=True, text=True)
    return result.stdout.splitlines() if result.returncode == 0 else None


def main():
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    paths = changed_paths(os.environ['GITHUB_EVENT_NAME'], event)
    selected = paths is None or needs_macos(paths)
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as handle:
        handle.write('macos=' + str(selected).lower() + '\n')


if __name__ == '__main__':
    main()
