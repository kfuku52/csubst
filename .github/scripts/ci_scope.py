#!/usr/bin/env python3
"""Conservatively select numerical and macOS validation coverage."""

import ast
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


def version_only_patch(before, after):
    """A docs-only patch bump must not conceal import code or a new release."""
    def parse(text):
        module = ast.parse(text)
        versions = [node for node in module.body if isinstance(node, ast.Assign)
                    and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == '__version__']
        if len(versions) != 1:
            raise ValueError('Expected one version assignment')
        value = ast.literal_eval(versions[0].value)
        if not isinstance(value, str) or not re.fullmatch(r'\d+\.\d+\.\d+', value):
            raise ValueError('Expected a semantic version')
        module.body.remove(versions[0])
        return tuple(map(int, value.split('.'))), ast.dump(module)
    try:
        old, old_code = parse(before)
        new, new_code = parse(after)
        return old_code == new_code and old[:2] == new[:2] and new[2] > old[2]
    except (ValueError, SyntaxError, TypeError):
        return False


def needs_numerical(paths, patch_only=False):
    if not paths:
        return True
    for path in paths:
        if path == 'csubst/__init__.py' and patch_only:
            continue
        if '/' not in path and path.endswith('.md'):
            continue
        if path.startswith(('docs/', 'reports/')) and path.endswith(('.md', '.json', '.tsv', '.txt', '.png', '.svg', '.pdf')):
            continue
        return True
    return False


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
    patch_only = False
    if paths is not None and 'csubst/__init__.py' in paths:
        base = (event.get('before') if os.environ['GITHUB_EVENT_NAME'] == 'push'
                else event.get('pull_request', {}).get('base', {}).get('sha'))
        result = subprocess.run(['git', 'show', str(base) + ':csubst/__init__.py'], capture_output=True, text=True)
        if result.returncode == 0:
            patch_only = version_only_patch(result.stdout, Path('csubst/__init__.py').read_text())
    numerical = paths is None or needs_numerical(paths, patch_only)
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as handle:
        handle.write('macos=' + str(selected).lower() + '\n')
        handle.write('numerical=' + str(numerical).lower() + '\n')


if __name__ == '__main__':
    main()
