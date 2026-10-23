"""Audit whether substitution-pattern vectorization can affect supported CLI work.

Run from the repository root:
    python reports/performance_round3_20261023/proposal10.py

This deliberately runs no synthetic performance benchmark: the target helper is
not called by the audited production source. Only proposal10 report files change.
"""

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tokenize


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASELINE_COMMIT = '624bb8a6dae536d0fa10891f666684ef1ad3dab0'
SYMBOL = 'calc_substitution_patterns'


def audit_package(root):
    files = sorted((root / 'csubst').rglob('*.py'))
    definitions, active_references, literal_lookups, comments = [], [], [], []
    source_hashes = {}
    for path in files:
        name = path.relative_to(root).as_posix()
        source = path.read_text()
        tree = ast.parse(source, filename=name)
        for node in ast.walk(tree):
            entry = {'path': name, 'line': getattr(node, 'lineno', None)}
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == SYMBOL:
                definitions.append(entry)
            if ((isinstance(node, ast.Attribute) and node.attr == SYMBOL)
                    or (isinstance(node, ast.Name) and node.id == SYMBOL)):
                active_references.append(entry | {'kind': type(node).__name__})
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and SYMBOL in node.value:
                literal_lookups.append(entry)
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT and SYMBOL in token.string:
                comments.append({'path': name, 'line': token.start[0], 'text': token.string})
        if name in ('csubst/cli.py', 'csubst/main_analyze.py', 'csubst/combination.py'):
            source_hashes[name] = hashlib.sha256(source.encode()).hexdigest()
    assert len(definitions) == 1, definitions
    assert not active_references, active_references
    assert not literal_lookups, literal_lookups
    assert len(comments) == 1, comments
    return {'python_files_parsed': len(files), 'definitions': definitions,
            'active_identifier_references': active_references,
            'literal_name_lookups': literal_lookups,
            'commented_references': comments, 'source_sha256': source_hashes}


def audit_cli(root):
    path = root / 'csubst/cli.py'
    tree = ast.parse(path.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    search = functions['command_search']
    imports = [node for node in ast.walk(search) if isinstance(node, ast.ImportFrom)
               and node.module == 'csubst.main_analyze'
               and any(item.name == 'main_analyze' for item in node.names)]
    calls = [node for node in ast.walk(search) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'main_analyze']
    assert len(imports) == len(calls) == 1
    registration = functions['_register_search_parser']
    parser = [node for node in ast.walk(registration) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_parser'
              and node.args and isinstance(node.args[0], ast.Constant)
              and node.args[0].value == 'search']
    assert len(parser) == 1
    aliases = next(item.value for item in parser[0].keywords if item.arg == 'aliases')
    assert ast.literal_eval(aliases) == ['analyze']
    handlers = [node for node in ast.walk(registration) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == 'set_defaults'
                and any(item.arg == 'handler' and isinstance(item.value, ast.Name)
                        and item.value.id == 'command_search' for item in node.keywords)]
    assert len(handlers) == 1
    return {'commands': ['search', 'analyze'], 'handler': 'command_search',
            'orchestrator': 'csubst.main_analyze.main_analyze',
            'parser_line': parser[0].lineno, 'handler_line': handlers[0].lineno,
            'orchestrator_call_line': calls[0].lineno}


def tracked_symbol_matches(revision=None):
    command = ['git', 'grep', '-n', '-E',
               'calc_substitution_patterns|S_sub_pattern_id|N_sub_pattern_id']
    if revision:
        command.append(revision)
    command.extend(['--', '.', ':!reports'])
    output = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert output.returncode in (0, 1), output.stderr
    prefix = revision + ':' if revision else ''
    return [line.removeprefix(prefix) for line in output.stdout.splitlines()]


def main():
    baseline = os.environ.get('CSUBST_PERF_BASELINE')
    if not baseline:
        baseline = Path('/tmp/csubst-round3-baseline-path').read_text().strip()
    baseline = Path(baseline)
    baseline_package = audit_package(baseline)
    current_package = audit_package(ROOT)
    assert baseline_package['source_sha256'] == current_package['source_sha256']
    # Check the snapshot's relevant files against the named committed baseline.
    for name, digest in baseline_package['source_sha256'].items():
        committed = subprocess.check_output(['git', 'show', BASELINE_COMMIT + ':' + name], cwd=ROOT)
        assert hashlib.sha256(committed).hexdigest() == digest
    baseline_matches = tracked_symbol_matches(BASELINE_COMMIT)
    current_matches = tracked_symbol_matches()
    assert baseline_matches == current_matches
    assert len(current_matches) == 2, current_matches
    result = {
        'proposal': 10, 'baseline': BASELINE_COMMIT,
        'idea': 'Sort the substitution-count matrix once and factorize sorted rows in first-seen order.',
        'decision': 'rejected_inactive_supported_cli_path',
        'reason': 'No executable reference or literal-name lookup calls the helper in package source; its sole former orchestration call is commented out.',
        'environment': {'python': sys.version, 'platform': platform.platform()},
        'baseline_audit': baseline_package, 'working_tree_audit': current_package,
        'baseline_cli_dispatch': audit_cli(baseline), 'working_tree_cli_dispatch': audit_cli(ROOT),
        'tracked_repository_symbol_matches_excluding_reports': current_matches,
        'performance_measurements': None,
        'performance_claim': None,
        'coverage': {'static_package_identifier_and_literal_name_audit': True,
                     'tracked_repository_symbol_search': True,
                     'baseline_snapshot_hash_checked_against_commit': True,
                     'cli_dispatch_ast_checked': True,
                     'prototype_contract_parity': False,
                     'timed_benchmark': False,
                     'runtime_cli_route_exhaustion': False,
                     'external_python_library_callers_audited': False},
        'skipped': 'No prototype, workload timing, peak-memory measurement, or regression suite: production code is unchanged and an optimization cannot affect the audited CLI route.',
    }
    (HERE / 'proposal10.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (HERE / 'proposal10.patch').write_text('')
    print(json.dumps({'proposal': 10, 'decision': result['decision'],
                      'package_python_files': current_package['python_files_parsed'],
                      'active_references': 0, 'symbol_matches': current_matches}, indent=2))


if __name__ == '__main__':
    main()
