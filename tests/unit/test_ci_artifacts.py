"""Guard against skipped platform coverage and source-shadowed wheel checks."""

import importlib.util
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import cli_runner


SCRIPT_DIR = Path(__file__).resolve().parents[2] / '.github' / 'scripts'


def load_script(name):
    spec = importlib.util.spec_from_file_location('ci_test_' + name, SCRIPT_DIR / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ci_scope = load_script('ci_scope')
installed_package = load_script('_installed_package')


@pytest.mark.parametrize('paths, expected', [
    (['csubst/runtime.py'], True),
    (['csubst/substitution_cy.pyx'], True),
    (['setup.py'], True),
    (['pyproject.toml'], True),
    (['.github/workflows/pytest.yml'], True),
    (['tests/unit/test_runtime_threads.py'], True),
    (['README.md', 'docs/ARCHITECTURE.md'], False),
    (['csubst/omega_statistics.py'], False),
])
def test_platform_coverage_tracks_build_and_runtime_changes(paths, expected):
    assert ci_scope.needs_macos(paths) is expected


@pytest.mark.parametrize('event_name, event', [
    ('schedule', {}),
    ('workflow_dispatch', {}),
    ('push', {'before': '0' * 40}),
    ('push', {'before': None}),
    ('pull_request', {}),
])
def test_unknown_or_periodic_diff_keeps_platform_coverage(event_name, event):
    assert ci_scope.changed_paths(event_name, event) is None


def test_missing_git_history_keeps_platform_coverage(monkeypatch):
    monkeypatch.setattr(ci_scope.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=128))
    assert ci_scope.changed_paths('push', {'before': 'a' * 40}) is None


def test_pull_request_compares_against_base_commit(monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout='csubst/runtime.py\nREADME.md\n')

    monkeypatch.setattr(ci_scope.subprocess, 'run', run)
    paths = ci_scope.changed_paths('pull_request', {'pull_request': {'base': {'sha': 'b' * 40}}})
    assert paths == ['csubst/runtime.py', 'README.md']
    assert commands == [['git', 'diff', '--name-only', 'b' * 40, 'HEAD']]


def test_wheel_guard_rejects_source_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'prefix', str(tmp_path / 'venv'))
    monkeypatch.setitem(sys.modules, 'csubst', SimpleNamespace(__file__=str(tmp_path / 'checkout/csubst/__init__.py')))
    with pytest.raises(RuntimeError, match='Expected csubst inside'):
        installed_package.require_installed_package()


@pytest.mark.parametrize('corruption', ['missing', 'python_fallback', 'external_binary'])
def test_wheel_guard_requires_every_installed_binary(tmp_path, monkeypatch, corruption):
    prefix = tmp_path / 'venv'
    package = prefix / 'lib/site-packages/csubst'
    monkeypatch.setattr(sys, 'prefix', str(prefix))
    monkeypatch.setitem(sys.modules, 'csubst', SimpleNamespace(__file__=str(package / '__init__.py')))
    for name in installed_package.EXTENSIONS:
        path = package / (name + EXTENSION_SUFFIXES[0])
        monkeypatch.setitem(sys.modules, 'csubst.' + name, SimpleNamespace(__file__=str(path)))
    target = 'csubst.substitution_cy'
    assert installed_package.require_installed_package() == package.parent
    if corruption == 'missing':
        monkeypatch.setitem(sys.modules, target, None)
    else:
        path = (package / 'substitution_cy.py' if corruption == 'python_fallback'
                else tmp_path / ('substitution_cy' + EXTENSION_SUFFIXES[0]))
        monkeypatch.setitem(sys.modules, target, SimpleNamespace(__file__=str(path)))
    with pytest.raises((RuntimeError, ModuleNotFoundError)):
        installed_package.require_installed_package()


def test_installed_cli_subprocess_cannot_inherit_source_override(monkeypatch, tmp_path):
    monkeypatch.setenv('PYTHONPATH', str(tmp_path / 'checkout'))
    monkeypatch.setenv('CSUBST_STRICT_EXTENSIONS', '1')
    env = installed_package.package_subprocess_env(tmp_path, installed=True)
    assert 'PYTHONPATH' not in env
    assert env['CSUBST_STRICT_EXTENSIONS'] == '1'
    monkeypatch.setenv('CSUBST_TEST_INSTALLED', '1')
    monkeypatch.setattr(cli_runner.subprocess, 'run', lambda *a, **k: k['env'])
    cli_env = cli_runner.run_csubst(['--version'], tmp_path)
    assert 'PYTHONPATH' not in cli_env
    assert cli_env['CSUBST_STRICT_EXTENSIONS'] == '1'
