"""Keep artifact checks on the installed package instead of the checkout."""

import importlib
from importlib.machinery import EXTENSION_SUFFIXES
import os
from pathlib import Path
import sys


EXTENSIONS = ('combination_cy', 'omega_cy', 'parser_iqtree_cy', 'recoding_cy',
              'substitution_cy', 'substitution_sparse_cy')


def require_installed_package():
    import csubst

    prefix = Path(sys.prefix).resolve()
    package = Path(csubst.__file__).resolve()
    if not package.is_relative_to(prefix):
        raise RuntimeError('Expected csubst inside {}, imported {}'.format(prefix, package))
    for name in EXTENSIONS:
        module = importlib.import_module('csubst.' + name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(prefix) or not any(str(path).endswith(s) for s in EXTENSION_SUFFIXES):
            raise RuntimeError('Expected installed native extension {}, imported {}'.format(name, path))
    return package.parent.parent


def configure_package_imports(repo_root, installed=False):
    repo_root = Path(repo_root).resolve()
    if installed:
        sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != repo_root]
        return require_installed_package()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def package_subprocess_env(repo_root, installed=False):
    env = os.environ.copy()
    if installed:
        env.pop('PYTHONPATH', None)
    else:
        env['PYTHONPATH'] = str(repo_root) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env
