__version__ = '1.11.10'

from importlib import import_module


def _load_optional_extension(module_name):
    try:
        return import_module('.' + str(module_name), __name__)
    except Exception:  # pragma: no cover - compiled extensions are optional
        return None


combination_cy = _load_optional_extension('combination_cy')
omega_cy = _load_optional_extension('omega_cy')
parser_iqtree_cy = _load_optional_extension('parser_iqtree_cy')
recoding_cy = _load_optional_extension('recoding_cy')
substitution_cy = _load_optional_extension('substitution_cy')
substitution_sparse_cy = _load_optional_extension('substitution_sparse_cy')
