import pytest

import csubst
from csubst import _extensions


def test_optional_extension_compatibility_attributes_exist():
    for name in (
        'combination_cy',
        'omega_cy',
        'parser_iqtree_cy',
        'recoding_cy',
        'substitution_cy',
        'substitution_sparse_cy',
    ):
        assert hasattr(csubst, name)


def test_load_optional_extension_returns_none_only_when_target_is_missing(monkeypatch):
    qualified_name = 'csubst.missing_extension'

    def _missing(_name):
        raise ModuleNotFoundError("missing", name=qualified_name)

    monkeypatch.setattr(_extensions, 'import_module', _missing)
    assert _extensions.load_optional_extension('missing_extension') is None


def test_load_optional_extension_preserves_nested_import_errors(monkeypatch):
    def _broken(_name):
        raise ModuleNotFoundError("missing dependency", name='numpy')

    monkeypatch.setattr(_extensions, 'import_module', _broken)
    with pytest.raises(ModuleNotFoundError, match='missing dependency'):
        _extensions.load_optional_extension('broken_extension')
