import warnings

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
    monkeypatch.delenv('CSUBST_DISABLE_EXTENSIONS', raising=False)
    qualified_name = 'csubst.missing_extension'

    def _missing(_name):
        raise ModuleNotFoundError("missing", name=qualified_name)

    monkeypatch.setattr(_extensions, 'import_module', _missing)
    assert _extensions.load_optional_extension('missing_extension') is None


def test_load_optional_extension_preserves_nested_import_errors(monkeypatch):
    monkeypatch.delenv('CSUBST_DISABLE_EXTENSIONS', raising=False)
    def _broken(_name):
        raise ModuleNotFoundError("missing dependency", name='numpy')

    monkeypatch.setattr(_extensions, 'import_module', _broken)
    with pytest.raises(ModuleNotFoundError, match='missing dependency'):
        _extensions.load_optional_extension('broken_extension')


def test_load_optional_extension_can_be_disabled(monkeypatch):
    monkeypatch.setenv('CSUBST_DISABLE_EXTENSIONS', '1')
    monkeypatch.setattr(
        _extensions,
        'import_module',
        lambda _name: pytest.fail('disabled extensions should not be imported'),
    )
    assert _extensions.load_optional_extension('combination_cy') is None


def test_extension_environment_flags_accept_false_and_reject_typos(monkeypatch):
    monkeypatch.setenv('CSUBST_DISABLE_EXTENSIONS', 'false')
    monkeypatch.setattr(_extensions, 'import_module', lambda _name: 'loaded')
    assert _extensions.load_optional_extension('combination_cy') == 'loaded'

    monkeypatch.setenv('CSUBST_DISABLE_EXTENSIONS', 'sometimes')
    with pytest.raises(ValueError, match='boolean value'):
        _extensions.load_optional_extension('combination_cy')


def test_extension_fallback_strict_mode_reraises(monkeypatch):
    monkeypatch.setenv('CSUBST_STRICT_EXTENSIONS', 'true')
    error = RuntimeError('broken accelerator')
    with pytest.raises(RuntimeError, match='broken accelerator'):
        _extensions.warn_extension_fallback('fast-path', error, set())


def test_extension_fallback_warns_only_once(monkeypatch):
    monkeypatch.delenv('CSUBST_STRICT_EXTENSIONS', raising=False)
    warned = set()
    with pytest.warns(RuntimeWarning, match='fast-path'):
        _extensions.warn_extension_fallback('fast-path', RuntimeError('broken'), warned)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        _extensions.warn_extension_fallback('fast-path', RuntimeError('broken'), warned)
    assert len(records) == 0
