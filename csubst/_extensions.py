from importlib import import_module
import os
import warnings


_TRUE_VALUES = frozenset({'1', 'true', 'yes', 'on'})
_FALSE_VALUES = frozenset({'0', 'false', 'no', 'off'})


def _env_flag(name):
    value = os.environ.get(name, '').strip().lower()
    if value == '':
        return False
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(
        '{} should be a boolean value such as 1/0, true/false, or yes/no.'.format(name)
    )


def load_optional_extension(module_name):
    if _env_flag('CSUBST_DISABLE_EXTENSIONS'):
        return None
    qualified_name = '{}.{}'.format(__package__, str(module_name))
    try:
        return import_module(qualified_name)
    except ModuleNotFoundError as exc:
        if exc.name == qualified_name:
            return None
        raise


def warn_extension_fallback(
    fastpath_name,
    exc,
    warned,
    fallback_name='Python',
    warning_key=None,
    accelerator_name='Cython fast path',
):
    """Warn once about a fast-path failure, or re-raise in strict mode."""

    if _env_flag('CSUBST_STRICT_EXTENSIONS'):
        raise exc
    if warning_key is None:
        warning_key = fastpath_name
    if warning_key in warned:
        return
    warned.add(warning_key)
    warnings.warn(
        '{} "{}" failed ({}: {}). Falling back to {} implementation.'.format(
            accelerator_name, fastpath_name, type(exc).__name__, exc, fallback_name
        ),
        RuntimeWarning,
        stacklevel=2,
    )
