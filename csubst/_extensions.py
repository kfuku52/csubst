from importlib import import_module


def load_optional_extension(module_name):
    qualified_name = '{}.{}'.format(__package__, str(module_name))
    try:
        return import_module(qualified_name)
    except ModuleNotFoundError as exc:
        if exc.name == qualified_name:
            return None
        raise
