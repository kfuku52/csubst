"""Shared Matplotlib text policy and lazy setup for site renderers."""

import importlib

font_size = 8
TREE_LINE_CAPSTYLE = 'round'
VESM_XTICK_LABEL_GAP_POINTS = 1.0
_matplotlib_module = None
_pyplot_module = None


def configure_text_rendering(module):
    """Keep point sizes while avoiding Helvetica's small-size native hinting error."""
    # Matplotlib 3.11 defaults to native hinting, which can divide by zero in
    # macOS Helvetica's prep program. Auto-hinting preserves our pre-3.11 policy.
    module.rcParams['text.hinting'] = 'auto'


def _load_matplotlib_modules():
    global _matplotlib_module, _pyplot_module
    if _matplotlib_module is None:
        module = importlib.import_module('matplotlib')
        pyplot = importlib.import_module('matplotlib.pyplot')
        configure_text_rendering(module)
        module.rcParams['font.size'] = font_size
        module.rcParams['font.family'] = 'sans-serif'
        module.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Nimbus Sans', 'DejaVu Sans']
        module.rcParams['svg.fonttype'] = 'none'
        module.rc('xtick', labelsize=font_size)
        module.rc('ytick', labelsize=font_size)
        module.rc('font', size=font_size)
        module.rc('axes', titlesize=font_size)
        module.rc('axes', labelsize=font_size)
        module.rc('legend', fontsize=font_size)
        module.rc('figure', titlesize=font_size)
        _matplotlib_module = module
        _pyplot_module = pyplot
    return _matplotlib_module, _pyplot_module


class _LazyMatplotlibProxy:
    def __getattr__(self, name):
        module, _ = _load_matplotlib_modules()
        return getattr(module, name)


class _LazyPyplotProxy:
    def __getattr__(self, name):
        _, pyplot = _load_matplotlib_modules()
        return getattr(pyplot, name)


matplotlib = _LazyMatplotlibProxy()
plt = _LazyPyplotProxy()
