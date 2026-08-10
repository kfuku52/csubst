import os
import platform
from numpy import get_include
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

try:
    from Cython.Build import cythonize
except ImportError:  # pragma: no cover - exercised in isolated builds without Cython
    cythonize = None


NUMPY_INCLUDE = get_include()
ROOT = os.path.dirname(os.path.abspath(__file__))
CYTHON_MODULES = [
    'combination_cy',
    'omega_cy',
    'parser_iqtree_cy',
    'recoding_cy',
    'substitution_cy',
    'substitution_sparse_cy',
]


class CSubstBuildExt(build_ext):
    def build_extensions(self):
        if platform.system() == 'Darwin':
            linker_so = list(getattr(self.compiler, 'linker_so', []))
            seen_rpaths = set()
            deduped = []
            for flag in linker_so:
                if str(flag).startswith('-Wl,-rpath,'):
                    if flag in seen_rpaths:
                        continue
                    seen_rpaths.add(flag)
                deduped.append(flag)
            self.compiler.linker_so = deduped
        super().build_extensions()


def normalize_extension_sources(extensions):
    for extension in extensions:
        extension.sources = [
            os.path.relpath(source, ROOT) if os.path.isabs(source) else source
            for source in extension.sources
        ]
    return extensions


def build_extensions():
    if os.environ.get('CSUBST_SKIP_EXTENSIONS', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
        return []
    use_cython_token = os.environ.get('CSUBST_USE_CYTHON', 'auto').lower()
    pyx_sources_available = all(
        os.path.exists(os.path.join(ROOT, 'csubst', module_name + '.pyx'))
        for module_name in CYTHON_MODULES
    )
    if use_cython_token == 'auto':
        use_cython = (cythonize is not None) and pyx_sources_available
    else:
        use_cython = use_cython_token in {'1', 'true', 'yes'}
    if use_cython and cythonize is None:
        raise RuntimeError('CSUBST_USE_CYTHON is set, but Cython is not installed.')
    if use_cython and (not pyx_sources_available):
        raise RuntimeError('CSUBST_USE_CYTHON is set, but one or more Cython .pyx sources are missing.')
    source_suffix = '.pyx' if use_cython else '.c'
    extensions = [
        Extension(
            f'csubst.{module_name}',
            [f'csubst/{module_name}{source_suffix}'],
            include_dirs=[NUMPY_INCLUDE],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        )
        for module_name in CYTHON_MODULES
    ]
    if use_cython:
        return normalize_extension_sources(
            cythonize(
                extensions,
                compiler_directives={'language_level': '3'},
            )
        )
    return extensions

setup(
    ext_modules      = build_extensions(),
    cmdclass         = {'build_ext': CSubstBuildExt},
    include_dirs     = [NUMPY_INCLUDE],
)
