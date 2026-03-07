import ast
import os
import platform
import re
from numpy import get_include
from setuptools import Extension, find_packages, setup

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


def normalize_extension_sources(extensions):
    for extension in extensions:
        extension.sources = [
            os.path.relpath(source, ROOT) if os.path.isabs(source) else source
            for source in extension.sources
        ]
    return extensions


def build_extensions():
    use_cython = os.environ.get('CSUBST_USE_CYTHON', 'auto').lower()
    if use_cython == 'auto':
        use_cython = cythonize is not None
    else:
        use_cython = use_cython in {'1', 'true', 'yes'}
    if use_cython and cythonize is None:
        raise RuntimeError('CSUBST_USE_CYTHON is set, but Cython is not installed.')
    source_suffix = '.pyx' if use_cython else '.c'
    extensions = [
        Extension(
            f'csubst.{module_name}',
            [f'csubst/{module_name}{source_suffix}'],
            include_dirs=[NUMPY_INCLUDE],
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

if platform.system() == 'Darwin':
    # https://stackoverflow.com/questions/39114132/cython-fatal-error-numpy-arrayobject-h-file-not-found-using-numpy
    os.environ["CPPFLAGS"] = os.getenv("CPPFLAGS", "") + " -I" + NUMPY_INCLUDE

with open(os.path.join('csubst', '__init__.py')) as f:
    match = re.search(r'__version__\s+=\s+(.*)', f.read())
version = str(ast.literal_eval(match.group(1)))

setup(
    name             = 'csubst',
    version          = version,
    description      = 'Tools for molecular convergence detection in coding sequences',
    license          = "MIT License",
    author           = "Kenji Fukushima",
    author_email     = 'kfuku52@gmail.com',
    url              = 'https://github.com/kfuku52/csubst.git',
    keywords         = 'molecular convergence',
    python_requires  = '>=3.10',
    packages         = find_packages(),
    install_requires = ['ete4>=4.3.0','numpy','scipy','pandas','matplotlib'],
    extras_require   = {
                            'simulate': [],
    },
    scripts          = ['csubst/csubst',],
    ext_modules      = build_extensions(),
    include_dirs     = [NUMPY_INCLUDE],
    include_package_data = True,
    package_data     = {
                            '':['substitution_matrix/*.dat',
                                'dataset/*',
                                ],
                            'csubst._vendor.pyvolve':['LICENSE.txt'],
    }
)
