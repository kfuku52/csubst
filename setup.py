import re
import ast
import platform
import os

from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from numpy import get_include # for "cimport numpy"

if platform.system() == 'Darwin':
    # https://stackoverflow.com/questions/39114132/cython-fatal-error-numpy-arrayobject-h-file-not-found-using-numpy
    os.environ["CPPFLAGS"] = os.getenv("CPPFLAGS", "") + " -I" + get_include()

with open(os.path.join('csubst', '__init__.py')) as f:
    match = re.search(r'__version__\s+=\s+(.*)', f.read())
version = str(ast.literal_eval(match.group(1)))

setup(
    name             = 'csubst',
    version          = version,
    description      = 'Tools for molecular convergence detection in coding sequences',
    license          = "BSD 3-clause License",
    author           = "Kenji Fukushima",
    author_email     = 'kfuku52@gmail.com',
    url              = 'https://github.com/kfuku52/csubst.git',
    keywords         = 'molecular convergence',
    python_requires  = '>=3.1',
    packages         = find_packages(),
    install_requires = ['ete3','numpy','scipy','pandas','joblib','cython','matplotlib','threadpoolctl'],
    scripts          = ['csubst/csubst',],
    setup_requires   = ['setuptools>=18.0','cython'],
    ext_modules      = cythonize(
                                'csubst/*.pyx',
                                include_path = [get_include(),],
                                compiler_directives={'language_level' : "3"}
    ),
    cmdclass         = {'build_ext': build_ext},
    include_dirs     = [get_include(),],
    include_package_data = True,
    package_data     = {
                            '':['substitution_matrix/*.dat',
                                'dataset/*',
                                ],
    }
)
