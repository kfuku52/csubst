from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from numpy import get_include # for "cimport numpy"
import os
import platform

if platform.system() == 'Darwin':
    # https://stackoverflow.com/questions/39114132/cython-fatal-error-numpy-arrayobject-h-file-not-found-using-numpy
    os.environ["CPPFLAGS"] = os.getenv("CPPFLAGS", "") + "-I" + get_include()

setup(
    name             = 'csubst',
    version          = "0.1",
    description      = 'Tools for molecular convergence detection in DNA/codon/protein sequences',
    license          = "BSD 3-clause License",
    author           = "Kenji Fukushima",
    author_email     = 'kfuku52@gmail.com',
    url              = 'https://github.com/kfuku52/csubst.git',
    keywords         = '',
    packages         = find_packages(),
    install_requires = ['ete3','numpy','pandas','joblib','cython'],
    scripts          = ['csubst/csubst',],
    ext_modules      = cythonize(
                                'csubst/*.pyx',
                                include_path = [get_include(),],
                                compiler_directives={'language_level' : "3"}
    ),
    cmdclass         = {'build_ext': build_ext},
    include_dirs     = [get_include(),],
)
