from setuptools import setup, find_packages

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
    install_requires = ['ete3','numpy','pandas','joblib'],
    scripts          = ['csubst/csubst',],
)