![](logo/logo_csubst_large.svg)

## Overview
**csubst** is a tool for analyzing combinatorial substitutions of codon sequences in phylogenetic trees. Main features include:

- Accurate detection of the rate of molecular convergence with null expectation obtained by:
    - Among-site substitution randomization
    - Empirical codon substitution model
    - Mechanistic codon substitution model
- Analysis of higher-order convergence involving more than two branches
- Simulated sequence evolution under specified scenarios of convergent evolution
- Flexible specification of "foreground" lineages and its comparison with neighboring branches

## Input files
**csubst** takes as inputs: 
- a [Newick](https://en.wikipedia.org/wiki/Newick_format) file for the rooted tree
- a [FASTA](https://en.wikipedia.org/wiki/FASTA_format) file for the multiple sequence alignment of in-frame coding sequences

## Dependency
**csubst** runs on python 3 (tested with >3.5.0) and depends on the following python packages. **csubst** installation with `pip install` will automatically install them except for **pyvolve** and **matplotlib**.
* [IQ-TREE](http://www.iqtree.org/) (version 2.0.0 or later)
* [python 3](https://www.python.org/)
* [ete3](https://github.com/etetoolkit/ete)
* [numpy](https://github.com/numpy/numpy)
* [scipy](https://www.scipy.org/)
* [pandas](https://github.com/pandas-dev/pandas)
* [joblib](https://github.com/joblib/joblib)
* [cython](https://cython.org/)
* [pyvolve](https://github.com/sjspielman/pyvolve) (Optional: required for `csubst simulate`)
* [matplotlib](https://matplotlib.org/3.1.1/index.html) (Optional: required for `csubst site`)

## Installation
**csubst** can be installed by the following command. **IQ-TREE** should also be installed in the same environment; See [here](http://www.iqtree.org/doc/Quickstart#installation) for instruction. Try `conda install -c bioconda iqtree` if you are familiar with `conda`.
```

# Installation with pip
pip install numpy cython # NumPy and Cython should be available upon csubst installation
pip install git+https://github.com/kfuku52/csubst

# Installation if successful if this command shows help messages
csubst -h 
```

## Getting started
**csubst** contains out-of-the-box datasets. It may take a couple of minutes to complete the test run below, which will be successful if output files like `csubst_cb_2.tsv` are produced. See [csubst wiki](https://github.com/kfuku52/csubst/wiki) for detailed usage and interpretation of output files.

```
# Generate a test dataset
csubst dataset --name PGK

# Run csubst analyze
csubst analyze \
--alignment_file alignment.fa \
--rooted_tree_file tree.nwk \
--omega_method rec \
--foreground foreground.txt \
--force_exhaustive yes \
--max_arity 2 \
--threads 4
```

## Licensing
**csubst** is BSD-licensed (3 clause). See [LICENSE](LICENSE) for details.