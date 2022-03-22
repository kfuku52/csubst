![](logo/logo_csubst_large.svg)

## Overview
**CSUBST** ([/si:sʌbst/](http://ipa-reader.xyz/?text=si:s%CA%8Cbst&voice=Salli)) is a tool for analyzing **C**ombinatorial **SUBST**itutions of codon sequences in phylogenetic trees. 
Main features include:

- Accurate detection of the rate of molecular convergence with null expectation obtained by:
    - Empirical or mechanistic codon substitution model
    - Urn sampling from site-wise substitution frequencies (**experimental**)
- Flexible specification of "foreground" lineages and its comparison with neighboring branches
- Analysis of higher-order convergence involving more than two branches
- Simulated sequence evolution under specified scenarios of convergent evolution
- Mapping 

## Combinatorial substitutions
Here, we mean combinatorial substitutions by substitutions occurring at the same protein site in multiple independent branches.
If the substitutions result in the same amino acid, they are considered convergent amino acid substitutions.

## Input files
**CSUBST** takes as inputs: 
- [Newick](https://en.wikipedia.org/wiki/Newick_format) file for the rooted tree
- [FASTA](https://en.wikipedia.org/wiki/FASTA_format) file for the multiple sequence alignment of in-frame coding sequences

## Dependency
**CSUBST** runs on python 3 (tested with >=3.6.0). **CSUBST** installation with `pip install` will automatically install required packages except for the followings.
* [IQ-TREE](http://www.iqtree.org/) (version 2.0.0 or later)
* [pyvolve](https://github.com/sjspielman/pyvolve) (Optional: required for `csubst simulate`)
* [matplotlib](https://matplotlib.org/3.1.1/index.html) (Optional: required for `csubst site`)
* [PyMOL](https://pymol.org/2/) (Optional: required for `csubst site --pdb`; Open-source version can be installed by [conda](https://anaconda.org/conda-forge/pymol-open-source) or [brew](https://github.com/brewsci/homebrew-bio))
* [pypdb](https://github.com/williamgilpin/pypdb) (Optional: required for `csubst site -pdb besthit`)
* [biopython](https://biopython.org/) (Optional: required for `csubst site -pdb besthit`)

## Installation
**CSUBST** can be installed by the following command. **IQ-TREE** should also be installed in the same environment; See [here](http://www.iqtree.org/doc/Quickstart#installation) for instruction. Try `conda install -c bioconda iqtree` if you are familiar with `conda`.
```

# Installation with pip
pip install numpy cython # NumPy and Cython should be available upon csubst installation
pip install git+https://github.com/kfuku52/csubst

# Installation if successful if this command shows help messages
csubst -h 
```

## Getting started
**CSUBST** contains out-of-the-box datasets. It may take a couple of minutes to complete the test run below, which will be successful if output files like `csubst_cb_2.tsv` are produced. See [CSUBST wiki](https://github.com/kfuku52/csubst/wiki) for detailed usage and interpretation of output files.

```
# Generate a test dataset
csubst dataset --name PGK

# Run csubst analyze
csubst analyze \
--alignment_file alignment.fa \
--rooted_tree_file tree.nwk \
--foreground foreground.txt \
--max_arity 2 \
--threads 4
```

## Licensing
**CSUBST** is BSD-licensed (3 clause). See [LICENSE](LICENSE) for details.