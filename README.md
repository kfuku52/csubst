![](logo/logo_csubst_large.svg)

## Overview
**CSUBST** ([/si:sʌbst/](http://ipa-reader.xyz/?text=si:s%CA%8Cbst&voice=Salli)) is a tool for analyzing **C**ombinatorial **SUBST**itutions of codon sequences in phylogenetic trees.
A combinatorial substitution is defined as recurrent substitutions that occur at the same protein site in multiple independent branches.
If multiple substitutions result in the same amino acid, they are considered convergent amino acid substitutions.
The main features of **CSUBST** include:

- Error-corrected rate of protein convergence with null expectation obtained by:
    - Empirical or mechanistic codon substitution model
    - Urn sampling from site-wise substitution frequencies (**experimental**)
- Flexible specification of "foreground" lineages and its comparison with neighboring branches
- Heuristic detection of higher-order convergence involving more than two branches
- Simulated sequence evolution under specified scenarios of convergent evolution
- Convergent substitution mapping to protein structure

![](logo/method.png)

## Input files
**CSUBST** takes as inputs: 
- [Newick](https://en.wikipedia.org/wiki/Newick_format) file for the rooted tree
- [FASTA](https://en.wikipedia.org/wiki/FASTA_format) file for the multiple sequence alignment of in-frame coding sequences

## Installation and usage
**CSUBST** runs on python 3 (tested with >=3.6.0). See [CSUBST wiki](https://github.com/kfuku52/csubst/wiki) for detailed usage.
For a quick installation and test run, try:
```angular2html
# Installation with pip
pip install numpy cython # NumPy and Cython should be available upon csubst installation
pip install git+https://github.com/kfuku52/csubst

# Generate a test dataset
csubst dataset --name PGK

# Run csubst analyze
csubst analyze \
--alignment_file alignment.fa \
--rooted_tree_file tree.nwk \
--foreground foreground.txt
```

## Licensing
**CSUBST** is BSD-licensed (3 clause). See [LICENSE](LICENSE) for details.