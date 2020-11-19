![](logo/logo_csubst.svg)

## Overview
**csubst** is a tool for molecular convergence detection in DNA/codon/protein sequences.

## Input files
**csubst** takes as inputs a rooted tree, a multiple sequence alignment, and posterior probabilities or maximum-likelihood estimates of ancestral sequences that are generated by phylogenetic softwares. Currently, ancestral sequences from following programs are supported.
* [IQ-TREE](http://www.iqtree.org/): version 2.0.0 or later
* [PhyloBayes](http://www.atgc-montpellier.fr/phylobayes/)

## Dependencies
**csubst** runs on python 3.5+ and depends on the following python packages.
* [python 3](https://www.python.org/)
* [ete3](https://github.com/etetoolkit/ete)
* [numpy](https://github.com/numpy/numpy)
* [pandas](https://github.com/pandas-dev/pandas)
* [joblib](https://github.com/joblib/joblib)
* [cython](https://cython.org/)
* [pyvolve](https://github.com/sjspielman/pyvolve) (Optional: required for `csubst simulate`)

## Installation
```
# Installation with pip
pip install git+https://github.com/kfuku52/csubst

# Check the complete set of options
csubst -h 
```

## Test run
```
# Download test data
# PGK data are also available in csubst/data.
# If svn is not installed in your environment, 
# download zip for the entire repo from "Clone or download". 
svn export https://github.com/kfuku52/csubst/trunk/data/PGK

# Enter the directory
cd ./PGK

# Run IQ-TREE (version >=2.0.0) to get output files 
# (.state, .rate, and .treefile)
# It's included in the downloaded directory 
# so you don't have to run IQ-TREE in this test.
# tree.nwk should be rooted.

iqtree \
-s alignment.fa \
-te tree.nwk \
-m ECMK07+F3X4+R4 \
--seqtype CODON1 \
--threads-max 4 \
--ancestral \
--rate \
--redo

# Run csubst analyze
csubst analyze \
--ncbi_codon_table 1 \
--aln_file alignment.fa \
--tre_file tree.nwk \
--infile_type iqtree \
--max_arity 2 \
--nslots 4
```
## Simulation of molecular convergence
```
# Run csubst simulate
csubst simulate \
--ncbi_codon_table 1 \
--aln_file alignment.fa \
--tre_file tree.nwk \
--infile_type iqtree \
--foreground foreground.txt \
--fg_stem_only yes \
--num_simulated_site 5000 \
--convergence_intensity_factor 100 \
--background_omega 0.1 \
--foreground_omega 0.1 \
--tree_scaling_factor 5 \
--convergent_amino_acids random1 \
--num_partition 100

# Run IQ-TREE
iqtree \
-s simulate.fa \
-te tree.nwk \
-m ECMK07 \
--seqtype CODON1 \
--threads-max 4 \
--ancestral \
--rate \
--redo

# Run csubst analyze
csubst analyze \
--ncbi_codon_table 1 \
--aln_file simulate.fa \
--tre_file tree.nwk \
--infile_type iqtree \
--max_arity 2 \
--foreground foreground.txt \
--fg_stem_only yes \
--fg_force_exhaustive yes \
--asrv each \
--nslots 4

```


## Licensing
**csubst** is BSD-licensed (3 clause). See [LICENSE](LICENSE) for details.