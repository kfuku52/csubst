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

## Installation and test run
**CSUBST** runs on python 3 (tested with >=3.6.0). 
For a quick installation and test run, try:
```angular2html
# IQ-TREE installation with conda
conda install iqtree

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

## Basic usage
CSUBST is composed of several subcommands. 
`csubst -h` shows the list of subcommands, and the complete set of subcommand options are available from `csubst SUBCOMMAND -h` (e.g., `csubst analyze -h`). 
Many options are available, but those used by a typical user would be as follows. 
More advanced usage is available in [CSUBST wiki](https://github.com/kfuku52/csubst/wiki). 

- `csubst dataset` returns out-of-the-box test datasets.
  - `--name`: Name of dataset. For a small test dataset, try `PGK` (vertebrate phosphoglycerate kinase genes).
- `csubst analyze` is the main function of CSUBST. This subcommand returns various files including a table for ω<sub>C</sub>, dN<sub>C</sub>, and dS<sub>C</sub>.
  - `--alignment_file`: PATH to input in-frame codon alignment.
  - `--rooted_tree_file`: PATH to input rooted tree. Tip labels should be consistent with `--alignment_file`.
  - `--genetic_code`: NCBI codon table ID. 1 = "Standard". See [here](https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi) for details.
  - `--iqtree_model`: Codon substitution model for ancestral state reconstruction. Base models of "MG", "GY", "ECMK07", and "ECMrest" are supported. Among-site rate heterogeneity and codon frequencies can be specified. See [IQTREE's website](http://www.iqtree.org/doc/Substitution-Models) for details.
  - `--threads`: The number of CPUs for parallel computations (e.g., `1` or `4`).
  - `--foreground`: Optional. A text file to specify the foreground lineages. The file should contain two columns separated by a tab: 1st column for lineage IDs and 2nd for regex-compatible leaf names.
- `csubst site` maps combinatorial substitutions onto protein structure.
  - `--alignment_file`: PATH to input in-frame codon alignment.
  - `--rooted_tree_file`: PATH to input rooted tree. Tip labels should be consistent with `--alignment_file`.
  - `--genetic_code`: NCBI codon table ID. 1 = "Standard". See [here](https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi) for details.
  - `--iqtree_model`: Codon substitution model for ancestral state reconstruction. Base models of "MG", "GY", "ECMK07", and "ECMrest" are supported. Among-site rate heterogeneity and codon frequencies can be specified. See [IQTREE's website](http://www.iqtree.org/doc/Substitution-Models) for details.
- `csubst simulate` generates a simulated sequence alignment under a convergent evolutionary scenario.
  - `--alignment_file`: PATH to input in-frame codon alignment.
  - `--rooted_tree_file`: PATH to input rooted tree. Tip labels should be consistent with `--alignment_file`.
  - `--genetic_code`: NCBI codon table ID. 1 = "Standard". See [here](https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi) for details.
  - `--iqtree_model`: Codon substitution model for ancestral state reconstruction. Base models of "MG", "GY", "ECMK07", and "ECMrest" are supported. Among-site rate heterogeneity and codon frequencies can be specified. See [IQTREE's website](http://www.iqtree.org/doc/Substitution-Models) for details.
  - `--foreground`: A text file to specify the foreground lineages. The file should contain two columns separated by a tab: 1st column for lineage IDs and 2nd for regex-compatible leaf names.

## Citation
Fukushima K, Pollock DD. 2022. Detecting macroevolutionary genotype-phenotype associations using error-corrected rates of protein convergence. bioRxiv 487346 [DOI: 10.1101/2022.04.06.487346](https://doi.org/10.1101/2022.04.06.487346)

## Licensing
**CSUBST** is BSD-licensed (3 clause). See [LICENSE](LICENSE) for details.