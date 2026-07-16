![](logo/logo_csubst_large.png)

[![Pytest](https://github.com/kfuku52/csubst/actions/workflows/pytest.yml/badge.svg)](https://github.com/kfuku52/csubst/actions/workflows/pytest.yml)
[![GitHub release](https://img.shields.io/github/v/tag/kfuku52/csubst?label=release)](https://github.com/kfuku52/csubst/releases)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/csubst.svg)](https://anaconda.org/bioconda/csubst)
[![Python](https://img.shields.io/badge/python-3.10--3.14-blue)](https://github.com/kfuku52/csubst)
[![Platforms](https://img.shields.io/conda/pn/bioconda/csubst.svg)](https://anaconda.org/bioconda/csubst)
[![Downloads](https://img.shields.io/conda/dn/bioconda/csubst.svg)](https://anaconda.org/bioconda/csubst)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview
**CSUBST** (/si:sʌbst/) is a tool for analyzing **C**ombinatorial **SUBST**itutions in codon sequences on phylogenetic trees.
A combinatorial substitution is a recurrent substitution at the same protein site on multiple independent branches.
When independent substitutions lead to the same amino acid, they are interpreted as convergent amino acid substitutions.
The main features of **CSUBST** are:

- Error-corrected rates of protein convergence, with null expectations based on:
    - Empirical or mechanistic codon substitution models
    - Urn sampling from site-wise substitution frequencies (**experimental**)
- Flexible specification of "foreground" lineages and comparisons with neighboring branches
- Heuristic detection of higher-order convergence involving more than two branches
- Sequence simulation under user-defined scenarios of convergent evolution
- Mapping convergent substitutions onto protein structures

![](logo/method.png)

## Input files
**CSUBST** requires the following input files:

- A [Newick](https://en.wikipedia.org/wiki/Newick_format) file containing the rooted tree
- A [FASTA](https://en.wikipedia.org/wiki/FASTA_format) file containing a multiple sequence alignment of in-frame coding sequences

## Installation
**CSUBST** supports Python 3.10–3.14. Installation via [Bioconda](https://anaconda.org/bioconda/csubst) is recommended because it installs IQ-TREE and the required Python dependencies automatically. `pip` installs the core Python dependencies automatically, but [IQ-TREE](https://iqtree.github.io/) and a C compiler must be available separately.

#### Option 1: Install with `conda`
```
conda install bioconda::csubst
```

#### Option 2: Install with `pip`
```
# Install IQ-TREE separately: https://iqtree.github.io/
pip install git+https://github.com/kfuku52/csubst
```

Protein-structure mapping additionally requires PyMOL and MAFFT. PyMOL can be
installed with the `structure` extra; install the MAFFT executable separately:

```bash
pip install "csubst[structure] @ git+https://github.com/kfuku52/csubst"
```

VESM and other protein-language-model features use the optional `vep` extra:

```bash
pip install "csubst[vep] @ git+https://github.com/kfuku52/csubst"
```

## Test run
```
# Generate a test dataset
csubst dataset --name PGK

# Run csubst search
csubst search --alignment_file alignment.fa.gz --rooted_tree_file tree.nwk --foreground foreground.txt
```

## Usage
CSUBST provides ten main subcommands:

- [`csubst dataset`](https://github.com/kfuku52/csubst/wiki/csubst-dataset): generate built-in example datasets such as `PGK` and `PEPC`.
- `csubst download`: download and verify shared model resources without requiring an input alignment.
- [`csubst doctor`](https://github.com/kfuku52/csubst/wiki/csubst-doctor): validate input files, inferred IQ-TREE paths, and optional 3Di settings before longer runs.
- [`csubst search`](https://github.com/kfuku52/csubst/wiki/csubst-search) (legacy alias: `csubst analyze`): run convergence analysis and report metrics such as `omegaC`, `dNC`, and `dSC`.
- [`csubst scan`](https://github.com/kfuku52/csubst/wiki/csubst-scan): list recurrent nonsynonymous-state substitutions shared by foreground clades without the omegaC branch-combination search. Foreground support units can be defined by input lineage IDs, automatically split stem branches, or their complete foreground clades. The output includes unit-level support, configurable foreground-vs-control rate-enrichment statistics, posterior-sum or called-event rate counts, site evolutionary rates, and amino-acid/state conservation.
- [`csubst inspect`](https://github.com/kfuku52/csubst/wiki/csubst-inspect): summarize branch mappings and inspect ancestral states.
- [`csubst sites`](https://github.com/kfuku52/csubst/wiki/csubst-sites) (legacy alias: `csubst site`): compute site-wise combinatorial substitutions for selected branch combinations, generate tree and site-summary plots, and optionally map sites to protein structures.
- [`csubst simulate`](https://github.com/kfuku52/csubst/wiki/csubst-simulate): simulate codon sequence evolution under user-defined convergence scenarios.
- [`csubst benchmark`](https://github.com/kfuku52/csubst/wiki/csubst-benchmark): run `csubst search` over parameter grids on the same input data and summarize runtime and output metrics.
- [`csubst benchmark-plot`](https://github.com/kfuku52/csubst/wiki/csubst-benchmark-plot): collect existing benchmark outputs, compare performance across parameter settings, and write an overview figure.

Display available commands and options:

```bash
csubst -h
csubst SUBCOMMAND -h
csubst SUBCOMMAND --help-advanced  # include expert tuning options
```

Shared model resources are downloaded by the resource manager when requested
and can be prefetched for an offline or batch-compute environment. The default cache is
`${CSUBST_CACHE_DIR:-~/.cache/csubst}`. For example:

```bash
csubst download --resource vesm-35m
csubst download --resource vesm-35m --no_download yes  # local availability check
```

VESM-35M consists of the pinned `ntranoslab/vesm` `VESM_35M.pth` checkpoint and
the pinned `facebook/esm2_t12_35M_UR50D` base model. Downloads are published
only after file-size and SHA-256 validation. Resource-specific interprocess
locks prevent duplicate downloads when multiple CSUBST processes start at the
same time. The same lock implementation protects ProstT5 model downloads and
shared ProstT5/3Di cache writes.

Structures retrieved for `--pdb PDB_CODE` or `--pdb besthit` are also stored
under the shared cache (`structures/`) and published atomically under the same
interprocess lock. RCSB, SWISS-MODEL, AlphaFold, and AlphaFill downloads no
longer create structure files in the current working directory.

Typical workflow:

```bash
# 1) Prepare a toy dataset
csubst dataset --name PGK

# 2) Validate inputs and inferred IQ-TREE paths
csubst doctor \
  --alignment_file alignment.fa.gz \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt

# 3) Run convergence analysis
csubst search \
  --alignment_file alignment.fa.gz \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt

# 4) Scan foreground recurrent substitutions directly
csubst scan \
  --alignment_file alignment.fa.gz \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --scan_unit_mode lineage \
  --scan_other_scope all \
  --scan_rate_event_mode posterior_sum \
  --scan_rate_exposure q_weighted \
  --scan_pvalue_calibration full_scan \
  --scan_n_permutations 1000 \
  --threads 8

# 5) Inspect site-wise convergence for a branch pair
csubst sites \
  --alignment_file alignment.fa.gz \
  --rooted_tree_file tree.nwk \
  --branch_id 23,51 \
  --outdir csubst_sites \
  --output_prefix csubst
```

All analysis subcommands accept `--outdir`, `--output_prefix`, and `--log_file`.
The `sites` command creates one branch-selection directory under `--outdir` for
each requested branch set.

### VESM-35M variant-effect scores in `csubst sites`

`csubst sites` can score posterior-supported amino-acid substitutions on the
selected branches with VESM-35M. Use the **full-length, codon-aligned CDS** as
`--alignment_file`; do not pass an alignment from which columns were removed by
trimAl or a similar program. CSUBST reconstructs the full-length ancestral
protein context before applying its own internal site filters.
When `--pdb` is also requested, `besthit` searching and PDB-to-alignment MAFFT
mapping use the tip AA alignment translated directly from this original codon
alignment. Internal removal of tip-invariant sites therefore does not shorten
the structure-search query or change its alignment coordinates.

The bundled PEPC dataset includes a suitable full-length alignment. Its ordinary
`alignment.fa.gz` is trimmed, so use `untrimmed_cds.fa.gz` for VESM:

```bash
csubst dataset --name PEPC

# Replace ANC,DES with numerical branch IDs reported by csubst inspect/search.
csubst sites \
  --alignment_file untrimmed_cds.fa.gz \
  --rooted_tree_file tree.nwk \
  --mode lineage \
  --branch_id ANC,DES \
  --vep_model vesm-35m \
  --vep_min_event_pp 0.8 \
  --tree_site_plot_format png
```

`--mode intersection` is also supported, including multiple branch IDs and
`--branch_id fg`. VESM support for `--mode set` and nonsynonymous recoding is
intentionally disabled in the initial implementation. The event threshold is
inclusive: an atomic parent-AA/child-AA event is scored when the product of the
two marginal state probabilities is at least `--vep_min_event_pp`.

The run writes a long `*.vesm.tsv` event table, VESM columns in the ordinary
wide sites table, and `*.vesm_tree_site.tsv` plus a tree/branch-by-site figure.
In the figure, marker size is substitution posterior probability and color is
the raw VESM log-likelihood ratio (LLR; lower values indicate a more deleterious
substitution). With `--pdb`, mapped residues are colored on the same continuous
red-white-blue scale in the PyMOL session. When multiple events
map to one residue, `--vep_site_aggregate` controls the structure score.

A GPU is not required. `--vep_device auto` prefers CUDA, then Apple MPS, and
falls back to CPU. The first scoring run downloads the pinned model unless it
was prefetched with `csubst download --resource vesm-35m`; subsequent runs reuse
both the model and the interprocess-locked mutation-score cache.

For advanced settings, including foreground formats, higher-order search, structure mapping, and simulation parameters, see the [CSUBST Wiki](https://github.com/kfuku52/csubst/wiki).

## Citation
Fukushima K, Pollock DD. 2023. Detecting macroevolutionary genotype-phenotype associations using error-corrected rates of protein convergence. Nature Ecology & Evolution 7: 155–170. [DOI: 10.1038/s41559-022-01932-7](https://doi.org/10.1038/s41559-022-01932-7)

## Licensing
**CSUBST** is MIT-licensed. See [LICENSE](LICENSE) for details.
