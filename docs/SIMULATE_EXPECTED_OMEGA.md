# Q-based expected omegaC for simulated branch pairs

Named [simulation scenarios](SIMULATE_SCENARIOS.md) enable the same table.
For distance-dependent preference, the calculation also integrates the
joint distribution of the two preferred amino acids. Terminal focal
edges use their preference-specific endpoint kernels for simulated
counts and the background kernel for null counts. The paths to their
parents use the background Q in both calculations.

`csubst simulate --export_expected_omegaC yes` adds
`<output_prefix>_expected_omega.tsv` to the simulation output directory.
Export is enabled by default for named scenarios and disabled for `custom`.
The table covers `any2spe` for every pair of branches
without an ancestor–descendant relationship, including sisters. It uses
the simulated tree's branch IDs and node names. The foreground flags describe
the foreground annotation on that tree. Use `--exclude_sister_pair no` in
an exhaustive pairwise search if you want to compare the same pair set.

At each site the calculation uses the root frequencies and rate-scaled custom
matrices passed to Pyvolve, and the branch lengths and model assignments it
actually used. At sites selected by `--percent_foreground_sites`, branches
marked as foreground receive the foreground matrix and length multiplier. Invariant sites draw a root
codon from the background equilibrium distribution and copy it unchanged
to every node. `--simulate_asrv file` uses IQ-TREE's posterior-mean `Rate`
column when present, with `C_Rate` as the alternative.
The simulation log reports the share of nonsynonymous Q rates directed
toward the target amino acids. To remove directional bias, use
`--target_amino_acids random0`. To run all sites under background Q,
use `--percent_foreground_sites 0`.

A numeric foreground or background omega of zero retains synonymous rates.
The reducible background generator draws root codons from the specified codon
frequencies. A positive foreground omega can still use the original
nonsynonymous exchangeabilities when the background omega is zero.
For file-based site rates a reversible background generator can reuse its
eigensystem across scaled transition matrices.

An event is a change between the two branch endpoints. N
matches two amino-acid changes to the same derived amino acid. S matches
two synonymous codon changes to the same derived codon.

For each pair, the last common ancestor of the two branch subtrees supplies
the shared state. Conditional on that state, the probability of a matching
event is the sum of products of the two branches' probabilities for each
possible destination. Summing over the ancestor's state accounts for
dependence through shared ancestry. Each site's null keeps the same
distribution of ancestral states, site rate and branch lengths. It uses the
background Q for the two focal transitions. The table sums each of the four
probabilities across sites:

| Columns | Meaning |
| --- | --- |
| `dist_bl`, `dist_parent_bl` | Patristic distance between the branch parents on the simulation tree, after global tree scaling and before site-specific foreground scaling. Search uses the same endpoints on its fitted tree. |
| `muNany2spe_sim`, `muSany2spe_sim` | Expected N and S counts under simulation |
| `muNany2spe_null`, `muSany2spe_null` | Expected N and S counts under the background kernel on the focal branches |
| `omegaCany2spe_expected` | `(muN_sim / muN_null) / (muS_sim / muS_null)` |
| `OCNany2spe_true`, `OCSany2spe_true` | Realized counts from simulated endpoint codons |
| `omegaCany2spe_true` | Realized N and S counts substituted into the ratio with the same null denominators |

For comparison, search reports `count_ratio_omegaCany2spe` from the same
four-count ratio using its own observed and expected counts. It retains tiny
positive values before taking the ratio. `omegaCany2spe` continues to use
the configured count tolerance, pseudocounts and calibration.
`ocs_status_any2spe` distinguishes missing, zero, positive below tolerance,
positive below one after the tolerance check, and at least one observed
synonymous event. These labels describe posterior count mass.

A zero null expectation or a 0/0 rate ratio produces `NA`. A positive N rate
over a zero S rate produces `inf`. With `--export_true_asr no`, the three realized
columns are `NA`. The expected ratio describes a ratio of mean counts. For short
alignments the ratio of realized counts can fluctuate widely or be undefined.

The true-ASR `.treefile` stores the baseline branch lengths after
`--tree_branch_length_scale`. The Q-based table also accounts for the
`--foreground_branch_length_scale` applied at sites assigned the foreground process. A single Newick
tree cannot encode both sets of lengths for a mixture of site classes.

For an end-to-end comparison, refit `csubst search` to the simulated tips and
retain `OCNany2spe`, `OCSany2spe`, and `omegaCany2spe`. Recompute a fixed-denominator
ratio from the search O counts using the simulation table's null columns.
Comparing the two ratios shows the effect of using search's fitted expectations.
Check the N and S counts individually and report zero-S rows when comparing ratios.
