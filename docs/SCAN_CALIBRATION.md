# Scan calibration and its assignment null

`csubst scan` defaults to `--scan_pvalue_calibration full_scan`. Its empirical
values compare foreground assignments on **fixed data**. They do not by
themselves establish a biological null of no adaptive convergence, or control
error across genes, repeated runs, or choices of settings made after inspecting
the results. `score_rate_enrichment` is the ranking score (larger means greater enrichment).
Its asymptotic tail is a separate exploratory diagnostic; see
[scan scores and bootstrap inference](SCAN_INFERENCE.md).

## What is held fixed

The tree, branch lengths, posterior states, state masks, substitution tensors,
Q matrices, and scan settings are fixed. ASR is not rerun within an assignment
trial. Branch eligibility is determined by scan's branch metadata, including
its effective state-bearing parents. An eligible clade has at least one
analyzable candidate branch in the selected unit mode. Every observed component
must be eligible; components cannot be silently lost after sampling.

Size bins are computed from eligible clades independently of foreground labels.
The observed count in each bin is held fixed. In `lineage` mode, each lineage's
component count in each bin is held fixed as well, and the allocation of clades
to lineages is sampled. Clades within a configuration cannot overlap. The
observed configuration is always included in the allowed space.

The conditional assignment null assumes that the observed configuration is
uniform over this space, conditional on the fixed data and counts. For an
observational foreground, that is an additional scientific assumption. Similar
tip counts alone do not ensure comparable branch lengths, phylogenetic depth,
composition, missingness, ASR uncertainty, or control-branch opportunities.
Rate normalization does not establish label exchangeability. The diagnostics
provide clade sizes, depths, lengths, state availability and entropy for
checking these differences; they do not certify exchangeability.

Clade assignment calibration currently accepts **one trait per scan**. Independent randomization
of several traits would not preserve their observed dependence. A calibrated
multi-trait analysis needs a specified joint assignment model. Multi-trait
candidate listing remains available with `--scan_pvalue_calibration none`.
The fitted sequence `parametric_bootstrap` also supports multiple fixed traits;
it simulates a joint alignment rather than randomizing trait labels.
Separate single-trait runs do not correct across traits.

## Sampling and tail probabilities

The sampler first counts bin/group allocations ignoring clade overlap. If that
proposal space has at most 10,000 configurations, it enumerates all valid
allocations. If the valid space size `M` is no greater than
`--scan_n_permutations`, all `M` configurations are evaluated, including the
observation. The exact upper-enrichment tail is the fraction whose ranking
score is at least the observed score; there is **no extra +1** in this
complete enumeration.

Otherwise, the requested `B` Monte Carlo configurations are drawn with
replacement, including the possibility of the observation. When a valid list
is available, draws are uniform from that list. For a larger space, each bin's
full allocation is proposed uniformly, then the **whole configuration** is
rejected if clades overlap. Clades are not chosen sequentially from a pool
conditioned on previous accepted choices. Accepted complete proposals are
uniform over the valid space. There is a limit of 10,000 proposals per trial;
exhausting it makes calibration unavailable, without changing the space.

Monte Carlo tails use `(1 + extreme_count) / (B + 1)`, with ties included.
Repeated configurations remain in this denominator; the number of unique
configurations is a diagnostic, not a replacement denominator. The nominal
resolution is `1/M` for enumeration or `1/(B+1)` for Monte Carlo, although ties
can make attainable values coarser. A nonuniform foreground assignment process
is not repaired simply by adding 1 or by using this uniform sampler.

## Candidate selection and the testing family

| Mode / column | Interpretation |
| --- | --- |
| `candidate_fixed`, `p_rate_enrichment_empirical` | Re-evaluates the observed candidate under each assignment. Useful for a candidate specified independently of these data, or as an exploratory comparison after discovery. It does not repeat selection or correct the selected family. |
| `full_scan`, `p_rate_enrichment_empirical` | Candidate-key tail with rediscovery: an absent or explicitly untestable key contributes score −∞. This is not the scan-wide adjustment. |
| `full_scan`, `p_rate_enrichment_empirical_maxT` | Compares each observed score with the maximum score over **all testable candidates rediscovered** in each assignment. |

The full-scan family contains sites and match classes searched for this trait
with the specified settings. A valid assignment null supports a conditional
complete-null/global test. Strong FWER control with true and false nulls mixed
requires additional assumptions or validation; it is not claimed here.

BH is applied only within each trait × match output family.
BH columns computed from asymptotic or candidate-wise empirical P values are
exploratory after foreground-based selection. Applying BH does not fix selection
bias, establish FDR control, or extend the family across genes and runs.

## Empty scans and failures

A successful full-scan trial with no testable candidates contributes maximum
score −∞. Finite nonnegative event mass with zero target/control exposure is an
explicit no-test outcome: the row is marked `scan_rate_testable=false`, its P
values stay undefined, and the trial remains in the denominator. This does not
repair the instantaneous-Q versus endpoint-change mismatch.

Other undefined or invalid statistics, or failed trials, make empirical P/q
unavailable for the run. The denominator is never reduced to successful trials.
An invalid observed statistic also makes assignment calibration unavailable.

`--scan_permutation_sample_original` now defaults to `yes`, and
`--scan_permutation_retry_sample_original` defaults to `no`. When calibration is
enabled for `candidate_fixed` or `full_scan`, the former `no` / `yes` combination is rejected with a migration
message. There is no fallback to a different assignment space.

## Outputs

In addition to `csubst_scan.tsv` and `csubst_scan_units.tsv`, each scan writes
`csubst_scan_calibration.json` under the configured output directory/prefix.
The JSON is written even if there are no candidate rows or calibration is
disabled. Schema version 1 records:

- the null assumption, inference scope, resolved exposure, settings and seed;
- fixed bin boundaries, eligible clades, group counts, and known proposal/valid
  space sizes (unknown valid sizes are `null`, not an overlap-ignorant count);
- the observed configuration and its stable ID;
- per-trial configuration, ID, seed (Monte Carlo), proposal count, candidate
  count, testable-candidate count, maximum score (`null` for −∞), status and failure reason;
- aggregate successes, failures, distinct configurations, observation inclusion
  count, sampling mode and P-value resolution.

Configuration groups contain numerical stem branch IDs. The clade catalog maps
these to leaf names, parent IDs and diagnostics. Rejected complete proposals
are counted but are not retained as successful trials. If no candidates were
observed, the status is `no_observed_candidates`, no trials are needed, and the
global non-rejection value is 1. The assignment diagnostics have status `disabled` for `none` or
`parametric_bootstrap`; bootstrap status is recorded separately in
`csubst_scan_inference.json` and its bootstrap manifest.
`conditional_assignment` means computation succeeded under the documented
assignment null, **not** that biological calibration has been demonstrated.

The table adds `scan_calibration_status`, `scan_calibration_null`,
`scan_calibration_scope`, `scan_permutation_sampling`,
`scan_permutation_unique_count`, `scan_permutation_space_size`, and
`scan_pvalue_resolution`. Existing success/failure counters count evaluated
configurations, which can be fewer than the requested number in exact mode.
P/q values and `scan_pvalue_resolution` use scientific notation in the TSV,
independently of the fixed precision used for other continuous measurements.

## Verification boundary

Small-space regression tests enumerate the assignment null, check uniform
sampling, preserve lineage allocations, repeat candidate discovery, and verify
that nonfinite or failed trials cannot create small P values. These checks
establish the implemented assignment law and tail calculations.

Biological calibration additionally requires independent sequence generation,
model fitting and ASR, then the complete discovery and decision procedure.
Vary balanced/pectinate trees, foreground sizes and separation, branch/rate
heterogeneity, random and foreground-correlated missingness, and mixtures of
null and signal sites. Count datasets without discoveries in the outer
denominator. Report FWER with binomial intervals, power, unavailable-run rates,
and the exact tested settings. A label-randomization check or a passing unit
suite is not a substitute for this experiment.

The reproducible [finite-state calibration check](../.github/scripts/scan_calibration_check.py)
generates sequences, fits a global rate on the supplied tree, reconstructs
marginal states with an independently tested pruning implementation, and runs
CSUBST's substitution tensor builder and full scan:

```bash
python .github/scripts/scan_calibration_check.py \
  --replicates 1000 --workers 2 --sites 64 --states 20 \
  --require-calibration-bound \
  --output reports/generated/scan_id6_calibration.json
```

This check uses `state_aware` exposure, raw fitted lengths, distinct clade-size
bins (`min_clade_bin_count=1`), and an equal-frequency finite-state model. Its
matched assignment scenarios include balanced/pectinate trees, larger clades,
heterogeneous lengths and random missingness. Foreground-correlated missingness
and mixed signal/null sites are reported as sensitivity analyses. Every
dataset, including no-discovery and unavailable runs, is reported; unavailable
runs cannot pass the matched-null validation criterion. The default criterion
requires the upper endpoint of a two-sided 95% binomial interval to be at most
`alpha + 0.01`, with zero unavailable runs. This is a measured boundary for this
simplified model, not validation of codon-Q or native 3Di inference.

For the assumptions behind random-permutation and conditional Monte Carlo
tests, see [Hemerik & Goeman (2018)](https://link.springer.com/article/10.1007/s11749-017-0571-1).
