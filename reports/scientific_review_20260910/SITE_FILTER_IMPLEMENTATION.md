# ID 11: analysis-site selection implementation and validation

2026-09-10. Implemented in the review worktree against the source task's current
uncommitted native 3Di fixes (source HEAD `dd37bee67b39199a1920600cf00c0ecb8566f477`).
The source directory was read-only during the initial implementation. The later
user-requested integration into the main repository is recorded below.

## Implemented behavior

- All sites are retained by default in the CLI and parameter normalization.
  Inspect/cache normalization agrees with the new default.
- Mask construction is separate from state/rate slicing. Existing explicit
  criteria retain their meaning; neither is described as zero expected mass.
- Opt-in `--site_filter_report yes` produces fixed-full-model OCN/OCS/ECN/ECS
  partitions for both criteria, by analyzed branch combination and requested
  category. It requires an unfiltered model-based search, supports native 3Di N
  plus codon S, and rejects unsupported urn reporting rather than presenting a
  different attribution as equivalent.
- Filtered searches and diagnostic searches write selection metadata and
  original-coordinate masks. Metadata states that selection is not repeated in
  the production count-null and records the analyzed child/parent pairs and
  tolerance. Observed zero-mass classification precedes `min_sub_pp`.
  Supplemental clade-permutation computations cannot overwrite the primary
  search report.
- The remaining direct 3Di urn prefilter is removed: fitting/ASR now uses all
  columns. Cache format 6 preserves the direct mask for that route as well as
  the native-model route; older caches rebuild/reject according to auto/yes.
  This is a consequence of changing urn fit semantics, not a claim that the
  previously fixed native mask/cache regression remained unresolved.
- Production omega P-values are unchanged. The report contains raw counts
  before pseudocounts and long-tail calibration and does not claim adjusted P/q
  values or biological FPR control.

The interface and limitations are documented in
[the site-filter guide](../../docs/SITE_FILTERING.md).

## Real-data sensitivity: bundled PGK

All three searches used the bundled PGK alignment, rooted tree and existing
IQ-TREE intermediate files, copied to temporary storage. Other settings were
identical: codon-model expectation, arity 2, one thread, default posterior,
pseudocount and long-tail options. Full inputs contained 417 codon sites. All
three searches returned the same 1,620 branch pairs.

| Selection | Retained sites | Median omegaCany2any | Median uncalibrated omegaCany2any |
| --- | ---: | ---: | ---: |
| no | 417 | 0.48425 | 0.48540 |
| tip_invariant | 402 | 0.47715 | 0.47775 |
| zero_sub_mass | 403 | 0.47760 | 0.47855 |

Fixed-full-model contributions excluded by `tip_invariant` summed to
OCNany2any = 7.569833e-8, OCSany2any = 0.000195, ECNany2any = 1.724921,
and ECSany2any = 1.589776. For `zero_sub_mass`, observed contributions were
zero, but ECNany2any = 1.536775; ECSany2any was zero in this dataset. These
are sums across overlapping branch combinations, not independent tree-wide
event counts. Neither criterion guarantees expected zero in general.

Base-category all = retained + excluded holds within 1.03e-14 after TSV
round-trip. Derived dif categories differ by at most 1e-9, consistent with the
run's `float_tol=1e-9` zeroing convention. Rank correlations with the unfiltered
run exceed 0.99996, but this does not establish equivalence: the largest absolute
omegaC change was 255.161 for tip filtering, in the high-ratio tail. No population
claim or biological significance follows from this one-family comparison.
Numerical results are in [site_filter_pgk.json](site_filter_pgk.json).

To reproduce, copy all `csubst/dataset/PGK.alignment.fa*` files and
`PGK.tree.nwk` into a temporary input directory. Pass those five explicit
`--iqtree_iqtree`, `--iqtree_log`, `--iqtree_rate`, `--iqtree_state`, and
`--iqtree_treefile` paths to each search, with `--max_arity 2 --threads 1`.
Use separate output directories and enable `--site_filter_report yes` only
for the `no` run. This prevents the comparison from refitting or rewriting
bundled IQ-TREE inputs.

## Limited null pilot

[calibrate_site_selection.py](calibrate_site_selection.py) generates a known,
symmetric four-codon CTMC on a fixed four-tip tree. Every pseudo-alignment gets
new exact pruning posteriors, the actual CSUBST mask, codon rescaling and expected
count kernels. The candidate pair A/C is fixed in advance. The statistic uses
any2any counts with fixed symmetric alpha=1 on observed and expected counts,
identically in reference and evaluation draws, with long-tail off. This avoids
claiming the unresolved production ID2/ID4 path is fixed.

The pilot's model parameters are known: it does not refit IQ-TREE, simulate 3Di,
estimate an empirical prior, test ASRV options, or redo candidate discovery.
Pruning was checked independently by complete enumeration of internal states.
Reference and evaluation samples use independent seed streams. Undefined
all-sites-excluded draws are no-test outcomes (P=1), retained in the evaluation
denominator and included as a minus-infinity atom in the reference statistic.

With seed 20260910, 12 sites, 199 reference and 200 evaluation replicates:

| ASR | Criterion | Rejections at 0.05 | Rate | 95% binomial interval |
| --- | --- | ---: | ---: | --- |
| posterior | no | 10/200 | 0.050 | 0.0242–0.0900 |
| posterior | tip_invariant | 11/200 | 0.055 | 0.0278–0.0963 |
| posterior | zero_sub_mass | 10/200 | 0.050 | 0.0242–0.0900 |
| ML | no | 7/200 | 0.035 | 0.0142–0.0708 |
| ML | tip_invariant | 7/200 | 0.035 | 0.0142–0.0708 |
| ML | zero_sub_mass | 7/200 | 0.035 | 0.0142–0.0708 |

The binomial intervals are conditional on the shared finite reference sample;
they do not include reference Monte Carlo uncertainty. This is a pilot, not a
predeclared equivalence or FPR acceptance test. Its deliberately mismatched
unfiltered-reference comparison does **not** establish FPR inflation here.
The scientific case for differentiating estimands comes from the excluded
contributions and different pipeline/statistic definitions, not a blanket claim
that every filter inflates false positives. No power study was run.

See [the 12-site results](site_filter_null_pilot.json). A second two-site pilot
(seed 20260911) exercises sparse counts and all-sites-excluded outcomes; see
[the two-site results](site_filter_null_sparse.json). In that pilot, tip filtering
produced 12/200 posterior and 13/200 ML no-test outcomes; ML zero-mass filtering
also produced 13/200. They remained in the evaluation denominator.

Run the script with `--reference 199 --evaluation 200 --sites 12
--seed 20260910 --output /tmp/site-filter-null.json` to reproduce the first pilot.

## Validation and remaining completion gates

Final full sequential suite: **1583 passed, 3 skipped**. The three skips require
PyTorch >=2.6; this environment has 2.2.2. The existing requests dependency
warning remains. `make lint`, `make typecheck`, and `git diff --check` passed.
The report tests also passed with Cython extensions disabled (6 passed).
`make test` could not start because pytest-xdist was not installed; the supported
sequential full suite above covered all test lanes instead. The ID11-only patch
was checked for applicability against the source directory without applying it.

Numerical regression tests cover positive observed mass at tip-invariant sites,
positive expected mass at observed-zero sites, retained-site expectation changes,
empty partitions, category selection, all/retained/excluded additivity, and
report-on/off equality. Both expm/eigen and conventional/native model routes are
covered. Existing mask tests cover missing/ambiguous tips and selected branches;
cache tests cover preserved direct masks and missing-mask rejection.

The full-pipeline selection-adjusted P/q-value work remains dependent on consistent
pseudocount transformations and joint dif null generation (IDs 2 and 4). Before
claiming production FPR calibration, define a joint codon/3Di generator where
applicable, repeat fitted nuisance estimation and search as intended, predeclare
acceptable error, and run independent simulation grids with adequate reference
and evaluation sizes, uncertainty and power reporting. These are scientific
completion gates, not functionality supplied by the count-report flag.


## Main-repository integration

The user subsequently requested committing the implementation to the main
repository. ID11 was applied on top of `cac5fe3` (the integrated ASRV work),
preserving that work and the source task's native 3Di changes. The native 3Di prerequisites are recorded in preceding commit `40f0c99`;
ID11 is committed separately. No push was requested.

Closing the ID11 task does not imply that production selection-adjusted P/q
values are calibrated. The remaining completion gates above are retained in
this committed report and the site-filter guide so they do not depend on the
conversation remaining open. ID2/ID4 statistical work remains separate.

Integration validation on the main repository: **1630 passed, 3 skipped**
(PyTorch-version skips), with the existing requests dependency warning.
`make lint`, `make typecheck`, and diff whitespace checks passed.
