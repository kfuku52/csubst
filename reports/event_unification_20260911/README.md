# Cross-command event posterior unification

Search, sites and scan now use the same fitted codon model selection and
observation eligibility. All explicitly supplied IQ-TREE intermediate files
retain their reported model; changing the CLI model default does not silently
refit them. Explicit `--iqtree_redo yes` still requests a new fit.

GY+F/GY+FQ, including rate-category models, select matching precise sidecars
when available. Report-only fits remain conditional on the same reported
parameters in every command. Serialization intervals detect sidecar mismatches;
the fitted bootstrap retains its independent likelihood check. Derived rate
matrices are built after model precision is selected.

The original tip emissions determine eligible edges. Missing tips remain latent
in pruning, but unobserved child subtrees do not contribute events, synonymous
counts, exposures or analytical target branches. S/N reporting lengths use the
same eligible-site denominator. Conservation summaries count original observed
tips. TSV boundaries display excluded entries as `NA` with coverage fields,
while numerical tensors remain finite. Site positions are mapped to the original
alignment, including the one-based sites output convention. Cache fingerprints
include fitted generators, rates, tree, original emissions and projections.

## Verification

`verify.py` compares every N event for search, sites and scan on GY+F, GY+FQ,
GY+F+G4, MG+F3X4, PGK and PEPC (18 command runs). Search requests `spe2spe` to
retain the full inspectable tensor. These are deterministic conditional
probability comparisons, not comparisons of separately refitted models.

Across the verified runs, Q, pi and eligibility masks agree exactly. All-event
probabilities agree within `atol=1e-12, rtol=1e-10`; the largest observed
absolute difference is below `3.4e-16`. Separate PGK verification with default
search projections agrees exactly for maximum events and within `1.7e-15` for
branch/site sums. Unit checks cover projected/full storage, rate mixtures,
missing observations, true zero-length edges, filtering, bridge masks, analytical
targets, precision sources, cache invalidation and TSV coverage/NA semantics.

A GY+F fitted bootstrap with two missing tip observations completed all three
refits and produced 57 candidate rows. Three replicates are a workflow smoke
test, not a useful calibration resolution.

One fully missing site in an IQ-TREE 2.3.6 GY+F fit with an absent codon exposed
an upstream likelihood discrepancy: its `.sitelh` reports `0.693147`, whereas a
normalized CTMC necessarily gives `0` for that site. Informative sites agree
within the printed `.sitelh` precision. The original verification rejected
that fit. Following the requested policy change, a finite mismatch now emits a
warning and bootstrap continues; the discrepancy is retained in provenance.
No guessed correction or change to the comparison tolerance is introduced. The
shared joint event inference and missing-data reporting remain valid.

The final required checks passed: **2,175 tests**, 5 optional-dependency skips,
16 native checks, lint and type checks. See `validation.json` for measurements
and source hashes, and `checks.log` for the check output.

## Runtime and memory

`benchmark.py` runs fresh single-threaded search processes on identical PGK/PEPC
fits. Each phase has one warmup and three measured runs, with alternating phase
order. The baseline restores the pre-unification paths exercised by search using
`benchmark-baseline.patch`, since the older git HEAD also lacks earlier work in
this task. Scientific cb columns on eligible rows retain equivalent values.

| Dataset | Before, median seconds | After, median seconds | Before peak RSS, MiB | After peak RSS, MiB |
|---|---:|---:|---:|---:|
| PGK | 2.404 | 2.409 | 222.9 | 223.8 |
| PEPC | 6.354 | 6.603 | 412.4 | 410.5 |

PGK is effectively unchanged; PEPC takes about 4% longer with input fingerprint
validation and reporting eligibility. Memory is comparable. No speedup claim is
made. Emission fingerprints hash packed support and nonzero values to avoid
repeated hashing of the dense codon axis.

## Reproduction

Use the same Python runtime/dependencies as the checkout and IQ-TREE on PATH:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reports/event_unification_20260911/verify.py --workdir /tmp/csubst-event-check
```

For timing, copy the current `csubst` package into an isolated baseline directory,
apply `benchmark-baseline.patch` there with `patch -p1`, then run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reports/event_unification_20260911/benchmark.py --baseline-root /tmp/csubst-event-baseline --workdir /tmp/csubst-event-timing
```

The reverse patch restores only the paths used by this search benchmark. It is
not a supported alternative implementation or a baseline for scan bootstrap.

## Warning policy follow-up

The originally rejected 300-site alignment now emits a likelihood warning and
completes all three fitted-bootstrap replicates (57 candidate rows, no failed
replicates). Provenance retains the mismatched likelihoods and check status.
All 2,175 tests and the native/lint/type checks passed again.
`warning-followup.json` records this run and the updated bootstrap source hash;
`validation.json` retains the original verification snapshot.
