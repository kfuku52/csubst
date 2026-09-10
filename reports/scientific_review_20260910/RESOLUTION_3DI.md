# Issue 1: independent fitted 3Di N and codon S expectations

2026-09-10. The initial guard against all 3Di/codon_model combinations has been
replaced with automatic model routing for direct ASR with uniform GTR.
Issues 2–13 in the original review remain outside this change.

With `--nonsyn_recode 3di20`, the default `--expectation_method codon_model`
now uses the fitted 3Di model for N and the existing codon model for S.
An explicit `urn` choice remains unchanged. Translate ASR, nonuniform rates,
and other 3Di model families are rejected for model-based expectations.

## Implementation

- Read IQ-TREE's ModelMorph exchangeabilities from its checkpoint (approximately
  ten significant digits), reconstruct the normalized GTRX+FQ generator, and
  explicitly map the fitted morphological symbols to the internal 3Di order.
  No amino-acid generator is used for 3Di. The initial model is fitted GTR,
  not a published fixed Q.3Di.AF or Q.3Di.LLM model.
- Carry independent Q, stationary frequencies, unit site rates and fitted
  branch lengths into materialized, sparse-fused and tensor-free expectations.
  Check branch identities by descendant taxa and numerical labels.
- Preserve invariant sites during fitting. Apply requested analysis filtering
  afterward, slicing the model's site axis with the state tensors.
- Store the complete context with state posteriors in version 4 caches.
  Validate dimensions, state order, finiteness, stationarity, normalization and
  site/branch axes. Incompatible caches are rebuilt with `auto`, rejected with
  `yes`; invalid model contexts cannot fall back to amino-acid Q.
- Compute codon S exposure from synonymous and amino-acid observed changes,
  so 3Di changes cannot contaminate codon branch-length rescaling. S Q and
  codon site rates retain their conventional behavior.

The model definitions follow [IQ-TREE's documentation](https://www.iqtree.org/doc/Substitution-Models#binary-and-morphological-models).
The checkpoint upper-triangle exchangeability order follows
[IQ-TREE 2.3.6 ModelMarkov](https://github.com/iqtree/iqtree2/blob/v2.3.6/model/modelmarkov.cpp).

## Validation

- Full sequential suite: **1563 passed, 3 skipped**. The three skips require
  PyTorch >=2.6; this environment has 2.2.2. The pre-existing requests dependency
  warning remains. `make lint`, `make typecheck`, and `git diff --check` passed.
- Real IQ-TREE 2.3.6 fits with four states, twenty states and noncontiguous
  symbols; invariant-site retention and normalized model context verified.
- Expected states checked against independent SciPy matrix exponentials.
  Materialized, fused sparse and tensor-free projections agree.
- Consistent state-label permutations preserve aggregate N statistics.
- Model/state cache round trips verified; invalid or missing Q rejected.
- Complete synthetic search runs (predictor replaced with a deterministic
  fixture, both IQ-TREE fits executed): fresh and cached CB tables identical.
  With a four-state structural fixture and actual synonymous changes, all
  OCS/ECS and uncalibrated dSC columns match the conventional codon run on the
  same sites. Summed OCSany2any = 40.9131; ECSany2any = 37.2486 across eight pairs.

Default long-tail calibration uses the N distribution to transform dSC.
Consequently calibrated dSC can differ even though OCS/ECS and uncalibrated
codon dSC are unchanged. Use the `_nocalib` columns or disable long-tail
calibration when checking component parity.

## Remaining limits

This change fixes the state-space mismatch; it does not establish biological
calibration of structural omegaC. The existing marginal-posterior approximation,
3Di prediction uncertainty and long-tail calibration are unchanged. Uniform
GTR may be underdetermined for short alignments. Published fixed 3Di matrices,
rate-mixture integration and translate-model validation remain future work.

The original `reproduction_results.json` is the pre-fix snapshot. The
`reproduction_results_after_3di_fix.json` records the initial guard-only stage;
the direct codon-to-3Di matrix conversion remains correctly prohibited.
Native-model regression tests are in `tests/integration/test_3di_model_expectations.py`.

## Follow-up audit and fixes (2026-09-10)

A deeper review after the initial implementation found three reproducible
problems, now fixed. The validation numbers above describe the earlier stage.

1. **Internal-node identity was changed by rerooting.** For an unrooted
   `(A,B,(C,D)Node2)Node1`, the old transfer function named the newly inserted
   root `Node1` and erased `Node1` from its original vertex. ASR rows were then
   attached to the wrong location. Names are now restored using the incident
   partitions of leaf taxa, which identify an unrooted internal vertex.
   This correction applies to the shared codon tree reader as well as 3Di.
   A newly inserted root no longer borrows an unrelated posterior; for uniform
   3Di GTR its posterior is calculated by scaled pruning with the fitted Q.
   Codon roots without an actual ASR row remain missing, so the existing
   sub-root exclusion policy correctly applies. Previously reported branch
   counts and eligible combinations can therefore change.
2. **Zero input root lengths erased the fitted edge.** When both input
   root-adjacent lengths were zero, the old split multiplied the fitted length
   by zero twice. Both halves now receive half of the fitted total; a positive
   input total still specifies the split ratio. Negative/nonfinite input root
   lengths are rejected. This does not claim to identify the root position
   from a reversible model.
3. **Native 3Di filtering could fall back to codon invariance.** Keeping all
   sites during fitting had bypassed the old 3Di prefilter mask. The native
   model context now retains the full tip-invariant 3Di mask, saves it in
   version 5 caches and slices it with the other site arrays. Fresh and cached
   analyses use the same structural site criterion. Older caches are invalid.

Validation after these fixes:

- Full suite: **1569 passed, 3 skipped**, with the same PyTorch-version skips
  and pre-existing requests warning. Lint, type checks and diff checks passed.
- Real IQ-TREE fits: each retained internal-node posterior matches its original
  `.state` row. An independent traversal conditioning on both sides of each
  edge matches these posteriors within IQ-TREE output precision, across all
  three tested alphabets. Inserted-root likelihood checked independently,
  including partially and wholly missing columns.
- Zero-root-length splitting preserves the fitted edge total. Existing
  missing-internal-node contraction tests retain their purpose using explicit
  synthetic-node fixtures rather than depending on the old name-loss bug.
- Model cache reload and 3Di-mask site filtering verified together.
- The complete synthetic search smoke was rerun: fresh/cached output matches,
  and OCS/ECS and uncalibrated dSC match the conventional codon analysis with
  the same settings. After the root correction, summed OCSany2any=25.0768 and
  ECSany2any=37.2486. The earlier observed total above is superseded.

The pseudocount/P-value inconsistency was reproduced again but intentionally
left for the next implementation step; see
[the proposed plan](PLAN_PSEUDOCOUNT_PVALUES.md). No false-positive-rate
calibration or predictor-accuracy claim follows from these numerical checks.
