# Review issue 9: prediction measurement foundation

This implementation preserves and measures prediction uncertainty. It does not
complete the biological calibration of structural omegaC. GTR remains the
intended model; no published Q.3Di.AF/LLM matrix is introduced.

The subsequent [experimental pilot and fixed-GTR inference report](../3di_pilot_20260910/README.md)
records the next completed stage. The limits below describe this initial phase.

## Implemented

- An opt-in record API retains all 20 encoder logits, canonical state order,
  model and preprocessing identities. Existing hard predictions and ASR APIs
  retain their behavior. Greedy ProstT5 records explicitly lack probability
  estimates; none are fabricated from its character strings.
- Versioned, pickle-free artifacts preserve logits and provenance, validate
  identities and shapes, and refuse overwriting existing outputs. They are
  separate from the hard-prediction cache and the ASR state cache.
- The offline validation tool accepts residue-matched references and reports
  masked coverage, confusion matrices, Q20, macro-family Q20, balanced accuracy,
  Brier score, log loss and reliability/ECE. It rejects declared family/sequence
  leakage between calibration and test splits and separates reference sources.
- A specified observation-error sampler supports state-dependent confusion,
  shared errors across tips and blocks of sites. It is a stress-test component,
  not an evolutionary simulation or an omegaC null generator.

See [usage, input contract and remaining gates](../../docs/STRUCTURAL_VALIDATION.md).

## Automated validation

- Full sequential suite in this worktree: **1,534 passed, 32 skipped**. All skips
  require compiled Cython extensions, which were not built in this environment.
  This is not the source task's suite including its uncommitted issue-1 changes.
- Focused prediction/validation/CLI-artifact tests: **64 passed**.
- `make lint`, `make typecheck`, additional mypy checks for both prediction and
  validation modules, checks for the new tool scripts, and `git diff --check`
  passed. External Wiki links were not checked against a separate Wiki checkout.

Tests include analytically known uniform-probability scores, stable extreme
log loss, label-order/tie handling, missing references, split leakage, artifact
corruption/identity mismatches, hard-only output, OOM retry, known zero-error
behavior, correlated observation errors, and command-line artifact rescores.

## Actual checkpoint checks

[Machine-readable results](runtime_checks.json) record CPU checks with the
pinned ESM3Di-35M and ProstT5-CNN models. Inputs include 1- and 40-residue
sequences, a duplicate and an empty sequence (41 unique nonempty residues).
Both models passed hard-prediction equality, finite logits, normalized softmax
and exact artifact round trips. These small checks verify the output contract;
they do not measure accuracy against a real structure. The generator's
hard-only record path is covered by fixtures, not a new checkpoint experiment.

## Shared-error reproduction

Run `python tools/reproduce_3di_observation_error.py` from the repository root.
With a known all-A ancestor and two true all-A descendants, only an A-to-C
observation error of 0.2 is added. For 100,000 sites and seed 9:

| Observation errors | Residue accuracy | Both endpoints falsely C | Analytical probability |
| --- | ---: | ---: | ---: |
| Independent | 0.800575 | 0.04057 | 0.04 |
| Fully shared | 0.799680 | 0.20032 | 0.20 |

Thus the same marginal prediction accuracy can accompany fivefold different
joint false-endpoint probabilities in this constructed model. No ASR or
omegaC significance test is run; these numbers are not empirical omegaC FPRs.

## Not established

No target-group experimental structure panel was supplied for this phase.
There is no fitted probability calibrator or observation likelihood, no
error-aware GTR fitting/ASR, no translate posterior sampling, and no end-to-end
omegaC FPR or power estimate. These remain explicit gates in the linked guide.
The source task's issue-1 model/root/cache changes and ongoing issue-2/4 work
are not copied or modified by this isolated change. Integration must preserve
those changes; the new implementation is concentrated in the prediction and
validation modules.
