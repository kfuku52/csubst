# Review issue 9: experimental pilot and fixed-GTR inference

The five-structure pilot and fixed-parameter inference checks are complete.
Structural omegaC remains uncalibrated. No default predictor, production ASR,
GTR fitting, published fixed Q, codon S or significance-testing behavior changes.

## Validation

- Full sequential suite: **1,557 passed, 32 skipped** in this worktree. The skips
  require compiled Cython extensions, unavailable in this environment.
- New reference/inference/comparison tests: **23 passed**, including Gemmi-based
  mapping tests. All five actual structures were also extracted with Foldseek;
  all three actual predictor checkpoints completed inference on the panel.
- Repository lint, type checks, additional mypy for the new research modules,
  tool-script lint, documentation checks and `git diff --check` passed.
  The external Wiki checkout was not checked.
- Reference extraction was repeated after identifier validation was tightened;
  the complete reference manifests were identical, so predictor outputs did
  not need to be regenerated.


### Main-checkout integration verification

After applying this work to the main checkout on 2026-09-10, the complete
Python-path suite passed: **1,714 passed, 32 skipped** (30.42 s). Existing
compiled extensions target x86_64 while the validation runtime is arm64, so
this run explicitly used `CSUBST_DISABLE_EXTENSIONS=1`; native execution was
not verified. The initial run without that flag stopped during collection.

Repository Ruff and documentation checks, new-module mypy, tool-script Ruff,
and hygiene/diff checks scoped to all 60 delivered files passed. Whole-repository
hygiene and type checks encountered concurrent, unrelated long-tail changes:
a home path in `reports/longtail_20260910/INTEGRATION_VALIDATION.md` and missing
`ge`/`valid` type annotations in `csubst/omega_calibration.py`. Those files were
not edited as part of this delivery. The external Wiki was not checked.

## Reference panel and provenance

The panel contains selected A chains from PGK 2X15, GH19 4IJ4/4MCK, and
A1RDF1 5G4I/5G4J (the latter are stored under the source data's AGT2 directory).
The folder name is not treated as independent evidence of functional annotation.
All five files declare X-ray diffraction; the source files were read only.

- [Input panel](panel.json) and [reference manifest](references/manifest.json).
- Each reference subdirectory retains normalized coordinates, raw descriptors,
  Foldseek logs and a `residue_map.json` with author and label identifiers.
- The manifest records source/export hashes, experimental methods, Gemmi 0.7.5,
  Foldseek version/binary hash and masking protocol. Extraction used official
  Foldseek 10-941cd33; the macOS universal download archive SHA-256 was
  `6d8b07e188d443044f0c98db30aa3d4cca6750e97b50679372b54b45a664cae5`.

Predictors receive full chain sequences including unresolved residues, not a
concatenation of coordinate fragments. Scoring excludes undefined descriptors,
missing geometry, modified/unknown neighborhoods and affected chain breaks or
compressed partner distances. See the [exact protocol](../../docs/STRUCTURAL_OBSERVATION.md).

| Reference | Full chain length | Scored positions |
| --- | ---: | ---: |
| PGK 2X15 A | 416 | 365 |
| GH19 4IJ4 A | 205 | 145 |
| GH19 4MCK A | 201 | 189 |
| A1RDF1 5G4I A | 446 | 421 |
| A1RDF1 5G4J A | 446 | 421 |
| Total | 1,714 | 1,541 |

The two A1RDF1 structures have identical chain sequences. Of 421 jointly scored
positions, 39 (9.26%) have different reference 3Di labels. The references should
not be interpreted as a unique structural state determined by sequence alone.
This observation does not partition conformational differences, coordinate
uncertainty, and discretization effects.

## Three-predictor comparison

All backends ran on the same panel using MPS, with offline pinned checkpoints.
Package/config/source provenance is retained in each `predictions.npz`. Model
manifest files are snapshots of `references/manifest.json`; their `residue_map`
links resolve against the `references` directory. No calibrator was fitted; all
records belong to the declared pilot test split. Training overlap is unknown.

| Family / aggregation | ESM3Di-35M Q20 | ProstT5-CNN Q20 | ProstT5 generator Q20 |
| --- | ---: | ---: | ---: |
| PGK | 41.37% | 61.64% | 72.88% |
| GH19 | 38.62% | 63.47% | 41.02% |
| A1RDF1 | 40.86% | 58.67% | 70.55% |
| Pooled residues | 40.49% | 60.42% | 64.70% |
| Equal family weights | 40.28% | 61.26% | 61.48% |

Thus the pooled generator advantage depends on the family composition and
repeated A1RDF1 structures. This panel does not establish a general predictor
ranking and is not a basis for changing defaults.

ESM3Di/CNN log losses were 1.855/1.083 and multiclass Brier scores 0.730/0.522.
Generator probabilities were not obtained; its probability scores remain absent.
The longest contiguous scored error runs were 15/10/16 residues for
ESM3Di/CNN/generator. Given an error at the previous eligible position, error
fractions at the next position were 0.658/0.463/0.526. These are descriptive
statistics, not estimates of an independent-error model.

All predictor pairs had a larger pooled both-wrong fraction than the product
of their pooled marginal error fractions. Family/state heterogeneity and shared
references can contribute; this is not an independence test. The
[comparison JSON](comparison.json) includes these counts, structure pairs and
whole-family bootstrap percentiles. Only three families are represented, so
the bootstrap intervals must not be promoted to population-level guarantees.

Retained outputs:

- [ESM3Di metrics](esm3di-35m/metrics.json), [prediction artifact](esm3di-35m/predictions.npz).
- [ProstT5-CNN metrics](prostt5-cnn/metrics.json), [prediction artifact](prostt5-cnn/predictions.npz).
- [ProstT5 generator metrics](prostt5/metrics.json), [prediction artifact](prostt5/predictions.npz).

## Fixed-GTR inference and misspecification

The new reference API computes all node marginals and parent-child endpoint
joints with log-space pruning/outside recursion. It assumes a fixed normalized
reversible Q, branch lengths, stationary frequencies and independent tip
likelihoods. Tests compare against complete finite state enumeration, including
nonuniform four-state GTR; 20-state axes, rerooting, relabeling, missing sites,
zero-length edges and severe likelihood underflow are also covered.

[Synthetic results](fixed_gtr_results.json) use 10,000 sites and seed 9 with
four states and known ancestors. For the internal edge, the true endpoint-change
frequency was 0.0981:

| Data-generating observation error | Mean change probability: ignore error | Mean change probability: independent-error likelihood |
| --- | ---: | ---: |
| None | 0.0947 | 0.0947 |
| Independent, marginal error 0.15 | 0.1261 | 0.0939 |
| Fully shared across tips, marginal error 0.15 | 0.0809 | 0.0639 |
| Shared blocks, marginal error 0.15 | 0.1040 | 0.0793 |

Error integration improves the mean probability in the matched independent
scenario. It does not fix shared errors: the independence assumption is wrong
in the last two scenarios. These are conditional endpoint checks, not omegaC
FPRs. No parameter fitting, codon S or multiple-testing procedure is included.

## Reproduction and remaining work

Use the commands in [the workflow guide](../../docs/STRUCTURAL_OBSERVATION.md)
to build the panel, run each backend, compare artifacts and repeat fixed-GTR
simulations. Source-data paths are supplied through `--source-root`; none are
written by this workflow. Resource preparation is separate from offline inference.

Next scientific gates are a larger independently separated target-group panel,
an observation model accounting for the measured error dependencies, error-aware
GTR/branch-length fitting, and end-to-end omegaC null/power evaluation after
issues 2 and 4. Translate AA-ancestor sampling and issue-8 multi-branch history
dependence remain separate. The validation APIs remain separate from the
production native-GTR inference and expectation paths.
