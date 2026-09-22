# Test-suite pruning, 2026-09-22

Scope: all 141 tracked pytest modules at `76c9124`, including tests of validation
scripts, plus the output-lifecycle module and foreground regression added by
`25419ca` during this audit. The final change is based on `25419ca` (1.16.5) and
publishes as 1.16.6. Test helpers and pytest/Makefile/CI selection were checked for
references affected by the removals. Untracked experiments were not changed.

The decision was whether removal could hide a realistic defect, weighed against
duplicate setup, subprocess execution, mock maintenance, and coupling to private
implementation choices. No coverage percentage or test-count target was used.

## Decisions

| Area | Removed or consolidated | Remaining regression signal |
| --- | --- | --- |
| Packed sampling | Two fake Cython classes copied Python bit operations | One real native/Python comparison, known packed bytes across a byte boundary, existing urn distribution tests |
| Logo layout | Fake transforms, glyphs, axes and copied sizing formulas | Actual Matplotlib glyph bounds check centering, slot fit and narrower I; missing-logo rendering still checked |
| Figure sizing | Four tests repeated constants and formulas | Growth with leaves/spacing and custom/default height limits in one scenario |
| Rendering details | Cap/join style and title-coordinate constant snapshots; placeholder dimensions | Actual SVG/PDF workflows, missing states, labels, highlights and root-marker clearance remain; exact cosmetic constants are intentionally not frozen |
| Site labels | Private dictionaries, channel indices, repeated label helpers | Bar-chart row labels/order, set-mode rendered labels, channel-label output and scalar branch IDs |
| Tree helpers | Tiny-tree unique-ID check and duplicate clade/species helpers | 64-leaf ID boundary, complete highlight routing and species-overlap detection/rendering |
| Pseudocounts | Repeated low-level default/invalid-input checks and internal weight-map shape | Public parameter validation, prevalidated auto-alpha output, all-category null and additive-identity checks |
| Omega calibration | Wrapper-versus-callee p-value comparison, repeated quantile rejection and column renaming | Hand-calculated p-values, calibrated nulls, longtail preservation and rejection of double calibration |
| FASTA | Plain alignment smoke test, isolated whitespace case, eight whitespace/compression combinations | Combined spaces/tabs/CRLF/multiline records in both plain and gzip paths; identifier and sequence assertions |
| Dependency layout | AST scan forbidding every Bio import | Clean dependency installation and actual FASTA/vendor behavior; source import spelling is not a runtime contract |
| IQ-TREE state | Lookup-table internals and duplicate root-row loading | Leaf ambiguity/native parity and streaming root-state values |
| Rate models | Weaker diagonal check and repeated fixed-recoding PNG generation | Canceling nonzero diagonals, real PCA output, and no-recoding orchestration |
| Recoding plots | Repeated no-recoding/disabled-3Di writes and nested outlier helper | Disabled predictor assertion folded into fixed-PCA output; enabled predictor and inset decisions retained |
| Tables | Duplicate cutoff parsing errors, compound/alternation parsing | Parameter boundary errors, whitespace compound evaluation, actual regex matching and comma quantifiers |
| Simulation metrics | Perfect-AUC and average-precision helper smoke tests | Full foreground metric output now checks precision-at-k as well; tied AUC retained |
| Scan | Two-state spectral duplicate, helper-only 3Di fallback and zero-event checks | Independent bridge quadrature, asymmetric spectral comparison, complete 3Di/zero-threshold scan workflows |
| Scan storage/calibration | Memmap descriptor smoke test and finite-maxT smoke test | Worker context reopening/aliasing, exact assignment tails and competing-candidate maxT values |
| Foreground logging | Entire mocked search setup to assert an exact progress sentence | Real candidate generation, dependency filtering and multi-trait selection |
| CLI validation | Repeated numeric error formatting and ordinary help option lists | Parameter range tests, representative clean CLI errors, advanced help visibility and real command workflows |
| Input protection | 36-way option/alias/parser product reduced to 12 cases | All alias x parser paths for alignment, plus registration of each other input option and inferred input protection |
| Shared options | Repeated backend/command, seed, posterior and integer-text products | All command defaults, search/scan opt-out branches, distinct backend destinations, detailed seed validation and exact int64-limit text |

Retained suites protect independent scientific reference calculations, rare-event
precision, missing-data semantics, invalid input, real dense/sparse/native parity,
cache invalidation, process cleanup and concurrent resource publication. Mocks
remain where they inject external failures or inspect a meaningful external
command. Their presence alone was not grounds for deletion. The newly added
output-lifecycle tests protect distinct overwrite, archival and finalization
failures, so they were retained.

The initial collection had 2,100 cases; the incoming commit added 14. The final
collection has 1,990 cases: 124 fewer than its parent. There are 64 removed test
functions and three consolidated replacements. Test code shrank by 984 lines
(1,076 removed, 92 added). These are scope measurements, not performance claims.

## Verification

Platform: macOS arm64, Python 3.12.14. Native extensions built locally. Optional
Torch, Transformers, PEFT and Gemmi dependencies were installed for final checks.
An initially unusable cached ETE4 binary was rebuilt from source in the isolated
environment; repository dependency constraints were not changed.

| Command | Result |
| --- | --- |
| `make test` before edits at `76c9124` | 2,091 passed, 9 optional-dependency skips |
| Focused edited scenarios | 130 passed |
| `make test` after integration with `25419ca` | 1,987 passed, 3 skipped |
| `make test-native` | 15 passed |
| `CSUBST_DISABLE_EXTENSIONS=1 make test` | 1,957 passed, 33 skips (30 native-only, 3 IQ-TREE) |
| `make lint typecheck` | Passed |
| `make package` | sdist/wheel built; both passed Twine checks |

The three final ordinary-suite skips require an IQ-TREE executable, which was
not available. Remote Wiki validation and hosted CI/platform lanes were not run
locally.
