# ID 3: independent context and clade cross-fitting

2026-09-10. Implementation based on repository commit `c4e6d82`.

## Outcome

The response-derived context path is removed. Active weighting requires an external
context table and provenance, uses monophyletic outer holdouts and inner tuning,
and applies exactly the resulting per-branch probabilities to ASRV sn expectations.
Insufficient clades and unsupported combinations fail explicitly. Off remains the default.

**This resolves direct response reuse in conditional count prediction; it does not
establish a biological epistasis correction.** The independent simulations below
do not demonstrate predictive benefit for the supplied low-dimensional context.
Therefore positive beta remains a tuning outcome, not a test result.

## Design and coverage

- External context uses rooted descendant-taxon keys, exact dimension checks,
  finite values, complete branch coverage and an input hash. Independence is
  explicitly user-declared rather than inferred from file format.
- Topology determines clade blocks, with ancestor connectors excluded from training
  and scoring. Branch order and zero-event responses cannot change the fold assignment.
- Each outer block receives parameters selected using only other blocks and a
  site prior estimated from those same training rows. Auto clip/alpha are inner-fold
  choices. Depth bins preserve equal depths and require sufficient clades.
- Beta candidates are nonnegative with exact zero; ties choose the smallest beta.
  The former negative-beta/null alias and all-data clipping quantile are removed.
- The same probabilities enter the actual urn expectations; full-data ASRV weights
  cannot reintroduce the held-out response after fitting. Unselected channels retain
  ordinary ASRV. JSON records each fold and its tuned beta=0 comparator.
- Active weighting is restricted to urn/ASRV sn. Custom training/concentration,
  generic ASRV diagnostics and omega P values are rejected with specific reasons.
  The dedicated report is `csubst_epistasis.json` (prefix follows normal output rules).

## Independent simulation

Run from the repository root:

```bash
python tools/validate_epistasis.py --replicates 1000 --seed 20260910 \
  --scenarios iid,binary_null,binary_epistatic,codon_null,codon_epistatic,codon_convergence,codon_epistatic_convergence \
  --output reports/scientific_review_20260910/id3_validation.json
```

[All 7,000 replicate results](id3_validation.json). These were run with Python
3.10.14 / NumPy 1.26.4. Score-relevant code was unchanged during the run;
later edits added diagnostic metadata only.

| Scenario | Replicates | Outer gain (nat/event) | Monte Carlo SE | Positive-beta branch fraction |
|---|---:|---:|---:|---:|
| iid | 1000 | -0.000663 | 0.000046 | 0.3556 |
| binary_null | 1000 | -0.035929 | 0.002530 | 0.4635 |
| binary_epistatic | 1000 | -0.042553 | 0.004874 | 0.4662 |
| codon_null | 1000 | -0.004293 | 0.000441 | 0.4783 |
| codon_epistatic | 1000 | -0.005271 | 0.000446 | 0.4537 |
| codon_convergence | 1000 | -0.004308 | 0.000374 | 0.4635 |
| codon_epistatic_convergence | 1000 | -0.005845 | 0.000475 | 0.4240 |

Gain compares with a matched, independently tuned beta=0 prior on the same outer
clades. All scenario averages are negative. Selection fractions are **not FPRs**.
The 1,000-replicate Monte Carlo errors quantify simulation precision, not
uncertainty for a particular observed phylogeny.

The IID null has 500 branches, 20 sites, one uniform event per branch and independent
random context. Other experiments use six sites on a fixed four-clade phylogeny,
heterogeneous site mutation rates and site fields. Pair coupling is either zero
or 0.7 on a graph specified before evolution. Codon simulations use 61 sense codons
and single-nucleotide moves. Convergence is a separately imposed field shift on
two fixed clades. No observed convergence labels determine structure features.
The root is uniform; no equilibrium root is assumed. Context is the true parent
state mean before each branch evolves, an oracle unavailable in ordinary analyses.

The generator is independently checked against full fitness differences and
detailed balance in a small exact binary state space. A 10,000-trajectory
Gillespie experiment is compared with the matrix exponential. Codon tests check
all legal moves, fitness differences, and unchanged synonymous proposal rates.

## Integration and limits

A real bundled PGK search completed with both N and S weighting and independently
seeded random feature/context tables (seed 50). Both channels selected beta=0
in every outer fold. The run produced finite search output, per-fold diagnostics,
and ASRV provenance marking clade-excluded training. It is an interface smoke,
not evidence that random context is biologically useful.

The count-layer invariance tests change held-out responses, including to zero,
and assert unchanged predictions and selected parameters for those branches.
They cover joint tuning, multiple features, masks, depth bins, row permutations,
zero context, insufficient clades, and N/S/NS application.

Full-tip ASR posteriors, learned site features, and masks are conditioned on;
ASR is not recomputed within folds. Full-pipeline predictive validation remains
a separate scientific requirement. The included history simulator and PGK smoke
must not be described as establishing that independence.

IDs 2 and 4 still govern smoothed omega statistics and joint category nulls.
This change does not supply their calibration or refitting machinery, so omega
P values with active weighting remain unavailable. ID 1 fitted native 3Di/codon
model routing is preserved; no published Q.3Di matrix or new model expectation
is introduced.

## Documentation

See [the input contract and examples](../../docs/STRUCTURE_WEIGHTING.md).
[Wiki changes](id3_wiki.patch) were prepared and validated in a separate local
clone; they are not published. Implementation and validation initially ran in an isolated worktree. The user
subsequently requested integration and a local commit in the main checkout.
The changes preserve the intervening `feab6e4` pseudocount/null fixes. Wiki
publication is not included in this local commit. External-data integration and
further biological validation are outside the agreed scope; the research
[residue-pair oracle](../pair_epistasis_oracle_20260910/README.md) is retained
with its explicit limitations.

## References

- [Roberts et al. (2017)](https://doi.org/10.1111/ecog.02881): cross-validation for structured data.
- [Patel et al. (2022)](https://pubmed.ncbi.nlm.nih.gov/35575390/): separating site fields and pair interactions in sequence simulations.

## Verification

- Python 3.12.11 full suite: **1,764 passed, 6 skipped** (1,760 non-process + 4 process).
  Skips require optional Torch (4) or gemmi (2), absent in the temporary environment.
- Strict native lane: **7 passed**; all six native extensions built successfully.
- Targeted prediction/CLI/simulator tests: **69 passed** before the final added
  edge cases; all final edge cases are included in the full suite above.
- Lint, repository hygiene, typing and documentation checks passed, including
  the separate Wiki clone (45 documents checked).
- Modern NumPy typing required explicit array annotations for two unchanged
  calibration counters in `omega_calibration.py`; no numerical behavior changed.
  Its affected calibration/long-tail tests: **60 passed**.
- The first Python 3.12 run used ETE3 because a temporary ETE4 build contained
  duplicate macOS rpaths. Rebuilding ETE4 with clean linker flags restored ETE4;
  the full suite above then passed. No product fallback or test relaxation was used.
- Source distribution and native wheel built successfully; Twine checks passed.
  The wheel contains `csubst/epistasis.py`.

## Main-checkout integration

Integrated onto `feab6e4` on the existing `master` branch, preserving its
pseudocount and joint-null changes. The integrated Python 3.12 suite passed
**1,839 tests, with 6 optional-dependency skips**; the strict native suite
passed **7 tests**. The first attempt encountered pre-existing NumPy-1-built
local extensions under the NumPy-2 validation environment. Testing used the
already validated extensions with identical Cython sources; original local
binaries were restored after checks. This required no source
fallback or dependency pin. Four counter variables in the newly integrated
`omega_null.py` received explicit array/dictionary annotations for NumPy-2
typing compatibility; their targeted tests passed (70 tests including the
pair oracle). Lint, configured type checks and documentation checks passed.
