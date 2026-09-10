# Remaining worktree integration review

## Decisions

- **Arity-six benchmark: include.** The changes add a reproducible workload,
  resource/context recording, and numerical parity checks for the endpoint
  implementation already in master. They do not change production inference.
  The existing report distinguishes non-equivalent estimands, background load,
  and retained/excluded measurements. Saved separately in commit b50b0c0.
- **Scan joint/bridge: include as explicit options.** Joint endpoints and
  posterior mean jump counts answer different observation questions. Independent
  enumeration/Frechet checks, comparison with the shared endpoint engine, and
  fitted-bootstrap integration provide a basis for exposing them. The held-out
  report does not demonstrate better detection and records greater runtime/RAM;
  marginal observations and existing calibration defaults therefore remain.
  Supported uniform codon models and invalid option combinations are checked
  explicitly. The finite-time endpoint exposure is also available separately.
- **Epistasis worktree: do not reapply.** Its substantive implementation and
  oracle are already integrated. Remaining code differences would undo newer
  pseudocount/joint-null handling or endpoint validation. Oracle summary values
  match after newline normalization; the script difference is CSV line endings.
  Its old completion notes and Wiki patch also predate the integration records
  and corrected master-branch link. Retain the current master versions.
- **Other two worktrees:** clean, with heads already reachable from master.

The original worktrees are preserved. A dirty old worktree does not imply that
its implementation is absent from master.

## Integration fixes

The analytical endpoint likelihood now reads the original tip emissions when
joint/bridge pruning has imputed missing or ambiguous tip posteriors. Otherwise
the same data would be passed back as inferred observations. A regression check
changes inferred states while holding saved emissions fixed and verifies that
the analytical P remains identical.

Both analytical and joint/bridge CLI/config options are preserved after merging
their overlapping additions. The analytical inference JSON remains present.
The three new inference modules are included in the routine type-check targets.
An earlier captured test log had its local home-directory prefix redacted to
satisfy repository hygiene checks.

## Validation

Tests ran in a GeneGalleon Docker runtime, Python 3.12, with the integrated
Cython extensions built from source in a copied checkout. Original host
extensions were not replaced.

- Non-process suite: 2,083 passed, three optional-dependency skips.
- Process suite: four passed.
- Strict native suite: eight passed (also covered by the broad suite).
- Related scan/CLI subset before broad checks: 266 passed.
- Six actual CLI configurations: marginal/joint/bridge, each with analytical
  inference plus a frozen profile or fitted parametric bootstrap (B=2).
  Every configuration completed; emitted P values and inference metadata were
  checked. B=2 is an integration check, not calibration evidence.
- Repository lint, hygiene, documentation checks and type checks passed.
  External Wiki links were not checked because no Wiki checkout was supplied.
- No SIF execution; Docker results do not establish SIF compatibility.

The initial broad run passed 2,081 tests and failed two because an existing
report-based fixture was missing from the copied checkout. Copying that fixture
and rerunning those two passed; no production code was changed for this issue.
The optional skips are transformers (one) and gemmi (two).

Prior scientific/performance experiments are retained in the three scan
reports. They are historical measurements, not rerun performance claims about
this merge. See [the matched fitted-bootstrap study](../scan_integration_20260910/README.md)
and [the analytical mixture study](../scan_analytic_training_20260910/REPORT.md).
