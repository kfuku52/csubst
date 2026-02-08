---
name: csubst-validation-parity
description: Reproducible CSUBST validation for output parity, runtime, and peak RAM. Use when benchmarking speedups, checking regressions after refactors, validating branch-to-branch consistency, or comparing selective and full loading paths.
---

# CSUBST Validation Parity

## Overview

Run CSUBST validation in isolated temp directories, compare outputs deterministically, and report runtime and peak RAM with reproducible evidence.

## Workflow

1. Create an isolated benchmark directory under `/tmp`.
2. Copy only required inputs into that directory.
3. Prefer precomputed IQ-TREE outputs for routine benchmark runs unless the user explicitly asks for fresh IQ-TREE execution.
4. Execute CSUBST from inside the temp directory to avoid polluting the repository root with generated files.
5. Measure runtime and peak RAM with `/usr/bin/time -l -p`.
6. If peak RAM cannot be collected because of sandbox restrictions, rerun with escalation.
7. Compare deterministic outputs with checksums first.
8. Compare table-level parity with `scripts/compare_tsv_parity.py` and explicitly ignore known runtime columns such as `elapsed_sec`.
9. Report exact file paths, hashes, and any column-level differences.

## Frequent Failure Points

- Writing outputs in the repository root by mistake:
  - Always `cd` into the temp run directory before invoking `csubst/csubst`.
- Reporting runtime without peak RAM:
  - Use `/usr/bin/time -l -p` and include both elapsed and peak memory footprint.
- Calling a run "identical" while ignoring hidden table drift:
  - Hash key outputs and run table-level parity checks.
- Comparing runs that use different intermediates:
  - Keep IQ-TREE intermediates identical unless the goal is explicitly to test fresh inference.

## Resources

- Read `references/validation-checklist.md` for command templates and reporting format.
- Use `scripts/compare_tsv_parity.py` for numeric and categorical TSV parity checks.
