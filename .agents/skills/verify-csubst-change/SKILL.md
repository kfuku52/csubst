---
name: verify-csubst-change
description: Select and run focused CSUBST regression checks for a code change, then report coverage and gaps. Use for local change verification, not performance measurement, release publication, or scientific recalibration.
---

# Verify a CSUBST change

Inputs: the intended behavior change, changed paths/diff, and the available Python
environment. If the diff includes unrelated edits, keep the verification scope
explicit without discarding those edits.

1. Work from the repository root. Read `CONTRIBUTING.md` for environment setup
   and `TESTING.md`'s **Choosing checks** section for the maintained selection
   table. Inspect the implementation and callers with `rg`; select tests that
   assert the changed observable contract, not just similarly named files.
2. State the selected checks and what they establish. Read the relevant method
   document for numerical changes. Distinguish mocked integration checks from
   real IQ-TREE, structural prediction, and calibration runs. Do not download
   model weights or start a large benchmark to satisfy an ordinary code check.
3. Run the focused pytest selection through the chosen Python, then use the
   applicable Makefile lanes from TESTING.md. Keep nested-process tests in the
   serial lane. Test outputs belong in pytest temporary directories; manual CLI
   runs need fresh temporary directories and copied inputs.
4. Check exit status, assertions, warnings, and skips. If a dependency or binary
   is missing, identify the missing prerequisite and report that path as
   unverified. An all-skipped native run is not accelerator verification. On a
   failure, inspect the first causal error; do not suppress it, weaken assertions,
   or update scientific reference values just to get a green run.
5. Review `git diff --check` and confirm verification did not modify tracked
   datasets or introduce generated output. Report commands, interpreter version,
   pass/fail/skip counts, and remaining coverage gaps. Label source-only evidence
   separately from installed-artifact or real scientific validation.

Worked selection: for output-path protection or manifest finalization, use the
output-lifecycle row in TESTING.md, inspect the affected command integration
tests, then broaden according to the change's callers. That row exercises
failure paths and temporary-file protection without requiring external data.

Deliverable: a concise verification report tied to the change. Successful
execution plus assertions for the intended contract establish the result;
collection alone or a static review must be reported as such. Publication uses
the separate existing push skill after local verification.
