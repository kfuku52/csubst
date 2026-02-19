## CSUBST Agent Guide

### Project Guardrails

- Run heavy validation and benchmarking in isolated directories under `/tmp`, not in the repository root.
- Prefer precomputed IQ-TREE outputs for benchmark loops unless a fresh IQ-TREE run is explicitly requested.
- Report both runtime and peak RAM in validation summaries.
- Treat `gh` API/TLS/network errors as transient first; retry before declaring failure.
- Include reproducible evidence in issue comments: command lines, artifact paths, hashes, and commit SHA.

## Skills

A skill is a local instruction set stored in a `SKILL.md` file. Use these skills for CSUBST work.

### Available skills

- `csubst-validation-parity`: Reproducible CSUBST parity/performance validation with peak RAM tracking and deterministic TSV comparisons.  
  file: `.codex/skills/csubst-validation-parity/SKILL.md`

### How to use skills

- Trigger rules:
  - Use a skill when the user explicitly names it.
  - Use a skill when the request clearly matches its description.
  - Use multiple skills when needed, in minimal sequence.
- Execution:
  - Read `SKILL.md` first.
  - Load only referenced files needed for the current request.
  - Prefer bundled scripts over reimplementing repeated logic.
- Reporting:
  - State which skill(s) are being used and why in one short line.
  - If a skill is missing or blocked, state the issue and continue with best-effort fallback.
