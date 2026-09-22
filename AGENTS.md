<!-- BEGIN KF AGENT POLICY: source=https://github.com/kfuku52/kf-agent-policy; version=10; sha256=82e3c0eb467582a414d9a6b2feaaaf6f5c8ae330d30f2e3efbf8c303155d0e2e -->
# Common agent policy

Repository-specific instructions override these defaults.

- Follow the user's task scope within higher-priority instructions and execution
  permissions. Complete implementation through affected verification and a result
  report; a plan or investigation ends with its requested deliverable. Continue
  authorized work without repeated approval; identify actual blocking boundaries.
- Inspect the worktree and preserve unrelated changes. Refresh remote information
  when needed; do not merge, rebase, or switch branches merely to inspect it.
- Prefer the default branch when starting work without an established branch.
  Preserve an existing task branch; follow explicit user branch instructions.
  Never create or switch branches solely for a commit, push, release, or PR.
- Change or recommend branch protection only when explicitly asked. Honor explicit
  repository-specific direct-push exceptions; otherwise report a rejected push
  without bypassing protection or inventing a branch or PR.
- Unpublished implementation details may be redesigned; preserve existing public
  APIs, file formats, and saved-data compatibility unless a breaking change is
  authorized. Update affected producers, consumers, tests, examples, and docs.
- Fix verified root causes; do not hide failures with fallbacks or weaker checks.
  Document unavoidable workarounds and their removal conditions.
- Read relevant docs and run the repository's check entrypoint for the change and
  phase. Verify affected behavior; report checks run and omitted. Repeat or broaden
  successful checks only for new changes, failures, or unresolved concerns.
- For library metadata, require demonstrated incompatibility for exact pins or
  upper bounds; keep reproducibility locks separate.
- When editing READMEs, keep them concise with useful visuals inline; put extended
  guides in linked documentation.
- For GitHub push/release work, use `prepare-github-push` in `.agents/skills/`.
  Local-only commits need no version bump; GitHub pushes require one.
- For software performance work, use `benchmark-performance` in `.agents/skills/`.
  Performance claims require comparable measurements and equivalent output.
- For GitHub Actions edits, use `optimize-github-actions` in `.agents/skills/`.
  Preserve required coverage; never run untrusted PR code on self-hosted runners.
<!-- END KF AGENT POLICY -->

# CSUBST working guide

- Start with `git status --short`, [CONTRIBUTING.md](CONTRIBUTING.md) for setup,
  and [TESTING.md](TESTING.md#choosing-checks) for checks selected by change.
  Read [README.md](README.md) for user-facing behavior and
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before changing module boundaries.
- CLI parsing/dispatch starts in `csubst/cli.py`; command orchestration is in
  `main_*`. Parameter validation starts in `param.py`; numerical work is in
  `substitution*`, `omega*`, and the `scan_*` modules. Follow the architecture
  guide for context, I/O, and accelerator contracts.
- Run commands from the repository root in the selected development environment:
  `python -m pip install -e '.[dev]'`, `python -m csubst --help`,
  `make test-fast`, `make lint`, `make typecheck`. Use `make PYTHON=...` when
  needed. Full, native, fallback, and artifact lanes are in TESTING.md; a fast
  pass alone does not cover integration or numerical parity.
- Use `.agents/skills/verify-csubst-change/SKILL.md` when selecting and reporting
  verification for a code change. Use the existing push skill and
  [RELEASING.md](RELEASING.md) for publication.
- Preserve scientific defaults and reference values unless their change is
  explicitly intended. Joint endpoint posteriors and legacy marginal estimates
  are different estimators; consult [ENDPOINT_POSTERIORS.md](docs/ENDPOINT_POSTERIORS.md)
  and [SCAN_CTMC.md](docs/SCAN_CTMC.md) for supported models and rate assumptions.
  Read the relevant method document before changing null models, filtering,
  calibration, seeds, or thresholds. Do not relax tolerances or regenerate
  reference data merely to make a failing test pass.
- Preserve CLI aliases, output schemas/names, cache compatibility, and failure
  behavior; see [CLI_SAFETY.md](docs/CLI_SAFETY.md). Check input protection and
  manifest finalization when touching output code.
- Keep analysis runs in fresh temporary directories. Do not edit bundled
  `csubst/dataset/` fixtures, substitution matrices, vendored sources, or
  `.github/performance_baseline.tsv` as incidental cleanup. Generated C/binaries,
  `build/`, `dist/`, caches, and local environments are not source edits.
  Preserve existing research reports; new artifacts follow [reports/README.md](reports/README.md).
- Finish by reviewing `git diff --check` and the diff. Report changed behavior,
  exact checks and outcomes (including skips), and checks not run with reasons.
  Do not describe source tests as installed-wheel or scientific calibration proof.
