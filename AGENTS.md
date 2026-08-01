<!-- BEGIN KF AGENT POLICY: source=https://github.com/kfuku52/kf-agent-policy; version=1; sha256=4d221330e9579c117e14f7d1a87be2ec0e29cac2ae135f3c9b96d5a1ca0580ab -->
# Common agent policy

These rules are defaults for repositories managed by Kenji Fukushima. A later,
repository-specific instruction may override a rule only when it states the
rule ID, scope, and reason for the exception.

## Working tree and scope

- **SCOPE-001:** Preserve unrelated user changes. Do not discard, overwrite,
  reformat, stage, or commit them as part of another task.
- **SCOPE-002:** Keep changes within the requested scope. Inspect broadly when
  needed for correctness, but do not turn a focused request into unrelated
  cleanup or repository administration.

## Git workflow and default branches

- **GIT-001:** Work on the repository's default branch (`main` or `master`)
  unless the user explicitly requests a different existing branch. Do not
  create a branch merely because code will be changed.
- **GIT-002:** A request to commit, push, publish, release, or open a pull
  request does not by itself authorize creating or switching to another
  non-default branch. Preserve a dirty worktree; do not switch branches through
  unrelated changes.
- **GIT-003:** Do not add, enable, recommend, or modify branch protection rules
  or repository rulesets for `main` or `master` unless the user explicitly asks
  for that repository-setting change.
- **GIT-004:** If existing protection prevents a requested direct push, report
  the exact blocker. Do not bypass it, create a workaround branch, or open a
  pull request without user direction.

## Dependency policy

- **DEP-001:** Do not add an upper version bound or exact version pin merely as
  a precaution against an untested future release. In library package metadata,
  prefer the broadest range supported by the code and verified behavior.
- **DEP-002:** Reproducibility locks for applications, deployed environments,
  containers, and test fixtures are distinct from compatibility constraints in
  published library metadata. Keep locks when reproducibility is their stated
  purpose; do not present them as evidence that newer versions are incompatible.
- **DEP-003:** Add or retain a version constraint only with concrete evidence,
  such as a reproduced failure, an incompatible API actually used by the
  repository, an upstream compatibility declaration, or a platform resolver
  requirement. Document the evidence, scope, and removal condition for a
  temporary upper bound.
- **DEP-004:** Prefer fixing compatibility and testing the current stable
  dependency over retaining a preventive cap. Test declared minimum versions
  when they matter and the newest supported versions in CI where practical.
- **DEP-005:** Before adding or removing a direct dependency, search source,
  build configuration, plugins, runtime entry points, tests, and documentation
  for direct use. A package should remain a direct runtime dependency only when
  the project directly requires it; validate removals in a clean environment.

## Validation and reporting

- **VAL-001:** Run checks proportional to the change, including focused tests
  and the repository's standard suite when practical. State exactly what ran
  and distinguish passed checks from checks that could not be run.
- **VAL-002:** Do not claim compatibility, release readiness, or successful
  publication from static inspection alone when an executable verification is
  available.
<!-- END KF AGENT POLICY -->
