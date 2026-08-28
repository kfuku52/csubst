<!-- BEGIN KF AGENT POLICY: source=https://github.com/kfuku52/kf-agent-policy; version=6; sha256=9a1279ed25a1782ca13ed3a1bebc74d389c35e34a52f8a8ec762bb7790d05186 -->
# Common agent policy

Repository-specific instructions override these defaults.

- Before edits, inspect the worktree and preserve unrelated user changes. When
  remote state matters, update from the default branch without discarding local
  work.
- Use the default branch unless the user explicitly requests another existing
  one. Never create or switch branches solely for a commit, push, release, or
  pull request.
- Change or recommend branch protection only when explicitly asked. If it
  blocks a requested direct push, report it; never bypass it or create a branch
  or pull request.
- In library metadata, exact pins or upper bounds require demonstrated
  incompatibility. Keep reproducibility locks separate; prefer fixing and
  testing compatibility.
- Interface, option, format, filename, or schema changes must update all
  producers, consumers, tests, examples, and documentation.
- Keep each repository's top-level README concise: do not add feature-specific
  guides or extended examples. Put them in dedicated documentation or the wiki,
  linking from the README only when needed for discoverability.
- Changes confined to unpushed local commits need no backward compatibility.
- Prefer verified root-cause fixes to fallbacks or relaxed validation that only
  hide failures. Document unavoidable workarounds and their removal conditions.
- When changing GitHub Actions, preserve required coverage while minimizing
  runner use: avoid duplicate push/PR runs and high-frequency polling, cancel
  superseded CI, combine short jobs when isolation is unnecessary, and retain
  artifacts only as long as needed.
- Use platform-specific runners only when their coverage is required, gate them
  by relevant paths, schedules, releases, or manual dispatch, and never execute
  untrusted pull-request code on self-hosted runners.
- Run focused checks and, when practical, the standard suite. Directly exercise
  affected behavior or rendered artifacts; report exactly what did and did not
  run.
- For performance work, benchmark representative workloads before and after,
  verify equivalent output, and report wall time and peak memory when relevant.
- Individual local commits need no version bump. Before GitHub pushes, bump the
  version even if unrequested, using the repository's scheme or Semantic
  Versioning (`MAJOR.MINOR.PATCH`) if absent.
<!-- END KF AGENT POLICY -->
