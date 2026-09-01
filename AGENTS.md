<!-- BEGIN KF AGENT POLICY: source=https://github.com/kfuku52/kf-agent-policy; version=8; sha256=74c7b7e2e0f64f6dbdac0aec9acab92873fbcf7011eee38fedefd261fcd43d35 -->
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
- Keep top-level READMEs concise and retain useful visuals inline. Put
  feature-specific guides and extended examples in dedicated documentation or
  the wiki, linking only as needed.
- Proactively use visuals when they improve understanding.
- Changes confined to unpushed local commits need no backward compatibility.
- Prefer verified root-cause fixes to fallbacks or relaxed validation that only
  hide failures. Document unavoidable workarounds and their removal conditions.
- When changing GitHub Actions, preserve required coverage and never execute
  untrusted pull-request code on self-hosted runners.
- Run focused checks and, when practical, the standard suite. Directly exercise
  affected behavior or rendered artifacts; report exactly what did and did not
  run.
- Performance claims require representative before-and-after measurements and
  equivalent output.
- Individual local commits need no version bump. Before GitHub pushes, bump the
  version even if unrequested, using the repository's scheme or Semantic
  Versioning (`MAJOR.MINOR.PATCH`) if absent.
<!-- END KF AGENT POLICY -->
