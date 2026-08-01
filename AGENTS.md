<!-- BEGIN KF AGENT POLICY: source=https://github.com/kfuku52/kf-agent-policy; version=2; sha256=9361b6d2342940b167ec77884d09c9ae337e5964b9fc94a80386fac5f1fa7e95 -->
# Common agent policy

Repository-specific instructions override these defaults.

- Work on the default branch unless the user explicitly requests another
  existing branch. Do not create or switch branches merely to commit, push,
  release, or open a pull request.
- Do not modify or recommend branch protection unless explicitly asked. If it
  blocks a requested direct push, report the blocker instead of bypassing it or
  creating a branch or pull request.
- In library metadata, use exact pins or upper bounds only for demonstrated
  incompatibility. Treat reproducibility locks separately, and prefer fixing
  and testing compatibility.
- Before adding or removing a direct dependency, confirm direct use in code,
  configuration, tests, or documentation. Validate removals in a clean
  environment.
<!-- END KF AGENT POLICY -->
