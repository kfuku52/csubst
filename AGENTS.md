<!-- BEGIN KF AGENT POLICY: source=https://github.com/kfuku52/kf-agent-policy; version=3; sha256=361e91830b2e55ec00baa462f4915dbb6732649962859691aee58d3f94cc86a4 -->
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
- Before pushing changes to GitHub, ensure the project version is bumped even
  if the user did not request it. Use the repository's versioning scheme, or
  Semantic Versioning (`MAJOR.MINOR.PATCH`) if none is defined.
<!-- END KF AGENT POLICY -->
