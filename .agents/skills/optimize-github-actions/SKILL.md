---
name: optimize-github-actions
description: Review or modify GitHub Actions workflows while preserving required coverage, reducing unnecessary runner use, and protecting self-hosted runners. Use when changing files under .github/workflows or optimizing CI behavior.
---

# Optimize GitHub Actions

Before editing a workflow, identify its event matrix, required checks, permissions, runner requirements, and any downstream workflow or release dependency. Preserve the coverage the repository relies on while removing work that does not contribute to that coverage.

Avoid duplicate push and pull-request runs for the same change. Cancel superseded validation when it is safe, combine short jobs when isolation is unnecessary, and retain artifacts only as long as their consumer needs them. Use platform-specific runners only for coverage that requires that platform, and gate expensive jobs by relevant paths, schedules, releases, or manual dispatch where this preserves required validation.

Never execute untrusted pull-request code on a self-hosted runner. Keep permissions and credentials scoped to the steps that need them, and preserve existing write gates unless the user explicitly requests a security or release-policy change.

Validate both workflow syntax and event behavior. Exercise important conditions or inspect the parsed workflow rather than relying only on text matching. Report the coverage retained, runner work removed or gated, checks run, and any event that still requires manual verification.
