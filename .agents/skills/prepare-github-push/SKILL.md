---
name: prepare-github-push
description: Prepare and carry out a GitHub push or release with worktree review, repository versioning, validation, and exact remote-target checks. Use when the user asks to push commits or tags, publish a release, or prepare one of those actions.
---

# Prepare a GitHub Push

Inspect the worktree, current branch, upstream, intended remote, and commits to be pushed. Preserve unrelated user changes. Update remote state when it affects the push decision without discarding local work.

Follow the repository's version scheme before a GitHub push, even when the user did not separately request a version change. Use the artifact-specific scheme when the repository has more than one; use Semantic Versioning only when no scheme exists. Keep local-only commits unversioned until a push is actually intended, and include related metadata or changelog updates only when the repository's established release process requires them.

Run focused checks and, when practical, the standard suite. Review the final diff and outgoing commits so the push contains the intended changes and no credentials, generated debris, or unrelated files.

Use the default branch unless the user explicitly requested another existing branch. Do not create or switch branches solely for the push, release, or pull request, and do not bypass or alter branch protection. If protection rejects a requested direct push, report the exact blocker.

Push only the intended commits and refs. Create or publish a tag, release, or pull request only when the user requested it. Report the destination, resulting commit and version, checks run, and any remote action that could not be completed.
