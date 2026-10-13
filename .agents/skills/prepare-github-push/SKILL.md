---
name: prepare-github-push
description: Prepare or execute a requested GitHub push or release. Use for push or release preparation, not history inspection or general planning.
---

# Prepare a GitHub Push

Inspect the worktree, current branch, upstream, intended remote, and commits to be pushed. Preserve unrelated user changes. Update remote state when it affects the push decision without discarding local work.

Use the requested destination, outgoing commit range, repository instructions, and delivery entrypoint as inputs. Record the destination branch's pre-push SHA after refreshing it, and use that fixed revision when the entrypoint needs a comparison base. Retain it for post-push verification; a remote-tracking branch may advance to the pushed commit and hide the changes from version checks.

Follow the repository's version scheme before a GitHub push, even when the user did not separately request a version change. Use the artifact-specific scheme when the repository has more than one; use Semantic Versioning only when no scheme exists. Keep local-only commits unversioned until a push is actually intended, and include related metadata or changelog updates only when the repository's established release process requires them.

Use the repository's delivery check entrypoint as the source of required checks; do not maintain a separate test list in this skill. Run checks appropriate to the change and all repository-required delivery checks. After success, expand or repeat checks only for new changes, failures, or unresolved concerns. Review the final diff and outgoing commits so the push contains the intended changes and no credentials, generated debris, or unrelated files.

If a required dependency, base revision, credential, or check is unavailable or fails, stop publication and report the failing command and missing prerequisite. Do not substitute a weaker check. The deliverable is the reviewed change plus a verification record identifying the base and checks run; distinguish executed checks, static inspection, and checks blocked by the environment.

Preserve the established task branch and follow explicit user branch instructions. Prefer the default branch only when no task branch is established; do not create or switch branches solely for publication. Follow explicit repository-specific direct-push exceptions only for their stated actor and scope. Such an exception does not authorize force pushes, protection changes, or other bypasses. Otherwise, if protection rejects a requested direct push, report the exact blocker.

Push only the intended commits and refs. Create or publish a tag, release, or pull request only when the user requested it. Report the destination, resulting commit and version, checks run, and any remote action that could not be completed.
