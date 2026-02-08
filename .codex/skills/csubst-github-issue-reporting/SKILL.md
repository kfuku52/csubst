---
name: csubst-github-issue-reporting
description: Structured GitHub issue operations for CSUBST using gh CLI. Use when creating roadmap issues, posting progress updates, sharing benchmark evidence, reporting parity results, or closing issues after verification.
---

# CSUBST GitHub Issue Reporting

## Overview

Publish high-signal issue updates with reproducible commands, artifact paths, and parity conclusions, while handling transient `gh` failures safely.

## Workflow

1. Resolve the target issue explicitly with `gh issue view <id>`.
2. Gather factual evidence before drafting:
   - branch and commit hash
   - exact command lines used
   - output file paths
   - parity hashes or metric summaries
3. Draft the comment in `/tmp/<issue>_<topic>.md`.
4. Post with `gh issue comment <id> --repo kfuku52/csubst --body-file <file>`.
5. If `gh` fails with network/TLS/API transient errors, retry the same command.
6. Close an issue only after fix commits are on remote and the comment includes commit and version.

## Frequent Failure Points

- Posting to the wrong issue:
  - Validate title and state with `gh issue view` before commenting.
- Losing structured evidence in ad-hoc comments:
  - Use a file-based template and include sections.
- Treating transient API failures as hard failures:
  - Retry when errors mention TLS handshake timeout or connection errors.
- Closing issues without traceable fix:
  - Include branch, commit SHA, version, and validation summary in the closing comment.

## Resource

- Read `references/issue-comment-template.md` and use it as the default issue update format.
