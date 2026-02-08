# Issue Comment Template

## Summary

- What changed:
- Why this was needed:

## Validation

- Dataset(s):
- Command(s):
- Runtime and peak RAM:
- Parity checks:

## Evidence

- Branch and commit:
- Artifact paths:
- Hashes or numeric parity outputs:

## Follow-up

- Next work item:
- Blockers:

## Posting Pattern

1. Draft to local file:
   - `cat > /tmp/issue<id>_<topic>.md <<'EOF'`
2. Post:
   - `gh issue comment <id> --repo kfuku52/csubst --body-file /tmp/issue<id>_<topic>.md`
3. Retry on transient network errors.

## Close Issue Pattern

- Confirm fix commit is pushed.
- Include commit SHA and version bump in the close comment.
- Run:
  - `gh issue close <id> --repo kfuku52/csubst --comment 'Closing as fixed in <branch> (commit <sha>, version <x.y.z>).`
