# Research report artifacts

This directory contains compact, reviewable evidence for investigations and
performance decisions. Keep each report self-contained: include the command,
software version or commit, input dataset name, platform, and a short result
summary. Commands must use repository-relative paths or `python -m csubst`.

Do not commit caches, model downloads, virtual environments, raw temporary run
directories, or artifacts containing user-home paths. Put regenerable output in
`reports/generated/` (ignored) and link to durable external storage when a
single report file would exceed 5 MiB. Existing larger scientific fixtures
under `csubst/dataset/` are not report output and are intentionally retained.
