# CSUBST 1.17.0 publication preparation

Date: 2026-10-23. Destination: `origin/master`
(`https://github.com/kfuku52/csubst.git`). The refreshed, fixed pre-push base is
`624bb8a6dae536d0fa10891f666684ef1ad3dab0` (version 1.16.22).

The outgoing implementation commits are `c371a1b` and `6bc51fa`:
[round-three evidence](../performance_round3_20261023/README.md) and
[round-four evidence](../performance_round4_20261023/README.md). The initializer
changes only its version to 1.17.0; its remaining AST is unchanged. This minor
version uses the documented [release workflow](../../RELEASING.md): successful
push-triggered Pytest, automated annotated tag/GitHub Release, then wheel builds
for Linux x86_64, macOS x86_64 and macOS arm64. No manual tag is required.

Local artifact verification used Python 3.12.15 on macOS arm64 in an isolated
source snapshot and a new non-editable wheel environment outside the checkout.
All product Python files match the outgoing tree. Commands, versions, exit
codes, artifact SHA-256 values and the initial environment failure are retained
in [verification.json](verification.json); temporary paths are placeholders.

Completed checks:

- `make package`: sdist/wheel built; Twine passed both artifacts.
- `CSUBST_USE_CYTHON=0 python -m pip wheel --no-deps dist/csubst-*.tar.gz --wheel-dir dist/from-sdist`: generated-C rebuild passed.
- `python .github/scripts/package_artifact_check.py` and Twine: passed.
- Installed wheel: `pip check`, CLI version 1.17.0, and 15 strict native tests passed.
- Installed parallel-safe full suite: 2,249 passed, 9 skipped; serial process suite: 5 passed. Branch coverage 75.53%, exceeding the required 70%.
- Installed PGK/PEPC numerical parity: passed; omegaCany2spe values 1.910709 and 0.030734 respectively. [Parity metrics](parity.tsv) retain the complete measured summary. This local run used `--numerical-only`; Linux runtime/RSS thresholds remain a CI check.

The 9 skips require Torch (4), gemmi (2), or real IQ-TREE (3). Eight existing
Matplotlib pending-deprecation warnings appeared. Earlier source verification
also covered the complete extension-disabled lane; see the performance reports.

The first fresh environment received an invalid locally cached ete4 4.4.0
wheel: its 16,727-byte tree bundle had no `PyInit_tree` export. Rebuilding that
same version with `--no-cache-dir --no-binary=ete4` restored import and the checks
above. Logical temporary paths were also normalized to physical paths before
rerunning the directory-based test taxonomy. No dependency restriction, test
tolerance or scientific reference was changed. This local environment repair
is unnecessary when installing a valid upstream/source-built ete4 artifact.

Remote CI, release creation and all-platform wheel completion must be verified
against the resulting pushed commit. They are not claimed by this pre-push
record. Sanitizers, other Python versions/architectures and Linux performance
checks are delegated to the repository's maintained GitHub Actions lanes.
