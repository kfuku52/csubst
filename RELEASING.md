# Releasing CSUBST

Keep `csubst/__init__.py` on the next semantic version for every change merged
to `master`.

The `Pytest` workflow validates each push. After it succeeds, the release
workflow checks the version from the exact tested commit:

- Versions whose patch component is nonzero (for example, `1.14.5`) remain
  available from `master`, but do not receive a Git tag or GitHub Release.
- Major and minor versions whose patch component is zero (for example,
  `1.15.0` or `2.0.0`) receive an annotated `v<version>` tag and a GitHub
  Release automatically.

Bioconda discovers tagged upstream releases. Consequently, its `csubst`
recipe is updated for major and minor releases only; patch-only versions are
intentionally not autobumped.

Do not create release tags manually unless recovering the automated workflow.
If recovery is necessary, point the annotated tag at the commit that passed
`Pytest` and preserve the existing `v<version>` tag format.
