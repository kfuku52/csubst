# CSUBST vendoring notes

The files in this directory were imported from Pyvolve 1.1.0, upstream commit
`6145db82049941d559c9d7ba9756525fa6419ea1`. The upstream license is preserved
in `LICENSE.txt`.

CSUBST keeps this private copy to make the simulation backend reproducible.
Local changes should be limited to package-relative integration and explicitly
documented in the commit that changes them. To update it:

1. Fetch a signed or tagged upstream release and record its immutable commit.
2. Replace the vendored Python sources while preserving `LICENSE.txt` and this
   file.
3. Run the simulation unit/integration tests and the full package-artifact test.
4. Update the release and commit in this file and `THIRD_PARTY_NOTICES.md`.
