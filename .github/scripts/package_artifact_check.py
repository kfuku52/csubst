#!/usr/bin/env python3
import email
import glob
import os
import tarfile
import zipfile


CYTHON_MODULES = (
    "combination_cy",
    "omega_cy",
    "parser_iqtree_cy",
    "recoding_cy",
    "substitution_cy",
    "substitution_sparse_cy",
)


def _one(pattern):
    paths = glob.glob(pattern)
    if len(paths) != 1:
        raise RuntimeError("Expected exactly one artifact for {!r}; found {}.".format(pattern, paths))
    return paths[0]


def _require_suffixes(names, suffixes, artifact):
    missing = [suffix for suffix in suffixes if not any(name.endswith(suffix) for name in names)]
    if missing:
        raise RuntimeError("{} is missing required files: {}".format(artifact, ", ".join(missing)))


sdist_path = _one(os.environ.get("CSUBST_SDIST_GLOB", "dist/csubst-*.tar.gz"))
wheel_path = _one(os.environ.get("CSUBST_WHEEL_GLOB", "dist/from-sdist/csubst-*.whl"))
required_sources = ["/csubst/{}.pyx".format(module) for module in CYTHON_MODULES]
required_notices = [
    "/LICENSE",
    "/THIRD_PARTY_NOTICES.md",
    "/licenses/BIOPYTHON_LICENSE.rst",
    "/csubst/_vendor/pyvolve/LICENSE.txt",
]

with tarfile.open(sdist_path, "r:gz") as archive:
    _require_suffixes(archive.getnames(), required_sources + required_notices, sdist_path)

with zipfile.ZipFile(wheel_path) as archive:
    names = archive.namelist()
    _require_suffixes(
        names,
        ["/THIRD_PARTY_NOTICES.md", "/BIOPYTHON_LICENSE.rst", "/csubst/_vendor/pyvolve/LICENSE.txt"],
        wheel_path,
    )
    metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
    if len(metadata_names) != 1:
        raise RuntimeError("Expected exactly one METADATA file in {}.".format(wheel_path))
    metadata = email.message_from_bytes(archive.read(metadata_names[0]))

requirements = metadata.get_all("Requires-Dist", [])
normalized_requirements = [requirement.lower().replace("_", "-") for requirement in requirements]
for distribution in ("ete4", "numpy", "scipy", "pandas", "matplotlib", "biopython", "requests"):
    if not any(requirement.startswith(distribution) for requirement in normalized_requirements):
        raise RuntimeError("Wheel metadata is missing dependency {!r}.".format(distribution))
if not any(
    "pymol-open-source" in requirement
    and ">=3.2.0a0" in requirement
    and "<3.3" in requirement
    and "structure" in requirement
    for requirement in normalized_requirements
):
    raise RuntimeError("Wheel metadata is missing the structure extra dependency on pymol-open-source.")

print("Verified source distribution: {}".format(sdist_path))
print("Verified wheel: {}".format(wheel_path))
