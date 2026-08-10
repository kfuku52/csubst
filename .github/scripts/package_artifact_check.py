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


def _forbid_suffixes(names, suffixes, artifact):
    unexpected = [suffix for suffix in suffixes if any(name.endswith(suffix) for name in names)]
    if unexpected:
        raise RuntimeError("{} contains runtime-unnecessary files: {}".format(artifact, ", ".join(unexpected)))


sdist_path = _one(os.environ.get("CSUBST_SDIST_GLOB", "dist/csubst-*.tar.gz"))
wheel_path = _one(os.environ.get("CSUBST_WHEEL_GLOB", "dist/from-sdist/csubst-*.whl"))
required_sources = ["/csubst/{}.pyx".format(module) for module in CYTHON_MODULES]
required_notices = [
    "/LICENSE",
    "/THIRD_PARTY_NOTICES.md",
    "/licenses/BIOPYTHON_LICENSE.rst",
    "/csubst/_vendor/pyvolve/LICENSE.txt",
]
required_test_support = [
    "/pytest.ini",
    "/TESTING.md",
    "/tests/conftest.py",
    "/tools/evaluate_epistasis_simulation.py",
    "/.github/scripts/_safe_workdir.py",
]
required_typed_package_files = [
    "csubst/config_types.py",
    "csubst/expected_sparse.py",
    "csubst/py.typed",
    "csubst/tsv.py",
]

with tarfile.open(sdist_path, "r:gz") as archive:
    _require_suffixes(
        archive.getnames(),
        required_sources + required_notices + required_test_support + required_typed_package_files,
        sdist_path,
    )

with zipfile.ZipFile(wheel_path) as archive:
    names = archive.namelist()
    _require_suffixes(
        names,
        [
            "/THIRD_PARTY_NOTICES.md",
            "/BIOPYTHON_LICENSE.rst",
            "/csubst/_vendor/pyvolve/LICENSE.txt",
        ]
        + required_typed_package_files,
        wheel_path,
    )
    _forbid_suffixes(
        names,
        ["csubst/{}.c".format(module) for module in CYTHON_MODULES]
        + ["csubst/{}.pyx".format(module) for module in CYTHON_MODULES],
        wheel_path,
    )
    if "csubst/csubst" in names:
        raise RuntimeError("{} contains a duplicate package copy of the CLI script.".format(wheel_path))
    metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
    if len(metadata_names) != 1:
        raise RuntimeError("Expected exactly one METADATA file in {}.".format(wheel_path))
    metadata = email.message_from_bytes(archive.read(metadata_names[0]))

requirements = metadata.get_all("Requires-Dist", [])
normalized_requirements = [requirement.lower().replace("_", "-") for requirement in requirements]
for distribution in ("ete4", "numpy", "scipy", "pandas", "matplotlib", "defusedxml", "requests"):
    if not any(requirement.startswith(distribution) for requirement in normalized_requirements):
        raise RuntimeError("Wheel metadata is missing dependency {!r}.".format(distribution))
if any(requirement.startswith("biopython") for requirement in normalized_requirements):
    raise RuntimeError("Wheel metadata unexpectedly depends on Biopython.")
with zipfile.ZipFile(wheel_path) as archive:
    entry_points = [name for name in archive.namelist() if name.endswith('.dist-info/entry_points.txt')]
    if len(entry_points) != 1:
        raise RuntimeError("Wheel should contain exactly one console entry-point file.")
    entry_point_text = archive.read(entry_points[0]).decode('utf-8')
    if 'csubst = csubst.cli:main' not in entry_point_text:
        raise RuntimeError("Wheel metadata is missing the csubst console entry point.")
if not any(
    requirement.startswith("matplotlib") and "<3.11" in requirement
    for requirement in normalized_requirements
):
    raise RuntimeError("Wheel metadata is missing the Matplotlib <3.11 compatibility constraint.")
if not any(
    "pymol-open-source" in requirement
    and ">=3.2.0a0" in requirement
    and "<3.3" in requirement
    and "structure" in requirement
    for requirement in normalized_requirements
):
    raise RuntimeError("Wheel metadata is missing the structure extra dependency on pymol-open-source.")
if metadata.get("Requires-Python") != ">=3.10":
    raise RuntimeError("Wheel metadata has an unexpected Requires-Python value.")
if metadata.get("Description-Content-Type") != "text/markdown":
    raise RuntimeError("Wheel metadata is missing the Markdown long-description content type.")
if "Programming Language :: Python :: 3.14" not in metadata.get_all("Classifier", []):
    raise RuntimeError("Wheel metadata is missing the Python 3.14 classifier.")

print("Verified source distribution: {}".format(sdist_path))
print("Verified wheel: {}".format(wheel_path))
