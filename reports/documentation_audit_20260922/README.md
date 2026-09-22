# Documentation / implementation audit — 2026-09-22

## Baseline and scope

- Source: `master`, `82cfe7a1a1bbc435e97be833a92672f13e7a4788` (1.16.7), clean worktree.
- Wiki: `edda2e5d7d0bd97f3a4c3b1617451ae932f3615e`, clean temporary clone.
- Environment: macOS arm64, Python 3.14.7. Isolated venv, non-editable local
  source installation using `python -m pip install '.[dev]'`. pip built and
  installed a native cp314 macOS arm64 wheel. No existing environment changed.
- Prioritized installation metadata, README PGK example, Wiki installation and
  typical workflow, search/sites/scan/doctor paths, foreground selection,
  output schemas, coordinates, cache precedence, and reruns. Read parser
  definitions, processing code, and relevant tests, not just help.
- Applied AGENTS.md, CONTRIBUTING.md, TESTING.md, RELEASING.md and the push
  skill. Publication metadata advances to 1.16.8; analysis behavior is unchanged.

## A — corrected documentation

| Location | Before / issue | Evidence and correction |
| --- | --- | --- |
| Wiki `Typical-workflow.md` | Default joint scan explicitly requested `q_weighted`, which fails. | `param._normalize_state_parameters`, `substitution_scan.validate_scan_configuration`, `tests/unit/test_joint_defaults.py`, and the actual command's exit 2. Use `endpoint`, consistent with `docs/SCAN_CTMC.md`. Explain that reduced permutations only check execution. |
| README cache paragraph; Wiki `csubst-download.md` | Cache precedence omitted `XDG_CACHE_HOME`; README also omitted the CLI override. | `resource_cache.resolve_cache_dir`, explicit-path test in `tests/integration/test_resource_cache.py`, and direct isolated environment-variable checks. Document CLI, CSUBST environment variable, XDG, home fallback in order. |
| Wiki `Foreground-specification.md` | Said `sites` accepts foreground stem selection options. | `_register_sites_parser` has no foreground parent; `main_sites._read_foreground_branch_combinations` reads `branch_id_*` and `is_fg*` from `--cb_file`. Remove sites from that statement and explain `--branch_id fg` with the actual default search path. |
| README test run; `docs/CLI_SAFETY.md` | Missing local explanation of result location, fit reuse/refitting, independent IQ-TREE directory, doctor executable checks, and dataset overwrite behavior. | `runtime.ensure_output_layout` / `ensure_iqtree_layout`, `parser_misc.generate_intermediate_files`, `main_dataset._copy_dataset_files`, `main_doctor`, output-lifecycle tests and real PGK runs. Add concise result/status pointers and centralize operational details in CLI_SAFETY. |
| README installation link | Promised checked distribution versions/ranges on a guide that instead links to current availability. | Read the checked-out Wiki installation page. Describe it as availability and version-selection guidance. No assertion about today's Bioconda builds added. |
| Wiki output interpretation | Current output guide did not explain estimator changes or point readers from count columns to unavailable-observation semantics. | `docs/ENDPOINT_POSTERIORS.md`, `event_reporting`, `tsv.write_dataframe`, and actual eligibility columns. Link the existing method definition instead of duplicating or redefining it. |

No historical release description or research report was rewritten. No generated
manual was edited. Wiki edits are published in the separate Wiki repository.

## B — unresolved implementation/diagnostic suspicion

A successful isolated installation reports `CSUBST missing dependency packages:
cython` during search, sites, scan and doctor. Cython is a **build** requirement
in `pyproject.toml`, not a runtime dependency; PEP 517 supplies it to the separate
build environment. The native wheel and PGK search/sites run successfully
without installing Cython into the runtime environment.

Reproduction: create a fresh venv, install the checkout with
`python -m pip install '.[dev]'`, then execute the README PGK commands below.
`csubst/param.py:DEPENDENCY_DISTRIBUTIONS` includes `cython`, and
`get_global_parameters` labels every missing member as a missing dependency.
`tests/unit/test_param_core.py` checks this reporting mechanism but does not
separate runtime requirements from build tools. This could mislead users into
repairing a valid installation. No runtime behavior or dependency metadata was
changed to conceal the issue; correcting the diagnostic is separate work.

## C — unresolved scientific intent

No specific contradictory scientific statement was established in this bounded
review. Scientific calibration and equivalence to the 2023 marginal-estimator
results were not established by these execution checks. Existing method documents
explicitly distinguish joint and marginal estimators; their scientific defaults,
reference values, and interpretation were retained.

## Executed checks

All analysis commands ran outside the checkout in a new temporary directory,
using the isolated installed package (confirmed from its site-packages path).
The following commands use `csubst` from that venv's PATH.

```bash
csubst dataset --name PGK
csubst search --alignment_file alignment.fa.gz --rooted_tree_file tree.nwk --foreground foreground.txt
csubst sites --alignment_file alignment.fa.gz --rooted_tree_file tree.nwk --branch_id 23,51 --outdir csubst_sites --output_prefix csubst
csubst doctor --alignment_file alignment.fa.gz --rooted_tree_file tree.nwk --foreground foreground.txt
```

- Dataset and search: exact README commands, exits 0. All referenced files
  exist. Bundled compatible IQ-TREE fit reused; no new external fit.
- Search: `csubst_search/csubst_cb_2.tsv` has 1,682 rows / 45 columns, including
  `branch_id_1/2`, `OCNany2spe`, `ECNany2spe`, `omegaCany2spe`, `dNCany2spe`,
  `dSCany2spe`; branch table 65 rows / 10 columns; summary 1 row / 83 columns.
  `csubst_search_run.json` has status `complete`.
- Sites: exact Wiki workflow command, exit 0. The branch directory is
  `csubst_sites/csubst.branch_id23,51`; `csubst.tsv` has 417 rows / 41 columns,
  with `codon_site_alignment` exactly 1–417. State TSVs and plot outputs exist.
  Manifest file sizes were asserted equal to the final files.
- Doctor: exact Wiki command, exit 2: 18 pass / 1 fail because `iqtree` is absent.
  This is expected from its documented default executable check, not proof
  that a refit works. Appending `--check_iqtree_exe no` succeeds, exit 0.

Original Wiki scan, executed literally, exits 2 before analysis:

```bash
csubst scan --alignment_file alignment.fa.gz --rooted_tree_file tree.nwk --foreground foreground.txt --scan_unit_mode clade --scan_other_scope all --scan_rate_event_mode posterior_sum --scan_rate_exposure q_weighted --scan_pvalue_calibration full_scan --scan_n_permutations 1000 --threads 8
```

Corrected **reduced verification**, not the full 1,000-permutation example:

```bash
csubst scan --alignment_file alignment.fa.gz --rooted_tree_file tree.nwk --foreground foreground.txt --scan_unit_mode clade --scan_other_scope all --scan_rate_event_mode posterior_sum --scan_rate_exposure endpoint --scan_pvalue_calibration full_scan --scan_n_permutations 2 --threads 1
```

Exit 0; `csubst_scan/csubst_scan.tsv` has 12 rows / 121 columns, units table
2 rows / 9 columns. Verified `codon_site_alignment == site + 1` on this unfiltered
input. Two permutations are not evidence for meaningful calibrated p-values.

Additional checks:

- Repeated the exact search command: exit 0, earlier table archived under
  `.csubst_search_history`; SHA-256 equals the new table, current run complete.
- Repeated dataset command without force: exit 1 / `FileExistsError`, refusing
  existing files. Did not execute destructive `--force yes`.
- Ran `csubst sites --alignment_file alignment.fa.gz --rooted_tree_file tree.nwk
  --branch_id fg --cb_file csubst_search/csubst_cb_2.tsv`: exit 0.
- Asserted explicit cache directory overrides CSUBST_CACHE_DIR, which overrides
  XDG_CACHE_HOME; checked XDG appends `/csubst`. No model downloads performed.
- `make docs-check WIKI_DIR=<temporary-wiki-clone>`: passed, including Wiki.
- `make lint typecheck` with the isolated Python: passed (typecheck only covers
  Makefile targets). These are source checks, not installed-wheel test lanes.
- `python -m pytest -q tests/unit/test_joint_defaults.py
  tests/integration/test_main_dataset.py tests/cli/test_output_lifecycle.py
  tests/integration/test_resource_cache.py -m 'not process'`:
  **46 passed, 2 deselected**, no skips. Process tests were deliberately excluded.
- Final source and Wiki `git diff --check`: passed.

## Limits

The local-source pip installation is an alternative validation of packaging,
not execution of the README's Git URL install or Bioconda install. No clean
sdist rebuild/full installed-artifact suite, other Python/platform installs,
real IQ-TREE refit, optional structure/3Di/VESM model downloads, external services,
simulation or benchmark grids, full scan calibration, or full regression suite
was run. Only documentation/version metadata changed, so focused checks were
selected. Primary examples used complete PGK, not PEPC or large user datasets.

The offline checker parses commands and checks local file/Wiki-page links; it
does not validate external URLs or heading anchors. All current repository/Wiki
pages were included in this mechanical check; detailed semantic review was
limited to the scope above. Deep method derivations, every docstring, optional
configuration/profile formats and every experimental mode remain outside this
review. Historical release notes and research reports are not current promises.
