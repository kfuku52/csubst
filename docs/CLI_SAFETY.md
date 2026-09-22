# CLI inputs, logs, and failure status

`--log_file` is resolved inside `--outdir` when relative, or used as an absolute
path when supplied that way. Before opening or appending to it, CSUBST rejects
a collision with a file input. This includes symlink/hardlink aliases, inferred
IQ-TREE inputs, and errors encountered while parsing unrelated arguments. A
collision exits with status 2 and leaves the input unchanged. Use a separate
log filename; do not redirect the shell's stdout/stderr onto an input file,
because shell redirection happens before CSUBST can check it.

Doctor checks all report destinations before opening its log. Search reserves
its table and run-record destinations before opening the log as well. Shared
output-path and TSV/manifest writers reject input/log aliases for dynamically
resolved outputs. A rejected destination exits with status 2.

Search reruns preserve earlier tables in `.csubst_search_history/<run-id>/`
inside the output directory, including tables from arities no longer reached.
Unrelated files are left in place. `<prefix>_search_run.json` identifies the
current tables, archived paths, and `running`, `complete`, or `failed` status.
Only a `complete` run should be consumed as a finished analysis; partial tables
from a failed run remain available for diagnosis. History is retained until the
user removes it. Concurrent searches cannot write the same table namespace.

Output manifests are refreshed after the CLI log closes, including on ordinary
command failures, so recorded file sizes describe the final files. A forcibly
terminated process cannot perform this finalization.

Trait labels remain unchanged in input tables and plot labels. Filenames use
percent encoding for unsafe characters: `C4/CAM` becomes `C4%2FCAM`, while a
literal `C4%2FCAM` becomes `C4%252FCAM`. Simple existing filenames are unchanged;
the encoding can be reversed with standard URL percent decoding.

FASTA readers and IQ-TREE site-count inference share one streaming parser.
Spaces, tabs, CRLF, and wrapped sequence lines do not add biological sites;
gzip inputs use the same rules. Site-count inference stops after the first
record rather than materializing the entire alignment. Tree-to-alignment
mapping uses both complete headers and their first whitespace-delimited
identifiers. Duplicate identifiers are rejected instead of overwriting or
concatenating sequences. Exported alignments retain existing node names; unnamed
nodes use `csubst_branch_<branch_id>` with a numeric suffix on collision. Both
alignment exporters use the same rule and continue to exclude root records.

`benchmark --benchmark_keep_going yes` (the default) continues with remaining
configurations after a failed run. `no` stops at the first failure. In both
cases, the summary, per-run results, and failure logs are written before the
command exits with status 2 if any run failed. All-success benchmarks exit 0.
Automation should check the exit status and retain the summary for diagnosis.

## Input paths and bundled examples

Relative alignment, tree, foreground, and `--iqtree_outdir` paths are resolved
from the working directory, independently of `--outdir`. The default IQ-TREE
directory is `csubst_iqtree`; changing the analysis output directory does not
move the fitted inputs.

`dataset --name PGK` writes `alignment.fa.gz`, `tree.nwk`, `foreground.txt`,
and bundled IQ-TREE intermediate files with a provenance manifest. Repeating
it in the same directory refuses existing destinations unless `--force yes`
is supplied. This is different from the search history behavior above.

For analysis, inferred IQ-TREE files are reused only when all five files
(`.iqtree`, `.log`, `.rate`, `.state`, `.treefile`), provenance, and model are
compatible. Otherwise CSUBST attempts a new fit. Explicitly supplying all five
`--iqtree_*` file paths uses their reported model without requiring the
provenance manifest; `--iqtree_redo yes` requests a new fit.

`doctor` checks the IQ-TREE executable by default even when a bundled fit can
be reused. For a check of existing inputs without the executable check, use
`--check_iqtree_exe no`; this does not verify that a future refit will work.

`sites` selects branches with `--branch_id`, not `--foreground`. To select
foreground combinations from search results, use `--branch_id fg` and supply
`--cb_file csubst_search/csubst_cb_2.tsv` for the default search layout. The
`--cb_file` default is `csubst_cb_2.tsv` in the working directory.

## Model resource checks

`download --resource vesm-35m` always checks file sizes and SHA-256 hashes;
`--no_download yes` also rejects missing or corrupt local files without a
replacement download. ProstT5 checks that the tokenizer/model can load locally,
which is an availability check rather than a CSUBST SHA-256 guarantee.

The old `--verify` option is deprecated. VESM accepts it with a warning, and
`--verify no` does not disable mandatory verification. Requesting
`--verify yes` for `prostt5` or `all` exits 2 before preparing any resource.
This also applies to `prostt5-cnn`, whose encoder uses the ProstT5 loading
contract. Its CNN head and all `esm3di-35m` files are always SHA-256 checked;
`esm3di-35m` accepts the deprecated verification flag without disabling checks.
For local-only checks, omit `--verify` and use `--no_download yes`.

VESM weights use the CSUBST cache. ProstT5 weights instead use Hugging Face's
cache or `--prostt5_local_dir`; the CSUBST cache contains its download lock,
not those weights. See the [download guide](https://github.com/kfuku52/csubst/wiki/csubst-download)
for offline preparation.

The new encoder resources use the CSUBST cache under `models/3di`. Predictor
identity is included in both sequence and derived state-cache keys. The
backend-neutral `--sa_no_download`, `--sa_cache` and `--sa_cache_file` aliases
retain the original `--prostt5_*` destinations and file-collision checks.
