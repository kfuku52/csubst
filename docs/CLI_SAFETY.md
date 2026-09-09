# CLI inputs, logs, and failure status

`--log_file` is resolved inside `--outdir` when relative, or used as an absolute
path when supplied that way. Before opening or appending to it, CSUBST rejects
a collision with a file input. This includes symlink/hardlink aliases, inferred
IQ-TREE inputs, and errors encountered while parsing unrelated arguments. A
collision exits with status 2 and leaves the input unchanged. Use a separate
log filename; do not redirect the shell's stdout/stderr onto an input file,
because shell redirection happens before CSUBST can check it.

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
