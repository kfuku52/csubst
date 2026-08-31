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
concatenating sequences.

`benchmark --benchmark_keep_going yes` (the default) continues with remaining
configurations after a failed run. `no` stops at the first failure. In both
cases, the summary, per-run results, and failure logs are written before the
command exits with status 2 if any run failed. All-success benchmarks exit 0.
Automation should check the exit status and retain the summary for diagnosis.
