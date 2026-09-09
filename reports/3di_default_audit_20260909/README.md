# ESM3Di default and functional audit (2026-09-09)

`esm3di-35m` is now the default AA-to-3Di predictor. CLI parsing, parameter
normalization, direct API calls, cache identity, PCA routing, the workflow
benchmark shell script, README and Wiki use the same selection. The Python
constant is defined once in `csubst/recoding_config.py`. Ordinary unrecoded
runs still do not import PyTorch, Transformers or PEFT during argument
normalization. Explicit `--sa_backend prostt5` retains the old generator.

## Reproduced defect and fix

The shared 3Di state-cache reader checked dimensions and metadata but accepted
readable NPZ files containing NaN, infinity, negative/out-of-range probability
values, unnormalized rows, or a reversed state order. Six regression cases
failed before the fix because these invalid caches were accepted.

The reader now checks the canonical 20-state order, finite probabilities in
[0, 1], and row sums of one (within IQ-TREE's rounding tolerance). All-zero rows
remain valid for missing/unloaded states. Checks use bounded row chunks.
`--sa_state_cache auto` recomputes an invalid cache; `yes` reports an error
instead of reusing or silently repairing it. Tests verify both behaviors.

## Actual model and workflow checks

| Check | Result |
| --- | --- |
| Default omitted vs explicit ESM3Di | Same predictor, cache identity and 3Di alignments |
| CPU lengths 0, 1, 2, 7, 31, 418, 960, 1022 | Exact lengths, valid alphabet, finite inference |
| 1023 residues | Historical CSUBST guard rejected input; guard later removed after actual long-sequence testing (linked below) |
| Automatic batch vs singleton vs repeat | Identical predictions on tested inputs |
| Sequence cache | Reused without loading a model |
| CPU vs automatic MPS selection | MPS selected; identical predictions on all tested residues |
| PyTorch 2.6.0 / Transformers 4.57.6 / PEFT 0.18.1 | Real checkpoint loaded; all 1378 benchmark residues matched the newer environment; 29 focused tests passed |
| Fresh resource preparation | Production download command succeeded |
| Offline resources | SHA-256 verification succeeded without a download call; same-size tokenizer corruption was rejected |
| `inspect`, direct and translate | Full 15-node state tensors; implicit, explicit and cache-only 3Di alignments identical |
| `search`, direct and translate | Completed; observed substitution-count columns finite and nonnegative |
| `sites`, direct and translate | Completed on branch IDs 1,4, including default plotting/output generation |
| `scan`, direct and translate | Completed, including state-aware opportunity and 20 permutations |
| PCA 3Di feature | 190 finite values, identical with implicit and explicit ESM3Di |

The CLI dataset was `data/GH19_chitinase_tiny`: eight tips and a 15-node tree;
these runs used all branches, not the previous smoke-mode branch restriction.
IQ-TREE's bundled codon intermediates were supplied explicitly, with a locally
available IQ-TREE executable for direct 3Di ancestral reconstruction. The
main environment was native Apple M2 Max / macOS 26.6.2, Python 3.12.14,
PyTorch 2.14.0, Transformers 5.16.1 and PEFT 0.20.0. See [checks.json](checks.json)
for the case matrix, boundary results and alignment hashes.

The final installed-wheel test run passed **1,516 non-process tests and four
process tests**. Seven strict native-path tests also passed. Lint, type checks,
repository/Wiki documentation checks, sdist/arm64 wheel builds, Twine checks
and the repository package-artifact inspection passed.

## Reproduction

Prepare `csubst[3di]` and `csubst download --resource esm3di-35m`. From the
repository root, this exercises the default (no `--sa_backend` argument):

```bash
data_dir=data/GH19_chitinase_tiny
csubst inspect --alignment_file "$data_dir/alignment.fa" \
  --rooted_tree_file "$data_dir/tree.nwk" \
  --full_cds_alignment_file "$data_dir/alignment.fa" \
  --iqtree_treefile "$data_dir/alignment.fa.treefile" \
  --iqtree_state "$data_dir/alignment.fa.state" \
  --iqtree_rate "$data_dir/alignment.fa.rate" \
  --iqtree_iqtree "$data_dir/alignment.fa.iqtree" \
  --iqtree_log "$data_dir/alignment.fa.log" --iqtree_redo no \
  --nonsyn_recode 3di20 --sa_asr_mode direct --sa_device cpu \
  --sa_no_download yes --blas_threads 4 --threads 1 \
  --outdir reports/generated/esm3di-default-inspect
```

Use `--sa_asr_mode translate` for the other mode; compare with explicit
`--sa_backend esm3di-35m`. `search` accepts the same common inputs. `sites`
also needs `--branch_id`; `scan` needs a foreground specification. The repository
unit tests cover cache corruption, default selection, backend isolation,
output validation, OOM backoff and the real tiny-model checkpoint loader.

## Limits

No further functional defect was reproduced within these checks. This does
not establish biological equivalence to ProstT5 or accuracy against known
structures. The ESM3Di checkpoint was trained on viral BFVD sequences, and
arbitrary proteins or reconstructed ancestors still need scientific validation.
Correction: the 1022-residue CSUBST guard tested above was based on an incorrect
assumption, not a model limitation. It has been removed after
[full-length inference through 8192 residues](../3di_long_sequences_20260909/README.md).
CUDA hardware and other Python versions were not tested locally. The MPS/CPU
and cross-version agreement results apply to the tested inputs; floating-point
argmax ties can differ on other inputs or platforms.
