# Choosing a 3Di predictor

`--nonsyn_recode 3di20` converts amino-acid sequences to the 20-state Foldseek
3Di alphabet. Choose the predictor with `--sa_backend`:

| Backend | Method | Intended use |
| --- | --- | --- |
| `prostt5` | ProstT5 encoder-decoder, sequential generation | Preserve the existing prediction method |
| `prostt5-cnn` | ProstT5 encoder plus the upstream two-layer CNN | Evaluate encoder-only inference on CPU or GPU |
| `esm3di-35m` (default) | ESM2-35M with trained ESM3Di LoRA adapters and classifier | Evaluate a smaller predictor, especially on CPU |

These are different predictors, not interchangeable numerical accelerators.
Their 3Di sequences, ancestral reconstructions and convergence statistics can
differ. ESM3Di-35M's released checkpoint was trained on the viral BFVD dataset;
its accuracy on other protein families and on reconstructed ancestors needs
evaluation. ESM3Di-35M is the default; select `--sa_backend prostt5` to reproduce
the previous prediction method. ESM++ and Foldseek-based inference are
not implemented as CSUBST backends.

## Installation and preparation

Install the optional inference dependencies:

```bash
python -m pip install "csubst[3di] @ git+https://github.com/kfuku52/csubst"
csubst download --resource prostt5-cnn
csubst download --resource esm3di-35m
```

`prostt5-cnn` prepares both the ProstT5 model and its CNN head. ESM3Di-35M
includes its complete trained backbone in the checkpoint; CSUBST downloads
only the configuration and tokenizer from the base ESM2 repository, not a
second set of backbone weights. PyTorch 2.6 or newer is required to load the
encoder predictor checkpoints with `weights_only=True`.

ESM3Di files and the CNN head live under
`$CSUBST_CACHE_DIR/models/3di/<backend>/v1` (default cache root
`~/.cache/csubst`). They are pinned and SHA-256 verified, including on offline
loads. ProstT5 encoder/decoder weights retain the existing Hugging Face cache
or `--prostt5_local_dir` location and local-load validation contract.

```bash
csubst download --resource esm3di-35m --no_download yes
csubst download --resource prostt5-cnn --no_download yes
```

Copy both cache locations to an offline host if using ProstT5-CNN. The
`--prostt5_model`, `--prostt5_revision` and `--prostt5_local_dir` options apply
to both ProstT5 backends, not ESM3Di. A custom CNN encoder must be compatible
with the published ProstT5 CNN (1024-dimensional embeddings and the same
training representation); merely matching the embedding dimension is not
evidence of prediction quality.

## CPU example

Use a codon alignment and matching rooted tree. Supply a full CDS alignment
when the analysis alignment has been trimmed:

```bash
csubst inspect --alignment_file alignment.fa --rooted_tree_file tree.nwk \
  --full_cds_alignment_file full-cds.fa --nonsyn_recode 3di20 \
  --sa_asr_mode translate --sa_backend esm3di-35m --sa_device cpu \
  --blas_threads 4 --sa_batch_size 4 --sa_no_download yes
```

The same predictor options are available in `search`, `sites`, and other
commands using the shared recoding options. They work with both ancestral
reconstruction modes: `direct` predicts tip 3Di sequences before IQ-TREE
ancestral reconstruction, whereas `translate` predicts 3Di from reconstructed
amino-acid sequences. PCA's optional 3Di feature also uses the selected backend.
IQ-TREE can omit morphological states absent from its input alignment. CSUBST
restores those states as zero-probability columns in the 20-state tensor and
rejects missing columns for states that were present in the input.

`--sa_device`, `--sa_no_download`, `--sa_cache`, and `--sa_cache_file` are
backend-neutral aliases of the existing `--prostt5_device`,
`--prostt5_no_download`, `--prostt5_cache`, and `--prostt5_cache_file` options.
The legacy sequence cache filename `csubst_prostt5_cache.tsv` is retained.
Sequence and derived 3Di-state caches distinguish the backend and model
weights, so a cached prediction from another model is never reused.
State caches also validate the 20-state order and posterior probabilities.
An invalid cache is recomputed in `--sa_state_cache auto` mode and rejected in
`--sa_state_cache yes` mode; missing/unloaded rows may contain all zeros.

## Batching and sequence limits

`--sa_batch_size 0` selects automatic batching. The encoder backends sort
uncached unique sequences by length, cap batches at four sequences on CPU
(16 on accelerators), and additionally limit padded batches to 4096 tokens.
A sequence longer than that budget is processed individually. An OOM retries
with a smaller batch; failure for one sequence is reported without truncation.
Set `--sa_batch_size 1` to disable batching, or a positive value to change the
sequence-count cap. This cap is independent of `--blas_threads`, which controls
native-library CPU threads. Existing `prostt5` automatic batching is retained;
an explicit `--sa_batch_size` overrides its sequence-count cap while preserving
its equal-length generation groups.

ProstT5-CNN retains a zero-masked end-of-sequence slot for the CNN and removes
padding before applying it. This prevents the CNN's biased convolutions from
changing short-sequence terminal predictions depending on other batch members.
Both encoder backends use float32 inference. Small floating-point differences
near an argmax tie can still occur across hardware or numerical libraries.

ESM3Di-35M uses rotary positions and has no fixed 1022-residue input cap.
Full-length inference was verified through 8192 residues on CPU and Apple MPS,
including natural human dystrophin (3685 residues). Longer inputs require more
compute and memory; 8192 is a tested length, not a model limit. CSUBST does not
truncate sequences or splice independently predicted windows. Execution checks
do not establish prediction accuracy for long proteins. See the
[long-sequence validation](../reports/3di_long_sequences_20260909/README.md).

## Validation and benchmarking

Use [the CPU benchmark script](../tools/benchmark_3di_backends.py) after preparing
the models. It records model loading separately from repeated inference,
disables sequence caches, fixes thread counts and records predictions and
peak process RSS. The [measured CPU comparison](../reports/3di_cpu_backends_20260909/README.md)
includes predictions, timings, memory and model-agreement values.
The [default-backend functional audit](../reports/3di_default_audit_20260909/README.md)
records boundary, device, compatibility, cache and full CLI checks.
See the tool's `--help` for custom FASTA inputs. Compare predicted
3Di against structure-derived 3Di and check downstream convergence results
before treating a model switch as scientifically equivalent. Agreement with
the original ProstT5 predictions alone is not an accuracy measurement.

## Sources

- [ProstT5 paper](https://doi.org/10.1093/nargab/lqae150) and
  [encoder-CNN reference implementation](https://github.com/mheinzinger/ProstT5/blob/3f6c0666ac61d1025ce9473e34d3f67fc893a589/scripts/predict_3Di_encoderOnly.py).
- [ESM3Di implementation](https://github.com/DessimozLab/ESM3di) and
  [ESM2-35M checkpoint](https://huggingface.co/cactuskid13/esm2small_3di/tree/2227cb08ffa533bc1a7f2b968fd6316f287134d1).
- Resource revisions and expected checksums are recorded in
  [structural_prediction.py](../csubst/structural_prediction.py).

The [additional robustness audit](../reports/3di_robustness_audit_20260909/README.md)
covers low-complexity inputs, unknown residues and cache handling. A direct
ancestral-state file with only a header is rejected; it cannot stand in for
missing ancestral predictions. Sequence cache paths expand `~` consistently
when reading and writing.
Exported alignments retain existing node names. Unnamed nodes receive
`csubst_branch_<branch_id>` (with a numeric suffix if that name already exists),
so FASTA identifiers are nonempty. This applies to codon, amino-acid and 3Di
alignment exports, including the fast inspect exporter; the root remains
excluded from these alignment files.
Both encoder predictors also passed [CUDA verification on audrey1](../reports/3di_cuda_audrey1_20260909/README.md)
(RTX 6000 Ada, PyTorch 2.6/CUDA 12.4), including CPU/CUDA prediction agreement
on the tested inputs and long-input checks for ESM3Di.
