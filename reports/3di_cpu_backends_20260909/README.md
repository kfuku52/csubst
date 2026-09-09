# CPU AA-to-3Di backend comparison (2026-09-09)

On the two full-length proteins tested, encoder-only classification reduced
inference time substantially. This is a **model comparison**, not a numerically
equivalent optimization: changing the predictor changes the predicted 3Di.
The initial implementation retained `prostt5` as its default. A subsequent
requested change selects `esm3di-35m` by default; the measurements below use
explicit backend selections and remain a record of the measured versions.

| Backend | Median inference | Range, 3 runs | Relative speed | Peak process RSS |
| --- | ---: | ---: | ---: | ---: |
| Original `prostt5` | 346.250 s | 302.118–348.597 s | 1× | 13.059 GiB |
| `prostt5-cnn` | 5.739 s | 5.673–5.967 s | 60.3× | 5.699 GiB |
| `esm3di-35m` | 0.365 s | 0.364–0.367 s | 949.3× | 0.815 GiB |

Loading the already cached models took 1.515, 1.124, and 1.791 seconds,
respectively. Loading, downloading, warmup, and validation are excluded from
inference timing. The very short ESM inference time does not describe total
CLI latency, which also includes imports, model loading, and downstream work.
RSS is the lifetime process high-water mark, including model loading and
warmup; it is not parameter size or an incremental allocation measurement.
The old generator had a 13% range relative to its slowest run, so the relative
speeds should be treated as approximate workload-specific measurements.

## Workload and environment

- Native Apple M2 Max, 12 CPU cores, 64 GiB RAM; macOS 26.6.2 arm64.
- Python 3.12.14, PyTorch 2.14.0, Transformers 5.16.1, PEFT 0.20.0.
- CPU only, float32, four PyTorch threads, automatic batching,
  sequence prediction caches disabled. No other benchmark/test/build workloads
  ran concurrently with the reported timed inference runs.
- The original harness set `torch.set_num_threads(4)` and inherited unset
  BLAS/OpenMP environment variables. The reusable runner additionally sets
  OpenMP, OpenBLAS, MKL, Accelerate and NumExpr environment caps to four.
  Neural inference uses PyTorch in both cases; the additional environment
  controls are recorded here rather than implying identical harnesses.
- [Input FASTA](input.fa): human PGK (418 residues) and maize PEPC (960 residues),
  1,378 residues total. These are translations of the first record in each
  bundled `PGK.untrimmed_cds.fa` and `PEPC.untrimmed_cds.fa`; gaps are removed
  and nonstandard residues are converted to X, as in CSUBST inference.
- One model per isolated process, a 32-residue warmup, then three sequential
  repetitions on both full-length sequences. Files were already cached locally.
- Original implementation: commit `286ac2b7edcf83ca63e2ae8c1a0eba0d8ab37c24`.
  New implementation: the uncommitted change containing this report. Source
  SHA-256 hashes and pinned model identities are included in the JSON results.

## Reproduction

Install `.[3di]`, prepare the resources, then run from the repository root:

```bash
csubst download --resource prostt5-cnn
csubst download --resource esm3di-35m
python tools/benchmark_3di_backends.py \
  --input-fasta reports/3di_cpu_backends_20260909/input.fa \
  --output-dir reports/generated/3di-cpu-backends --threads 4 --repeats 3
```

The default FASTA selection in the tool uses the same two bundled records.
The original generator was measured from a copy made before edits; a separate
run through the then-default ProstT5 dispatcher confirmed exactly the same outputs
as all three original runs. Its extra timing was used only for validation and
is not included in this comparison. Individual measurements and predictions
are in [prostt5.json](prostt5.json), [prostt5-cnn.json](prostt5-cnn.json), and
[esm3di-35m.json](esm3di-35m.json); [summary.tsv](summary.tsv) is machine-readable.

## Output differences and validation

All backends returned valid 20-state symbols and the exact input lengths.
Repeated predictions were identical. Both new backends also produced identical
predictions with single-sequence and automatic mixed-length batches. The CNN
uses the upstream single-sequence mask convention, including a zero-masked EOS
slot; its head runs separately on each unpadded sequence to remove dependence
on the padding lengths of other batch members.
The real CNN checkpoint also matched an independent calculation using the
upstream CNN class and masking steps; the real ESM3Di checkpoint matched its
unmerged LoRA reference at all 1,378 residues.

Agreement with the old generator was **73.0% for ProstT5-CNN** and **38.3% for
ESM3Di-35M** on these inputs. See [model_agreement.json](model_agreement.json).
These are agreement values, **not accuracy against known structures**. The
small ESM3Di checkpoint was trained on viral BFVD sequences; these nonviral
proteins and reconstructed ancestors require independent accuracy assessment.
No downstream scientific equivalence or universal speedup is claimed.

The implementation supports both direct and translated ancestral-state modes,
with model-specific caches and offline resource verification. A subsequently
identified, unnecessary 1022-residue guard was removed after
[actual long-sequence validation](../3di_long_sequences_20260909/README.md).
There is no silent truncation or windowing.
See the [predictor guide](../../docs/STRUCTURAL_ALPHABET.md) for provenance,
installation, model limits, and CLI examples.

[Validation results](validation.json): 1,498 non-process and four process tests
passed, as did seven strict native-path tests, lint, type checks, repository
and Wiki documentation checks, arm64 wheel/sdist builds and Twine checks.
Both new backends completed `inspect` in `translate` and `direct` modes on
`data/GH19_chitinase_tiny` (eight tips, smoke selection of two non-root branches
plus the root); their output manifests and 3Di alignments were verified. The
smoke runs exposed and fixed a pre-existing importer assumption that IQ-TREE
always writes all 20 columns: absent input states are now restored as zeros,
while missing probability columns for observed states still raise an error.
GPU execution, other dependency/Python versions, and structure-based accuracy
were not tested in this local run.
