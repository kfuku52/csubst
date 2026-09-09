# audrey1 CUDA verification — 2026-09-09

**Both ESM3Di-35M and ProstT5-CNN passed actual CUDA inference checks.**
Their CPU and CUDA predictions matched at every residue in the comparison set.
No production-code changes were required by this verification.

## Environment and scheduling

Host audrey1, NVIDIA RTX 6000 Ada (49140 MiB), driver 550.127.05, Python 3.12,
PyTorch 2.6.0+cu124, Transformers 5.16.1 and PEFT 0.20.0. A separate audit
virtual environment reads the existing CUDA runtime's packages; missing
packages were installed only in the new environment. Neither existing
environments nor other jobs were modified.

Slurm job **25610** (ESM3Di) completed with exit 0:0 at 20:00:32 JST; job
**25612** (ProstT5-CNN), dependent on its success, completed with exit 0:0 at
20:03:38 JST. Both ran on September 9. The original next-day start estimate
moved earlier when the preceding GPU allocation ended. Each job requested four
CPU threads, 10 GiB host memory, one GPU allocation and a 30-minute limit.

## Results

| Check | ESM3Di-35M | ProstT5-CNN |
|---|---|---|
| Actual model parameters on CUDA | Pass | Pass |
| 35 sequences, 2817 residues: CPU versus CUDA | 0 differing residues | 0 differing residues |
| Singleton versus mixed batches | Identical | Identical |
| Repeated prediction | Identical | Identical |
| Output lengths and finite logits | Pass | Pass |
| Public API automatic CUDA selection | Pass | Pass |
| Cache hit without loading a model | Pass | Pass |
| Long CUDA input: 1023, 2048, 3685, 4096, 8192 residues | Pass; hashes match prior CPU results | Not tested here |

The comparison set covers unknown-only sequences, all 20 amino-acid
homopolymers, short inputs and seeded random sequences through 1025 residues.
The long ESM3Di probes use complete human dystrophin (3685 residues), its
prefixes, and repeated-sequence length probes. See the
[long-sequence report](../3di_long_sequences_20260909/README.md) for the source
FASTA and prior CPU results. No truncation or independent-window splicing was
used. The checkpoint loaders verify ESM3Di and CNN resource hashes; transferred
source-file hashes also match the local source recorded in [status.json](status.json).

The measured peak PyTorch CUDA allocation in these probes was 458083328 bytes
for ESM3Di and 6536511488 bytes for ProstT5-CNN. This is allocated tensor memory,
not total GPU memory usage or a universal capacity estimate. Recorded times
are diagnostics; the CPU and CUDA ESM3Di workloads include different extra
checks, so these times must not be used to calculate a speedup.

Raw records: [ESM3Di results](results-esm3di-35m.json),
[ProstT5-CNN results](results-prostt5-cnn.json),
[ESM3Di predictions](esm3di-35m-predictions.json), and
[ProstT5-CNN predictions](prostt5-cnn-predictions.json).

## Reproduction and limits

The [validation script](validate.py) expects the tested CSUBST source on
PYTHONPATH, prepared encoder resources in `cache/models/3di`, ProstT5 in
`models/ProstT5`, and the earlier report's `dystrophin.fasta` in the working
directory. Set `CSUBST_AUDIT_BACKENDS=esm3di-35m` or `prostt5-cnn` for separate
runs, `HF_HUB_OFFLINE=1`, `CSUBST_DISABLE_EXTENSIONS=1` and four BLAS threads.
Use an empty output directory/cache for a fresh public-API inference check.

This validates the two encoder prediction paths on one GPU/runtime combination.
It does not establish biological accuracy, all-GPU compatibility or numerical
identity for arbitrary inputs. The legacy ProstT5 encoder-decoder and full
CUDA-backed downstream CLI workflows were not exercised in these jobs.

## Equal-workload timing results — 2026-09-10

Job **25616** completed with exit 0:0. [benchmark.py](benchmark.py) used the
same PGK/PEPC [input FASTA](benchmark-input.fa) (418 + 960 = 1378 residues) on
CPU and CUDA, in a batch of two, with one full warmup and three timed repeats.
CPU used four threads of an AMD EPYC 7713. Model loading was excluded; CUDA was
synchronized around timing. All repetitions and CPU/CUDA predictions matched.

| Backend | CPU median (s) | CUDA median (s) | CPU/CUDA ratio |
|---|---:|---:|---:|
| esm3di-35m | 0.901930879 | 0.015422144 | 58.48× |
| prostt5-cnn | 27.861853033 | 0.317185637 | 87.84× |

[Raw results](benchmark-results.json) include every repetition, source input
hash, output hashes and separate loading observations. ESM3Di GPU timings were
0.022375485, 0.015422144 and 0.015390404 seconds; its first measured run was
slower. Three repetitions of two proteins do not establish a universal speedup.
These are inference timings, not full CLI or cold-start latency.

The ESM3Di test data, conditions and results are
[published in the Wiki](https://github.com/kfuku52/csubst/wiki/ESM3Di-CPU-GPU-benchmark)
(report version 1.0.0, Wiki commit `723ca6d`). The page explicitly identifies
the tested source as an unreleased development worktree. Existing unrelated
Wiki edits and the uncommitted main-source changes were preserved.
