# PEPC simulation performance after the audit

The benchmark ran the full `simulate` command with expected omegaC and
true-ASR export. Both versions used the same bundled PEPC inputs, 71 tips,
8,516 scored branch pairs, seed 101, and one thread. All sites used the
background Q, so the fix to foreground assignments did not change the
experiment being timed. The baseline was the first expected-omega
implementation on CSUBST 1.17.5 (base `ed229ac`), before this audit.
Runs used Python 3.12.15, NumPy 2.5.5 and SciPy 1.18.1 on macOS arm64,
Apple M2 Max. Each measurement starts a fresh process and includes imports,
plotting, simulation, expectation calculation, and file writing.

| Site rates | Sites | Wall time before → after | Peak RSS before → after |
| --- | ---: | ---: | ---: |
| Uniform | 1,000 | 3.42 → 2.96 s | 431 → 319 MiB |
| IQ-TREE Rate column | 192 | 16.87 → 9.06 s | 705 → 657 MiB |

Values are medians of three repetitions. ASRV took 46% less wall time.
The uniform runs had more relative run-to-run variation. See the
[individual measurements](performance.tsv).

The ASRV subset had 146 distinct values for 192 site rates. The simulator
now reuses the custom model for each repeated rate and preserves the site
order passed to Pyvolve. This lets the simulator reuse exponentials for
repeated models and the expected-count calculation process each such
site group once. The expected-count calculation also shares the simulated
and null event calculation for homogeneous partitions. Profiling still
showed work for distinct rate values in both simulation and expectation.

Every matched before-and-after run had byte-identical tip FASTA, ancestral
FASTA, state, tree and rate files. The expected-omega tables matched (all
columns, including zeros and unavailable values) within relative tolerance
`1e-13` with zero absolute tolerance. The maximum relative difference over
the four expected count columns was `4.2e-15`. The custom-frequency values
agreed within `4.5e-16` absolute.

## Reproduce

Use the same Python environment for both checkouts. Copy the PEPC input
and fit files as in [the PEPC report](README.md), and run in the temporary
directory containing the copies. For each checkout, set `PYTHONPATH` to
its repository root. Give each run a fresh output directory. Use the same
command for the uniform case with `--num_simulated_site 1000 --simulate_asrv no`.

```bash
export MPLBACKEND=Agg VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/usr/bin/time -l python -m csubst simulate \
  --alignment_file PEPC.alignment.fa --rooted_tree_file PEPC.tree.nwk \
  --iqtree_treefile PEPC.alignment.fa.treefile --iqtree_iqtree PEPC.alignment.fa.iqtree \
  --iqtree_log PEPC.alignment.fa.log --iqtree_state PEPC.alignment.fa.state \
  --iqtree_rate PEPC.alignment.fa.rate --background_omega 0.2 --foreground_omega 0.2 \
  --percent_convergent_site 0 --simulate_seed 101 --export_expected_omega yes \
  --threads 1 --blas_threads 1 --num_simulated_site 192 --simulate_asrv file \
  --outdir run-asrv --output_prefix sim
```
