# Additional 3Di robustness audit — 2026-09-09

Three additional reproducible defects were fixed. They affected shared 3Di
infrastructure, including the newly selected default backend.

| Defect | Reproduction before fix | Corrected behavior |
|---|---|---|
| Sequence-cache reads did not expand `~`, whereas writes did | All three backend cache-hit tests failed and tried to load a model despite a valid cache | Reads and writes resolve the same home-relative path; cached prediction works without loading a model |
| A header-only direct IQ-TREE `.state` file returned tip states and zero ancestor states | A regression test expected an error but none was raised | Reject the empty ancestral output before downstream analysis |

Six failing regression cases were reproduced before their respective fixes.
The fixes do not change model predictions or valid ancestral-state inputs.
Unnamed exported nodes now have usable FASTA identifiers.

The third defect was reproduced with the bundled GH19 CLI run: an unnamed
internal node was exported as a bare `>` header. Both alignment exporters now
assign `csubst_branch_<branch_id>`, with a numeric suffix on collision, preserving
existing names and tree state. Regression tests exercise both writers, deliberate
name collisions and FASTA round trips. The alignment exports intentionally omit
the root: this eight-tip tree has 14 exported records, not 15.

## Additional actual-model checks

On an Apple M2 Max, Python 3.12, PyTorch 2.14.0, Transformers 5.16.1 and PEFT
0.20.0, both actual pinned encoder backends processed 35 sequences (2817 total
residues) on CPU with four PyTorch threads: single residues, unknown-only
sequences, all 20 amino-acid homopolymers, and seeded random sequences through
1025 residues. Production inference checked finite logits. Output lengths were
correct and singleton versus mixed-batch predictions matched for every residue.
See [inference.json](inference.json).

An additional 11-sequence set (2039 residues) combined unknown-only,
low-complexity and seeded random inputs through 1025 residues. ESM3Di predictions
were identical on CPU and Apple MPS: [device.json](device.json).

The audit also reviewed backend dispatch, lazy optional imports, checkpoint
identity and strict loading, tokenizer offsets and padding, finite-logit checks,
OOM batch reduction, model-specific sequence/state caches, resource SHA checks,
CLI aliases and direct/translate integration. Related automated tests cover
malformed caches, resource locking, independent-process sequence-cache merging,
state-table validation and missing dependencies. Earlier actual-checkpoint
[compatibility and CLI checks](../3di_default_audit_20260909/README.md) and
[long-sequence checks](../3di_long_sequences_20260909/README.md) remain applicable.

These checks test functional robustness, not structure-reference accuracy.
The released ESM3Di weights were trained on viral BFVD data. No CUDA device was
available; CUDA behavior and biological accuracy remain unverified locally.

## Final verification

The rebuilt installed arm64 wheel passed 1524 non-process tests and four
process tests (1528 total), plus seven strict native-path checks. Focused
regression/resource tests passed (77), as did the sequence utility suite (45).
Lint, type checks, source/Wiki documentation checks, source/wheel artifact
verification and Twine checks passed.

Fresh `inspect` runs on the bundled GH19_chitinase_tiny dataset completed in
both `direct` and `translate` modes with both prediction and state caches
disabled. Each exported alignment had all 14 non-root records with nonempty,
unique identifiers: [CLI checks](cli.json).

For reproducible extra model probes, install `csubst[3di]`, prepare both model
resources, and run `inference_check.py` or `device_check.py` from an empty output
directory. The scripts use fixed random seeds and write the corresponding JSON
result to that working directory. The device check requires Apple MPS.
