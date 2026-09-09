# ESM3Di long-sequence verification — 2026-09-09

The previous 1022-residue guard in CSUBST was based on an incorrect assumption.
Actual checkpoint inference succeeds beyond it, and the guard has been removed.
The pinned model uses rotary positions; its `max_position_embeddings=1026`
configuration value is not a fixed position-table boundary. See the upstream
[rotary implementation](https://github.com/facebookresearch/esm/blob/main/esm/rotary_embedding.py).

## Actual checkpoint results

Apple M2 Max, 64 GiB RAM, CPU with four PyTorch threads, or Apple MPS.
Current runtime: Python 3.12, PyTorch 2.14.0, Transformers 5.16.1, PEFT 0.20.0.
Compatibility runtime: PyTorch 2.6.0, Transformers 4.57.6, PEFT 0.18.1 (CPU).
All tested runs used the installed pinned ESM3Di-35M checkpoint, cache disabled,
float32 and full sequences, without truncation or windowing. All output lengths
matched input lengths, every residue's logits were finite, and predicted labels
belonged to the 20-state alphabet. All environments produced identical predictions
at shared tested lengths. A short prediction remained unchanged after long inputs.

| Residues | Input | CPU | MPS | Compatibility CPU |
|---:|---|---|---|---|
| 1022 | natural sequence prefix | Pass | Pass | Pass |
| 1023 | natural sequence prefix | Pass | Pass | Pass |
| 1024 | natural sequence prefix | Pass | Pass | Pass |
| 1025 | natural sequence prefix | Pass | Pass | Pass |
| 2048 | natural sequence prefix | Pass | Pass | Pass |
| 3685 | complete natural sequence | Pass | Pass | Pass |
| 4096 | repeated natural sequence | Pass | Pass | Pass |
| 8192 | repeated natural sequence | Pass | Pass | Not run |

The natural input is complete human dystrophin, 3685 residues, from
[UniProt P11532](https://www.uniprot.org/uniprotkb/P11532/entry), sequence version 4.
Shorter inputs are prefixes; 4096 and 8192 are repeated-sequence length probes,
not naturally occurring proteins. The source sequence and SHA256 hashes are
included. See [CPU](cpu.json), [MPS](mps.json), and [compatibility](compat-cpu.json)
records for output hashes and diagnostic times. Times are single invocations,
not a repeated performance benchmark. Process RSS on MPS does not describe total
GPU/unified memory usage.

8192 residues is a verified length, not a newly imposed maximum. The tests did
not seek an out-of-memory boundary. Required compute and memory grow with input
length and attention implementation. No structure-reference accuracy evaluation
was performed; successful execution does not validate long-sequence accuracy.

## Reproduction

Install CSUBST with its `3di` extra and prepare `esm3di-35m` resources using
`csubst download --resource esm3di-35m`. Run this directory's `verify.py` with
`--device cpu` or `--device mps` and an empty `--output-dir`. The script uses
production tokenization and prediction, including finite-logit validation.
Use `--max-length 4096` for the compatibility run.

The public default `predict_3di` API also passed mixed 128/3685/8192-residue input,
matched direct inference for both long sequences, and reproduced all outputs
from the sequence cache: [public API verification](public.json).

After removing the guard, 1518 non-process tests and four process tests passed
using the rebuilt installed arm64 wheel. Seven strict native-path tests also
passed. Focused source tests passed (87), as did lint, type checks, repository
and Wiki documentation checks, wheel/sdist verification and Twine checks.
The earlier benchmark and default-audit reports now explicitly correct their
former length-limit statements.
