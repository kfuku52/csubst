# Proposal 04: bounded batch scores for fixed-candidate calibration

Decision: **reject**. The prototype preserves direct numerical scores in the
checked cases, but changes which invalid candidate determines a replicate's
recorded failure reason. No production modules or tests were changed.

The proposed optimization defers rate scoring until at most 256 candidates
have accumulated. Counts, exposure calculations and assignment sampling retain
their existing implementations. The batch helper combines finite checks,
`logaddexp`, `sqrt` and `log_ndtr` calls, retaining scalar `math.log` and
`math.exp` to avoid changing their established rounding. This costs up to 256
pending summary dictionaries plus O(256) numeric temporaries; peak RSS was
not measured. The accompanying patch is a rejected review artifact.

Reproduce from the repository root:

```sh
CSUBST_PERF_BASELINE='<BASELINE_624bb8a>' python reports/performance_round3_20261023/proposal04.py
```

The baseline snapshot must contain the package and bundled datasets. The
session alternatively resolves it through `/tmp/csubst-round3-baseline-path`.
Baseline is `624bb8a6dae536d0fa10891f666684ef1ad3dab0`. Environment: macOS
27.0.1 ARM64, 12 logical CPUs, Python 3.14.7, NumPy 2.5.3, SciPy 1.17.1,
pandas 3.0.5. Numerical-library thread variables are set to one before imports.
Source SHA256 hashes and all correctness results are in `proposal04.json`.

The direct comparison passes all 25,754 score/P cases exactly, with no
warnings. It covers zero and signed zero, negative values, subnormals, extreme
finite numbers, NaN/infinities, 5,000 seeded wide-log random cases, and
adjacent enrichment boundaries. Empty `(0, 4)` batches preserve empty outputs;
four invalid shapes are rejected. Ten direct calibration checks retain
zero-exposure no-test outcomes, invalid-statistic failures and endpoint
impossibility handling.

A separate ordering check executes the original and prototype candidate
loops in isolated namespaces with controlled upstream helpers. Its first
candidate has an impossible endpoint statistic: a positive event on a branch
with zero exposure can leave positive aggregate exposure, but the established
summary deliberately marks that statistic undefined. Its second candidate
raises an independent exposure error. The baseline validates the first row
immediately and reports:

```text
ValueError: Undefined rate score in selected candidates; null replicate cannot be omitted.
```

The prototype queues that row, reaches the later candidate and instead reports:

```text
RuntimeError: Later candidate exposure failed.
```

Both paths still reject the replicate; neither silently removes a failed
trial. The first reported failure nevertheless changes, violating the task's
failure-behavior requirement. This controlled check does not claim that the
bundled datasets contain that failure. It demonstrates a reachable ordering
regression in the deferred validation design.

No timings were collected. An initial process was terminated while waiting
for the shared benchmark lock after this correctness issue was identified.
The optional `--benchmark` harness retains PGK/PEPC fixed-candidate chunk
comparisons against both the baseline and proposal 03's scalar finite checks,
with five warmups and seven alternating measured repetitions under
`bench_common.benchmark_lock()`. These comparisons were **not run**; there is
no speedup claim and no gain attributed to proposal 04 or double-counted from
proposal 03. Full CLI, installed-wheel, native/fallback parity and biological
calibration were not run for this rejected prototype.

Verification: `python reports/performance_round3_20261023/proposal04.py`,
`python -m py_compile reports/performance_round3_20261023/proposal04.py`,
`python -m ruff check reports/performance_round3_20261023/proposal04.py`, and
`git diff --check` completed successfully. The JSON intentionally records the
failure-order mismatch as the rejection evidence.
