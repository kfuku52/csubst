# Validation Checklist

## Use This Flow

1. Create an isolated run directory:
   - `RUN=/tmp/csubst_validation_$(date +%Y%m%d_%H%M%S)`
   - `mkdir -p "$RUN"`
2. Copy only required input files into `$RUN`.
3. Change directory to `$RUN` before execution.
4. Run with peak RAM capture:
   - `/usr/bin/time -l -p /Users/kf/miniforge3/bin/python <repo>/csubst/csubst analyze ...`
5. Save logs:
   - `> run.stdout.log 2> run.stderr.log`
6. Hash key files:
   - `shasum -a 256 csubst_instantaneous_rate_matrix.tsv csubst_cb_2.tsv`
7. Compare table parity with script:
   - `python <repo>/.codex/skills/csubst-validation-parity/scripts/compare_tsv_parity.py a.tsv b.tsv --ignore-col-regex '^elapsed_sec$'`

## Report Template

- Run directory:
- Commit and branch:
- Command:
- Runtime (`real`):
- Peak memory footprint:
- Hash parity:
- TSV parity:
- Notes and caveats:

## Known Caveats

- If `/usr/bin/time -l` is blocked by sandbox permissions, rerun with escalation.
- Always state which columns were excluded from parity checks.
