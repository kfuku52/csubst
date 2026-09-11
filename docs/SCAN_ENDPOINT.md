# Finite-time endpoint exposure in scan

`--scan_rate_exposure endpoint --scan_rate_length raw` compares observed endpoint
changes with their finite-time codon-model expectation. It accounts for paths
through intermediate codons even when the direct instantaneous rate is zero.
This is the default exposure for joint scan. Explicit marginal observation
requests default to the legacy `q_weighted` exposure instead.

## Supported inputs

Joint observations support fitted GY, MG/MGK, ECMK07 and ECMrest models,
including discrete G/R/I rates. MG/MGK use counted F1X4/F3X4 frequencies.
For each site the expectation is summed over P(category | all tips) times the
category-conditional parent posterior and exp(Q × category rate × length).
The fitted category table is required; posterior mean rates cannot replace it.
Marginal/bridge endpoint exposure remains limited to uniform unit rates.
The full codon Q, stationary frequencies, codon/group order and posterior
states must be available.

The model's raw IQ-TREE branch lengths are required. `n_rescaled` and
`sn_rescaled` use observed-change summaries and are not the time parameter of
the fitted CTMC. `posterior_sum` events are required; `called` would compare
thresholded observations with an unthresholded expectation. Discovery and
foreground support still use the configured event/support thresholds.

Codon-derived amino-acid/reduced states and native `3di20` joint observations
are supported. Native 3Di uses the independently fitted uniform GTRX+FQ
20-state generator, stationary frequencies and structural branch lengths;
its state-group mapping is the identity. Codon synonymous counts retain their
independent codon fit and rate mixture. Direct ASR is required. Analytical,
parametric and bridge 3Di scan routes remain unsupported and are rejected.
Use `--scan_pvalue_calibration none --scan_n_permutations 0` for exploratory
scores without calibration. Explicit legacy 3Di `q_weighted` requests still
resolve to `state_aware`.

For example, fit a uniform model and run scan with independent foreground units:

```bash
python -m csubst scan \
  --alignment_file csubst/dataset/PEPC.alignment.fa \
  --rooted_tree_file csubst/dataset/PEPC.tree.nwk \
  --foreground reports/csubst_scan_pepc_20260625/PEPC.foreground.independent.txt \
  --iqtree_model ECMK07+F --iqtree_outdir /tmp/pepc-uniform-fit \
  --scan_rate_exposure endpoint --scan_rate_length raw \
  --scan_rate_event_mode posterior_sum --scan_match any2spe \
  --scan_pvalue_calibration none --scan_site_plot no \
  --threads 1 --outdir /tmp/pepc-endpoint
```

Do not reuse a mixture-model ASR fit as if it were uniform. For a new/old
comparison, reuse the same uniform fit, and vary only exposure/length options.

## What is calculated

Let `h(i)` map codon `i` to its reported state, `C` denote the candidate state
pairs, and `pi_b,s(i)` the parent codon posterior. A branch contributes

```text
E_b,s = sum_i,j pi_b,s(i) * exp(Q * t_b)[i,j]
        over (h(i), h(j)) in C and h(i) != h(j).
```

The full codon generator is exponentiated **before** destination states are
grouped. Same-group endpoints, including synonymous codon changes, are excluded.
There is no additional multiplication by branch length. Each branch contributes
at most one expected endpoint event, whereas a CTMC jump count can exceed one.
Endpoint changes do not count an excursion that returns to the starting group.

Transitions are prepared once per distinct branch length and reused across
sites, candidates and foreground permutations. The stored transition array has
length x codon x reported-state axes, not branch x site x codon-pair axes.
Workers can share large arrays through the existing scan memmap mechanism.

For joint/bridge observations, eligibility is derived from original tip
emissions, before missing tips are imputed. The same mask excludes observed
event mass and exposure. For explicit marginal observations, only branch/site
pairs with nonmissing codon and reported-state rows at both endpoints contribute. Nonmissing rows must sum to one within ASR
text precision; parent codon rows are normalized to absorb that rounding.
Very small positive transition probabilities are retained. Negative numerical
roundoff is removed only after validating the transition matrix.

## Output and zero diagnostics

| Column | Meaning |
|---|---|
| `target_exposure`, `other_exposure` | Denominators actually used, for all exposure modes |
| `scan_exposure_units` | `expected_endpoint_events` or `legacy_weighted_branch_length` |
| `scan_endpoint_model` | Fitted codon model used by endpoint exposure |
| `scan_observation_method` | `joint_endpoint_posterior` by default; `marginal_posterior_product` for explicit marginal |
| `scan_inference_method` | `exploratory_poisson_lrt` |
| `rate_status` | `ok`, `positive_event_zero_exposure`, `zero_target_exposure` or `zero_other_exposure` |
| `{target,other}_zero_exposure_branch_count` | Nonmissing branches with zero exposure |
| `{target,other}_positive_event_zero_exposure_branch_count` | Branches carrying positive observed mass but no exposure |
| `{target,other}_missing_exposure_branch_count` | Branch/site pairs excluded for missing endpoint states in endpoint mode |
| `{target,other}_exposure_diagnostics` | Counts by cause, separated by semicolons |

The old `{target,other}_exposure_branch_length` columns retain their values in
legacy modes and are undefined in endpoint mode. Consumers should use the new
generic columns together with `scan_exposure_units`; units must not be inferred
from an old column name. Event rates in endpoint mode are observed/expected
endpoint-event ratios, not inferred instantaneous CTMC rates.

Endpoint zero causes distinguish `missing_state`, `zero_model_length`,
`zero_source_mass`, `unreachable` and `numerical_zero`. Reachability is calculated
from positive generator edges, including intermediate states. `numerical_zero`
means a path is possible but its computed probability is zero, e.g. underflow.
Legacy zeros are labeled `legacy_zero_exposure`; legacy missingness handling
and numerical statistics are unchanged.

A positive observation on a zero-exposure branch makes endpoint rates and P
values undefined even when the aggregate exposure of its group is positive.
No epsilon or pseudocount is added. An undefined endpoint statistic in a
permutation is reported as a failed permutation, not treated as a configuration
with no discovered candidates. Failure counts/reasons remain part of the
existing output; exclusion of failed configurations is not a calibration proof.

## Interpretation and remaining limits

With explicit `--scan_observation marginal`, this changes only the finite-time
opportunity. Observations remain products of marginal posteriors, and analytical
P/q values are exploratory. Legacy foreground permutations keep ASR fixed.

For true joint parent/child posteriors, hidden within-edge jumps, and a
parametric bootstrap that repeats ASR and candidate selection, see
[Joint posterior and CTMC bridge scan](SCAN_CTMC.md). Bridge and calibration are separate opt-in
modes; their rate-model and fitting limitations are
explicitly documented there.

## Reproduce the runtime comparison

After the example has prepared the uniform fit:

```bash
python .github/scripts/scan_endpoint_benchmark.py \
  --fit-dir /tmp/pepc-uniform-fit --outdir /tmp/scan-comparison --repeats 3
```

The script compares legacy default, Q-weighted raw lengths and endpoint raw
lengths, excludes one warmup each, rotates run order and records wall time,
scan-stage time, peak RSS, input hashes and exact commands. It checks that
candidates, support and observed event counts are identical. Add
`--calibration full_scan --niter 20` for a permutation workload; 20 iterations
are a runtime workload, not a significance-calibration study.
