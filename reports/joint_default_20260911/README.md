# Joint defaults, MG correction and scan rate mixtures

Joint is now the CLI default for search/analyze, sites/site, inspect, benchmark
and scan. `--substitution_posterior marginal` selects the legacy estimator.
In scan, `--scan_observation` is an explicit override (also supporting bridge).
Unspecified scan exposure/length options resolve to endpoint/raw for joint or
bridge and q_weighted/n_rescaled for marginal. Explicit incompatible settings
fail; unsupported models never silently switch estimators.

## Implementation

- MG/MGK now use target nucleotide frequencies at the changed position. F1X4
  and F3X4 are counted from the original fitted alignment, excluding unknown
  codons, and checked against reported codon frequencies when available.
  Codon stationary probabilities include the stop-codon normalization.
  Earlier MG used GY-style target-codon weighting. This correction applies to
  **marginal as well as joint** and to simulation; the MG bug is not preserved
  as a legacy option. True-ASR exports retain MG nucleotide frequencies so
  reimport does not estimate a different generator from simulated tip counts.
- Joint scan uses the shared blocked endpoint engine. Both observed events and
  predictive opportunities integrate the site-wide posterior category weights.
  Exposure retains each category's conditional parent state; it does not use a
  posterior-mean rate or multiply two separately averaged quantities.
- Fixed-model `parametric` calibration draws one prior category per site across
  the tree and recomputes category posteriors in each simulated data set.
  Scan diagnostics record the resolved estimator, category rates/priors and
  missing-tip policy, including empty output. Shared worker transport includes
  the retained category states. Existing fitted-bootstrap child commands
  preserve the common posterior switch and scan override.

The MG correction follows the [IQ-TREE codon implementation](https://github.com/iqtree/iqtree2/blob/master/model/modelcodon.cpp)
and its [frequency counting](https://github.com/iqtree/iqtree2/blob/master/alignment/alignment.cpp).

## Verification

The [reproducible check](verify.py) fits four PGK tips (300 codons) with IQ-TREE
2.3.6 and computes their likelihood independently using CSUBST's reconstructed
MG Q and scaled pruning on the fitted tree. Verbose fitted parameters and
counted frequencies are used.

| Model | Absolute log-likelihood difference from IQ-TREE |
|---|---:|
| MG+F1X4 | 4.25e-7 |
| MG+F3X4 | 7.57e-8 |
| MGK+F3X4 | 8.37e-8 |

Each model completes search with both joint and marginal. MG+F3X4 also completes
scan, sites, inspect, benchmark and simulation. For the default ECMK07+F+R4
scan, all 46 output rows match between implicit joint, explicit joint and two
workers. The fixed-model mixture bootstrap completes all three smoke-test
replicates with zero failures. Three replicates verify execution only, not
calibration accuracy. A full PEPC default joint/R4 scan with two workers also
completes (99 candidate rows, 10 foreground units).

Independent tiny-tree tests enumerate every ancestral/tip assignment and rate
category. They compare joint events, category-weighted node posteriors and
conditional exposure, including a zero-rate component, varying tip evidence,
a wholly missing site and block sizes 1/3/64. MG tests cover stationary Q,
unit time scale, forbidden multi-nucleotide instantaneous changes, mismatched
input frequencies and simulation export/reimport. CLI tests cover all affected
commands, aliases, explicit marginal and scan override/exposure resolution.

Checks: 2,146 non-process tests + 4 process tests passed; 5 optional-dependency
skips. The 16 native checks passed (overlapping the full suite). After the final
bootstrap metadata addition, all 20 scan-CTMC integration tests passed again.
Lint, repository hygiene, local documentation checks and configured type checks
passed. The external wiki was not checked. See [checks](checks.log) and
[results and source hashes](validation.json).

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python reports/joint_default_20260911/verify.py --workdir /tmp/joint-default-check
make test test-native lint typecheck
```

The separate PEPC check uses bundled PEPC alignment/tree/state/rate/iqtree/log
files, the independent foreground file from reports/csubst_scan_pepc_20260625,
`--threads 2 --blas_threads 1 --scan_site_plot no --scan_pvalue_calibration none`.
No runtime improvement is claimed by this report.

## Remaining contracts

Native 3Di **scan** and mixtures of different Q matrices remain unsupported.
Bridge requires uniform unit rates. Fitted `parametric_bootstrap` still requires
uniform GY+F/GY+FQ; the new rate-mixture support covers joint observations,
conditional exposure and fixed-model `parametric` calibration. Default
`full_scan` remains foreground-clade permutation at fixed posterior tensors.
Changing the observation estimator does not establish P-value calibration.

At this verification snapshot, scan and search had distinct missing-tip reporting
policies. The subsequent [cross-command unification](../event_unification_20260911/README.md)
aligned them: both integrate missing latent states during inference and exclude
unobserved child subtrees from reported events. Joint scan retains
O(categories × nodes × sites × states) category states for exposure plus its
grouped event tensor; raw codon-pair temporaries and pruning are blocked.

See [endpoint documentation](../../docs/ENDPOINT_POSTERIORS.md) and
[scan documentation](../../docs/SCAN_CTMC.md) for supported modifiers, legacy
selection, statistical interpretation and true-ASR handling.
