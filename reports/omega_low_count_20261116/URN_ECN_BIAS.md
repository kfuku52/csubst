# Sources of ECN error in the empirical PEPC urn

This comparison reuses 18 simulated 520-codon PEPC alignments on the same
71-tip topology. Each alignment contributes its saved ECMK07_GY0K, GY,
MGK, and ECMK07 ancestral-state fit. Four fixed branch folds define each
training pool. The fit for a pair uses eligible branches outside the
folds of both evaluated branches. Simulation labels define the biased
subset for scoring. For each fit, all experimental urn methods share
OCN, OCS, and the original empirical-urn ECS. The ordinary codon model
uses its own ECN and ECS. [Previous comparison](CODON_MODEL_COMPARISON.md)

The two background conditions pool 50,262 pair-replicates each. The
biased condition pools 216 pair-replicates with both branches affected.
Each condition uses six seeds. We score ECN against the simulator's
background-Q expectation using a pooled count ratio and a pairwise RMSE.
We also score `log(1 + omegaC)` against the simulator's generating ratio
on rows with finite ratios for all methods of a given ASR fit.

## Separate the site weights from the destination totals

The current N site profile uses the inferred parent amino acid and
cross-fitted route weights multiplied by the empirical relative site
rate. The site rate is the site's total training N+S endpoint mass plus
0.5, normalized to mean one. For each branch and N destination, the
urn then allocates that branch's observed destination total using the
profile. All route fits keep the existing destination-pooled prior of
0.5 except the explicit unpooled-route control. [Empirical urn design](URN_EMPIRICAL_ROUTES.md)

To change the site component, replace the empirical rate in both the
route exposure and the evaluation-site profile. The controls use raw
N+S mass, synonymous mass, fitted posterior mean site rates, or known
simulator rates. The fitted-Q profile control instead uses the fitted
predictive N endpoint probabilities as site weights. In those controls
the urn still uses the observed N destination totals.

To change the draw totals with the empirical site profile fixed, compare
three estimates. Pooled destinations divides the observed branch-wide N
total among destinations in proportion to the learned route profile
summed across sites. Pooled N+S uses the observed branch-wide N+S total
as the measure of branch effort and lets the learned site profile and
routes determine the N count for each destination. It learns one scale
factor per fold so that the N count summed across training branches
matches their observed N count. The fitted-Q totals control uses the
fitted predictive counts for each branch and destination.

The cross-validated site estimate mixes the observed site distribution
with a uniform distribution. Within each training pool, leave one branch
out and score that branch's site counts under the remaining training
branches' mixed site distribution. Choose the uniform weight that
maximizes the sum of these multinomial log scores. Repeat this separately
for every pair of folds. The combined method uses those weights in the
route fit and estimates draw totals with pooled N+S. It still uses the
original empirical S calculation for comparison.

## What drives the bias

In the mixed-rate condition, the fixed site pseudocount assigns a mean
relative rate of 0.118–0.125 to simulated rate-zero sites across the four
ASR models. It reduces the mean at rate-three sites to 2.618–2.675.
For rare draws, spreading weight over more sites reduces the chance that
two branches draw the same site. Removing the uniform pseudocount raises
pooled ECN from 0.795–0.849 to 0.936–0.996 of the Q reference. It also
raises pairwise ECN RMSE from 0.0166–0.0168 to 0.0182–0.0187 and increases
pooled ECN in the two unit-rate conditions. Using the known site rates
gives 0.887–0.930 of the Q reference in the mixed-rate condition. This
keeps the observed destination totals and the empirical route model.

Changing the draw totals with site weights held fixed improves the
individual-pair errors. Pooling only within N gives mixed-rate ECN RMSE
of 0.0132 across the four ASR fits. Pooling from total N+S gives
0.0104–0.0110. In mixed rates the latter still gives 0.842–0.899 of the
generating ECN. Removing all shrinkage from the N route table with the
original totals yields 1.136–1.197 of the expected mixed-rate ECN and
1.538–1.576 at unit rate. Substituting fitted-Q totals alone gives
0.827–2.982 across models in the mixed-rate condition.

## Combine site regularization and pooled N+S totals

The learned uniform weight averages 55% in the unit-rate background,
8% in the mixed-rate background and 58% in the biased unit-rate
condition. With the pooled N+S totals, cross-validation improves ECN
RMSE relative to the current empirical urn in all 72 fits. The table
shows pooled ECN divided by its Q reference, with the pairwise ECN RMSE
in parentheses, for the ECMK07_GY0K ASR fits.

| Method | Background, unit rate | Background, mixed rates | Both branches biased |
| --- | ---: | ---: | ---: |
| Fitted codon model | 1.090 (0.00410) | 1.002 (0.00815) | 1.130 (0.00345) |
| Current empirical urn | 1.049 (0.00984) | 0.795 (0.01675) | 1.080 (0.01091) |
| Remove the site pseudocount only | 1.114 (0.01063) | 0.936 (0.01819) | 1.149 (0.01183) |
| Pool N+S totals only | 1.090 (0.00403) | 0.842 (0.01094) | 1.188 (0.00408) |
| Pool N+S totals with no site pseudocount | 1.157 (0.00453) | 0.992 (0.00920) | 1.262 (0.00485) |
| Pool N+S totals with cross-validated site weights | 0.943 (0.00382) | 0.887 (0.01028) | 1.022 (0.00296) |

In mixed rates, the cross-validated combination yields 0.887–0.950
of the Q expectation and ECN RMSE 0.00988–0.01038 across the four models.
Its pooled omegaC is 1.038–1.058 with the original empirical ECS, and
pairwise log omegaC RMSE goes from 1.004–1.150 to 0.817–0.859. The
no-pseudocount combination has lower ECN RMSE in that condition, alongside
16–19% pooled ECN inflation at unit rate and 26–28% inflation on the
biased pairs. The adaptive combination is the more balanced candidate
on this topology. Its remaining mixed-rate ECN deficit and the contrast
between pooled and pairwise error motivate testing other trees and rate
mixtures before generalizing the result.
[Per-fit counts and errors](urn_ecn_bias_summary.tsv) ·
[Site rates by fold and generating rate](urn_ecn_site_rates.tsv)

## Reproduction

Use the saved fits from the [previous comparison](CODON_MODEL_COMPARISON.md#reproduction).
The replay used editable CSUBST 1.17.5 based on `c39484a`, Python 3.12.15,
and IQ-TREE 3.1.4 fits on macOS arm64. The script checks the generator
hash and the ASR projections against the saved search counts.

```bash
export MPLBACKEND=Agg VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python reports/omega_low_count_20261116/urn_ecn_bias.py \
  "$panel" "$runs" "$second_panel" "$work/fits" "$work/ecn.tsv" \
  --site-rates-output "$work/site-rates.tsv" --jobs 8
```
