# Method review for `csubst scan`

Date: 2026-06-26

This report reviews the theoretical status of the current `csubst scan` implementation and compares it with methods that address similar questions: explicit convergent amino-acid substitution counting, PCOC/profile-shift methods, CSUBST omegaC-style convergence-rate tests, and ESL-PSC-style predictive approaches.

## Current `csubst scan` Model

The current implementation does the following:

1. Infer nonsynonymous-state substitutions from ancestral-state posterior tensors.
2. Keep branch-site-state events whose posterior event probability is at least `--scan_min_event_pp`.
3. Build candidate substitution patterns from foreground branches.
4. Keep candidates whose lineage-level support is at least `--scan_min_support`.
5. For each retained candidate, compute:
   - foreground event count: sum of matching event mass in foreground branches
   - other event count: sum of matching event mass in the configured control branch set
   - foreground exposure: branch length multiplied by opportunity
   - other exposure: same for the configured control branch set
6. Under the default `--scan_rate_event_mode posterior_sum`, candidate discovery/support remain thresholded, but rate P values use all matching posterior event mass. `called` is available for the previous thresholded-count behavior.
7. Under the default `--scan_rate_exposure q_weighted`, opportunity is posterior parent-state mass in candidate ancestral states multiplied by the candidate nonsynonymous transition weight. With the default `--scan_rate_length n_rescaled`, this weight is the conditional probability of the candidate transition among all nonsynonymous outgoing transitions from the parent state; with raw/S+N lengths, it is the nonsynonymous instantaneous rate toward the candidate derived state. This excludes post-hit descendant branch length when the parent state has already reached the candidate derived state. `state_aware` is available as the same state filter without Q weighting.
8. A one-sided Poisson LRT compares target rate vs other rate.
9. By default, `--scan_pvalue_calibration full_scan` reruns candidate discovery on foreground-clade permutations and reports a maxT-style empirical P value.

In formula form, for a fixed candidate pattern:

```text
event_b = sum posterior mass of substitutions matching the candidate on branch b
exposure_b = selected_branch_length_b * Pr(parent state can produce candidate | data) * candidate_transition_weight

target_event = sum_{b in target} event_b
target_exposure = sum_{b in target} exposure_b
other_event = sum_{b in configured control set} event_b
other_exposure = sum_{b in configured control set} exposure_b
```

The P value tests:

```text
H0: lambda_target = lambda_other
HA: lambda_target > lambda_other
```

where the observed counts are posterior event mass, not integer counts from a fully specified generative substitution model.

For the default full-scan empirical calibration, each permutation produces a complete scan table and the minimum analytic P value is used as the permutation statistic:

```text
p_empirical_maxT = (1 + #{min_p_perm <= p_obs}) / (1 + n_successful_permutations)
```

## Bottom-Line Assessment

The method is theoretically defensible as a **screening and auditing statistic for recurrent posterior substitutions**, not as a fully calibrated confirmatory test of adaptive convergence.

It is strongest when used to answer:

```text
Given this inferred candidate substitution, is posterior event mass enriched on foreground branches after accounting for branch length and parent-state opportunity?
```

It should not be over-interpreted as:

```text
This site is adaptively convergent with a calibrated post-selection P value under a complete codon or amino-acid model.
```

The main reason is candidate selection. Candidates are discovered from foreground events and then tested for foreground enrichment using the same ASR/posterior data. This makes `p_rate_enrichment` a post-selection screening P value.

## Literature Context

### CSUBST / omegaC

CSUBST was designed to test genotype-phenotype associations using error-corrected rates of protein convergence. Its key conceptual advantage is that convergence signals are calibrated against expected or synonymous substitution behavior, reducing false positives caused by chance, phylogeny, or background rate variation. Source: [Fukushima and Pollock 2023, Nature Ecology & Evolution](https://www.nature.com/articles/s41559-022-01932-7).

`csubst scan` is intentionally narrower than omegaC. It does not search branch combinations or estimate omegaC. It lists and ranks recurrent substitutions. Therefore it gives up some of omegaC's null-model depth in exchange for transparency.

Implication:

- Good: transparent branch/site/from/to event table.
- Risk: no synonymous or expected-substitution correction is included in the current P value, so `p_rate_enrichment` is less model-calibrated than omegaC.

### Explicit Convergent-Substitution Counting

Classical molecular convergence studies count parallel or convergent amino-acid substitutions inferred on independent branches, often comparing observed counts with expectations under a model or with background substitutions. This family includes early explicit convergence definitions and tools derived from Zhang and Kumar-style substitution mapping. These methods are close in spirit to `csubst scan`: infer substitutions, identify recurrent changes, and ask whether they occur more often than expected.

Relevant concerns from this literature:

- ASR uncertainty matters.
- Chance convergence is common at fast-evolving or chemically constrained sites.
- Identical amino-acid substitutions are interpretable but can be too strict.
- Multiple substitutions and reversions can mislead simple parsimonious counting.

`csubst scan` addresses some of this by using posterior event mass and state-aware exposure, but it does not yet provide a full model-expected count for each candidate state change.

### PCOC and Profile-Shift Methods

PCOC (Profile Change with One Change) detects sites associated with convergent phenotype transitions using likelihood models that combine amino-acid preference shifts and a OneChange component. It compares a convergent model to a null model at each site. Sources: [Rey et al. 2018, Molecular Biology and Evolution](https://academic.oup.com/mbe/article/35/9/2296/5050468), [PCOC GitHub](https://github.com/CarineRey/pcoc).

PCOC is a stronger confirmatory model for site-level phenotype-associated amino-acid shifts, because it explicitly models amino-acid profiles on convergent branches. It is not primarily an event-table tool.

Comparison:

| Axis | `csubst scan` | PCOC |
| --- | --- | --- |
| Main object | branch-site substitution events | site-level convergent profile model |
| Data type | codon-derived amino-acid/nonsyn states, recodings | amino-acid alignment |
| Output | from/to states, support branches, foreground-vs-control rates | posterior/site evidence for PCOC/PC/OC models |
| Statistical null | Poisson rate equality after candidate selection | explicit likelihood model comparison |
| Strength | transparent candidate auditing | model-calibrated site detection |
| Weakness | post-selection screening P value | less direct event-level audit table |

Implication:

`csubst scan` should be framed as complementary to PCOC. PCOC can say a site fits a convergent amino-acid preference shift; `csubst scan` can say which explicit substitutions on which branches drive the signal.

### Topology-Based Convergence Claims and Their Critiques

Convergence studies based only on phylogenetic topology or clustering can be vulnerable to rate heterogeneity, model misspecification, and sampling artifacts. Critiques of high-profile molecular convergence claims in echolocation studies emphasize that apparent convergence can arise by chance or from background evolutionary constraints. Example source: [Thomas and Hahn 2015, MBE](https://academic.oup.com/mbe/article/32/8/2085/2925574).

`csubst scan` is safer than topology-only approaches because it uses a rooted tree, branch lengths, ancestral-state posteriors, and foreground/control branch sets. However, its current P value still lacks a full candidate-specific expected substitution model, so it should not be presented as the final evidence for adaptive molecular convergence.

### ESL-PSC / Predictive Sparse-Learning Methods

ESL-PSC uses paired species contrast and sparse predictive modeling to identify genes and sites associated with convergent traits. The toolkit extends this workflow with GUI/CLI support and related analyses. Sources: [Allard and Kumar 2025, Nature Communications](https://www.nature.com/articles/s41467-025-58428-8), [ESL-PSC Toolkit arXiv preprint](https://arxiv.org/abs/2605.27677), [ESL-PSC GitHub](https://github.com/kumarlabgit/ESL-PSC).

ESL-PSC is a predictor/feature-selection approach, not a branch-substitution rate test. It is strong for proteome-scale gene/site ranking, but it does not directly estimate ancestral substitutions or branch-level event rates.

Comparison:

| Axis | `csubst scan` | ESL-PSC |
| --- | --- | --- |
| Main object | candidate substitution events | predictive sequence features |
| Phylogeny use | explicit branch history and ASR | paired species contrasts |
| Branch lengths | used in exposure | not central |
| Output scale | gene/alignment-level scan, batchable | genome/proteome-scale sparse model |
| Strength | mechanistic event audit | large-scale trait predictor discovery |
| Weakness | not a predictive model | not an ancestral-event table |

Implication:

ESL-PSC can prioritize genes/sites across genomes; `csubst scan` can then inspect whether those candidates have explicit foreground-enriched substitutions.

### Post-Selection Inference and Circular Analysis

The current `scan` P value is vulnerable to a well-known issue: using the same data to select candidates and test them inflates apparent significance unless selection is accounted for. This is the general "double dipping" or post-selection inference problem. A classic warning in biological data analysis is [Kriegeskorte et al. 2009, Nature Neuroscience](https://www.nature.com/articles/nn.2303), with open discussion also available through [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2841687/). General selective-inference theory similarly emphasizes that naive P values after model/feature selection are not confirmatory.

Implication:

`p_rate_enrichment` should be explicitly documented as a screening or ranking statistic unless a selection-aware null is added.

## Theoretical Issues Found

### Issue 1: Candidate Selection Bias

Severity: high for confirmatory inference; acceptable for screening.

Candidates are selected because they recur in foreground branches. The same events are then used to test foreground enrichment. This makes small P values anti-conservative if interpreted as unconditional evidence.

Recommended framing:

```text
p_rate_enrichment is a post-selection screening statistic for called candidate substitutions.
```

Implemented / remaining fixes:

- foreground-clade permutation matched by clade size, with `full_scan` as the default calibration
- split-sample or split-branch discovery/testing
- simulation under fitted null models
- report empirical P values alongside the current analytic P value

### Issue 2: Thresholded Posterior Events Are Not Model Counts

Severity: medium to high.

`extract_atomic_events` drops branch-site substitutions below `--scan_min_event_pp` for candidate discovery and support. If `--scan_rate_event_mode called` is used, the P value also tests enrichment of **called high-confidence events**, not the full posterior mass. The default `posterior_sum` mode avoids this thresholding for rate counts.

Example risk:

- Foreground events have high posterior probability and are counted.
- Background has many low-posterior matching events and they are dropped.
- Background event count is underestimated, making foreground enrichment stronger.

Implemented fix:

- Use thresholding for candidate discovery/support, but compute rate P values from all posterior mass for the candidate state change.
- Use `--scan_rate_event_mode called` when exact continuity with thresholded event counts is desired.

### Issue 3: Poisson Approximation Is a Ranking Model, Not a Full Substitution Model

Severity: medium.

The Poisson LRT assumes independent event counts with rate proportional to exposure. Real substitution histories are CTMC processes on a tree, with state-dependent rates, site-specific constraints, multiple hits, and posterior dependence among adjacent branches.

The approximation is reasonable when:

- events are rare,
- branch lengths are not too long,
- the candidate is fixed before testing,
- exposure captures most opportunity variation.

It is weaker when:

- branches are long,
- multiple/reverse substitutions are likely,
- site-specific exchangeability differs strongly across amino acids,
- candidate selection is target-biased.

Potential fixes:

- Add model-expected candidate counts from codon or amino-acid rate matrices.
- Use parametric simulation to calibrate P values.
- Keep analytic P values for fast screening and label them accordingly.

### Issue 4: Q-Weighted Exposure Is The Most Model-Aware Fast Denominator

Severity: medium.

`q_weighted` exposure first applies the state-aware filter, so branches whose parent state cannot produce the candidate substitution contribute little or nothing. It then weights each possible parent-to-derived transition by the fitted nonsynonymous instantaneous rate matrix. When the selected length is `n_rescaled`, the weight is normalized by the total nonsynonymous outgoing rate from that parent state, because the branch length has already been rescaled to nonsynonymous substitutions. This is the most model-aware fast denominator currently implemented.

This handles the user's stem-completion concern in a fair way and avoids the origin-weight correction problem discussed separately. It includes candidate-specific exchangeability, codon mutational accessibility, codon degeneracy, and nonsynonymous recoding through the fitted Q matrix. It still does not include:

- site-specific amino-acid preference,
- uncertainty in the fitted substitution model,
- a full CTMC likelihood for the selected candidate after foreground discovery.

Implemented / remaining fixes:

- Implemented: `q_weighted` exposure, defined as parent posterior mass times candidate transition weight; the weight is conditional on nonsynonymous outgoing rate under `n_rescaled` branch length.
- Remaining: parametric simulation or model-expected candidate counts for confirmatory calibration.
- Remaining: optional site-rate-stratified empirical nulls.

### Issue 5: Other Branches Are Heterogeneous

Severity: medium.

The `other` branch set can be a heterogeneous mixture of many clades, not a matched control population. This can make foreground-vs-other comparisons sensitive to background composition. `--scan_other_scope all|sister` makes the control branch set explicit: `all` uses all non-foreground branches, while `sister` uses sister branches of foreground units.

Remaining fixes:

- Provide local-control P values using sister branches when available.
- Add matched-clade controls beyond the current `sister` option.
- Report background composition diagnostics.

### Issue 6: Multiple Testing Correction Is Useful but Not Fully Calibrated

Severity: medium.

BH q-values are useful for ranking, but the tested rows are highly dependent:

- overlapping match classes can share the same candidate event mass.
- match classes overlap, especially `any2any`, `any2spe`, and `spe2spe`.
- candidates are selected before testing.

Potential fixes:

- Add q-values stratified by `trait` and `scan_match`.
- Add candidate-level q-values, using one row per `scan_id` per hypothesis family.
- Prefer empirical null calibration for final significance claims.

### Issue 7: Site Rate and Conservation Are Annotations, Not Null-Model Terms

Severity: low.

Including site rate and conservation in the output is theoretically clean as prioritization metadata. They are not currently used in the P value, so they do not create circularity in the test. However, if users select hits after looking at these annotations, the final biological claim becomes a post hoc prioritization.

Potential fixes:

- Document site-rate/conservation columns as prioritization fields.
- Optionally add stratified empirical nulls by site-rate quantile.

### Issue 8: Origin/Stem Weighting Should Remain Out

Severity if included globally: high.

The proposed origin/stem weighting is attractive for foreground clades, but it is not cleanly symmetric for the non-foreground complement because non-foreground is often paraphyletic and has no meaningful single stem. Adding that correction to the current P value would likely make the null model ambiguous and foreground-favorable.

Current decision:

- Do not add origin/stem weighting to `p_rate_enrichment`.
- Keep `state_aware` exposure, which is symmetric and interpretable.

## Comparison Matrix

| Method | Detects explicit substitutions | Uses ASR uncertainty | Uses branch length | Has full likelihood/null model | Handles exact AA convergence | Handles profile shifts | Proteome-scale feature selection | Best role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `csubst scan` | yes | yes, via posterior event mass | yes | partial, screening Poisson LRT | yes | partially via broad match classes | no, but batchable | audit and rank recurrent substitutions |
| CSUBST omegaC/search | yes, through combinations | yes | yes | stronger expected/error-corrected framework | yes | not profile-shift focused | gene-by-gene/batchable | convergence-rate inference |
| PCOC | not as primary table | likelihood model | yes | yes | not limited to exact AA | yes | gene-by-gene/batchable | site-level convergent profile detection |
| Classical convergent substitution counting | yes | variable | variable | variable | yes | no | variable | explicit parallel/convergent event counts |
| ESL-PSC | no | no explicit ASR table | not central | predictive sparse model, not substitution rate null | feature-level only | feature-level only | yes | genome-scale predictor discovery |

## Recommended Changes Before Treating `scan` P Values As Publication-Grade

### Must-Do Documentation

1. State in help/docs that `p_rate_enrichment` is a screening statistic after candidate discovery.
2. State that candidate discovery/support are thresholded, while rate counts use `--scan_rate_event_mode`.
3. Recommend reporting support counts, site-rate/conservation, and foreground-vs-control exposure/counts along with P values.

### High-Value Implementation

1. Add parametric simulation calibration in addition to the implemented clade-permutation calibration.

### Lower-Priority But Useful

1. Branch-level companion table for each `scan_id`.
2. PCOC bridge: export/import site scores.
3. ESL-PSC bridge: aggregate scan features across genes.
4. Site-rate-stratified empirical nulls.

## Final Verdict

There is no fatal theoretical problem if `csubst scan` is presented as:

```text
a transparent, posterior-event-based scanner and ranking tool for recurrent foreground substitutions
```

There is a serious theoretical problem if `p_rate_enrichment` is presented as:

```text
an unconditional, fully calibrated proof of adaptive convergence
```

The current method is therefore best described as a practical bridge between explicit convergent-substitution counting and CSUBST's richer convergence-rate framework. It is particularly useful for:

- listing candidate substitutions,
- comparing foreground support with foreground-vs-control rate summaries,
- prioritizing slow/conserved sites,
- auditing exact branch-level evolutionary histories.

For publication-grade hypothesis testing, add empirical calibration or model-expected candidate counts.

## Primary Sources Checked

- CSUBST: Fukushima K, Pollock DD. 2023. Detecting macroevolutionary genotype-phenotype associations using error-corrected rates of protein convergence. Nature Ecology & Evolution. https://www.nature.com/articles/s41559-022-01932-7
- PCOC: Rey C, Guéguen L, Sémon M, Boussau B. 2018. Accurate detection of convergent amino-acid evolution with PCOC. Molecular Biology and Evolution. https://academic.oup.com/mbe/article/35/9/2296/5050468
- PCOC software: https://github.com/CarineRey/pcoc
- Critique of molecular convergence claims: Thomas GWC, Hahn MW. 2015. Determining the null model for detecting adaptive convergence from genomic data. Molecular Biology and Evolution. https://academic.oup.com/mbe/article/32/8/2085/2925574
- ESL-PSC: Allard A, Kumar S. 2025. Evolutionary sparse learning reveals the shared genetic basis of convergent traits. Nature Communications. https://www.nature.com/articles/s41467-025-58428-8
- ESL-PSC Toolkit preprint: Allard A, Kumar S. 2026. ESL-PSC Toolkit: A graphical software environment linking shared genetic changes to convergent phenotypes. https://arxiv.org/abs/2605.27677
- ESL-PSC software: https://github.com/kumarlabgit/ESL-PSC
- Circular/post-selection warning: Kriegeskorte N et al. 2009. Circular analysis in systems neuroscience: the dangers of double dipping. Nature Neuroscience. https://pmc.ncbi.nlm.nih.gov/articles/PMC2841687/
