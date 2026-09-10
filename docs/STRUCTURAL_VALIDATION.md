# Scientific validation of predicted 3Di states

Structural omegaC remains uncalibrated. The tools here establish a reproducible
measurement layer for review issue 9; they do not integrate prediction errors
into ASR, turn softmax into observation likelihoods, or change search/scan results.
Continue fitting GTR for the supported direct model path. Published Q.3Di.AF/LLM
matrices are not part of this work.

The next stage now supplies an [experimental-reference and fixed-GTR workflow](STRUCTURAL_OBSERVATION.md)
and a [five-structure pilot report](../reports/3di_pilot_20260910/README.md).
Those checks are separate from end-to-end omegaC calibration.

## Record predictions without discarding uncertainty

`csubst.structural_prediction.predict_3di_records(sequences, config)` returns
`PredictionRecord` objects. ESM3Di-35M and ProstT5-CNN records retain raw logits
with canonical columns `ACDEFGHIKLMNPQRSTVWY`. `record.probabilities` computes
the corresponding softmax in float64; its kind is `uncalibrated_softmax`.
The hard prediction preserves the model's original tie-breaking order.

The autoregressive `prostt5` backend supplies hard predictions with
`logits=None`, `probabilities=None`, and kind `unavailable`. Its greedy-path
conditional token probabilities have not been implemented as residue marginals.
Missing scores are never replaced with one-hot vectors.

The opt-in record API bypasses the existing character-only cache, because that
cache cannot reconstruct logits. Ordinary `predict_3di` and ASR behavior remain
unchanged. Explicit prediction artifacts are versioned, pickle-free NPZ files;
they store sequences, state order, logits, backend/model identities, preprocessing
identity and caller provenance. Softmax is reproducible from the saved logits.
`save_predictions` atomically publishes a new artifact and refuses overwrite.
`load_predictions(..., expected_model_key=...)` can enforce model identity.
These artifacts are not the versioned ASR state cache.

## Reference manifest and offline evaluation

Prepare a JSON manifest with the following shape (labels below are a synthetic
format example, not a real protein structure):

```json
{
  "schema_version": 1,
  "records": [
    {
      "id": "example",
      "amino_acids": "MKL",
      "structure_3di": "AC?",
      "family": "example-family",
      "split": "test",
      "group": "example-target-group",
      "truth_source": "simulated",
      "source": "synthetic format example"
    }
  ]
}
```

Each reference character must correspond to the same residue in the uppercase,
gapless AA sequence. Use `?` or `-` to mask unknown reference labels; these are
not deletions of positions from the AA sequence. AA `X` is accepted as an unknown
residue. Extract experimental labels with a fixed Foldseek version, preserving
chain/residue mapping, and record that version and the structure accession or
input hash in `source` (additional manifest metadata is retained). The tool
does not fetch structures or verify the declared mapping or provenance itself.

`truth_source` must be `experimental`, `simulated`, `predicted`, or
`reconstructed`. An AlphaFold-derived label is `predicted`; a resurrected ASR
sequence/structure is `reconstructed`, not a known historical ancestor. Group
names describe the target population; define them before inspecting results.

`split` is `calibration` or `test`. A family or identical AA sequence cannot
occur in both splits. The validator cannot detect undeclared homology or unknown
predictor pretraining overlap. Use family-level external separation, audit that
overlap, and keep the final test set untouched during calibration design.
No calibration parameters are fitted by this tool.

Prepare model resources separately, then run each backend on the same manifest:

```bash
python tools/validate_3di_predictions.py --manifest reference.json \
  --backend esm3di-35m --output-dir validation-esm
python tools/validate_3di_predictions.py --manifest reference.json \
  --backend prostt5-cnn --output-dir validation-cnn
python tools/validate_3di_predictions.py --manifest reference.json \
  --backend prostt5 --output-dir validation-generator
```

Downloads are disabled unless `--allow-download` is supplied. The default device
is CPU. Each output directory must be new. It receives `predictions.npz`, a
`manifest.json` snapshot and `metrics.json`. New inference records configuration,
package versions, platform and source-code hashes. No model is needed to rescore:

```bash
python tools/validate_3di_predictions.py --manifest reference.json \
  --predictions validation-esm/predictions.npz --output-dir validation-rescored
```

Reports separate calibration/test and reference-source strata, with target-group
subsets. Confusion matrices use true-state rows and predicted-state columns.
Q20 is a fraction, not a percentage. Macro-family Q20 gives equal weight to each
family with at least one scored residue. Balanced accuracy averages recall over
represented true states only. Coverage reports the fraction of reference
positions that could be scored. All-masked strata have undefined metrics (`null`).

Probabilistic scores are multiclass Brier score (sum over the 20 states, no
division by 20), natural-log loss and top-label reliability/ECE in ten fixed
equal-width confidence bins. The last bin includes 1. Log loss is calculated
from logits without a probability floor. The number of scored probabilistic
residues is explicit; hard-only records do not acquire probability metrics.
These descriptive metrics do not include confidence intervals or certify
probability calibration. In a substantive study, use family-level uncertainty
estimates and report state-specific and out-of-domain failures.

## Shared-error stress test

```bash
python tools/reproduce_3di_observation_error.py
```

The known ancestor and both true descendants are A at every site. The toy
observation model miscalls A as C with probability 0.2. Independent errors yield
a probability 0.04 of two false A-to-C endpoints; perfectly shared errors yield
0.2, despite identical marginal accuracy of 0.8. The script reports an empirical
reproduction with 100,000 sites and a fixed seed.

`simulate_observation_errors` accepts a row-normalized true-by-predicted confusion
matrix, integer tip-by-site states, a shared-error fraction and site block size.
It couples uniform quantiles across tips/blocks while retaining the specified
one-site marginal probabilities. This is an explicit stress-test construction,
not a learned model of protein evolution. It does not run ASR, codon S, omegaC,
or a hypothesis test, and its joint false-endpoint rate is not an omegaC FPR.

## Remaining scientific work and acceptance gates

1. Obtain matched experimental structures for the actual target protein groups;
   compare all three pinned predictors on identical residues and family splits.
   Examine alternative structures, missing residues, state-dependent confusion,
   sequence-distance effects, contiguous errors and errors shared across clades.
   Agreement between predictors is not independent validation.
2. Fit an externally estimated observation model `P(prediction | true 3Di)`.
   Discriminative softmax is not that likelihood. Validate any posterior-to-
   likelihood conversion and its reference prior. Fit GTR/branch lengths and
   reconstruct ancestors consistently with the observation model; merely
   softening tip tensors after hard-state fitting is a sensitivity analysis.
   Check identifiability and states absent from the observed alignment.
3. Evaluate known-ancestor simulations and, where available, experimental
   lineages. For translate, propagate AA sequence uncertainty through the whole
   predictor, preserve parent/child dependence where possible, and evaluate
   statistics per draw. Do not average P values. Marginal-posterior products
   still require the separate issue-8 assessment.
4. Build both GTR-consistent technical nulls and model-misspecified nulls with
   matched codon/3Di dependence and prediction error. Refit and rerun the actual
   pipeline, including site selection, pseudocounts, long-tail transformation
   and any search. Independently generating codon and 3Di data is insufficient
   to validate their biological N/S relationship. Issues 2 and 4 must supply
   consistent omegaC and joint-category null calculations first.
5. Predeclare FPR tolerance, Monte Carlo precision and power requirements for
   held-out target groups. For example, a chosen 5% test could require its 95%
   FPR upper confidence bound to stay below 6%, alongside useful power. This is
   a proposed gate, not an achieved result. Extend to search/multiple-testing
   guarantees only after the corresponding issues 6 and 13 are evaluated.

Until those gates are met, describe results as candidates for repeated changes
of predicted structural states. Residue prediction validation alone does not
establish structural adaptation or a calibrated structural omegaC.

## Sources

- [Foldseek original paper](https://doi.org/10.1038/s41587-023-01773-0): 3Di
  describes residue-neighbor geometry, not an independent codon state.
- [ProstT5 original paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11616678/):
  sequence prediction and homology-search performance are different endpoints.
- [ESM3Di implementation](https://github.com/DessimozLab/ESM3di).
- [Predictor configuration and model provenance](STRUCTURAL_ALPHABET.md).
