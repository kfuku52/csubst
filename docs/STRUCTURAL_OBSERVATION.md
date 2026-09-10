# Experimental references and fixed-GTR observation inference

This research workflow extends the [prediction measurement API](STRUCTURAL_VALIDATION.md).
It does not fit an error model or Q, change production ASR/search/scan, or provide
calibrated structural omegaC. The [pilot report](../reports/3di_pilot_20260910/README.md)
contains real-checkpoint measurements and synthetic inference results.

## Build a reference panel

Install the optional CIF reader with `pip install '.[structural-validation]'`.
Prediction also requires the `3di` extra and prepared model resources. Reference
extraction uses the official [Foldseek release 10-941cd33](https://github.com/steineggerlab/foldseek/releases/tag/10-941cd33),
with version string `941cd33ff0771cd2e3f144e3293e22a2b87e9fda`. The version is
fixed because the workflow reads the descriptor exporter, including its partner
distance fields; other versions need explicit verification before support.
No protein language model is used to generate reference labels.

A panel JSON contains `schema_version: 1` and a `structures` list. Each entry has
`id`, relative `path`, `label_asym_id`, optional `model` (default 1), `family`,
`group`, and `split`. IDs are unique letters/digits/underscore/hyphen names.
Paths are relative to `--source-root`. See the checked-in
[five-structure panel](../reports/3di_pilot_20260910/panel.json) for an example.

```bash
python tools/build_3di_validation_manifest.py \
  --panel reports/3di_pilot_20260910/panel.json \
  --source-root SOURCE_DATA_DIRECTORY \
  --foldseek FOLDSEEK_EXECUTABLE --output-dir reference-panel
python tools/validate_3di_predictions.py --manifest reference-panel/manifest.json \
  --backend esm3di-35m --device cpu --output-dir validation-esm
```

Source files are read only. New output directories retain the normalized PDB,
raw Foldseek descriptors/log, residue map, manifest and input panel. Provenance
includes the source-file hash, entry ID, experimental method, Gemmi version,
Foldseek version/binary hash, export hash and mask protocol. A failed extraction
does not publish a completed manifest; its diagnostic files remain available.

### Residue correspondence and conservative masking

The full `_pdbx_poly_seq_scheme` supplies the predictor's chain sequence, including
unresolved positions. The selected model and label chain determine observed
atoms. Atom monomers and available author identifiers must agree with the scheme.
The map retains label positions, author chain/numbers, insertion codes, selected
alternate conformer, observed atoms and every mask reason. Coordinate-derived
3Di is projected back to the full chain; missing residues are not compressed out
of the predictor input.

For alternate atoms, select the conformer with highest total selected-atom
occupancy, retaining common atoms and using lexical tie-breaking. Export N, CA,
C, O and CB without rebuilding absent atoms. PDB coordinates are rounded to
0.001 Angstrom; the exact export is retained. MSE is explicitly represented as
M/MET for prediction/extraction, while descriptor neighborhoods touching MSE or
UNK are excluded from scoring. Other unsupported monomers fail explicitly.

The extractor currently accepts declared X-ray, neutron, electron microscopy/
crystallography and solution/solid-state NMR methods; model-archive predicted
structures and unknown methods are rejected. This is a conservative pilot
protocol, not universal structural-file support.

Foldseek can assign an alphabet character to an undefined descriptor. Therefore
character validity alone is insufficient. The workflow masks:

- Missing CA positions and undefined/terminal descriptors.
- Descriptors whose query or partner three-residue neighborhoods lack N/CA/C,
  contain MSE/UNK, or cross nonconsecutive polymer indices.
- Neighbor CA separations outside the declared 2–4.5 Angstrom interval.
- Query/partner pairs whose compressed coordinate indices change their true
  polymer sequence distance because intervening residues are absent.

Partner indices are recovered from the pinned export's signed log-distance
feature and checked against its clipped sequence-distance feature. These
features follow [the release's descriptor implementation](https://github.com/steineggerlab/foldseek/blob/10-941cd33/lib/3di/structureto3di.cpp).
Unknown unobserved spatial partners cannot be reconstructed by these masks.
Their absence, coordinate uncertainty and genuine conformational variation
remain limitations of the reference labels.

## Compare the common-reference predictions

```bash
python tools/compare_3di_validation.py --manifest reference-panel/manifest.json \
  --predictions esm3di-35m=validation-esm/predictions.npz \
  --predictions prostt5-cnn=validation-cnn/predictions.npz \
  --predictions prostt5=validation-generator/predictions.npz \
  --output comparison.json
```

The comparison requires one split/reference source and exactly matched IDs and
sequences. It reports family-weighted results, whole-family bootstrap percentiles,
adjacent error persistence, common predictor errors and reference disagreement
for identical sequences with multiple structures. Bootstrap resampling uses
10,000 draws and seed 9; with very few families these are descriptive intervals,
not population-level guarantees. Shared structures/residues are not independent
replicates. Pooled common-error rates versus products of marginals are descriptive,
not statistical independence tests. The output refuses overwrite.

## Fixed-GTR observation likelihoods and endpoint joints

`csubst.structural_observation.observation_likelihoods(observed, confusion)`
maps observed characters (integer indices; -1 missing) to `P(observed | true)`.
Confusion rows are true states and columns observed states. Missing rows give
likelihood one for every state. Softmax is not substituted for this likelihood.

`fixed_gtr_posteriors(parents, branch_lengths, q, pi, tip_likelihoods)` uses
log-space pruning and an outside recursion. Inputs use one common state order;
Q must be reversible with unit stationary mean rate, pi strictly positive, and
the rooted tree has one -1 parent and zero incoming length at the root. Supply
likelihood arrays for exactly its leaves. This supports positive-support state
spaces; inactive states in a production context cannot be silently retained or
removed without consistently remapping every array.

Outputs include node-by-site-by-state posterior, child-indexed site-by-parent-
state-by-child-state joints and site log likelihoods. Entirely missing sites
return model priors. This differs from production missing-data zero tensors;
callers must preserve the original observation mask separately before integration.
Joints describe branch endpoints, not numbers of CTMC jumps, nor the joint
history across several branches. Q, branch lengths and confusion are fixed.
Conditional independence of the tip observations is an explicit assumption.

```bash
python tools/validate_3di_fixed_gtr.py --sites 10000 --seed 9 \
  --output fixed-gtr-results.json
```

This synthetic experiment retains true ancestral and descendant states and
compares ignoring error with a known independent-error likelihood. Scenarios
include no error, independent error, shared errors across tips and shared blocks.
The last two intentionally violate the fitted observation assumption. It reports
root and joint endpoint log losses/Brier scores and change-probability reliability.
It does not simulate codon S, refit parameters, or test omegaC significance.

## Remaining integration gates

Estimate an observation model from independently separated, larger target-group
data before fitting error-aware GTR and branch lengths. Jointly estimating a
flexible confusion matrix and Q from the same small alignment risks confounding.
The current pilot provides no fitted calibrator. Preserve the source task's
native-GTR N/codon-S split, root identity and state-cache fixes on integration.
Connect to omegaC after issues 2/4 have consistent statistics and joint-category
nulls; propagate AA ancestral uncertainty for translate separately. Neither
pilot accuracy nor fixed-parameter posterior checks complete those gates.
