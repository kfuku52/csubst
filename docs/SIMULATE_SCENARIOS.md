# Simulation scenarios

Use `csubst simulate --simulate_scenario NAME` to start from a scenario
used in the PEPC and PGK evaluations. Explicit options such as
`--num_simulated_site`, `--background_omega`, and `--simulate_asrv` override
the preset. Named scenarios enable `--export_expected_omega` by default.
The `custom` scenario uses the ordinary simulation options and defaults.

```bash
csubst simulate --alignment_file alignment.fa --rooted_tree_file tree.nwk \
  --simulate_scenario epistasis_inverse --simulate_seed 1013
```

## Presets for phylogenetic sensitivity checks

| Scenario | Settings |
| --- | --- |
| `baseline`, `biased` | 520 codons, background N/S mass ratio 0.2. `biased` gives 20% of sites a shared preferred amino acid across foreground branches. |
| `short_alignment`, `short_alignment_bias`, `long_alignment` | 120 or 1,400 codons. The biased short alignment uses the `biased` settings. |
| `short_branches`, `short_branches_bias`, `long_branches`, `long_branches_bias` | Multiply the input tree lengths by 0.3 or 2.5. The biased variants use the `biased` settings. |
| `rate_mixture`, `rate_mixture_bias` | Repeat site rates 0, 0.2, 0.8, and 3. The biased variant uses the `biased` settings. |
| `strong_constraint`, `weak_constraint` | Background N/S mass ratios 0.05 and 0.6, respectively, with neutral foregrounds. |
| `foreground_acceleration`, `foreground_relaxation` | On 20% of the sites, triple foreground branch lengths or set foreground N/S mass to 0.8, respectively. Neither condition selects a shared amino acid. |
| `ecmk07`, `ecmk07_rate_mixture`, `ecmk07_biased` | Use the fitted background Q at its original scale. The default IQ-TREE model is `ECMK07+F+R4`. A complete set of supplied IQ-TREE files determines the fitted model. |

The foreground-effect presets use `--foreground` to specify branches.
To use the same four-rate pattern with other settings, specify
`--simulate_asrv mixture`. To preserve any supported fitted background Q,
specify `--background_omega model`. `--foreground_omega background`
matches the resulting background N/S mass ratio for legacy directional
bias, acceleration, and relaxation.

## Distance-dependent preference

The `epistasis_neutral`, `epistasis_shared`, and `epistasis_inverse`
presets use 1,200 codons and the fitted Q. They provide a neutral control,
a shared-preference control, and distance-dependent convergence.

- For `epistasis_shared`, every focal terminal edge favors the same
  amino acid at each selected site.
- For `epistasis_inverse`, a latent state changes along the tree and
  determines which amino acid each focal terminal edge favors. For
  parent-to-parent distance (d), the retention component is
  (1/(1+d/d_0)). The probability of matching preferences includes a
  chance match between independently drawn amino acids.
- For `epistasis_neutral`, all sites use the background generator.

The two preference scenarios select 50% of sites and multiply the
instantaneous N rates into the preferred amino acid by 8 on the focal
edges. The latent state is read at the parent and held fixed for that
edge. These scenarios represent a dependence of site preferences on an
evolving background. Use `--simulate_epistasis_distance_scale` to set
(d_0) (default 0.18) and `--simulate_epistasis_preference_multiplier`
to change the rate multiplier. Distances use the selected simulation
tree after tree scaling.

Without `--foreground`, the presets sample up to 24 terminal edges at
even intervals in leaf order. Set `--simulate_focal_tips` to change this
limit. With a foreground file, the matched tips define the terminal
edges. Focal lengths default to 0.08 and can be set using
`--simulate_focal_branch_length`. Use `input` to keep their scaled
lengths. The generated foreground IDs are written to
`csubst_foreground_branch.txt` with the current prefix.

`--export_expected_omega yes` integrates the latent preferences and
ancestral states to produce the usual expected-count TSV. True-ASR
export also adds the realized counts. [Output and formulas](SIMULATE_EXPECTED_OMEGA.md)

The branch-length, alignment-length, and selection settings come from the
[PEPC and PGK sensitivity panel](../reports/simulate_sensitivity_20261116/README.md).
