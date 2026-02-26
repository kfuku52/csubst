# Issue #94 artifacts (simulate bug-fix comparison)

Dataset: PGK
Seed: 20260226

Commands:
- before_default: `python csubst/csubst simulate --alignment_file csubst/dataset/PGK.alignment.fa --rooted_tree_file csubst/dataset/PGK.tree.nwk --foreground csubst/dataset/PGK.foreground.txt --simulate_seed 20260226 --threads 1`
- after_omega0.2: `python csubst/csubst simulate --alignment_file csubst/dataset/PGK.alignment.fa --rooted_tree_file csubst/dataset/PGK.tree.nwk --foreground csubst/dataset/PGK.foreground.txt --simulate_seed 20260226 --background_omega 0.2 --threads 1`
- after_default: `python csubst/csubst simulate --alignment_file csubst/dataset/PGK.alignment.fa --rooted_tree_file csubst/dataset/PGK.tree.nwk --foreground csubst/dataset/PGK.foreground.txt --simulate_seed 20260226 --threads 1`

Artifacts and SHA256:
- `reports/bugfix_issue94_20260226/simulate_diff_vs_before.png`: `d898db7feabf0b9c154b95730996d6216f59d7fa878cbb0991edf9011055b06f`
- `reports/bugfix_issue94_20260226/simulate_runtime_ram.png`: `97e6d9fbec8465f5553bb9b8946b3bb84f7716d49215a6b7bdee604d11aa8a43`
- `reports/bugfix_issue94_20260226/scenario_summary.tsv`: `a06a8ebd00f2a4d1313d9ffa06168b97e31f784afc92c689bcb36b54174a21ca`
- `reports/bugfix_issue94_20260226/pairwise_vs_before.tsv`: `a98d408b8934364bb86c3d5198cc2b7ad58db92af34e2c35c34fb952ee79d33e`
- `reports/bugfix_issue94_20260226/runtime_ram.tsv`: `1b8cfc1e398cec62857ed1d22c88cfb160760874f3d3fef38ee6bd83e5b74f6d`
- `reports/bugfix_issue94_20260226/comparison_summary.json`: `825f36f59cb37c5baeed6ee478bbe6ef85e5e4847b4f9fe825f8bb3cae1a2c1a`

Raw run directory (not committed):
- `/tmp/csubst_sim_compare_20260226_163916/runs2`