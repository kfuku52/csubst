# Bootstrapping (PEPC foreground) rerun after modelfree engine update

## Scope
- Branch: `codex/bootstrapping`
- Code commit under test: `5e8d5df`
- Dataset: PEPC with foreground file (`PEPC.foreground.txt`, `--fg_format 1`)
- Bootstrap settings: `--bootstrap yes --bootstrap_mode site --bootstrap_niter 30 --bootstrap_seed 7`
- Arity/search: `--max_arity 2 --exhaustive_until 2`

## Reproduction commands
### 1) modelfree rerun (latest engine)
```bash
cd /tmp/csubst_bootstrap_pepc_fg_modelfree_rerun_20260227_172349/modelfree
/usr/bin/time -l -p python /tmp/csubst-bootstrapping/csubst/csubst analyze \
  --alignment_file PEPC.alignment.fa \
  --rooted_tree_file PEPC.tree.nwk \
  --iqtree_state PEPC.alignment.fa.state \
  --iqtree_rate PEPC.alignment.fa.rate \
  --foreground PEPC.foreground.txt \
  --fg_format 1 \
  --omegaC_method modelfree \
  --bootstrap yes --bootstrap_mode site --bootstrap_niter 30 --bootstrap_seed 7 \
  --threads 4 --max_arity 2 --exhaustive_until 2 --calc_omega_pvalue no \
  --cb yes --b no --s no --cs no --bs no --cbs no \
  > run.stdout.log 2> run.time.log
```

### 2) submodel reference (previous run kept as baseline)
```bash
cd /tmp/csubst_bootstrap_pepc_fg_20260227_090423/submodel
/usr/bin/time -l -p python /tmp/csubst-bootstrapping/csubst/csubst analyze \
  --alignment_file PEPC.alignment.fa \
  --rooted_tree_file PEPC.tree.nwk \
  --iqtree_state PEPC.alignment.fa.state \
  --iqtree_rate PEPC.alignment.fa.rate \
  --foreground PEPC.foreground.txt \
  --fg_format 1 \
  --omegaC_method submodel \
  --bootstrap yes --bootstrap_mode site --bootstrap_niter 30 --bootstrap_seed 7 \
  --threads 4 --max_arity 2 --exhaustive_until 2 --calc_omega_pvalue no \
  --cb yes --b no --s no --cs no --bs no --cbs no \
  > run.stdout.log 2> run.time.log
```

## Updated comparison table
| method | source | rows | real_sec | peak_RAM_GiB | fg_total | CI_low>1_all | CI_low>1_fg | CI_low>1_mf | CI_low>1_mg | omega_median | omega_p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| modelfree | rerun_modelfree_latest_engine | 8308 | 125.26 | 3.151 | 20 | 35 | 0 | 0 | 0 | 0.38015 | 1.81610 |
| submodel | baseline_submodel_prev_run | 8308 | 46.91 | 7.176 | 20 | 190 | 6 | 0 | 0 | 0.81230 | 4.25416 |

## Artifacts
- Updated table: `/tmp/csubst_bootstrap_pepc_fg_modelfree_rerun_20260227_172349/compare_summary_foreground_updated.tsv`
- modelfree rerun output: `/tmp/csubst_bootstrap_pepc_fg_modelfree_rerun_20260227_172349/modelfree/csubst_cb_2.tsv`
- modelfree rerun stats: `/tmp/csubst_bootstrap_pepc_fg_modelfree_rerun_20260227_172349/modelfree/csubst_cb_stats.tsv`
- modelfree runtime/RAM log: `/tmp/csubst_bootstrap_pepc_fg_modelfree_rerun_20260227_172349/modelfree/run.time.log`
- submodel baseline output: `/tmp/csubst_bootstrap_pepc_fg_20260227_090423/submodel/csubst_cb_2.tsv`
- submodel runtime/RAM log: `/tmp/csubst_bootstrap_pepc_fg_20260227_090423/submodel/run.time.log`

## Notes
- Foreground independence handling remained active in rerun (`Number of non-independent foreground branch combinations ...: 94 / 8,308`).
- Updated modelfree rerun changed runtime but not fg-level CI-low>1 count (still `0` for `omegaCany2any`).
