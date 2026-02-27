#!/usr/bin/env bash
set -euo pipefail
models=(hypergeom poisson poisson_full nbinom)
stats=(any2spe any2any)
for stat in "${stats[@]}"; do
  for model in "${models[@]}"; do
    echo "== stat=$stat model=$model =="
    /usr/bin/time -l -p \
    python "/Users/kf/Library/CloudStorage/GoogleDrive-kenji.fukushima@nig.ac.jp/My Drive/psnl/repos/csubst/csubst/csubst" analyze \
      --alignment_file simulate.fa \
      --rooted_tree_file tree.nwk \
      --foreground foreground.txt \
      --omegaC_method modelfree \
      --output_stat "$stat" \
      --calc_omega_pvalue yes \
      --omega_pvalue_null_model "$model" \
      --omega_pvalue_nbinom_alpha auto \
      --omega_pvalue_niter 1000 \
      --omega_pvalue_rounding round \
      --calibrate_longtail yes \
      --epistasis_site_metric auto \
      --epistasis_beta 0 \
      --threads 1 \
      --branch_dist no --b no --s no --cs no --bs no --cbs no --cb yes \
      > "analyze_${stat}_${model}.stdout.log" 2> "analyze_${stat}_${model}.stderr.log"
    cp csubst.log "csubst_${stat}_${model}.log"
    cp csubst_cb_2.tsv "csubst_cb_2_${stat}_${model}.tsv"
    python - "$stat" "$model" <<'PY'
import pandas as pd, numpy as np, sys
stat=sys.argv[1]; model=sys.argv[2]
df=pd.read_csv(f'csubst_cb_2_{stat}_{model}.tsv',sep='\t')
rows=[]
for kind,col in [('nocalib',f'pomegaC{stat}_nocalib'),('calib',f'pomegaC{stat}')]:
    if col not in df.columns: continue
    p=pd.to_numeric(df[col],errors='coerce').to_numpy(float)
    fin=np.isfinite(p)
    rows.append(dict(output_stat=stat,null_model=model,kind=kind,n=int(fin.sum()),frac_p_le_0_05=float(np.mean(p[fin]<=0.05)) if fin.any() else np.nan,mean_p=float(np.nanmean(p)) if fin.any() else np.nan,median_p=float(np.nanmedian(p)) if fin.any() else np.nan))
out=pd.DataFrame(rows)
out.to_csv(f'summary_{stat}_{model}.tsv',sep='\t',index=False)
print(out.to_string(index=False))
PY
    python - "$stat" "$model" <<'PY'
import pandas as pd,re,sys
stat=sys.argv[1]; model=sys.argv[2]
err=open(f'analyze_{stat}_{model}.stderr.log').read()
real=re.search(r'^real\s+([0-9.]+)$',err,re.M)
rss=re.search(r'^\s*([0-9]+)\s+maximum resident set size$',err,re.M)
peak=re.search(r'^\s*([0-9]+)\s+peak memory footprint$',err,re.M)
df=pd.DataFrame([dict(output_stat=stat,null_model=model,real_sec=float(real.group(1)) if real else None,maxrss_bytes=int(rss.group(1)) if rss else None,peak_mem_bytes=int(peak.group(1)) if peak else None)])
df.to_csv(f'runtime_{stat}_{model}.tsv',sep='\t',index=False)
print(df.to_string(index=False))
PY
  done
done
python - <<'PY'
import pandas as pd,glob
s=pd.concat([pd.read_csv(f,sep='\t') for f in sorted(glob.glob('summary_*_*.tsv'))],ignore_index=True)
r=pd.concat([pd.read_csv(f,sep='\t') for f in sorted(glob.glob('runtime_*_*.tsv'))],ignore_index=True)
s.to_csv('null_model_summary_all.tsv',sep='\t',index=False)
r.to_csv('null_model_runtime_all.tsv',sep='\t',index=False)
print('\n=== summary ===')
print(s.sort_values(['output_stat','kind','null_model']).to_string(index=False))
print('\n=== runtime ===')
print(r.sort_values(['output_stat','null_model']).to_string(index=False))
PY
