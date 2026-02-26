#!/usr/bin/env bash
set -euo pipefail
min_sub_pp_levels=(0 0.05 0.2)
stats=(any2spe any2any)
for stat in "${stats[@]}"; do
  for min_sub_pp in "${min_sub_pp_levels[@]}"; do
    tag=$(echo "$min_sub_pp" | tr '.' '_')
    echo "== stat=$stat min_sub_pp=$min_sub_pp =="
    /usr/bin/time -l -p \
    python "/Users/kf/Library/CloudStorage/GoogleDrive-kenji.fukushima@nig.ac.jp/My Drive/psnl/repos/csubst/csubst/csubst" analyze \
      --alignment_file simulate.fa \
      --rooted_tree_file tree.nwk \
      --foreground foreground.txt \
      --omegaC_method modelfree \
      --output_stat "$stat" \
      --calc_omega_pvalue yes \
      --min_sub_pp "$min_sub_pp" \
      --omega_pvalue_niter 1000 \
      --omega_pvalue_rounding round \
      --calibrate_longtail yes \
      --threads 1 \
      --branch_dist no --b no --s no --cs no --bs no --cbs no --cb yes \
      > "analyze_${stat}_min_sub_pp_${tag}.stdout.log" 2> "analyze_${stat}_min_sub_pp_${tag}.stderr.log"
    cp csubst.log "csubst_${stat}_min_sub_pp_${tag}.log"
    cp csubst_cb_2.tsv "csubst_cb_2_${stat}_min_sub_pp_${tag}.tsv"
    python - "$stat" "$min_sub_pp" "$tag" <<'PY'
import pandas as pd, numpy as np, re, sys
stat=sys.argv[1]
min_sub_pp=float(sys.argv[2])
tag=sys.argv[3]
df=pd.read_csv(f'csubst_cb_2_{stat}_min_sub_pp_{tag}.tsv',sep='\t')
rows=[]
for kind,col in [('nocalib',f'pomegaC{stat}_nocalib'),('calib',f'pomegaC{stat}')]:
    if col not in df.columns:
        continue
    p=pd.to_numeric(df[col],errors='coerce').to_numpy(float)
    fin=np.isfinite(p)
    rows.append(dict(output_stat=stat,min_sub_pp=min_sub_pp,kind=kind,n=int(fin.sum()),frac_p_le_0_05=float(np.mean(p[fin]<=0.05)) if fin.any() else np.nan,mean_p=float(np.nanmean(p)) if fin.any() else np.nan,median_p=float(np.nanmedian(p)) if fin.any() else np.nan))
out=pd.DataFrame(rows)
out.to_csv(f'summary_{stat}_min_sub_pp_{tag}.tsv',sep='\t',index=False)
print(out.to_string(index=False))
PY
    python - "$stat" "$min_sub_pp" "$tag" <<'PY'
import re,sys,pandas as pd
stat=sys.argv[1]; min_sub_pp=float(sys.argv[2]); tag=sys.argv[3]
err=open(f'analyze_{stat}_min_sub_pp_{tag}.stderr.log').read()
real=re.search(r'^real\s+([0-9.]+)$',err,re.M)
rss=re.search(r'^\s*([0-9]+)\s+maximum resident set size$',err,re.M)
peak=re.search(r'^\s*([0-9]+)\s+peak memory footprint$',err,re.M)
df=pd.DataFrame([dict(output_stat=stat,min_sub_pp=min_sub_pp,real_sec=float(real.group(1)) if real else None,maxrss_bytes=int(rss.group(1)) if rss else None,peak_mem_bytes=int(peak.group(1)) if peak else None)])
df.to_csv(f'runtime_{stat}_min_sub_pp_{tag}.tsv',sep='\t',index=False)
print(df.to_string(index=False))
PY
  done
done
python - <<'PY'
import pandas as pd,glob
s=pd.concat([pd.read_csv(f,sep='\t') for f in sorted(glob.glob('summary_*_min_sub_pp_*.tsv'))],ignore_index=True)
r=pd.concat([pd.read_csv(f,sep='\t') for f in sorted(glob.glob('runtime_*_min_sub_pp_*.tsv'))],ignore_index=True)
s.to_csv('safe_guard_summary_all.tsv',sep='\t',index=False)
r.to_csv('safe_guard_runtime_all.tsv',sep='\t',index=False)
print('\n=== summary ===')
print(s.sort_values(['output_stat','kind','min_sub_pp']).to_string(index=False))
print('\n=== runtime ===')
print(r.sort_values(['output_stat','min_sub_pp']).to_string(index=False))
PY
