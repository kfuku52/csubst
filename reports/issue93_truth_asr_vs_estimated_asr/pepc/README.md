# PEPC truth-ASR vs estimated-ASR omegaC comparison

Artifacts for issue93 follow-up validation.

## Plot
- `omegaC_truth_vs_estimated_scatter_log10_panel_3x2_ocn_ge1_color.png`
  - 3x2 panel, log10 axes.
  - Top row: calibrated `omegaCany2any`, `omegaCany2dif`, `omegaCany2spe`.
  - Bottom row: corresponding `_nocalib` metrics.
  - Color split by estimated OCN threshold: red `OCN>=1.0`, blue `OCN<1.0`.

## Summary table (TSV excerpt)
metric	ocn_column	n	n_ocn_ge_1	n_ocn_lt_1	pearson_r_log10	median_abs_dlog10
omegaCany2any	OCNany2any	2400	21	2379	0.4997380984051721	0.0
omegaCany2dif	OCNany2dif	1774	10	1764	0.4666047375693763	0.0
omegaCany2spe	OCNany2spe	1203	2	1201	0.4738289264460165	0.0
omegaCany2any_nocalib	OCNany2any	8207	21	8186	0.2766398045017383	0.0
omegaCany2dif_nocalib	OCNany2dif	8192	10	8182	0.2111898318701189	0.0
omegaCany2spe_nocalib	OCNany2spe	8278	2	8276	0.1780955247294724	0.0

## Runtime / peak RAM (TSV excerpt)
step	real_sec	max_rss_mb
simulate	55.28	315.84375
analyze_estimated_exh2	28.1	1544.37109375
analyze_truth_exh2	3.94	1544.1640625

## Repro commands
See issue body for full command lines (simulate + analyze estimated + analyze truth).

## Hashes
```
fd3dacf732257eeb1242e02e2515df0d106875263934f43ef3dbc8f9d477ac5e  omegaC_truth_vs_estimated_scatter_log10_panel_3x2_ocn_ge1_color.png
93241d40797c29dccebb22e58aff958212520a787b48d596ed39a937f6abe8a8  omegaC_truth_vs_estimated_log10_panel_3x2_ocn_ge1_color_summary.tsv
3a5412b15de3c0b1bebace7197001d473c3ac358add032909cb3efd350569a4d  runtime_peak_ram.tsv
```
