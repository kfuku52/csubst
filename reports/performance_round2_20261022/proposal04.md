# 案04: 解析的検定の変更されない部分木を再利用

**判定: 条件付き候補として保留。** siteごとにまとめて処理する版はPGKで改善したが、
実PEPCの測定範囲が重なった。通常scanでは使わないopt-in解析的検定だけの案であり、
CLI全体の改善は未確認。製品コードは変更していない。

`EndpointEnrichment.log_likelihoods`のうち、対象枝を子孫に含まない部分木は
state contrastに依存しない。固定model・profile・私有tip snapshotから
同じshape・子順序でscaled partialとscaleを準備し、対象枝の祖先だけ再計算した。
行列積のshapeを変えず、対象枝のnull atomでも既存の再正規化を繰り返して丸めを保った。

候補順のままcacheするとPEPCで再構築が多いため、同じ案の改良として内部ではsite順に処理し、
結果を元の行順へ戻した。snapshot構築回数は194→94になった。
公開メソッドをこの私有backendへ置き換える変更は今回行っていない。

## 測定

基準1.16.21 (`88528ce`)。PGK/PEPCの同梱fixed fitを一時領域へコピーし、
`scan_match=any2spe,spe2spe`で実際に選ばれた候補と解析的target枝を使用した。
Python 3.14.7 / macOS arm64、詳細は[JSON](proposal04.json)。
BLAS 1、warmup 1回後に3回、実行順を交互にして測定。
共有lockで他案のCPU計測と排他した。各試作のcacheは計測開始時に空で、
snapshot構築・copy・内部整列を含む。入出力、joint inference、transitionのcold準備、
TSV出力は含まない。単位は秒、wall中央値（最小–最大）。

| 条件 | 現行 | 試作 | 評価 |
| --- | ---: | ---: | --- |
| PGK実候補23件/12site・site順処理 | 0.3406 (0.3311–0.3738) | 0.2777 (0.2753–0.3088) | 18.5%短縮、範囲分離 |
| PEPC実候補198件/94site・site順処理 | 3.5589 (3.2240–4.9598) | 3.1680 (3.0250–4.5228) | 範囲重複、改善未確定 |
| PGK人工同site20contrast | 0.1912 (0.1872–0.2058) | 0.1367 (0.1339–0.1397) | 28.5%短縮、範囲分離 |
| PEPC人工同site20contrast | 0.2882 (0.2872–0.3284) | 0.2146 (0.1992–0.2365) | 25.5%短縮、範囲分離 |

人工20contrastは、実候補1siteのtipとtargetを使い、20種類のdestinationを調べる
償却条件の試験である。新しい独立データや通常の候補分布を表してはいない。

初期の候補順版はPGK 0.2459→0.2264秒、PEPC 2.7313→3.5368秒で、
どちらも測定範囲が重なる。これと改良版の異なるbaseline同士を比較しない。
全反復のCPU時間・wall時間はJSONに保存した。

site順版の追加保持配列はPGK 7,931,808 bytes、PEPC 32,969,304 bytes。
cacheの**配列payload**上限32 MiBであり、Python dict/containerや一時workspaceは別。
RSSは未測定。単一siteだけ保持する実装による削減は今回は測定していない。

## 同等性と採用条件

- 実候補、人工20contrastとも、全null/alternative log尤度・P・log-eがbytes一致。
- missing/ambiguity、0 rate、0 branch、複数target、partial participation、infinite tiltを含む
  648合成条件でbytes一致。共有hidden stateを含む16観測patternは独立した全状態列挙とも一致
  (`rtol=2e-13, atol=1e-16`)。
- 葉key不足、root branch、negative/NaN/shape不正tipの5条件で同じValueErrorを確認。
- 既存`test_scan_analytic.py`を、既存entrypointのkey/branch検証を残したadapterで
  試作数値backendへ差し替え、**7 passed、0 skipped**。

採用する場合も公開entrypointの葉key→branch→葉値という検証順を維持し、
その後で固定fit/profileと私有emission snapshotを準備する必要がある。
`PreparedSite`をそのまま公開メソッドへ置換した場合の、複数同時invalid入力の失敗順互換は
今回証明していない。共有cacheをannotation runの外へ持ち出したり、変更可能tipを
object identityだけで識別したりする設計は想定しない。
PEPCの安定した実行全体改善とRSS費用を確認してから判断する。

## 再現・資料

```bash
python reports/performance_round2_20261022/proposal04.py
python reports/performance_round2_20261022/proposal04.py --verify-only
python -m ruff check reports/performance_round2_20261022/proposal04.py
```

`--grouped-only`は同じscriptの初期測定JSONにsite順版だけを追加するための補助。
入力hash・元関数hashをJSONに記録する。製品full suite・installed wheel・校正simulationは未実行
（製品コードを変更せず、限定した試作と既存解析的契約だけを検証したため）。

[Kobert et al. 2017](https://pubmed.ncbi.nlm.nih.gov/27576546/)の冗長partial計算の再利用を参考にしたが、
本案はsite-repeat圧縮の実装ではない。
混合log尤度は既存の[SciPy logsumexp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html)
のまま保持した。科学的契約は[SCAN_ANALYTICAL](../../docs/SCAN_ANALYTICAL.md)に従う。
