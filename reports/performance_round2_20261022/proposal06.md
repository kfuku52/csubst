# 案06: 共通maxT参照の一括tail検索

**判定: 大量候補向けの条件付き候補。** 同じmaxT参照を候補ごとに検証・走査する代わりに、
一度検証して独立した整列配列を作り、全候補とglobal statisticをまとめて検索する。
同梱入力ではこの最終集計は数ms以下であり、全CLIの主要ボトルネックとは確認できない。
本体、公開scalar API、候補固有の参照分布、null試行数、乱数、選択規則は変更していない。

## 数値同等性

現行は`reference >= observed`または`np.isclose(reference, observed, rtol=1e-12, atol=1e-12)`を
上側tailに含める。単純に`observed - (atol + rtol * abs(observed))`へ二分探索するだけでは、
減算の丸めによって境界上の値を過剰・過少に数える。
試作は検索位置の前後を**同じ現行predicate**で確認し、違いがあるqueryだけ、
そのpredicateを直接用いた二分探索で補正する。近似tieの幅は変えない。

- 4,517 query/不正入力比較で、全p値の厳密配列・bytes一致を確認した。
- 極小値、最大有限値、負値、0、NaN、±infinity、境界の前後4 ulp、重複値、exact/Monte Carloを検査。
- 単純な閾値検索が誤る59参照条件を検出し、118 queryで境界補正を実際に実行した。
- 空/2次元/NaN/+infinity/非数値参照、試行数不一致で例外型と文言を比較した。
  orchestrationの「各configurationは必ず1値」という前段エラーの順序も別途保持した。
- 外部参照配列を変更しても準備済みsnapshotは変わらず、保持配列は書込禁止。
- 実PGK/PEPCのfull-scan 20成功試行から、共通参照、候補スコア、global scoreを捕捉し、
  現行が計算した全maxT p値とのbytes一致も確認した。

公開scalar関数はそのままにし、私有のfloat64一次元batch境界として導入する条件で評価した。
これは新しい統計検定・科学的校正の証明ではない。

## 測定

基準`88528ce` / CSUBST 1.16.21。macOS arm64、Python 3.14.7、
NumPy 2.5.3、SciPy 1.17.1、pandas 3.0.5。BLASはimport前に1へ制限し、
入力準備・捕捉・計測は共通排他lock内で行った。
warmup 1組＋順序交互の5反復。短い処理は同じbatchを複数回実行し、
1 batchの秒へ正規化した。各batchで参照の変換・検証・sortをやり直すため、準備費用を含む。

以下は**最終maxT tail集計stage**のwall time、ms、中央値（最小–最大）。
permutationの再探索や全CLI時間ではない。

| 入力 | 候補+global query / null draw | 現行 | 試作 |
| --- | ---: | ---: | ---: |
| 実PGK | 13 / 20 | 0.1948 (0.1883–0.2054) | 0.0412 (0.0398–0.0445) |
| 実PEPC | 104 / 20 | 2.0549 (1.4642–2.8470) | 0.1424 (0.0632–1.5557) |
| 合成・既定draw規模 | 103 / 1,000 | 7.2291 (3.7492–9.9559) | 0.1667 (0.1530–0.1782) |
| 合成・大量候補 | 10,000 / 1,000 | 384.9711 (356.6030–468.3305) | 1.1311 (1.1060–1.3641) |
| 合成・多数draw | 2,000 / 10,000 | 494.4920 (475.8690–526.3530) | 0.8446 (0.7592–0.9093) |

PGKと全合成条件では範囲が分離した。PEPCはwall範囲が一部重なるので、
wall短縮率は確定しない。PEPC CPU中央値は1.8525→0.1108 msで範囲が分離した。
合成入力は新しい生物学的標本ではなく、問い合わせ数と参照数のサイズ試験である。

整列参照の追加保持は実入力160 bytes、合成で8,000/80,000 bytes。
返値はquery数×8 bytesで、ほかにquery数に比例する閾値・index・境界mask作業配列を使う。
これは配列サイズでありpeak RSSではない。RSSの低減は測っておらず主張しない。

20試行の実入力では節約される絶対時間が小さく、既定1,000試行の全scanを測った結果ではない。
大量候補を出すfull_scanでは検討価値があるが、候補ごとに異なる参照を毎回sortする用途には適用しない。
既定経路への優先採用より、局所負荷が大きい条件での導入を推奨する。

## 根拠・再現・範囲

[NumPy searchsorted](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html)は
整列配列に対して二分探索を行う。
[NumPy isclose](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)は
第2引数に依存した非対称の許容幅を使うため、引数順と浮動小数点の境界判定を維持した。
これらは実装設計の根拠で、上表の速度はローカル測定で判断した。
[scan校正契約](../../docs/SCAN_CALIBRATION.md)の分母・inclusive tailは維持する。

```bash
python reports/performance_round2_20261022/proposal06.py
python -m ruff check reports/performance_round2_20261022/proposal06.py
```

[再現ハーネス](proposal06.py)、[全反復・入力hash・境界検査](proposal06.json)。
入力は一時領域へcopyし、生成ログ・実行出力は保存対象外。
Ruff・構文・JSON・パス・空白検査を実行した。本体未変更なので全suite、wheel、
他OS、parallel/fallback backendの検証は本案の証拠に含めない。
