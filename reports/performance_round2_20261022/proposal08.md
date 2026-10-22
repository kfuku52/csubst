# 案08: unit supportのmembership準備とsegmented maximum

**通常採用は見送り。** 枝membershipを配列化しても、実PGK/PEPCでは安定した
短縮がない。大規模な人工membershipでだけ局所的な効果が確認できた。
本体コード、既定値、候補選択、閾値、校正nullは変更していない。

## 試作

`_scan_unit_support_plan`の結果から、unitごとの区間、圧縮枝IDと復元indexを
1割当て内だけ保持する。候補イベントのpandas `groupby.sum`は従来のままとし、
支持する枝の値を並べた後、`np.maximum.reduceat`でunitごとの最大値を計算する。
支持値の和・平均、unit順序、lineage挿入順、枝ID文字列の順は既存規則を保持する。

[NumPy `reduceat`](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.reduceat.html)は
空区間を自然には表さないため、空unitを除いた開始位置だけに適用する。
[`unique(return_inverse=True)`](https://numpy.org/doc/stable/reference/generated/numpy.unique.html)で
枝IDを圧縮し、負ID・欠番・巨大IDがあっても最大IDに比例する配列を作らない。
これは方法の根拠であり、速度は以下の実測で判断した。

準備済みの私有planだけに試作を適用し、未準備の公開呼出しは現行関数を通る。
再利用は一定のunit membershipをもつ1割当て内に限定する。
可変な外部DataFrameをidentityだけでキャッシュする仕組みは導入しない。

## 測定

基準は1.16.21 / `88528ce`。macOS arm64、Python 3.14.7、NumPy 2.5.3、
SciPy 1.17.1、pandas 3.0.5。BLASはimport前に1スレッドへ制限し、
全案共通の排他ロックで準備・検証・計測を直列化した。
同梱PGK/PEPCの固定fitを一時領域にコピーし、通常joint scan・校正none・seed8から
実際の全support呼出しを捕捉した。

1 warmup + 5交互反復。各batchで空のprepared planから開始するので構築費用を含む。
以下は**support reducer batchのみ**のwall秒、中央値（最小–最大）。
CLI全体や候補固定校正全体の時間ではない。

| 入力・batch | 現行 | 試作 |
| --- | ---: | ---: |
| PGK、198呼出し、2 units・16 memberships | 0.02195 (0.02120–0.02435) | 0.02521 (0.02336–0.02678) |
| PEPC、546呼出し、9 units・15 memberships | 0.06311 (0.06047–0.06536) | 0.06714 (0.06279–0.07107) |
| 人工、40呼出し、200 units・20,000 memberships | 0.12776 (0.12419–0.13159) | 0.04663 (0.04224–0.04699) |

実2入力の範囲は重なり、中央値も改善しない。実PGKの最大unitは15枝、PEPCは5枝と小さく、
追加のgather・mask・reductionの費用を回収できなかった。
CPU中央値もPGK 0.02193→0.02504秒、PEPC 0.06287→0.06643秒で減少しない。

人工条件は2,000枝IDを200 unitに繰り返し配置したoverlapの多いstress入力で、
壁時計時間は63.5%減り、範囲が分離した。独立した生物学的標本でも、
校正nullを満たす非重複clade構成の実証でもない。この結果を通常scanへ外挿しない。

追加保持配列はPGK432 B、PEPC576 B、人工340,800 B。
これは元planに加えた配列の`nbytes`合計であり、Python object、一時配列、RSSを含まない。
候補ごとの作業配列もmembership数に比例して増える。RSS低減は測定・主張しない。

## 検証と範囲

- 実入力の全198/546呼出しで、全返値・文字列・キー・Python型を厳密比較した。
  各反復も全返値を現行の結果と照合した。
- 計測外counterで、準備済み経路とsegmented maximumがPGK198回・PEPC546回、
  人工40回それぞれ実行されたことを確認した。
- 205境界検査に成功。空unit/空membership、重複membership、負/欠番/巨大枝ID、
  lineageのCSV escaping・日本語・改行、空イベント、NaN/0/負/無限質量、
  compensated sumが必要な重複イベント、閾値の`nextafter`/NaN/±inf/不正文字列を含む。
  未準備経路、target不正、イベント列欠落、支持unit ID不正の返値/例外も比較した。
- 手計算の独立oracleで、重複membership、unitごとの最大、枝の重複除去と
  lineage集計を検査した。
- Ruff、構文、JSON、空白・パス検査に成功。本体未変更のため製品全suite、
  並列worker、fallback、installed wheel、他OS、科学的校正は実行していない。

枝イベントの補償付き集約を省略すれば別の速度が出る可能性はあるが、
それは本案の同等性条件を満たさない。通常の小さなunitには現行の単純な処理を維持する。

## 再現

```bash
python reports/performance_round2_20261022/proposal08.py
python -m ruff check reports/performance_round2_20261022/proposal08.py
```

[試作・検証コード](proposal08.py)、[全反復・入力hash・counter・境界結果](proposal08.json)。
全入力・CLI出力・ログは新しい一時領域で生成し、元fixtureへ書き込んでいない。
