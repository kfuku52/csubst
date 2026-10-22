# 案05: omega経験的tail変換の列chunk化

**通常の採用は見送り。RAM制約が強い場合だけ条件付き候補。**
既存のnull count行列を列方向に分割してrate・omega・tail countを計算すると、
一時配列は減るが、実PGK/PEPCの対象stageは約2倍遅くなった。
本体・乱数生成・入力fixtureは変更していない。

`_calc_omega_empirical_upper_tail_counts`で、1枚のfloat64作業行列を約4 MiB以下にする。
各chunkの整数exceedance数・valid数を加算し、元と同じcountを返す。
quantile/gamma校正は**全行を保持して列だけ分割**する。行方向の分割は
校正母集団を変えるため使わない。列sliceはviewで、元count行列2枚は保持する。
4 MiBは1枚の作業配列の上限であり、全workspaceやRSSの上限ではない。

## 計測

基準1.16.21 / `88528ce`、Python 3.14.7、macOS arm64、NumPy 2.5.3、
SciPy 1.17.1、pandas 3.0.5、process/BLAS各1。
共通排他lockの下で入力準備と計測を行った。
warmup1組＋順序を交互にした3組。全反復は[JSON](proposal05.json)に保存した。

実入力は同梱固定fitを一時領域へcopyし、joint urn searchの
`any2spe`・Poisson・1000 draw・長尾校正なしで、対象関数に渡る行列を捕捉した。
下表は**その統計変換stageだけ**のwall秒、中央値（最小–最大）。
推論・null標本生成・CLI全体の時間は含まない。

| 入力行列 | 現行 | 列chunk |
| --- | ---: | ---: |
| PGK、1,682行×1,000 draw | 0.01672 (0.01629–0.01699) | 0.03233 (0.03148–0.03304) |
| PEPC、8,446行×1,000 draw | 0.10743 (0.10352–0.14743) | 0.21315 (0.18121–0.24673) |

両方で範囲が分離し、CPU中央値もPGK 0.01663→0.03205秒、
PEPC 0.10474→0.19393秒へ増えた。PGKは311列、PEPCは62列ずつ処理する。
count行列の保持量は各26.9 MB / 135.1 MBで、この入力保持量自体は減らない。
校正・擬似countの専用経路は`omega_null`/`omega_calibration`で既にdrawをblock化しており、
この関数の最適化がすべてのomega検定へ適用されるわけではない。
自動scheduleの100/1000 drawと段階的なactive row選択も維持する必要がある。
今回の実入力比較はcustom単一1000 draw stageで、通常CLI全体の短縮を示していない。

## メモリ試験

6,000行×1,000 drawの**合成**Poisson count行列2枚（計96 MB）を、
同じseedで各独立Python subprocess内に生成した。discardしたwarmup1組と測定3組。
`resource.ru_maxrss`でimport・入力生成・変換を含むprocess peak RSSを測る。
wall/CPU欄だけは標本生成・起動を除いた対象変換stageである。

| 指標 | 現行 | 列chunk |
| --- | ---: | ---: |
| peak RSS MB、中央値（範囲） | 496.8 (486.0–520.9) | 455.2 (440.8–459.1) |
| stage wall秒、中央値（範囲） | 0.08051 (0.06611–0.10019) | 0.15881 (0.13120–0.16034) |
| stage CPU秒、中央値 | 0.07952 | 0.13689 |

RSSは41.6 MB（8.4%）減り範囲が分離したが、実行時間の悪化も範囲が分離した。
MBは10進。入力やimportが占める分もあり、4 MiBという配列上限からRSS低減率を推定しない。
RAM不足を避ける必要がある大規模行列では費用交換になりうるが、
通常の既定経路へ速度改善として導入する根拠はない。

## 同等性・範囲

- 実PGK/PEPCの全比較で、返却する2本のint64配列が厳密一致。
- 84検査で空行/空列、rateのNaN/±inf/0、ゼロ分母、readonly/strided入力、
  scalar・行・列・全行列のalpha、quantile/gamma校正、shape・alpha・校正名エラーを照合。
  成功返値または例外型・文字列が一致した。
- 独立した`Fraction`の有理数tail oracleでも両実装が一致。
  実行時のchunk関数呼出しを追跡し、分割が実際に実行され、
  各校正chunkが元の全行を保つことを確認した。
- 合成RSS比較でも全返値のhashが一致。乱数標本の数や生成順は変更していない。
- Ruff・構文・JSON・差分空白検査を実施。本体未変更のため全test suiteは実行していない。
  全CLI出力・installed wheel・他OS・独立した科学的校正の証拠には数えない。

## 根拠と再現

[NumPyのbasic slicing](https://numpy.org/doc/stable/user/basics.indexing.html)はviewを作る。
[`isclose`](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)は参照値側の相対誤差を使うため、
元のinclusive tail計算と`rtol=8*eps, atol=0`をそのまま呼んだ。
[長尾校正](../../docs/LONGTAIL_CALIBRATION.md)と
[擬似countの契約](../../docs/PSEUDOCOUNT_PVALUES.md)に従い、校正の行母集団を変えていない。
文書は設計上の根拠で、速度・RSSの根拠は上記の比較測定である。

```bash
python reports/performance_round2_20261022/proposal05.py
python -m ruff check reports/performance_round2_20261022/proposal05.py
```

[試作・検証コード](proposal05.py)、[環境・入力hash・全反復・検査](proposal05.json)。
