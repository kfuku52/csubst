# 07: 保存度計算のサイト分割

`substitution_scan._calc_conservation` の `state[tip_ids, :, :]` は、
全tips × sites × statesを一度にコピーする。選択tipsの順序を保ったまま
サイトを分割し、このコピーの目標サイズを16 MiBにする。
1サイトを最小単位とするため、上限は `max(16 MiB, 選択tipsの1サイト分)`。
返却配列と入力配列のサイズはこの上限に含まれない。

小さい入力は従来の計算に入り、大きい入力だけ同じ関数をサイトsliceへ
適用して4つの返却配列を組み立てる。tipsの加算順序、閾値、欠損の扱い、
major stateの同率時の選択を維持する。科学的な既定値は変更しない。

測定対象は局所kernelであり、scan全体の速度や科学的校正を示すものではない。
PGK/PEPCの同梱alignmentを直接one-hot化したtipsを用い、未解決codonを
ゼロにした入力で、アミノ酸20状態とsense codon61状態を比較する。
完全なCLI/ASR処理の代わりになる検査ではない。

## 再現

基準は `624bb8a`。リポジトリrootで次を実行する。

```bash
task_baseline=$(mktemp -d)
git archive 624bb8a | tar -x -C "$task_baseline"
CSUBST_PERF_BASELINE="$task_baseline" python reports/performance_round3_20261023/proposal07.py
```

Python 3.14.7を使用し、数値ライブラリのthread数をimport前に1へ固定する。
入力生成、同等性検査、計測全体は共通 `benchmark_lock()` 内で直列化する。
各実装1回のwarmup後、実行順を交互にした7反復を行う。
小さい入力では1反復を約50 ms CPUに近づけるよう1–50回をまとめ、
1呼び出し当たりへ正規化する。入力の構築は計測区間に含めない。

メモリはNumPyのデータ領域を含む `tracemalloc` の追加割当peakであり、
事前に存在する入力、import済みmodule、allocator/RSS overheadは含まない。
fresh-process RSSは主担当の追加検証対象である。

## 結果

判定: **メモリ削減の有望案**。最終16 MiB版は156 edge caseで612配列が
完全一致し、無効入力3例の例外型も一致した。下記6つのbenchmark入力の
24返却配列も完全一致した。

| 入力 (tips × sites × states) | wall中央値ms 基準→候補 | CPU中央値ms 基準→候補 | 追加割当peak MiB 基準→候補 |
| --- | ---: | ---: | ---: |
| PGK amino 33 × 417 × 20 | 0.676 → 0.600 | 0.558 → 0.552 | 2.52 → 2.52 |
| PGK codon 33 × 417 × 61 | 2.528 → 1.944 | 1.441 → 1.441 | 7.63 → 7.63 |
| PEPC amino 71 × 971 × 20 | 2.006 → 2.118 | 1.696 → 1.883 | 11.53 → 11.53 |
| PEPC codon 71 × 971 × 61 | 7.076 → 5.308 | 6.449 → 4.476 | 34.96 → 17.49 |
| large float32 192 × 8192 × 20 | 36.664 → 27.220 | 34.734 → 26.817 | 128.79 → 17.46 |
| large float64 192 × 8192 × 20 | 63.247 → 39.567 | 50.180 → 36.709 | 253.75 → 17.18 |

大きい入力の追加割当peakはfloat32で86.4%、float64で93.2%低下した。
返却配列と元の入力を含むprocess全体のRSSが同じ割合で下がるという意味ではない。
小さい未分割入力でもwallの変動が見られ、PGKの見かけのwall短縮を
速度向上として解釈しない。PEPC aminoも分割せず計算は同じであり、
その約0.19 msのCPU差には再測定の余地がある。
分割を行う3入力ではCPU中央値も低かったが、本体への適用後の追加検証が必要。

環境はmacOS 27.0.1 / arm64 / 12 CPUs、NumPy 2.5.3、SciPy 1.17.1、
pandas 3.0.5。全反復のwall/CPU範囲はJSONに保存した。

8 MiB版は探索として棄却した。追加割当peakを大きく減らせたが、
PEPCのアミノ酸入力でも分割が発生し、CPU中央値が1.66 msから2.52 msへ増えた。
この実測を受けて目標を16 MiBへ上げ、PGK全入力とPEPCアミノ酸入力を
従来経路に残す。8 MiB版の数値を最終patchの改善率として扱わない。
初期版はbatchなしの7反復で、raw値をJSONの
`discarded_exploration_8MiB` に残した。

## 検査範囲

float32/64、空tips/site/state、欠損、曖昧な分布、同率major state、
非連続slice、read-only配列、read-only memmap、重複/逆順/負のtip ID、
閾値境界/負値/NaN、NaN/Infを含む入力を比較する。
数値配列はdtype/shapeに加えてbytesも照合し、文字列配列は要素を照合する。
無効indexや不整合なordersが起こす例外の型も比較する。

本体とtestsには変更を加えていない。focused pytest、全体test/lint/typecheck、
CLI、native/fallback、installed-wheel、独立した科学的校正は本調査では未実施。
採用する場合は主担当が本体への適用後に必要なlaneを実施する。

試作品は [proposal07.patch](proposal07.patch)、測定値は
[proposal07.json](proposal07.json)、実行コードは [proposal07.py](proposal07.py)。
