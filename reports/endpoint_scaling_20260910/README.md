# PEPC: joint / legacy のarity・CPU・出力条件別比較

**サイト別出力を無効にしても、全条件でlegacyに近い速度にはならない。** 全枝arity 2・枝別表なしではjointはlegacyの1.1〜1.3倍の時間だが、全枝arity 3では約1.5〜2.2倍、前景12枝のarity 4では約3.5〜4.1倍、枝別表ありのarity 2では約3.7倍になった。

今回、計算本体への追加最適化は行っていない。前回の最適化後jointをそのまま測り、性能差の原因を調べた。

## 実行時間

各セルはウォームアップ1回を除く3回の**中央値（最小–最大）**。時間は入力読み込みからCB書き出しまでの秒数で、最大arityまでの累積。`threads`はCSUBSTのプロセス数指定、`blas_threads`は行列ライブラリのスレッド数指定。

| 条件 | threads | blas_threads | legacy / marginal（秒） | joint（秒） | joint / legacy |
|---|---:|---:|---:|---:|---:|
| 全枝・最大arity 2、bなし | 1 | 1 | 4.90 (4.61–5.10) | 5.48 (5.34–5.82) | 1.12× |
| 全枝・最大arity 2、bなし | 2 | 1 | 4.50 (4.49–4.60) | 5.57 (5.38–5.75) | 1.24× |
| 全枝・最大arity 2、bなし | 4 | 1 | 4.35 (4.33–4.55) | 5.64 (5.53–5.76) | 1.30× |
| 前景12枝・最大arity 4 | 1 | 1 | 2.31 (2.30–2.50) | 8.05 (7.96–8.63) | 3.49× |
| 前景12枝・最大arity 4 | 2 | 1 | 2.21 (2.19–2.36) | 9.15 (8.35–9.39) | 4.13× |
| 前景12枝・最大arity 4 | 4 | 1 | 2.39 (2.34–2.58) | 8.33 (7.81–9.66) | 3.48× |
| 全枝・最大arity 3（全探索） | 1 | 1 | 63.93 (57.95–65.87) | 126.81 (125.41–148.63) | 1.98× |
| 全枝・最大arity 3（全探索） | 4 | 1 | 49.42 (49.14–52.34) | 107.52 (107.19–113.67) | 2.18× |
| 全枝・最大arity 2、bなし | 4 | 4 | 4.35 (4.18–4.64) | 5.51 (5.27–5.63) | 1.27× |
| 全枝・最大arity 3（全探索） | 4 | 4 | 40.23 (40.16–41.85) | 60.56 (59.54–60.90) | 1.51× |
| 全枝・最大arity 2、bあり | 1 | 1 | 4.16 (4.13–4.18) | 15.37 (14.94–15.53) | 3.69× |

各条件で実行順を交互に入れ替え、ベンチマーク同士は同時実行していない。合計66本測定＋22ウォームアップ。条件間にはホスト負荷等によるばらつきがあるため、`threads=4`と`threads=1`の小さな時間差を並列化の効果とは解釈しない。

## RAM

ピークRSSの中央値、単位MiB。表には親プロセスのOS計測ピークRSSを使用した。これは子プロセスを含む総RSSではない。

| 条件 | threads / blas_threads | legacy | joint |
|---|---:|---:|---:|
| 全枝・最大arity 2、bなし | 1/1 | 440.5 | 373.2 |
| 全枝・最大arity 2、bなし | 2/1 | 433.8 | 388.0 |
| 全枝・最大arity 2、bなし | 4/1 | 437.7 | 377.6 |
| 前景12枝・最大arity 4 | 1/1 | 298.2 | 540.7 |
| 前景12枝・最大arity 4 | 2/1 | 285.4 | 535.4 |
| 前景12枝・最大arity 4 | 4/1 | 294.0 | 540.0 |
| 全枝・最大arity 3（全探索） | 1/1 | 1464.3 | 1848.6 |
| 全枝・最大arity 3（全探索） | 4/1 | 1430.7 | 1811.8 |
| 全枝・最大arity 2、bなし | 4/4 | 468.4 | 420.5 |
| 全枝・最大arity 3（全探索） | 4/4 | 1467.3 | 1849.1 |
| 全枝・最大arity 2、bあり | 1/1 | 448.3 | 471.3 |

JSONには50 ms間隔でサンプリングしたプロセスツリーRSSの合計も保存した。この値はピークを取り逃す可能性があり、複数プロセスが共有するページは重複計上されうる。親プロセスのOS計測値と混ぜて比較していない。88run中87runでは子プロセスを観測せず、CPU 1のlegacyの1runだけ最大1子プロセスを観測した（種類は記録していない）。CPU 4指定の全runでは子プロセス0だった。

## 実際に計算した条件

- 同梱PEPC: 71 tips、971 codon sites。既存IQ-TREE出力を両方式で再利用し、IQ-TREEの再推定時間は含めない。
- 全条件で `--s no --cs no --bs no --cbs no --calibrate_longtail no --endpoint_block_size 64 --random_seed 8`。`--b`は「bあり」の行だけyes。ほかの解析オプションは既定値。
- 全枝・最大arity 2: legacyは8,308組、jointは8,446組。
- 全枝・最大arity 3: `--exhaustive_until 3`。arity 2に加え、legacyは307,432組、jointは315,740組のarity 3を計算。
- 件数差は、legacyでは根の祖先状態がなく、根直下の枝を含む組み合わせに除外制約が加わるため。全枝arity 3ではjointの対象が2.70%多い。legacyの対象はすべてjointにも含まれる。計算法間の結果そのものは同値とはみなさない。
- 前景12枝: `PEPC.foreground.txt`の12 tip名を用い、各tipに別のlineage IDを与えたTSVを生成。`--fg_format 2 --exhaustive_until 1 --cutoff_stat OCNany2spe,0`。両方式ともarity 2/3/4でそれぞれ**64/200/406組**を計算し、枝IDの集合も同一。
- 前景指定なしの予備実行はarity 3へ進めなかったが、追加調査でlegacyにも共通する前景候補フィルタの問題と分かった。jointによる高次信号消失を示す結果ではない（[検出の追加調査](../endpoint_detection_20260910/README.md)）。その後この問題は[修正・通常CLIで再検証](../search_candidate_fix_20260910/README.md)した。高次の負荷比較には上記の明示的な全探索・前景条件を使用した。全枝arity 4の全探索や、このMac以外の環境は測定していない。

## CPU指定とボトルネック

`--threads 4 --blas_threads 1`の全runで、子プロセスは観測されなかった。jointの事前集計済みprojectionを使う経路は逐次処理で、既存の高次・距離計算スケジューラもこの負荷では1 workerを選ぶ。後半のrunで確認したOpenBLAS/MKLの実設定も1スレッド、平均CPU使用量は約0.95コアだった。

`--threads 4 --blas_threads 4`では、OpenBLAS/MKLの両方が実際に4スレッド設定となった。全枝arity 3の平均CPU使用量の中央値はlegacy 1.26コア、joint 1.97コア。jointは60.56秒まで短縮したが、legacy 40.23秒より約1.5倍遅い。4コアを常時使う構造ではない。

反復測定とは別に、1 CPU・BLAS 1で両方式をcProfile計測した（各条件1回。速度表の集計には含めない）。[CSUBST関連関数の記録](profiles.json)から以下を確認した。

| 条件・処理 | legacyの累積秒 | jointの累積秒 | 解釈 |
|---|---:|---:|---|
| 全枝arity 3: `_calc_dense_arity3_projection_products` | 6.25 | 61.33 | 大きな行列による高次集計が差の中心 |
| 全枝arity 3: TSV書き出し `write_dataframe` | 16.52 | 17.11 | 書き出しの差では説明できない |
| 前景12枝arity 4: `prep_state` | 0.45 | 6.79 | legacyは23/141ノードだけを読み込めるが、jointは全tipで条件付けして推論・集計用データを作る |
| 全枝arity 2・bあり: `prep_state` | 1.59 | 13.07 | b表の置換文字列用に個別イベントが必要となり、直接集計する高速経路が使われない |

profileの累積時間は入れ子の呼び出しを含むため、重複する関数の値を足し合わせない。

次の最適化対象は、高次集計での不要なゼロ列・パディングの除去と、同じ集計データを複製せずに処理を並列化すること。前景検索では全tipでの条件付けを維持しつつ、対象外の枝のイベント出力・集計作成を省けるかを検討する。b表では最大確率イベントの取得を直接集計に統合する余地がある。これらの追加高速化は未実装・未測定。

## 数値・検証

- [数値照合結果](validation.json): 同一計算法の反復・CPU・BLAS間で106表を照合し、さらにb表オン/オフで2表を照合した。全108比較が通過。枝ID、注釈、NaN・正負Infの位置も一致。
- 観測・期待カウントの最大絶対差は **6.40e-14**（検査基準 `rtol=atol=1e-10`）。
- 比率は一律に許容誤差を緩めず、対応する分子・分母の差から誤差を伝播させて検査。最大相対差は **6.16e-10**。小さい分母を使う `OCNCoD` では最大絶対差 **6.60e-4** があり、文字列・丸め後の完全一致は主張しない。
- 比率の比較ヘルパーは、同一のマスク済みゼロ（分母NaNを含む）に不要な誤差伝播を行って失敗するケースを修正。異なる出力には従来の検査を適用し、ゼロを誤った非ゼロへ変えた場合は拒否する回帰テストを追加した。計算本体・許容誤差は変更していない。
- 計測した計算本体・拡張・入力ファイルのSHA-256が、現行ソースと保存したbaselineに一致することを検証した。
- 比較ヘルパーのテスト: **2 passed**。`make lint`（ruff、repository hygiene、documentation check）通過。計算本体を変えていないため、本調査では全テストの再実行はしていない。

## 再現条件とファイル

- Apple M2 Max、12 CPU、64 GiB RAM、macOS 26.6.2。実行Pythonはx86_64/Rosetta、Python 3.10.14、NumPy 1.26.4、SciPy 1.15.2、pandas 2.2.3。
- BLAS: NumPyのOpenBLAS 0.3.23.devとSciPy側のMKL 2020.0.4。異なるBLAS・ネイティブARM・Linuxの性能へは一般化しない。
- legacy baseline: commit `b591efc`直後に保存したソースと互換ビルド済み拡張。joint: 前回の最適化後の未コミットソース。全source/input SHA-256は各JSONに記録。
- 計測区間には通常のTSVと照合用の丸め前pickleの書き出しを両方式で同様に含む。Pythonプロセス起動と計測後のライブラリ情報取得は含まない。JSONの `process_wall_seconds` はこれらの補助処理も含む診断用値で、上の速度表には使用しない。後半のrunのみ追加のCPU秒・ライブラリ設定情報を持つ。
- 個々のコマンド・反復値・範囲: [arity 2](matrix.json)、[前景arity 4](foreground.json)、[全枝arity 3](exhaustive.json)、[BLAS 4](blas.json)、[枝別表あり](branch.json)。[代表ログ](logs/)も保存。

```sh
# baseline-rootには、同じ環境で実行できるb591efcのソース/拡張を指定する。
python .github/scripts/benchmark_endpoint_scaling.py \
  --baseline-root /path/to/b591efc \
  --workdir /tmp/endpoint-scaling-repro --result /tmp/endpoint-scaling-repro.json \
  --scenarios pair foreground4 exhaustive3 pair_b --cpus 1 2 4 --blas 1 --repeats 3
python .github/scripts/benchmark_endpoint_scaling.py \
  --baseline-root /path/to/b591efc \
  --workdir /tmp/endpoint-scaling-repro-blas --result /tmp/endpoint-scaling-repro-blas.json \
  --scenarios pair exhaustive3 --cpus 4 --blas 4 --repeats 3
python .github/scripts/verify_endpoint_scaling.py \
  --reports /tmp/endpoint-scaling-repro.json /tmp/endpoint-scaling-repro-blas.json \
  --result /tmp/endpoint-scaling-repro-validation.json
```

測定時の正確な条件集合は各JSONに記録されている。上の再現例はCPU 2の全枝arity 3等も追加して実行する。照合用pickleは、このスクリプトがローカルで生成したものだけを使う。
