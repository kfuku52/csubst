# Joint endpoint: 逐次CB集計と共有遷移計算

2026-09-10。先行実装と前回の最適化を **b591efc** にコミットし、そのコードを比較基準にした。
本報告の追加実装は作業worktreeの変更。追加commit・pushは行っていない。

## 新旧比較

同一x86_64/Rosetta環境、Python 3.10.14、NumPy 1.26.4、SciPy 1.15.2。
CPU/BLASは1 thread。各条件1回ウォームアップ後、独立プロセスで3回計測。
下表は中央値（最小–最大）。反復ごとに実行順序を反転。最終測定中に本タスクのテストは実行していない。
Cython拡張は新旧で同じものを使用し、arm64への移行や依存パッケージ更新はしていない。

| データ | 方式 | 実行時間 秒 | ピークRAM MiB |
|---|---|---:|---:|
| PGK | 旧marginal（参考） | 2.03 (1.90–2.04) | 196.6 (195.1–204.8) |
| PGK | 今回の変更前joint | 5.43 (5.19–7.21) | 319.1 (312.9–334.2) |
| PGK | 今回の変更後joint | 2.20 (2.19–2.25) | 208.4 (206.4–211.3) |
| PEPC | 旧marginal（参考） | 4.65 (4.40–5.04) | 453.1 (446.9–453.6) |
| PEPC | 今回の変更前joint | 13.07 (12.83–14.17) | 797.9 (771.6–802.6) |
| PEPC | 今回の変更後joint | 5.72 (5.47–5.96) | 386.7 (370.0–394.7) |

変更前joint比で、PGKは**時間59.5%減・RAM34.7%減**、PEPCは**時間56.2%減・RAM51.5%減**。
両データとも最適化後の時間・RAMの測定範囲が変更前を下回った。
PGKは33 tips・417 codon sites、PEPCは71 tips・971 codon sites。

入力読込、推定、期待値、arity=2のCB集計、出力までを測定。IQ-TREE再フィットは含まない。
`--b no --s no --cs no --bs no --cbs no --calibrate_longtail no`、block size 64、既定の出力統計。
比較のため丸め前の表も保存する費用を、新旧とも時間・RSSに含めた。
前回の数値を転記せず、コミットした基準を今回の環境で再測定した。

## 精度・出力の検証

- PGKのCB表1,682行×39列、PEPCのCB表8,446行×39列を丸め前の値で照合。
  観測・期待カウントの最大絶対差はPGK **2.59e-13**、PEPC **1.05e-12**。
  カウント・注釈等は `rtol=1e-10, atol=1e-10` の検査に合格。
- 比率のNaN・±infの位置は一致。CoDやomegaC等の差は、検証済みの分子・分母の差を
  伝播させた誤差の範囲内であることを確認した。一律に全出力の許容誤差を広げていない。
- ほぼ等しいカウントの差を分母にする比率では、加算順序による誤差が増幅される。
  最大相対差はPGK **1.09e-10**、PEPC **1.20e-11**。
  既定4桁のTSVではPGKのOCNCoD **1セル**が122888.9247→122888.9246となり、
  PEPCは全数値セルが一致した。PGKも完全な文字列一致とは主張しない。
- 63節20状態、63節61状態、255節20状態の合成データで、サイト別Brier lossを比較。
  変化有無の最大差 **2.43e-16**、状態対の最大差 **3.20e-16**。前回のjointの推定精度を維持。
- 8状態の鎖で極小の多段遷移を作り、110桁のDecimal行列指数級数とも比較した。
  非可逆モデル、ゼロ生成行列、ゼロ長・短枝・長枝も検証。
- 別途PGKの`--b yes`を実行し、枝表65行×8列（置換文字列を含む）とCB表を照合した。
  この追加検証は上のタイミング集計には含めていない。

小さいposteriorを切り捨てる変更はない。これらの検証は全ASRパイプラインの較正や
omegaC偽陽性率の新しい評価を意味しない。

## 実装した改善

1. **逐次CB集計**：arity=2ではサイトブロックごとに枝ペア行列を加算し、全サイト分の
   projection CSR・ソート用配列・一時ファイルを省く。Sの無効なパディング列も計算しない。
   枝・サイト別カウントと、必要なら枝表用の最大確率の置換を保持する。
2. **キャッシュ解放**：最終期待値の使用後、active reducerだけでなくendpoint cache側の
   参照も削除する。解析が再利用不要と宣言した場合だけ解放し、clade permutationは保持する。
   weak referenceを使ったテストで最後の参照が解放されることを確認。
3. **不要なASR確率読込の省略**：通常のunrecoded joint searchでは、再推定する内部節の
   probability rowsの読込・変換を省く。tip観測、列定義、既存のモデル入力検査は維持する。
   後続処理のための状態配列自体は引き続き保持する。
4. **遷移計算の共有**：`R=I+Q/mu`、`mu=max(-diag(Q))`として、非負の行列累乗を再利用する
   uniformizationを実装。`exp(-mu*t) sum_n (mu*t)^n/n! R^n`を計算し、Poisson tailを
   最小の正の部分和との相対値で検査する。powersは対応状態数で最大8 MiB。
   時間が大きい場合や容量内で収束しない場合は既存のexpmを使用する。
   NumPyの行列積で逐次集計し、別BLASライブラリの初回呼出し費用も避ける。

枝ペア行列は枝数の二乗で増えるため、pair行列・block・カウント・積の一時配列の見積りが
**64 MiB以内**の場合に逐次経路を使う。これはプロセス全体のRAM上限ではない。
大きな木、高次arity、site-filter report、clade permutationはprojection保持経路を使う。
個別イベントが必要な設定は従来のfull-event経路を維持する。
このため上表の削減率を全オプション・全入力サイズに外挿しない。

[利用ガイド](../../docs/ENDPOINT_POSTERIORS.md)に適用範囲を記載。
manifestの`observed_storage`で`pairwise`／`projections`／`full_events`を確認できる。

## テスト・再現

- 全体テスト：**1824 passed, 5 skipped**。その後追加した比較検査・作業域制限の2テストも成功。
- 最終関連テスト：**103 passed**。strict native：**8 passed**。
- lint、repository hygiene、文書検査、型検査、diff検査成功。
- 5スキップは既存のPyTorchバージョン3件、gemmi未導入2件。
- 全体テストは既存の一時test venvでxdist/processを分離して実行。計測は元のPythonで実行。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python .github/scripts/benchmark_endpoint_optimization.py \
  --baseline-root /tmp/csubst-endpoint-b591efc \
  --workdir /tmp/endpoint-streaming-bounded-final --repeats 3 \
  --result endpoint_streaming.json
```

基準source treeは`b591efc`から復元できる。新旧とも同じ環境でビルドした拡張を用意する。
今回のCython sourceはそのコミットから変更していない。
[生の測定値・コマンド・環境・SHA-256](benchmark.json)、
[カウントの精度](count_accuracy.json)、[合成データの精度](accuracy.json)、
[検証結果](validation.json)。ログ内のhomeディレクトリは`${HOME}`へ正規化した。
