# Joint endpoint: RAM・実行時間の最適化

2026-09-10。前回のjoint実装を基準に、推定式と確率の閾値を変えずに最適化した。
arm64への移行は行っていない。作業worktreeのみ変更し、commit・pushなし。

## 結果

同一のx86_64/Rosetta環境、Python 3.10.14、NumPy 1.26.4、SciPy 1.15.2。
CPU/BLASは1 thread。各条件1回ウォームアップ後、独立プロセスで3回測定した中央値
（括弧内は最小–最大）。条件の順序は反復ごとに逆転。測定中に本タスクのテストは実行していない。

| データ | 方式 | 実行時間 秒 | ピークRAM MiB |
|---|---|---:|---:|
| PGK | 旧marginal | 2.24 (2.13–2.56) | 199.8 (195.1–202.3) |
| PGK | 変更前joint | 11.61 (9.43–15.20) | 496.8 (494.2–508.8) |
| PGK | 最適化joint | 5.09 (4.88–5.25) | 304.9 (293.4–314.3) |
| PEPC | 旧marginal | 4.32 (4.27–4.45) | 442.3 (438.5–449.3) |
| PEPC | 変更前joint | 31.75 (31.67–38.98) | 1418.5 (1413.0–1424.4) |
| PEPC | 最適化joint | 13.56 (13.32–19.18) | 793.5 (787.8–801.3) |

変更前joint比で、PGKは時間56.2%減・RAM38.6%減、PEPCは時間57.3%減・RAM44.1%減。
速度にはばらつきがあるが、両データで最適化後の時間・RAMの範囲は変更前を下回った。
旧marginalよりは依然として時間・RAMを要する。過去の測定値をそのまま比較せず、
今回保存した変更前コードを同じ環境で再測定した。

PGKは33 tips・417 codon sites、PEPCは71 tips・971 codon sites。
入力読込、joint推定、期待値、arity=2のCB集計、出力を含む。既存IQ-TREEファイルを使い、
再フィットは除外。`--b no --s no --cs no --bs no --cbs no --calibrate_longtail no`、
block size 64、既定の出力統計。前回と同じ代表入力・設定を使用した。

## 精度・出力

- PGKのCB表1,682行×39列とPEPCのCB表8,446行×39列は、全列で照合成功。
  TSVの数値列の最大絶対差は両方とも0（保存時の丸め精度）。
- 63節20状態、63節61状態、255節20状態の合成データで、変更前後jointのサイト別
  Brier lossを比較。変化有無の最大差4.17e-17、状態対の最大差5.56e-17。
  前回jointで得られた精度を維持している。これを新たな統計的精度の改善とは主張しない。
- PGKでは別途`--b yes`でもCB表と枝表65行×8列が一致。
  `N_sitewise`の置換文字列も照合した。この追加実行はタイミング集計に含めていない。
- 単体テストで欠損、ゼロ長枝、速度0を含むカテゴリ混合、recoding/3Di、複数block size、
  Cython/NumPy経路、必要なときの全イベント保持を検証した。

元のmarginalとjointの統計的比較は[前回の報告](../endpoint_posterior_20260910/README.md)。
今回の目的はjointの計算費用削減であり、omegaC偽陽性率や全ASRパイプラインの較正は追加評価していない。

## 実装

1. searchで使う観測projectionと枝・サイト別カウントを逐次保存し、全イベントCSRを省いた。
   枝表には各サイトの最大確率の置換だけを保持する。
2. 子node marginalはBLAS行列積で計算し、伝播のための全状態ペア展開を除去。
   `--b no`かつ補助AAストリームが不要な場合、Sは同義グループ内の異なるコドンのみ、
   Nは異なるグループ間の遷移を直接集計する。観測・期待値の両方に適用。
3. 残ったS集計をCythonで融合し、パディングされた配列の反復生成・sumを省いた。
   Cython導入前のPEPCプロファイルではprojection transformに累積約6.2秒、
   全NumPy reduceに合計約4.6秒を要していた（重複を含むため足し合わせない）。
   拡張がない環境用の同値なNumPy経路も残す。

小さい確率を切り捨てる変更はない。大きな不変確率の差し引きではなく、非負項を足し合わせる。
全状態ペアが必要なsites/scan、spe2spe、CS/CBS、正のmin_sub_pp、urn、P値、long-tail、
epistasis、ASRV診断、ASRV学習枝制限では全イベントの経路を維持する。
`--b yes`では保存RAMは削減するが、最大確率の置換を求めるためblock内のイベントは計算する。
上表の時間削減率をこれらの設定へそのまま外挿しない。

適用範囲は[利用ガイド](../../docs/ENDPOINT_POSTERIORS.md)。各モデルのmanifestには
`observed_storage`と`direct_projection`を記録する。

## 検証・再現

- 全体: **1799 passed, 5 skipped**（1795非process＋4 process）。
- strict native: **8 passed**。lint、repository hygiene、文書検査、型検査、diff検査成功。
- スキップは既存のPyTorchバージョン3件、gemmi未導入2件。
- 前回の一時test venvで所定のxdist/process分離テストを実行。性能測定は元のPythonで実行。
- 元環境の古いsetuptoolsは既存pyprojectのlicense形式を受理しなかったため、
  benchmark用拡張は同じPython・Cython・コンパイラで新旧それぞれ直接buildした。
  リンカのrpath重複除去はrepositoryのbuild_extと同じ処理を使用。
  user環境のパッケージ更新や元repositoryの変更はしていない。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python .github/scripts/benchmark_endpoint_optimization.py \
  --baseline-root /tmp/csubst-endpoint-before-20260910 \
  --workdir /tmp/endpoint-optimization-final --repeats 3 \
  --result endpoint_optimization.json
```

保存した変更前source treeと現在のsource treeを使用。比較差分を
[optimization.patch](optimization.patch)に保存したため、現在のコードの使い捨てコピーに
`patch -R -p1`で適用すると、測定時の変更前5ファイルを復元できる。
復元後は同じ環境で拡張をbuildする。作業中のworktreeへ逆適用しない。

[生の測定値・環境・コードと入力のSHA-256](benchmark.json)、
[合成データの精度比較](accuracy.json)、[検証結果](validation.json)。
出力中のhomeディレクトリは`${HOME}`へ正規化した。
