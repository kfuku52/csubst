# ID 8: joint endpoint posterior implementation and comparison

2026-09-10。作業用worktreeで実装。元ディレクトリ `${HOME}/repos/csubst` への変更、commit、pushなし。基点は `c4e6d82`。

## 結果

単一枝のjoint endpoint posteriorを実装した。祖先シナリオの列挙は不要。ゼロ長枝で生じていた偽の変化を除去し、合成データのBrier scoreも改善した。一方、実データの一連の処理ではRAMと時間が増加した。旧方式は再現・比較用の既定値として残し、新方式を明示的に選択する。

```bash
csubst search --alignment_file alignment.fa --rooted_tree_file tree.nwk \
  --substitution_posterior joint --endpoint_block_size 64
```

実装・対応範囲は [ENDPOINT_POSTERIORS.md](../../docs/ENDPOINT_POSTERIORS.md)。

## 実データでの処理全体

1 CPU/BLAS thread、ウォームアップ1回後の3回測定の中央値。括弧内は最小–最大。時間は入力読込、イベント生成、期待値、arity=2のCB集計、出力を含み、IQ-TREEの再フィットは含まない。long-tail補正は無効。RAMは各独立プロセスの最大RSS。

| データ | 方式 | 時間 秒 | ピークRAM MiB |
|---|---|---:|---:|
| PGK | 旧・周辺確率の積 | 2.55 (1.87–2.96) | 192.8 (187.3–198.4) |
| PGK | joint、64サイト/block | 11.05 (9.72–16.13) | 475.8 (472.7–503.7) |
| PGK | joint、16サイト/block | 11.86 (9.70–12.56) | 463.5 (455.0–473.6) |
| PEPC | 旧・周辺確率の積 | 5.37 (4.16–5.81) | 443.0 (412.7–448.5) |
| PEPC | joint、64サイト/block | 39.36 (36.97–48.46) | 1455.3 (1438.8–1458.1) |
| PEPC | joint、16サイト/block | 41.41 (36.52–53.70) | 1423.8 (1394.4–1458.2) |

PGKは33 tips・417 codon sites、PEPCは71 tips・971 codon sites。joint/64のRAM比はそれぞれ約2.47倍・3.28倍、時間比は約4.34倍・7.32倍。時間にはばらつきがあり、細かな速度差の優劣は主張しない。

ブロック化は尤度・posteriorの一時配列を制御するが、丸められたASRファイルより細かな非ゼロ確率が多くなり、最終CSRテンソルと集計バッファは増える。16サイト/blockでも処理全体のピークは大きくは下がらなかった。小確率の切捨てはしていない。

初期実装のPGKピークRSSは約798 MiBだった。尤度バッファ再利用と、疎テンソルの投影・要約を枝ごとに処理する変更で約476 MiBに低減。改修前後のjoint CB表と、旧方式の改修前後のCB表が許容誤差1e-10で一致することを別途確認した。

## 精度

- 5節・3状態の全祖先状態列挙との最大絶対差：node marginal 6.11e-16、edge joint 4.44e-16、条件付き期待値 3.89e-16。単一速度と速度0を含む3カテゴリ混合、block size 1/2/64で照合。
- 対称2状態のゼロ長内部枝：旧方式の非対角質量0.5、新方式0。正の短枝でも解析解 `(1-exp(-2t))/2` に一致。
- 以下のBrier scoreは「枝端点が違う確率」と既知の真値0/1の二乗誤差平均。小さいほどよい。真値は既知CTMCから生成した。時刻計測の反復は同じデータを使うため、精度の独立反復とは数えていない。

| 合成データ | 旧Brier | joint Brier | 誤差減少 | joint−旧の95%区間 |
|---|---:|---:|---:|---|
| 63節・2000 sites・20状態 | 0.00561414 | 0.00531857 | 5.26% | [-0.00037823, -0.000212899] |
| 63節・1000 sites・61状態 | 0.00487557 | 0.00470146 | 3.57% | [-0.000234366, -0.000113851] |
| 255節・2000 sites・20状態 | 0.00706054 | 0.00657920 | 6.82% | [-0.00054246, -0.00042022] |

区間はサイト単位の対応のある誤差差から計算した正規近似。サイト内の枝を独立標本とは扱わない。各合成条件は1 seed・1パラメータ設定であり、一般的な生物学的精度やomegaCの偽陽性率の較正を示すものではない。20状態の合成CTMCであり、3Di予測器のベンチマークではない。

## 計算本体のメモリ制御

旧方式には同じモデルから得た高精度のnode marginalsを渡し、新方式にはtip likelihoodsを渡した。したがって新方式に追加されたpruningの計算時間は含まれる。以下の時間は計算本体のみ、RAMは入力読込を含むプロセスのピーク。全イベント保存はせず、両方式とも同じ誤差・質量へ逐次集計する。

| 合成データ | 方式 | 計算本体 秒 | ピークRAM MiB |
|---|---|---:|---:|
| 63節・2000 sites・20状態 | 旧 | 0.132 | 154.4 |
| 63節・2000 sites・20状態 | joint/64 | 3.167 | 166.2 |
| 63節・2000 sites・20状態 | joint/16 | 4.086 | 159.1 |
| 63節・1000 sites・61状態 | 旧 | 0.348 | 194.9 |
| 63節・1000 sites・61状態 | joint/64 | 5.723 | 255.9 |
| 63節・1000 sites・61状態 | joint/16 | 5.984 | 187.3 |
| 255節・2000 sites・20状態 | 旧 | 0.558 | 219.9 |
| 255節・2000 sites・20状態 | joint/64 | 8.201 | 221.0 |
| 255節・2000 sites・20状態 | joint/16 | 11.289 | 202.9 |

block size 16/64の結果は、全実データのCB表と全合成データの誤差配列で一致した。メモリ制御できるのは一時領域であり、最終テンソルの費用とは区別する。

## 実装・検証範囲

- `endpoint.py`: スケーリング付きpruning、カテゴリposteriorでの積分、枝単位のjoint/条件付きprediction、32 MiBの遷移行列cache。
- `endpoint_io.py`: codon S/N・recoding・native 3Diの分離、CSRの一時ファイル経由構築、期待値projection、モデルmanifest、filter時のcache無効化。
- search/sites/scanのテンソル生成とVESMの独自外積計算を新APIへ接続。state plotの旧高速bypassをjointでは使わない。
- 期待値は `sum_c P(c|D) P(parent=a|D,c) P_c(a,d)`。既存の観測カウント由来の枝長再スケールはjointの期待値へ適用しない。
- 従来のchild marginalだけを返す期待値ヘルパーはjoint時に明示エラー。新方式は両軸を保持するprojection APIを使う。
- 全体テスト: **1776 passed, 5 skipped**（1772非process＋4 process）。skipはPyTorchのバージョン3件、gemmi未導入2件。strict nativeテスト7件成功。lint、型検査、diff検査成功。
- 既存環境にpytest-xdistがなかったため、一時的なsystem-site-packages venvへpytest-xdistを追加し、Makefile所定の並列＋process分離テストを実行。ベンチマークは元のPython環境。
- ベンチマーク終了後の変更は入力モデル/型の検査、型注釈、検証と文書のみ。測定対象の推定・集計アルゴリズムは同じ。

## 未検証・残る制約

- codon Q・速度カテゴリは丸められたIQ-TREE report由来。PGKの既存node posteriorとの最大差は0.0045007、平均絶対差は1.24e-6だった。これは同一の未丸めモデルを再現したという保証ではない。新方式は読み取ったパラメータに条件付けて再推定する。
- omegaC・scanの偽陽性率、モデル誤指定、3Di予測誤差、全パイプラインbootstrapは未評価。threshold選択の較正も別課題。
- 複数枝のスコアの積は残る。全枝の共同posteriorやstochastic mappingを実装したわけではない。
- legacyの既定値は変更していない。MG、未知のモデルmodifier、モデル混合、ASC、translate 3Di、ML binarizationはjointで明示的に拒否。

## 再現

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python .github/scripts/benchmark_endpoints.py \
  --workdir /tmp/csubst-endpoint-benchmark --result endpoint_benchmark.json --repeats 3
```

測定環境: macOS 26.6.2 x86_64、Python 3.10.14、NumPy 1.26.4。NumPy OpenBLAS / SciPy MKLはいずれも1 thread。全入力・測定時コードのSHA-256、各反復の数値・実行コマンドは [benchmark.json](benchmark.json)。厳密解照合と出力同値確認は [accuracy.json](accuracy.json)。

状態対ごとのBrier score（非対角の全from/toカテゴリの誤差和）も改善した：

- 63節20状態: 0.01028785 → 0.01019346（0.92%減少）。
- 63節61状態: 0.00661695 → 0.00652066（1.46%減少）。
- 255節20状態: 0.01228468 → 0.01213393（1.23%減少）。

補足環境: SciPy 1.15.2、Rosetta translated=True。実測はこの環境に限定され、ネイティブarm64実行の性能は未測定。
