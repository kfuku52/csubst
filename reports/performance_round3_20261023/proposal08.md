# 案08: 疎な枝積が空になった後の有限行の乗算を省略

**判定: 不採用。** 人工的な早期ゼロの入力ではCPUを43–47%削減したが、
実データの呼出しで有効になることを確認できなかった。既存実装のままとする。
[試作patch](proposal08.patch) は本体へ適用していない。

`aggregate_sparse_branches` の積は、選択されたCSR行を順に乗算する。
試作は中間積が空で、次の行の明示値がすべて有限の場合にだけ乗算を省く。
`getrow()` を必ず呼び、無効な枝IDの拒否を維持する。
NaN/Infを含む行は従来どおり乗算し、暗黙のゼロとの積を保持する。
入力dtype、重複IDの反復適用、空選択、未知のoperation、和の処理は変更しない。

実際の呼出し元は `main_sites.plot_state` の和・積集約だけだった。
`should_plot_state` は `intersection` でのみ真となるため、set式の評価は
この最適化を利用しない。PGK/PEPCの同梱alignmentと固定IQ-TREE fitを
新しい一時ディレクトリへコピーし、実際の `sites --mode intersection`
から得られるN/Sテンソルを `plot_state` の入口で捕捉した。
枝1–16の明示的な16枝選択と、その先頭2枝を測定した。
いずれも最終行の前に積が空にならず、shortcutの実行回数はゼロだった。
PEPC Sの16枝積だけは最後の行で空になった。

| 実データkernel | CPU中央値: 基準→試作（ms） | 試作/基準 |
| --- | --- | --- |
| PGK N、2枝 | 0.1164→0.1373 | 1.180 |
| PGK N、16枝 | 4.7101→4.2834 | 0.909 |
| PGK S、2枝 | 0.0565→0.0544 | 0.963 |
| PGK S、16枝 | 1.8214→1.7726 | 0.973 |
| PEPC N、2枝 | 2.7028→2.5795 | 0.954 |
| PEPC N、16枝 | 9.8268→12.6190 | 1.284 |
| PEPC S、2枝 | 0.8296→0.8040 | 0.969 |
| PEPC S、16枝 | 3.5729→3.4443 | 0.964 |

| 人工入力kernel | CPU中央値: 基準→試作（ms） | 試作/基準 |
| --- | --- | --- |
| 16枝、1,600列、各行120 nonzero | 0.5350→0.2824 | 0.528 |
| 16枝、2,000,000列、各行5,000 nonzero | 0.6940→0.3983 | 0.574 |
| 32枝、10,000,000列、float32、各行5,000 nonzero | 1.4253→0.7824 | 0.549 |
| 16枝、2,000,000列、共通supportあり | 0.7618→0.8173 | 1.073 |

小さい2枝controlはCPU比1.004、広い2枝controlは1.102だった。
和のコードは不変だが、controlのwall比は1.011–1.553と変動した。
実データのwall比も0.664–1.672と不安定で、改善の証拠として扱えない。
全入力でtracemallocのピークは同じか80 byte以内の差で、メモリ削減は示さない。

基準commitは `624bb8a6dae536d0fa10891f666684ef1ad3dab0`。
現在の作業ツリーではなく、変更前snapshotの `csubst.substitution` をimportし、
そのprivate namespaceで関数だけを置換した。
macOS arm64、Python 3.14.7、NumPy 2.5.3、SciPy 1.17.1、pandas 3.0.5。
import前にBLAS等のthread数を1へ固定した。
入力準備・同等性検査・全計測は共通の `benchmark_lock()` 内で直列化し、
各実装1 warmup後に実行順を交互にした7反復を実施した。
各反復を40 ms CPU目標、上限100呼出しでbatch化し、1呼出しに正規化した。
全sample、範囲、入力SHA256、packed幅、dtype、積のnonzero推移を
[proposal08.json](proposal08.json) に保存した。

同等性は501ケースを比較し、302出力でCSR data/indices/indptrがbyte単位で一致し、
199ケースで例外の型とmessageが一致した。148ケースは独立したdense参照とも一致した。
bool/int8/int32/uint64/float32/float64/complex128、重複・並べ替え・文字列化ID、
空選択、空のsite/state軸、有限値の相殺、ゼロ除去、NaN/±Inf、readonly入力、
無効ID・未知operation・dense入力を含む。非有限値を含む27枝積では、
列supportの和集合を使う独立したscalar疎参照にも一致した。
正規化されたreadonly出力と入力CSRの不変性も確認した。
全計測入力についても、計測前に出力の厳密一致を確認した。

再現コマンド:

```bash
python reports/performance_round3_20261023/proposal08.py
```

`CSUBST_PERF_BASELINE` で変更前snapshotを指定できる。
未指定時は同じ検証sessionの `/tmp/csubst-round3-baseline-path` を読む。
スクリプトの構文検査、`git apply --check`、`git diff --check` も通過した。
不採用で本体に変更がないため、既存pytest全suite、installed-wheel、native/fallback lane、
全CLI性能、独立した科学的校正は本案では実施していない。
実データ計測は集約kernelだけであり、ASRや描画を含むCLI速度ではない。
tracemallocは既存入力とnative allocatorの全量を含まず、peak RSSではない。
