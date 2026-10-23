# 案01: endpoint推論の遷移行列参照をblock間で再利用

**判定: 採用見送り。** transition呼出しは大幅に減るが、実入力の改善は安定しない。
PEPCの対応する反復のCPU比中央値は0.998で、効果がほぼ見えない。
製品コードは変更せず、再検討用の未適用patchを残す。

## 試作と範囲

`EndpointModel.iter_blocks`は枝・rateカテゴリごとの同じ行列を、各blockの
pruningとforward traversalで`transition()`から取得している。
試作は最初のpruningで得た行列の参照を局所tableへ保存し、以降で再利用する。
行列計算、block size、float64、rate積分、推定量、閾値は変更しない。

tableは`nodes × categories <= min(_cache_items, 8192)`の場合だけ作る。
全行列が既存32 MiB cacheへ収まる保守的な条件であり、参照数も8192以下に制限する。
条件を満たさないモデルは既存の呼出し経路を使う。
初回の遷移行列計算は元と同じpruning位置で行い、空サイトでは実行しない。

固定fitモデルの推論を対象とする。generator実行中のlength/rate変更は対象外。
privateなLRUの最終順序は変わるが、cacheのキー集合、全行列と容量は同じ。
参照tableは行列をcopyせず、generator終了時に解放される。

## 実測

基準`624bb8a`、macOS 27.0.1 arm64、Python 3.14.7、NumPy 2.5.3、
SciPy 1.17.1、pandas 3.0.5。import前に各BLAS環境変数を1へ設定した。
共有lock下で、同梱PGK/PEPC固定fitを新規一時ディレクトリへcopyし、
通常joint scanのモデル・tips・native reducerを捕捉した。
独立したモデル/cacheを両版へ与え、block 64、1 warmup後に9反復を交互順で測定した。
table作成・入力検査を含む`iter_blocks`全stageの測定であり、CLI全体の時間ではない。

| 入力 | 基準wall秒、中央値 (範囲) | 試作wall秒、中央値 (範囲) | 基準CPU秒、中央値 (範囲) | 試作CPU秒、中央値 (範囲) |
| --- | ---: | ---: | ---: | ---: |
| PGK、65 nodes、417 sites、4 categories | 0.5953 (0.3922–1.4008) | 0.6394 (0.3601–1.4744) | 0.3839 (0.3431–0.4058) | 0.3666 (0.3326–0.3916) |
| PEPC、141 nodes、971 sites、4 categories | 2.2554 (1.7675–2.6516) | 2.1701 (1.9060–11.2242) | 1.6767 (1.5967–1.6916) | 1.6473 (1.6074–1.9613) |

PGKのCPU中央値は4.5%、PEPCは1.8%小さい。
ただし対応する反復のCPU比中央値はPGK 0.966、PEPC 0.998であり、
速かった反復はそれぞれ6/9、5/9にとどまる。
同一マシンで共有lockの対象外のCPU処理も確認され、wall値のばらつきが大きい。
中央値差だけを根拠に、実入力の高速化や悪化を断定しない。
小さく混在するstage効果を受けて、全CLI反復は追加しなかった。

| 入力 | transition呼出し、基準→試作 | tableのPython object bytes | warmed stage tracemalloc peak、基準→試作 |
| --- | ---: | ---: | ---: |
| PGK | 3,584→256 | 2,392 | 17,859,910→17,862,471 B |
| PEPC | 17,920→560 | 4,824 | 37,016,782→37,021,622 B |

table bytesはouter/row listを`sys.getsizeof`で数えたもの。
tracemallocは計測外の別passで取得した追加allocationのpeakであり、process RSSではない。
cacheをwarmupした後の比較で、既存のモデル・入力・遷移行列は計測開始前に存在する。
RSS低減は主張しない。

## 同等性とチェック

- 実入力の全recordについてnode、category node、reduced出力を厳密配列一致と
  bytes一致で比較した。PGK 18,968,913値、PEPC 96,269,795値が一致。
  cacheキー集合・遷移行列も一致し、容量内に収まることを確認した。
- 合成36条件、892,896値がbytes一致。block 1/64/200、joint/predictive、
  選択枝・非選択枝transform、ゼロrateを含む4カテゴリ、ゼロ長枝、
  曖昧・欠損・非unit singleton・空サイト、cache容量不足の既存経路を含む。
- ゼロ尤度、不正block size、負値、NaN、葉不足の5条件で同じValueError文字列を確認。
  空サイトは遷移呼出し0、容量不足の小モデルは両版とも8呼出しだった。
- `python -m ruff check reports/performance_round3_20261023/proposal01.py`: pass。
- `git apply --check reports/performance_round3_20261023/proposal01.patch`: pass。

本体への適用は見送り、製品のpytest/Makefile全laneはこの案として実行していない。
全CLI、installed wheel、source-only reducer、Linux/Intel、独立した科学的校正は未検証。
この限定的なcache制御を追加する根拠として、今回の時間差は不十分だった。

## 再現

固定した`624bb8a`のsource snapshotと互換なnative extensionsを用意する。
`CSUBST_PERF_BASELINE`でそのrootを渡す。harnessはendpoint.pyのhashも検査し、
後日のlive checkoutを基準として暗黙に使用しない。

```bash
CSUBST_PERF_BASELINE="$BASELINE_ROOT" \
  python reports/performance_round3_20261023/proposal01.py
python -m ruff check reports/performance_round3_20261023/proposal01.py
git apply --check reports/performance_round3_20261023/proposal01.patch
```

[試作・検証コード](proposal01.py)、[全反復・環境・入力hash](proposal01.json)、
[未適用patch](proposal01.patch)。既存の実入力capture harnessをimportして再利用した。
