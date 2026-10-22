# 案5: scanの同一サイトの状態検証・正規化を共有

**評価: 現在の安全な試作は既定導入を見送り。** PGKの全match条件ではCPU費用が減ったが、
通常scanでのwall time改善は確認できず、PEPCではsnapshotとcacheの保持費が大きい。
既存の候補単位exposure cacheを再導入する案ではなく、from/to集合が異なる候補でも共通する
同一siteのparent状態・missingness・正規化・カテゴリー別parent状態を共有する案を検証した。

## 根拠と試作の境界

対象は `csubst/scan_endpoint.py:expected_events`。
現行関数は候補ごとに4組の枝×状態のadvanced indexing、有限・非負・行和の検証、
missingness判定、parent正規化を行う。rate mixtureではカテゴリー別parentの読取りと
parent marginalとの一致検証も繰り返す。これらは同じ固定fitの同一site内で不変である。

- [NumPy公式indexing](https://numpy.org/doc/stable/user/basics.indexing.html):
  advanced indexingはcopyを作るため、同じ枝/site選択の反復に配列生成費がある。
- [NumPy公式ufunc](https://numpy.org/doc/stable/reference/ufuncs.html) と
  [einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html):
  正規化と候補依存の縮約を分離する根拠。計算式・縮約順を変更しない。
- Williams, Waterman & Patterson, [Roofline](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html):
  データ再利用とメモリ移動を評価する一般的背景。今回hardware counterやRoofline実測は行っていない。

公開関数の入力検証を削除したり、配列identityだけを不変性の証拠にしたりはしない。
試作 `PreparedSnapshot` はcodon/nonsyn状態とcontext内のndarrayをコピー所有しread-onlyにする。
カテゴリー状態のsealed `SiteArray` は同じrunの不変ファイルとして保持する。
fit/filter/bootstrapが変われば新snapshotを作る。外部のmutable配列を直接変更しても旧snapshotへ
影響せず、新snapshotは変更を取り込む。このコピー費用も計測に含めた。

site単位のprepared配列は32MiB上限のLRUで共有し、eviction後は再読取り・再検証する。
候補のfrom/to検証、候補別weights/expected/reasonsは現行関数のソースから取り出し、そのまま実行する。
製品コードと既存の候補単位cacheは変更していない。

## 測定条件

`4498c43`、macOS 27.0.1 arm64、Python 3.14.7、NumPy 2.5.3、SciPy 1.17.1、pandas 3.0.5。
import前にVECLIB/OpenBLAS/OMP/MKL/BLISのthread環境変数を1に設定。
共有ファイルロックで他案のCPU測定と直列化した。
同梱PGK/PEPCのfitを新しい一時ディレクトリへコピーし、実CLI scanの
`--scan_match any2spe` と `all`、`--scan_pvalue_calibration none` から実際のexposure呼出し列を捕捉した。
カテゴリー状態は同じdisk-backed表現を維持し、一時runの削除後も読める独立コピーを保持した。

同じcontext・state・候補順に対して、現行関数と新snapshotによる全呼出しを再生する。
snapshot構築とcold site cacheからの評価を含む。入力準備・CLI・出力比較は計測から除外。
warmup1回後、実行順序を交互にして3回測定。以下は中央値でありCLI全体の時間ではない。

| 条件 | 呼出し / site | wall秒・現行→試作 | wall変化 | CPU秒・現行→試作 |
| --- | ---: | ---: | ---: | ---: |
| PGK any2spe | 12 / 12 | 0.00498 → 0.00506 | +1.8% | 0.00493 → 0.00506 |
| PGK all | 146 / 28 | 0.16689 → 0.13026 | −21.9% | 0.14088 → 0.10876 |
| PEPC any2spe | 103 / 94 | 0.06929 → 0.06804 | −1.8% | 0.06850 → 0.06777 |
| PEPC all | 811 / 153 | 1.42269 → 1.45397 | +2.2% | 1.36759 → 1.32771 |

全条件でwall timeの最小–最大が重なった。PGK allは現行0.13848–0.21947秒、
試作0.09112–0.21077秒。PEPC allは現行1.40126–1.81824秒、試作1.45390–1.53800秒。
PGK allのCPUは約22.8%減り、CPU範囲も分離したが、通常のany2speで同じ利益はない。
PEPC allではCPU約2.9%減にとどまり、wall改善を主張しない。

## Cacheとメモリ

| 条件 | hit / miss / eviction | snapshot ndarray MiB | peak cache ndarray MiB |
| --- | ---: | ---: | ---: |
| PGK any2spe | 0 / 12 / 0 | 19.76 | 2.15 |
| PGK all | 118 / 28 / 0 | 19.76 | 5.01 |
| PEPC any2spe | 9 / 94 / 13 | 91.26 | 31.68 |
| PEPC all | 140 / 671 / 590 | 91.26 | 31.68 |

PEPC allは153のunique siteに対してcacheが小さく、候補順の再訪問でevictionが多い。
安全なsnapshotコピーの保持に加えて最大約32MiBを使い、この試作をRAM削減策にはできない。
表は明示的なndarrayペイロードであり、Pythonオブジェクト、`SiteArray`内の1-site cache、
一時配列、OS file cache、全process peak RSSは含まない。ピークRSSは未測定。

## 正確性・残る確認

全測定ペアのexpected値・missing mask・reason文字列を全要素厳密一致で照合し、最大差0だった。
通常・missing・zero-length・zero-source・unreachable・eligibility・mixture・bridgeの8小例も一致。
NaN・負値・不正行和・カテゴリーposterior不一致・候補範囲外の5条件で同じValueError文言を確認した。
snapshot後の元state変更からの隔離、新snapshotでの再構築、容量110 bytesでのeviction/再検証も成功。
実PGK/PEPCのrate mixtureとdisk-backedカテゴリー入力も上記の比較に含む。

現行の任意mutable公開入力に黙ってcacheを付ける案は採らない。
将来、内部scan stageがstateの排他的な不変所有権を明示できればsnapshotコピーを避けられるが、
その設計・速度・alias安全性は本実験で検証していない。
候補順を内部でまとめて再利用率を上げる案も未実装。
CLI最終表/校正JSON、実worker、3Di、native/fallback全レーン、installed wheel、
Linux性能基準や科学的校正は今回の検証に含めない。

## 再現

```bash
python reports/performance_candidates_20261022/proposal05.py
python -m ruff check reports/performance_candidates_20261022/proposal05.py
git diff --check
```

[スクリプト](proposal05.py) と [全測定値・入力hash・edge checks](proposal05.json)。
上記コマンドは成功。最初のedge検査では、2 siteが収まる200-byte予算を誤って
1 site用と扱ったfixture assertionが失敗した。予算を110 bytesに直して再実行し全確認を完了した。
科学的な閾値や製品コードには変更していない。source試作の同値確認であり、科学的校正の証明ではない。
