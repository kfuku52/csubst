# 案09: 到達距離を使ったuniformizationの開始次数

**判定: 今回の採用は見送り。** 微小遷移と出力の同等性は確認できたが、
実PGK/PEPCでwall timeの安定した短縮を示せなかった。本体は変更していない。

現行の非負uniformization級数は、少なくとも`states - 1`次まで進んでから
打切りを検討する。試作は正の遷移の有向最短距離の最大値を準備し、
その次数から検討する。実codon modelでは開始下限が60から2になった。
最小の正の部分和に対する既存`eps * 0.25`のtail条件、べきworkspace上限、
大きいtauでの`expm`経路は維持した。

到達距離だけでは十分でない。最初の経路の積がunderflowしても、
より長い経路が表現可能な確率を持つ場合がある。
**到達可能pairの部分和に0が残れば打切りを禁止**し、以降の項を計算する。
これを非常に短い枝の合成条件で実際に通過させている。
固有値再構成や絶対誤差だけによる打切りには変更しない。

## 測定

基準`88528ce` / 1.16.21、macOS arm64、Python 3.14.7、NumPy 2.5.3、
SciPy 1.17.1、pandas 3.0.5。import前にBLASスレッド環境変数を1へ設定し、
他案と共通の排他ロック内で測定した。固定fit PGK/PEPCを一時領域へcopyし、
joint scanの実モデル・tip likelihood・native transformを捕捉した。
warmup1組＋順序を交互にした3反復。

wall秒、中央値（最小–最大）。**coldはmodel生成・graph準備・全枝/全rateの遷移を含む**。
warmは遷移cacheを準備済みの`iter_blocks`全stageであり、CLI全体ではない。

| 条件 | 現行 | 試作 |
| --- | ---: | ---: |
| PGK cold model・全遷移 | 0.02226 (0.02220–0.02499) | 0.02274 (0.01923–0.02468) |
| PEPC cold model・全遷移 | 0.05966 (0.04851–0.10187) | 0.05047 (0.04199–0.07496) |
| PGK warm endpoint stage | 0.36950 (0.32982–0.41962) | 0.42868 (0.31476–0.45877) |
| PEPC warm endpoint stage | 3.30223 (2.35375–4.66741) | 2.49695 (2.02594–3.54245) |

全wall範囲が重なるため、中央値差を確定した改善・悪化とは扱わない。
PEPC cold CPUだけは0.05579 (0.04698–0.06810)→0.04292 (0.04084–0.04606)秒で
範囲が分離したが、短い準備stageのCPU差をCLIの短縮と見なさない。
全CPU反復は[JSON](proposal09.json)に保存した。

追加保持は到達mask 3,721 bytes/実モデル。準備時のdistance行列は29,768 bytesで、
演算用の一時配列やPython objectは別である。既存の8 MiBべきcacheと
32 MiB遷移cacheの上限は変更しない。これらは配列サイズであり、peak RSSではない。

## 同等性・検証

- 実モデルの受理次数はPGK 5–26、PEPC 5–23。60未満の実行を確認し、
  単なる準備追加や現行同士の比較になっていないことを検査した。
- 全枝/rate遷移の現行との差は0。実recordの全node/category/reduced出力は
  PGK18,968,913値、PEPC96,269,795値について変更数0。
  `1e-12`と`0.5`の閾値をまたいだ値も0だった。
- dense、chain、reducible、directed cycle、underflowする最短経路、zero Qの
  6モデル×7枝長、および混合rate/曖昧tip/predictiveの2条件を検査。
  別途128状態の容量不足fallbackと、吸収状態を持つ有向chainも確認した。
- 0より大きく1以下の枝長では420桁Decimalの独立Taylor級数参照を使用し、
  それ以外はSciPy `expm`と比較。許容差は既存rare-event検査と同じ
  `rtol=3e-14, atol=1e-300`。表現可能な正の確率を0へ失う例はなかった。
- 既存`tests/unit/test_endpoint.py`を試作classに差し替えて実行し、
  **26 passed、0 failed、0 skipped**。独立全状態列挙、失敗動作、微小遷移も含む。

検査ハーネスの初回は、directed cycleの`t=.1`の微小entryで
SciPy Pade参照と約`3.6e-9`の相対差が出た。現行と試作は一致していた。
許容差を緩めず、その範囲もDecimal参照で検証した。これは測定前の参照検査の修正で、
初回の失敗を成功した性能測定に数えていない。

全CLI、installed wheel、nativeなしfallback、別OS/CPU、独立した科学的校正は未実施。
既定値・参照値・入力fixtureは変更していない。

## 根拠と再現

[Fox and Glynn (1988)](https://web.stanford.edu/~glynn/papers/1988/FG88.html)は
Poisson確率の打切り誤差とunderflow/overflow管理を扱う。
[Sherlock (2021)](https://doi.org/10.1007/s00180-021-01102-6)
（[著者preprint](https://arxiv.org/abs/1809.07110)）もCTMC計算を扱う。
これらを背景資料とし、今回のgraph下限とreachable-zero guardは個別に検証した。
資料の存在だけから相対精度や速度を保証したとは扱わない。

```bash
python reports/performance_round2_20261022/proposal09.py
python -m ruff check reports/performance_round2_20261022/proposal09.py
```

[再現コード](proposal09.py)、[環境・入力hash・全反復・検証結果](proposal09.json)。
追加検証だけを再実行する`--verify-only`は、既存JSONの測定値を再利用して
容量/有向chain検査と既存26テストの結果だけを更新する。
