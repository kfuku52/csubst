# 案06: endpoint露出量のサイト単位検査再利用 — 不採用

`624bb8a` の `scan_endpoint.expected_events` で、候補に依存しない親・子の
posterior抽出、妥当性検査、欠損判定、親posterior正規化、category条件付き
親posteriorをサイト単位で再利用する試作。
候補の露出量ベクトルと同じ32 MiB・4,096項目のLRUを使用した。
本体・既存テストは変更していない。[試作差分](proposal06.patch) は採用用ではない。

実データで再利用は成立したが、PEPCでは大きなサイト項目が既存の候補ベクトルを
追い出し、再計算が増えた。PGKの既定設定では再利用が一度もなかった。
追加の保持メモリと複雑さに対する利点を実証できず、不採用とする。

## 計測範囲と条件

```bash
python reports/performance_round3_20261023/proposal06.py
```

Python 3.14.7、macOS 27.0.1 arm64、NumPy 2.5.3、SciPy 1.17.1、pandas 3.0.5。
BLAS関連thread数はimport前に1へ固定。正確な環境、入力SHA256、ソースSHA256、
各反復は [JSON](proposal06.json) に保存した。
基準snapshotは `CSUBST_PERF_BASELINE`、未指定時はセッションの一時pointerを使用する。

同梱PGK/PEPCの固定fitとforegroundを新規一時ディレクトリへコピーし、実際の
joint/R4推論で得た入力を使用。`candidate_fixed`、10試行、seed 8。
`any2spe` と `any2spe,spe2spe` を比較した。warmup 1回を除外し、5反復を交互順で実施。
入力準備・正しさの検査・計測は共通の `benchmark_lock()` 内で直列化した。

「初回core」は `_scan_substitutions_core`、
「全scan」は `scan_substitutions` と固定候補校正まで。
CLI起動、ファイル読み込み、joint推論、結果ファイル出力は時間に含めない。

**下表は計装込みのCPU時間中央値であり、本番の速度差を表さない。**
cacheのhit/build数と保持payloadを記録するwrapperを計測内で使用した。
項目ごとの保持量走査はcache miss増加時に余分な負荷を加える。
またwall時間のばらつきが大きく、採用を支持する速度向上は検証できていない。
実行時間比を改善率や本番での悪化率として扱わない。

| データ・match | 候補 / サイト | 初回core CPU秒: 基準→試作 | 全scan CPU秒: 基準→試作 | 実サイトhit数 |
| --- | ---: | ---: | ---: | ---: |
| PGK any2spe | 12 / 12 | 0.1361 → 0.1346 | 0.2942 → 0.2995 | 0 |
| PEPC any2spe | 103 / 94 | 0.4922 → 0.4459 | 1.1695 → 1.6667 | 162 |
| PGK any2spe,spe2spe | 23 / 12 | 0.2179 → 0.2204 | 0.4002 → 0.3837 | 22 |
| PEPC any2spe,spe2spe | 198 / 94 | 0.7857 → 0.9004 | 1.5267 → 2.6775 | 356 |

| データ・match | 初回core wall秒: 基準→試作 | 全scan wall秒: 基準→試作 |
| --- | ---: | ---: |
| PGK any2spe | 0.7315 → 0.2443 | 1.1771 → 0.7903 |
| PEPC any2spe | 1.1015 → 0.6167 | 2.3803 → 3.2199 |
| PGK any2spe,spe2spe | 0.6305 → 0.5322 | 1.0656 → 0.9951 |
| PEPC any2spe,spe2spe | 0.9638 → 1.2863 | 1.9346 → 4.7314 |

例えばPGK any2speの全scan wall時間は基準0.321–3.388秒、試作0.354–2.025秒。
大きなwall中央値差は安定した改善の証拠にならない。

## 保持量とcacheの競合

payloadはプロセスpeak RSSではなく、cacheに保持した配列のhigh-water値。
counterは最後の反復の初回探索・校正worker chunkを合わせた値。

| データ・match | 基準payload bytes | 試作payload bytes | 候補vector build: 基準→試作 | 候補vector hit: 基準→試作 |
| --- | ---: | ---: | ---: | ---: |
| PGK any2spe | 80,640 | 2,330,112 | 48 → 48 | 216 → 216 |
| PEPC any2spe | 1,514,100 | 33,278,280 | 412 → 1,339 | 1,854 → 927 |
| PGK any2spe,spe2spe | 154,560 | 2,404,032 | 92 → 92 | 414 → 414 |
| PEPC any2spe,spe2spe | 2,910,600 | 33,278,280 | 792 → 2,574 | 3,564 → 1,782 |

試作payloadは既存の33,554,432 bytes上限以内に収まり、追加のglobal cacheはない。
worker transportで既存のexposure cacheとともに破棄される。
サイト項目は入力全配列への参照を保持せず、抽出済みの配列のみを保存する。
同じmissing配列を候補vectorとサイト項目が参照した場合も両者の計上に含めるため、
共有payloadの計上は保守的。
PEPCではサイト項目だけで31,984,680 bytesを保持し、候補vectorの再計算が増えた。
この実測された競合と、未検証の速度利得を採用判断の根拠とした。

## 正しさの確認と未確認範囲

- 4 workloadの全反復で、scan表・unit表・校正diagnosticsが基準と完全一致。
- 48条件×5候補で露出量・missing mask・理由文字列が完全一致。
  長さ0・1e-8・1・100、通常・recoding group、欠損、eligibility、bridge、
  mixtureのメモリ配列・disk-backed SiteArrayを確認した。
- 9失敗条件をcache有無の両方で確認し、例外型とメッセージが完全一致。
  候補範囲外と不正posteriorを同時に与えた際の失敗順序も保持。
- cacheを使用した後でも、通常の6引数呼び出しはmutable入力を再検査し、不正値を拒否。
- cache上限超過項目の非保持、共有budget内でのLRU evictionを確認。

本体への採用を見送ったため、pytest/full・native/fallback・installed-wheelの追加検査は
実施していない。IQ-TREE再fit、新しい科学的校正、独立したpeak RSS計測、
計装を外した性能比較は未実施。科学的既定値、閾値、seed、参照値を変更していない。
