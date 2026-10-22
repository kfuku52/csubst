# 性能改善の本体実装・追加検証（2026-10-22）

**4件を個別コミットで実装し、1.16.21として公開準備した。**
候補表示、候補固定校正、Fisher inclusionに加え、再プロファイルで見つけた
集約済み枝イベントの再groupbyを削減した。科学的既定値、推定量、乱数、閾値、
出力形式、参照値、性能基準は変更していない。

| コミット | 変更 |
| --- | --- |
| `7975b67` | 候補分布の表示で小型Series生成を省き、pandasと同じtie/NaN順を保持 |
| `8886d86` | 候補固定校正は共通の数値コアを使い、出力用の枝数・枝長・理由文字列を省略 |
| `55fedb0` | Fisher inclusionの加算順を保ってサイト方向へベクトル化 |
| `128864f` | 枝ごとに1行のイベントは再groupbyを省略。重複行は従来の補償付き集約を維持 |

前回の[10案の調査](../performance_candidates_20261022/README.md)は
`13c31dc`で保存した。リモート更新を取り込んだ公開比較元は
`8548fbf74e9ee68cc6c1a35ca143e2dac7955bce`。
数値コードは前回の`4498c43`と同じだった。

## 追加測定

macOS 27.0.1 arm64、Python 3.14.7、NumPy 2.5.3、SciPy 1.17.1、pandas 3.0.5。
同じnative拡張・同梱固定fit・seedを使い、CSUBSTは1プロセス
（並列条件のみ2）、BLASは1スレッド。warmup1回、順序を交互にした計測3回。
モデル再推定・ネットワーク・描画は除外した。原入力と実行出力は一時領域に置いた。
コマンド・入力/ソースhash・全反復・RSSはJSONに保存する。
保存版JSONの実行パスは変数表記へ置換している。

以下は**CLI全体のwall time**、秒、中央値（最小–最大）。

| 条件 | 現行版 | 最終実装 | 中央値短縮 |
| --- | ---: | ---: | ---: |
| PEPC・候補固定校正50割当て | 9.465 (6.993–9.745) | 5.941 (5.751–6.784) | 37.2% |
| PGK・候補固定校正20割当て・2プロセス | 3.589 (3.259–4.774) | 2.923 (2.894–3.208) | 18.5% |
| PGK・full-scan校正20割当て・2プロセス | 3.851 (3.784–3.873) | 3.399 (3.354–3.502) | 11.7% |
| PGK・Fisher urn検索（追加groupby改善前） | 13.427 (12.400–14.565) | 10.276 (9.940–10.422) | 23.5% |

これらは同じ条件の最小–最大範囲が分離した。両並列条件は校正JSONの
`backend=multiprocessing, n_jobs=2, success_count=20`でも確認した。
1,000割当て既定の全実行を測ったものではない。
Fisherの関数単体では971サイト・draw100で約14倍だったが、CLIの倍率と区別する。
その重みは合成入力であり、実urn検索の時間は上表で別に測った。

通常PEPC scanの最終測定は4.089→4.819秒、全統計検索の初期測定は
8.728→10.880秒だったが、いずれも測定範囲が重なる。これらの改善・悪化は断定しない。
全統計検索は変更した計算経路を使わない対照条件でもある。
別段階のbaseline時間同士を比較せず、RSSの一般的な低減も主張しない。

[初期7条件](cli.json)、[最終4条件](cli_final.json)、[条件付き4条件](conditionals.json)の
計120有効CLI実行・60比較組で、主結果表の**全列・型・行順が厳密一致**した。
scanの校正・推論JSONもversion情報以外で一致した。通常の数値許容差による検査に加えて
`check_exact=True`で全比較組を追検証した。実行時間等の動的manifestは同値比較対象外。

## 残っていたボトルネック

優先3案の実装後にPEPCの50割当てをcProfileで調べると、
`_rate_summary`が5,253回呼ばれ、枝別集約を繰り返していた。
posterior抽出済み入力は枝ごとに1行なので、重複がなければ再groupbyは不要だった。
NaN質量を0とする既存の扱いを保ち、重複があるcalledイベントでは従来のsumを使う。

50割当ての試作比較ではPGK/PEPCのCPU中央値が約33%減り、全スコア・キー・反復診断が一致した。
この局所比較のwall範囲は重なるため、確実な短縮率とは扱わない。
全CLIの効果は上表で評価した。profileのgroupby.sum呼出し数は5,801→548へ減った。
[試作比較](branch_events.json)、[変更前profile](profile_before.json)、
[変更後profile](profile_after.json)。profileの時間は診断用であり、比較ベンチマークではない。

## 条件付き候補の扱い

- 案1のパターン圧縮は、通常PGK/PEPCでは費用が上回り、展開時メモリも増えるため保留。
- 案2のblock 64→128は、実CLIでPGK 2.126→1.988秒、PEPC 5.260→4.842秒だが範囲が重なる。
  PEPCには既存の書込workspace上限もあり、推論単体の倍率を適用できない。既定を維持した。
- 案4のspool mmapは、高次arityでの条件付き効果に対しRSS増があるため保留。
- 案10のout指定は、PEPCで4.819→4.120秒だが範囲が重なり、
  人工反復PEPC×4は14.021→14.685秒で改善しなかった。今回は採用しない。

PEPC×4はサイズ試験で、独立した生物学的標本ではない。
これらの保留案を速度だけを理由に既定経路へ入れていない。

## 検証

Python 3.12.15、NumPy 2.5.4、SciPy 1.18.1、pandas 3.0.6、Matplotlib 3.11.2、ETE4 4.4.0。
[環境](test_environment.json)と[実行コマンド・結果](checks.json)に詳細を保存する。

| チェック | 結果 |
| --- | --- |
| `make test-fast` | 1,668 passed・6 skipped |
| `make lint` / `make typecheck` | Ruff・hygiene・文書チェック、対象25モジュール成功 |
| `make test` | 2,167 passed・9 skipped、process 5 passed |
| 6拡張の直接import / `make test-native` | import成功、15 passed |
| native無効の全source suite | 2,137 passed・39 skipped、process 5 passed |
| PGK/PEPC科学的参照値 | 成功。omegaCany2spe 1.910709 / 0.030734、サイト分類も一致 |
| `make package` / 生成Cからsdist wheel再構築 / artifact check | 全て成功 |
| 別環境の非editable wheel全suite | 2,167 passed・9 skipped、process 5 passed |
| installed wheelのPGK/PEPC参照値 | 成功 |
| nativeなしsdistインストール・抽出source全suite | 2,137 passed・39 skipped、process 5 passed |

通常の9 skipはtorch不在4件、gemmi不在2件、IQ-TREE不在3件。
fallbackで増える30 skipはnative専用の検査。該当する外部バックエンドの証明には数えない。
Matplotlibの`set_bad`に関する既存PendingDeprecationWarningが8件あった。
installed wheelはcheckout外から`CSUBST_TEST_INSTALLED=1`で実行し、
packageと6拡張のprefix検査も通過した。
科学的参照は固定fitの回帰検査であり、新しい校正や実IQ-TREE再推定ではない。
Linux性能基準・sanitizer・他Python版・macOS Intelのローカル検証は行わず、push後のCIで確認する。

## 検証ハーネスで修正した点

条件付き比較の初回に、既存ハーネスが合成PEPC×4をcheckout内へ生成した。
該当計測を破棄し、新規8ファイルを一時領域へ移動した。追跡済みfixtureの差分はない。
baseline/candidateの固定スナップショットを使い、checkoutをbaselineへ指定すると
実行前に拒否する保護を追加して、条件付き比較を最初から再実行した。

また、sdistの拡張不在検査をcheckout内で始めた初回は、checkoutのnative moduleを拾って失敗した。
checkout外・isolated modeでinstalled prefixも検査して再実行し成功した。
初回の失敗記録は残し、弱い検査や拡張無効化による成功へ置き換えていない。

既存`build/`と`dist/`は一時領域へ退避してから新規生成した。旧成果物を削除していない。
生成C・バイナリ・実行ディレクトリ・モデル・キャッシュはコミット対象外。

## 再現

[benchmark.py](benchmark.py)は前回の再利用可能なハーネスを呼び、
baselineの製品コードを一時スナップショットとして指定する。
各スナップショットには同じPython用のnative拡張を用意する。
`BASELINE_ROOT`は`13c31dc`、`PRODUCTION_ROOT`は`128864f`相当、
`BEFORE_BRANCH_ROOT`は`55fedb0`相当の製品コードと同梱入力を配置する。
`check-results`は再実行直後の未置換JSONへ使用する。

```bash
python reports/performance_implementation_20261022/benchmark.py phases --baseline-root "$BASELINE_ROOT" --workdir /tmp/csubst-production-phases --result /tmp/csubst-production-phases.json
python reports/performance_implementation_20261022/benchmark.py cli --baseline-root "$BASELINE_ROOT" --candidate-root . --baseline-label 13c31dc --candidate-label production --workdir /tmp/csubst-production-cli --result /tmp/csubst-production-cli.json --cases pepc_scan_defaults pepc_fixed50 pgk_fixed_parallel20 pgk_scan_parallel20 pgk_fisher
python reports/performance_implementation_20261022/benchmark.py check-results /tmp/csubst-production-cli.json
python reports/performance_implementation_20261022/benchmark.py conditionals --baseline-root "$PRODUCTION_ROOT" --candidate-root "$PRODUCTION_ROOT" --workdir /tmp/csubst-conditionals --result /tmp/csubst-conditionals.json --cases block_pgk block_pepc out_pepc out_pepc4
python reports/performance_implementation_20261022/benchmark.py branch-events --baseline-root "$BEFORE_BRANCH_ROOT" --workdir /tmp/csubst-branch-events --result /tmp/csubst-branch-events.json
```
