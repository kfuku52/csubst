# 追加候補の本体実装・検証（2026-10-22）

基準は`88528ce` / 1.16.21、公開対象は1.16.22。
[追加10案](../performance_round2_20261022/README.md)から3案を実装し、
再プロファイルで不要なendpoint診断配列の確保も削減した。
科学的既定値・推定量・乱数・閾値・出力schema・参照値は維持する。

| コミット | 変更 |
| --- | --- |
| `944b87d` | 固定候補校正の1割当て内で枝metadata列とtarget/control maskを共有 |
| `57df867` | 固定候補校正は既存cacheにread-only枝vectorを保持。不要なDataFrameは保持しない |
| `65d96c1` | 構造重み付けCVのlog-prior・mask・採点応答をfold内で再利用 |
| `0693e8d` | endpoint返値で上書きされる診断配列の事前確保を省略 |
| `1e5e824` | pandas 3.1で非推奨になった統計列追加を`.loc`へ変更 |
| `dd42117` | 数値summary列をobject型へ変換してから空文字を補完 |

枝vectorは配列/objectサイズを既存32 MiB/4096件の上限に合算し、
収まらない場合は既存DataFrame経路を使う。単回探索・公開イベント表・calledイベントの
補償付き集約は維持する。worker transportではcacheを破棄する。
CVの再利用は1回のinner selection内だけで、同じ32 MiB/1024件の上限を使う。
held-out応答は採点専用で、training priorへ入れない。

## 実測

macOS 27.0.1 arm64、Python 3.14.7、NumPy 2.5.3、SciPy 1.17.1、pandas 3.0.5。
BLAS1、固定fit、seed8。入力は一時領域へcopyし、candidateのPython sourceを
計測開始前にsnapshot化した。同一native拡張を使う独立processで、warmup1組と
順序交互の測定3組以上を実行し、排他lockでCPU測定と検証を直列化する。
全反復・source/input hash・command・CPU・RSSは各JSONへ保存する。

以下は全CLIのwall秒、中央値。別の行のbaseline同士は比較しない。

| 独立比較 | 現行 | 試作/実装 | 結果 |
| --- | ---: | ---: | --- |
| 枝metadata、PGK・固定200 | 2.213 | 1.936 | 範囲分離 |
| 枝metadata、PEPC・固定200 | 7.890 | 5.911 | 範囲分離 |
| 枝metadata、PGK・固定200・2 process | 3.567 | 3.402 | 範囲分離 |
| vector、PEPC・固定1000（metadata適用済み版との比較） | 13.128 | 7.167 | 範囲分離 |
| vector、PEPC・固定200・2 process | 11.333 | 10.285 | 範囲分離 |
| CV、PGK固定fit＋合成context/features | 6.023 | 5.279 | 範囲分離 |
| CV、PEPC固定fit＋合成context/features | 29.982 | 27.508 | wall範囲重複、CPUは28.38→25.02秒で分離 |
| 診断配列省略、PEPC・固定1000（3変更適用済み版との比較） | 6.888 | 6.787 | 範囲分離 |

CVのcountは実際の同梱alignmentから計算したが、context/featuresは明示的な合成入力。
実構造・外部contextの独立性・新しい生物学的校正を検証したものではない。
通常検索と校正なしscanは対照条件として比較し、一般的な高速化は主張しない。
固定fitのPGK/PEPC参照値はomegaCany2spe 1.910709 / 0.030734で維持された。

最終合成版を元の1.16.21と比較した[再測定](final_confirm.json)では、PEPC固定1,000回の
wall中央値は23.102→7.060秒（69.4%減、3.27倍）だった。
測定範囲は22.826–23.244 / 6.960–7.343秒で分離し、科学的出力は一致した。
同じ最終版のPGK固定200・2 processも3.563→3.419秒で範囲が分離した。
full-scan並列条件や通常検索、校正なしscanでは、全般的な改善を主張できない。

最初の通常PEPC検索でpeak RSSが434→464 MBとなったため、
[追加5反復](memory_control.json)で調べた。中央値469→464 MB、範囲445–473 / 457–475 MBと
重なり、増加は再現しなかった。RAM改善も主張しない。
全8比較の完了分はwarmupを含め98組・196 CLI実行で、うち時間が有効な条件は90組。
個々の中央値はwarmupを除いた3反復（RSS追加確認だけ5反復）から計算する。

個別記録: [metadata](metadata_cli.json)、[vector](vectors_cli.json)、
[CV](structure_cli.json)、[診断配列](diagnostic_cli.json)、[最終版対照](final_cli22.json)。

## 追加のボトルネックと保留案

PEPC固定200のprofileでは`_rate_summary`が20,703回呼ばれ、
endpoint返値に上書きされる`U24`文字列配列を毎回確保していた。
実CLIで追跡すると、この配列だけで累計278,248,320 bytesを余分に生成していた。
削除後は0となり、期待値側で必要な配列206回・2,768,640 bytesは維持した。
これは**累計確保量**で、peak RSSの低減ではない。
[計測記録](allocations.json)と[変更前profile](profile_before.json)を残した。
残る主要費用はjoint endpoint推論と校正のrate計算であり、
数値同値性と安定した全CLI改善を確認できない追加kernel変更は採用しない。

案04はsiteごとにsnapshotを1つだけ保持する追加試作でも、全CLIでPGK
1.838→1.862秒、PEPC6.314→6.523秒となり、範囲も重なったため保留。
[記録](analytic_cli.json)は試作の比較で、本体へは取り込んでいない。
案05はRAM制約の実証がない通常経路へ、遅延と引き換えの列chunkを導入しない。
案06の共通tail検索はこの負荷では絶対費用が小さく、今回のprofileの優先対象ではない。
既に見送った01/07/08/09の既定経路も変更しない。

## 回帰・配布物の検証

Python 3.12.15の既存開発環境（pandas 3.0.6）と、別の非editable artifact環境
（pandas 3.1.0）で検査する。実行記録には修正前の成功/警告や再現失敗も残し、
最終source/配布物のチェックを区別する。

枝metadata/vectorの95＋95境界と、CVの16条件を変更前sourceと直接比較し、
返値・例外・配列bytesが一致した。新しいcache上限、疎な巨大ID、readonly所有権、
fit/siteキー、hold-out独立性、overflow、tie、evictionを検査した。
[境界比較](regression.json)と[98組の最終出力比較](output_audit.json)に結果を保存した。

新しい依存環境では、統計列の`.at`追加と、数値summary列への空文字fillが
pandas 3.1の非推奨操作として検出された。`.loc`追加と、object型への明示変換で対応し、
該当warningをエラーにする回帰検査を追加した。科学的値や出力列を変更する対応ではない。

配布物はsdistから生成Cを使ってwheelを再構築し、別prefixへ非editable installする。
installed wheelの6拡張prefix、full/native/process、科学的参照値を検査する。
nativeなしsdistのprefix検査・抽出sourceの全suite、およびsourceの拡張無効化経路も検査する。
省略したoptional backendや警告の詳細は検証記録へ記載する。

最終結果は[checks.json](checks.json)の`release-*`、
配布物とsourceのhash照合は[delivery.json](delivery.json)、
実行環境は[test_environment.json](test_environment.json)を参照する。
以前の`final-*`等は途中の版の記録で、最終配布物の証拠とは区別する。

| 最終確認 | 結果 |
| --- | --- |
| source fast | 1,671 passed / 6 skipped |
| source full・serial process | 2,184 passed / 9 skipped、別lane 5 passed |
| pandas 3.1 source full、Pandas4Warningをerror扱い | 2,184 passed / 9 skipped |
| 非editable wheel full・process・strict native | 2,184 passed / 9 skipped、5 passed、15 passed |
| nativeなしsdistの抽出source full・process | 2,154 passed / 39 skipped、5 passed |
| source拡張無効 full・process | 2,154 passed / 39 skipped、5 passed |
| lint・docs-check・typecheck | 成功（typecheckは指定25 module） |
| sdist/wheel生成・内容・metadata・prefix | 成功 |
| installed wheelの固定fit科学的参照 | PGK・PEPCとも既存値に一致 |

通常の9skipはtorch 4件、gemmi 2件、実IQ-TREE 3件の未導入による。
nativeなし経路はnative専用30件が追加skipとなる。
既存Matplotlib PendingDeprecationWarning 8件が残るが、pandas警告は最終suiteで0件。
実IQ-TREEの再fit、optional model download、独立した大規模科学的校正は行っていない。
固定fitの実CLI・installed artifact・局所回帰は、それらの検証の代用ではない。

## 計測上の修正

追加pytestが重なった診断配列比較の最後の並列条件と、依存環境の読込が重なった
最終比較のPEPC条件は、時間を無効として記録した。出力比較は保持し、該当条件は
排他lock下の別実行で測り直す。無効な値を改善率の根拠に使わない。
[時刻による確認](measurement_timestamps.json)も保持する。

最終比較の初回は、意図した`csubst_version`差で厳密比較が停止した。
その項目だけを別に記録するよう修正し、科学的内容は厳密比較のままfresh directoryで再実行した。
統計metadata表も列・型・行順を比較し、動的な`elapsed_sec`だけを対象外とする。
動的manifest/logは同値比較対象外。

## 再現

`BASELINE`は`88528ce`の製品sourceと同梱入力のsnapshotとし、同じPython用の
native拡張を用意する。live checkoutをbaselineに指定すると拒否する。
各workdirは新規の一時ディレクトリとする。

```bash
python reports/performance_round2_implementation_20261022/benchmark.py --baseline "$BASELINE" --workdir "$FRESH_WORKDIR" --result "$RESULT" --cases pepc_fixed1000 pgk_fixed_parallel200 pgk_full_parallel20 pgk_search pepc_search pepc_plain
python reports/performance_round2_implementation_20261022/regression.py --baseline "$BASELINE" --result "$REGRESSION_RESULT"
make test-fast lint typecheck test test-native package
```

元fixture、科学的参照、性能baseline、vendored sourceは編集していない。
生のrun directory、model、cache、binary、環境はコミット対象外。
