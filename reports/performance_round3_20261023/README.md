# 性能改善10案の独立検証と本体への採用（2026-10-23）

**10件を異なるサブエージェントで検証し、5件を本体へ採用、5件を見送った。**
基準は `624bb8a` / CSUBST 1.16.22。主担当が有望案をレビューし、
最終パッチの追加計測・回帰検証を実行した。担当記録は [agents.json](agents.json)。
ローカルコミットのためversionは変更せず、push/releaseは行わない。
科学的既定値、推定量、乱数、閾値、出力schema、既存fixtureと参照値は維持する。

| 案 | 改善内容 | 判断と根拠 |
| --- | --- | --- |
| [01](proposal01.md) | endpoint遷移行列参照の再利用 | **見送り**。出力115M値がbytes一致したが、PGK/PEPCのCPU差が小さく不安定。追加cache分岐に見合う改善なし |
| [02](proposal02.md) | Python経路のendpointイベント射影 | **限定採用**。広い61状態案は棄却。実際の20状態・Nのみ・一対一mappingで、最終patchのkernel CPUが39.1–39.6%減。native・一般mappingは既存経路 |
| [03](proposal03.md) | rate scoreのスカラー入力検査 | **採用**。25,822入力で返値・例外・警告が一致。3万callのCPU中央値38.3%減。数値式は維持 |
| [04](proposal04.md) | 固定候補校正のrate score一括計算 | **見送り**。数値25,754例は一致したが、異常値検出の遅延で先に報告される失敗理由が変わった。性能測定せず棄却 |
| [05](proposal05.md) | 固定候補校正のunit表生成省略 | **採用**。48設定・256 configurationで枝集合、結果、診断が一致。PGK/PEPC校正stageのCPU中央値41.3%/8.8%減 |
| [06](proposal06.md) | endpoint露出量のサイト検査再利用 | **見送り**。PEPCで32 MiB cacheが候補vectorを追い出し、再計算を増やした。時間は計装込みで、未計装の改善は未実証 |
| [07](proposal07.md) | 保存度計算の選択tipコピーを分割 | **採用**。16 MiB目標、最小1site。大規模合成float64入力の一時割当peak93.2%減。統合版のfresh-process RSSでも改善を確認 |
| [08](proposal08.md) | 疎テンソル積の空集合後の短絡 | **見送り**。人工入力だけCPU改善。実PGK/PEPCでは短絡0回。501同等性ケースは通過したが、適用先で効果なし |
| [09](proposal09.md) | 組合せ数DPの使用済み子状態を解放 | **メモリ目的で採用**。2箇所をpopへ変更。400独立全列挙を含む568ケース、322-bit整数まで一致。小木では数十μsのCPU増があり、一般的な高速化は主張しない |
| [10](proposal10.md) | 置換パターン分類の配列化 | **見送り**。baseline・作業版各94ファイルのASTで実行可能な呼出し0件。唯一の呼出しはコメント。未使用処理を変更しない |

上表のCPU割合は独立した各局所比較で、全CLIの短縮率ではない。改善率を加算しない。
案01/04/06/08/10のpatchは検証資料として保存するが、本体へは適用していない。

## 主担当による追加検証

案02は型保持を修正した最終patchそのものを7交互反復で再測定した。
96個のidentity/permutation/dtype条件、混合dtype、complex/int、空/strided/readonly、
一般mappingとnative dispatchを確認し、CPU時間の範囲が両blockサイズで分離した。
混雑したhost上のwall範囲は重なるため、信頼できるwall改善や構造解析CLI全体の倍率は示さない。
案03/05の初期全CLI比較は [scalar_cli.json](scalar_cli.json) に残す。
途中に別担当の正しさ検査がlock外で重なったため、時間は性能主張に使わず、出力一致のみ保持した。
その担当の実行は修正し、最終統合版を新規ディレクトリで再測定した。

[memory_followup.json](memory_followup.json) は本体の統合版を独立processで3回比較した結果。
全反復で出力digestが一致した。

| 測定 | baseline中央値 | 統合版中央値 | 範囲と解釈 |
| --- | ---: | ---: | --- |
| 保存度: 192 tips × 8,192 sites × 20 float64、peak RSS | 693.8 MB | 454.4 MB | 692.7–696.0 / 451.9–461.2 MB。入力・import込み、34.5%減 |
| 組合せ数: 8,192-tip pectinate、arity16、姉妹除外、tracemalloc peak | 17.337 MB | 0.150 MB | 各3回同値、DP実行中のみ。tree/importを除き、process RSSではない |

初回の組合せ数follow-upは入力木のconstructor呼出しで失敗したため、明示Newickへ修正した。
成功済み保存度比較を繰り返さず、未測定の組合せ数だけ再開した。経緯はJSONに記録する。

## 全CLIと回帰の最終結果

[final_cli.json](final_cli.json) は6条件、各warmup1組＋順序交互5組の**72 CLI実行**。
全組で主要表の列・型・行順・数値と科学的JSONが厳密一致した。
CB統計のelapsed_sec、manifest/log等の動的情報は比較対象に含めない。
以下はCPU秒の観測中央値であり、小さな差を一律の改善と扱わない。

| CLI条件 | baseline CPU秒 | 統合版 CPU秒 | 判断 |
| --- | ---: | ---: | --- |
| PEPC candidate_fixed 1,000回 | 7.81 | 7.19 | 7.9%減、対応5組すべてCPU減 |
| PGK candidate_fixed 200回 | 2.16 | 2.12 | 差は小さい |
| PGK candidate_fixed 200回・2 process | 6.03 | 5.75 | 範囲重複、一般的な並列高速化は未確定 |
| PGK full_scan 20回 | 3.09 | 3.05 | 対照条件、範囲重複 |
| PEPC 校正なしscan | 4.40 | 4.40 | 対照条件、CPU中央値同じ |
| PGK search | 2.34 | 2.36 | 対照条件、範囲重複 |

PEPC固定1,000回のwallは8.784→8.455秒だが、範囲7.716–12.868 / 6.941–11.088秒が重なる。
wall時間の確実な短縮や全workflow共通の倍率は主張しない。

この条件の最初のRSS中央値575→604 MB増を調べるため、
[rss_control_cli.json](rss_control_cli.json) でPEPC固定1,000回・校正なしscanを
それぞれ新規warmup＋5組、計**24 CLI実行**で再確認した。
固定校正RSSは604→598 MB、範囲571–643 / 544–606 MBで初回の増加は再現しなかった。
校正なしscanも598→593 MB、範囲584–638 / 556–615 MBで重なった。
実データ全般のRSS改善は主張しない。
追加比較の固定校正CPUは7.76→7.23秒（6.8%減）で、出力も全24実行で一致した。

[checks.json](checks.json) の最終source検査は以下の通り。

| 検査 | 結果 |
| --- | --- |
| affected endpoint/scan/kernel・組合せunit | 364＋8 passed、skipなし |
| make test-fast | 1,679 passed / 6 skipped |
| make test | 2,196 passed / 9 skipped、serial process別lane 5 passed |
| make test-native（strict）・6拡張import | 15 passed / skipなし、6拡張すべて実体を確認 |
| 拡張無効化 make test | 2,166 passed / 39 skipped、serial process 5 passed |
| make lint・typecheck・変更後docs-check | 成功。typecheckはMakefile指定moduleのみ |
| PGK/PEPC固定fit参照値 | 成功、[結果TSV](scientific_reference.tsv)。omegaCany2spe 1.910709 / 0.030734を維持 |

通常9skipはtorch4件、gemmi2件、実IQ-TREE3件の未導入による。
拡張無効化ではnative専用30件が追加skipとなる。
既存Matplotlib PendingDeprecationWarning 8件を確認し、許容値や参照値を緩めていない。
固定fit検査は製品sourceと入力を一時snapshotへコピーし、strict nativeで行った。
報告scriptのRuff初回検査でimport配置と未使用importを検出し、import前のBLAS制限を
維持して修正後に再通過した。数値kernelや測定条件の修正ではない。

包装・依存・Cython sourceは未変更のため、package/非editable wheel・sdist再構築、
他OS CI、sanitizer、実IQ-TREE再fit、optional model取得、新しい科学的校正は実施していない。
source/native/fallbackと固定fitの等価性の証拠を、それらの検証の代用とはしない。

## 測定・正しさの範囲

性能測定はmacOS 27.0.1 arm64、Python 3.14.7、NumPy 2.5.3、SciPy 1.17.1、pandas 3.0.5。
回帰チェックは既存Python 3.12.15開発環境（NumPy 2.5.4、SciPy 1.18.1、pandas 3.0.6）。
BLAS thread数は性能ハーネスのimport前に1へ設定し、実CLIは明記した1/2 process。
全CPU負荷のある計測・検査を `bench_common.benchmark_lock()` で直列化する。
外部の無関係なhost負荷は制御していないため、wall/CPUと範囲を併記し、
不安定なwallの中央値差だけで改善と結論しない。

実CLI入力は同梱PGK/PEPCのalignment/tree/固定fitを新規一時ディレクトリへコピーする。
IQ-TREEの再fit、外部構造モデルdownload、独立した大規模科学的校正を含まない。
案07の大規模入力と案02のkernel入力は明示的な合成データ。
source検査はinstalled-wheel、他OS、Linux性能baselineの検証を代替しない。

## 再現

リポジトリrootで開発環境を選び、baselineはlive checkoutと別に作成する。
同じinterpreter用の既存native拡張を共有し、製品Python sourceとdatasetを固定する。

```sh
BASELINE="$(mktemp -d)"
git archive 624bb8a csubst | tar -x -C "$BASELINE"
export CSUBST_PERF_BASELINE="$BASELINE"
python - <<'PY'
import os
from pathlib import Path
import shutil
for path in Path('csubst').glob('*_cy*.so'):
    shutil.copyfile(path, Path(os.environ['CSUBST_PERF_BASELINE']) / 'csubst' / path.name)
PY
python reports/performance_round3_20261023/proposal03.py
python reports/performance_round3_20261023/proposal02.py --phase narrow
python reports/performance_round3_20261023/memory_followup.py --baseline "$BASELINE" --result "$RESULT"
python reports/performance_round3_20261023/compare_cli.py --baseline "$BASELINE" --workdir "$FRESH_WORKDIR" --result "$CLI_RESULT" --cases pepc_fixed1000 pgk_fixed200 pgk_fixed_parallel200 pgk_full20 pepc_plain pgk_search --repeats 5
make PYTHON="$VERIFY_PYTHON" test-fast lint typecheck test test-native
CSUBST_DISABLE_EXTENSIONS=1 make PYTHON="$VERIFY_PYTHON" test
```

各案の再現scriptは同名JSON/patchを更新するため、既存結果を別に保存してから実行する。
各案の実行方法は、それぞれの報告に記載する。
保存patchは空白検査を通すためzero-context形式に統一した。
baselineへの適用検査には `git apply --unidiff-zero --check` を使う。
9つの非空patchすべてで適用検査済みで、数値実装の変更ではない。
生のrun directory、cache、binary、環境はコミット対象外。
