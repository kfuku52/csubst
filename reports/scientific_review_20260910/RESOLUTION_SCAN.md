# 科学レビュー ID5: scan の統計量・診断値・較正の分離

2026-09-10。実装基点は `c4e6d82`。専用worktreeで実装後、本体masterの変更を取り込んで再レビュー・統合検証を行った。pushは実施していない。

## 実装した範囲

- `score_rate_enrichment` と漸近診断値 `p_rate_enrichment_asymptotic` を分離した。小標本2/0・等exposureの漸近値は約0.04794548であり、独立整数Poissonの条件付き二項検定の0.25と同じではない。`called` もfractional posterior massであり、整数検定へ置換しない。
- ユーザー指定どおり、BHは **trait × match別の出力候補群** のみ。global/trait単独のq列を削除し、群ID・候補数・有効診断値数を出力する。漸近PとそのBH値は探索的診断として明記した。
- 既存のclade較正も同じscoreで比較する。失敗した帰無反復があればempirical P/qを未定義にし、成功反復だけの分母を使わない。空の検定可能候補群を持つ成功反復は最大score=-∞で保持する。
- `parametric_bootstrap` を追加した。観測データに適合した一様rateのGY+F/GY+FQから完全配列を生成し、各反復でIQ-TREEパラメータ・枝長・ASR・自動recoding・site filter・exposure・候補選択を再実行する。観測データで学習したASRやsite maskを使い回さない。
- 最大scoreによるPの補正範囲は **同一runの全trait・match・検定可能候補**。trait×matchのBHとは別で、maxT値に再度BHをかけない。
- 生成モデルはcheckpointのkappa/omegaと高精度頻度から再構成し、独立pruningによる尤度がIQ-TREEの報告値に一致することを検査する。乱数seed、コマンド、モデル、各反復結果・失敗理由を保存する。
- 全site除外は正常なno-test。有限非負massでexposureが0の候補も明示的なno-testとして扱い、反復は分母に残す。NaN・負値等の原因不明な計算異常は失敗のままにする。
- CLI説明、列名移行、表示フィルター、JSON推論メタデータ、使用ガイドを更新した。

使用法・列の意味・対応条件は [SCAN_INFERENCE.md](../../docs/SCAN_INFERENCE.md) を参照。

## 検証結果

全体テストは **1774 passed / 5 skipped**（87.25秒）。その収集後に追加した全site除外の回帰テストを含むintegrationファイルは **23 passed**。skipはtorch>=2.6を要求する3件（環境は2.2.2）とgemmi未導入の2件。既存のrequests依存版警告が1件。`make test` はpytest-xdist未導入で起動できなかったため、同じ全テストを `python -m pytest -q` で逐次実行した。`make lint`、`make typecheck`、新規2モジュールのmypy、`git diff --check`も通過。記録は [checks/](scan_id5/checks/) にある。

実IQ-TREE 2.3.6ではGY+F、GY+FQ、欠損、filter、called/posterior_sum、自動6群recodingを含む3組×3反復を完了した。さらにGY+F・any2spe＋spe2speで **99/99成功、失敗0** を確認した。99反復中40反復は検定可能な候補群が空で、そのまま参照分布に保持された。最小Pは0.01。モデル再構成尤度は -1817.900419434838、IQ-TREE報告は -1817.900419。

99反復の初回実行では、多段階endpoint変化に対する瞬間Q exposureが0となり、未定義scoreが反復失敗として検出された。明示的なtestability条件と回帰テストを追加し、同じseedで最初から99反復をやり直した。失敗後の成功反復だけを集めた較正ではない。現在の仕様ではzero exposureはno-test、それ以外の未説明な計算異常は失敗である。

独立検証の最終結果は以下。名目水準5%、各条件の参照9999・完全帰無検証5000・部分対立200。

| 条件 | tip invariant filter | 完全帰無の誤検出率 | 95%区間 | 上限≤6% | 部分対立でのnull site誤検出率 |
| --- | --- | ---: | --- | --- | ---: |
| sparse | なし | 2.10% | 1.72%–2.54% | 達成 | 3.00% |
| sparse | あり | 1.90% | 1.54%–2.32% | 達成 | 2.00% |
| unequal | なし | 5.50% | 4.88%–6.17% | 未達 | 5.00% |
| unequal | あり | 5.60% | 4.98%–6.27% | 未達 | 6.00% |
| uncertain | なし | 4.86% | 4.28%–5.49% | 達成 | 4.00% |
| uncertain | あり | 4.80% | 4.22%–5.43% | 達成 | 3.50% |

6条件中4条件で基準を満たしたが、不均一枝長の2条件は未達。全条件合格とは判定しない。部分対立の200反復は精度が低く、strong FWERの根拠にはしない。詳細は [extended/summary.json](scan_id5/extended/summary.json)。

## 検証の解釈と残る制約

独立シミュレーターは4 codon CTMCと独立な厳密周辺posterior pruningを使い、実際のCSUBSTの枝長rescaling、filter、候補選択、score/maxT計算を通す。疎な変化、不均一な枝長、祖先状態の不確実性と既知の非一様site rate、2 trait × 2 match、filter有無を検査した。ただしパラメータは既知で、IQ-TREEの再推定は含まない。**この誤検出率をfitted bootstrapのFWER検証と呼んではならない。** 実IQ-TREEの検証は手続きが最後まで正しく動くことの検証である。

先行pilot（参照999、検証1000/条件）ではuncertain＋filter条件が7.7%、95%区間6.12–9.53%となり、事前基準「区間上限≤6%」を満たさなかった。その結果を削除せず [pilot/summary.json](scan_id5/pilot/summary.json) に保存した。共有する有限参照標本の変動と処理上の問題を区別するため、一度だけ参照9999・検証5000に拡張した。拡張版では役割ごとに独立seed streamを使用し、参照数の変更で検証datasetがずれないようにした。区間は固定した参照標本を条件とする二項区間であり、参照標本自体の不確実性は含まない。合格するまでseedや基準を変更することはしていない。

GY+F/GY+FQ、一様site rate、通常AA/codon recodingのみをfitted bootstrapでサポートする。3Di、混合rate、モデル選択、部分曖昧codonは明示的に未対応。IQ-TREE 2.3.6でモデル再構成を検証した。他版のcheckpoint/logが非互換ならエラーにする。

この変更はモデル条件付きbootstrapを提供し、従来の診断値の過大解釈を防ぐ。任意のモデル誤指定・部分対立下でのstrong FWER・複数run横断の制御は保証しない。clade交換可能性（ID6）、瞬間Q exposureとendpoint変化の不一致（ID7）、ASR/3Di・site選択など他レビュー項目の妥当性を、この実装だけで解決したとは扱わない。0 exposure候補のno-test化もID7自体の修正ではない。

## 保存物と再現

`scan_id5/inputs/` に実IQ-TREE検証の配列・樹・foregroundを保存した。`validated/` はGY+F、`filtered_called/` は欠損＋自動6群recoding＋called＋filter、`fq/` はGY+FQ＋欠損＋filterの各3反復。`bootstrap99/` はGY+Fの99反復検証。各フォルダーの `example_command.json` は実際の子反復コマンドで、保存時に個人環境のパスを `${RUN_ROOT}`・`${WORKTREE}`・`${USER_HOME}` に置換した。再実行時は入力・出力パスを置き換える。`run.log` には親の設定がある。manifest、生成モデル、最終表も保存した。全反復の中間ファイルは実行環境の一時出力先に残してある。

独立検証の再現コマンド（本worktreeのPython環境で実行）:

```sh
python tools/validate_scan_calibration.py --outdir /tmp/scan-validation-new \
  --reference 9999 --validation 5000 --partial-alternatives 200 \
  --sites 12 --workers 4 --seed 51052026
```

pilotの乱数列は旧flat streamであり、v2の同じseedからは再現しない。監査用にpilotと拡張版の各datasetのseed・score・候補数・診断値の棄却有無をgzip JSONで保持する。

## commit前の追加レビュー

- 極端なexposure比で確率が0/1へ丸められ、有限のrate scoreが未定義になっていた計算を対数空間へ変更した。通常条件の独立尤度計算との一致、1e-300対1e300の境界条件を回帰テストした。
- bootstrapの再較正が失敗した際に、入力表に残った以前のP値を出力しないよう初期化した。負のbootstrap seedはASR開始前に拒否する。
- 全site除外の早期終了でも、無効な候補選択設定を正常終了として扱わないよう、閾値・match等をASR読み込み前に検査する。
- 過去のscan報告書は保存表の列名を維持し、現在の仕様への案内を追記した。
- GY+FQ・欠損・自動6群recoding・called・tip filter・bootstrap表示filterを同時指定し、再び実IQ-TREEで3/3反復を完了。全9候補のP値は保存された帰無最大scoreからの独立再計算と一致し、PDF生成も成功した。

本体の `6b45270`（epistasis）と `71dba8c`（clade配置較正）を統合した最終状態では、**1911 passed / 5 skipped**（55.78秒）。lint・型検査・新規モジュールのmypyも成功した。skipの理由は上記と同じ。全site除外経路を含む入力検証、真の不正scoreと明示的no-testの区別、完全列挙とMonte Carloの異なる分母、複数trait bootstrapの制約分離を追加確認した。

統合後の実IQ-TREE検証でも、2 trait × 2 matchのbootstrapは3/3成功し、4 BH群・18候補のP値をmanifestから独立再計算して一致を確認した。clade完全列挙は28/28配置成功、9候補のmaxT P値を照合し、global P=6/28となった。最終記録は `scan_id5/integrated_bootstrap/`、`integrated_exact/`、`checks/final_*.log` に保存した。

ID6で追加された固定配置空間・観測配置を含む一様抽出・完全列挙は維持した。その較正も新しいscoreへ移行し、ゼロexposureの明示的no-testは配置を除外せずscore −∞として保持する。その他の不正値は較正全体を利用不可にする。較正JSONの `min_p` / `finite_pvalue_count` は `max_score` / `testable_candidate_count` へ移行した。過去の保存記録は実行時のschemaのままである。

本体checkoutへの反映後も、score・bootstrap・clade較正の49テストが通過した。保存ログの空白とTSV末尾の欠損セル表記（NA）は保存時に正規化している。
