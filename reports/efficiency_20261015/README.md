# リポジトリの実行効率調査（2026-10-15）

基準コミットは `5df0919b9164652f73174027faaf7896d6749d3b`。
通常解析の重複処理と、大きな木で増える準備コストを改善した。
科学的な既定値、推定量、乱数、閾値、出力形式、参照データは変更していない。

## 調査範囲と判断

CLI・設定、入力パーサー、木、joint endpoint、置換テンソルと疎行列、
omega、scan、並列実行、サイト保存、出力、構造解析の主要経路と既存テストを横断確認した。
PGK/PEPC の実データでは、検索・scan・sites の4条件を cProfile で調べた。
[診断用プロファイルの要約](profile_summary.json) は測定環境の負荷と
プロファイラのオーバーヘッドを含む。累積時間は入れ子なので合計しない。

- PEPC の全統計検索では、joint 推定を含む状態準備が支配的だった。
  既にブロック処理、Cython のイベント集約、疎な投影、容量制限付きキャッシュを使う。
  今回はこの数値カーネルを変更していない。
- PEPC scan は候補評価に約6.24秒、PGK の20回校正付き scan は
  候補評価に約20.26秒を使った。後者ではイベントマスクに約5.02秒、
  支持数集計に約4.66秒かかり、候補間・候補内の重複処理が確認できた。
- sites は選択枝のイベントと集計専用の他枝を分け、入力・保存領域も既に削減されている。
  今回の変更による全体時間の改善は測定結果に限定して解釈する。
- 大きな木のラベル付けは、各ノードから子孫の葉を再走査していた。
  EndpointModel の構築も、訪問済みノードのリスト検索を繰り返していた。
  どちらも推定式を変えずに改善できる。
- 任意モデルのダウンロード、GPU/3Di/VESM の実推論、IQ-TREE の再推定は
  この測定に含めない。既存の研究レポート、Linux 性能基準、CI、テスト選択は変更していない。

## 実装

1. **scan の共通情報を再利用。** foreground の枝リストを候補ごとに文字列から
   解析せず、trait と割当てごとに一度だけ準備する。対象・対照の枝集合も同じ単位で準備する。
   割当てをまたぐキャッシュにはしていない。
2. **不要な候補処理を削減。** 支持数の判定と出力に同じ集計結果を使う。
   支持条件を満たさない候補の posterior rate イベントを読み出さない。
   イベントマスクは NumPy で対象サイトの状態だけ比較し、候補行の pandas Series 生成も省く。
3. **木の準備を改善。** 子孫ビット署名を postorder で子から集約し、ラベルの順位と
   同順位の順序を保つ。EndpointModel の訪問済み判定には set を使い、走査順序は保つ。
   Python の任意精度整数を維持し、単分岐・多分岐・64葉以上も扱う。

## 測定条件と再現

macOS 27.0.1 arm64、12 CPU、64 GiB、Python 3.14.7、NumPy 2.5.3、
SciPy 1.18.1、pandas 3.0.5。両版で同じ6個の native 拡張を使う。
基準版は一時ディレクトリへ保存し、すべての解析出力も新しい一時ディレクトリへ書いた。
ウォームアップ1回と計測3回。CLI は before/after の実行順序を交互にする。
CSUBST は1プロセス（並列校正条件のみ2）、BLAS は1スレッド、seed は8。
校正の seed は既定の1、反復数は20。既定の1,000反復全体を測ったものではない。

PGK は33配列×417コドン、PEPC は71配列×971コドン。
PEPC×4 は配列とサイト情報を4回繰り返した3,884コドンのサイズ試験であり、
独立した生物学的標本ではない。固定の同梱 IQ-TREE fit を再利用し、
モデル・枝長の再推定、ネットワーク、プロットは性能測定から除外する。

CLI 全体の wall time と `/usr/bin/time -l` のピークRSSを記録する。
RSS は worker の使用量を足した総物理メモリではない。
同じ端末の他の処理も動いているため、測定範囲が重なる差は改善と断定しない。
scan 単体では同じ準備済み入力を両実装へ交互に渡し、wall time と process CPU time を記録する。
木の単体測定は import と Newick の読み込みを時間に含めない。

基準版と候補版を別ディレクトリに用意し、同じ環境・native 拡張を使って再実行する。
`BASELINE_ROOT`、`CANDIDATE_ROOT` はそれぞれのリポジトリルートを指定する。
基準版の `csubst/dataset/` には同梱データを配置する。
ベンチマーク用の依存パッケージ `psutil` も必要。

```bash
python reports/efficiency_20261015/benchmark.py cli \
  --baseline-root "$BASELINE_ROOT" --candidate-root "$CANDIDATE_ROOT" \
  --baseline-label 5df0919 --candidate-label efficiency-20261015 \
  --cases pepc_scan_defaults pgk_scan_calibration20 pgk_scan_parallel20 \
    pepc4_scan_defaults pepc_full pepc_sites \
  --workdir /tmp/csubst-efficiency-cli --result /tmp/csubst-efficiency-cli.json
python reports/efficiency_20261015/benchmark.py check-results /tmp/csubst-efficiency-cli.json
PYTHONPATH="$CANDIDATE_ROOT" python reports/efficiency_20261015/benchmark.py scan-core \
  --baseline-root "$BASELINE_ROOT" --workdir /tmp/csubst-efficiency-core \
  --result /tmp/csubst-efficiency-core.json
python reports/efficiency_20261015/benchmark.py trees \
  --baseline-root "$BASELINE_ROOT" --candidate-root "$CANDIDATE_ROOT" \
  --result /tmp/csubst-efficiency-trees.json
```

`scan-core` は候補版を import できる環境・作業ディレクトリから実行する。
主結果TSVは全列・行順・型を `rtol=1e-8, atol=1e-9` で照合する。
校正・推論JSONはバージョン情報以外を比較する。
scan 単体の結果は全列を厳密一致で比較し、木はラベルと走査順序の SHA-256 を照合する。
実行時間などの動的メタデータは結果表の同値比較に含めない。

完全なCLI引数、入力・ソースのハッシュ、全計測値と表の比較結果は
[CLI測定記録](results.json) に保存する。

## CLI全体の測定結果

各セルは中央値（最小–最大）。時間は秒、RSSはMiB。

| 条件 | 時間 before | 時間 after | 時間増減 | RSS before | RSS after |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEPC scan・校正なし | 13.76 (11.20–14.44) | 14.43 (10.73–15.36) | +4.9% | 565.16 (539.48–600.41) | 591.86 (559.36–602.81) |
| PGK scan・20回校正・1プロセス | 19.56 (19.43–35.52) | 14.41 (13.98–14.79) | -26.4% | 335.17 (327.41–347.20) | 319.09 (312.86–328.09) |
| PGK scan・20回校正・2プロセス | 15.50 (14.03–17.83) | 10.56 (10.15–11.84) | -31.8% | 316.91 (315.89–330.41) | 325.08 (321.12–341.27) |
| PEPC×4 scan・校正なし | 24.07 (17.21–25.96) | 17.65 (16.12–21.79) | -26.7% | 1397.03 (1392.47–1423.53) | 1421.06 (1398.08–1515.83) |
| PEPC 全統計検索 K=2 | 10.26 (10.12–10.49) | 11.47 (9.19–12.11) | +11.7% | 509.03 (498.02–518.30) | 498.50 (479.95–504.03) |
| PEPC sites・2枝 | 4.20 (3.88–4.58) | 3.99 (3.70–4.95) | -4.8% | 501.39 (483.47–526.81) | 498.83 (485.44–520.33) |

PGK の20回校正は1プロセスで約26%、2プロセスで約32%短縮した。
2プロセス条件は校正JSONの `n_jobs=2` でも確認した。
PEPC×4 は中央値で約27%短縮したが、範囲が重なるため再現する削減率とは断定しない。
通常PEPC scan、全統計検索、sites の差も測定範囲が重なり、速度の改善・悪化を断定しない。
今回の結果から総RAMの低減は主張しない。6条件・全48実行は正常終了し、
主結果の全列と scan の校正・推論JSONが同値だった。

## 処理単体の測定結果

同じ準備済み入力を交互に評価した結果。各値は3回の中央値。
[全測定値と厳密な表の一致確認](scan_core.json)。

| scan 候補評価 | wall before → after（秒） | CPU before → after（秒） | CPU短縮 |
| --- | ---: | ---: | ---: |
| PGK | 0.281 → 0.170 | 0.250 → 0.159 | 36.5% |
| PEPC | 1.424 → 0.818 | 1.184 → 0.646 | 45.4% |

以下は木の準備のみの測定。[全測定値とラベル・走査順序のハッシュ](trees.json)。

| 準備処理 | before（秒） | after（秒） | 倍率 |
| --- | ---: | ---: | ---: |
| ラベル付け・平衡木2,048葉 | 0.0511 | 0.0054 | 9.5× |
| ラベル付け・偏った木2,048葉 | 1.1379 | 0.0104 | 109.0× |
| EndpointModel・2状態16,384ノード | 1.7052 | 0.0084 | 202.4× |

木の倍率は準備処理に対する値であり、解析全体の倍率ではない。
木のラベル付けは子孫の再走査を省くが、任意精度整数のコストと順位付けのソートは残る。
測定時の数値コードは公開版と同じで、公開時にはバージョン定数のみ更新する。

## 検証

最終検証は別の一時環境の Python 3.12.14 で行った。
NumPy 2.5.4、SciPy 1.18.1、pandas 3.0.6、Matplotlib 3.11.2、ETE4 4.4.0。
これは上記の性能測定環境とは別である。コマンド、結果、skip理由は
[検証記録](verification.json) に保存した。

| チェック | 結果 |
| --- | --- |
| Python 3.14 の変更箇所に絞った4モジュール | 130 passed |
| `make test-fast` | 1,632 passed、6 skipped |
| `make lint` | Ruff・hygiene・文書チェック成功 |
| `make typecheck` | 対象25モジュール成功 |
| `make test` | 並列レーン2,102 passed・9 skipped、直列processレーン5 passed |
| 6拡張の直接importと `make test-native` | import成功、15 passed |
| `CSUBST_DISABLE_EXTENSIONS=1` の関連unit/integration/parity選択 | 283 passed、native専用7 skipped |
| `sites_parity_check.py --numerical-only` | PGK/PEPC の検索・sites参照値成功 |
| ベンチマーク用スクリプトの Ruff | 成功 |

全体テストの9 skipは torch不在の4件、gemmi不在の2件、実IQ-TREE不在の3件。
該当する外部バックエンドの検証には数えない。
Matplotlib の `set_bad` に関する既存の PendingDeprecationWarning が8件出た。
最初の Python 3.12 テスト起動はキャッシュ由来の ETE4 バイナリが
`PyInit_tree` を公開せず失敗した。同じ ETE4 4.4.0 をソースから再構築して
importを確認後、失敗した検証と残りの全チェックを実行した。
リポジトリの依存条件は変更していない。

固定fitによる独立した数値参照は、PGK の `omegaCany2spe=1.910709`、
収束5・分岐7・空白405、PEPC の `omegaCany2spe=0.030734`、
収束0・分岐2・空白969と一致した。これは再校正や新しいIQ-TREE fitの検証ではない。
installed-wheel、sdist-only、sanitizer、Linux性能基準、その他のPython版のCIレーンは
ローカルでは実行していない。モデルダウンロードや任意構造モデルの実推論も対象外。
