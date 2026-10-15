# 追加調査に基づく効率化の実装（2026-10-15）

比較基準は `4382e1f84d01b7bff56855965c4c6e55c3512c51`（1.16.19）、
実装版は1.16.20。[追加調査](../efficiency_followup_20261015/README.md) のうち、
有効な重複削減とメモリ対策を実装した。推定量、科学的既定値、seed、閾値、
CLI・出力形式、永続キャッシュ署名、同梱データ、Linux性能基準は維持する。

## 変更

- scan の状態分布をグループキーごとに一括集計し、9 match に共通する4形状の要約を共有する。
  pandasの加算・表示順を維持し、既定any2speの小さなgroupbyも削減する。
- 反復する割当て校正では固定fit内の候補イベントとendpoint exposureを、それぞれ32 MiB・4,096件以下で再利用する。
  対象・対照の割当てに依存する支持数・率・候補発見は毎回計算する。
  キャッシュはworker転送時に除外し、worker chunkで作り直す。候補固定経路のID解析は
  chunkごと、枝集合の準備は割当てごとに行い、既存の投影も利用する。
  校正しない単一passや各反復でfitが変わる条件では候補ベクトルを保持しない。
- cladeの子孫をpreorder区間で保持し、必要なときだけ従来順の配列・葉名を作る。
  列挙キャッシュは8 MiB以下。校正の重なり判定とeligibility判定で全子孫を展開しない。
- 欠損maskをpostorderで集約する。作業bool領域は約8 MiB以下でサイトを分割し、
  選択枝に必要な部分木だけを読む。非負有限のposterior以外には従来の合計規則を保つ。
- inspectの子孫葉数をpostorderで集約する。root転送の頂点照合には子孫ビット集合を使い、
  内部名、枝長、外部の木署名・キャッシュキーを保つ。
- IQ-TREE状態行の数値検証を256行ずつまとめる。不正なサイト、重複、数値、後続の壊れた行の
  エラー優先順位を保ち、未知状態、gzip、選択ノードも従来どおり処理する。
- 任意の構造重み付けでは、内側CVの訓練集計とinteraction scoreをselection内だけで再利用する。
  合計32 MiB以下で、外側holdout、候補順、tie規則、選択診断を保つ。
- `maximum_score` は有限非負の列を直接集計し、無効値の厳密なno-test/失敗判定だけを行単位で処理する。
- CIに軽量な範囲判定jobを追加する。既知の文書・レポート形式だけの変更では数値・配布laneを省く。
  lint・型・hygiene・文書検証は維持し、ソース、テスト、研究Python、依存、配布、workflow、
  不明なdiff、schedule/manual、major/minor公開では従来の検証を行う。
  文書だけの変更に付随するpatch bumpはinitializerの残りのASTが同一の場合だけ認める。

予測endpointの追加縮約はCPU時間が改善しなかったため採用していない。
永続署名の表現や診断JSONに必要な全clade列挙は互換性を優先して保つ。
使用されていない関数の書き換えや科学的手法・反復数の変更で速度を作ることはしていない。

## 測定方法

macOS27.0.1 arm64、12 CPU、64 GiB、Python3.14.7、NumPy2.5.3、SciPy1.18.1、
pandas3.0.5、ETE4 4.4.0。同じ6個のnative拡張とBLAS1スレッドを両版で使う。
CLIは別のソースディレクトリから新しいプロセスで起動し、ウォームアップ1回、計測3回、
版の順序を交互にする。PGK/PEPCの同じ固定fit、seed8、明示的な校正設定を使い、
IQ-TREEの再推定、モデル取得、プロットは含めない。出力は新しい一時ディレクトリへ書く。

単体比較は旧ソースの対象モジュールと実装版を同じ依存環境へ読み込み、同じ入力を交互に評価する。
各組で表・状態配列・mask・診断・子孫順を照合する。単体時間はCLI全体の倍率ではない。
ホストは共有端末でwall timeの変動が大きいため、CPU timeも記録する。
範囲が重なる差は確実な改善としない。

## 単体比較

CPU時間の中央値、秒。[全反復とwall time](phases.json)。

| 処理 | 現行 | 実装版 | CPU短縮 |
| --- | ---: | ---: | ---: |
| 欠損判定・偏った2,048葉、64サイト、4状態 | 2.6811 | 0.02515 | 99.1% |
| inspect集計・同じ木 | 1.0855 | 0.00524 | 99.5% |
| clade準備・同じ木 | 2.7649 | 0.01282 | 99.5% |
| root頂点照合・同じ木 | 2.4846 | 0.00748 | 99.7% |
| 候補生成・合成4,000イベント・any2spe | 1.1768 | 0.6034 | 48.7% |
| 候補生成・同じ入力・全9 match | 6.8504 | 2.5631 | 62.6% |
| PEPC状態テキストの読み取り・検証 | 0.7980 | 0.5962 | 25.3% |
| 構造重み付け・内側CV、合成80枝×256サイト | 0.6527 | 0.4686 | 28.2% |

2,048葉の偏った木のclade準備では、旧版の子孫NumPy配列67,076,096 bytesに対し、
新区間indexのNumPy配列は180,160 bytesだった。これは配列ペイロードの比較であり、
Pythonオブジェクト、必要時に作る列挙キャッシュ、解析全体のRSSを含まない。
全cladeの出力を要求する診断JSON自体のサイズは縮小していない。

初回のCLI比較では全match・校正なし条件でRSS増加が見られたため、保持の有効性を調べ直した。
811候補についてイベント・exposureとも811回の呼び出しに対し再利用は0回で、
合計14,882,112 bytesのペイロードを不要に保持していた。[診断記録](cache_policy.json)。
このため単一passのキャッシュを無効にし、配布物・関連検証・CLI比較を最終版でやり直した。

平衡木、root照合の結果、10,000候補のmaximumも `phases.json` に保存する。
通常の小さな候補集合にmaximumの単体倍率を当てはめない。

## CLI全体の比較

各時間は中央値（最小–最大）、RSSは中央値。[全64実行、比較結果、引数、入力とソースのハッシュ](cli.json)。

| 条件 | 時間 before（秒） | 時間 after（秒） | 中央値増減 | RSS before→after（MiB） |
| --- | ---: | ---: | ---: | ---: |
| PEPC scan・校正なし | 4.178 (4.146–4.209) | 3.934 (3.908–4.338) | -5.8% | 563.0 → 553.7 |
| PEPC scan・全9 match・校正なし | 7.241 (6.966–7.528) | 6.522 (6.353–6.976) | -9.9% | 570.8 → 552.8 |
| PEPC scan・候補固定10反復 | 6.780 (6.110–8.767) | 5.179 (5.000–5.416) | -23.6% | 562.0 → 559.1 |
| PGK scan・full_scan20・1プロセス | 3.655 (3.598–3.850) | 3.147 (3.101–4.006) | -13.9% | 303.5 → 310.1 |
| PGK scan・full_scan20・2プロセス | 4.332 (4.202–5.189) | 3.836 (3.663–4.560) | -11.5% | 337.7 → 323.1 |
| PEPC 全統計検索 | 7.993 (7.758–8.365) | 8.328 (7.462–8.337) | +4.2% | 493.1 → 491.2 |
| PEPC marginal検索 | 3.278 (3.266–3.516) | 2.960 (2.919–3.216) | -9.7% | 485.5 → 457.8 |
| PEPC sites・2枝 | 3.244 (3.168–4.770) | 2.788 (2.673–3.091) | -14.1% | 485.6 → 479.4 |

64実行が正常終了し、32組の主結果表を全列・型・行順ごとに `rtol=1e-8, atol=1e-9` で照合した。
scanの5条件では校正・推論JSONも全20組で一致した（バージョン情報を除く）。
2プロセス条件は全反復の診断により実際の `n_jobs=2` も確認した。

候補固定校正はこの測定で約24%短縮した。marginal検索、sitesも測定範囲の重ならない短縮だった。
既定scan、全match、PGKの両プロセス条件は中央値では短縮したが範囲が重なるため、
一定の短縮率とは断定しない。全統計検索の+4.2%も範囲が重なり、改善・悪化は断定しない。
RSSはworker全体の物理メモリの合計ではなく、この結果から解析全般のRAM削減は主張しない。

CLI測定後に、スコア列のない不正な候補行を確実に拒否する空判定のガードを調整した。
正常入力の計算は同じで、専用回帰テストとsource/wheelの検証をやり直し、maximumの単体計測も更新した。
`cli.json` には測定時と最終版の当該ファイルのハッシュを区別して保存する。

## 検証と配布

Python3.12.14のソース環境、および別の新しい非editable wheel環境で検証した。
NumPy2.5.4、SciPy1.18.1、pandas3.0.6、Matplotlib3.11.2、ETE4 4.4.0。
配布物は別の一時checkoutで `make package` を実行し、sdistの生成Cからもwheelを再構築する。
wheel検証はcheckout外で `CSUBST_TEST_INSTALLED=1` とprefix guardを使う。

| チェック | 結果 |
| --- | --- |
| ソース `make test-fast` | 1,662 passed、6 skipped |
| ソース `make test` | 通常2,136 passed・9 skipped、process5 passed |
| ソースnative | 15 passed |
| ソース・拡張無効の関連選択 | 327 passed、native専用1 skipped |
| ソース `make lint typecheck` | 成功、型検査25対象 |
| source固定fitの数値参照 | PGK/PEPC成功 |
| `make package`、sdistからのwheel再構築、artifact内容検査 | 成功 |
| 新環境のinstalled wheel全体・process・native | 通常2,136 passed・9 skipped、process5 passed、native15 passed |
| installed wheel固定fitの数値参照 | PGK/PEPC成功 |
| workflow | actionlint1.7.12とYAML依存グラフの検査成功 |

正確なコマンド、skip、配布物とソースのハッシュは [verification.json](verification.json) に保存した。
固定fitの参照値は [source](source_parity.tsv) と [installed wheel](wheel_parity.tsv) にある。

9 skipはtorch不在4件、gemmi不在2件、実IQ-TREE不在3件。
既存のMatplotlib `set_bad` のPendingDeprecationWarningが8件ある。
新しいwheel環境の最初の検証は、ETE4のキャッシュバイナリが `PyInit_tree` を公開せず失敗した。
同じ4.4.0を `--no-cache-dir --no-binary=ete4 --force-reinstall --no-deps` で再構築し、
importと全installed検証をやり直した。CSUBSTの依存制約は変えていない。
actionlintの外部shellcheck呼び出しは無効にしており、shellcheck自体の検証には数えない。

新規の回帰検証には、枝IDが非連続な木、単分岐・多分岐、列挙順と葉名順、
lazy/dense samplerの同じseedでの抽選と試行数、キャッシュの上限とpickle、
新しいfitへの非持ち越し、割当て別率の再集計、選択枝とエラー優先順位、CIの範囲判定を含む。
科学的参照値や許容誤差は変更していない。

CSUBSTはCLI配布であり、常駐サービスのデプロイ設定はない。
`RELEASING.md` に従い、patch版はmasterで公開し、手動tag・GitHub Release・Bioconda更新は作らない。
push後のPytestと公開workflowを別途確認する。

## 再現

`BASELINE_ROOT` は4382e1fのソース、`CANDIDATE_ROOT` は実装版のリポジトリルート。
同じnative拡張を両版に用意し、同梱データを基準版へ置く。CLI測定にはpsutilとmacOSのtimeを使う。

```bash
python reports/efficiency_implementation_20261015/benchmark.py phases \
  --baseline-root "$BASELINE_ROOT" --result /tmp/csubst-implementation-phases.json
python reports/efficiency_implementation_20261015/benchmark.py cli \
  --baseline-root "$BASELINE_ROOT" --candidate-root "$CANDIDATE_ROOT" \
  --baseline-label 4382e1f --candidate-label 1.16.20 \
  --cases pepc_scan_defaults pepc_scan_all pepc_fixed10 pgk_scan_calibration20 \
    pgk_scan_parallel20 pepc_full pepc_marginal pepc_sites \
  --workdir /tmp/csubst-implementation-cli --result /tmp/csubst-implementation-cli.json
python reports/efficiency_implementation_20261015/benchmark.py check-results /tmp/csubst-implementation-cli.json
python reports/efficiency_implementation_20261015/benchmark.py cache-diagnostic \
  --workdir /tmp/csubst-cache-diagnostic --result /tmp/csubst-cache-diagnostic.json
```

`cache-diagnostic` は調査当時の常時キャッシュを実験プロセス内だけで強制して再現する。
通常の校正なし解析ではこの保持を行わない。

研究用の合成測定や固定fitの数値一致は、新しいfit、1,000反復の科学的再校正、
実GPU/3Di/VESM推論、未実行のLinux性能基準の証明ではない。
