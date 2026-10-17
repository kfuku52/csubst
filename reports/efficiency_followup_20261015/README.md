# 追加の効率調査（2026-10-15）

対象は、前回の改善を公開した `4382e1f84d01b7bff56855965c4c6e55c3512c51`
（1.16.19）。今回も改善余地が確認できた。特に **候補固定校正の重複計算**、
**scan の候補生成**、**偏った木の欠損判定・子孫情報**を優先する。
今回は調査と試作比較であり、`csubst/`、科学的既定値、テスト、CI は変更していない。
試作はこのレポート内に限定し、製品コードへの組み込みや追加の公開は行っていない。

## 優先順位

時間は各処理単体の中央値。CLI 全体の短縮率ではない。
「試作」は今回の入力で比較した実験コードであり、公開に必要な全検証を終えた実装ではない。

| 優先度 | 対象 | 確認結果 | 適用条件・残る課題 |
| --- | --- | --- | --- |
| 高 | scan 候補固定校正 | PEPC・10反復の scan 本体が 2.832 → 1.756秒、38.0%短縮。CPU は37.6%短縮 | 割当てに依存しないイベントと exposure のみ再利用。プロセスごとの容量制限と無効化が必要 |
| 高 | scan 候補生成 | PEPC・全9 match の生成が 0.907 → 0.670秒、26.1%短縮 | 同じ形のグループ集計を再利用。既定の any2spe 単独にはこの削減率を適用できない |
| 高・大きな木 | 入力の欠損判定 | 2,048葉の偏った木で 2.657 → 0.0225秒、約118倍 | 64サイト・4状態の合成入力。選択枝APIとメモリ制限を組み込み時に検証 |
| 高・大きな木 | clade permutation の子孫キャッシュ | 同じ木で NumPy 子孫配列だけで63.97 MiB、葉名参照209.8万個 | 二次増大を実測。区間表現への変更は設計段階で、削減後RSSは未測定 |
| 中 | inspect の子孫葉数 | 同じ木で 0.695 → 0.00546秒、約127倍 | postorder 集約の試作で全出力行とノード対応が一致 |
| 中 | full-scan 校正の候補集計 | PGK・20反復で候補生成21回、診断プロファイルの累積2.97秒 | 既定 match の小さな groupby を一括集計する余地。最終実装・改善率は未測定 |
| 低〜中 | 内側CVの訓練集計 | 合成入力で CPU 0.642 → 0.519秒 | オプションの構造重み付けのみ。外側fold間で共有しない |
| 要設計 | 文書だけ変更した場合のCI | 現状でも数値・配布・sanitizer 等の通常10ジョブを起動 | 必須check名、Pytest完了を待つ公開処理、文書チェックのカバー範囲を維持する条件分岐が必要 |

## 調査範囲と測定方法

前回確認した CLI、設定、入力、木、joint endpoint、置換テンソル、omega、
scan、プロセス並列、保存・出力に加え、残る候補集計、両校正モード、
欠損判定、inspect、clade permutation、構造重み付け、Actions の実行依存を追った。
関数に重い処理があるだけでなく、実運用の呼び出し元があるかも確認した。

macOS 27.0.1 arm64、12 CPU、64 GiB、Python 3.14.7、NumPy 2.5.3、
SciPy 1.18.1、pandas 3.0.5、ETE4 4.4.0。同じ checkout と6個の native 拡張を使用。
BLAS は1スレッド。PGK は33配列×417コドン、PEPC は71配列×971コドン。
同梱の固定 IQ-TREE fit を明示し、再推定・ネットワーク取得・プロットは除外した。
乱数や match、校正の既定値を変えて速度を比較することはしていない。

6条件を新しいプロセスの cProfile で調べた結果は [profiles.json](profiles.json) にある。
これは呼び出し回数と負担箇所の診断用であり、プロファイラ自体と import の負担を含む。
入れ子の累積時間を足したり、別条件間の速度比較に用いたりしない。
入力ファイルの SHA-256 と全CLI引数も保存した。

試作比較は準備済み入力に対してウォームアップ1回、計測3回、現行版と試作の順序を
交互にする。wall time と process CPU time を記録し、各組で出力を照合する。
同じ端末の他の処理による変動がある。小さい差や範囲が重なる結果を確実な改善とはしない。
ピークRSSの比較は今回行っていない。キャッシュのバイト数は保持配列・DataFrameの
ペイロードであり、Pythonコンテナや全プロセスの物理メモリを含まない。

## 1. 校正中の不変量を再利用する

該当箇所は `substitution_scan.py:2484` の `_candidate_fixed_permutation_scores`、
同 `:1756` の `_rate_summary`、`scan_endpoint.py:133` の `expected_events`。
現在は候補×反復ごとに、同じサイト・from/to集合の posterior イベント抽出と
endpoint exposure 計算をやり直す。後者には親・子の4つの状態配列の検証と、
遷移・カテゴリーの集約も含まれる。

対象・対照の枝割当ては毎回変わる。一方、固定fit、状態、候補、eligibility が同じなら
枝ごとのイベント・exposure ベクトルは不変なので、割当ての集計だけを毎回行える。
試作では各 scan 呼び出し内だけ、イベントと exposure をそれぞれ32 MiB以下のLRUに置いた。

| 条件 | wall 現行→試作（秒） | CPU 現行→試作（秒） | 判定 |
| --- | ---: | ---: | --- |
| PEPC・candidate_fixed・10反復 | 2.832 → 1.756 | 2.620 → 1.636 | 明確な改善 |
| PGK・full_scan・20反復 | 1.830 → 1.772 | 1.809 → 1.760 | 範囲が重なる小差。改善率を一般化しない |

PEPC ではイベント抽出1,133回のうち1,030回、exposure 1,133回のうち927回が再利用できた。
保持したペイロードはイベント302,396 bytes、exposure 3,028,200 bytes。
scan 結果表、unit表は全列・型・順序を厳密一致で照合し、校正診断JSONも一致した。
測定値・各反復の保持量は [scan.json](scan.json) にある。

組み込む場合は次を守る。

- `_build_scan_static_context` の寿命内か、同等の worker 内スコープに限定する。
  新しい fit、状態・サイトフィルター、bootstrap、遷移モデルへ持ち越さない。
- target/control を含む最終 rate 結果や、full-scan の発見済み候補を別割当てへ使い回さない。
  候補発見、支持数、空集合の扱い、null反復数、seed、診断を維持する。
- キャッシュ量は worker 数で増える。今回の64 MiB上限のままでも全worker総量ではない。
  容量到達・無効化・dense/sparse/disk-backed・実workerで検証する。
- 候補固定経路には、候補ごとのID文字列解析、traitのunit抽出、対象・対照集合生成、
  既存の `rate_event_projection` を使わない抽出も残る。これらは別途まとめられる。

## 2. 候補生成の小さな groupby を減らす

`substitution_scan.py:1301` の `build_candidates` は9種類の match を順に処理する。
実際のグループキーは `(site, from, to)`、`(site, to)`、`(site, from)`、`site` の4種類。
同じグループについて状態集合、from/toの分布文字列、イベント質量を繰り返し作る。
状態分布も候補ごとに小さな `groupby().sum().sort_values()` を2つ作っている。

4種類のグループ要約を再利用する試作では、PEPC の719イベントを入力した全match候補表が
0.907 → 0.670秒、CPU 0.902 → 0.667秒になった。全列・行順・型を厳密一致で確認した。
各match、空表、単一イベント等の追加16比較も一致した。

既定の `any2spe` 単独ではグループ形状の共有は効かない。
PGK full-scan20 の診断では `build_candidates` が21回・累積2.97秒で、
scan本体21回・累積5.46秒の大きな部分を占めた。
`(site,to,from)` の分布を一括集計し、状態集合と表示文字列をまとめて作るのが次の候補。
ただし浮動小数点の加算順、同質量の表示順、全matchの候補順は維持して検証する。
この追加案の実装効果はまだ測っていない。

## 3. 木の子孫を繰り返し走査する経路

`parser_iqtree.py:896` の `get_internal_site_nonzero_masks` は各内部ノードの
子・姉妹cladeについて、全子孫の葉を集め、葉×サイト×状態の配列をコピー・合計する。
偏った木では同じ大きな部分木を何度も読む。
葉の非欠損boolを postorder でOR集約すれば、有効な非負のposteriorでは同じ判定を作れる。
これは marginal の入力や inspect 等で使われる経路であり、既定jointの全解析に
同じ短縮率が生じるわけではない。

`main_inspect.py:41` の `_collect_branch_rows` も各ノードから葉を数え直している。
子の葉数を postorder で足す試作で、全出力行とノード対応を保てた。

| 処理・2,048葉 | 平衡木 現行→試作（秒） | 偏った木 現行→試作（秒） |
| --- | ---: | ---: |
| 欠損判定・64サイト・4状態 | 0.0805 → 0.0179 | 2.657 → 0.0225 |
| inspect の枝行生成 | 0.0139 → 0.00550 | 0.695 → 0.00546 |

[synthetic.json](synthetic.json) に512葉も含む全計測値を保存した。
欠損判定の標準試作は非負・有限の全状態入力を対象とする。
単分岐、多分岐、全欠損、部分欠損の追加6比較は一致したが、選択内部枝・
`leaf_nonmissing_sites` のAPI全体を置き換えたものではない。
製品実装では既存の選択枝、root挿入、欠損規則を保ち、大きなサイト数では
ノード×サイトboolの常駐量も制限する必要がある。

## 4. clade permutation の二次メモリ

`foreground.py:83` の `_build_trait_clade_permutation_cache` は、各枝について
子孫branch ID配列、子孫index配列、子孫葉名tupleをすべて保持する。
`scan_permutation.py:57` の `build_plan` はさらに子孫を frozenset にする。
同じ配列を参照するdict/listは二重計上せず、元のNumPy配列だけを数えた。

| 木 | 子孫NumPy配列 | 葉名への参照数 | 構築wall（単回、秒） |
| --- | ---: | ---: | ---: |
| 平衡・512葉 | 0.125 MiB | 4,608 | 0.0096 |
| 平衡・2,048葉 | 0.625 MiB | 22,528 | 0.0380 |
| 偏り・512葉 | 3.992 MiB | 131,327 | 0.135 |
| 偏り・2,048葉 | 63.97 MiB | 2,098,175 | 2.584 |

後者にはPythonオブジェクト、tupleの参照領域、後続のfrozensetを含めていない。
構築時間は診断用の単回値で、速度比較の根拠にはしない。
ノード数を4倍にすると偏った木の配列量が約16倍になる。
Euler順の開始・終了位置で子孫集合を表し、必要時だけ列挙する設計なら、
祖先関係・重なり判定のために全子孫を保持する必要はない。

ただし現在の列挙順、枝ID、均一な非重複clade抽出、exact計数、provenance JSONを維持する
設計が必要。JSON自体に全候補cladeを出す部分は列挙サイズが下限になる。
`synthetic.json` の「3整数/ノード」は区間表現の設計上の配列見積りであり、
同等APIの実装や測定済みメモリ削減率を表さない。

## 5. 任意の構造重み付け

`epistasis.py:167` の `_select` は beta/alpha/clip の各候補と各内側foldで
同じ訓練カウントを再集計する。`context @ features.T` も beta/alpha/clip に依存しない。
fold内の訓練集計をalphaごと、interaction scoreをfoldごとに再利用できる。

80枝×256サイト×2特徴、5fold、744候補の有効な合成入力で比較した。
内側selectionだけで wall 0.702 → 0.576秒、CPU 0.642 → 0.519秒。
base計算3,720回のうち3,700回を再利用し、base配列655,360 bytes、score配列163,840 bytesを保持した。
選択されたパラメータと内側スコア・全fold診断は各組で厳密一致した。
[structure_cv.json](structure_cv.json) に全測定値を保存した。

これは既定では無効な機能の小規模合成試験であり、実CLIの短縮率ではない。
キャッシュは一度のselection内に限定する。外側holdoutの応答を共有せず、fold、
mask、alpha、候補順と1e-12のtie規則を保つ。
`docs/STRUCTURE_WEIGHTING.md` の独立性・予測上の意味は変更しない。

## 6. 入力・保存・CIの次候補

- **ASR状態テキストの読み取り**：`parser_iqtree.py:938` は1行ずつ数値を変換・検証する。
  PEPC marginal の診断では1回1.19秒、状態入力全体は1.30秒だった。
  少量バッチ化の候補だが、実装・速度は未測定。最初の不正行、重複サイト、
  skip対象ノードの検証、整数サイト文法とエラー順を保持する必要がある。
- **disk-backed scan**：`extract_candidate_posterior_events` は投影で回答できる場合にも、
  先に `SiteEventTensor.read_site` を呼ぶ。候補固定経路も既存投影を渡していない。
  読み取りの後回しと投影再利用は候補になる。今回の計測からI/O全体の改善率は主張しない。
- **木の署名・root対応**：`tree.py:165,183,192` に子孫列挙が残る。
  一部は IQ-TREE/3Di の永続キャッシュ署名に使われるので、単に表現を変えると互換性に影響する。
  外部署名を保つ内部indexや、署名計算の限定的再利用から検討する。
- **CIの変更範囲判定**：`.github/scripts/ci_scope.py` はmacOSの要否だけを判定する。
  文書・レポートだけの変更でもLinuxの数値・配布・sanitizer等は動く。
  今回の公開コミットの [成功run](https://github.com/kfuku52/csubst/actions/runs/38690129612)
  は実行10ジョブ合計1,027秒、完了まで約4分だった。[ジョブ記録](ci.json)。
  合計は並列ジョブの経過時間であり、課金分数でもそのまま削減できる時間でもない。
  文書専用変更の分岐を導入するなら、文書・hygiene・必要な静的検証と各checkの報告を残し、
  数値コード・依存・native・package・workflow変更、schedule、manualでは所要の検証を保つ。
  Pytest成功後に動く `tag_from_version.yml` との依存も確認が必要。

## 優先度を下げた案

- **predictive endpoint の追加縮約**：全統計PEPC検索の endpoint 構築に、
  32 MiB制限のgroup kernelを試作した。wall は12.739 → 11.446秒だが範囲が重なり、
  CPU は7.757 → 8.389秒と増えた。改善とは判断しない。
  observedの全pairwise値・枝サイト集計、predictiveの全pairwise値を
  `rtol=1e-12, atol=1e-13`、manifestを厳密一致で確認した。
  [endpoint.json](endpoint.json)。既定の投影経路には既にpredictive kernelの再利用がある。
- **maximum_score のベクトル化**：1,000行で13.7 ms、10,000行で150 msだった行Series生成を
  約0.04 msにできた。ただし通常の小さな候補集合では絶対時間が小さい。
  不正スコアの失敗とzero-exposure no-testを維持する必要があり、単純な `nanmax` では代用できない。
- **使われない処理の高速化**：`transfer_internal_node_names` はテスト以外の呼び出しがなく、
  `calc_substitution_patterns` の主経路呼び出しもコメントアウトされている。
  これらを速くしても今回の実解析には効かないため、実装優先度を下げる。
- **既存のCI共有・並列化の重複提案**：Python3.12 wheel/sdistのrun内共有、
  concurrencyのキャンセル、macOSの条件選択、pip cacheは既にある。
  pushはmasterだけなので、通常の作業枝でpush/PR双方を起動する重複も生じない。
  各Python版、fallback、配布物、sanitizer、数値parityは異なる保証を持つので一律に省かない。

## 再現と検証範囲

同じ開発環境でリポジトリルートから実行する。`--workdir` は存在しない新しいパスを指定する。
出力は一時ディレクトリに置き、同梱fixtureには書き込まない。

```bash
python reports/efficiency_followup_20261015/audit.py --mode scan \
  --workdir /tmp/csubst-followup-scan --result /tmp/csubst-followup-scan.json
python reports/efficiency_followup_20261015/audit.py --mode synthetic \
  --workdir /tmp/csubst-followup-trees --result /tmp/csubst-followup-trees.json
python reports/efficiency_followup_20261015/audit.py --mode endpoint \
  --workdir /tmp/csubst-followup-endpoint --result /tmp/csubst-followup-endpoint.json
python reports/efficiency_followup_20261015/structure_cv.py \
  --result /tmp/csubst-followup-cv.json
python reports/efficiency_followup_20261015/profile_paths.py \
  --workdir /tmp/csubst-followup-profile --result /tmp/csubst-followup-profile.json
python reports/efficiency_followup_20261015/check_prototypes.py
python -m ruff check reports/efficiency_followup_20261015
make lint
git diff --check
```

`audit.py` は現行のresource benchmarkのCLI引数生成器を読み込む。
すべての試作差し替えは実験プロセス内で復元する。
プロファイル6条件、scan/candidate/木/endpointの比較、構造CV比較は正常終了した。
追加の同値確認は候補表16件、欠損mask6件、maximumの値・失敗8件、
predictive変換9件、合計39件が成功した。
レポート用スクリプトのRuff、`make lint`、新規ファイルも含む差分の空白検査は成功した。
チェック結果と保存ファイルのハッシュは [verification.json](verification.json) にある。

本体に変更がないため、前回実行したフルテストや数値parityは今回再実行していない。
今回の比較はソース実行であり、installed wheel、科学的再校正、1,000反復校正、
新規IQ-TREE fit、実GPU/3Di/VESMモデル、Linux性能、実CIの条件変更は検証していない。
プロファイルで小さかった経路や、外部モデル推論の未測定経路に、ボトルネックが存在しないと
断定するものではない。
