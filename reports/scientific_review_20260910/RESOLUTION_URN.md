# ID 12: urn / ASRV / automatic recoding

2026-09-10。ユーザー承認後の実装と検証記録。
作業用worktreeの基点は `dd37bee67b39199a1920600cf00c0ecb8566f477`。
以下は主checkoutへ取り込む前の実装・検証記録。
この実装段階では主checkoutを読み取り専用で参照した。

## 実装した範囲

- ASRVの学習枝を `all`、`background`、明示的な枝ID集合から選択可能にした。
  背景指定では全traitのforeground/marginal targetの和集合と根を除く。
  学習するサイトmassのみを変え、評価対象枝の総mass・観測カウントは保持する。
- 既存のサイト当たりalphaに加え、固定総量の一様prior
  `--asrv_concentration` を実装した。既定値は変更していない。
  `sn/each/file_each`のみが対象で、サイト当たりalphaを置き換える。
- 新しい学習集合／総濃度指定とepistasisの併用は理由を示して拒否する。
  epistasis側の学習はまだ全枝とサイト当たりalphaを使うため、ID 3で
  整合させるまで「背景だけからの学習」として提供しない。
- 自動recodingへ別の学習用コドンアラインメントを渡せるようにした。
  キャッシュはファイル内容のSHA-256と遺伝暗号を含めて検証する。
  同じファイル名を使う反復でも古い群分け用統計を再利用しない。
- Walleniusの1／2抽出と一様重みを厳密に扱い、残る列挙範囲は従来の
  正重み20サイト以下に維持した。`exact`指定では範囲外の近似を拒否する。
  重みスケールに依存する数値問題と、有限重みの和のoverflowも検証した。
- urn provenance JSONとrecoding metadata JSONを追加した。
  詳細ASRV診断は `--asrv_report yes` で記録する。
- 独立CTMCデータ生成器と解析全体の検証実行器を追加した。
  各反復でIQ-TREE、ASR、recoding、ASRV、探索をやり直し、宣言した
  カウント閾値・枝除外・複数構成からの最大ωC選択を再現する。
  カウント帰無の `pomegaC*` は生成せず、独立した帰無参照データで
  選択後の統計量を検証する。

仕様、CLI例、manifest契約、統計的制約は
[URN_CALIBRATION.md](../../docs/URN_CALIBRATION.md)を参照。

## 数値的な再現と回帰検証

21サイト、重み `[20,1,...,1]`、2抽出について、重いサイトの包含確率は
旧近似の **0.7305825203** から **0.7564102564** になった。
新値は独立した順序付き2抽出の全列挙と一致する。
大きい一般ケースの近似誤差がすべて解消したという意味ではない。

背景枝学習では、foregroundのmassを8から80に変更しても重みは不変で、
評価対象枝の総massは8／80のままであることを確認した。
`each/file_each/sn/pool`とdense／sparseを含む。
有効サイト数が異なる欠測マスクでの総濃度、ゼロmass、無効な枝集合、
独立recoding学習、同名ファイル差替え、遺伝暗号変更も検証した。

`poisson_full`は観測枝別サイトmass由来のurn重複平均からPoisson
カウントを発生させる、という既存の生成方式を変更していない。
完全な系統生成モデルではなく、観測集中を吸収し得ることを明記した。
NB過分散推定器も変更していない。

## 実施したチェック

- 全テスト：**1572 passed / 3 skipped**。
  skipは既存のPyTorch 2.6要件に対して環境が2.2.2であるため。
  既存のrequests依存バージョン警告が1件ある。
  [ログ](id12_full_tests.log)（保存時にユーザー環境のパスだけ匿名化）
- lint、型チェック、文書リンク／CLI例チェック、差分の空白チェックは成功。
- `make test`の並列経路はpytest-xdist未導入で起動できなかったため、
  リポジトリで認められている通常の `python -m pytest -q` で全件を実行した。
- 元タスクの未コミット修正を含むソースを一時領域に複製し、ID 12の差分を
  重ねた。差分は競合なく適用でき、3Di期待値・根の対応・新ASRV／recoding
  機能を含む **112 passed**。元ディレクトリは変更していない。
  [統合ログ](id12_current_overlay_tests.log)
- その一時統合コードでも、実IQ-TREEを伴う背景ASRVの2データセット解析を
  完走した。これは較正用標本数ではなく、統合後の実行経路確認である。

## 解析全体のsmoke実験

独立したsingle-step sense-codon CTMCから300コドン、8tipのデータを生成。
帰無参照3、帰無検証3、合成シグナル検証2、計8データセットについて、
次の3構成、合計24解析を完走した。

1. 全枝学習、既存のサイト当たりalpha=1。
2. 背景枝学習、総濃度tau=2。
3. `kgbauto6`自動recoding、背景枝学習、総濃度tau=2。

統計量は宣言した対象集合の最大 `omegaCany2spe`、さらに構成間の最大。
`OCNany2spe >= 1` と `OCSany2spe >= 1` を全反復へ適用し、既存仕様で
ECが未定義になる根／根隣接枝は、既知の樹形から求めたIDを事前に除外した。
それ以外の対象行に未定義値があれば黙って捨てずに停止する。
対象集合が空の反復も負の無限大という定義済み統計量として保持した。

- [反復・構成ごとのスコア](id12_pipeline_smoke_scores.tsv)
- [独立検証側の結果](id12_pipeline_smoke_validation.tsv)
- [較正判定と信頼区間](id12_pipeline_smoke_summary.json)

帰無参照が3件なので最小P値は **0.25**。水準0.05での検出は不可能で、
`fpr_criterion_met=false` を正しく報告した。この実験をFPR／powerの
較正済み証拠として扱ってはいけない。3構成の比較も小規模な動作確認であり、
新しい既定値を推奨する根拠にはしていない。

再現コマンド（出力先は未使用ディレクトリを選ぶ）：

```bash
python tools/generate_urn_validation.py --outdir /tmp/urn-id12-input \
  --calibration 3 --validation 3 --alternatives 2 --sites 300
python tools/evaluate_urn_pipeline.py --manifest /tmp/urn-id12-input/manifest.json \
  --outdir /tmp/urn-id12-run
```

## 未完了の科学的評価と依存関係

以下は未完了であり、この実装だけではID 12全体を「較正済み」としない。

- 十分な独立反復数によるFPRとpower評価。短長配列、疎なS、arity、欠測、
  非一様率、モデル不一致、学習集合、alpha／tau、recoding選択を含む。
  生成器の大規模既定値は今回実行していない。
- 残るWallenius近似領域の代表的な誤差測定と受容基準。
- 背景由来の非一様prior、総濃度自動推定、clade単位cross-fittingなどの
  比較評価。未検証の方式を既定として追加していない。
- ID 2の観測／帰無共通変換、経験的prior／alpha推定、ID 4のdif共同乱数は
  元タスクの責任範囲。対応後の条件付きP値較正にこの検証基盤を接続する。
- ID 3のepistasis学習、ID 9の3Di測定、ID 10のlong-tail、ID 11のサイト除外、
  ID 13の探索familyは個別に整合した設定で検証する。

公開Q.3Di.AF／LLMの追加、branch protection、バージョン変更、公開操作は
この変更の対象に含めていない。

## 主checkoutへの取り込み検証（2026-09-10）

ユーザーのcommit依頼に基づき、ID 12の変更を主checkoutへ取り込んだ。
既存の別作業の変更は保持し、commit対象から除外した。
取り込み後の全テストは1616 passed、3 skipped。スキップはtorchのバージョン要件による。
`make lint typecheck`も成功した。これらは実装の回帰検証であり、
広範な偽陽性率・検出力の校正が完了したことを意味しない。
