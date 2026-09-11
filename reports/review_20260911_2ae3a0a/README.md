# 2026-09-11 コードレビュー

対象は `286ac2b..2ae3a0a`（9月8日から11日の変更）とレビュー開始時点の作業ツリー。`csubst/` と `tests/` だけで125ファイル、13,324行追加・2,249行削除がある。既存の未コミット変更を保持し、実装の修正・コミット・pushは行っていない。このディレクトリだけが今回のレビュー成果物である。

**追記: 以下の2件は後続の修正で解決。** [修正後の検証・性能監査](../resource_audit_20260911/README.md) を参照。 `+I` 境界値の回帰テストと実IQ-TREE入力で確認し、ログの個人パスも匿名化した。以下は修正前のレビュー記録。

**再現確認した指摘は2件。** 通常テストの成功だけでは検出できなかった有効入力の拒否と、必須チェックの失敗がある。

## 1. [P1] コミット済み検証ログが必須の hygiene 検査を失敗させる

- 箇所: `reports/event_unification_20260911/final-review-checks.log:2`（43、56、65行にも同様の内容）。
- ログに個人ホーム以下の Python パスが残っており、既存の `repository_hygiene_check.py` が拒否する。
- 再現: `make lint`。Ruff は成功し、その次の hygiene 検査が非ゼロ終了する。
- このファイルは `2ae3a0a` に含まれる追跡済みファイルであり、レビュー開始時点の未コミット作業が原因ではない。必須チェックを通せないため、修正が必要。
- 対応: 検証ログの環境依存パスを匿名化する。検査ルールを緩める必要はない。
- 証拠: [lint.log](lint.log)。

## 2. [P2] +I の割合が0に推定された有効な IQ-TREE 出力を joint 推定が拒否する

- 箇所: `csubst/endpoint_io.py:69-73`、`read_rate_mixture()`。
- IQ-TREE 2.3.6 で通常どおり `-m GY+FQ+I` を推定すると、不変サイト割合が0になる入力では `Model of rate heterogeneity: Invar` と `Proportion of invariable sites: 0.000000` を出力し、カテゴリ表を省略する。
- 読み込み処理は `Uniform` かカテゴリ表だけを受け付けるため、既定の joint search が `Joint endpoints require an IQ-TREE uniform model or rate-category table.` で終了する。
- 4種×100コドンの生成データで再現した。同じ fitted files を用いた joint は終了コード2、marginal は0。割合が約0.5となる対照入力ではカテゴリ表があり、問題は発生しない。`+I` 全般が壊れるという指摘ではない。
- 影響: この読み込みを共有する joint search/sites/scan と analytical scan。従来の marginal 入力を新しい既定値で処理すると停止しうる。
- 対応: 不変成分が正確に0である縮退モデルの表現を認識して、単一の単位速度カテゴリに変換する。正の割合を一律に無視する回避策は不適切。IQ-TREE 実出力を使った境界値テストを追加する。
- 再現: [reproduce_invariant.py](reproduce_invariant.py)。入力生成、IQ-TREE 実行、joint/marginal 比較を一時ディレクトリで実施する。
- 証拠: [invariant-evidence.txt](invariant-evidence.txt)。

## 実施した検証

環境: macOS、Python 3.10、NumPy 1.26.4、SciPy 1.15.2、pandas 2.2.3、IQ-TREE 2.3.6。

| 検証 | 結果 |
| --- | --- |
| 全非processテスト、4ワーカー | 2,194成功、5スキップ |
| processテスト | 4成功 |
| strict nativeテスト | 16成功（上記テストとの重複あり） |
| 拡張を無効化した全非processテスト | 2,153成功、46スキップ |
| 拡張を無効化したprocessテスト | 4成功 |
| Ruff | 成功 |
| 必須のmypy対象 | 23ファイル成功 |
| ドキュメント検査 | 21文書・23コマンド等を検証して成功 |
| repository hygiene | 指摘1により失敗 |

通常テストの5スキップは torch>=2.6 が必要な3件と gemmi が必要な2件。fallback の追加スキップはコンパイル済み拡張を必要とするテスト。Requests の依存バージョン警告等はログに残している。

[tests.log](tests.log)、[process.log](process.log)、[native.log](native.log)、[fallback.log](fallback.log)、[fallback-process.log](fallback-process.log)、[types.log](types.log)、[docs.log](docs.log) に結果を保存した。保存時に個人ホームのパスを匿名化した。

### 実コマンドと数値の追加検証

PGK の同梱 fitted files を固定し、11通りの有効なCLI構成が完了した。

- search: 既定値、全テーブル出力、urn、urn+P値、サイト除外、サイト監査、Dayhoff6、min_sub_pp=0.5。
- scan: joint と analytical endpoint mixture。
- sites: 指定枝のテーブル出力。

別途、既定のdif統計とhypergeom P値の非対応構成が説明付きで拒否されることも確認し、対応するany2speで再試験して成功した。これは不具合として数えていない。[cli-results.json](cli-results.json) に終了コードを記録した。

PGK の既定の投影計算と全イベント計算を、1,682枝組合せ・25数値列で照合した。NA位置は一致し、保存TSVの最大絶対差は約0.0001（小数4桁出力の1単位）。許容誤差 `atol=rtol=1e-4` で一致した。内部の全精度一致をこのTSV比較だけから主張してはいない。

さらに本レビュー用の [oracle.py](oracle.py) で、最適化された endpoint 推定と、別実装の対数空間 inside/outside 推定を比較した。3/4/8状態、枝長スケール1e-7/0.1/2/30、単一速度/3速度混合の24モデルを検証。欠損・部分尤度・全欠損サイトを含め、全節点周辺分布と全枝jointを比較し、最大絶対差は `1.4432899320127035e-14`。結果は [oracle-results.json](oracle-results.json)。

## レビューの重点と限界

直近の既定値変更に伴う model/ASR 読み込み、joint pruning、圧縮投影とfullイベント、キャッシュの無効化、観測可能性とNA表示、サイト座標・除外、scanの露出と検定、ASRV学習、Poissonカテゴリ・long-tail、3Diモデル・再符号化の接続を中心に差分と呼出元を追った。上記2件以外に、今回の検証で裏付けられる具体的な不具合は確認しなかった。

ただし、これは全条件での統計的妥当性の保証ではない。

- 新しい既定joint推定について、大規模な偽陽性率・検出力の実験は今回実施していない。既存の `issue46_pipeline_20260910` は以前のスナップショットとmarginal設定を評価しており、その結果を現HEADのjoint設定の検証とみなせない。
- 3Diの実モデル推論、GPU、外部ダウンロード、PyMOLの実描画は再実行していない。関連コードと実行可能なテストを確認した範囲に限る。
- wheel/sdistの再ビルド、他のPython版・OS、実GitHub Actions、外部wikiの検査は実施していない。リモートの更新を統合せず、ローカルHEADを対象とした。
- 既存の未コミット校正スクリプト等は全体テスト・lintの対象に含むが、保存済みの数百反復の実験は再実行していない。
