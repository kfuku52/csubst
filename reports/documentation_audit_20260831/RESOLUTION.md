# ドキュメント監査への対応 — CSUBST 1.14.10

修正前の記録は [README.md](README.md)。本体と Wiki の `master` を pull し、
監査 D1–D7 と補足4項目を修正した。Wiki は既存14ページを更新し、
`Typical-workflow` を1ページ追加した。数値計算の実装と依存パッケージは変更していない。
Wiki の修正 commit は `748456cd5e0009f221080bee3bf5809d3af1b5e5`。

## 対応内容

| 指摘 | 対応 |
| --- | --- |
| D1: PyPI を前提としたインストール | Wiki の本体・extras 指定を GitHub のソース URL に統一。依存不足時の Python エラーメッセージも更新 |
| D2: sites の旧出力名 | 関連5ページを `csubst_sites/csubst.<branch-selection>/` と `csubst.*` に更新。独自接頭辞、ログ、manifest の位置も説明 |
| D3: cs 表の集計単位 | `cs` は組み合わせを合算したサイト別、`cbs` は組み合わせ別と明記。行キー、0始まり/1始まりの座標、小数丸めも説明 |
| D4: verify の不整合 | VESM の常時 SHA-256 検証を維持し、重複する `--verify` を互換用として非推奨化。ProstT5 / all の未対応チェック要求は準備処理の前に終了2 |
| D5: モデル保存先 | CSUBST キャッシュ、ProstT5 の Hugging Face キャッシュ、明示的な local_dir、共有ロックの位置を区別。オフラインへのコピー手順を追加 |
| D6: プロットの no | `no` は無効化かつ既定値、`yes` は非対応、`all` / サイト指定は有効化と修正 |
| D7: 引用著者 | `Parler` を `Perler` に訂正 |
| 補足1: 配布チャネル | Python 3.14 のソース対応と Bioconda の配布範囲を分離。Wiki に確認日付きの配布状況と一次情報へのリンクを記載 |
| 補足2: benchmark の終了コード | 続行設定にかかわらず、失敗があれば結果保存後に終了2と追記 |
| 補足3: スレッド数比較 | 単一整数を渡すシェルループに変更し、実行ごとに出力先を分離 |
| 補足4: README の重複 | 詳細ワークフローと VESM ガイドを Wiki に集約。README は225行から128行へ整理し、ロゴと方法図を維持 |

照合中に確認した `inspect` の site-index-map ファイル名の help 誤記も修正した。
単枝の intersection でも state-summary ファイルが出ることは実行で確認し、
Wiki の説明に反映している。既存の `analyze` / `site` エイリアス、歴史的な Wiki URL、
移動した VESM セクションのアンカーは維持した。

## 検証

macOS 26.5.2 / arm64、Python 3.12.13、NumPy 2.5.2、SciPy 1.18.1、
pandas 3.0.5、Matplotlib 3.10.9、ete4 4.4.0 で確認した。

| 確認 | 結果 |
| --- | --- |
| 関連テスト | 73 passed |
| 全テスト | 1,399 passed（並列対象1,395件 + process 4件） |
| strict native | 7 passed（全テストにも含まれる） |
| Ruff / repository hygiene / mypy | 通過 |
| README等6文書 + Wiki29ページ | コマンド59件、help17件、内部リンク127件を確認。テンプレート2件は具体的引数の解析対象外 |
| シェル例の構文 | 48コードブロックが `bash -n` を通過 |
| 文書チェックの異常系 | 不明なオプションと存在しない画像リンクを、変更した一時コピーで検出 |
| PGK を使う実 CLI | 正常系9件は終了0。未対応の ProstT5 / all checksum 要求2件は終了2 |

PGK の実行対象は `dataset`、`doctor`、追加4表を有効にした `search`、
`sites` の intersection / lineage / set / 単枝 / 独自接頭辞、`inspect` のプロット無効化。
ファイル一覧と行スキーマを読み、文書の説明と一致することを確認した。
個々のコマンド・終了コード・出力一覧は [resolution_verification.json](resolution_verification.json) に保存した。

VESM の破損チェックは、小さな実ファイルと実際の SHA-256 検証を使用した。
同じサイズのまま内容を書き換えたファイルを、オフラインかつ `--verify` の
未指定 / no / yes の全ケースで拒否し、置き換えをダウンロードしないことを確認した。
ProstT5 / all に対する未対応のチェックは、いずれのモデルも準備し始める前に拒否する。

モデル本体のダウンロード・推論、オンライン構造検索は実施していない。
これらのローダー自体は変更していない。ローカルでは checkout を対象に検証し、
配布 wheel / sdist の検証は既存の push CI が担当する。

## 継続的な文書チェック

`.github/scripts/documentation_check.py` を追加し、既存の `make lint` に組み込んだ。
標準ライブラリと CLI parser のみを使い、`python -S` でも通過する。
CI ジョブ、追加の依存パッケージ、ネットワークアクセスは増やしていない。

```bash
make lint typecheck test test-native
make docs-check WIKI_DIR=path/to/csubst.wiki
```

文書中のシェルは実行しない。静的に解析できるループは代表値を使用し、
環境変数やコマンド置換を評価しない。外部 URL、すべての見出しアンカー、
科学的な主張や実際の出力ファイル名までは自動検証しないため、
変更した例は小さなデータで実行し、manifest も確認する手順を CONTRIBUTING に追加した。

監査前の `verification.json` は1.14.9時点の記録として保持し、修正後の記録と分けた。
特に search の実行オプションが異なるため、両記録の `cbs` 行数を同じ条件の
性能・数値比較として扱わない。
