# Worktree・未統合コミットの確認

2026-09-11、修正前HEAD `2ae3a0a` を基準に確認。`git fetch origin` 後も origin/master は `72c3c45` で、master は1コミット先行し、取り込むべきリモート更新はなかった。

## 現存するworktree

| 作業場所 | HEAD | masterへの統合 | 作業ツリー |
| --- | --- | --- | --- |
| 本体 | 2ae3a0a | 本体 | 今回の修正以外に、以前からIssue46の校正スクリプト・文書・結果が未コミット |
| 42d8 | 75b68b7 | ancestor判定で確認 | clean |
| cd1a | c4e6d82 | ancestor判定で確認 | epistasis作業の残存差分 |
| d82b | 2361ccd | ancestor判定で確認 | scan作業の残存差分 |

cd1a/d82b について、未追跡ファイルも含めて本体と照合した。本体に存在しないファイルは0件。全ファイルの分類と過去コミットとのblob一致は [worktree-audit.json](worktree-audit.json) に記録した。

- cd1a: epistasis/pair_epistasis本体、テスト、主要検証データ等は本体と同一。変更パッチの `csubst/`・`tools/`・`tests/` への追加行はすべて本体に残っている。異なる検証用TSVはCRLF/LF差、対応する生成スクリプトは本体でLF出力に修正済み。その他は別の修正との統合・文書整備による差。
- d82b: scanの新規モジュール等は master内の `1a83b17` と同じblob。その後 `2ae3a0a` でjoint既定値、共通fitted model、率カテゴリへの対応が更新された。旧差分に残るscan専用モデル準備や旧既定値を再適用すると、この統合を巻き戻してしまう。

確認した範囲で、これら2worktreeの**本体実装を取り込み忘れた形跡はない**。worktreeの差分を破棄・resetしたり、古い版で上書きしたりしていない。

## 最近の保存snapshot

通常のブランチ以外に、9月10–11日の保存snapshotを7件確認した。

- 最新 `769919e` のツリーは、修正前master `2ae3a0a` と完全一致。
- ASRV、long-tail、3Di、scanの実装を含むsnapshotについて、本体に存在しない製品コード・テスト・toolsファイルは0件。関連機能はその後の通常コミットに含まれる。
- **`6d3a73e` の `reports/pepc_longtail_20260910/`（25ファイル）は本体にない。** PEPCのempirical/independent-null比較報告・図・表・再現スクリプトであり、製品コードを変更するsnapshotではない。保存snapshotから回収可能な状態を維持している。本体の実装漏れと混同せず、未統合の実験成果物として報告する。

例えば `git show 6d3a73e:reports/pepc_longtail_20260910/README.md` で内容を確認できる。この報告は当時の設定とキャッシュ付き実験の記録であり、現行既定値の性能測定として流用していない。

## 古いブランチ

通常のローカル・リモートブランチには、2月以前の報告書・図表等の未統合コミットが残る。

- `4d496da` (`codex/bootstrapping`): `reports/issue_bootstrapping/` の報告書・ハッシュ。
- `a18ee29` (`codex/epistasis`): negative-control の図表6ファイル。
- `575c33b` (`issue53-parity-artifacts`): issue53の図表。
- `feaf3e2` (`issue7-phase6-artifacts`): dense/sparse比較の図表。
- 同ブランチ履歴の `83b438d` / `3a94efd` は、製品機能としては masterの `57aeb11` / `9a7805a` 等に対応する。コミットIDが未統合であることだけから、vendored Pyvolveやtrue ASR出力の実装漏れとは判断できない。
- Dependabotの `4203cd9`（setup-python v7）は通常履歴上未統合だが、masterは `fe055f0` 以降すでにv7の固定SHAを使用している。
- 2025年の `origin/ci/smoke` は現在のCIと異なる古い構成。現行CIへそのまま統合する対象にはしていない。

今回の「2件の修正」には、これら別目的の実験成果物、旧CI、以前から本体に残るIssue46作業を混在させていない。未コミット状態を含めて保持した。
