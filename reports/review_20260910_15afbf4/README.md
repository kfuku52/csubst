CSUBSTの直近変更レビュー（2026-09-10）

対象は9月8日以降の確定済み変更、基準 `5713ffb` → `15afbf4`。本体59ファイル、約6,953行追加・1,790行削除。ソースを固定した別ディレクトリで調査した。作業中のissue 46関連の未コミット変更は対象外。以下は修正前のレビュー記録。修正後の確認結果は末尾に記載。

**確認した問題**

1. **[P1] 背景ASRV学習に必要な枝が部分読み込みから欠落する。** `--exhaustive_until 1 --foreground ... --cb yes` で、`b/s/bs/cs/cbs`をすべて無効にすると、`parser_misc._can_use_selective_state_loading`は新しい学習設定を確認せず部分読み込みを許可する。`_get_required_state_branch_ids`も前景枝とその親だけを対象とする。一方、`asrv.resolve_training_ids`（60–77行）は木全体から背景枝IDを登録し、`substitution.get_sub_sites`（2617–2629行）と`omega._prepare_substitution_permutation_components`は未読み込み枝のゼロ配列を実際の学習量として集計する。処理は正常終了し、provenanceにも欠落を示さず背景枝IDが記録される。

   8末端・100コドンの同一入力で、`--expectation_method urn --asrv sn --asrv_training_branches background --asrv_report yes`を指定した。`--b no`では15ノード中5ノードのみ読み込まれ、S学習量が0。`--b yes`への変更だけでS学習量が64.4467068774となり、同じ枝対(0,7)のECNany2anyが6.3091→7.2153、ECSany2anyが0.0394→0.0470、omegaCany2speが1091.8701→1095.9118に変化した。出力の有無が統計量を変えてしまう。修正は状態読み込み前に学習枝とその親を必要集合へ含めるか、カスタム学習時は完全読み込みを必須にすること。背景枝だけでなく明示ID指定も確認が必要。単体テストのASRV学習用テンソルは最初から全枝を含み、この組み合わせを検出しない。

2. **[P2] サイト監査表が未読み込み末端を欠損として数える。** `site_filter.prepare`（45–47行）は部分読み込み済み`state_cdn`から`num_tips_with_codon_state`を計算する。この列は入力の欠損状況を報告するものだが、全100サイトに8配列の有効コドンがある入力を、`--site_filter_report yes`と同じ部分読み込み設定で解析すると全サイト2と報告する。`--b yes`だけで全サイト8へ変わる。入力配列から同じコドン解釈規則で数えるか、監査に必要な末端状態をすべて読み込む必要がある。今回のfixtureでは除外マスクに差は確認されず、確認できた影響は末端数の誤報告である。

**検証**

Dockerイメージ `local/genegalleon:csubst-scan-inference-dev`（GeneGalleon実行環境、Python 3.12、IQ-TREE 3.1.4）で固定ソースを実行。Linuxネイティブ拡張は、6個すべての対応するCythonソースが固定ソースと同一であることを確認した既存ビルドを使用した。

- `python -m pytest -q -n 4 --dist worksteal -m "not process"`: 2,086 passed、3 skipped、4 warnings。
- `python -m pytest -q -m process`: 4 passed。
- `CSUBST_STRICT_EXTENSIONS=1 python -m pytest -q -m native`: 8 passed（上記テスト集合の部分集合であり、独立件数に加算しない）。
- 今回追加の実CLI比較: 背景ASRV学習2条件、サイト監査2条件、すべて正常終了して上記不一致を再現。
- テストログと比較JSON、CLIログ、比較対象TSV、入力をこのディレクトリに保存。

重点確認範囲は、ASRV学習・平滑化・独立null校正、endpoint/bridgeとscanの校正経路、3Diモデル・キャッシュ・期待値、サイト除外と状態解放、高次探索・複数traitの候補処理、ID・表・木・ファイル処理。既存テストの成功は新しいオプションの組み合わせの正しさを保証しない。

transformersを必要とする1テストとgemmiを必要とする2テストは依存未導入でskip。今回、実際の3Di予測モデルによる新規推論、大規模な実データ解析、SIF/Apptainerでの検証は行っていない。これらの互換性・正確性を保証する結論ではない。

**再現方法**

本ディレクトリの`reproduce.py`を、対象CSUBSTを`PYTHONPATH`に設定したGeneGalleonコンテナ内で実行する。入力は同梱。出力先は未作成のディレクトリを指定する。

```sh
PYTHONPATH=/work python /report/reproduce.py --output /work/review-reproduced
```

今回の固定ソースはホストの`/tmp/csubst-review-15afbf4`。コンテナへこれを`/work`、本ディレクトリを`/report`としてマウントすれば同じ比較を再実行できる。ASRVの巨大な診断JSONは固定実行ディレクトリにあり、本報告には必要な値を抽出した`review-comparison.json`を保存した。

**修正後の確認（1.15.2）**

状態読み込みの判定に、経験的ASRV学習とサイト選択・監査の全枝依存を追加した。学習集合がall/background/明示IDのいずれでも経験的モードは全枝を読み込む。固定サイト重みのno/fileは部分読み込みを維持する。サイト除外が暗黙に出力する監査表も対応した。

追加した17ケースを含む通常テスト2,107件が成功、依存不足の3件がskip。ネイティブ必須チェック8件も成功。実CLIによる4条件の再実行では、枝出力の有無によらず、S学習量64.4467068774、ECNany2any 7.2153、ECSany2any 0.0470、omegaCany2spe 1095.9118が一致し、監査表は全サイト8配列となった。値は`fixed-comparison.json`に保存した。

Ruff・リポジトリ衛生・文書チェック・全指定モジュールのmypyが成功。sdistを作成し、生成済みCからwheelを再構築、配布物内容検証とtwineチェックも成功した。
