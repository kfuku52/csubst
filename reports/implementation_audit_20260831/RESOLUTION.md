# 監査指摘への対応 — CSUBST 1.14.9

対象は 1.14.8 / `8e80176135cb57ac80dbc40241f1f8bc15d59433` からの変更。
`master` を再度 pull した上で、監査 A–H と開発手順・CI・処理境界の改善を実施した。
計算式や統計基準の変更、依存パッケージの新たな上限制約、branch protection の変更は行っていない。

## 対応内容

| 指摘 | 対応 | 主な検証 |
| --- | --- | --- |
| A: ログによる入力上書き | ログを開く前に入力パス・同一 inode を検査。引数エラー時、既定入力、推定 IQ-TREE パス、省略オプションにも対応 | 入力内容を保持したまま終了 2。通常・symlink・hardlink・引数エラーを CLI subprocess で確認 |
| B: benchmark の失敗を成功扱い | 続行可否と最終終了コードを分離。結果・ログを書いてから、失敗があれば終了 2 | 全成功、一部失敗、全失敗、早期中断、失敗後の継続 |
| C: readonly 配列で native 脱落 | 入力専用の Cython memoryview を `const` に変更 | readonly ndarray / memmap / pandas、pseudocount の有無、strict native |
| D: FASTA 空白の不一致 | IQ-TREE と ETE の読込を共通 streaming reader に統一。サイト数推定は最初の record で停止し、tree mapping の重複 ID は拒否 | space / tab / CRLF / 複数行 / gzip、leaf state tensor、実 PGK 入力 |
| E: Accelerate の変数名 | NumPy import 前に `VECLIB_MAXIMUM_THREADS` を設定 | 別プロセスで設定順序と独立した環境変数の期待値を確認 |
| F: 自動タグから wheel が起動しない | wheel workflow を reusable にし、major/minor release job から対象 tag を明示して呼び出す | actionlint、対象 ref・権限・patch skip 方針の確認 |
| G: wheel の実計算未検証 | checkout 外の非 editable install を対象に全件テスト・PGK/PEPC parity。import 元と全 6 拡張を検査 | 通常 wheel の実計算、ソース混入・native 欠落を拒否するチェック |
| H: 開発ツール不足 | 実際に使う Ruff / mypy / build / Twine を `dev` extra に宣言し、Makefile を `$(PYTHON) -m ...` に統一 | 新規環境で `.[dev]` のみから lint・型検査・配布ビルド・pip check |

終了コードと入力保全の仕様は [CLI_SAFETY.md](../../docs/CLI_SAFETY.md)、
配布物を含む検証手順は [TESTING.md](../../TESTING.md) に記載した。

## 開発速度と変更の追いやすさ

- Python 3.12 の full / parity / packaging は、同じ run で生成した wheel を共有する。
  pure Python は sdist、sanitizer は専用 build のまま分離し、検証項目は残した。
- push は `master` に限定し、PR との重複を避ける。古い CI は新しい run で cancel する。
- macOS Intel は build・依存・runtime・CI 関連の変更で動かす。差分不明、週次、手動実行では必ず動く。
  release wheel は Linux / macOS Intel / macOS arm64 の全 Python 対象で数値 parity を行う。
- 中間配布 artifact は 2 日、parity metrics と release wheel artifact は 14 日保持する。
- `get_global_parameters` の 790 行を順序付きの検証段階へ分割し、入口を 32 行にした。
  CLI parser の入口も 600 行から 63 行へ整理。各コマンドの通常・advanced help の互換性を確認した。
- 配列のみを扱う統計処理を `omega_statistics`、tree/site 描画を `site_tree_plot`、
  遅延 Matplotlib 初期化を `plotting` に分離。既存の helper 名は元の module から再 export する。
  分割時は処理本体を AST で比較し、数値・描画テストを併用した。
- 型検査を 5 から 13 module に拡大し、`main_scan` の戻り値型を修正した。
  全 context key の型付けや全面的な書き直しは行わず、設定・配列・I/O・描画の境界から進めた。
- pandas 3 が通知する単一列 groupby の将来警告を、scalar grouper への変更で解消した。
  警告の一括抑制は行っていない。

配布 wheel のテストを実行したことで、spawn 先が pytest の仮想 test module を import
できない問題も見つかった。子プロセス用関数を `tests/support/process_workers.py` に移し、
ソース package を子プロセスの path に追加せず、従来の並行処理の検証を維持した。

wheel の数値検証は [cibuildwheel のテスト用一時環境](https://cibuildwheel.pypa.io/en/stable/options/#test-command)
で実行する。Linux container には検証 script が直接使用する GNU time も明示的に導入する。

## 検証環境と再現手順

ローカルは macOS 26.5.2 / arm64。Python 3.14.0 の既存環境と、Python 3.12.13 の
新規 dev / wheel / pure Python 環境を分けた。後者の主要依存は NumPy 2.5.2 /
SciPy 1.18.1 / pandas 3.0.5 / Matplotlib 3.10.9 / ete4 4.4.0。

ソースでの検証:

```bash
python -m pip install -e '.[dev]'
make lint typecheck
make test
make test-native
```

配布物のビルドと、別環境・checkout 外での全件テストは
[TESTING.md](../../TESTING.md#testing-the-installed-wheel) のコマンドを使用する。
pure Python CI は拡張を無効にして、`.github/scripts/run_sdist_tests.py` で
展開した sdist 内の全テストを実行する。

実データの確認は対象 wheel を通常 install した環境で、checkout の外から実行する。
`CSUBST_REPO` はこの repository の絶対パスを設定する:

```bash
CSUBST_STRICT_EXTENSIONS=1 python "$CSUBST_REPO/.github/scripts/sites_parity_check.py" \
  --installed --numerical-only --output parity_metrics.tsv --workdir parity_runs
CSUBST_STRICT_EXTENSIONS=1 python "$CSUBST_REPO/.github/scripts/omega_pvalue_calibration_check.py" \
  --installed --workdir calibration_runs
```

Mac の時間・RSS で Linux の性能 baseline は更新しない。`--numerical-only` は
数値チェックを残したまま、Linux 用の性能しきい値だけを適用しない設定である。
CI の Linux parity lane では従来どおり性能しきい値を適用する。

この patch では release tag を新規作成しない。major/minor release から reusable
workflow への実際の呼び出しは次の対象 release 時の検証となる。
外部サービス・全 optional model の download を網羅する検証は行っていない。

## 最終コードの検証結果

| 対象 | 結果 | pytest 表示時間 |
| --- | --- | ---: |
| Python 3.14.0、ソース、parallel-safe 全件 | 1,385 passed | 23.41 秒 |
| 同環境、process を直列実行 | 4 passed | 4.52 秒 |
| Python 3.12.13、通常 wheel、parallel-safe 全件、coverage あり | 1,385 passed | 15.88 秒 |
| 同環境、process を直列実行、coverage 継続 | 4 passed | 9.44 秒 |
| Python 3.12.13、拡張無効、展開した sdist の parallel-safe 全件 | 1,353 passed / 32 skipped | 15.31 秒 |
| 同環境、process を直列実行 | 4 passed | 4.07 秒 |
| strict native、Python 3.14 / 通常 wheel 3.12 | 各 7 passed | 1.46 / 2.42 秒 |

ソース・wheel の全件数は各 1,389 件。strict native はその中の 7 件を別条件で再実行したもの。
pytest 表示時間は環境・coverage・cache 条件が異なるため、行同士の速度比較には使わない。
wheel の statement coverage は **76.43%**、branch coverage は **63.74%**、合成値は
**72.65%**。既存の 70% gate を変更せずに通過した。

Ruff、repository hygiene、13 module の mypy、actionlint、`git diff --check` は成功。
科学計算パッケージを型検索対象から外した状態でも型検査を実行し、CI の lint 専用環境に
実行時依存を追加する必要がないことを確認した。新規 dev / wheel / pure Python 環境の
`pip check` も成功した。

sdist と、その generated C から作った wheel は内容検査・Twine を通過した。
wheel の Python 66 ファイルは checkout・通常 install 先と byte 単位で一致し、sdist の
test Python 104 ファイルも checkout と一致する。別環境の wheel から native 拡張を一時的に
外す故障注入と、ソースを先に import させる確認は、どちらも artifact guard が拒否した。
拡張を復元後に全 6 module の検査を再実行して成功している。

Accelerate を使う新規プロセスで、スレッド環境変数を未設定にして行列積を実行した。
`--threads 1` 相当の設定後、観測したプロセスの最大スレッド数は修正前 3、修正後 1。
これはスレッド上限が働くことの確認であり、異なるスレッド数での行列積を速度比較には使わない。

## 実データ・描画と性能

PGK / PEPC は warmup 各 1 回を除き、修正前後で各 3 回ずつ実行した。両方とも同じ
Python・科学計算依存の通常 wheel とし、同じ thread 数・事前に作った Matplotlib cache を使用した。
スレッド変数の修正の有無に左右されない比較のため、両方に `VECLIB_MAXIMUM_THREADS=1` を外から設定した。
反復ごとに実行順序を交互に入れ替えた。共有ホストの他作業は停止していないため、CPU 競合や
メモリ圧迫を完全には管理できていない。以下は中央値、括弧内は最小–最大。RSS は KiB。

| dataset / version | analyze 秒 | site 秒 | analyze peak RSS | site peak RSS |
| --- | ---: | ---: | ---: | ---: |
| PGK / 修正前 1.14.8 | 2.51 (1.71–2.67) | 4.41 (4.26–4.47) | 196,784 (192,144–197,616) | 273,712 (272,496–275,120) |
| PGK / 修正後 1.14.9 | 2.00 (1.92–2.60) | 4.01 (3.68–4.92) | 193,824 (192,992–196,688) | 286,080 (257,968–297,856) |
| PEPC / 修正前 1.14.8 | 4.03 (3.52–4.42) | 6.18 (5.31–6.47) | 419,728 (383,488–433,632) | 485,296 (406,944–498,432) |
| PEPC / 修正後 1.14.9 | 4.19 (2.87–4.92) | 6.34 (6.14–8.76) | 356,224 (342,624–455,920) | 506,224 (422,016–522,112) |

PGK の時間中央値は短くなった一方、PEPC の時間と両 dataset の site RSS 中央値は増えた。
反復間のばらつきもあり、この結果を一律の高速化や性能劣化なしの証拠とは扱わない。
主な開発速度対策は、CI の重複 build・重複起動の削減と開発手順の再現性である。
Linux CI の性能 baseline / しきい値は変更していない。生の測定値は
[performance_comparison.tsv](performance_comparison.tsv) に warmup を含めて記録した。

すべての実行が既存の数値基準を通過した。PGK の branch IDs は 23,51、ωCany2spe は
1.975050、convergent / divergent / blank は 5 / 7 / 390。PEPC は 9,108、
0.049466、0 / 2 / 954。代表反復で branch・combination・site の **8 TSV が byte 単位で一致**した。
**4 PDF の全ページで抽出テキストと 100 dpi の描画画素が一致**し、描画結果も目視した。
これは既存結果からの回帰確認であり、図のデザインは変更していない。

追加の PGK `--asrv sn` は、修正前の strict native では終了 2 / readonly buffer error を再現した。
修正後は strict native のまま成功し、修正前の fallback 実行と結果表が byte 単位で一致した。
空白・タブ・CRLF を挿入して gzip 化した入力でも最後まで計算でき、同じ結果表を得た。
途中のサイト数だけでなく、ETE の leaf sequence と state tensor を通す回帰テストを追加した。
測定値は [sn_comparison.tsv](sn_comparison.tsv)。

p 値校正は `niter=100`、`min_sub_pp=0,0.05`、`any2any / any2spe` の 4 設定で
既存の回帰基準を通過した。統計手法全体の妥当性を新たに証明したという意味ではない。

依存バージョン、配布物 hash、テスト結果、数値・画像比較、測定範囲の機械可読記録は
[resolution_verification.json](resolution_verification.json)。修正前の記録は
[verification.json](verification.json) と [README.md](README.md) に残している。
