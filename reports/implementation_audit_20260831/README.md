# 実装・検証・開発効率の監査 — 2026-08-31

これは **修正前の 1.14.8 を対象にした監査記録**。指摘への対応内容と修正後の検証は [RESOLUTION.md](RESOLUTION.md) を参照。

`master` を `git pull --ff-only origin master` で更新し、CSUBST **1.14.8 / `8e80176135cb57ac80dbc40241f1f8bc15d59433`** を調べた。更新前は `85d6e6b`、作業ツリーは clean。製品コード・依存定義・CI は変更せず、調査結果だけを追加した。commit / push / release は行っていない。

既存テストと PGK・PEPC の数値回帰は通る。一方、**入力ファイルの破壊、失敗時の成功終了、pandas 3 での高速経路の脱落**を再現した。開発効率では、CI の導入・ビルド時間、配布物の実計算テストの不足、開発環境の手順不足を先に直す価値が高い。全面的な書き直しを勧める根拠はない。

## 1. 優先順位

P1 はデータ保全のため優先して修正すべき問題、P2 は該当条件で機能・性能・検証・開発手順に支障がある問題とする。「再現」はこの監査で実行した結果、「構造確認」はソースと外部仕様を照合した結果であり、両者を区別する。

| ID | 優先度 | 指摘 | 確認方法 | 修正の方向 |
| --- | --- | --- | --- | --- |
| A | P1 | `--log_file` に入力ファイルを指定すると検証前に上書きする | ダミー FASTA で CLI 再現 | ログを開く前に入力との同一性を検査 |
| B | P2 | benchmark が全件失敗しても既定設定では終了コード 0 | 存在しない入力で CLI 再現 | 続行するかと、最終的に失敗を返すかを分離 |
| C | P2 | pandas 3 の読み取り専用配列で Cython 経路が失敗する | 既存テスト・PGK CLI で再現 | 読み取り専用入力に `const` memoryview を使用 |
| D | P2 | FASTA の空白を二つの読み込み処理が違って扱う | 最小入力・PGK コピーで再現 | FASTA の解釈を共通化 |
| E | P2 | Accelerate のスレッド制限用環境変数名が違う | 環境変数と Apple 仕様を照合 | `VECLIB_MAXIMUM_THREADS` を設定 |
| F | P2 | 自動作成タグでは wheel workflow が起動しない構成 | Actions 定義と GitHub 仕様を照合 | 明示的な workflow 呼び出しにする |
| G | P2 | wheel の import は確認するが、その実計算をテストしていない | 通常の wheel インストールで import 元を追跡 | インストール済み wheel を対象に数値回帰 |
| H | P2 | 開発手順どおりに導入しても lint 等が動かない | クリーン環境で `make lint` を再現 | 実際に使用する開発ツールを dev extra に宣言 |

## 2. 再現した実装上の問題

### A. ログと入力の衝突で、入力ファイルが失われる

対象: [csubst/cli.py](../../csubst/cli.py) L1304–1313。

`main()` は引数をパースした後、入力ファイルを検証する前にログを `open(..., 'w')` する。`--log_file` と `--alignment_file` が同じファイルでも拒否しない。ダミー FASTA と木を用意して `doctor` を実行すると、終了コードは 2 だが FASTA の内容は `CSUBST start:` から始まるログに置き換わった。実データは使用していない。

修正案: CLI の事前検証で入力とログのパスを比較してから開く。正規化したパスの一致に加え、既存ファイルの `samefile` で symlink / hardlink も扱う。引数エラー時の追記経路 L1304–1306 にも同じ保護が必要。入力が不正なときにログを残す場合も、入力への書き込みは許可しない。

受け入れ条件: 通常・引数エラー・symlink・hardlink の衝突を拒否し、入力のハッシュが変わらない。正常な別パスへのログ出力は維持する。

### B. benchmark の keep-going が、失敗の通知まで抑制する

対象: [csubst/main_benchmark.py](../../csubst/main_benchmark.py) L624–639、[csubst/cli.py](../../csubst/cli.py) L1009–1010。

既定の `--benchmark_keep_going yes` ではループを続行するだけでなく、最後の例外発生まで抑制する。存在しない alignment / tree を指定した実行で **`pass=0, fail=1`、終了コード 0** になった。シェルや CI が終了コードで成否を判断すると、失敗した実験を成功扱いする。

修正案: `keep_going` は残りの設定を実行するかだけを制御し、集計・成果物の書き出し後は失敗件数に応じて非 0 を返す。全件成功なら 0。失敗を許容する機能が必要なら独立した明示的オプションにし、ヘルプ・ドキュメント・呼び出し元も更新する。

受け入れ条件: 全件成功、一部失敗、全件失敗、早期中断を別々に確認。失敗時にも集計ファイルと各試行のログが残る。

### C. pandas 3 で高速経路が脱落し、strict モードでは解析が停止する

対象: [csubst/substitution.py](../../csubst/substitution.py) L2491–2515 / L2589–2595、[csubst/substitution_cy.pyx](../../csubst/substitution_cy.pyx) L339–350。

Python 3.12.13 / pandas 3.0.5 の新規環境では、`df['S_sub'].values` などが読み取り専用になる。Python 側の `asarray` と `astype(copy=False)` はその性質を保持するが、Cython 側の `double[:] weight_mv` は書き込み可能な buffer を要求する。実際には入力を読むだけである。

通常の全テストは通るが、`normalize_branch_site_weights_double` が `buffer source array is read-only` で失敗し、Python 実装への fallback 警告が出た。同じ既存テストを `CSUBST_STRICT_EXTENSIONS=1` で実行すると失敗する。PGK の実際の `search --expectation_method urn --asrv sn` も strict モードで終了コード 2 になった。

修正案: 入力用 memoryview を `const` にする。隣接する mask / branch ID も読み取り専用として扱えるか確認する。不要な配列コピーや pandas の上限制約で回避するより、入力契約を正す。native 専用の回帰テストでは意図しない fallback をエラーにし、意図的に fallback を検証するテストは別に保つ。

受け入れ条件: pandas 2 / 3、writeable / readonly ndarray、必要に応じて readonly memmap で native と Python の値が一致する。PGK `--asrv sn` が strict で完走する。通常モードでの数値誤りは今回確認しておらず、失われる高速化の割合も未測定。

### D. FASTA の塩基数の数え方が共通 reader と一致しない

対象: [csubst/parser_iqtree.py](../../csubst/parser_iqtree.py) L338–364、[csubst/sequence_io.py](../../csubst/sequence_io.py) L48–79。

共通 reader は配列中の空白を除去する。一方、IQ-TREE state 読み込み前のサイト数推定は `len(line.strip())` で内部の空白も数える。`>A\nATG GCT\n` は共通 reader では 6 塩基だが、後者は「3 の倍数ではない」と拒否した。PGK の配列行に空白を挿入したコピーでも、既計算の IQ-TREE 出力を明示指定した `search` が同じ理由で終了した。

修正案: 最初のレコードの長さを取得する処理も共通の FASTA 解釈を使う。大きな alignment 全体を長さ推定のためだけに読み込まず、共通の iterator / tokenizer を用意する。

受け入れ条件: 空白、tab、CRLF、複数行、gzip で塩基数・サイト数が一致する。形式だけを変えた PGK 入力で解析結果が変わらない。

### E. macOS / Accelerate 用のスレッド上限が設定されない

対象: [csubst/runtime.py](../../csubst/runtime.py) L18–34、[tests/unit/test_runtime_threads.py](../../tests/unit/test_runtime_threads.py) L8–14。

設定される名前は `VECLIB_NUM_THREADS` だが、Apple が示す名前は `VECLIB_MAXIMUM_THREADS`。`configure_native_threads(1)` 実行後も後者は未設定だった。今回の新規 NumPy 環境の BLAS backend は Accelerate なので、対応対象の環境である。[Apple の説明](https://developer.apple.com/documentation/accelerate/sparse-solvers-library?changes=_1__3&language=objc)

修正案: NumPy / SciPy の import 前に正しい変数を設定する。現在のテストは実装の変数名リストをそのまま期待値に使うため、スペルミスを検出できない。バックエンドの仕様に基づく独立した期待値を持たせる。

受け入れ条件: 別プロセスで設定順序と変数名を検証し、Accelerate 環境で実際のスレッド制御を確認する。今回測定したのは設定の欠落までであり、実際の過剰スレッド数や性能低下率は未測定。

## 3. 配布・開発手順の問題

### F. 自動リリースから wheel ビルドへのイベント連鎖が成立しない

対象: [.github/workflows/tag_from_version.yml](../../.github/workflows/tag_from_version.yml) L23–27 / L70–77、[.github/workflows/wheels.yml](../../.github/workflows/wheels.yml) L3–7。

自動リリースは checkout の既定 `GITHUB_TOKEN` でタグを push する。一方、wheel workflow の起動条件は tag push または手動 dispatch。GitHub は `GITHUB_TOKEN` による push から通常の push workflow を新たに起動しない。この構成では、次の対象 major / minor リリースで自動タグができても wheel ビルドが連動しない。手動起動は可能であり、過去の配布事故を観測したという指摘ではない。[GitHub のイベント起動仕様](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)

修正案: wheel ビルドを reusable workflow として呼ぶ、または対象タグを明示した `workflow_dispatch` を実行する。ビルド対象は CI を通過した commit に固定する。広い権限の PAT を追加する必要はない。

受け入れ条件: パッチ版では現行の skip 方針を保ち、対象版では一度だけ正しい commit の wheel が生成される。監査ではタグ作成・リリース・workflow dispatch を実行していない。

### G. 配布 wheel の実計算を保証するテストが不足している

対象: [.github/workflows/pytest.yml](../../.github/workflows/pytest.yml) L225–242、[tests/conftest.py](../../tests/conftest.py) L9–28、[.github/workflows/wheels.yml](../../.github/workflows/wheels.yml) L44–48。

wheel の smoke test は version・dataset・6 拡張の import を確認している。その後の pytest は展開した sdist 内で実行され、`conftest.py` が `csubst` をソース側へ強制的に差し替える。**sdist のテストとしては成立するが、インストール済み wheel の計算テストにはならない。**

同じ commit の wheel をビルドし、editable ではない新規環境にインストールして確認した。`conftest.py` 読み込み前は `site-packages/csubst` から読み込まれ、6 拡張すべてが見える。拡張のないソースコピーの `conftest.py` を読み込むとソース側に変わり、6 拡張すべてが見えなくなった。

修正案: ソース向けテストと配布物向けテストを明示的に分ける。後者では `sys.path` / `sys.modules` の差し替えを無効にし、import 元が対象 venv 内であることと native 拡張の存在を検査する。リポジトリ外の作業ディレクトリから、wheel に対する PGK / PEPC の数値回帰を追加する。

受け入れ条件: wheel 側を壊すと配布物テストが失敗する。単なる import 成功やソース側の fallback 成功では通らない。

### H. CONTRIBUTING の導入手順だけでは開発コマンドが使えない

対象: [CONTRIBUTING.md](../../CONTRIBUTING.md) L3–17、[pyproject.toml](../../pyproject.toml) L49–53、[Makefile](../../Makefile) L10–21。

案内は `pip install -e '.[test]'` だが、test extra にあるのは pytest / pytest-cov / pytest-xdist のみ。案内直後の `make lint`、`make typecheck`、`make package` が使う ruff / mypy / build / twine は入らない。クリーン環境で 4 パッケージが未導入であることを確認し、`make lint` は `ruff: No such file or directory` で失敗した。

修正案: Makefile から直接使われているこれらを dev extra に宣言し、CONTRIBUTING の手順を更新する。runtime 依存には追加しない。CI とローカルで同じ開発ツール集合を使い、クリーン環境からの bootstrap を確認する。

## 4. 開発速度を下げる要因と改善案

### CI はテスト実行より導入・ビルドが長い

取得した直近の成功 [Pytest run 31554813225](https://github.com/kfuku52/csubst/actions/runs/31554813225)（2026-08-12、`fe055f0`）を集計した。現在の HEAD 自体の CI 実行時間ではない。全 11 job の経過時間の合計は **1,362 秒 / 22.7 runner 分**。これは並列 job の合計であり、利用者の待ち時間や課金額ではない。

| job | job 全体（秒） | 導入（秒） | 主な実行部分（秒） |
| --- | ---: | ---: | --- |
| Full Python 3.12 | 148 | 93 | テスト 32 + process 12 |
| Fast Python 3.10 | 111 | 82 | テスト 17 |
| Fast Python 3.11 | 114 | 86 | テスト 17 |
| Fast Python 3.13 | 167 | 142 | テスト 15 |
| Fast Python 3.14 | 125 | 96 | テスト 18 |
| PGK / PEPC parity | 145 | 95 | parity 22 + calibration 15 |
| Pure Python | 58 | 26 | fallback 確認・テスト 20 |
| Packaging | 131 | tools 1 | sdist 15 + wheel 57 + smoke 20 + pytest 28 |
| macOS Intel | 183 | 166 | clean install / pip check / version の step |
| Sanitizers | 162 | 74 | sanitizer build 78 + テスト 4 |
| Lint | 18 | — | ツール導入・静的検査 7 |

秒数の根拠は [ci_timings.tsv](ci_timings.tsv)。run 一回の観測なので、中央値や将来の短縮率はまだ出せない。

改善の順序:

1. **同じ環境での繰り返しビルドを減らす。** 既に pip cache はあるが、各 job の editable install はプロジェクトの拡張を再ビルドする。Python 3.12 の full / parity / packaging で検証済み wheel を再利用できる構成を検討する。OS・architecture・Python ABI・NumPy / compiler / Cython・ソースの違いを区別し、誤ったバイナリを再利用しない。sanitizer と pure-Python lane は分離を保つ。job の依存関係追加で待ち時間が増えないかも比較する。
2. **同一変更の push / PR 二重実行を避ける。** 現在は push と pull_request が無条件で、同一リポジトリ内の PR では両方が起動し得る。`github.ref` が異なるため concurrency でも重複を消せない。通常は default branch の push と PR に分ける。後続 push に置き換えられた検証の cancellation も整理し、release の対象 commit の保証は維持する。
3. **プラットフォーム専用 job を必要な変更に対応させる。** macOS Intel の clean install は最長 job だが、日常の数値テストは行っていない。依存・ビルド・配布関連の変更、定期実行、release、手動実行での保証を残して起動条件を整理する。Mac 固有の E の検証を追加すると費用に見合う意味が増す。
4. **短い確認処理と artifact 保持を整理する。** 低コストな静的チェックの統合、必要な保持期間の指定を行う。coverage・Python 互換性・sanitizer・数値 parity を単純に削って高速化しない。

### 巨大な関数と広い可変状態が、変更の影響を追いにくくしている

vendor を除く production Python は 51 ファイル / 40,089 行、test Python は 99 ファイル / 25,646 行。`omega.py` は 4,464 行、`main_sites.py` は 4,046 行。`param.get_global_parameters()` は 790 行、CLI parser builder は 600 行である。行数だけを不具合とは扱わないが、設定・計算・描画・ファイル出力の境界を検討する手掛かりになる。

`g[...]` の文字列リテラルキーは静的に 345 種類、`get` / `setdefault` / `pop` も含めると 404 種類見つかる。設定だけでなく実行時状態も含む。一方、型定義のキーは 68 種類で、`make typecheck` は 5 ファイルのみ、import 先の型検査も省略する。既存の `RunContext` / `AnalysisConfig` を使い、まず入出力と stage 間の契約から検査を広げたい。

実際に `main_scan.py` / `main_analyze.py` / `main_download.py` を追加で mypy にかけると 2 エラーが出た。うち `main_scan()` は `-> None` と宣言されているのに L208 で `(g, scan_df, units_df)` を返している。もう 1 件は L23 の空リストの型注釈不足。通常の `make typecheck` は両方を見ていない。

改善案: CLI 引数から不変設定を作る部分、解析中の配列・cache・結果を持つ部分を段階的に分離する。次に omega の期待値計算 / 帰無モデル / p 値、および sites の解析 / 描画 / 構造連携を、小さな契約を持つ単位に分ける。PGK / PEPC parity を保ちながら変更ごとに進め、巨大な一括 refactor は避ける。

### テストの件数に比べて、CLI の失敗経路が手薄

vendor を除いた測定で statement coverage **76.41%**、branch coverage **63.98%**、coverage.py の合成値 **72.69%**。現行の 70% gate は満たす。一方、`main_analyze.py` の合成値は 45.90%、`main_scan.py` は 37.91%。subprocess 内の測定範囲にも限界があるため、これだけで未検証と断定はしないが、今回の A / B / C / D は全テスト成功だけでは検出できなかった。

改善案: helper の mock を増やすより、入力保全、非 0 終了、空の結果、readonly 配列、形式だけ変えた alignment を少数の CLI 回帰にする。配布 wheel / strict native / pure Python のどれを検証したかが結果から分かるようにする。

新規環境では `substitution_scan.py` L1543 の一列 `groupby` に対する `Pandas4Warning` も並列テストで 38 件出た。現状の計算失敗ではない。単一列なら scalar grouper を渡すことで警告をなくし、将来互換性の確認を追加する。警告の一括抑制や pandas の上限制約で隠さない。

## 5. 検証結果と限界

macOS 26.5.2 / arm64 で実行。追加環境とビルドには同じ HEAD の `git archive` コピーを使い、元の環境・ソースを変更しなかった。

| 環境・検証 | 結果 | pytest 表示時間 |
| --- | --- | ---: |
| 既存 Python 3.14.0、native、coverage あり、`not process` | 1,295 passed | 93.64 秒 |
| 同環境、process を直列実行 | 4 passed | 82.08 秒 |
| 新規 Python 3.12.13、native、`not process` | 1,295 passed、39 warnings | 63.68 秒 |
| 同環境、process を直列実行 | 4 passed、4 warnings | 30.83 秒 |
| 同環境、拡張無効、`not process and not parity` | 1,229 passed、25 skipped、38 warnings | 15.50 秒 |
| 元の環境で `make lint typecheck` | 成功 | — |
| 新規環境で既存の readonly 関連テスト、strict | 1 failed（C） | — |
| 通常 wheel のビルド・別 venv への install・6 拡張の import | 成功 | — |

新規環境の主な依存は NumPy 2.5.2 / SciPy 1.18.1 / pandas 3.0.5 / matplotlib 3.10.9 / ete4 4.4.0。既存環境は NumPy 2.4.2 / SciPy 1.17.1 / pandas 2.3.3 / matplotlib 3.10.8。native と記した通常テストは fallback を許す設定であり、C の脱落がある。拡張無効 lane は対象テストが異なり、速度比較には使えない。Python 3.12 / 3.14 も coverage・cache 条件が違うため性能比較ではない。

既存の PGK / PEPC 回帰スクリプトの `run_dataset` を strict native で実行し、`EXPECTED` と branch ID・ωC・サイト分類を比較した。ωC は絶対差 `1e-6` 以内で一致。既計算の IQ-TREE 出力を使用し、IQ-TREE 自体の再推定は行っていない。

| dataset | branch IDs | ωCany2spe | convergent / divergent / blank | analyze 秒 / peak RSS KiB | site 秒 / peak RSS KiB |
| --- | --- | ---: | --- | --- | --- |
| PGK | 23, 51 | 1.975050 | 5 / 7 / 390 | 2.28 / 201,056 | 6.18 / 290,688 |
| PEPC | 9, 108 | 0.049466 | 0 / 2 / 954 | 5.39 / 412,080 | 7.56 / 426,288 |

これは単一実行の現状測定。既存 Linux CI の性能しきい値は Mac に流用していない。さらに PGK の p 値校正回帰を `niter=100`、`min_sub_pp=0,0.05`、`any2any / any2spe` の 4 設定で実行して成功した。既存の回帰条件の成功であり、統計手法全体の正当性を新たに証明する検証ではない。

今回は Linux の sanitizer をローカルで再実行していない。Python 3.10 / 3.11 / 3.13、他 OS、巨大入力、全 optional dependency、外部 API / モデル download の全経路も網羅していない。提案を実装した後の性能改善率はまだ測定していない。数値・再現結果の機械可読版は [verification.json](verification.json)。

## 6. 再現コマンド

以下はリポジトリのルートから、対象の csubst を導入した環境で実行する。追加実験の出力先は `reports/generated/` または使い捨てディレクトリとし、実データを上書きしない。

通常検証:

```bash
python -m pytest -q -n auto --dist worksteal -m 'not process' --durations=25
python -m pytest -q -m process --durations=25
CSUBST_DISABLE_EXTENSIONS=1 python -m pytest -q -n auto --dist worksteal -m 'not process and not parity'
make lint typecheck
python -m mypy --follow-imports=skip csubst/main_scan.py csubst/main_analyze.py csubst/main_download.py
```

C の最小再現（pandas 3、ビルド済み拡張がある環境）:

```bash
CSUBST_STRICT_EXTENSIONS=1 python -m pytest -q tests/unit/test_substitution_dense_tensor.py::test_get_sub_sites_sn_applies_dirichlet_pseudocount
```

A / B / D / E の安全な最小再現（既存ファイルは指定しない）:

```python
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from csubst import parser_iqtree, runtime, sequence_io

with tempfile.TemporaryDirectory(prefix="csubst-audit-") as tmp:
    base = Path(tmp)
    alignment = base / "alignment.fa"
    tree = base / "tree.nwk"
    original = ">A\nATGGCT\n>B\nATGGCC\n"
    alignment.write_text(original)
    tree.write_text("(A:0.1,B:0.1);\n")
    proc = subprocess.run(
        [sys.executable, "-m", "csubst", "doctor",
         "--alignment_file", str(alignment), "--rooted_tree_file", str(tree),
         "--log_file", str(alignment), "--outdir", str(base / "doctor")],
        cwd=base, capture_output=True, text=True,
    )
    print("A:", proc.returncode, "input_unchanged=", alignment.read_text() == original)
    proc = subprocess.run(
        [sys.executable, "-m", "csubst", "benchmark",
         "--alignment_file", "missing.fa", "--rooted_tree_file", "missing.nwk",
         "--outdir", str(base / "benchmark")],
        cwd=base, capture_output=True, text=True,
    )
    print("B:", proc.returncode,
          [line for line in proc.stdout.splitlines() if "Benchmark summary:" in line])
    alignment.write_text(">A\nATG GCT\n")
    print("D reader:", sequence_io.read_fasta_records(alignment)[0].sequence)
    try:
        print("D sites:", parser_iqtree._infer_num_input_site_from_alignment_file(alignment))
    except (AssertionError, ValueError) as exc:
        print("D:", type(exc).__name__, str(exc))

os.environ.pop("VECLIB_MAXIMUM_THREADS", None)
runtime.configure_native_threads(1)
print("E:", json.dumps({key: os.environ.get(key) for key in
      ["VECLIB_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]}))
```

PGK の実 CLI 再現（C）:

```bash
CSUBST_STRICT_EXTENSIONS=1 python -m csubst search \
  --alignment_file csubst/dataset/PGK.alignment.fa \
  --rooted_tree_file csubst/dataset/PGK.tree.nwk \
  --foreground csubst/dataset/PGK.foreground.txt \
  --iqtree_model ECMK07+F+R4 \
  --iqtree_treefile csubst/dataset/PGK.alignment.fa.treefile \
  --iqtree_state csubst/dataset/PGK.alignment.fa.state \
  --iqtree_rate csubst/dataset/PGK.alignment.fa.rate \
  --iqtree_iqtree csubst/dataset/PGK.alignment.fa.iqtree \
  --iqtree_log csubst/dataset/PGK.alignment.fa.log \
  --expectation_method urn --asrv sn --exhaustive_until 1 --threads 1 \
  --outdir reports/generated/implementation_audit_20260831/strict-sn
```

実データ parity は `.github/scripts/sites_parity_check.py` の `run_dataset()` と `EXPECTED` の比較、校正回帰は次のコマンドを使用した。監査時の一時出力先を以下ではリポジトリ相対パスに置き換えている。

```bash
CSUBST_STRICT_EXTENSIONS=1 python .github/scripts/omega_pvalue_calibration_check.py \
  --niter 100 --min-sub-pp-levels 0,0.05 \
  --workdir reports/generated/implementation_audit_20260831/calibration \
  --output reports/generated/implementation_audit_20260831/calibration.tsv \
  --runtime-output reports/generated/implementation_audit_20260831/calibration-runtime.tsv
```

## 7. 実装するならこの順序

1. **入力保全と失敗通知**: A / B を修正し、CLI の失敗経路を回帰テストにする。
2. **互換性と環境差**: C / D / E を修正し、最新依存・strict native・pure Python で再検証する。警告の整理もここで行う。
3. **開発・配布の再現性**: H の dev 環境、G の wheel 数値検証、F の明示的 release 連携を整える。
4. **CI の重複削減**: 上記の保証を保ってビルド再利用・イベント・platform 起動条件を整理し、複数 run で待ち時間と runner 時間を比較する。
5. **変更しやすい構造へ段階移行**: 型検査を stage 境界から広げ、巨大な設定・計算・描画処理を分割する。変更ごとに数値 parity と必要な性能測定を添える。
