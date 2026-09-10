# 科学的レビュー ID 6: scan の条件付き foreground 較正

2026-09-10。CSUBST 1.15.0、base commit
`dd37bee67b39199a1920600cf00c0ecb8566f477` に対するローカル修正。
検証環境は Python 3.12.14 / macOS arm64。

## 修正と確認した問題

- 観測 foreground を除外せず、ラベルによらず固定した適格クレードと
  サイズ bin から非重複配置を一様抽出する。小空間は全列挙し、大空間は
  配置全体の一様提案と重複配置の棄却を用いる。
- lineage ごとの bin 別成分数を保持し、配置の一部として所属も抽出する。
  サイズ順で事後的に lineage へ割り当て直す処理を廃止した。
- 正常な「候補なし」は 1 を寄与する。NaN、ゼロ exposure による未定義値、
  例外は成功試行として扱わず、1 件でもあれば較正 P/q を出さない。
  成功試行だけに分母を縮めず、失敗後に抽出空間を切り替えない。
- 全列挙では正確な尾の割合、Monte Carlo では ties を含む
  `(1 + extreme_count)/(B + 1)` を使う。各配置は各候補へ一度だけ寄与する。
- 配置 ID、抽出数、失敗、候補数、有限統計量数、bin、枝長、深さ、状態の
  欠損・不確実性を独立した `scan_calibration.json` に記録する。
  候補ゼロでも診断を保存する。
- 較正は単一 trait に限定する。複数 trait の候補一覧は `none` で利用できるが、
  複数 trait の依存関係を保持する共同帰無は未実装。

scan 専用の [抽出器](../../csubst/scan_permutation.py) を追加した。
既存の omega 検索用 foreground permutation は変更していない。
[較正処理](../../csubst/substitution_scan.py)、CLI、出力、型定義、テスト、
[設計文書](../../docs/SCAN_CALIBRATION.md) を更新した。
Wiki は別のローカルコピーで更新し、変更を [wiki.patch](wiki.patch) に保存した。
Wiki の base は `7fdd7ccb607936ad795e1b362f6d26d0fe26aafb`。
パッチは文脈0行形式で保存しており、適用時は `git apply --unidiff-zero` を使う。
初回実装時点では元作業ディレクトリへの書き込み、commit、push は行っていない。
その後の本体統合・再監査は末尾に記録した。

### 回帰例

| 例 | 修正前 | 修正後 |
| --- | --- | --- |
| 4 末端から前景 2 個を一様に選ぶ 6 配置。固定イベントは A/C、B=99 | 観測 AC で P=0.01。名目 5% で 6 配置中 1 配置を棄却 | 両モードで全 6 配置を評価し P=1/6。名目 5% の棄却は 0 配置 |
| `((A,B)X,((C,D)Y,(E,F)Z)W)R` の内部クレード 2 個 | WX/XY/XZ/YZ の確率は 1/3, 5/24, 5/24, 1/4。重複無視の組合せ数は 6 | 有効な 4 配置を正確に列挙。各確率 1/4。大空間用抽出器も 12,000 回の抽出で照合 |
| 代替前景の exposure がゼロ | candidate_fixed で未定義値を分子から落とし、P=0.01 を生成できた | candidate_fixed/full_scan とも未定義試行を記録し、較正値は unavailable |
| A/C と B/D に別サイトの候補 | 候補固定と再探索は異なる検定範囲 | 全列挙で候補別 P=1/6、full_scan の minP 調整値=1/3 を確認 |

## 再現可能な配列生成・ASR 検証

[検証スクリプト](../../.github/scripts/scan_calibration_check.py) は等頻度 20 状態
CTMC から配列を生成し、固定系統上で全体の率を推定する。独立に実装した
pruning / inside-outside 計算で祖先状態を復元し、実際の CSUBST 置換テンソル
生成と full_scan に渡す。参照 ASR 自体は、小系統の潜在状態全列挙と照合した。

```bash
python .github/scripts/scan_calibration_check.py \
  --replicates 1000 --workers 2 --sites 64 --states 20 \
  --require-calibration-bound \
  --output reports/generated/scan_id6_calibration.json
```

seed=20260910、名目水準 0.05。state_aware exposure、raw fitted length、
min_clade_bin_count=1、any2spe、支持数 2、event PP 閾値 0.5 を用いた。
各条件は前景配置が 28 通りで、候補のある実行では全配置を評価する。
この空間では ties がなくても最小 P は 1/28 であり、名目 5% の検定は離散的。
候補なしのデータセットも外側反復の分母に含めた。

事前の実用基準は「二項分布の両側 95% 区間の上端が 0.06 以下、かつ
評価不能な実行が 0」。これはこの簡略化モデルでの検証基準であり、
あらゆる系統・モデルについての証明ではない。

| 条件（各 1,000 回） | 帰無サイトを一つ以上棄却 | 95% 二項区間 | 候補なし | 評価不能 |
| --- | ---: | ---: | ---: | ---: |
| 平衡系統、8 末端、前景クレード各 1 末端 | 1.7% | 0.99–2.71% | 957 | 0 |
| 櫛状系統、8 末端 | 2.6% | 1.71–3.79% | 936 | 0 |
| 平衡系統、16 末端、前景クレード各 2 末端 | 3.3% | 2.28–4.60% | 679 | 0 |
| 枝長不均一（枝ごとに 0.2 または 4 倍） | 2.8% | 1.87–4.02% | 852 | 0 |
| 独立なランダム欠損 20% | 2.1% | 1.30–3.19% | 971 | 0 |
| 前景 60% / 背景 10% の欠損 **〔感度分析〕** | 0.4% | 0.11–1.02% | 995 | 0 |
| 8/64 サイトに前景特異的な遷移 **〔感度分析〕** | 2.6% | 1.71–3.79% | 220 | 0 |

帰無 5 条件は全て基準を満たした。真陽性混在条件では、1,000 回中 500 回で
少なくとも一つの信号サイトを検出した。後ろの 2 条件は較正の保証範囲に含めない。
数値と設定は [calibration_summary.json](calibration_summary.json) に保存した。
全反復の出力は上記コマンドで再生成できる。

## 実行・ソフトウェア検証

最終ソースで以下を確認した。

```bash
python -m pytest -q -n auto --dist worksteal -m "not process"
python -m pytest -q -m process
CSUBST_STRICT_EXTENSIONS=1 python -m pytest -q -m native
make lint
make typecheck
```

- ネイティブ拡張あり: 非 process 1,553 passed + process 4 passed = **1,557 passed**。
- strict native: **7 passed**（上記全体テストの部分集合）。
- Python 経路でも全体 1,519 passed / 32 skipped、process 4 passed を確認。
  この時点の skip はネイティブ拡張未ロードによるもの。最終ネイティブ実行では skip なし。
- Ruff、repository hygiene、ドキュメント検査、リポジトリ指定の型検査を通過。
  更新した Wiki もドキュメント検査と patch の適用確認を通過した。
- PGK の同梱配列・系統・事前計算済み IQ-TREE 出力を一時領域へコピーして
  CLI を実行。候補 12 件、20/20 試行成功、配置空間 232、ユニーク配置 18、
  global assignment P=4/21 を確認した。これは実行・出力検証であり、PGK の
  生物学的帰無が正しいという検証ではない。

PGK の再現コマンド（同梱入力を作業用ディレクトリへコピーする）:

```bash
python - <<'PY'
from pathlib import Path
from shutil import copy2
target = Path('reports/generated/scan_id6_pgk/input')
target.mkdir(parents=True, exist_ok=True)
for source in Path('csubst/dataset').glob('PGK.*'):
    copy2(source, target / source.name)
PY
python -m csubst scan \
  --alignment_file reports/generated/scan_id6_pgk/input/PGK.alignment.fa \
  --rooted_tree_file reports/generated/scan_id6_pgk/input/PGK.tree.nwk \
  --foreground reports/generated/scan_id6_pgk/input/PGK.foreground.txt \
  --iqtree_treefile reports/generated/scan_id6_pgk/input/PGK.alignment.fa.treefile \
  --iqtree_state reports/generated/scan_id6_pgk/input/PGK.alignment.fa.state \
  --iqtree_rate reports/generated/scan_id6_pgk/input/PGK.alignment.fa.rate \
  --iqtree_iqtree reports/generated/scan_id6_pgk/input/PGK.alignment.fa.iqtree \
  --iqtree_log reports/generated/scan_id6_pgk/input/PGK.alignment.fa.log \
  --outdir reports/generated/scan_id6_pgk/output \
  --scan_n_permutations 20 --scan_site_plot no --threads 1
```

## 残る範囲・他項目との関係

- 交換可能性はサイズ bin だけでは成立しない。観察的な foreground については
  配置帰無の科学的な正当化が必要。診断出力はその判断材料であって証明ではない。
- 今回の配列検証は等頻度モデル、固定系統、率一つの推定、参照 ASR に限られる。
  codon-Q、native 3Di、IQ-TREE による全パラメータ再推定、全ゲノム、
  モデル選択を含む解析全体の較正は未検証。
- 大きな空間の Monte Carlo は一様抽出器の検証と PGK 実行で確認した。
  巨大な配置空間全般の生物学的 FWER を測ったものではない。
- candidate_fixed と候補別 BH は、同じ foreground による探索後には探索的な値。
  full_scan も部分対立下の強い FWER や複数実行をまたぐ補正を保証しない。
- ID 5 の解析的 Poisson P は変更していない。ID 7 の exposure、ID 1 の 3Di
  モデル・mask・根、ID 2/4 の omega 帰無は独立作業。これらを変更した後に
  対象モデルで較正を再確認する必要がある。

## 本体統合前の最終監査（2026-09-10）

本体 master `feab6e4` の変更と統合し、最終的に別タスクのcommit
`6b45270` も含めて再監査した。抽出空間、配置の一様性、
観測配置での統計量の再現、再探索、未定義値・失敗の扱いについて、追加の
計算上の不具合は見つからなかった。別タスクの staged / unstaged 変更を
保持し、重なった型定義は両方の項目を残した。今回の差分だけを本体 master
へのローカルcommit対象とする。pushは行わない。

- commit対象を本体HEADと統合した隔離コピー: 1,872 passed、2 skipped。
  skipは任意依存 `gemmi` がない構造参照テスト2件。native専用7件も成功。
- 別タスクの未コミット変更を含む本体のPython実装: 1,840 passed、34 skipped。
  skipは拡張無効化に伴うnative検証32件と上記任意依存2件。
- 本体の lint、repository hygiene、型検査、Wikiを含む文書検査は成功。
- 最新本体と統合したPGK実行: 12候補、20/20試行成功、配置空間232、
  異なる配置18、global P=4/21。JSONとTSVの較正状態・数値精度を照合。
- sdist生成、Twine検査、新規モジュール・検証スクリプト・文書・テストの
  アーカイブ同梱を確認。

本体に残っていたIntel用拡張とARM用Pythonの不一致による初回テストの
import失敗は、実行環境の組み合わせによるものだった。本体の既存バイナリを
変更せず、本体は `CSUBST_DISABLE_EXTENSIONS=1`、隔離コピーはCPU種別の
合う6拡張を使って上記検証を完了した。sdistの初回非隔離ビルドで不足した
Cython/wheelは、通常の隔離ビルドで解決した。

本タスクの実装と検証は完了。単一traitの条件付き配置帰無という適用範囲、
簡略モデルを超えたcodon/3Di全般の生物学的較正が未検証である点は、
引き続き上記の制限に従う。
