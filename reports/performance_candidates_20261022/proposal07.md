# 案7: 候補固定校正でスコア専用の集約を使う

**評価: 採用を勧める候補。** 同じ10割当ての候補固定校正で、PGKは19.2%、
PEPCは17.8%のwall time短縮が得られ、全スコア・候補キー・反復診断が厳密一致した。
製品コードは変更していない。これは校正workerの処理単体であり、CLI全体の改善率ではない。

## 文献・コード調査と案の改訂

当初の「iterrowsとID文字列解析を反復外へ出す」案は、現行 `4498c43` の
`substitution_scan._run_scan_permutation_chunk` が既に
`observed_df.to_dict('records')` とID解析をchunkごと一度行い、
`_fixed_candidates` に保持していたため、新規改善として採用しなかった。
[pandas iterrows公式文書](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iterrows.html)と
[to_dict公式文書](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_dict.html)の
推奨を機械的に当てはめず、実際の呼出し元まで確認した。

残っていた費用は、候補固定校正でも `_rate_summary` が最終TSV用の全フィールドを作り、
その後 `scan_statistics.calibration_score` が5項目しか参照しないことである。
不要な枝長・枝数の集計、rate ratio、理由文字列と
`np.unique(..., return_counts=True)` を使う診断作成を省く試作へ改訂した。
[NumPy unique公式文書](https://numpy.org/doc/stable/reference/generated/numpy.unique.html)が
定義する重複除去・カウントは最終出力では必要だが、校正スコアには使われない。

試作は同じ `_rate_summary` の入力検証・イベント集約・exposure計算・
`rate_score`・正のイベントに対するゼロexposure判定を残す。
返す項目を `score_rate_enrichment`, `target_event_count`, `other_event_count`,
`target_exposure`, `other_exposure` に絞り、現行と同じ `calibration_score` に渡す。
既存の候補・イベント・exposure cache、割当て生成、乱数、閾値は変えない。
通常の候補表・full-scan出力では従来の全フィールドを引き続き計算する。

## 測定

基準コミット `4498c43`。macOS 27.0.1 arm64、Python 3.14.7、
NumPy 2.5.3、SciPy 1.17.1、pandas 3.0.5。
NumPy/SciPy import前にVECLIB/OpenBLAS/OMP/MKLのthread環境変数を1に設定した。
共有ファイルロックで他案のCPU測定と直列化した。

同梱のPGK/PEPC固定fitを実CLIで準備し、校正workerの呼出し直前に捕捉した。
1 worker、10割当て、seedは同じまま、ウォームアップ1回後に順序を交互にして3回測定。
両版とも現行のchunk関数を使い、candidateの準備・ID解析も各chunkの時間に含める。
現行のworker unpack処理が各回でイベント・exposure cacheを初期化するので、
一方だけ前の計測の結果cacheを引き継ぐ比較にはしていない。
入力準備、joint推論、観測候補の発見、最終P値集約・TSV保存は時間に含まない。

各セルは3回の中央値（最小–最大）、秒。

| 条件 | 現行wall | スコア専用wall | 短縮 | 現行CPU → スコア専用CPU |
| --- | ---: | ---: | ---: | ---: |
| PGK・12候補×10割当て | 0.05712 (0.05554–0.05794) | 0.04615 (0.04337–0.04676) | 19.2% | 0.05600 → 0.04611 |
| PEPC・103候補×10割当て | 0.53671 (0.53257–0.57992) | 0.44132 (0.40751–0.44665) | 17.8% | 0.50371 → 0.42691 |

両条件でwall範囲が分離した。PEPCのCPU短縮は15.2%。
既定1,000割当て全体や、解析CLI全体の同じ割合の短縮を意味しない。
前段の試行ではPEPCのwall範囲が重なったため、実endpointの異常状態チェックを
追加した最終スクリプトの再実行結果を上表・JSONに記録した。

## 正確性・失敗契約・メモリ

- 各データ・各測定ペアで、全候補キーごとの10スコア、最大値、成功/失敗数、
  seed・configuration IDを含む全反復診断を厳密一致で直接比較した。
  両データとも10成功・0失敗だった。
- 空候補族のchunkも両データで厳密一致した。成功したno-testを失敗や省略にしない。
- 28条件で、raw/state-aware/Q-weighted exposure、対象/対照ゼロ長、欠損状態、
  負または無限イベント、空イベント、endpointの正イベントとゼロexposure、
  endpoint文脈の欠落・枝順不一致・不正な長さモードを照合した。
  スコア、明示的no-testの−∞、例外型・メッセージが一致した。
- 実PGK/PEPCから捕捉したendpoint入力でも、負値・NaN・非正規化stateを同じエラーで拒否し、
  子の欠損stateで同じスコア/no-testになることを確認した（追加8条件）。
- exposureの既存cacheはそのままで、fitをまたぐ新しい永続cacheは追加していない。
  スコア専用の返却dictは小さくなるが、保持配列・cacheは同じ。
  ピークRSSやworker合計物理RAMは測っておらず、RAM削減率は主張しない。

## 採用条件と残る検証

製品化では、イベント・exposure・スコア計算の共通数値コアを分離し、
全フィールド出力とスコア専用出力の両方から使うことが望ましい。
今回の実験用の関数複製をそのまま製品へ入れると、将来の仕様変更で両者がずれる危険がある。
公開の `_rate_summary` 全体を置換せず、候補固定校正だけに内部経路を限定する。

入力状態は現行と同じ固定fitのworker内の寿命で使用し、各割当ての対象・対照集計は
毎回行う。full-scanの候補発見を固定して高速化する変更は含まない。
複数traitの割当て校正は現行仕様で未対応であり、この試作で対応済みとはしない。
複数worker、source-only fallback、native/installed wheel、Linux/Windows、
全CLI表と最終P値集約を含む統合回帰は未実施。科学的校正の検証でもない。

## 再現

```bash
python reports/performance_candidates_20261022/proposal07.py
python -m ruff check reports/performance_candidates_20261022/proposal07.py
git diff --check
```

上記コマンドは成功。[再現スクリプト](proposal07.py)と
[全測定値・入力ハッシュ・環境](proposal07.json)に証拠を保存した。
製品コードの変更がないため通常のpytest全体は実行していない。
