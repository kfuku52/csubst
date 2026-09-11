# PEPC joint endpoint: 枝別表のイベント集計最適化

変更前は `72c3c45` の凍結ソース、変更後は同じコミットに本変更を適用した作業ツリー。PEPCのarity 6までのsearchで、枝別表ありの処理本体が約29–30%短縮した。全32試行で数値と検出結果の同等性を確認した。

## 測定結果

秒、ウォームアップ1回を除く3回中央値（最小–最大）。BLASは1。

| 枝別表 | CPU | 変更前 | 変更後 | 時間短縮 |
|---|---:|---:|---:|---:|
| あり | 1 | 19.30 (19.03–19.52) | 13.70 (13.49–15.54) | 29.0% |
| あり | 4 | 19.62 (19.14–20.14) | 13.76 (13.49–13.79) | 29.9% |
| なし | 1 | 10.22 (9.33–11.39) | 9.18 (9.17–9.22) | 効果と解釈しない |
| なし | 4 | 9.55 (9.29–9.68) | 9.34 (9.19–9.37) | 効果と解釈しない |

入力読み込み、推論、arity 2–6、TSV出力と検証用pickle保存を含む。Python起動と共通モジュール事前読み込みは含まない。起動を含む実時間、CPU時間、RSS、全コマンド、ソース・拡張・入力のSHA256は [matrix.json](matrix.json) に記録。

共有マシンでの測定であり、ホスト全体の稼働は約8.9–11.5 CPU相当だった。最初の4試行では別のpytestプロセスも観測した。変更前後を交互に実行し、枝別表ありでは各条件の測定範囲が重ならない短縮を確認した。ただし無負荷環境の速度保証ではない。枝別表なしの経路はほぼ同じで、時間差は負荷変動の影響を含むため高速化とは主張しない。

親プロセスのピークRSS中央値は枝別表ありCPU 1で820→754 MiB、CPU 4で808→764 MiB。

## 変更と正しさ

観測側では61×61コドンイベントの中間配列を展開してからS/Nへ変換する処理を、Cythonで最終的なS/Nイベントへ直接加算する処理に変更。期待値側は必要な周辺集計だけを計算する。最大確率の置換は速度カテゴリを混合した後に選ぶ。既存拡張に新カーネルがない環境は既存のNumPy計算経路を使う。全イベントを必要とする設定は既存経路を維持する。

全32試行のarity 2–6の丸め前表、枝ID、候補の閾値判定を比較。枝別表あり16試行では置換文字列を含む枝別表も比較した。件数の許容誤差は絶対・相対とも1e-10、比は検証済み分子・分母の差を伝播した誤差範囲内であることを確認。NaNと±Infの位置も一致した。詳細は [validation.json](validation.json)。

arity 2–6の候補行数は8446 / 416 / 87 / 14 / 1、閾値通過数は85 / 45 / 20 / 6 / 1。arity 6の枝IDは全試行で `[1, 26, 33, 35, 40, 102]`。

テストは2107 passed、5 skipped。ネイティブ専用は9 passed（全体テストと重複）。lint・文書検査・型検査も成功。混合速度、S/N・構造状態、欠損、ブロックサイズ、枝別最大置換の同等性を確認。[checks.log](checks.log) に型注釈修正前の失敗と修正後の成功も記録。

## 再現

環境: Apple M2 Max、64 GiB、12論理CPU。Python 3.10.14 / NumPy 1.26.4 / SciPy 1.15.2、macOS上のx86_64実行。両ソースツリーに互換性のあるビルド済み拡張が必要。

```bash
python .github/scripts/benchmark_endpoint_scaling.py \
  --baseline-root /tmp/csubst-arity6-before-72c3c45 --baseline-mode joint \
  --scenarios heuristic6_b heuristic6 --cpus 1 4 --blas 1 --repeats 3 \
  --workdir /tmp/csubst-endpoint-events-matrix \
  --result reports/endpoint_event_optimization_20260910/matrix.json
python reports/endpoint_event_optimization_20260910/verify.py \
  --workdir /tmp/csubst-endpoint-events-matrix
```

ワーカーの生データ・ログは上記workdirに保存。cutoffは `OCNany2spe,2.0|omegaCany2spe,5.0`、既存IQ-TREE入力を使用し再推定しない。
