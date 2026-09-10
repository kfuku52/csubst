# 擬似カウント併用 ωC P 値の修正計画

2026-09-10 に策定した計画。現在の[実装・検証結果](RESOLUTION_PSEUDOCOUNT_PVALUES.md)を別記した。
以下は計画策定時の内容。
対象は `expectation_method urn` の empirical P 値。3Di の GTR 期待値に
新しい P 値を追加する計画ではない。

## 確認済みの不整合

観測側 `omega.get_omega` は擬似カウントを加えた ωC を比較対象にするが、
帰無側 `omega_statistics._calc_permutation_omega_matrix` は未補正の比を使う。

再確認した最小例：OCN=1, ECN=10, OCS=10, ECS=10、symmetric alpha=1、
target=both。全 1,000 帰無反復を観測と同じカウントに固定すると、
観測 ωC=2/11=0.181818…、帰無 ωC=0.1 となり P=1/1001。
同じ変換なら全反復が同値となり P=1 であるべき。
この実験は比較処理の不整合を証明するもので、実データでの偽陽性率ではない。

## 採用する統計量と計算順序

各カテゴリ k について、観測・各帰無反復とも同じ関数を通す。

```
dN = (ON + alpha_ON[k]) / (EN + alpha_EN[k])
dS = (OS + alpha_OS[k]) / (ES + alpha_ES[k])
T  = dN / dS
```

- target=observed/expected/both に応じて四つの alpha を適切にゼロ化する。
- 生のカテゴリカウントを集計した後で平滑化する。平滑化した比同士の差を取らない。
- 0/0、正値/0、NaN、infinity、float_tol の規約も共通化する。
  alpha=0 / mode=none は従来の未補正計算と一致させる。
- long-tail 補正を使う場合は、観測・帰無とも「平滑化→dSC 補正→ωC」の順。
  帰無反復ごとに、同じ枝組合せ集合で補正を計算する。
- 有効反復数 B と上側同値を含む回数 r から (r+1)/(B+1) を計算する。
  微小な丸め誤差による同値の取りこぼしを独立に検証する。
  この式だけで、現在のカウント帰無モデル全体が厳密検定になるわけではない。

## 実装順序

### 1. 固定 symmetric alpha の比較処理を修正

`omega_statistics` に配列用の共通変換を置き、`get_omega` と帰無計算が
同じ実装を使う。`_get_pseudocount_context` の四つの alpha と対象カテゴリを
名前で対応付け、帰無側へ渡す。出力列の選択・順序が変わってもずれない設計にする。

この段階では既存の base カテゴリに対する固定 symmetric alpha を検証対象とする。
未検証のデータ依存 smoothing を固定値扱いして、対応済みとはしない。
対応完了前の `empirical` / `alpha=auto` と P 値の組合せは明示的に拒否する方針。
`pseudocount_report` のみ、mode=none、実効 alpha=0 は拒否しない。

### 2. dif カテゴリの帰無分布を整合させる

ID 4 の独立乱数問題は平滑化だけでは直らない。
同じ原子的イベント／ランダム化結果から any/spe と派生 dif を同時集計し、
包含関係とカテゴリ間の共分散を保存する。
負値のクリッピングや NaN 試行の除外で代用しない。
null_model ごとに実装可能性を検証し、対応前の dif P 値は明示的に制限する。

### 3. empirical prior と alpha=auto の推定手順を含める

推奨方針は、観測から推定した prior / alpha を無説明に固定せず、
各反復の擬似データ集合から同じ手順で再推定すること。
ここで E は既存の fitted count-null に条件付けた固定期待値として扱う。
E や ASRV 自体の再推定を含む完全なパイプライン bootstrap は別の推論対象となる。

- 観測で prior / alpha を推定した枝集合・カテゴリ集合を各反復でも使用する。
- 全カテゴリで同一の反復 ID を共有する。表示カテゴリだけから prior を
  再構成したり、カテゴリごとに独立の擬似データを作ったりしない。
- 各反復の全枝集合が必要なため、段階的反復で active rows だけに絞る既存処理を
  そのまま使わない。全体依存の推定・long-tail 補正には固定した行集合を保つ。
- メモリは反復方向のブロック処理で制御する。ブロックサイズを変えても
  seed と反復 ID に対する結果が変わらないようにする。
- 観測から推定した prior を固定する近似を別途導入するなら、推論対象と
  制約を明示し、再推定方式と区別する。今回の第一選択にはしない。

### 4. 出力・診断・文書を揃える

`pomegaC*` が実際に比較した raw/smoothed/calibrated 統計量を記録する。
既存の `_raw` / `_smoothed` / `_nocalib` 列と対応を一致させる。
有効 B、undefined 反復数、alpha の固定／再推定、prior の扱いを記録する。
CLI validation、benchmark、全呼出し元、例、説明文書も同時に更新する。
P 値修正後に q 値を再計算し、異なる処理の古い q 値を残さない。

## 必須検証・完了条件

1. 観測と全帰無カウントが同一なら P=1。target の三通りと複数カテゴリで確認。
2. alpha=0 / smoothing 無効は従来と一致。零カウント・零期待値・NaN・infinity を含む。
3. 小さな帰無空間を全列挙し、独立した計算と尾確率を比較する。
4. long-tail の on/off、カテゴリ順序、行順序、反復ブロックサイズ、
   単段／多段実行で統計量と反復 ID の整合性を検証する。
5. dif の包含関係、全イベントが specific のときの恒等的な差分ゼロを検証する。
6. empirical / auto は独立した素朴な反復再推定実装と比較する。
7. 疎な S、短い配列、異なる arity・ASRV・null_model を含む独立 null シミュレーションで
   名目水準と棄却率を比較する。反復数と二項信頼区間を報告し、
   機械的な一致テストだけで FPR 較正済みとは主張しない。
8. 全体テスト・lint・型チェックを通し、未対応の組合せを文書とエラーで一致させる。

## 根拠

- [Phipson & Smyth (2010)](https://pubmed.ncbi.nlm.nih.gov/21044043/):
  ランダムに抽出した permutation と Monte Carlo P 値の扱い。
- [Winkler et al. (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/):
  帰無反復で同じ統計量を計算することと、交換可能性の条件。
- 現行コード：`csubst/omega.py` の `get_omega`, `_get_pseudocount_context`,
  `add_omega_empirical_pvalues`、`csubst/omega_statistics.py` の
  `_calc_permutation_omega_matrix`。
