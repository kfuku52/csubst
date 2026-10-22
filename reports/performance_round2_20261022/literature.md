# 第2回候補の設計根拠

基準は CSUBST 1.16.21 (`88528ce`)。前回の10案と採用済み4変更は
[前回調査](../performance_candidates_20261022/README.md)と
[実装記録](../performance_implementation_20261022/README.md)を確認した。
文献・APIの性質から速度を推定せず、各試作で現行実装との差を測定する。

| 案 | 新しい検証対象 | 前回案との差 |
| --- | --- | --- |
| 01 | one-hot葉尤度の遷移行列列参照 | 全tipパターンの圧縮ではなく、末端messageだけの特殊化 |
| 02 | 割当てごとの枝metadata配列・対象mask準備 | score-only出力省略に加え、候補間で同じmetadataを再利用 |
| 03 | 既存イベントcache内の枝数値vector | groupby省略後に残るDataFrame変換の反復を対象とする |
| 04 | 解析的検定で変わらない部分木の尤度再利用 | 観測posteriorではなく、明示的な代替分布の再pruningを対象とする |
| 05 | omega経験的tailの列chunk化 | 推定量・乱数はそのまま、統計変換の一時配列を抑える |
| 06 | 共通maxT参照の整列・tail検索 | null試行を減らさず、同じ参照への反復問合せだけを置換 |
| 07 | synonymous変換の固定遷移積を再利用 | out bufferではなく、block間で一定のmatrix積の準備を共有 |
| 08 | unit supportの枝membershipを配列化 | 表示文字列ではなく、unitごとの支持計算を対象とする |
| 09 | uniformizationの到達距離に基づく開始次数 | 固有値近似を使わず、非負級数と相対誤差条件を維持できるか検証 |
| 10 | 構造重み付けCVのlog-prior等のfold内再利用 | 既存base/interaction cacheの後に残るgrid内反復を対象とする |

## 参照資料

- [NumPy `take`](https://numpy.org/doc/stable/reference/generated/numpy.take.html):
  指定軸の要素参照。案01ではone-hotの行列積との数値同等性と参照準備の費用を別に検査する。
- [NumPy copies and views](https://numpy.org/doc/stable/user/basics.copies.html):
  配列の所有権とcopy費用。案02・03の寿命、変更可能入力、追加保持を評価する根拠。
- [NumPy `searchsorted`](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html)と
  [`isclose`](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html):
  案06ではinclusive tailと既存の近似tieを両立する必要があり、丸め境界も検査する。
- [SciPy `logsumexp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html):
  尤度・混合分布の計算をlog空間に保つ。案04・10で安易な指数化や加算順変更を避ける。
- [Kobert et al. (2017), Efficient Detection of Repeating Sites to Accelerate
  Phylogenetic Likelihood Calculations](https://doi.org/10.1093/sysbio/syw075):
  部分計算再利用の先行研究。今回の案04はsite repeatの圧縮を実装するものではなく、
  同じ検定内で変更されない部分木を再利用する試作である。
- [Fox and Glynn (1988), Computing Poisson Probabilities](https://web.stanford.edu/~glynn/papers/1988/FG88.html)と
  [Sherlock (2021), Direct statistical inference for finite Markov jump processes
  via the matrix exponential](https://doi.org/10.1007/s00180-021-01102-6):
  案09の非負uniformization級数と打切り誤差を検討する資料。
  全体の絶対誤差だけで微小な遷移の相対精度を保証したと扱わない。

製品側の科学的契約は [endpoint](../../docs/ENDPOINT_POSTERIORS.md)、
[scan校正](../../docs/SCAN_CALIBRATION.md)、
[解析的検定](../../docs/SCAN_ANALYTICAL.md)、
[構造重み付け](../../docs/STRUCTURE_WEIGHTING.md)を優先する。
各担当者による追加資料・適用条件は各案のMarkdownに記載する。
