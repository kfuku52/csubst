# 文献調査と10案の選定（2026-10-22）

対象は CSUBST `4498c43`。2026-10-15 の既存調査と現行実装を確認してから、
既に導入済みの木の postorder 集約、区間 clade index、候補形状別 groupby 共有、
割当てに依存しないイベント・exposure の bounded cache は新規案に数えなかった。
文献の速度倍率を CSUBST の改善率には転用しない。

系統尤度計算、行列指数、配列・データフレーム処理、ファイル入出力、
重み付き非復元抽出を調査した。下記は選定根拠であり、全研究領域の系統的レビューではない。
10案それぞれを別のサブエージェントへ割り当て、現行コードと試作の比較を依頼した。

| 案 | 仮説 | 主な根拠・確認すべき条件 |
| --- | --- | --- |
| 1 | 完全一致する tip emission のサイトを一度だけ推定し、元の順へ展開 | パターン圧縮。曖昧状態の近似統合は禁止。サイト固有マスクは展開後に適用 |
| 2 | endpoint の site block を調整し、BLAS呼び出しを効率化 | BEAGLE のバッチ処理・問題サイズ依存。workspace 増加を評価 |
| 3 | 可逆 Q の固有分解を transition 作成で共有 | matrix exponential。短枝・稀な遷移の相対精度を維持できなければ見送り |
| 4 | 疎行列 spool の断片読み出しを mmap でまとめる | 小さい fromfile/seek の繰り返しを削減。最終CSR・寿命・空入力を保持 |
| 5 | scan の同一サイトの状態検証・正規化を一度だけ行う | 固定fit内の不変量。公開APIの検証と変化する割当ての再計算を維持 |
| 6 | 候補分布の表示用小型 Series の生成・sort を削減 | pandas/NumPy の組み込み処理。加算順・同値tie・文字列順を保持 |
| 7 | 候補固定校正ではスコアに必要な集計だけを作る | 現行descriptor準備は既実装と判明。最終出力用の枝長・理由文字列等の生成だけを省き、validation・score・反復診断を維持 |
| 8 | SiteArray の反復読み出しを read-only mmap へ置換 | 小範囲アクセス。thresholdのcopy、ファイル破損検出、worker寿命を保持 |
| 9 | Fisher inclusion DP の除外多項式集約をベクトル化 | 同じFisher分布の正確な対数DP。Walleniusへの置換や近似は禁止 |
| 10 | endpoint pruning の一時配列を out 引数とworkspace再利用で削減 | NumPy ufunc/matmul。逐次演算順・generatorの返却値所有権を保持 |

## 一次資料

- Kobert, Stamatakis & Flouri (2017), [Efficient Detection of Repeating Sites to Accelerate Phylogenetic Likelihood Calculations](https://doi.org/10.1093/sysbio/syw075)。
  木の各部分で繰り返すサイトの検出と再利用を扱う。全tip完全一致による案1はこの考え方の限定的な試作である。
- Rambaut & Lemey (2024), [BEAST: Pattern compression for fun and profit](https://beast.community/2024-07-17_pattern_compression.html)。
  同一モデル・rate における同一サイトの尤度再利用を説明する。
- Ayres et al. (2012), [BEAGLE](https://doi.org/10.1093/sysbio/syr100)。
  尤度計算をハードウェア向けに分離したライブラリ設計の一次論文。
- Ayres et al. (2019), [BEAGLE 3](https://escholarship.org/uc/item/32s6100w)。
  問題サイズ・状態数・機器に応じたバッチ計算、並列化、メモリの評価を扱う。
- Gangavarapu et al. (2026), [BEAGLE 4.1](https://arxiv.org/abs/2606.27607)。
  2026-06-25公開のpreprint。ARM NEON、GPU、状態数・サイト数に伴う性能変化を扱う。
  CSUBSTへの直接統合・GPU性能の検証は今回の試作に含めない。
- [RAxML-NG source](https://github.com/amkozlov/raxml-ng/blob/master/src/CommandLineParser.cpp)。
  パターン圧縮とsite repeatsが別の機能であることを確認した。
- SciPy公式 [expm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html)、
  [eigh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html)、
  [expm_frechet](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm_frechet.html)。
  固有分解、安定な行列指数、Frechet導関数の契約と参照文献。endpointとbridgeは異なる推定量である。
- NumPy公式 [matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)、
  [einsum_path](https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html)、
  [ufunc](https://numpy.org/doc/stable/reference/ufuncs.html)。
  BLAS、縮約順と中間配列、out/whereの挙動を確認した。
- NumPy公式 [memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)、
  [array indexing](https://numpy.org/doc/stable/user/basics.indexing.html)、
  [reading and writing files](https://numpy.org/doc/stable/user/how-to-io.html)。
  mmap・slice view・advanced indexing copyと、一時binary storageの制約を確認した。
- pandas公式 [groupby](https://pandas.pydata.org/docs/user_guide/groupby.html)、
  [iterrows](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iterrows.html)。
  組み込み集約と行単位Python処理の違いを確認した。
- Fog (2024), [Biased Urn Theory](https://cran.r-project.org/web/packages/BiasedUrn/vignettes/UrnTheory.pdf)、
  SciPy公式 [Fisher noncentral hypergeometric](https://docs.scipy.org/doc/scipy/tutorial/stats/discrete_nchypergeom_fisher.html)。
  FisherとWalleniusは異なる分布であり、高速化のための交換はできない。
- Cython公式 [typed memoryviews](https://docs.cython.org/en/latest/src/userguide/memoryviews.html)。
  native化を検討する際のconst入力・nogilの契約。既存native経路の再導入を新案とは数えない。

個々の追加資料、具体的な計測条件、反例・失敗結果は各 `proposalNN.md` / `.json` に記録する。
