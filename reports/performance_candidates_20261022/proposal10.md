# 案10: endpointの行列積を既存bufferへ直接出力

**評価: PEPCの推論単体に限る条件付き候補、優先度は低い。一般的な高速化は未確認。**
`EndpointModel.iter_blocks` の上向き・下向きの行列積を `np.matmul(..., out=post[...])`、
下向きの乗算を `np.multiply(..., out=post[...])` にする最小試作を比較した。
block size、推定量、精度、カテゴリー数、既存workspaceは変えていない。
製品ソースを編集せず、実験プロセス内で現行関数の該当2式だけを置換している。

## 一次資料と仮説

- [NumPy matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) は、
  `out`を省略すると結果配列を新たに確保すること、可能な場合にBLASを使うことを説明する。
- [NumPy ufunc](https://numpy.org/doc/stable/reference/ufuncs.html) は、出力先の指定と、
  重なりのある入力・出力の扱いを説明する。単純なin-place乗算により明示的な積配列を省ける。
  ただし内部のコピーやBLAS workspaceまで消えることを保証するものではない。
- [Gangavarapu et al. (2026), BEAGLE 4.1](https://arxiv.org/abs/2606.27607) は、
  系統尤度計算のハードウェア実装と問題サイズによる性能差を扱うpreprint。
  そのSIMD/GPU倍率を今回のNumPy試作の改善率へ転用しない。

現行はpruning用bufferを既に再利用しているが、上向きの行列積では結果を一度確保してから
bufferへコピーし、下向きでは行列積と要素積の結果をそれぞれ確保している。
そこを直接出力にする。`left`やcallback入力の寿命は変更しない。
式の順序は行列積→要素積のままである。

## 測定条件

基準 `4498c43`、macOS 27.0.1 arm64、Python 3.14.7、NumPy 2.5.3、
SciPy 1.17.1、pandas 3.0.5。NumPy/SciPy import前に
VECLIB/OpenBLAS/OMP/MKL/BLISのthread環境変数を1にした。
共有排他lockで他案のCPU計測と直列化した。

PGK/PEPCの同梱入力・固定IQ-TREE fitを新しい一時ディレクトリへコピーし、
実CLIからjoint推論直前の入力を捕捉した。両版は同じmodel、tip likelihood、
scan classified-event reducer、4率カテゴリー、block size 64を使う。
transition cacheをウォームアップし、各版warmup1回、順序を交互にした3回計測。
生成する全node/event/category配列の走査・checksumを含む推論単体であり、
入力準備、ファイル保存、校正、CLI全体、IQ-TREE再推定を含まない。

| 条件 | wall現行→試作（秒） | wallの現行範囲 / 試作範囲 | 中央値変化 | CPU現行→試作（秒） |
| --- | ---: | --- | ---: | ---: |
| PGK・417サイト | 0.3358 → 0.3369 | 0.3275–0.3400 / 0.3293–0.3837 | +0.3% | 0.3344 → 0.3336 |
| PEPC・971サイト | 1.8125 → 1.6394 | 1.7387–1.9475 / 1.5612–1.6996 | −9.6% | 1.6594 → 1.5809 |
| PEPC×4・3,884サイト | 6.8490 → 6.9030 | 6.8084–7.2129 / 6.7164–7.2635 | +0.8% | 6.6497 → 6.5482 |

PEPCだけでwall範囲が分離し、CPU中央値も4.7%減った。
PGKと人工反復PEPCでは範囲が重なり、wall改善は確認できない。
PEPC×4は独立した生物学的データではなく、同じsiteを4回繰り返すサイズ試験である。
この結果から解析全体や大入力一般の高速化は主張しない。

## 同値性・メモリ

実3条件では全node、全Nイベント、synonymous summary、カテゴリー別node posteriorを
現行と直接照合した。比較要素数はそれぞれ18,968,913、96,269,795、385,079,180、
最大絶対差は全条件で0だった（照合基準 `rtol=1e-12, atol=1e-13`）。

別途21条件で、uniformと不変カテゴリーを含むmixture、ゼロ・短枝、部分曖昧性、
全欠損、read-only/strided入力、選択枝、predictive、空site、返却recordを最後まで保持する場合、
left/rightのviewを返すcallbackを比較した。すべて最大差0。
callbackが既存bufferのviewを返す場合の現行alias挙動も一致し、新しいscratch再利用は導入していない。

既存 `tests/unit/test_endpoint.py` を試作へ差し替えて実行し、**26 passed**。
独立した全祖先状態列挙、短枝・微小遷移の参照を含む。製品ファイルへの差し替えではない。

追加の保持workspaceは0。主要bufferの明示配列容量はPGK 16,373,760 bytes、
PEPC/PEPC×4 35,518,464 bytesで変わらない。
省く明示結果配列の累積割当て量はPGK 156,284,928 bytes、PEPC 796,064,640 bytes、
PEPC×4 3,184,258,560 bytesだが、これは実行中の累積量であり、同時に減るRAM量ではない。
同時に存在する2つの結果配列は最大62,464 bytesにすぎない。
NumPy内部/BLAS workspace、allocator挙動、ピークRSSは測定しておらず、RAM削減率は主張しない。

## 採用条件・残る検証

小さな変更で数値・所有権の比較は通ったため候補には残るが、一般採用の優先度は低い。
PEPCの有利さが別OS/BLAS/Python版と実CLIで再現すること、PGKや長いalignmentを
悪化させないことを確認してから組み込む。
native/fallback両経路、全出力表、実worker、installed wheel、科学的校正は未検証。
今回の26件のsourceテストをそれらの証明とは扱わない。

## 再現

```bash
python reports/performance_candidates_20261022/proposal10.py
python -m ruff check reports/performance_candidates_20261022/proposal10.py
git diff --check
```

上記は成功。[全測定値・環境・入力hash](proposal10.json)／[試作](proposal10.py)。
通常pytest全体は実行していない。製品コード・既定値・参照値は変更していない。
