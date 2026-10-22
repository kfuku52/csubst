# 案9: Fisher inclusion の対数DP集約をサイト方向へベクトル化

**評価: 任意のFisher urn経路に対する優先採用候補。**
分布・近似・乱数を変えず、Pythonのサイト×drawの二重ループを、draw順のループと
NumPyのサイト方向の演算へ置き換える。現環境で全条件の旧新出力のbitwise一致を追検証した。
通常既定の`codon_model`経路の高速化ではなく、CLI全体の倍率も未測定である。

## 文献と対象

- Fog, [Biased Urn Theory](https://cran.r-universe.dev/BiasedUrn/doc/UrnTheory.pdf)
  （取得版2026-07-22）はFisherとWalleniusを異なる条件付き分布として定義する。
  今回はFisherの積重み付き部分集合を維持し、別分布への交換は行わない。
- [SciPy Fisher noncentral hypergeometric](https://docs.scipy.org/doc/scipy/tutorial/stats/discrete_nchypergeom_fisher.html)
  はFisher分布の正規化和とFogの資料を示す。これは2色の説明であり、
  本試作の多色性能を直接保証する資料ではない。
- [NumPy logaddexp](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html)
  は小確率を対数で加えるufuncと`out`を説明する。

対象は`omega._calc_fisher_inclusion_probabilities`の除外多項式の集約だけ。
prefix/suffixの対数DP、入力検証、drawの境界処理、加算するleft_pickの順序、
float32/float64への最終cast、clipは現行と同じにする。
Pythonスカラーの`logaddexp`呼び出しを、各サイトを独立に扱うベクトル呼び出しへ変える。
`make_prototype`は現行関数の対象部分だけをメモリ内で差し替え、製品ファイルを編集しない。

## 方法と結果

基準コミット`4498c43`、macOS 27.0.1 arm64、Python 3.14.7、
NumPy 2.5.3、SciPy 1.17.1、pandas 3.0.5。
NumPy/SciPy import前にBLAS/OpenMP関連環境変数を1へ設定し、
共通ファイルロック内で他案のCPU計測と直列化した。
同じ重み・draw・出力型でwarmup1回、交互順の計測3回。
入力作成、import、出力照合は計測から除く。

417/971はPGK/PEPCのサイト数に合わせた**合成重み**であり、実際の解析から抽出した
ASRV重みではない。uniformと、対数一様な偏り＋ゼロを含む重みを評価した。
全13条件の現行・試作のwall最小–最大範囲が分離した。
以下は関数単体のwall中央値（ミリ秒）。全反復・CPU時間はJSONに保存する。

| サイト数・重み | draw | 現行 → 試作 ms | 倍率 |
| --- | ---: | ---: | ---: |
| 417 uniform | 1 | 1.529 → 1.140 | 1.34× |
| 417 uniform | 10 | 4.539 → 1.342 | 3.38× |
| 417 uniform | 100 | 30.174 → 2.346 | 12.86× |
| 417 skew＋zero | 100 | 25.448 → 1.937 | 13.14× |
| 971 uniform | 1 | 3.550 → 2.663 | 1.33× |
| 971 uniform | 10 | 9.681 → 2.906 | 3.33× |
| 971 uniform | 100 | 70.897 → 5.501 | 12.89× |
| 971 skew＋zero | 100 | 60.165 → 4.713 | 12.77× |
| 4,096 skew＋zero | 128 | 344.214 → 24.585 | 14.00× |

重みとdrawが同じ場合の全ベクトルの要素値を各計測ペアで直接比較した。
全13条件で、追検証によるdtype・shape・bytesの一致も確認した。
サイト数・drawに依存する効果であり、1.3～14倍を解析全体へ一般化しない。
親レビュー後に`assert_array_equal`だけでは符号付きゼロ等のbitまで検査できない点を修正し、
旧新のdtype・shape・`tobytes()`も直接比較するassertを追加した。
428小標本比較、13計測条件、読取専用入力をlock下で正確性のみ再実行し、すべて成功した。
この追検証で上記のベンチマーク計時は再実行・上書きしていない。

## 正確性とメモリ

- 小標本の全組合せを`fractions.Fraction`で列挙した独立参照と比較した。
  空、ゼロ重み、全取得、draw超過、同一重み、偏った重み、
  `1e-100`から`1e100`の重み、float32/64を含む428比較で旧新が厳密一致。
  独立参照にはfloat64で`rtol=3e-13, atol=1e-14`、float32でその丸めに合う基準を用い、
  全て成功した。独立参照との差の最大値（float32を含む）は`2.97e-8`。
- 読取専用入力を保護し、負重み・負drawの例外文言も現行と一致した。
- 大きい条件のinclusion和とdrawの差は、旧新同じで最大絶対値`4.95e-12`だった。
  この丸め誤差を補正する変更は行わない。
- prefix/suffixは両版とも`2*(n_positive+1)*(draw+1)*8` bytes。
  例えば971サイト・draw100のuniform条件では1,570,752 bytes。
  新規の明示的scratchはfloat64ベクトル1本、`n_positive*8` bytesであり、
  971サイトなら7,768 bytes、4,096条件では29,784 bytesを追加する。
  inclusion自体は現行にもある同じ長さのベクトルを再利用する。
- prefix/suffix列の加算は`np.add(..., out=workspace)`、logaddexp、最終の加算・減算・expも
  `out=inclusion`を使うため、これらに別のサイト長一時配列を明示生成しない。
  元からあるclip出力、NumPy内部の演算buffer、Pythonオブジェクトまで含む
  ピークRSSの追加上限を測定したものではない。RAM削減は主張しない。

## 採用条件と検証

変更範囲が小さく、同じ対数DP・加算順のまま明確に速くなるため、Fisher経路に限定して
実装へ進める価値がある。drawの小さい条件でもこの環境では改善した。
通常の期待値計算法、Wallenius設定、サンプリングRNGは変更しない。
Fisherを利用する実urn解析全体、native/fallback全体、他のNumPy/Python/OS、
installed wheel、科学的較正は未検証。製品化時には関連数値回帰とその実入力での
end-to-end確認を行う。今回の結果はurnモデルの科学的妥当性を認定するものではない。

```bash
python reports/performance_candidates_20261022/proposal09.py
python -m ruff check reports/performance_candidates_20261022/proposal09.py
python -c 'import sys; sys.path.insert(0, "reports/performance_candidates_20261022"); from proposal09 import make_prototype; from bench_common import benchmark_lock; from csubst import omega; import pytest; omega._calc_fisher_inclusion_probabilities = make_prototype(); lock = benchmark_lock(); lock.__enter__(); result = pytest.main(["-q", "tests/unit/test_omega_sampling.py", "-k", "fisher or draw_one"]); lock.__exit__(None, None, None); raise SystemExit(result)'
git diff --check
```

上記の実験、Ruff、diff-checkは成功。既存Fisher関連テストをメモリ内の試作へ差し替えた実行は
**2 passed、17 deselected、0.14秒、exit 0**。通常のソース関数だけを検査した結果ではない。
全体pytestは実行していない。製品コード、参照値、科学的既定値は変更していない。

[再現スクリプト](proposal09.py)／[全測定値・環境・直接比較結果](proposal09.json)。
