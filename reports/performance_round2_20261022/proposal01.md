# 案1: one-hot葉尤度の遷移行列参照

**判定: 現試作は採用見送り。** 正しさは検証できたが、実入力で安定した短縮を示せない。
製品コード・既定値は変更していない。

初回試作には最終の`source.replace`がなく、コード作成の準備だけが追加され、
肝心のgatherが実行されていなかった。初回の計測と採用判定を無効とし、
修正後に全検証・計測を取り直した。以下と保存JSONには修正版の結果だけを掲載する。

one-hot観測 `e_j` の枝メッセージは `e_j @ P.T == P[:, j]` なので、
葉の全サイトがone-hotであるblockだけ行列積を列参照へ置き換えた。
曖昧・欠損・値が1以外のsingletonを含むblockには元の行列積を使う。
元の入力検証を全て終えてからsiteコード・適用可能マスクを準備する。
全体サイトパターンの圧縮、推定量の変更、乱数処理の変更は含まない。

葉状態の検索表は既存の系統尤度最適化でも使われる
（[Kobertほかの原著](https://academic.oup.com/sysbio/article/66/2/205/2420631)）。
ここで評価したのは、同論文のsubtree repeat法とは独立の単純なone-hot恒等式である。
NumPyのadvanced indexingはcopyを作るため、演算数の削減がそのまま高速化になるとは限らない
（[NumPy公式資料](https://numpy.org/doc/stable/user/how-to-index.html)）。

## 測定

基準は `88528ce` / 1.16.21。macOS arm64、Python 3.14.7、NumPy 2.5.3、
SciPy 1.17.1、pandas 3.0.5。BLAS環境変数はimport前に1へ設定した。
共有lockを取得して実入力の準備・検証・計測を行い、他案のCPU計測との重複を防いだ。
同梱の固定fit PGK/PEPCを一時ディレクトリへcopyし、通常joint scanからモデル・tip・transformを捕捉した。
両実装ともtransition cacheをwarmupし、block 64で1 warmup + 5反復、順序を交互にした。
コードの準備、尤度検証、siteコード作成は計測対象に含む。

以下は **`EndpointModel.iter_blocks` stageのみ** の秒、中央値（最小–最大）。
CLI全体やモデルfitの時間ではない。

| 入力 | 基準 wall | 試作 wall | 基準 CPU | 試作 CPU |
| --- | ---: | ---: | ---: | ---: |
| PGK、65 nodes、417 sites、61 states、4 categories | 0.3791 (0.3583–0.5481) | 0.4351 (0.3690–0.5555) | 0.3262 (0.3246–0.3322) | 0.3214 (0.3119–0.4036) |
| PEPC、141 nodes、971 sites、61 states、4 categories | 1.7579 (1.6242–2.0151) | 1.6848 (1.6312–2.0543) | 1.6633 (1.6128–1.8819) | 1.6468 (1.5974–1.8308) |

PGKのwall中央値は14.8%大きく、PEPCは4.2%小さいが、両方で範囲が大きく重なる。
CPU中央値の差はPGK約1.5%、PEPC約1.0%の減少と小さく、範囲も重なる。
したがって実入力の改善・悪化を断定しない。

PGKは231葉block中229、PEPCは1,136中667が適用可能だった。
追加保持するcodes/maskはPGK 123,849 bytes、PEPC 620,469 bytes。
参照結果の一時配列上限は31,232 bytesで、元のmatmul結果一時配列を置き換える。
これらは配列サイズであり、process peak RSSではない。RSS低減の主張はしない。

## 同等性と限界

- 計測外の1passをtraceし、gather代入行がPGK 916回、PEPC 2,668回実行されたことを確認。
  適用可能leaf block数×4カテゴリという独立した期待回数と一致する。
  ソース文字列の存在だけを、最適化の実行証拠として扱っていない。

- 実入力の全recordでnode、category node、全reduced出力を厳密配列一致とbytes一致で検査。
  PGK 18,968,913値、PEPC 96,269,795値が一致した。
- 合成24条件で、block 1/32/64/200、joint/predictive、選択枝と非選択枝transform、
  rate 0を含む4カテゴリ、ゼロ長枝、曖昧・欠損・非unit singleton・空サイトを検査。
  全595,264値がbytes一致した。
- ゼロ尤度、不正block size、負値、NaN、葉不足の5条件で同じValueError文字列を確認。
- 通常scanのnative reducerを含むstageを評価した。source-only reducer、全CLI、
  installed wheel、Linux/Intel、独立した科学的校正の検証は行っていない。

適用可能な葉blockの割合が高いPGKでも費用差が安定しないため、
このPython/NumPy試作をそのまま導入する根拠はない。
より低費用のnative leaf kernelを検討する場合は別案として再測定が必要。

## 再現

```bash
python reports/performance_round2_20261022/proposal01.py
python -m ruff check reports/performance_round2_20261022/proposal01.py
```

[試作・検証コード](proposal01.py)、[全反復・環境・入力hash・検証記録](proposal01.json)。
生成入力・全実行ディレクトリは保存対象外。
