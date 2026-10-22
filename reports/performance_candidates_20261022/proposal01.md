# 案1: 完全一致サイトパターン圧縮

**評価: 高反復データに限る条件付き採用候補。通常のPGK/PEPCへ既定適用する案は見送り。**
全tipの観測尤度が完全に一致するサイトだけを一度推定し、node posteriorとNイベントを
元のsite順へ展開する。独立サイト・共通Q・共通のrate-mixtureという現行EndpointModelの
条件で成立する。モデルや推定量を変更する近似は使わない。

## 文献とコード上の根拠

- [BEAST公式のpattern compression解説](https://beast.community/2024-07-17_pattern_compression.html)
  は、同一モデル/rateの完全一致サイトで尤度計算を共有できることを説明する。
  同ページ後半の曖昧な定常サイトをまとめる近似方式は今回使わない。
- [Kobert, Stamatakis & Flouri (2017)](https://doi.org/10.1093/sysbio/syw075)
  は、木の内部でも繰り返すサイトの検出と尤度ベクトル再利用を扱う一次論文。
  今回は実装負担の小さい全tip一致だけを対象とし、部分木単位の圧縮は未実装。
- [Ayres et al. (2019), BEAGLE 3](https://doi.org/10.1093/sysbio/syz020)
  はsite-pattern数やブロック分割が計算性能に影響することを示す。
  同論文のGPU/CPU高速化率はCSUBSTの改善率として流用していない。

対象は `csubst/endpoint.py:EndpointModel.iter_blocks` と、そのtip likelihoodを準備する
`csubst/endpoint_io.py`。現在は各サイトを個別に処理する。
本試作は全tipのfloat64 emissionをbyte単位で完全一致比較するため、部分的な曖昧さや
非二値の尤度も区別する。site固有のeligibilityは**展開後**に適用する。

## 測定

基準コミット `4498c43`、macOS 27.0.1 arm64、Python 3.14.7、NumPy 2.5.3、
SciPy 1.17.1、pandas 3.0.5。同一native event projectorを両実装で使用した。
NumPy/SciPy import前にVECLIB、OpenBLAS、OMP、MKLのthread環境変数を1に設定。
他案のベンチマークとは共有ファイルロックで直列化した。

固定の同梱fitを実CLIで読み込んだ後、endpoint inference直前の入力を捕捉した。
入力準備・import・出力比較を計測から除外し、両版とも新規EndpointModel構築、block size 64で
全node posteriorと全Nイベントの生成・保存、site順展開、eligibility適用を計測する。
圧縮版にはパターン検出・圧縮tip生成の時間も含める。
これはendpoint処理単体の比較であり、search/scan CLI全体の速度ではない。
1回のwarmup後、順序を交互にして3回測定した。

| 条件 | 全site / unique | 圧縮できる割合 | wall秒・現行→圧縮 | wall変化 | CPU秒・現行→圧縮 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PGK | 417 / 405 | 2.88% | 0.363 → 0.403 | +11.1% | 0.360 → 0.398 |
| PEPC | 971 / 966 | 0.51% | 1.880 → 2.090 | +11.1% | 1.855 → 2.046 |
| PEPCを4回反復 | 3,884 / 966 | 75.13% | 6.660 → 3.247 | −51.3% | 6.478 → 2.954 |

wallの最小–最大はPGKが現行0.359–0.404／圧縮0.390–0.467秒、
PEPCが1.791–2.842／1.957–2.276秒、反復PEPCが6.572–7.134／2.936–3.432秒。
実2データの範囲は重なるが、CPU中央値も約10%増えており、有効性を示す証拠はない。
反復PEPCの範囲は分離し、同じ入力・同じ出力で明確な改善がある。
人工反復は独立した生物学的データではなく、反復率の高い条件を切り分けたサイズ試験。

## 正確性とメモリ

全測定ペアで全node posterior・全Nイベントを `rtol=1e-12, atol=1e-14` で直接照合し、
最大絶対差は全条件で0だった。4率カテゴリー（ゼロ率を含む）、部分曖昧性、全欠損サイト、
非二値尤度、ゼロ長枝、並べ替えた反復サイト、反復間で異なるeligibilityを含む小例でも
node・event・カテゴリー別posteriorの最大絶対差が0だった。

返す配列のペイロードはPGK 152,872,200 bytes、PEPC 772,178,040 bytes、
反復PEPC 3,088,712,160 bytesで両版同じ。圧縮キー用stack、圧縮tip、indexについての
保守的な配列容量上限は順に13,437,312／67,301,912／269,184,464 bytes。
JSONの `pattern_temporary_upper_bytes` はこの明示的な配列だけの上限であり、
NumPy内部のsort workspace、Pythonオブジェクト、展開時の一時配列を含むピークRSSではない。
ピークRSSは未測定。圧縮版では圧縮出力と展開出力が一時的に共存するため、
この試作をRAM削減策としては評価しない。

## 採用条件・残る検証

- 高い反復率があるデータを選ぶ任意経路、または検出コストを含む利益判定を持つ経路なら候補。
  本結果から一般的な採用閾値は確定できない。
- 製品化時には元site順へ小ブロックで展開し、全圧縮出力と全展開出力の同時保持を避ける設計が必要。
- S、predictive endpoint、全出力統計、site filter後の無効化、3Di、実worker、CLIの最終表と
  manifest、native/fallback、Windows/Linuxについては未検証。site依存Q/rateを将来扱う場合は
  それらも圧縮キーに含める必要がある。
- 通常PGK/PEPCを高速化する優先度は低い。部分木内のrepeatsは別案として追加評価が必要。

## 再現と検証結果

```bash
python reports/performance_candidates_20261022/proposal01.py
python -m ruff check reports/performance_candidates_20261022/proposal01.py
git diff --check
```

[スクリプト](proposal01.py)／[全測定値・入力ハッシュ・環境](proposal01.json)。
上記3コマンドは成功。製品コードは変更していないため通常のpytest全体は実行していない。
source実験であり、installed wheel検証や科学的校正の証明ではない。
