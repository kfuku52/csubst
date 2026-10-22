# 案07: synonymous変換の固定遷移積を推論内で再利用

**判定: 現試作の採用は見送り。** 正しさは確認したが、実入力で安定した短縮を示せず、
追加保持配列も必要となった。製品コードは変更していない。

`scan_ctmc._joint_infer`のtransformは各block・枝・rate categoryごとに
`transition * synonymous_mask`を作る。同じtransitionに対する要素積を推論1回だけの
LRUに保存し、後続blockでは元と同じ行列積・要素積・sumへ渡す試作を比較した。
posterior、rate混合、event native reducer、加算順、block既定値は変えていない。

## 計測

基準 `88528ce` / 1.16.21。macOS arm64、Python 3.14.7、NumPy 2.5.3、
SciPy 1.17.1、pandas 3.0.5。import前にBLAS環境変数を1へ設定した。
同梱PGK/PEPCの固定fitを一時ディレクトリへコピーし、実scanから捕捉した引数を使った。
共同の排他lock内で準備・検証・計測し、warmup1組＋順序交互の測定3組。
各呼出しは空のtransition/product cacheから始まり、モデル準備とdisk出力を含む。

以下は **`_joint_infer` stageのみ**、秒、中央値（最小–最大）。

| 入力 | 現行 wall | 試作 wall | 現行 CPU | 試作 CPU |
| --- | ---: | ---: | ---: | ---: |
| PGK、65 nodes・417 sites・61 states・4 categories | 0.5186 (0.4788–0.5338) | 0.4729 (0.4675–0.7849) | 0.4748 (0.4673–0.4990) | 0.4603 (0.4453–0.5121) |
| PEPC、141 nodes・971 sites・61 states・4 categories | 3.0617 (2.9302–3.1927) | 3.5505 (2.9446–4.0257) | 2.6972 (2.5967–2.7286) | 2.8699 (2.5290–2.9997) |

PGK wall中央値は8.8%小さく、PEPCは16.0%大きいが、いずれも範囲が重なる。
CPUの範囲も重なるため、速度改善・悪化を断定しない。CLI全体は測っていない。
実際のcache hitはPGK 1,536回、PEPC 11,760回で、未実行の最適化を比較したものではない。
missは256/560回。準備費用を含めても安定した利点を示すには至らなかった。

## メモリと同等性

transition配列を強参照で保持してPython object IDの再利用を防ぎ、
元transitionと積の両配列を32 MiB上限へ計上する。上限超過時にはLRUで追い出す。
推論終了時にcacheを破棄し、summariesには数値の統計だけを残す。
追加の積配列はPGK **7,620,608 bytes**、PEPC **16,670,080 bytes**。
元transitionは既存model cacheと共有するが、LRUが延長する寿命も考慮し、
cache会計では両方を合計15,241,216 / 33,340,160 bytesとして数えた。
Python containerの費用やprocess peak RSSをこの数値に含めず、RSS低減は主張しない。

- 実入力のposterior、全event tensor、category posterior、synonymous counts、入力summary配列を
  bytes一致で照合。PGK 19,139,916値、PEPC 96,663,028値が一致した。
- dense/disk × normal・missing・ambiguity・zero Q・summaryなし・eligibility maskの12条件で
  全出力がbytes一致。0を含む混合rate categoryを使った。
- 不正tip（負値/NaN）、Q、pi、block sizeの5条件で例外型と文言が一致した。
- 小容量cacheで同一identityのhitとbyte上限による追出しを実行検査した。

maskとtransitionは推論中に変更されないprivate contextに限定する。
可変配列を受ける汎用の永続cacheとしては使えない。
source-only native fallback、installed wheel、他OS、全CLI、独立した科学的校正は未検証。
本体未変更のため製品の全suiteは再実行していない。

## 根拠・再現

[NumPy `multiply`](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html)は
`out`指定がなければ新しい配列を返す。ここでは同じ固定積をblockごとに再構築する費用を対象にした。
[Python `OrderedDict`](https://docs.python.org/3/library/collections.html#collections.OrderedDict)を
LRUの順序管理に使用した。APIの性質は設計根拠であり、速度は上記の実測で判断した。
[joint endpoint契約](../../docs/ENDPOINT_POSTERIORS.md)と
[scan CTMC契約](../../docs/SCAN_CTMC.md)の推定量は維持する。

```bash
python reports/performance_round2_20261022/proposal07.py
python -m ruff check reports/performance_round2_20261022/proposal07.py
```

[試作コード](proposal07.py)、[全反復・入力hash・cache統計・検証結果](proposal07.json)。
初回の合成準備ではツリーfactory名を誤り、計測開始前に停止した。
repositoryの`PhyloNode`へ修正後、境界検査と実入力計測を最初から実行した。
保存JSONは修正後の成功実行のみを含む。
