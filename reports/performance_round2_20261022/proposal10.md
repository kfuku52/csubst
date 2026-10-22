# 案10: 構造重み付けCVのlog-priorとheld-out応答要約の再利用

**条件付き採用候補。** 既定の自動探索gridを使う合成CV負荷で、
全nested crossfit stageが42.0–54.9%短縮した。
実際の外部構造contextやCLI全体は測っていないため、一般の解析速度への効果は未確定。
本体・既定値・推定量・乱数処理は変更していない。

## 変更の範囲

現行`_select`は既にfoldごとのbase priorとinteraction scoreを再利用している。
今回の試作は、その後に残るgrid候補ごとの`log(base)`、maskのコピー、
held-out countのコピー・正値選択・合計を減らす。

私有prior entryは、現行と同じbeta/clip・score検証を通った後にだけ、
保持するbase配列をlog-prior配列へ置き換える。
logweightの計算、最大値によるscale、指数化、正規化、log scoreの加算順は維持する。
held-out応答要約は現在のinner foldの採点にだけ使い、training priorには入れない。
mask・prior・score・応答要約を同じ32 MiB・1,024件のLRU上限へ合算し、
1回の`_select`を越えて共有しない。固定1候補の早期returnは変更しない。

[NumPyの`log`](https://numpy.org/doc/stable/reference/generated/numpy.log.html)の
`out`/`where`を現行と同じ形で使い、masked/zero priorを−∞として保持した。
[NumPyのcopy/view仕様](https://numpy.org/doc/stable/user/basics.copies.html)を踏まえ、
外部可変配列のidentityを鍵にした汎用cacheにはしていない。
[`sum`の仕様](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)では
浮動小数点加算の方法に依存する誤差があるため、形・順序を保ち、実行結果も厳密比較した。
[科学的契約](../../docs/STRUCTURE_WEIGHTING.md)にある外側held-outの独立性と
1e-12のtie規則はそのまま維持する。

## 測定

基準1.16.21 / `88528ce`。macOS arm64、Python 3.14.7、
NumPy 2.5.3、SciPy 1.17.1、pandas 3.0.5、BLAS1。
全案共通の排他lockを取り、入力準備・検証・CPU測定を直列化した。
現行実装の事前測定後、同じ入力・gridでwarmup1組＋順序交互の測定3組。
cache準備を含む**全nested crossfit stage**であり、単一predict kernelだけの時間ではない。

**入力は全て合成データ。** PGK/PEPCに近いbranch/site数を使うが、
Poisson count、独立Gaussian context/features、欠測mask、5つの固定group、
1つのpartitionをseed1210で生成した。実際のPGK/PEPC count、構造、外部contextではない。
beta31 × alpha4 × clip6 = 744候補の既定joint-auto gridと、
独自alphaを選ぶbeta0 comparatorを、外側5foldで全て評価した。
入力配列hashと全反復は[JSON](proposal10.json)に保存する。

秒、中央値（最小–最大）:

| 合成入力 | 現行wall | 試作wall | 現行CPU | 試作CPU |
| --- | ---: | ---: | ---: | ---: |
| 64枝 × 417site | 1.9918 (1.8784–2.1072) | 1.1549 (1.0626–1.7021) | 1.9548 (1.8471–1.9957) | 1.1127 (1.0457–1.3449) |
| 140枝 × 971site | 8.4384 (7.4651–9.3428) | 3.8050 (3.7973–3.9206) | 7.7874 (7.4262–8.3797) | 3.7669 (3.7232–3.8617) |

両負荷でwall/CPUの範囲が分離した。wall中央値の短縮率は42.0% / 54.9%。
各負荷でcache hit61,160回、log-prior変換160回、応答要約構築40回を確認し、
最適化経路が実行されていることも検査した。

保持配列payloadの最大値は930,176 / 4,663,104 bytes。
同じfold分割で現行cacheが保持するbase4種＋score1種から導いた基準値は
867,360 / 4,350,080 bytesなので、追加保持は62,816 / 313,024 bytes。
この導出は全行eligible・1partition・evictionなしという今回の入力条件に限る。
いずれも既存32 MiB上限内で、小容量cacheの追出しも検査した。
Python object/headerと一時作業配列をこれらの保持量に含めない。
process peak RSSは測定しておらず、RAM低減の主張はしない。

## 同等性と検証

- warmupを含む各反復で、予測・null予測・選択beta/alpha/clipの全配列がbytes一致。
  全候補定義・foldごとの選択結果・inner/outer診断も厳密一致。
- 16検査で、ゼロcontextのtie、全ゼロcount、全欠測行、buffer、depth partition、
  read-only入力、全inner score不定、非有限入力、固定1候補、clade不足を比較した。
  finite context/featuresの行列積がoverflowするbeta0条件も同じエラーになり、
  beta0を理由に検証を省略していない。
- outer held-out応答を999へ変更しても、そのfold自身の予測・選択パラメータは不変。
  buffer応答の変更も全予測・パラメータに影響しない。
- cache上限を1 KiBへ下げた場合も結果一致、実際のevictionと上限維持を確認。
  小型softmaxは独立した直接式ともrtol1e-14・atol0で一致した。
- `_select`だけを試作へ差し替えた既存`tests/unit/test_omega_epistasis.py`は
  **22 passed、0 skipped**。公開`predict`は変更していない。
- Ruff・py_compile・JSON読込・パス・差分空白検査が成功。

この証拠はsource試作と合成の条件付きcount予測についてのもの。
実構造入力、PDB処理、外部contextの独立性、CLI全体、installed wheel、
他OS、科学的校正は検証していない。本体未変更のため全suiteは再実行していない。
導入するなら、自動gridが実際のボトルネックとなる構造重み付け解析で追加測定し、
cacheの私有寿命と既存validation順を製品テストで維持する。

## 再現

```bash
python reports/performance_round2_20261022/proposal10.py
python -m ruff check reports/performance_round2_20261022/proposal10.py
```

[試作ハーネス](proposal10.py)、[環境・全反復・hash・検証結果](proposal10.json)。
