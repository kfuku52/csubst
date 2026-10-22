# 案4: endpoint疎行列spoolの断片読み出しをmmapに置換

**評価: 条件付き採用候補。大きいspoolのCPU費用を削減できるが、全経路への既定適用は勧めない。**
PEPCのCSR生成では改善が見られた。PGKのwall timeは改善を示さず、合成サイズ試験の
プロセスピークRSSは約51 MiB増えた。`finish`だけの倍率を解析全体の倍率とは扱わない。

## 根拠と試作

現行の `endpoint_io._Spool` はサイトブロックごとの疎なpayloadを匿名TemporaryFileに追記し、
`finish`で最終CSRのdata/indexを一度確保する。各fragmentについて2回seekし、2回`np.fromfile`
を呼ぶ。高次arityのprojected経路では、PGKで3,584断片、PEPCで15,264断片になった。

[NumPy memmap公式](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)は、
大きいファイルの部分アクセスと、Pythonのmmapをndarrayのbufferにする方法を説明する。
[fromfile公式](https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html)もbinary読取自体は
効率的としているため、今回は個々の読取アルゴリズムでなく呼び出し回数を減らす仮説を評価した。
[SciPy CSR公式](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)に従い、
data・indices・indptrの構造とsort済みindexを維持した。

試作はspoolをflushして2ファイルをread-only mmapし、fragment sliceを最終CSR配列へコピーする。
コピー後にmappingを閉じる。返すCSRは通常の独立したNumPy配列を持ち、TemporaryFileやmappingの
寿命に依存しない。payloadが空ならmappingを作らない。既存のappendとsort_indicesは変更しない。
全ファイルを追加の通常ndarrayへ一括readする方式は使わない。

## 測定方法

基準 `4498c43`、macOS 27.0.1 arm64、Python 3.14.7、NumPy 2.5.3、SciPy 1.17.1、
pandas 3.0.5。NumPy/SciPy import前にVECLIB/OpenBLAS/OMP/MKLのthread環境変数を1にした。
他案のCPUベンチマークとは共有ファイルロックで直列化した。

実CLIでPGK/PEPCの同梱fixed fitを準備し、endpoint `_build` 後に終了した。
`search --max_arity 3 --drop_invariant_tip_sites no --b no --s no --cs no --bs no --cbs no`
を使い、alignment、rooted treeとIQ-TREEのtreefile/state/rate/iqtree/logの5パスを明示する。
seed8、CSUBST/BLAS各1、longtailなし。新しい一時ディレクトリに出力した。
これは既定arity2のpairwise経路とは異なり、実際にspoolを保持する経路である。

実際の各spoolへ渡された同一append block列を捕捉し、次の2範囲を比較した。
各spoolでwarmup1回と交互順の測定3回を行い、各ペアのCSRを直接照合する。
import、入力準備、記録用blockコピー、出力照合の費用は時間に含めない。

1. `finish`だけ：既に作成済みの同じ一時ファイルから最終CSRを作成。
2. `append+finish`：同じblock列を新しいspoolへ追記し最終CSRを作成。書き込み費用も含む。

下表は**各repについて8個のspoolの測定時間を合計した後の中央値**。
連続したCLI全体のwall timeではなく、同パイプラインで使うCSR構築処理の合計である。
file cacheを破棄する測定はしておらず、cold filesystemの主張はしない。

| 条件 | 範囲 | wall秒・現行→mmap | wall変化 | CPU秒・現行→mmap |
| --- | --- | ---: | ---: | ---: |
| PGK | finish | 0.1503 → 0.1712 | +13.9% | 0.1499 → 0.0982 |
| PGK | append+finish | 0.3714 → 0.3629 | −2.3% | 0.3221 → 0.2657 |
| PEPC | finish | 0.7693 → 0.5695 | −26.0% | 0.7225 → 0.4773 |
| PEPC | append+finish | 1.8582 → 1.5292 | −17.7% | 1.5651 → 1.2999 |

PGK append+finishのwall範囲は現行0.329–0.453秒／mmap0.353–0.370秒で重なり、
小さい中央値差を改善の証拠としない。PGKのfinishはwallでは悪化した。
PEPC finishは0.753–0.774秒／0.562–0.601秒で分離している。
PEPC append+finishは1.738–1.864秒／1.489–1.763秒で一部重なり、削減率の再現性は未確定。
CPUは両データの両範囲で減った。PGKのpayloadは合計45.03 MiB、PEPCは197.77 MiBだった。

## メモリの代償

別の新規プロセスを現行/mmap各3回起動し、256行、8,192サイト、20特徴、block64の
合成spoolを作った。各blockの90%程度をゼロにした固定seed入力で、4,227,072 nnz、
payload48.375 MiB、32,768断片。`resource.ru_maxrss`をfinish直後、検証hashのbyteコピー前に記録した。
入力は小さい同じblockを反復し、現実データの多様性を模した試験ではない。

| 合成サイズ試験 | 現行 | mmap |
| --- | ---: | ---: |
| プロセスピークRSS中央値 | 210.45 MiB | 261.09 MiB |
| ピークRSS範囲 | 209.22–212.72 MiB | 260.42–261.58 MiB |
| finish wall中央値 | 0.833秒 | 0.192秒 |

mmapはファイルを追加の通常ndarrayへ一括コピーしないが、touchしたmapped pageもRSSへ計上される。
約50.64 MiBのRSS増はpayload容量と同程度だった。file-backed pageはOS cacheとも関わるため、
この差をシステム全体の追加物理RAMと同一視しない。長いspoolのプロセスRSS上限を維持するには
windowed mmap等を別途測る必要があり、今回の試作でRAM削減は主張しない。

## 正確性・採用条件・未検証

- 全実測ペアのCSR shape/dtype/data/indices/indptr/sort状態が厳密一致。
- float32/float64、int32/int64 index、空payload、branchとsite blockの交錯、
  read-only入力、finish再呼び出しを照合した。spoolを閉じた後も返却CSRが一致した。
- 製品コードは変更していない。既定pairwise arity2やscanのSiteArrayにはこの案の効果はない。
- PEPC程度以上でCSR構築のCPU費用が問題になる条件なら、RSS budgetと併せて導入候補。
  現時点では一律切り替えに必要なサイズ閾値は決められない。
- 最終search表/manifestまで含むCLI比較、Linux/Windows、native/fallback双方の回帰レーン、
  installed wheel、cold filesystem、windowed mmapは未検証。通常のpytest全体は実行していない。
  科学的校正や数値推定器の変更を評価したものではない。

## 再現

```bash
python reports/performance_candidates_20261022/proposal04.py
python -m ruff check reports/performance_candidates_20261022/proposal04.py
git diff --check
```

3コマンドは成功。RSS計測はhash一時配列の費用を除くよう修正後、`--rss-only`で追試した。
[試作スクリプト](proposal04.py)／[全測定値・環境・入力hash](proposal04.json)。
