# 案8: SiteArrayの反復読取りにread-only mmapを使う

**評価: 既定導入は見送り。** 現行の返却値の所有権を維持した試作では、
実PGK/PEPCのscan本体のwall timeを改善できず、順次アクセスのピークRSSも増えた。
製品コードは変更していない。

## 文献・適用範囲

[NumPy memmap公式文書](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
は、大きいファイルの一部へアクセスできる一方、派生viewとmappingの寿命管理に注意が必要とする。
[NumPy indexing公式文書](https://numpy.org/doc/stable/user/basics.indexing.html)では基本sliceはviewで、
advanced indexingはcopyになる。[Python mmap公式文書](https://docs.python.org/3/library/mmap.html)
はread-only mappingとファイル記述子・mappingの関係を説明する。
これらはCSUBSTにおける速度改善を保証するものではない。

現行 `SiteArray.read_sites` は独立したwritable配列を返し、`read_site` は
そのコピーをread-onlyの単一サイトcacheへ置く。単純にmmap viewへ変えると、
scratch fileを同じサイズのまま書き換えただけで**以前に返した値も変化する**。
この反例を確認したため、公開APIへzero-copy viewを返す案は採用しなかった。

主試作は `read_site` のcache missだけを `mmap → copy` に置換する。
公開 `read_sites`・連続block読取り・書込みは現行のまま。
site範囲、seal、必要部分のファイル長を確認し、thresholdはコピーへ適用する。
pickleにはmappingを入れず、borrowerは初回読取りで再mappingする。

## 測定方法

基準 `4498c43`。macOS 27.0.1 arm64、Python 3.14.7、NumPy 2.5.3、
SciPy 1.17.1、pandas 3.0.5。NumPy import前のVECLIB/OpenBLAS/OMP/MKL/BLIS環境変数を1にした。
共有ファイルロックで他案のベンチマークと直列化した。

PGK/PEPCは同梱の固定fitを新しい一時ディレクトリへコピーし、実scanの入力準備を通した。
準備済みscan coreを実行して実際の `read_site` 呼出し順を捕捉する。
PGKは24回、PEPCは206回のread呼出しがあり、イベントとカテゴリー別状態の2ファイルを読む。
両版で同じファイル・同じ順序を使い、OS page cacheはwarm、sampleごとmappingと単一site cacheを
破棄して初回mapping費用も含めた。warmup 1回、順序を交互にした計測3回。
入力推定・ファイル作成はreal scan coreの時間には含めず、別のライフサイクル測定で評価した。
CLI全体の計測ではない。

## 結果

秒、3回の中央値（最小–最大）。

| 条件 | 現行 | mmap→copy | 中央値の変化 |
| --- | ---: | ---: | ---: |
| PGK・実read順の再生 | 0.00125 (0.00101–0.00134) | 0.01553 (0.01551–0.02504) | 悪化 |
| PGK・準備済みscan core | 0.09967 (0.09955–0.10248) | 0.11385 (0.11371–0.11401) | +14.2% |
| PEPC・実read順の再生 | 0.01631 (0.01563–0.01767) | 0.01493 (0.01145–0.02151) | −8.4%、範囲重複 |
| PEPC・準備済みscan core | 0.43720 (0.40552–0.48781) | 0.47421 (0.41198–0.50522) | +8.5%、範囲重複 |
| PEPC形状・作成/書込/全site読取/削除 | 0.19399 (0.17995–0.25733) | 0.23531 (0.16801–0.27982) | +21.3%、範囲重複 |

最後の条件は `(141, 971, 1, 20, 20)` の合成配列で、入力blockの生成を計測外、
ファイルの作成・全block書込・seal・全971サイト読取・unmap・削除を計測内とした。
保存形式・配列の大きさは同一。CPU中央値は0.1630→0.1453秒へ減ったが、wall timeの改善証拠はない。
PGKは小さい読取り量に対しmapping初期化の待ち時間が勝った。
PEPCのread単体の差も範囲が重なり、実scan全体での改善を示していない。

## メモリ

PEPC形状の438,115,200 bytesのファイルを別processで最初から最後まで読む合成試験。
子processにも同じimport、入力、checksum集計を使い、warmup後3回のピークRSSを測定した。

| 条件 | 現行 | mmap→copy |
| --- | ---: | ---: |
| peak RSS中央値（MiB） | 162.80 | 581.80 |
| peak RSS最小–最大（MiB） | 161.56–166.30 | 579.19–584.00 |
| wall中央値（秒） | 0.06831 | 0.07005 |

全checksumは一致した。mappingのページを順次touchするとRSSへ計上され、約419 MiB増えた。
これはprocessのpeak RSSであり、OSのfile cacheと共有するページを含む。
全workerの物理RAMが同じ量だけ増えたという測定ではない。

## 正確性・採用判断

- 実scan表とunit表は各ペアで全列・順序・型を厳密一致で照合した。
- float32/64、threshold 0/.3/.9、同一siteのcache identity、順序の変化について
  36回の全値比較に成功した。
- public blockはwritableで独立し、その変更がcache/fileへ伝播しないことを確認した。
- pickleにmappingと大きいpayloadが含まれないこと、borrowerの値が一致することを確認した。
- unsealed、範囲外、逐次的なtruncationに対する拒否、既に返したsnapshotがfile変更後も
  保たれることを確認した。外部processによる同時truncationとの競合安全性は未検証である。
- 試作を適用した既存 `tests/unit/test_site_storage.py` は **10 passed、skipなし**。

今回の証拠では既定導入の利点はない。特にzero-copyを公開返却値へ広げて得た速度を、
同じ契約での高速化と扱うことはできない。
再検討するなら、返却値を保持しない内部consumer専用の短命viewとmappingの容量上限が必要になる。
その設計・有効性は今回の試作では検証していない。

候補固定校正の長い反復、全full-scan、実spawn workerへのpickle輸送、Windows/Linux、
installed wheel、全科学的parity、CLI全体は未検証。保存block経路は変更対象から外した。

## 再現

```bash
python reports/performance_candidates_20261022/proposal08.py
python reports/performance_candidates_20261022/proposal08.py --check-existing-tests
python -m ruff check reports/performance_candidates_20261022/proposal08.py
git diff --check
```

[全測定値・環境・入力ハッシュ](proposal08.json)／[試作スクリプト](proposal08.py)。
既存テスト・Ruff・diff-checkは成功。製品コードを変更していないため、リポジトリ全pytestは実行していない。
