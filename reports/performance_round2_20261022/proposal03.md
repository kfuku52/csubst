# 案03: posteriorイベントキャッシュに枝別数値ベクトルを併置

**反復校正の内部処理に採用可能な候補。** 既存のイベントキャッシュへ、
posteriorイベントの枝別float64ベクトルを一度だけ作って保存する。
候補固定校正で同じイベントを参照するたびに行っているpandasの一意性検査、
欠損補完、辞書化を省く。前回採用したgroupby省略の先に残った費用を対象とする。
メタデータやtarget/controlマスクを準備する案02とは独立に評価した。

試作は固定候補関数のprivate namespaceだけを差し替える。
公開DataFrame、単回探索、calledイベントは従来の経路を保つ。
重複calledイベントのpandas補償付き集約も変更しない。
ベクトルは書込禁止にし、元DataFrameとともに同じ`_event_cache`の
32 MiB・4,096件の上限で計上する。新しいfitやworker chunkを越えて共有しない。

## 測定

基準は1.16.21 / `88528ce`。Python 3.14.7、macOS arm64、
NumPy 2.5.3、SciPy 1.17.1、pandas 3.0.5。
同梱PGK/PEPCの固定fitを一時ディレクトリへコピーし、
joint endpoint、candidate_fixed、50割当て、seed8、1 process・BLAS1を使用した。
全エージェントで共通の排他ロックを使い、CPU測定を重複させていない。
warmup1回と、順序を交互にした測定3回。
各worker測定は空キャッシュから始め、初回の構築費用も含む。

以下は**候補固定校正worker chunk**のwall time、秒、中央値（最小–最大）。

| 入力 | 現行 | 数値ベクトル | 中央値短縮 |
| --- | ---: | ---: | ---: |
| PGK・12候補・50割当て | 0.13550 (0.12862–0.14039) | 0.10267 (0.10232–0.10394) | 24.2% |
| PEPC・103候補・50割当て | 0.97565 (0.96622–0.98825) | 0.70128 (0.69007–0.72701) | 28.1% |

両方とも測定範囲が分離した。CPU中央値もPGK 0.13531→0.10258秒、
PEPC 0.97548→0.70079秒へ減った。推論、CLI起動、出力書込の時間は含まない。

同じ条件を独立Python subprocessの**全CLI**でも比較した。
importsと両経路に共通のwrapper起動費用を含む。

| 入力 | 現行 | 数値ベクトル |
| --- | ---: | ---: |
| PGK・candidate_fixed50 | 2.202 (2.124–2.408) | 2.183 (1.907–2.403) |
| PEPC・candidate_fixed50 | 6.911 (6.620–7.879) | 6.603 (6.100–7.357) |

全CLIは両方とも範囲が重なるため、**全体の速度向上は確定しない**。
CLIのpeak RSS中央値はPGK 331.4→326.8 MB、PEPC 633.4→652.3 MBで、
こちらも範囲が重なる。RSSの改善・悪化を断定しない。
[全反復JSON](proposal03.json)の`cli_cases`にwall・子CPU・peak RSSを記録した。
局所費用の削減として有望だが、安定した全体効果を採用条件にする場合は保留となる。

## 同等性とメモリ

- 各入力の50試行について、全スコア・キー・診断が厳密一致。全試行成功、失敗0。
- warmupを含む16 CLI実行・8比較組で、scan/units表の全列・型・行順を
  `check_exact=True`で照合し、校正・推論JSONも全内容が一致した。
- 42件の境界検査で、枝順・欠番・範囲外ID・NaN・空イベント・負値・無限大、
  重複行の補償付き集約、全4 exposure方式、full/score-only、endpointの
  長さ・枝順・contextエラー、cache再利用/無効化・上限追出しを確認した。
- worker unpackが既存の`_event_cache`を落とすことも実行して確認した。
- 追加の数値配列はPGK 6,240 B、PEPC 116,184 B。
  配列ヘッダ・recordの見積りを含む追加量は各8,160 B / 132,664 B。
  キャッシュ計上合計は25,184 B / 435,060 Bで、既存上限内。
  これは保持データの計上値であり、プロセスRSSの低減を意味しない。

巨大な木では枝数に比例するベクトルが疎なDataFrameより高価になる。
同じバイト上限に合算しても保持可能な候補数は減るため、疎な巨大入力や
キャッシュが頻繁に追い出される条件には追加評価が必要。
公開DataFrameへのattrs埋込や外部可変入力のidentityだけに頼るキャッシュは使わず、
固定fitのprivate entry内に限定して導入する。

## 再現と範囲

```bash
python reports/performance_round2_20261022/proposal03.py
python -m ruff check reports/performance_round2_20261022/proposal03.py
```

[試作ハーネス](proposal03.py)、[全反復・hash・境界検査](proposal03.json)。
本体コードは変更していないため、本体テストsuite・wheel・他OS・実IQ-TREE再推定は
今回の試作証拠に含めない。導入時にはworker経路を含む製品テストが必要。

NumPyの[書込フラグ](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.setflags.html)
と[コピー・ビューの仕様](https://numpy.org/doc/stable/user/basics.copies.html)を確認し、
内部保持配列をread-onlyにし、整数配列による枝選択は独立した作業配列として扱った。
これは実装方法の根拠であり、速度は上記のローカル測定で判断した。
