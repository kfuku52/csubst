# 案6: 候補分布の小型Series生成を省く

**評価: 採用を勧める候補。候補生成単体の改善は明確で、変更範囲も小さい。**
現行の集約・加算順はそのままにし、各from/to状態分布の表示文字列を作る際の
`pd.Series(...).sort_values(ascending=False)` をNumPy配列のソートへ置き換える。
既に導入済みのgroupby形状別共有や一括分布集約を再提案したものではない。

## 一次資料と正確性の注意

- [pandas Series.sort_values](https://pandas.pydata.org/docs/reference/api/pandas.Series.sort_values.html)
  は既定ソートがquicksortであることを示す。
- [pandas 3.0.5のsorting実装](https://github.com/pandas-dev/pandas/blob/v3.0.5/pandas/core/sorting.py)
  と実行環境のソースを確認した。降順は入力を反転してからascending argsortし、結果を再反転する。
  単に `argsort(-mass)` へ変えると同量の要素の順が一致する保証がない。
- [NumPy argsort](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html)
  もquicksortの同値要素の順序を保証しないと説明する。試作はpandasと同じ入力反転と
  `kind='quicksort'` を用い、NaNを元の順で末尾へ置く。
- [pandas groupby](https://pandas.pydata.org/docs/user_guide/groupby.html)
  の組み込み集約は維持した。統計量の加算、candidateの選別、閾値、校正は変更しない。

試作の `make_prototype` は現行 `build_candidates` の該当表示式が1か所だけであることを
確認し、その式だけを置換する。製品ファイルの書き換えは行わない。

## 条件と計測結果

基準 `4498c43`、macOS 27.0.1 arm64、Python 3.14.7、NumPy 2.5.3、
SciPy 1.17.1、pandas 3.0.5。同梱PGK/PEPCの入力・既存IQ-TREE fitを新しい
一時ディレクトリへコピーし、通常のjoint scan入力準備を通した。
IQ-TREEの再推定・ネットワーク・描画は行わない。
NumPy/SciPy import前にVECLIB/OpenBLAS/OMP/MKLのスレッド環境変数を1に設定し、
他案のCPU測定とは共通lockで直列化した。
各条件でwarmup1回、現行/試作の順を交互にして3回計測した。

候補生成は同じ準備済みcalled eventsを入力する。以下はwall timeの中央値。
input準備・import・結果比較は含まない。

| 入力・match | event行 / 候補行 | 現行→試作（秒） | 短縮率 | CPU現行→試作（秒） |
| --- | ---: | ---: | ---: | ---: |
| PGK any2spe | 229 / 198 | 0.0330 → 0.0202 | 38.8% | 0.0330 → 0.0201 |
| PGK all | 229 / 825 | 0.1220 → 0.0713 | 41.5% | 0.1213 → 0.0712 |
| PEPC any2spe | 719 / 546 | 0.0904 → 0.0569 | 37.0% | 0.0902 → 0.0561 |
| PEPC all | 719 / 2,336 | 0.3242 → 0.1881 | 42.0% | 0.3213 → 0.1880 |
| 合成6,000イベント all | 6,000 / 17,654 | 2.5072 → 1.7210 | 31.4% | 2.3301 → 1.4483 |

全5条件でwall timeの現行/試作の最小–最大範囲は分離した。
合成入力はランダムな状態対・重複index・離散的なfractional質量を含むサイズ試験であり、
独立した生物学的な検証データではない。

同じ準備済み入力による既定any2speのscan coreでも比較した。
候補生成・支持判定・rate集計・表作成を含み、ASR、入力準備、保存、校正反復は含まない。

| scan core | wall現行→試作（秒） | 中央値短縮 | 現行の範囲 / 試作の範囲（秒） | CPU現行→試作（秒） |
| --- | ---: | ---: | --- | ---: |
| PGK | 0.1150 → 0.1048 | 8.8% | 0.1061–0.1172 / 0.1002–0.1102 | 0.1100 → 0.1015 |
| PEPC | 0.4079 → 0.3589 | 12.0% | 0.3687–0.4118 / 0.3475–0.3669 | 0.4071 → 0.3551 |

PGK coreは範囲が重なるため確実な8.8%改善とはしない。
PEPC coreの範囲は分離しているが、全CLIの短縮率ではない。
default full-scan校正は候補生成を反復するので候補となるが、その全反復の倍率は未計測。

## 同値性・メモリ・残る確認

- 5種の合成入力（空、単一、tie、fractional、大規模）について各9matchとallの
  計50比較で、全列・型・行順・浮動小数点値・表示文字列が厳密一致した。
- 表示ヘルパー単体は1～128状態、同値質量、ランダム、NaN、全NaN、float32、
  大きいint64を含む540比較で文字列が厳密一致した。
- 実データの候補表は全測定ペアで厳密一致した。
  scan coreのscan表・unit表も全ペアと最初の現行参照に対して厳密一致した。
- 両版の返却候補表のdeep payloadは同じ。any2speのPGKは117,692 bytes、
  PEPCは324,509 bytes。保持キャッシュは追加せず、配列workspaceは各表示分布の状態数に比例する。
  ピークRSSは測定しておらず、RAM削減率は主張しない。
- 製品化する場合は表示ヘルパーを通常の関数として実装し、サポート対象のpandas/NumPy版で
  tie/NaN/文字列順を回帰検証する。source文字列の置換を製品に持ち込む案ではない。
  実worker、校正全反復、CLI最終TSV、installed wheel、他OS・他Python版は未検証。

初回の調査ハーネス起動ではCLIのkeyword引数名との不一致で入力準備時に停止した。
callback名を修正し、最終版はコピー入力による再実行を正常完了した。数値比較の失敗はなかった。

## 再現

```bash
python reports/performance_candidates_20261022/proposal06.py
python -m ruff check reports/performance_candidates_20261022/proposal06.py
git diff --check
```

上記は成功。[全測定値・環境・入力hash](proposal06.json)、[試作スクリプト](proposal06.py)。
製品コード変更はなく、通常pytest全体は実行していない。source上の試作比較であり、
installed artifactや科学的校正を検証したという意味ではない。
