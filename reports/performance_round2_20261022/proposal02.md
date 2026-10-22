# 案02: 割当て内の枝metadata配列とtarget/control maskの再利用

**採用可能。** 固定候補校正で、候補ごとに繰り返していたID変換、`np.isin`、
length列のboxingを、同じ前景割当て内で共有する。
イベント集約・露出計算・スコア・乱数・加算順は現行のまま。
本体にはまだ変更していない。

## 測定

基準`88528ce`、1.16.21。環境・全反復・入力hashは[JSON](proposal02.json)。
macOS arm64、Python 3.14.7、BLAS/CSUBST各1スレッド、同梱固定fit、seed8。
共有ロックで他案の測定と直列化し、warmup1組＋順序交互の測定3組。
50割当て、PGKは12候補・64枝、PEPCは103候補・140枝。

wall秒、中央値（最小–最大）:

| 範囲 | 現行 | 試作 | 中央値短縮 |
| --- | ---: | ---: | ---: |
| PGK校正worker chunk | 0.1332 (0.1319–0.1370) | 0.0833 (0.0804–0.0838) | 37.4% |
| PEPC校正worker chunk | 0.9596 (0.9542–0.9891) | 0.4970 (0.4937–0.5139) | 48.2% |
| PGK全subprocess | 1.8739 (1.8367–1.9130) | 1.7686 (1.7152–1.7696) | 5.6% |
| PEPC全subprocess | 4.7753 (4.7643–4.8263) | 4.2633 (4.2216–4.2918) | 10.7% |

全subprocessは独立Python起動・import・試作ラッパー・全scan処理を含む。
両条件とも同じラッパーを読み、試作側のみ関数をプロセス内で置換する。
全条件で現行と試作のwall範囲は分離した。CPU中央値もPGK全体1.84→1.72秒、
PEPC全体4.74→4.22秒で減少した。他CPU・OS・既定1000割当ての測定ではない。

追加の枝IDコピーと2maskは1familyでPGK640 bytes/PEPC1400 bytes。
length列は既存配列への参照で、割当て終了時に全て解放する。
state-awareのparent IDコピーは別途8 bytes/枝。
小さなPython objectの保持量をこの配列数には含めない。
全subprocessのpeak RSS中央値はPGK321.4→330.1 MB、PEPC577.6→562.2 MB
（10進、`/usr/bin/time -l`）。範囲は重なるのでRSS改善・悪化は断定しない。

## 同等性と実装条件

- 実PGK/PEPCの全50割当てで、スコア、候補キー、最大値、成功数、失敗数、診断が厳密一致。
  各条件とも成功50・失敗0。空候補familyも一致。
- 全subprocessの8比較組で、scan表とunits表の全列・型・行順が`check_exact=True`で一致。
  校正JSONと推論JSONも全項目が厳密一致。
- 95の合成シナリオを検査。legacy3露出×3length、重複/重なるtarget/control、
  疎な枝ID・行順、空イベント、NaN/負イベント、0露出、欠測、endpoint露出の検証を含む。
  full/score-only両返値、未準備の公開呼出し、例外型・文言・失敗順序を比較した。
  合成endpoint境界はstubであり、実endpoint計算は固定fitのPGK/PEPCで別に通す。

再利用は1割当て・一定のrate-length・不変のbranch metadataに限定する。
target/control maskはtrait×target-class別で、controlからtargetを除く現行規則を保持する。
metadataの変換をlazyにし、イベント列の検査とlength列の検査の順序も維持した。
fit・割当て・worker間で持ち回る汎用キャッシュにはしない。
本実装時は専用prepared型と所有範囲を明示し、既存の未準備呼出し経路を残すのがよい。

この検査はsource試作・固定fitの回帰比較であり、新しい科学的校正、installed wheel、
並列worker、nativeなしfallbackを検証していない。本体未変更のため全suiteは実行していない。
試作PythonのRuffと`git diff --check`は成功。

## 根拠と再現

[`np.isin`](https://numpy.org/doc/stable/reference/generated/numpy.isin.html)は毎回boolean配列を返す。
[`Series.to_numpy`](https://pandas.pydata.org/docs/reference/api/pandas.Series.to_numpy.html)は
dtype変換時にコピーが必要になる場合があるため、不変なmetadataの再変換を避ける。
[校正契約](../../docs/SCAN_CALIBRATION.md)では枝metadataや固定fitが割当て内で不変。
この不変条件からの最適化判断であり、一般の可変DataFrameへの適用は想定しない。

```bash
python reports/performance_round2_20261022/proposal02.py
python -m ruff check reports/performance_round2_20261022/proposal02.py
```

入力は実行時に一時ディレクトリへコピーする。元fixtureへの書込はない。
実行ログ・主結果表は一時領域に置き、保存JSONには入力名とhash、比較結果を残す。
