# 案05: 固定候補校正で使わないunit表を生成しない

**判定: 有望。** `candidate_fixed` の校正stageでCPU時間を削減し、
PGKでは7反復すべてが改善した。PEPCは6/7反復が改善した。
本体への採用前に全CLIと既存回帰を追加確認する。

`_build_permuted_trait_context` は、固定候補の再採点にもunit別のDataFrame、
葉名、文字列化した枝ID、sister枝を毎試行で構築していた。
試作では既存samplerの後で `include_units=False` を選び、foregroundと
rate用cladeの枝集合を保持する。`scan_other_scope=sister` のときだけ
sister枝を構築し、`all` では省略する。固定候補の内部contextは
`units=None` を明示する。既定のbuilder呼出しと `full_scan` は従来の
unit表と戻り値keyを保つ。sampling law、seed、試行回数、失敗処理を変更しない。

計測は基準commit `624bb8a6dae536d0fa10891f666684ef1ad3dab0` のソースと、
そのprivate namespace上の試作を比較した。実データPGK/PEPCの同梱alignment、
foreground、tree、固定IQ-TREE fitを新しい一時ディレクトリへコピーし、
CLIが呼ぶ固定候補校正chunkを捕捉して測定した。各呼出しでworker cacheを
作り直し、50 assignmentを処理する。PGKは12候補、PEPCは103候補。
入力SHA256、全反復と環境は [proposal05.json](proposal05.json) に記録した。

| 実データstage | CPU中央値: 基準→試作 | CPU削減 | wall中央値: 基準→試作 |
| --- | --- | --- | --- |
| PGK | 0.075381→0.044249秒 | 41.3% | 0.326298→0.081998秒 |
| PEPC | 0.287897→0.262543秒 | 8.8% | 0.361270→0.311811秒 |

macOS arm64、Python 3.14.7、NumPy 2.5.3、SciPy 1.17.1、pandas 3.0.5。
BLAS等のthread数をimport前に1へ固定し、共有lockの下で2 warmup、
7反復を実施した。基準と試作の実行順を交互に入れ替えた。
wall時間の変動が大きく、特にPGKのwall改善率は安定した期待値として
扱えない。CPUの基準/試作範囲はPGK 0.055531–0.101390 / 0.042069–0.054915秒、
PEPC 0.273051–0.322070 / 0.242065–0.290784秒だった。

すべての実データ反復でscore、候補key、configuration、configuration ID、
seed、sampling attempts、成功/失敗数、trial diagnosticsが厳密一致した。
小さい木ではstem/clade/lineage、all/sister、foreground stem設定、
sister stem設定、および非単系統lineageと内部cladeを組み合わせた48設定で、
列挙可能な全256 configurationのtarget/control枝集合、既定context、
exact scan結果と診断を比較した。空scan、zero-exposure no-test、
非有限入力、無効なnull統計、context失敗、proposal枯渇、Monte Carlo、
`full_scan`、無効な明示configurationと原foreground除外の拒否も比較する。
proposal枯渇のattempts=37も保持する。

再現コマンド:

```bash
python reports/performance_round3_20261023/proposal05.py
```

[試作patch](proposal05.patch) は本体へ未適用。
この測定範囲は校正chunkであり、ASRや候補発見を含む全CLIの速度を示さない。
peak RSSは測定しておらず、メモリ削減率は主張しない。
独立した生物学的校正、installed-wheel、別OSの性能は未検証。
`git apply --check` とスクリプトの構文検査も通過した。
