# 高次探索の候補選別: 修正と検証

前景指定なしの通常CLIで、legacy (`marginal`) とjointの両方がPEPCのarity 6まで探索し、同じ1組を検出するよう修正した。診断用の関数差し替えは不要。

## 修正内容

1. 前景指定なしでも高次候補を前景ラベルで除いていた。前景なしでは、閾値を通過した全候補を拡張する。
2. `--max_combination` による順位付けが `--cutoff_stat` の指定順を無視し、表の列順になっていた。指定順で順位付けし、正規表現の重複一致は順位付けで重複させない。
3. 前景外の候補を除く前に候補数の上限を適用し、有効な前景候補が上限枠から押し出されていた。前景適格性の判定後に上限を適用する。
4. 複数形質の候補を先に混ぜることで、ある形質で非独立な組み合わせが、生成に関与しない別形質の判定を通じて残っていた。形質ごとに生成・依存関係を確認してから統合する。別形質が実際に独立な候補として生成できる場合は残し、形質ごとの依存注釈も保持する。

終了メッセージも、候補不足を常に統計閾値や系統独立性だけの問題と断定しない表現に変更した。`--max_combination` は生成後のK行数ではなく、拡張に使うK-1候補数の上限であることをCLIヘルプと[探索ガイド](../../docs/HIGHER_ORDER_SEARCH.md)に明記した。置換確率やomegaCの計算式は変更していない。

## PEPCの通常CLI検証

同梱alignment/tree/IQ-TREE出力、`--max_arity 6 --exhaustive_until 2`、`OCNany2spe >= 2.0` かつ `omegaCany2spe >= 5.0`、long-tail補正なし、サイト別出力なし、BLAS 1、block size 64、seed 8。

各方式をCPU 1・4（`--b no`）、CPU 1（`--b yes`）で実行した6条件すべてが正常終了。arity 2〜6の標準TSVは、方式内でCPU数・枝別出力の切替、および[修正前の診断](../endpoint_detection_20260910/README.md)と一致した（30表比較）。これは標準TSVの一致であり、丸め前のビット単位の一致を主張するものではない。

| arity | legacy: 計算数 / 閾値通過数 | joint: 計算数 / 閾値通過数 |
|---|---:|---:|
| 2 | 8,308 / 107 | 8,446 / 85 |
| 3 | 446 / 39 | 416 / 45 |
| 4 | 66 / 20 | 87 / 20 |
| 5 | 13 / 6 | 14 / 6 |
| 6 | 1 / 1 | 1 / 1 |

arity 6の枝IDは **1, 26, 33, 35, 40, 102**。OCNany2speはlegacy約2.499416、joint約2.498897。omegaCany2speは両方inf（極小の同義置換値が既存閾値でゼロ扱いになるため）。

前景指定ありも両方式で検証した4条件が正常終了。同梱ファイルの全行lineage ID 1では高次の独立な前景候補がなくarity 2で終了し、12 tipを別々のlineage IDとした場合はarity 5まで進む。この制約は前景指定なしの停止バグとは異なる。既存の前景条件を緩和していない。

## 自動検証

- `make test`: 1,848 passed、5 skipped。スキップは既存のtorchバージョン条件3件とgemmi未導入2件。
- `make test-native`: strict nativeで8 passed。
- `make lint`、`make typecheck`: 成功。
- 新しい回帰テストは、前景なしの実際の候補生成でarity 6まで進むこと、全探索からヒューリスティックへの移行、CPU指定、候補順位・上限、閾値境界・NaN・inf・空候補を確認。
- 複数形質の候補生成は、arity 2〜4、全探索範囲内外について独立に実装した総当たりとの120ケースの照合で確認。生成元の形質と独立性の両条件を満たす場合だけ候補が残る。
- 従来の複数形質テストの一つは、生成元で非独立な候補を別形質が無条件に救済する挙動を期待していた。生成可能な別形質がある場合・ない場合に分け、意図した独立性と候補生成の条件を検証するテストへ修正した。

今回の調査範囲は候補選別・高次生成・前景依存注釈と、それらを通るlegacy/jointの探索。既報のjointの高次集計や`--b yes`の速度差を解消する変更ではなく、性能の再測定も行っていない。

## 記録と再現

[検証JSON](verification.json)に6条件のCLI引数、表比較、前景4条件の結果を記録。[全体テスト](tests.log)、[lint・型チェック](checks.log)、[strict native](native.log)を保存。実データの各実行ログとarity 6のTSVもこのディレクトリに保存している。

以下の`normal`は候補生成関数を差し替えず通常CLIを実行する。スクリプトは検証用に丸め前の表も保存する。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python reports/endpoint_detection_20260910/reproduce.py marginal /tmp/pepc-fixed-marginal normal
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python reports/endpoint_detection_20260910/reproduce.py joint /tmp/pepc-fixed-joint normal
```
