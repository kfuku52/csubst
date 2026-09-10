# PEPCの高次検出: legacyとjoint

**更新:** 以下は修正前の診断記録。前景指定なしの停止バグと関連する候補選別の問題は修正済みで、[通常CLIでの再検証](../search_candidate_fix_20260910/README.md)でもarity 6を確認した。

**jointでも同じarity 6の組み合わせが閾値を満たす。** ただし、現行の前景指定なしの通常検索には、閾値通過候補を前景フィルタで除いてしまいarity 3へ進めない問題があり、これはlegacyにも共通する。

以下はこの候補選別を切り分ける診断。高次候補の生成時だけ `get_node_combinations(..., cb_all=True)` を両方式で使用した。計算本体や作業ツリーのパッケージファイルは変更していない。**通常CLIをそのまま実行した結果ではない。**

## 同一の探索ルールでの比較

同梱PEPC alignment/tree/IQ-TREE出力、最大arity 6、全探索はarity 2まで、以後は閾値通過組み合わせから拡張。閾値は `OCNany2spe >= 2.0` かつ `omegaCany2spe >= 5.0`。long-tail補正なし、b/s/cs/bs/cbs出力なし、CPU/BLAS 1、block size 64、seed 8。

| arity | legacy通過数 | joint通過数 | 共通する組み合わせ数 |
|---|---:|---:|---:|
| 2 | 107 | 85 | 68 |
| 3 | 39 | 45 | 38 |
| 4 | 20 | 20 | 19 |
| 5 | 6 | 6 | 6 |
| 6 | 1 | 1 | 1 |

arity 6の枝IDは両方式とも **1, 26, 33, 35, 40, 102**。

| 指標 | legacy | joint |
|---|---:|---:|
| OCNany2spe | 2.499416 | 2.498897 |
| ECNany2spe | 2.939116e-7 | 9.535080e-9 |
| OCSany2spe | 9.236544e-20 | 3.565179e-20 |
| omegaCany2spe | inf | inf |

omegaCのinfは、極小の観測同義置換数が既存の閾値処理でゼロ扱いになるため。統計的有意性や無限の証拠を意味しない。低次arityでは失う組み合わせと新たに得る組み合わせがあり、両計算法の全結果が同一という意味ではない。

## 通常検索が止まる理由

`main_analyze.cb_search` は高次候補生成へ `cb_all=False` を渡す。`combination.get_node_combinations` は、渡された閾値通過表からさらに `is_fg/is_mf/is_mg == Y` の行だけを使う。前景指定なしではPLACEHOLDERのこれらのラベルがすべてNとなり、候補が空になる。

通常の前景なしlegacyではarity 2で107組が閾値を通るが、arity 3を作れず終了する。jointも85組が通るが同様に終了する。したがって、以前の速度調査での「jointがarity 3で終了」はjointによる高次信号消失の根拠にはならない。

同梱 `PEPC.foreground.txt`（全行lineage ID 1）をそのまま指定した場合も、現行の前景依存判定によって両方式で高次候補が残らなかった。12 tipを別々のlineage IDにした対照では、両方式ともarity 5まで検出した。これらの前景指定は以前の実行条件と同一と確認できていないため、上の全枝診断と混同しない。

候補選別のパッケージ本体の修正は、この調査では行っていない。

## 証拠と再現

- [結果JSON](results.json)、`legacy_passed_arity*.tsv` と `joint_passed_arity*.tsv` に閾値通過組み合わせを丸め前の値で保存。
- [legacy全枝診断ログ](all-legacy6.log)、[joint全枝診断ログ](all-joint6.log)。ログのCLI行だけでは診断用の関数差し替えを表せないので、下のスクリプトを参照。
- [診断スクリプト](reproduce.py)。`all_candidates` は実行中の候補生成関数だけを差し替える。`bundled` は同梱前景ファイル、`independent` は12 tipを別lineage IDにした対照。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python reports/endpoint_detection_20260910/reproduce.py marginal /tmp/pepc-legacy6 all_candidates
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python reports/endpoint_detection_20260910/reproduce.py joint /tmp/pepc-joint6 all_candidates
```
