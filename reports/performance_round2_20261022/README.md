# 追加10案の性能検証（2026-10-22）

基準は **CSUBST 1.16.21 / `88528ce`**。
前回の[候補調査](../performance_candidates_20261022/README.md)と
[採用4変更](../performance_implementation_20261022/README.md)を踏まえ、
異なる処理を対象に追加10案を選んだ。
各案を別のサブエージェントが試作・実測・同等性検査し、主担当がコードと判定をレビューする。

**実CLIで確認した採用候補は1件。用途を限定した条件付き候補が5件、見送りが4件。**
10件それぞれ別のサブエージェントによる検証を行った。本体への導入は今回の対象外。

## 評価表

時間はwall中央値。同じ行の現行版と試作だけを比較する。
異なる案のbaseline時間同士は測定時刻や処理範囲が異なるため比較しない。

| 案 | 改善案 | 主な測定結果 | 判定 |
| --- | --- | --- | --- |
| [01](proposal01.md) | one-hot葉の行列積を列参照へ特殊化 | 推論stage PGK 0.379→0.435秒、PEPC 1.758→1.685秒。両方範囲重複 | 見送り |
| [02](proposal02.md) | 割当てごとの枝metadata・target/control mask再利用 | **全CLI PGK 1.874→1.769秒、PEPC 4.775→4.263秒**。両方範囲分離 | **採用候補** |
| [03](proposal03.md) | 既存イベントcache内の枝別数値vector | 校正stage 24.2%/28.1%短縮。全CLIは範囲重複 | 条件付き |
| [04](proposal04.md) | 解析的検定で変更されない部分木の尤度再利用 | PGK解析stage 0.341→0.278秒。PEPCは範囲重複、保持配列増 | 条件付き |
| [05](proposal05.md) | omega経験的tailを列chunk処理 | 合成RSS 496.8→455.2 MB、実統計stageは約2倍に遅延 | 通常見送り、RAM制約時のみ条件付き |
| [06](proposal06.md) | 共通maxT参照の整列と一括tail検索 | 実PGKの集計0.195→0.041 ms。1万候補の合成負荷で明確な短縮 | 大量候補のみ条件付き |
| [07](proposal07.md) | synonymous用の固定遷移積cache | 推論stageで範囲重複。追加積配列7.62/16.67 MB | 見送り |
| [08](proposal08.md) | unit/branch membershipと支持maxの配列化 | 実PGK/PEPCで改善未確認。多数unitの合成条件だけ短縮 | 通常見送り |
| [09](proposal09.md) | uniformizationの到達距離に基づく開始次数 | 全遷移準備PGK 0.0223→0.0227秒、PEPC 0.0597→0.0505秒。wall範囲重複 | 見送り |
| [10](proposal10.md) | 構造重み付けCVのfold内log-prior等の再利用 | 合成PGK/PEPC形状のnested CVが42.0%/54.9%短縮。全返値厳密一致 | 条件付き |

## 測定条件と判定

macOS 27.0.1 arm64、Python 3.14.7、NumPy 2.5.3、SciPy 1.17.1、pandas 3.0.5。
BLAS環境変数を数値ライブラリのimport前に1へ設定し、実CLIは1 process。
同梱PGK（417 codon sites）とPEPC（971 codon sites）の固定fitを一時領域へコピーした。
IQ-TREE再推定・構造モデルの取得・新たな科学的校正は含まない。

各案はwarmup後、順序を交互にした3回以上の反復を行った。
全案共通の`bench_common.benchmark_lock()`で入力準備・CPU計測を直列化した。
同じマシン上の他の負荷や温度変動を完全に除去したものではない。
中央値に差があってもmin–max範囲が重なる条件は、確実な改善と扱わない。
wall/CPU、stage/全CLI、実入力/人工負荷、配列サイズ/peak RSSを区別する。

各案のJSONに全反復、環境、対象入力hash、同等性の結果を保存する。
速度の根拠はこの測定であり、文献や演算量の削減だけから倍率を推定していない。
[設計根拠・資料](literature.md)と各案の詳細報告を参照。
別々の10担当と完了状態は[担当記録](agents.json)に保存した。

## 採用に向けた優先度

案02は実CLIでPGK **5.6%**、PEPC **10.7%**の中央値短縮があり、
全列・型・行順と校正/推論JSONが厳密一致した。
追加配列は1familyあたり約640/1400 bytesで、割当て終了時に解放できる。
固定fit・一定のmetadata・一定のrate-lengthという私有スコープを明示して実装する価値がある。
公開の未準備呼出し経路と失敗順序は維持する。

案03は局所短縮が明確だが、全CLIでは未確定。
案02との併用効果を足し算して見積もらない。
案04・06は特定機能・規模向けで、通常scan全体の倍率として示さない。
案05は速度改善として採用せず、RAM上限が支配する条件を別途示せる場合に限り再検討する。

案10はopt-in構造重み付けの既定joint grid（744候補、5clade）の
全nested CVで1.992→1.155秒、8.438→3.805秒となり、両方の範囲が分離した。
PGK/PEPCと同じ形状を持つ**合成count/context/features**の結果であり、
実構造データや全CLIの効果ではない。
fold内のlog-prior、mask、held-out responseを同じ32 MiB上限へ計上し、
全予測・選択パラメータ・診断とheld-out独立性を検査した。
実入力で確認できれば、案02に続く優先候補となる。

## 正しさの検査と限界

出力の比較方法、境界条件、実行した既存テストは各案の報告に記載する。
今回の変更は試作・検証ハーネスと報告のみで、本体の数値実装・既定値・fixture・参照値を変更していない。
本体への導入完了や他OSでの速度、installed wheel、独立した生物学的校正を証明するものではない。
導入時にはaffected tests、並列worker、source-only fallback等の製品回帰検証が必要。

案01は主担当のレビューで試作sourceの最終置換漏れを発見した。
初回計測・判定を無効化し、gather行の実行回数を追跡してから全検査・計測をやり直した。
保存したJSONと上表は修正版だけの結果である。
以後の案もcache hitや対象経路の実行など、最適化が実際に動いた証拠を確認した。

主担当の最終チェックは[検証記録](checks.json)に保存する。
全11本の検証Pythonに対するRuff・compileall、`make lint`、strict JSON読込、
ローカル参照先・home path混入・新規ファイルも含む空白検査が成功した。
`make lint`には本体Ruff、repository hygiene、文書22件・command23件・local link73件の検査を含む。
Wikiの16参照はcloneを指定しておらず対象外。
既存テストを試作backendへ差し替えた案04/09/10では、それぞれ7/26/22 passed・skipなし。
本体未変更の調査なので本体全suite・typecheck・package・他OS CIは再実行していない。

## 再現

リポジトリrootの開発環境で、必要な案だけ実行する。
本体の関数は子プロセスまたは一時的な私有namespace内で差し替え、終了時に戻す。
元fixtureへの書込はせず、生の実行ディレクトリは一時領域へ置く。

```bash
python reports/performance_round2_20261022/proposal01.py
python reports/performance_round2_20261022/proposal02.py
python reports/performance_round2_20261022/proposal03.py
python reports/performance_round2_20261022/proposal04.py
python reports/performance_round2_20261022/proposal05.py
python reports/performance_round2_20261022/proposal06.py
python reports/performance_round2_20261022/proposal07.py
python reports/performance_round2_20261022/proposal08.py
python reports/performance_round2_20261022/proposal09.py
python reports/performance_round2_20261022/proposal10.py
```

各ハーネスは該当する保存JSONを更新するので、再測定時には旧結果を別に保管する。
案02・03の全CLI用RSS取得にはmacOSの`/usr/bin/time -l`を使う。
案05のRSSは独立子プロセスの`resource.getrusage`で測る。
