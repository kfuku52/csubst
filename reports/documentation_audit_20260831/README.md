# ドキュメント監査 — 2026-08-31

現行 CLI と一致しない説明が見つかった。優先すべき修正は、Wiki のインストール先、
`sites` の出力パス、`cs` 表の集計単位である。モデルの検証オプションには、
ドキュメントだけでなく実装側も整理する必要がある。

これは修正前の監査記録であり、以下の行番号と外部配布状況は監査時点のもの。
監査段階では README・Wiki 本文・実装を変更していない。
その後の修正内容と検証結果は [RESOLUTION.md](RESOLUTION.md) に記録する。

## 対象と確認方法

- 本体: `master` を fast-forward pull 後の `37868fddffeef77bb4a1403f877632ce8232e198`、CSUBST 1.14.9。
- Wiki: `d08f84cfaf2d68dc9269b1c07adee52c02478272`、最終更新 2026-08-18。
- 対象: README、CONTRIBUTING、TESTING、RELEASING、`docs/` の2文書、Wiki 全28ページ。合計34文書。
- ソース・CLI 定義との照合に加え、コードブロック内の具体的な CSUBST コマンド62件を argparse で確認。
  help / テンプレート13件はこの62件に含めない。引数解析エラーはなかった。
- 相対ファイル・画像参照と明示的な Wiki ページリンク113件を確認し、参照先の欠落はなかった。
- PGK で `dataset`、`sites`、追加表を有効にした `search`、プロット無効の `inspect` を実行。全4件が終了0。
- PyPI、Bioconda 配布メタデータ、Hugging Face 公式文書、引用論文の書誌情報も照合した。

環境は macOS 26.5.2 / arm64、Python 3.12.13、NumPy 2.5.2、SciPy 1.18.1、
pandas 3.0.5、Matplotlib 3.10.9、ete4 4.4.0。機械可読の確認記録は同じディレクトリの
`verification.json` に収録した。

## 現行仕様と一致しない記述

### D1 — P1: Wiki の PyPI インストール手順は利用できない

該当箇所は次の5ページ・8箇所。

| ページ | 行 |
| --- | --- |
| [Installation and test run](https://github.com/kfuku52/csubst/wiki/Installation-and-test-run) | 20、39、42 |
| [Dependency](https://github.com/kfuku52/csubst/wiki/Dependency) | 19、21 |
| [csubst download](https://github.com/kfuku52/csubst/wiki/csubst-download) | 50 |
| [csubst sites](https://github.com/kfuku52/csubst/wiki/csubst-sites) | 43 |
| [Structure mapping](https://github.com/kfuku52/csubst/wiki/Mapping-convergent-substitutions-on-protein-structures) | 110 |

`pip install csubst` と PyPI を前提とした extras 指定が記載されているが、
[PyPI の CSUBST メタデータ](https://pypi.org/pypi/csubst/json) は HTTP 404。
通常の PyPI を使う新規環境では、この経路で配布物を取得できない。
本体 README の GitHub URL 指定は正しく、Wiki と食い違っている。

修正案は、現行ソースのインストールを次に統一し、「現在のリリース」と区別すること。

```bash
python -m pip install git+https://github.com/kfuku52/csubst
python -m pip install "csubst[structure] @ git+https://github.com/kfuku52/csubst"
python -m pip install "csubst[vep] @ git+https://github.com/kfuku52/csubst"
```

同じ利用不能な extras 指定が
[依存不足時のエラーメッセージ](https://github.com/kfuku52/csubst/blob/37868fddffeef77bb4a1403f877632ce8232e198/csubst/model_resources.py#L43)
にもあるため、利用者向け案内をまとめて直す必要がある。

### D2 — P2: `sites` の出力ディレクトリ・ファイル名が旧仕様

主な該当箇所は [csubst sites](https://github.com/kfuku52/csubst/wiki/csubst-sites) 67–80行、
[出力解説](https://github.com/kfuku52/csubst/wiki/Interpreting-output-files-of-csubst-site) 8–138行、
[モード解説](https://github.com/kfuku52/csubst/wiki/csubst-site-modes) 28–70行。
[構造へのマッピング](https://github.com/kfuku52/csubst/wiki/Mapping-convergent-substitutions-on-protein-structures)
と [偽陽性の点検](https://github.com/kfuku52/csubst/wiki/Inspecting-spurious-convergence-using-csubst-site)
にも同じ旧接頭辞が残っている。

Wiki は `csubst_sites.branch_id23,51/` や `csubst_sites.tsv` と説明するが、
現在の CLI 既定値は `--outdir csubst_sites --output_prefix csubst`。
PGK の実行で確認した出力は次のとおり。

```text
csubst_sites/
  csubst.log
  csubst.branch_id23,51/
    csubst.tsv
    csubst.pdf
    csubst.outputs.tsv
    csubst.tree_site.tsv
    csubst.tree_site.pdf
    csubst.state.pdf
    csubst.state_N.tsv
    csubst.state_S.tsv
```

修正案は5ページを一括で更新し、一般形を `<outdir>/<output_prefix>.<branch-selection>/`
として示すこと。PDB・VESM・Chimera 出力も同じ接頭辞のルールと照合する。
内部 helper の旧既定値を、そのまま CLI の既定値として転載しない。
根拠は [CLI の既定値](https://github.com/kfuku52/csubst/blob/37868fddffeef77bb4a1403f877632ce8232e198/csubst/cli.py#L106)
と [呼び出し時のパス構築](https://github.com/kfuku52/csubst/blob/37868fddffeef77bb4a1403f877632ce8232e198/csubst/main_sites.py#L1971)。

### D3 — P2: `csubst_cs.tsv` の集計単位の説明が不正確

[search 出力解説](https://github.com/kfuku52/csubst/wiki/Interpreting-output-files-of-csubst-analyze)
24–25行では、`cs` と `cbs` を両方とも枝の組み合わせ別・サイト別の表として説明している。
実際の `cs` は対象の組み合わせを合算したサイト別の表で、枝 ID 列を持たない。
組み合わせ別の値が必要な場合は `cbs` を使う。

PGK の K=2 の実行結果:

| 出力 | 行の単位 / キー | 行数 |
| --- | --- | ---: |
| `csubst_s.tsv` | `site` | 417 |
| `csubst_bs.tsv` | `branch_id`, `site` | 27,105 |
| `csubst_cs.tsv` | `site`。対象組み合わせを合算 | 417 |
| `csubst_cbs.tsv` | `branch_id_1`, `branch_id_2`, `site` | 676,374 |

`cs` の合算処理は [sparse 実装](https://github.com/kfuku52/csubst/blob/37868fddffeef77bb4a1403f877632ce8232e198/csubst/substitution.py#L946)
と [dense 実装](https://github.com/kfuku52/csubst/blob/37868fddffeef77bb4a1403f877632ce8232e198/csubst/substitution.py#L1094)
の両方で確認した。

修正時にはキーと座標の基数も表にする。今回の実出力では search の `site` は0始まり、
sites の `codon_site_alignment` は1始まり。後者に結合するときは単なる同値結合にしない。
既定の小数4桁で保存された表同士の再集計は丸め誤差を含むため、完全一致の説明も避ける。

### D4 — P2: `download --verify` の説明と実装の契約が揃っていない

[download 解説](https://github.com/kfuku52/csubst/wiki/csubst-download) 34–38行と
[CLI help](https://github.com/kfuku52/csubst/blob/37868fddffeef77bb4a1403f877632ce8232e198/csubst/cli.py#L919)
は、`--verify yes` で既存リソースの SHA-256 を再計算すると説明している。

実装の記録用 fake を使った確認では、`yes` と `no` は次の同じ処理になった。

- VESM: [download の入口](https://github.com/kfuku52/csubst/blob/37868fddffeef77bb4a1403f877632ce8232e198/csubst/main_download.py#L25)
  は常に `verify_existing=True` を渡す。
  [リソースの読み出し](https://github.com/kfuku52/csubst/blob/37868fddffeef77bb4a1403f877632ce8232e198/csubst/model_resources.py#L124)
  でも必ずハッシュを検証する。
- ProstT5: [読み込み処理](https://github.com/kfuku52/csubst/blob/37868fddffeef77bb4a1403f877632ce8232e198/csubst/structural_alphabet.py#L249)
  はこのフラグを参照せず、Transformers のローカル読み込みと必要時のダウンロードを行う。
  CSUBST がこのオプションで SHA-256 を再計算する処理はない。

VESM の `--verify yes` が検証しないという意味ではない。検証は通常時にも行われており、
オプションの切り替えとして機能していない、という不整合である。
また、ProstT5 の読み込み成功を CSUBST による明示的な SHA-256 検証と同一視できない。

修正案は、まずリソースごとの検証仕様を決め、実装・help・Wiki を揃えること。
現行動作の説明では「VESM は常時 SHA-256 検証」「ProstT5 はローカル読み込み可否の確認」を区別する。
重量級のモデル取得は行わず、ソースと記録用 fake で呼び出し内容を確認した。

### D5 — P2: README のモデル保存先は適用範囲の説明が不足

[README 106–121行](https://github.com/kfuku52/csubst/blob/37868fddffeef77bb4a1403f877632ce8232e198/README.md#L106)
は共有モデルの既定キャッシュを `${CSUBST_CACHE_DIR:-~/.cache/csubst}` とまとめて説明する。
ただし、この指定で ProstT5 の重みの保存先まで変わるわけではない。

| 対象 | 現行の保存先 |
| --- | --- |
| VESM のモデルファイル | CSUBST キャッシュの `models/vesm-35m/v1/` |
| ProstT5 のモデルファイル | Hugging Face キャッシュ。または明示した `--prostt5_local_dir` |
| CSUBST が管理する ProstT5 ダウンロード用ロック | CSUBST キャッシュ。モデル本体とは別 |

[ProstT5 の呼び出し](https://github.com/kfuku52/csubst/blob/37868fddffeef77bb4a1403f877632ce8232e198/csubst/structural_alphabet.py#L312)
は `from_pretrained()` に `resource_cache_dir` を渡していない。
Hugging Face の標準キャッシュは `~/.cache/huggingface/hub` で、`HF_HOME` / `HF_HUB_CACHE`
などの設定に従う。[Hugging Face 公式説明](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)

オフライン環境に CSUBST キャッシュだけをコピーすると、ProstT5 のモデル本体を持ち込めない可能性がある。
README は保存先がモデルごとに異なる旨を短く補足し、詳細な表とオフライン手順を Wiki に置く。

### D6 — P3: 祖先状態プロットの `no` は現在も有効

[Ancestral state tree plots](https://github.com/kfuku52/csubst/wiki/Ancestral-state-tree-plots)
40行は `yes` / `no` 形式を受け付けなくなったと説明しているが、29–30行の形式表では `no` が含まれる。
現在も `no` は無効化の値かつ既定値であり、両オプションに `no` を指定した実 CLI は終了0だった。

修正案: `yes` は非対応、`no` は無効化、`all` またはサイト指定は有効化、と明記する。

### D7 — P3: 引用文献の著者名に誤記

[How ω and ωC are different](https://github.com/kfuku52/csubst/wiki/How-%CF%89-and-%CF%89_C-are-different)
3行の `Parler et al. (1980)` は `Perler et al. (1980)` が正しい。
リンク先 DOI に対応する [論文書誌](https://pubmed.ncbi.nlm.nih.gov/7388949/) で確認した。

## 誤りと断定せず、補足すべき点

1. **ソース対応と Bioconda 配布の区別（P2）**。
   README 34行の Python 3.10–3.14 対応は現行ソースについて正しい。
   一方、監査時点の [Bioconda 配布メタデータ](https://api.anaconda.org/package/bioconda/csubst)
   は 1.14.4、Python 3.10–3.13 用で、
   [レシピ](https://github.com/bioconda/bioconda-recipes/blob/master/recipes/csubst/meta.yaml)
   も Python 3.14 以上を除外している。インストール欄に配布チャネルの差を補足し、
   Python 3.14 では対応するソース経路を案内する。プロジェクト全体が3.14非対応という意味ではない。
2. **benchmark の終了コード（P2）**。
   [Wiki](https://github.com/kfuku52/csubst/wiki/csubst-benchmark) 58–59行は失敗後の続行可否しか説明していない。
   1.14.9 の「1件でも失敗すれば、結果保存後に終了2」を追記する。
   [CLI_SAFETY.md](https://github.com/kfuku52/csubst/blob/37868fddffeef77bb4a1403f877632ce8232e198/docs/CLI_SAFETY.md#L19)
   には既に正しく記載されている。
3. **スレッド数の比較表現（P3）**。
   [性能調整ガイド](https://github.com/kfuku52/csubst/wiki/Parallel-processing-and-performance-tuning) 47行の
   `--threads 1,2,4,8` は文中の略記だが、そのままは実行できない。CLI は単一の整数を受け取る。
   1、2、4、8をそれぞれ別の実行で指定すると書くか、出力先も分けるループにする。
4. **README と Wiki の重複（保守性）**。
   VESM の長い使い方と詳細ワークフローは専用ガイドへ集約し、README のロゴ・方法図は残す。
   出力名・既定値は CLI と小さな実行の manifest から確認できる形にし、手書きの重複を減らす。
   本体と別リポジトリである Wiki も、インターフェース変更時の更新対象に含める。

## 再現コマンドと限界

インストール済みの対象バージョンを使い、空の作業ディレクトリで実行した。
既存のデータを上書きする `--force` は使っていない。

```bash
python -m csubst dataset --name PGK
python -m csubst sites --alignment_file alignment.fa.gz --rooted_tree_file tree.nwk --branch_id 23,51
python -m csubst search --alignment_file alignment.fa.gz --rooted_tree_file tree.nwk \
  --foreground foreground.txt --exhaustive_until 1 --s yes --bs yes --cs yes --cbs yes
python -m csubst inspect --alignment_file alignment.fa.gz --rooted_tree_file tree.nwk \
  --plot_state_aa no --plot_state_codon no
```

この監査ではコードブロック62件の全解析を実行したが、全コマンドの計算を実行したわけではない。
optional モデルのダウンロード・推論、オンライン構造検索、全外部 URL の到達性、全 heading anchor、
全画像の科学的内容は今回の実行確認には含めない。34文書の文章・例と実装を照合し、
モデル関連はネットワークや重量級依存を必要としない呼び出し記録を併用した。
コードを変更していないため、全テストスイートの再実行も行っていない。

`analyze` / `site` は有効な互換エイリアスなので、旧名や歴史的 URL だけを誤りとは扱っていない。
Wiki 内で旧結果と明示された PEPC の数値例や、日付付きの過去の監査・性能レポートも、
現在の既定値を説明する文書とは区別した。
