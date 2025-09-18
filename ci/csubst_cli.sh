#!/usr/bin/env bash
set -euo pipefail

echo "== csubst CLI tests =="

which csubst
csubst --version || true

# --- ヘルプが期待どおり出るか（主要サブコマンドと必須引数名） ---
csubst -h | head -n 20
csubst -h | grep -E '\banalyze\b' >/dev/null
csubst -h | grep -E '\bdataset\b' >/dev/null
csubst analyze -h | grep -- '--alignment_file' >/dev/null
csubst analyze -h | grep -- '--rooted_tree_file' >/dev/null
csubst analyze -h | grep -- '--foreground' >/dev/null

# 作業ディレクトリ（smoke と同じ場所を再利用）
WORKDIR="${RUNNER_TEMP:-$(mktemp -d)}/csubst_smoke"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 必要なら最小データを用意
if [ ! -s alignment.fa ]; then
  csubst dataset --name PGK
fi
test -s alignment.fa && test -s tree.nwk && test -s foreground.txt

# --- 異常系：存在しない入力で失敗すべき ---
set +e
csubst analyze --alignment_file __NO_FILE__.fa \
  --rooted_tree_file tree.nwk --foreground foreground.txt --threads 1 >/dev/null 2>&1
rc=$?
set -e
if [ $rc -eq 0 ]; then
  echo "ERROR: 存在しない入力で analyze が成功してしまいました"; exit 1
else
  echo "OK: 異常系（欠損入力）は非0で失敗"
fi

# --- 正常系(1)：既定モデル（ECM系） ---
rm -f alignment.fa.{iqtree,log,rate,state,treefile} || true
env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst analyze \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --threads 1

shopt -s nullglob
CB1=(csubst_cb_*.tsv)
[ ${#CB1[@]} -ge 1 ] || { echo "ERROR: ECM 実行後に cb TSV がありません"; exit 1; }
[ $(wc -l < "${CB1[0]}") -ge 2 ] || { echo "ERROR: cb TSV が空のようです"; exit 1; }
grep $'\t' "${CB1[0]}" >/dev/null || { echo "ERROR: cb TSV がタブ区切りでない可能性"; exit 1; }

# --- 正常系(2)：GY+F3x4+R2（別分岐も生存確認） ---
rm -f alignment.fa.{iqtree,log,rate,state,treefile} || true
env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst analyze \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --iqtree_model GY+F3x4+R2 \
  --threads 1

CB2=(csubst_cb_*.tsv)
[ ${#CB2[@]} -ge 1 ] || { echo "ERROR: GY 実行後に cb TSV がありません"; exit 1; }

# アーティファクト収集
ART="$WORKDIR/_artifacts_cli"
mkdir -p "$ART"
cp -v csubst_cb_*.tsv "$ART" 2>/dev/null || true
cp -v alignment.fa.{iqtree,log,rate,state,treefile} "$ART" 2>/dev/null || true

echo "CLI tests OK"
