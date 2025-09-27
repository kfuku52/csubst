#!/usr/bin/env bash
set -euo pipefail

echo "== csubst CLI tests =="

which csubst
csubst --version || true

# --- ヘルプ（主要サブコマンド/引数の存在） ---
csubst -h | head -n 20
csubst -h | grep -E '\banalyze\b' >/dev/null
csubst -h | grep -E '\bdataset\b' >/dev/null
csubst analyze -h | grep -- '--alignment_file' >/dev/null
csubst analyze -h | grep -- '--rooted_tree_file' >/dev/null
csubst analyze -h | grep -- '--foreground' >/dev/null

# 作業ディレクトリ（smoke と共用）
WORKDIR="${RUNNER_TEMP:-$(mktemp -d)}/csubst_smoke"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 最小データが無ければ生成
if [ ! -s alignment.fa ]; then
  csubst dataset --name PGK
fi
test -s alignment.fa && test -s tree.nwk && test -s foreground.txt

# --- 異常系：欠損入力で失敗すること ---
set +e
csubst analyze --alignment_file __NO_FILE__.fa \
  --rooted_tree_file tree.nwk --foreground foreground.txt --threads 1 >/dev/null 2>&1
rc=$?
set -e
[ $rc -ne 0 ] || { echo "ERROR: 欠損入力で成功してしまった"; exit 1; }
echo "OK: 異常系で非0終了を確認"

# --- 正常系(ECM系：デフォルト) ---
rm -f alignment.fa.{iqtree,log,rate,state,treefile} || true
env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst analyze \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --threads 1

shopt -s nullglob
CB1=(csubst_cb_*.tsv)
[ ${#CB1[@]} -ge 1 ] || { echo "ERROR: ECM 実行後に cb TSV が無い"; exit 1; }
[ $(wc -l < "${CB1[0]}") -ge 2 ] || { echo "ERROR: cb TSV が空"; exit 1; }
grep $'\t' "${CB1[0]}" >/dev/null || { echo "ERROR: TSV がタブ区切りでない可能性"; exit 1; }

# --- 正常系(GY系：別分岐) ---
rm -f alignment.fa.{iqtree,log,rate,state,treefile} || true
env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst analyze \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --iqtree_model GY+F3x4+R2 \
  --threads 1
CB2=(csubst_cb_*.tsv)
[ ${#CB2[@]} -ge 1 ] || { echo "ERROR: GY 実行後に cb TSV が無い"; exit 1; }

# アーティファクト収集（任意）
ART="$WORKDIR/_artifacts_cli"
mkdir -p "$ART"
cp -v csubst_cb_*.tsv "$ART" 2>/dev/null || true
cp -v alignment.fa.{iqtree,log,rate,state,treefile} "$ART" 2>/dev/null || true

echo "CLI tests OK"
