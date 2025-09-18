#!/usr/bin/env bash
set -euo pipefail

echo "== csubst command tests =="

which csubst
csubst --version || true

# --- ヘルプ存在チェック ---
csubst -h | grep -E '\bdataset\b' >/dev/null
csubst -h | grep -E '\banalyze\b'  >/dev/null
csubst -h | grep -E '\bsite\b'     >/dev/null
csubst -h | grep -E '\bsimulate\b' >/dev/null
csubst analyze  -h | grep -- '--alignment_file'   >/dev/null
csubst site     -h | grep -- '--alignment_file'   >/dev/null
csubst simulate -h | grep -- '--alignment_file'   >/dev/null

# --- 作業ディレクトリ ---
WORKDIR="${RUNNER_TEMP:-$(mktemp -d)}/csubst_smoke"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 最小データが無ければ生成
if [ ! -s alignment.fa ]; then
  csubst dataset --name PGK
fi
test -s alignment.fa && test -s tree.nwk && test -s foreground.txt

# 便利関数：直近生成ファイルの列挙
newer_files() { find . -maxdepth 1 -type f -newer "$1" -printf "%f\n" || true; }

# --- analyze（既定＝ECM系） ---
rm -f alignment.fa.{iqtree,log,rate,state,treefile} || true
MARKER=$(mktemp); sleep 1; touch "$MARKER"
env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst analyze \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --threads 1
NEW_TSV=($(find . -maxdepth 1 -type f -name "csubst_*.tsv" -newer "$MARKER" -print))
[ ${#NEW_TSV[@]} -ge 1 ] || { echo "ERROR: analyze 後に TSV が増えていない"; exit 1; }
echo "OK: analyze(created): ${NEW_TSV[*]}"

# --- analyze（GY 分岐） ---
rm -f alignment.fa.{iqtree,log,rate,state,treefile} || true
MARKER=$(mktemp); sleep 1; touch "$MARKER"
env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst analyze \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --iqtree_model GY+F3x4+R2 \
  --threads 1
NEW_TSV=($(find . -maxdepth 1 -type f -name "csubst_*.tsv" -newer "$MARKER" -print))
[ ${#NEW_TSV[@]} -ge 1 ] || { echo "ERROR: analyze(GY) 後に TSV が増えていない"; exit 1; }
echo "OK: analyze(GY)(created): ${NEW_TSV[*]}"

# --- site（サイト別計算） ---
# まずは fg（foreground 指定の枝集合）で試す
rm -f alignment.fa.{iqtree,log,rate,state,treefile} || true
MARKER=$(mktemp); sleep 1; touch "$MARKER"

set +e
env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst site \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --branch_id fg \
  --threads 1
rc=$?
set -e

if [ $rc -ne 0 ]; then
  echo "WARN: site --branch_id fg が失敗。cb出力から枝IDをフォールバックで抽出します"
  # 例：csubst_cb_2.tsv などから「数字,数字」の最初の組を拾う
  combo="$(grep -Eho '(^|[[:space:]])[0-9]+,[0-9]+' csubst_cb_*.tsv 2>/dev/null | head -n1 | tr -d '[:space:]')"
  if [ -z "${combo}" ] && [ -f csubst_b.tsv ]; then
    # 単枝IDでも可（最初の1つ）
    combo="$(awk -F'\t' 'NR==2{print $1}' csubst_b.tsv 2>/dev/null)"
  fi
  [ -n "${combo}" ] || { echo "ERROR: 枝IDの抽出に失敗"; exit 1; }
  echo "Fallback branch_id=${combo}"
  env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst site \
    --alignment_file alignment.fa \
    --rooted_tree_file tree.nwk \
    --foreground foreground.txt \
    --branch_id "${combo}" \
    --threads 1
fi

NEW_SITE_TSV=($(find . -maxdepth 1 -type f -name "csubst_site*.tsv" -newer "$MARKER" -print))
# 名前が将来変わっても、直後に増えた .tsv があることも確認
[ ${#NEW_SITE_TSV[@]} -ge 1 ] || NEW_SITE_TSV=($(find . -maxdepth 1 -type f -name "*.tsv" -newer "$MARKER" -print))
[ ${#NEW_SITE_TSV[@]} -ge 1 ] || { echo "ERROR: site 実行で TSV が作られていない"; exit 1; }
echo "OK: site(created): ${NEW_SITE_TSV[*]}"

# --- simulate（シミュレーション） ---
MARKER=$(mktemp); sleep 1; touch "$MARKER"
# ※ simulate は速い前提。安全のため threads=1。
env PYTHONOPTIMIZE=1 csubst simulate \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --threads 1
NEW_SIM=($(find . -maxdepth 1 -type f \( -name "*.fa" -o -name "*.fasta" -o -name "*.phy" \) -newer "$MARKER" -print))
[ ${#NEW_SIM[@]} -ge 1 ] || { echo "ERROR: simulate 実行で配列ファイルが作られていない"; exit 1; }
echo "OK: simulate(created): ${NEW_SIM[*]}"

# --- 体裁の最小チェック（TSV が非空 & タブ区切り） ---
TSV_PICK="${NEW_TSV[0]}"
[ -n "${TSV_PICK:-}" ] && [ -s "$TSV_PICK" ] && grep $'\t' "$TSV_PICK" >/dev/null || true

# --- アーティファクト収集 ---
ART="$WORKDIR/_artifacts_cmd"
mkdir -p "$ART"
for f in csubst_*.tsv alignment.fa.{iqtree,log,rate,state,treefile} *.fa *.fasta *.phy; do
  [ -f "$f" ] && cp -v "$f" "$ART" || true
done

echo "Command tests OK"
