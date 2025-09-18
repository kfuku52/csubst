#!/usr/bin/env bash
set -euo pipefail

echo "== csubst command tests =="

which csubst
csubst --version || true

# --- ヘルプ確認（存在チェック） ---
csubst -h | grep -E '\bdataset\b' >/dev/null
csubst -h | grep -E '\banalyze\b'  >/dev/null
csubst -h | grep -E '\bsite\b'     >/dev/null
csubst -h | grep -E '\bsimulate\b' >/dev/null
csubst analyze  -h | grep -- '--alignment_file'   >/dev/null
csubst site     -h | grep -- '--alignment_file'   >/dev/null
csubst simulate -h | grep -- '--alignment_file'   >/dev/null

# --- 作業ディレクトリ（smoke と共通） ---
WORKDIR="${RUNNER_TEMP:-$(mktemp -d)}/csubst_smoke"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 最小データが無い場合は生成
if [ ! -s alignment.fa ]; then
  csubst dataset --name PGK
fi
test -s alignment.fa && test -s tree.nwk && test -s foreground.txt

# 便利関数：新規ファイル検出
newer_than() { find . -maxdepth 1 -type f -newer "$1" -printf "%f\n" || true; }

# --- analyze（既定＝ECM系） ---
rm -f alignment.fa.{iqtree,log,rate,state,treefile} || true
MARKER=$(mktemp); sleep 1; touch "$MARKER"
env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst analyze \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --threads 1
# 生成物（TSV）が増えたか
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

# --- site（サイト別計算：最小オプション） ---
rm -f alignment.fa.{iqtree,log,rate,state,treefile} || true
MARKER=$(mktemp); sleep 1; touch "$MARKER"
# ※ site も内部で祖先状態を使うため、念のため assert 無効化を局所適用
env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst site \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --threads 1
NEW_SITE_TSV=($(find . -maxdepth 1 -type f -name "csubst_site*.tsv" -newer "$MARKER" -print))
# ファイル名が将来変わっても拾えるよう、site 実行直後に新規 TSV があるかも見る
[ ${#NEW_SITE_TSV[@]} -ge 1 ] || NEW_SITE_TSV=($(find . -maxdepth 1 -type f -name "*.tsv" -newer "$MARKER" -print))
[ ${#NEW_SITE_TSV[@]} -ge 1 ] || { echo "ERROR: site 実行で TSV が作られていない"; exit 1; }
echo "OK: site(created): ${NEW_SITE_TSV[*]}"

# --- simulate（シミュレーション：デフォルト） ---
MARKER=$(mktemp); sleep 1; touch "$MARKER"
# simulate は多くの場合 IQ-TREE を起動しないので速いはず。安全のため threads=1。
env PYTHONOPTIMIZE=1 csubst simulate \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --threads 1
NEW_SIM=($(find . -maxdepth 1 -type f \( -name "*.fa" -o -name "*.fasta" -o -name "*.phy" \) -newer "$MARKER" -print))
[ ${#NEW_SIM[@]} -ge 1 ] || { echo "ERROR: simulate 実行で配列ファイルが作られていない"; exit 1; }
echo "OK: simulate(created): ${NEW_SIM[*]}"

# --- 体裁の最小チェック（TSV ヘッダ & 非空） ---
TSV_PICK="${NEW_TSV[0]}"
[ -n "${TSV_PICK:-}" ] && [ -s "$TSV_PICK" ] && grep $'\t' "$TSV_PICK" >/dev/null || true

# --- アーティファクト収集 ---
ART="$WORKDIR/_artifacts_cmd"
mkdir -p "$ART"
# 直近の生成物をざっくり集める
for f in csubst_*.tsv alignment.fa.{iqtree,log,rate,state,treefile} *.fa *.fasta *.phy; do
  [ -f "$f" ] && cp -v "$f" "$ART" || true
done

echo "Command tests OK"
