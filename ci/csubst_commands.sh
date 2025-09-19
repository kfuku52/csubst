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

# --- site（サイト別計算：最小オプションで可視化前処理を確認） ---
CBFILE="$(ls -1t csubst_cb_*.tsv 2>/dev/null | head -n1 || true)"
if [ -z "${CBFILE}" ]; then
  echo "ERROR: cb テーブルが見つかりません（csubst_cb_*.tsv）"; exit 1
fi

# IQ-TREE 中間を消して site 側の再生成も確認（または再利用させる）
rm -f alignment.fa.{iqtree,log,rate,state,treefile} || true

MARKER=$(mktemp); sleep 1; touch "$MARKER"

# ログを保存しつつ終了コードを保持
set +e
env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 ${MPLBACKEND:+env MPLBACKEND=$MPLBACKEND} csubst site \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --cb_file "$CBFILE" \
  --branch_id fg \
  --threads 1 2>&1 | tee site.log
rc=${PIPESTATUS[0]}
set -e

# fg が通らない場合は cb から最初の枝IDを抽出して再実行
if [ $rc -ne 0 ]; then
  echo "WARN: site --branch_id fg が失敗。cb テーブルから枝IDを抽出して再実行します"
  combo="$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++){if($i ~ /branch_?id|branches/i) col=i}} NR==2 && col{print $col}' "$CBFILE")"
  [ -n "$combo" ] || combo="$(grep -Eho '^[[:space:]]*[0-9]+,[0-9]+' "$CBFILE" | head -n1 | tr -d '[:space:]')"
  [ -n "$combo" ] || { echo "ERROR: cb テーブルから枝IDを取得できませんでした"; exit 1; }

  set +e
  env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 ${MPLBACKEND:+env MPLBACKEND=$MPLBACKEND} csubst site \
    --alignment_file alignment.fa \
    --rooted_tree_file tree.nwk \
    --branch_id "$combo" \
    --threads 1 2>&1 | tee site.log
  rc=${PIPESTATUS[0]}
  set -e
fi

[ $rc -eq 0 ] || { echo "ERROR: csubst site が非0終了"; exit 1; }

# ---- 成功判定：ログ＋中間ファイル（.mmap を必須にしない） ----
grep -E "csubst site end|Generating memory map" site.log >/dev/null || {
  echo "ERROR: site.log に実行完了の痕跡がありません"; exit 1; }

# IQ-TREEの中間が直近でできていること（少なくとも state/rate）
test -s alignment.fa.state && test -s alignment.fa.rate || {
  echo "ERROR: alignment.fa.state / .rate が見つかりません"; exit 1; }
# 実行後に更新されたかを軽く確認（無理ならスキップしてOK）
if [ -n "$(find . -maxdepth 1 -name 'alignment.fa.state' -newer "$MARKER" -print -quit)" ]; then
  echo "OK: site 再計算で state/rate を生成/更新"
else
  echo "NOTE: site は既存の IQ-TREE 中間を再利用した可能性があります"
fi

# もし TSV が出来ていれば記録（必須にしない）
SITE_TSV=($(find . -maxdepth 1 -type f -name "csubst_site*.tsv" -newer "$MARKER" -print))
[ ${#SITE_TSV[@]} -ge 1 ] && echo "NOTE: site TSV: ${SITE_TSV[*]}"

# アーティファクト収集に site.log も追加
ART="$WORKDIR/_artifacts_cmd"
mkdir -p "$ART"
cp -v site.log "$ART" || true

echo "OK: site finished (verified by logs and IQ-TREE intermediates)"

# --- round-trip: simulate の出力を analyze へ ---
MARKER=$(mktemp); sleep 1; touch "$MARKER"

env PYTHONOPTIMIZE=1 csubst simulate \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --threads 1

SIM_ALN="$(find . -maxdepth 1 -type f \( -name "*.fa" -o -name "*.fasta" -o -name "*.phy" \) -newer "$MARKER" | head -n1)"
[ -s "${SIM_ALN:-}" ] || { echo "ERROR: simulate 出力の配列ファイルが見つかりません"; exit 1; }

MARKER2=$(mktemp); sleep 1; touch "$MARKER2"
env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst analyze \
  --alignment_file "$SIM_ALN" \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --threads 1

NEW_TSV2=($(find . -maxdepth 1 -type f -name "csubst_*.tsv" -newer "$MARKER2" -print))
[ ${#NEW_TSV2[@]} -ge 1 ] || { echo "ERROR: simulate→analyze で TSV が生成されていません"; exit 1; }
echo "OK: round-trip simulate→analyze(created): ${NEW_TSV2[*]}"

echo "Command tests OK"

{
  echo "# csubst CI Summary"
  echo
  echo "## Generated files"
  ls -lh | sed 's/^/- /'
} > summary.md

cp -v summary.md "$ART" || true
