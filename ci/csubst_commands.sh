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

# IQ-TREE 中間を消して site 側の再生成も確認
rm -f alignment.fa.{iqtree,log,rate,state,treefile} || true

MARKER=$(mktemp); sleep 1; touch "$MARKER"

set +e
env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 ${MPLBACKEND:+env MPLBACKEND=$MPLBACKEND} csubst site \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --cb_file "$CBFILE" \
  --branch_id fg \
  --threads 1
rc=$?
set -e

# fg が通らない時は cb テーブルから枝IDを抽出して再実行（念のため）
if [ $rc -ne 0 ]; then
  echo "WARN: site --branch_id fg が失敗。cb テーブルから枝IDを抽出して再実行します"
  combo="$(awk -F'\t' '
    NR==1{
      for(i=1;i<=NF;i++){if($i ~ /branch_?id/i || $i ~ /branches/i){col=i}}
    }
    NR==2 && col {print $col}
  ' "$CBFILE")"
  if [ -z "$combo" ]; then
    combo="$(grep -Eho '^[[:space:]]*[0-9]+,[0-9]+' "$CBFILE" | head -n1 | tr -d '[:space:]')"
  fi
  [ -n "$combo" ] || { echo "ERROR: cb テーブルから枝IDを取得できませんでした"; exit 1; }

  env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 ${MPLBACKEND:+env MPLBACKEND=$MPLBACKEND} csubst site \
    --alignment_file alignment.fa \
    --rooted_tree_file tree.nwk \
    --branch_id "$combo" \
    --threads 1
fi

# ---- 成功判定：TSV は必須にしない ----
# 1) IQ-TREE の中間が生成されていること
test -s alignment.fa.state
test -s alignment.fa.rate

# 2) サイトテンソルのメモリマップが生成されていること（N/S どちらかで可）
MMAPS=($(ls -1 tmp.csubst.sub_tensor.*.mmap 2>/dev/null || true))
[ ${#MMAPS[@]} -ge 1 ] || { echo "ERROR: site 実行で .mmap が見つかりません"; exit 1; }

# 3) もし TSV が出来ていればログに出す（必須ではない）
SITE_TSV=($(find . -maxdepth 1 -type f -name "csubst_site*.tsv" -newer "$MARKER" -print))
[ ${#SITE_TSV[@]} -ge 1 ] && echo "NOTE: site TSV: ${SITE_TSV[*]}"

echo "OK: site generated IQ-TREE intermediates and site tensor mmap(s): ${MMAPS[*]}"

echo "Command tests OK"
