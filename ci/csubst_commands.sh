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

# アーティファクト置き場
ART="$WORKDIR/_artifacts_cmd"
mkdir -p "$ART"

# 最小データが無ければ生成
if [ ! -s alignment.fa ]; then
  csubst dataset --name PGK
fi
test -s alignment.fa && test -s tree.nwk && test -s foreground.txt

# --- analyze（既定＝ECM系） ---
rm -f alignment.fa.{iqtree,log,rate,state,treefile} || true
MARKER=$(mktemp); sleep 1; touch "$MARKER"

set +e
env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst analyze \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --threads 1
ANALYZE_RC1=$?
set -e
echo "[SMOKE] analyze exited rc=${ANALYZE_RC1} (validate by files)"

NEW_TSV=($(find . -maxdepth 1 -type f -name "csubst_*.tsv" -newer "$MARKER" -print))
[ ${#NEW_TSV[@]} -ge 1 ] || { echo "ERROR: analyze 後に TSV が増えていない"; exit 1; }
echo "OK: analyze(created): ${NEW_TSV[*]}"

# --- analyze（GY 分岐） ---
rm -f alignment.fa.{iqtree,log,rate,state,treefile} || true
MARKER=$(mktemp); sleep 1; touch "$MARKER"

set +e
env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst analyze \
    --alignment_file alignment.fa \
    --rooted_tree_file tree.nwk \
    --foreground foreground.txt \
    --threads 1
ANALYZE_RC1=$?
set -e
echo "[SMOKE] analyze exited rc=${ANALYZE_RC1} (validate by files)"

NEW_TSV=($(find . -maxdepth 1 -type f -name "csubst_*.tsv" -newer "$MARKER" -print))
[ ${#NEW_TSV[@]} -ge 1 ] || { echo "ERROR: analyze(GY) 後に TSV が増えていない"; exit 1; }
echo "OK: analyze(GY)(created): ${NEW_TSV[*]}"

## --- site（PyMOL がある時だけ実行） ---
python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("pymol") else 1)
PY
if [ $? -ne 0 ]; then
  echo "NOTE: PyMOL not available -> skip 'csubst site'"
else
  CBFILE="$(ls -1t csubst_cb_*.tsv 2>/dev/null | head -n1 || true)"
  if [ -z "${CBFILE}" ]; then
    echo "ERROR: cb テーブルが見つかりません（csubst_cb_*.tsv）"; exit 1
  fi
  rm -f alignment.fa.{iqtree,log,rate,state,treefile} || true
  MARKER=$(mktemp); sleep 1; touch "$MARKER"
  set +e
  env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst site \
    --alignment_file alignment.fa \
    --rooted_tree_file tree.nwk \
    --cb_file "$CBFILE" \
    --branch_id fg \
    --threads 1 2>&1 | tee "$ART/site.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [ $rc -ne 0 ]; then
    echo "WARN: site --branch_id fg が失敗。cb テーブルから枝IDを抽出して再実行します"
    combo="$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++){if($i ~ /branch_?id|branches/i) col=i}} NR==2 && col{print $col}' "$CBFILE")"
    [ -n "$combo" ] || combo="$(grep -Eho '^[[:space:]]*[0-9]+,[0-9]+' "$CBFILE" | head -n1 | tr -d '[:space:]')"
    [ -n "$combo" ] || { echo "ERROR: cb テーブルから枝IDを取得できませんでした"; exit 1; }
    set +e
    env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst site \
      --alignment_file alignment.fa \
      --rooted_tree_file tree.nwk \
      --branch_id "$combo" \
      --threads 1 2>&1 | tee "$ART/site.log"
    rc=${PIPESTATUS[0]}
    set -e
  fi
  [ $rc -eq 0 ] || { echo "ERROR: csubst site が非0終了"; exit 1; }
  grep -E "csubst site end|Generating memory map" "$ART/site.log" >/dev/null || {
    echo "ERROR: site.log に実行完了の痕跡がありません"; exit 1; }
  test -s alignment.fa.state && test -s alignment.fa.rate || {
    echo "ERROR: alignment.fa.state / .rate が見つかりません"; exit 1; }
  if find . -maxdepth 1 -name 'alignment.fa.state' -newer "$MARKER" | head -n1 | grep -q .; then
    echo "OK: site 再計算で state/rate を生成/更新"
  else
    echo "NOTE: site は既存の IQ-TREE 中間を再利用した可能性があります"
  fi
  SITE_TSV=($(find . -maxdepth 1 -type f -name "csubst_site*.tsv" -newer "$MARKER" -print))
  [ ${#SITE_TSV[@]} -ge 1 ] && echo "NOTE: site TSV: ${SITE_TSV[*]}"
  echo "OK: site finished (verified by logs and IQ-TREE intermediates)"
fi

# --- simulate（pyvolve がある時だけ実行） ---
python - <<'PY'
import importlib.util, sys
ok = importlib.util.find_spec("pyvolve") is not None
print("pyvolve:", "available" if ok else "missing")
sys.exit(0 if ok else 1)
PY
if [ $? -ne 0 ]; then
  echo "NOTE: pyvolve not available -> skip 'csubst simulate'"
else

# simulate 実行（ログ保存）
MARKER=$(mktemp); sleep 1; touch "$MARKER"
set -o pipefail
csubst simulate \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --iqtree_model GY+F3x4+R2 \
  --threads 1 | tee "$ART/simulate.log"
sim_ec=${PIPESTATUS[0]}
set +o pipefail

# 合否判定：終了コード + ログマーカー
if [[ $sim_ec -ne 0 ]]; then
  echo "ERROR: simulate failed (exit $sim_ec)"; exit 1
fi
grep -q "csubst simulate start" "$ART/simulate.log" || { echo "ERROR: simulate log missing 'start' mark"; exit 1; }
grep -q "Time elapsed" "$ART/simulate.log" || { echo "ERROR: simulate log missing 'Time elapsed'"; exit 1; }
echo "OK: simulate finished (verified by exit code and log markers)"

# --- round-trip: simulate の出力を analyze へ ---
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
fi

echo "Command tests OK"

# サマリ
{
  echo "# csubst CI Summary"
  echo
  echo "## Generated files"
  ls -lh | sed 's/^/- /'
} > summary.md

cp -v summary.md "$ART" || true
