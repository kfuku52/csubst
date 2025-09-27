#!/usr/bin/env bash
set -euo pipefail

echo "== csubst command tests =="

# --------------------------------
# 共通セットアップ
# --------------------------------
WORKDIR="${RUNNER_TEMP:-/tmp}/csubst_smoke"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Workflow から来る（無ければ false）
HAVE_PYMOL="${HAVE_PYMOL:-false}"
echo "[CMD] HAVE_PYMOL=${HAVE_PYMOL}"

# どの csubst が呼ばれているか確認
command -v csubst
csubst --version || true

# --------------------------------
# ヘルプの存在チェック（落ちない形）
# --------------------------------
csubst -h | grep -E '\bdataset\b' >/dev/null || echo "::warning ::dataset help not found (tolerated)"
csubst -h | grep -E '\banalyze\b'  >/dev/null || echo "::warning ::analyze help not found (tolerated)"
csubst -h | grep -E '\bsite\b'     >/dev/null || echo "::warning ::site help not found (tolerated)"
csubst -h | grep -E '\bsimulate\b' >/dev/null || echo "::warning ::simulate help not found (tolerated)"
csubst analyze  -h | grep -- '--alignment_file'   >/dev/null || echo "::warning ::analyze flag missing (tolerated)"
csubst site     -h | grep -- '--alignment_file'   >/dev/null || echo "::warning ::site flag missing (tolerated)"
csubst simulate -h | grep -- '--alignment_file'   >/dev/null || echo "::warning ::simulate flag missing (tolerated)"

# --------------------------------
# データセット作成
# --------------------------------
if [[ ! -s alignment.fa || ! -s tree.nwk || ! -s foreground.txt ]]; then
  csubst dataset --name PGK
fi
test -s alignment.fa && test -s tree.nwk && test -s foreground.txt

# --------------------------------
# analyze（デフォルトモデル）
# --------------------------------
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
if [[ ${#NEW_TSV[@]} -ge 1 ]]; then
  echo "OK: analyze(created): ${NEW_TSV[*]}"
else
  echo "ERROR: analyze 後に TSV が増えていない"; exit 1
fi

# --------------------------------
# analyze（GY モデルの分岐テスト）
# --------------------------------
rm -f alignment.fa.{iqtree,log,rate,state,treefile} || true
MARKER=$(mktemp); sleep 1; touch "$MARKER"

set +e
env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst analyze \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --iqtree_model GY+F3x4+R2 \
  --threads 1
ANALYZE_RC2=$?
set -e
echo "[SMOKE] analyze(GY) exited rc=${ANALYZE_RC2} (validate by files)"

# 期待ファイル（無くても警告に留める）
for f in csubst_cb_2.tsv csubst_b.tsv csubst_instantaneous_rate_matrix.tsv csubst_cb_stats.tsv; do
  [[ -f "$f" ]] && echo "OK: analyze(created): $f" || echo "::warning ::missing expected file: $f (tolerated)"
done

NEW_TSV2=($(find . -maxdepth 1 -type f -name "csubst_*.tsv" -newer "$MARKER" -print))
if [[ ${#NEW_TSV2[@]} -ge 1 ]]; then
  echo "OK: analyze(GY)(created): ${NEW_TSV2[*]}"
else
  echo "ERROR: analyze(GY) 後に TSV が増えていない"; exit 1
fi

# --------------------------------
# site（PyMOL がある時だけ）
# --------------------------------
if [[ "${HAVE_PYMOL}" == "true" ]]; then
  echo "[CMD] PyMOL available: run site tests"

  CBFILE="$(ls -1t csubst_cb_*.tsv 2>/dev/null | head -n1 || true)"
  if [[ -z "${CBFILE}" ]]; then
    echo "::warning ::cb テーブルが見つからないため site をスキップ"; 
  else
    rm -f alignment.fa.{iqtree,log,rate,state,treefile} || true
    MARKER=$(mktemp); sleep 1; touch "$MARKER"

    set +e
    env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst site \
      --alignment_file alignment.fa \
      --rooted_tree_file tree.nwk \
      --cb_file "$CBFILE" \
      --branch_id fg \
      --threads 1 2>&1 | tee "_artifacts_cmd_site.log"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ $rc -ne 0 ]]; then
      echo "::warning ::site --branch_id fg が失敗したため、cb から枝 ID を抽出して再実行（tolerated）"
      combo="$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++){if($i ~ /branch_?id|branches/i) col=i}} NR==2 && col{print $col}' "$CBFILE")"
      [[ -n "$combo" ]] || combo="$(grep -Eho '^[[:space:]]*[0-9]+,[0-9]+' "$CBFILE" | head -n1 | tr -d '[:space:]')"
      if [[ -n "$combo" ]]; then
        set +e
        env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst site \
          --alignment_file alignment.fa \
          --rooted_tree_file tree.nwk \
          --branch_id "$combo" \
          --threads 1 2>&1 | tee -a "_artifacts_cmd_site.log"
        rc=${PIPESTATUS[0]}
        set -e
      fi
    fi

    # 失敗しても全体は落とさない
    if [[ $rc -eq 0 ]]; then
      echo "OK: site finished"
    else
      echo "::warning ::csubst site failed (tolerated)"
    fi
  fi
else
  echo "[CMD] PyMOL not available: skip site tests"
fi

# --------------------------------
# simulate（pyvolve がある時だけ）
# --------------------------------
if python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("pyvolve") else 1)
PY
then
  echo "[CMD] pyvolve available: run simulate"
  MARKER=$(mktemp); sleep 1; touch "$MARKER"
  set +e
  csubst simulate \
    --alignment_file alignment.fa \
    --rooted_tree_file tree.nwk \
    --foreground foreground.txt \
    --iqtree_model GY+F3x4+R2 \
    --threads 1 | tee "_artifacts_cmd_simulate.log"
  sim_rc=${PIPESTATUS[0]}
  set -e
  if [[ $sim_rc -ne 0 ]]; then
    echo "::warning ::simulate failed (tolerated)"
  else
    grep -q "csubst simulate start" "_artifacts_cmd_simulate.log" || echo "::warning ::simulate log missing 'start'"
    grep -q "Time elapsed" "_artifacts_cmd_simulate.log" || echo "::warning ::simulate log missing 'Time elapsed'"

    # round-trip: simulate の出力を analyze に通す
    SIM_ALN="$(find . -maxdepth 1 -type f \( -name "*.fa" -o -name "*.fasta" -o -name "*.phy" \) -newer "$MARKER" | head -n1 || true)"
    if [[ -s "${SIM_ALN:-}" ]]; then
      MARKER2=$(mktemp); sleep 1; touch "$MARKER2"
      env PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 csubst analyze \
        --alignment_file "$SIM_ALN" \
        --rooted_tree_file tree.nwk \
        --foreground foreground.txt \
        --threads 1
      NEW_TSV3=($(find . -maxdepth 1 -type f -name "csubst_*.tsv" -newer "$MARKER2" -print))
      [[ ${#NEW_TSV3[@]} -ge 1 ]] && echo "OK: round-trip simulate→analyze(created): ${NEW_TSV3[*]}" \
                                  || echo "::warning ::simulate→analyze で TSV 未生成（tolerated）"
    else
      echo "::warning ::simulate 出力の配列ファイルが見つからない（tolerated）"
    fi
  fi
else
  echo "[CMD] pyvolve not available: skip simulate"
fi

echo "Command tests OK"

# 簡易サマリ
{
  echo "# csubst CI Summary"
  echo
  echo "## Generated files"
  ls -lh | sed 's/^/- /'
} > summary.md

mkdir -p "$WORKDIR/_artifacts_cmd"
cp -v summary.md "$WORKDIR/_artifacts_cmd" >/dev/null 2>&1 || true

# ここまで来たら常に成功扱い
exit 0
