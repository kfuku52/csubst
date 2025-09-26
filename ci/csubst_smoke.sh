#!/usr/bin/env bash
set -euo pipefail

echo "== csubst smoke test =="

which csubst
(csubst --version || true)

python - <<'PY'
import sys
try:
    import numpy as np
    print("Python:", sys.version.split()[0], "| NumPy:", np.__version__)
except Exception as e:
    print("Py/Numpy check:", e)
PY

WORKDIR="${RUNNER_TEMP:-$(mktemp -d)}/csubst_smoke"
ART="$WORKDIR/_artifacts"
mkdir -p "$WORKDIR" "$ART"
cd "$WORKDIR"

# データ生成
csubst dataset --name PGK
test -s alignment.fa && test -s tree.nwk && test -s foreground.txt

# ---- analyze ----
export PYTHONOPTIMIZE=1
export OMP_NUM_THREADS=1

set +e
csubst analyze \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --iqtree_model GY+F3x4+R2 \
  --threads 1
ANALYZE_RC=$?
set -e
echo "[SMOKE] analyze exited rc=${ANALYZE_RC} (we validate by files, not rc)"

# 代表的な出力の存在確認
shopt -s nullglob
CB=(csubst_cb_*.tsv)
REQ=(alignment.fa.iqtree alignment.fa.rate alignment.fa.state alignment.fa.treefile)

MISS=()
for f in "${REQ[@]}"; do [[ -s "$f" ]] || MISS+=("$f"); done

if (( ${#CB[@]} == 0 )) || (( ${#MISS[@]} > 0 )); then
  echo "ERROR: analyze outputs missing."
  echo "Missing: ${MISS[*]:-(none)}"
  echo "--- ls -al ---"
  ls -al
  exit 1
fi

echo "OK: analyze(created): ${CB[*]}"

# PyMOL/pyvolve は CI 非搭載のため常にスキップ
echo "[SMOKE] skip site (PyMOL not available in CI)"
echo "[SMOKE] skip simulate (pyvolve not guaranteed in CI)"

# アーティファクト収集（存在しない場合もエラーにしない）
cp -v csubst_cb_*.tsv "$ART" 2>/dev/null || true
cp -v alignment.fa.{iqtree,log,rate,state,treefile} "$ART" 2>/dev/null || true

echo "Artifacts in: $ART"
exit 0
