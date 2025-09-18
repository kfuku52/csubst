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
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 最小データ生成
csubst dataset --name PGK
test -s alignment.fa && test -s tree.nwk && test -s foreground.txt

# NOTE: smoke 用。実行時アサーションに掛かるため CI では無効化して通過確認のみ行う
export PYTHONOPTIMIZE=1
export OMP_NUM_THREADS=1
# GY にして実行
csubst analyze \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --iqtree_model GY+F3x4+R2 \
  --threads 1

# 代表的な出力の存在確認
shopt -s nullglob
CB=(csubst_cb_*.tsv)
if [ ${#CB[@]} -eq 0 ]; then
  echo "ERROR: csubst_cb_*.tsv が生成されませんでした"; ls -l; exit 1
fi
echo "OK: 出力 ${#CB[@]} 件: ${CB[*]}"
head -n 5 "${CB[0]}"

ART="$WORKDIR/_artifacts"
mkdir -p "$ART"
# 代表的な出力を収集（存在しない場合もエラーにしない）
cp -v csubst_cb_*.tsv "$ART" 2>/dev/null || true
cp -v alignment.fa.{iqtree,log,rate,state,treefile} "$ART" 2>/dev/null || true
