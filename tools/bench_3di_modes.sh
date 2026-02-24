#!/usr/bin/env bash
set -euo pipefail

# Benchmark csubst analyze across:
#   1) no recoding
#   2) 3di20 direct
#   3) 3di20 translate
#
# This script runs the local repository code (not site-packages), writes per-mode logs,
# and prints summary.tsv with wall time and max RSS.
#
# Usage:
#   bash tools/bench_3di_modes.sh [DATA_DIR] [OUT_DIR]
#
# Examples:
#   bash tools/bench_3di_modes.sh
#   bash tools/bench_3di_modes.sh data/GH19_chitinase_tiny /tmp/csubst_bench_run1
#
# Environment overrides:
#   THREADS=12
#   PROSTT5_DEVICE=auto|cpu|cuda|mps
#   PROSTT5_CACHE=yes|no
#   MODES=no,direct,translate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR_INPUT="${1:-$REPO_ROOT/data/GH19_chitinase_tiny}"
OUT_DIR="${2:-/tmp/csubst_speed_bench_$(date +%Y%m%d_%H%M%S)}"

if [[ "$DATA_DIR_INPUT" = /* ]]; then
  DATA_DIR="$DATA_DIR_INPUT"
else
  DATA_DIR="$REPO_ROOT/$DATA_DIR_INPUT"
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Data directory not found: $DATA_DIR" >&2
  exit 2
fi

if [[ -z "${THREADS:-}" ]]; then
  THREADS="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 1)"
fi
PROSTT5_DEVICE="${PROSTT5_DEVICE:-auto}"
PROSTT5_CACHE="${PROSTT5_CACHE:-yes}"
MODES="${MODES:-no,direct,translate}"
IFS=',' read -r -a MODE_LIST <<< "$MODES"

mkdir -p "$OUT_DIR"

cat > "$OUT_DIR/run_local_csubst.py" <<PY
import runpy
import sys
ROOT = r"$REPO_ROOT"
while ROOT in sys.path:
    sys.path.remove(ROOT)
sys.path.insert(0, ROOT)
ns = runpy.run_path(ROOT + '/csubst/csubst', run_name='csubst_cli')
sys.argv[0] = 'csubst'
ns['_main']()
PY

link_common_inputs() {
  local work_dir="$1"
  ln -sf "$DATA_DIR/alignment.fa" "$work_dir/alignment.fa"
  ln -sf "$DATA_DIR/tree.nwk" "$work_dir/tree.nwk"
  ln -sf "$DATA_DIR/foreground.txt" "$work_dir/foreground.txt"
  ln -sf "$DATA_DIR/alignment.fa.treefile" "$work_dir/alignment.fa.treefile"
  ln -sf "$DATA_DIR/alignment.fa.state" "$work_dir/alignment.fa.state"
  ln -sf "$DATA_DIR/alignment.fa.rate" "$work_dir/alignment.fa.rate"
  ln -sf "$DATA_DIR/alignment.fa.iqtree" "$work_dir/alignment.fa.iqtree"
  ln -sf "$DATA_DIR/alignment.fa.log" "$work_dir/alignment.fa.log"
}

extract_time_metrics() {
  local stderr_path="$1"
  python - <<PY
import re
txt = open(r"$stderr_path", encoding="utf-8", errors="replace").read()
patterns = {
    "real_sec": r"([0-9.]+) real",
    "user_sec": r"([0-9.]+) user",
    "sys_sec": r"([0-9.]+) sys",
    "maxrss_bytes": r"\\s*([0-9]+)\\s+maximum resident set size",
}
for key, pat in patterns.items():
    m = re.search(pat, txt)
    print(f"{key}\\t{m.group(1) if m else ''}")
PY
}

extract_elapsed_lines() {
  local stdout_path="$1"
  python - <<PY
for line in open(r"$stdout_path", encoding="utf-8", errors="replace"):
    if "Elapsed time:" in line:
        print(line.strip())
PY
}

run_mode() {
  local mode="$1"
  local work_dir="$OUT_DIR/$mode"
  mkdir -p "$work_dir"
  link_common_inputs "$work_dir"
  pushd "$work_dir" >/dev/null

  local args=(
    analyze
    --alignment_file alignment.fa
    --rooted_tree_file tree.nwk
    --foreground foreground.txt
    --threads "$THREADS"
    --iqtree_redo no
    --iqtree_treefile alignment.fa.treefile
    --iqtree_state alignment.fa.state
    --iqtree_rate alignment.fa.rate
    --iqtree_iqtree alignment.fa.iqtree
    --iqtree_log alignment.fa.log
    --drop_invariant_tip_sites tip_invariant
  )
  if [[ "$mode" == "direct" || "$mode" == "translate" ]]; then
    args+=(
      --nonsyn_recode 3di20
      --sa_asr_mode "$mode"
      --full_cds_alignment_file alignment.fa
      --prostt5_device "$PROSTT5_DEVICE"
      --prostt5_cache "$PROSTT5_CACHE"
    )
  fi
  if [[ "$mode" == "direct" ]]; then
    args+=(--sa_iqtree_model GTR)
  fi

  set +e
  /usr/bin/time -l python "$OUT_DIR/run_local_csubst.py" "${args[@]}" > stdout.log 2> time_stderr.log
  local rc=$?
  set -e
  echo "$rc" > exit_code.txt

  extract_time_metrics "time_stderr.log" > time.tsv
  extract_elapsed_lines "stdout.log" > elapsed_lines.txt

  popd >/dev/null
}

for mode in "${MODE_LIST[@]}"; do
  mode="$(echo "$mode" | tr -d '[:space:]')"
  if [[ -z "$mode" ]]; then
    continue
  fi
  if [[ "$mode" != "no" && "$mode" != "direct" && "$mode" != "translate" ]]; then
    echo "Unsupported mode in MODES: $mode" >&2
    exit 2
  fi
  run_mode "$mode"
done

python - <<PY > "$OUT_DIR/summary.tsv"
import os
from pathlib import Path

out = Path(r"$OUT_DIR")
modes = [m.strip() for m in r"$MODES".split(",") if m.strip()]
print("mode\\texit_code\\treal_sec\\tuser_sec\\tsys_sec\\tmaxrss_bytes\\tfirst_elapsed_line")
for mode in modes:
    mode_dir = out / mode
    rc = (mode_dir / "exit_code.txt").read_text(encoding="utf-8").strip() if (mode_dir / "exit_code.txt").exists() else ""
    metrics = {}
    if (mode_dir / "time.tsv").exists():
        for line in (mode_dir / "time.tsv").read_text(encoding="utf-8", errors="replace").splitlines():
            if "\\t" in line:
                k, v = line.split("\\t", 1)
                metrics[k] = v
    first_elapsed = ""
    if (mode_dir / "elapsed_lines.txt").exists():
        lines = [s.strip() for s in (mode_dir / "elapsed_lines.txt").read_text(encoding="utf-8", errors="replace").splitlines() if s.strip()]
        if lines:
            first_elapsed = lines[0]
    print("\\t".join([
        mode,
        rc,
        metrics.get("real_sec", ""),
        metrics.get("user_sec", ""),
        metrics.get("sys_sec", ""),
        metrics.get("maxrss_bytes", ""),
        first_elapsed,
    ]))
PY

echo "OUT_DIR=$OUT_DIR"
cat "$OUT_DIR/summary.tsv"
