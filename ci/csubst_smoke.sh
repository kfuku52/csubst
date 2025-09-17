#!/usr/bin/env bash
set -euo pipefail

echo "== csubst smoke test =="

# どの csubst を使っているか表示
which csubst
(csubst --version || true)

# ヘルプが出れば CLI は生きている
csubst -h | head -n 20

# 作業用ディレクトリ
WORKDIR="${RUNNER_TEMP:-$(mktemp -d)}/csubst_smoke"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 最小データ生成（PGK）
# README / Wiki に記載の手順そのまま
csubst dataset --name PGK

# 入力ができているか軽く確認
test -s alignment.fa
test -s tree.nwk
test -s foreground.txt

# スレッドは 1 にして最小限で実行（CI 安定化）
export OMP_NUM_THREADS=1
csubst analyze \
  --alignment_file alignment.fa \
  --rooted_tree_file tree.nwk \
  --foreground foreground.txt \
  --threads 1

# 代表的な出力（cb テーブル）ができたか確認
shopt -s nullglob
CB=(csubst_cb_*.tsv)
if [ ${#CB[@]} -eq 0 ]; then
  echo "ERROR: csubst_cb_*.tsv が生成されませんでした"; ls -l; exit 1
fi

echo "OK: 出力 ${#CB[@]} 件: ${CB[*]}"
head -n 5 "${CB[0]}"

# 成果物をワークフローのアーティファクトに載せるために残す
pwd
