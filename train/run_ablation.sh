#!/usr/bin/env bash
# run_ablation.sh — one arm/seed/seq cell of the FORGE ablation.
#
# Runs on sol (host), inside the unsloth-intel image with the Arc Pro B50.
# The HF cache is mounted so the model is not re-downloaded every cell.
#
# usage: run_ablation.sh <arm> <seed> <maxlen> <steps> <rows> [lr]
#   arm    : v6-sft | harvested | mixed   (must exist as /tmp/data/<arm>.jsonl
#            with a sibling <arm>.gate.json, or training refuses)
#   seed   : 42 | 1337 | 2026
#
# Writes /tmp/runs/<arm>-s<seed>-L<maxlen>/ with the adapter + result.json
set -euo pipefail

ARM=${1:?arm}
SEED=${2:?seed}
MAXLEN=${3:?maxlen}
STEPS=${4:?steps}
ROWS=${5:?rows}
LR=${6:-2e-4}

OUT="/tmp/runs/${ARM}-s${SEED}-L${MAXLEN}"
NAME="abl-${ARM}-s${SEED}-L${MAXLEN}"

if [ ! -f "/tmp/data/${ARM}.jsonl" ]; then
  echo "MISSING data: /tmp/data/${ARM}.jsonl" >&2; exit 1
fi

rm -rf "$OUT"; mkdir -p "$OUT"

echo "[run] arm=$ARM seed=$SEED maxlen=$MAXLEN steps=$STEPS rows=$ROWS lr=$LR"
docker run --rm --name "$NAME" \
  --device /dev/dri \
  -e HF_HOME=/hf \
  -e PYTHONIOENCODING=utf-8 \
  -v /home/zcoder/hf-cache:/hf \
  -v /tmp/data:/data:ro \
  -v /tmp/forge:/forge:ro \
  -v "$OUT":/out \
  unsloth-intel:local python /forge/tools/tm/train_arc.py \
    --data "/data/${ARM}.jsonl" \
    --out /out \
    --steps "$STEPS" \
    --rows "$ROWS" \
    --max-len "$MAXLEN" \
    --lr "$LR" \
    --seed "$SEED" \
  > "$OUT/train.log" 2>&1

# docker runs as root; make the artifacts readable/pullable
chmod -R a+rX "$OUT" 2>/dev/null || true

echo "[run] done -> $OUT"
cat "$OUT/result.json" 2>/dev/null || echo "[run] NO result.json"
