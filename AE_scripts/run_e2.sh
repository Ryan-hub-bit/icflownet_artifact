#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the extracted ICFlowNet AE package}"

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_ROOT/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs}"
GPU="${GPU:-0}"
SAMPLE_ROOT="$DATA_ROOT/train_sample"
MODEL_ROOT="$DATA_ROOT/model"
CLEAN_ROOT="$DATA_ROOT/cleantest"
E2_ROOT="$OUTPUT_ROOT/e2"
TRAIN_ROOT="$E2_ROOT/train-smoke"

python "$SCRIPT_ROOT/check_setup.py" "$DATA_ROOT" --skip-checksums
mkdir -p "$E2_ROOT/hub" "$E2_ROOT/nohub"

echo "[E2 1/3] One-epoch Dual-Hub training smoke test"
cd "$SAMPLE_ROOT"
python "$REPO_ROOT/src/train_single.py" \
  --graph_dir "$SAMPLE_ROOT/graph" \
  --output_dir "$TRAIN_ROOT" \
  --target_edge_types indirectcall \
  --train_all_task_graphs \
  --skip_validation \
  --skip_duplicate_filtering \
  --epochs 1 \
  --gpu "$GPU" \
  --use_gch \
  --use_gdh

test -s "$TRAIN_ROOT/indirectcall/best_model.pt"
echo "E2 training smoke PASS"

echo "[E2 2/3] FP32 clean-test evaluation of Hub and No-Hub checkpoints"
cd "$REPO_ROOT"
python scripts/eval_single_cleantest.py \
  --checkpoint "$MODEL_ROOT/ic_hub_model.pt" \
  --clean_root "$CLEAN_ROOT" \
  --task indirectcall \
  --scope both \
  --output "$E2_ROOT/hub" \
  --gpu "$GPU" \
  --disable_amp

python scripts/eval_single_cleantest.py \
  --checkpoint "$MODEL_ROOT/ic_nohub_model.pt" \
  --clean_root "$CLEAN_ROOT" \
  --task indirectcall \
  --scope both \
  --output "$E2_ROOT/nohub" \
  --gpu "$GPU" \
  --disable_amp

echo "[E2 3/3] Verify claim C2"
python "$SCRIPT_ROOT/verify_e2.py" "$E2_ROOT"
