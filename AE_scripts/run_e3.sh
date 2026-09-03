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
E3_ROOT="$OUTPUT_ROOT/e3"
TRAIN_ROOT="$E3_ROOT/train-smoke"

python "$SCRIPT_ROOT/check_setup.py" "$DATA_ROOT" --skip-checksums
for result in "$E2_ROOT/hub/overall.json" "$E2_ROOT/hub/long_range.json"; do
  if [[ ! -f "$result" ]]; then
    echo "E3 ERROR: missing $result; run E2 first" >&2
    exit 1
  fi
done
mkdir -p "$E3_ROOT"

echo "[E3 1/3] One-epoch four-task MTL training smoke test"
cd "$SAMPLE_ROOT"
python "$REPO_ROOT/src/train_multi.py" \
  --graph_dir "$SAMPLE_ROOT/graph" \
  --output_dir "$TRAIN_ROOT" \
  --target_edge_types ret jumptable indirectcall tailcall \
  --train_all_task_graphs \
  --skip_validation \
  --skip_duplicate_filtering \
  --epochs 1 \
  --gpu "$GPU"

if ! find "$TRAIN_ROOT" -name best_model_trial_preset.pt -print -quit | grep -q .; then
  echo "E3 ERROR: smoke-training checkpoint was not created" >&2
  exit 1
fi
echo "E3 training smoke PASS"

echo "[E3 2/3] Two-GPU FP32 clean-test evaluation of the MTL checkpoint"
cd "$REPO_ROOT"
for scope in overall long_range; do
  output_dir="$E3_ROOT/mtl/$scope"
  mkdir -p "$output_dir"
  torchrun --standalone --nproc_per_node=2 \
    scripts/eval_cleantest_ddp.py \
    --checkpoint "$MODEL_ROOT/mtl_model.pt" \
    --clean_root "$CLEAN_ROOT" \
    --scope "$scope" \
    --output_dir "$output_dir" \
    --disable_amp
done

echo "[E3 3/3] Verify claim C3"
python "$SCRIPT_ROOT/verify_e3.py" "$E2_ROOT" "$E3_ROOT"
