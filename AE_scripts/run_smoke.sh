#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the extracted ICFlowNet AE package}"

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_ROOT/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs}"
SMOKE_ROOT="$OUTPUT_ROOT/smoke"

python "$SCRIPT_ROOT/check_setup.py" "$DATA_ROOT"
mkdir -p "$SMOKE_ROOT"

torchrun --standalone --nproc_per_node=2 \
  "$REPO_ROOT/scripts/eval_cleantest_ddp.py" \
  --checkpoint "$DATA_ROOT/model/mtl_model.pt" \
  --clean_root "$DATA_ROOT/cleantest" \
  --scope overall \
  --max_records 4 \
  --output_dir "$SMOKE_ROOT" \
  --disable_amp

python - "$SMOKE_ROOT/result.json" <<'PY'
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
result = json.loads(result_path.read_text(encoding="utf-8"))
if result.get("precision") != "fp32" or result.get("amp_enabled") is not False:
    raise SystemExit("Basic evaluation FAIL: result is not FP32")
if result.get("world_size") != 2:
    raise SystemExit("Basic evaluation FAIL: evaluator did not use two processes")
if result.get("metrics", {}).get("evaluated_binaries") != 4:
    raise SystemExit("Basic evaluation FAIL: expected four evaluated binaries")
print("Basic evaluation PASS")
PY
