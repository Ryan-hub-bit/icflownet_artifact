#!/usr/bin/env python3
"""Verify MTL versus Dual-Hub single-task indirect-call performance."""

import json
import sys
from pathlib import Path


def fail(message):
    raise SystemExit("E3 FAIL: " + message)


def require(condition, message):
    if not condition:
        fail(message)


def load(path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_e3.py E2_OUTPUT_ROOT E3_OUTPUT_ROOT")

    e2_root = Path(sys.argv[1])
    e3_root = Path(sys.argv[2])
    for scope in ("overall", "long_range"):
        single = load(e2_root / "hub" / f"{scope}.json")
        mtl = load(e3_root / "mtl" / scope / "result.json")
        require(single.get("scope") == mtl.get("scope") == scope, f"{scope}: scope mismatch")
        require(single.get("task") == "indirectcall", f"{scope}: E2 result is not indirect-call")
        require(single.get("threshold") == mtl.get("threshold") == 0.5, f"{scope}: threshold mismatch")
        require(single.get("threshold_source") == "fixed_evaluator", f"{scope}: single-task threshold source mismatch")
        single_clean_root = single.get("clean_root")
        mtl_clean_root = mtl.get("clean_root")
        require(bool(single_clean_root) and bool(mtl_clean_root), f"{scope}: clean-test root is absent")
        require(
            Path(single_clean_root).resolve() == Path(mtl_clean_root).resolve(),
            f"{scope}: clean-test roots differ",
        )
        for label, result in (("Single", single), ("MTL", mtl)):
            require(result.get("precision") == "fp32", f"{scope}: {label} evaluation is not FP32")
            require(result.get("amp_enabled") is False, f"{scope}: {label} AMP must be disabled")
            require(result.get("checkpoint_artifact_format") == "icflownet_model_v1", f"{scope}: {label} checkpoint format mismatch")
        require(not single.get("cpu_fallback_records"), f"{scope}: single-task evaluation used CPU fallback")
        require(mtl.get("world_size") == 2, f"{scope}: MTL evaluation did not use two processes")

        config = mtl.get("model_config", {})
        require(config.get("use_gch") and config.get("use_gdh"), f"{scope}: MTL config is not Dual-Hub")
        require("indirectcall" in config.get("task_weights", {}), f"{scope}: MTL indirect-call head is absent")

        single_metrics = single.get("metrics", {})
        mtl_metrics = mtl.get("metrics", {}).get("task_metrics", {}).get("indirectcall", {})
        require(
            (single_metrics.get("support"), single_metrics.get("pairs"))
            == (mtl_metrics.get("support"), mtl_metrics.get("pairs")),
            f"{scope}: evaluated indirect-call pair counts differ",
        )
        require(single_metrics.get("pairs", 0) > 0, f"{scope}: no indirect-call pairs were evaluated")
        require(mtl_metrics.get("f1", 0) > single_metrics.get("f1", 0), f"{scope}: MTL F1 is not higher")
        print(
            f"{scope}: MTL F1={mtl_metrics['f1']:.6f}, "
            f"Single F1={single_metrics['f1']:.6f}"
        )
    print("E3 PASS")


if __name__ == "__main__":
    main()
