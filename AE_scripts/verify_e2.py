#!/usr/bin/env python3
"""Verify the Dual-Hub versus No-Hub clean-test comparison."""

import json
import sys
from pathlib import Path


def fail(message):
    raise SystemExit("E2 FAIL: " + message)


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
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_e2.py E2_OUTPUT_ROOT")

    root = Path(sys.argv[1])
    for scope in ("overall", "long_range"):
        hub = load(root / "hub" / f"{scope}.json")
        nohub = load(root / "nohub" / f"{scope}.json")
        require(hub.get("scope") == nohub.get("scope") == scope, f"{scope}: scope mismatch")
        require(hub.get("task") == nohub.get("task") == "indirectcall", f"{scope}: task mismatch")
        require(hub.get("threshold") == nohub.get("threshold") == 0.5, f"{scope}: threshold mismatch")
        require(
            hub.get("threshold_source") == nohub.get("threshold_source") == "fixed_evaluator",
            f"{scope}: threshold source mismatch",
        )
        hub_clean_root = hub.get("clean_root")
        nohub_clean_root = nohub.get("clean_root")
        require(bool(hub_clean_root) and bool(nohub_clean_root), f"{scope}: clean-test root is absent")
        require(
            Path(hub_clean_root).resolve() == Path(nohub_clean_root).resolve(),
            f"{scope}: clean-test roots differ",
        )
        for label, result in (("Hub", hub), ("No-Hub", nohub)):
            require(result.get("checkpoint_artifact_format") == "icflownet_model_v1", f"{scope}: {label} checkpoint format mismatch")
            require(result.get("precision") == "fp32", f"{scope}: {label} evaluation is not FP32")
            require(result.get("amp_enabled") is False, f"{scope}: {label} AMP must be disabled")
            require(not result.get("cpu_fallback_records"), f"{scope}: {label} used CPU fallback")

        hub_config = hub.get("model_config", {})
        nohub_config = nohub.get("model_config", {})
        require(hub_config.get("use_gch") and hub_config.get("use_gdh"), f"{scope}: Hub config is not Dual-Hub")
        require(not nohub_config.get("use_gch") and not nohub_config.get("use_gdh"), f"{scope}: No-Hub config enables a hub")

        hub_metrics = hub.get("metrics", {})
        nohub_metrics = nohub.get("metrics", {})
        require(
            (hub_metrics.get("support"), hub_metrics.get("pairs"))
            == (nohub_metrics.get("support"), nohub_metrics.get("pairs")),
            f"{scope}: evaluated pair counts differ",
        )
        require(hub_metrics.get("pairs", 0) > 0, f"{scope}: no pairs were evaluated")
        require(hub_metrics.get("f1", 0) > nohub_metrics.get("f1", 0), f"{scope}: Hub F1 is not higher")
        print(
            f"{scope}: Hub F1={hub_metrics['f1']:.6f}, "
            f"No-Hub F1={nohub_metrics['f1']:.6f}"
        )
    print("E2 PASS")


if __name__ == "__main__":
    main()
