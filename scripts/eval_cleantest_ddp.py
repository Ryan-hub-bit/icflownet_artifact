#!/usr/bin/env python3
"""Evaluate a multi-task checkpoint on a clean-test pair scope."""

import argparse
import json
import os
import sys
from types import SimpleNamespace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import torch

from multi_task_model import MultiTaskGAT
from clean_eval_utils import (
    cleanup_distributed,
    distributed_setup,
    evaluate_clean_full,
    prepare_clean_records,
    seed_everything,
)


def load_checkpoint(path, device):
    artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise RuntimeError("Expected a versioned model artifact dictionary.")
    if artifact.get("artifact_format") != "icflownet_model_v1":
        raise RuntimeError("Unsupported model artifact format: {}".format(
            artifact.get("artifact_format")
        ))
    model_config = artifact.get("model_config")
    if not isinstance(model_config, dict):
        raise RuntimeError("Checkpoint does not contain model_config.")
    model_config = dict(model_config)
    model_config.pop("architecture_version", None)
    if int(model_config.get("gdh_radius", -1)) != 2:
        raise RuntimeError("Checkpoint GDH radius is not 2.")
    model = MultiTaskGAT(**model_config)
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, model_config, artifact.get("artifact_format")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clean_root", required=True)
    parser.add_argument("--scope", choices=("overall", "long_range"), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument(
        "--enable_amp",
        action="store_true",
        help=(
            "Explicitly opt into FP16 autocast. FP32 is the default because "
            "FP16 is numerically unsafe for the MTL Ret branch."
        ),
    )
    precision_group.add_argument(
        "--disable_amp",
        action="store_true",
        help="Deprecated compatibility flag; FP32 is already the default.",
    )
    args = parser.parse_args()
    amp_enabled = bool(args.enable_amp)
    if amp_enabled:
        print(
            "[WARNING] FP16 autocast was explicitly enabled. MTL Ret metrics "
            "may be numerically invalid; use the default FP32 mode for reported results.",
            flush=True,
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full clean-test evaluation.")
    rank, local_rank, world_size, device = distributed_setup()
    if world_size != 2:
        cleanup_distributed(world_size)
        raise RuntimeError("This evaluation requires exactly two GPU processes.")
    seed_everything(args.seed, rank)

    clean_root = os.path.abspath(args.clean_root)
    clean_resource_root = os.path.join(clean_root, "resources")
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    try:
        clean_records, preflight = prepare_clean_records(
            clean_root,
            clean_resource_root,
            scope=args.scope,
        )
        if args.max_records is not None:
            clean_records = clean_records[: args.max_records]
        model, model_config, artifact_format = load_checkpoint(
            os.path.abspath(args.checkpoint),
            device,
        )
        model_holder = SimpleNamespace(module=model)
        metrics = evaluate_clean_full(
            model_holder,
            clean_records,
            device,
            rank,
            world_size,
            output_dir,
            args.log_every,
            amp_enabled,
        )
        if rank == 0:
            result = {
                "format": "icflownet_cleantest_eval_v1",
                "scope": args.scope,
                "checkpoint": os.path.abspath(args.checkpoint),
                "checkpoint_artifact_format": artifact_format,
                "threshold": 0.5,
                "clean_root": clean_root,
                "clean_resource_root": clean_resource_root,
                "world_size": world_size,
                "precision": "fp16_autocast" if amp_enabled else "fp32",
                "amp_enabled": amp_enabled,
                "model_config": model_config,
                "preflight": preflight,
                "metrics": metrics,
            }
            result_path = os.path.join(output_dir, "result.json")
            with open(result_path, "w") as handle:
                json.dump(result, handle, indent=2)
            print(json.dumps({
                "result": result_path,
                "scope": args.scope,
                "evaluated_binaries": metrics["evaluated_binaries"],
                "macro_f1": metrics["macro_f1"],
                "task_metrics": metrics["task_metrics"],
            }, indent=2), flush=True)
    finally:
        cleanup_distributed(world_size)


if __name__ == "__main__":
    main()
