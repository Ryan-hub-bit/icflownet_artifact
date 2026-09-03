#!/usr/bin/env python3
"""Evaluate one versioned single-task checkpoint on the released clean test."""

import argparse
import contextlib
import gc
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import torch
import torch.nn.functional as F

from clean_eval_utils import (
    TASKS,
    confusion_metrics,
    discard_unused_full_code_features,
    load_compressed_graph,
    prepare_clean_records,
)
from single_task_model import SingleTaskGAT


def graph_feature_dims(graph):
    code_data = graph.nodes["code"].data
    code_features = code_data.get("featmean", code_data.get("feat"))
    if code_features is None:
        raise RuntimeError("Clean graph is missing code featmean/feat.")

    data_feat_dim = 1
    if "data" in graph.ntypes and graph.num_nodes("data") > 0:
        data_features = graph.nodes["data"].data.get("feat")
        if data_features is not None:
            data_feat_dim = int(data_features.shape[1])
    return int(code_features.shape[1]), data_feat_dim


def load_checkpoint(path, task, probe_graph, device):
    artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict) or artifact.get("artifact_format") != "icflownet_model_v1":
        raise RuntimeError("Expected an icflownet_model_v1 single-task checkpoint.")
    model_config = dict(artifact.get("model_config") or {})
    model_config.pop("architecture_version", None)
    stored_task = model_config.get("task_type")
    if stored_task != task:
        raise RuntimeError(
            "Checkpoint task {!r} does not match requested task {!r}.".format(
                stored_task, task
            )
        )

    code_feat_dim, data_feat_dim = graph_feature_dims(probe_graph)
    model = SingleTaskGAT(
        code_feat_dim=code_feat_dim,
        data_feat_dim=data_feat_dim,
        **model_config,
    )
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    model.decision_threshold = 0.5
    model.to(device)
    model.eval()
    return model, model_config, artifact.get("artifact_format")


def evaluate_record(
    model,
    graph,
    rows,
    candidate_meta,
    device,
    amp_enabled,
):
    """Evaluate one clean-test graph and return aggregate edge metrics."""
    graph = graph.to(device)
    pairs = torch.tensor(
        [(row["src"], row["dst"]) for row in rows],
        dtype=torch.long,
        device=device,
    )
    labels = torch.tensor(
        [row["label"] for row in rows], dtype=torch.float32, device=device
    )

    autocast_context = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if amp_enabled
        else contextlib.nullcontext()
    )
    with torch.no_grad(), autocast_context:
        logits = model(graph, pairs, candidate_meta=candidate_meta)
        logits = logits.reshape(-1)

    scores = torch.sigmoid(logits)
    predictions = scores >= 0.5
    positives = labels == 1
    negatives = ~positives
    local_tn = int(((~predictions) & negatives).sum().item())
    local_fp = int((predictions & negatives).sum().item())
    local_fn = int(((~predictions) & positives).sum().item())
    local_tp = int((predictions & positives).sum().item())
    local_loss = float(
        F.binary_cross_entropy_with_logits(logits, labels, reduction="sum").item()
    )
    return local_tn, local_fp, local_fn, local_tp, local_loss


def evaluate(args):
    if args.task not in TASKS:
        raise ValueError("Unsupported task: {}".format(args.task))
    if args.gpu >= 0 and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device("cuda:{}".format(args.gpu) if args.gpu >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    clean_root = os.path.abspath(args.clean_root)
    resource_root = os.path.join(clean_root, "resources")
    records, preflight = prepare_clean_records(clean_root, resource_root, args.scope)
    all_eligible = [record for record in records if record["task_rows"].get(args.task)]
    indexed_eligible = list(enumerate(all_eligible, start=1))
    end_record = args.end_record or len(indexed_eligible)
    indexed_eligible = indexed_eligible[args.start_record - 1 : end_record]
    if args.max_records is not None:
        indexed_eligible = indexed_eligible[: args.max_records]
    if not indexed_eligible:
        raise RuntimeError("No clean-test records contain task {}.".format(args.task))

    probe_graph = discard_unused_full_code_features(
        load_compressed_graph(indexed_eligible[0][1]["graph_path"])
    )
    model, model_config, artifact_format = load_checkpoint(
        os.path.abspath(args.checkpoint), args.task, probe_graph, device
    )
    del probe_graph

    tn = fp = fn = tp = 0
    loss_sum = 0.0
    pair_count = 0
    details = []
    cpu_model = None
    cpu_fallback_records = []
    started = time.time()
    amp_enabled = device.type == "cuda" and args.enable_amp

    for selected_index, (record_index, record) in enumerate(indexed_eligible, start=1):
        rows = record["task_rows"][args.task]
        graph = discard_unused_full_code_features(
            load_compressed_graph(record["graph_path"])
        )
        if (
            graph.num_nodes("code") != record["expected_code_nodes"]
            or graph.num_nodes("data") != record["expected_data_nodes"]
        ):
            raise RuntimeError(
                "Clean graph node count mismatch for {}.".format(record["binary_key"])
            )
        try:
            (
                local_tn,
                local_fp,
                local_fn,
                local_tp,
                local_loss,
            ) = evaluate_record(
                model,
                graph,
                rows,
                record["candidate_meta"],
                device,
                amp_enabled,
            )
        except RuntimeError as error:
            if device.type != "cuda" or not args.cpu_fallback:
                raise
            print(
                "[{}] CUDA evaluation failed for {}; retrying this record on CPU: {}".format(
                    record_index, record["binary_key"], error
                ),
                flush=True,
            )
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
            if cpu_model is None:
                cpu_device = torch.device("cpu")
                cpu_model, _, _ = load_checkpoint(
                    os.path.abspath(args.checkpoint), args.task, graph, cpu_device
                )
            (
                local_tn,
                local_fp,
                local_fn,
                local_tp,
                local_loss,
            ) = evaluate_record(
                cpu_model,
                graph,
                rows,
                record["candidate_meta"],
                torch.device("cpu"),
                False,
            )
            cpu_fallback_records.append(
                {
                    "record_index": record_index,
                    "binary": record["binary_key"],
                    "graph_id": record["graph_id"],
                    "cuda_error": str(error),
                }
            )
        tn += local_tn
        fp += local_fp
        fn += local_fn
        tp += local_tp
        loss_sum += local_loss
        pair_count += len(rows)

        if args.save_details:
            details.append(
                {
                    "binary": record["binary_key"],
                    "graph_id": record["graph_id"],
                    "pairs": len(rows),
                    "confusion_matrix": [[local_tn, local_fp], [local_fn, local_tp]],
                }
            )
        if args.log_every and (
            selected_index % args.log_every == 0
            or selected_index == len(indexed_eligible)
        ):
            print(
                "[{}/{} selected; source record {}/{}] {} records evaluated ({:.1f}s)".format(
                    selected_index,
                    len(indexed_eligible),
                    record_index,
                    len(all_eligible),
                    args.task,
                    time.time() - started,
                ),
                flush=True,
            )

        del graph
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = confusion_metrics(tn, fp, fn, tp)
    metrics["loss"] = loss_sum / max(pair_count, 1)
    result = {
        "format": "icflownet_single_cleantest_eval_v1",
        "scope": args.scope,
        "task": args.task,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_artifact_format": artifact_format,
        "model_config": model_config,
        "threshold": 0.5,
        "threshold_source": "fixed_evaluator",
        "precision": "fp16_autocast" if amp_enabled else "fp32",
        "amp_enabled": amp_enabled,
        "clean_root": clean_root,
        "preflight": preflight,
        "available_eligible_records": len(all_eligible),
        "eligible_records": len(indexed_eligible),
        "selected_record_indices": [index for index, _ in indexed_eligible],
        "cpu_fallback_records": cpu_fallback_records,
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }
    if args.save_details:
        result["record_details"] = details
    return result


def evaluate_both_scopes(args):
    """Evaluate overall and long-range pairs with one graph encoding per record."""
    if args.task not in TASKS:
        raise ValueError("Unsupported task: {}".format(args.task))
    if args.gpu >= 0 and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device("cuda:{}".format(args.gpu) if args.gpu >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    clean_root = os.path.abspath(args.clean_root)
    resource_root = os.path.join(clean_root, "resources")
    scopes = ("overall", "long_range")
    records_by_scope = {}
    preflight_by_scope = {}
    eligible_by_scope = {}
    for scope in scopes:
        records, preflight = prepare_clean_records(clean_root, resource_root, scope)
        records_by_scope[scope] = {
            record["binary_key"]: record for record in records
        }
        preflight_by_scope[scope] = preflight
        eligible_by_scope[scope] = [
            record["binary_key"]
            for record in records
            if record["task_rows"].get(args.task)
        ]

    ordered_keys = []
    seen_keys = set()
    for record in records_by_scope["overall"].values():
        key = record["binary_key"]
        if any(
            records_by_scope[scope][key]["task_rows"].get(args.task)
            for scope in scopes
        ):
            ordered_keys.append(key)
            seen_keys.add(key)
    for key in records_by_scope["long_range"]:
        if key not in seen_keys and records_by_scope["long_range"][key][
            "task_rows"
        ].get(args.task):
            ordered_keys.append(key)

    end_record = args.end_record or len(ordered_keys)
    selected_keys = ordered_keys[args.start_record - 1 : end_record]
    if args.max_records is not None:
        selected_keys = selected_keys[: args.max_records]
    if not selected_keys:
        raise RuntimeError("No clean-test records contain task {}.".format(args.task))

    probe_record = records_by_scope["overall"].get(
        selected_keys[0], records_by_scope["long_range"][selected_keys[0]]
    )
    probe_graph = discard_unused_full_code_features(
        load_compressed_graph(probe_record["graph_path"])
    )
    model, model_config, artifact_format = load_checkpoint(
        os.path.abspath(args.checkpoint), args.task, probe_graph, device
    )
    del probe_graph

    accumulators = {
        scope: {
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "loss_sum": 0.0,
            "pair_count": 0,
            "record_count": 0,
            "details": [],
        }
        for scope in scopes
    }
    started = time.time()
    amp_enabled = device.type == "cuda" and args.enable_amp

    for selected_index, key in enumerate(selected_keys, start=1):
        base_record = records_by_scope["overall"].get(
            key, records_by_scope["long_range"][key]
        )
        combined_rows = []
        scope_slices = {}
        for scope in scopes:
            record = records_by_scope[scope].get(key)
            rows = record["task_rows"].get(args.task, []) if record else []
            if not rows:
                continue
            start = len(combined_rows)
            combined_rows.extend(rows)
            scope_slices[scope] = (start, len(combined_rows), rows)

        graph = discard_unused_full_code_features(
            load_compressed_graph(base_record["graph_path"])
        )
        if (
            graph.num_nodes("code") != base_record["expected_code_nodes"]
            or graph.num_nodes("data") != base_record["expected_data_nodes"]
        ):
            raise RuntimeError(
                "Clean graph node count mismatch for {}.".format(key)
            )

        graph = graph.to(device)
        pairs = torch.tensor(
            [(row["src"], row["dst"]) for row in combined_rows],
            dtype=torch.long,
            device=device,
        )
        labels = torch.tensor(
            [row["label"] for row in combined_rows],
            dtype=torch.float32,
            device=device,
        )
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=torch.float16)
            if amp_enabled
            else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast_context:
            logits = model(
                graph, pairs, candidate_meta=base_record["candidate_meta"]
            ).reshape(-1)

        for scope, (start, end, rows) in scope_slices.items():
            local_logits = logits[start:end]
            local_labels = labels[start:end]
            predictions = torch.sigmoid(local_logits) >= 0.5
            positives = local_labels == 1
            negatives = ~positives
            local_tn = int(((~predictions) & negatives).sum().item())
            local_fp = int((predictions & negatives).sum().item())
            local_fn = int(((~predictions) & positives).sum().item())
            local_tp = int((predictions & positives).sum().item())
            local_loss = float(
                F.binary_cross_entropy_with_logits(
                    local_logits, local_labels, reduction="sum"
                ).item()
            )
            accumulator = accumulators[scope]
            accumulator["tn"] += local_tn
            accumulator["fp"] += local_fp
            accumulator["fn"] += local_fn
            accumulator["tp"] += local_tp
            accumulator["loss_sum"] += local_loss
            accumulator["pair_count"] += len(rows)
            accumulator["record_count"] += 1
            if args.save_details:
                accumulator["details"].append(
                    {
                        "binary": key,
                        "graph_id": base_record["graph_id"],
                        "pairs": len(rows),
                        "confusion_matrix": [
                            [local_tn, local_fp],
                            [local_fn, local_tp],
                        ],
                    }
                )

        if args.log_every and (
            selected_index % args.log_every == 0
            or selected_index == len(selected_keys)
        ):
            print(
                "[{}/{} selected] {} records evaluated for both scopes ({:.1f}s)".format(
                    selected_index,
                    len(selected_keys),
                    args.task,
                    time.time() - started,
                ),
                flush=True,
            )

        del graph
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed_seconds = time.time() - started
    results = {}
    for scope in scopes:
        accumulator = accumulators[scope]
        metrics = confusion_metrics(
            accumulator["tn"],
            accumulator["fp"],
            accumulator["fn"],
            accumulator["tp"],
        )
        metrics["loss"] = accumulator["loss_sum"] / max(
            accumulator["pair_count"], 1
        )
        result = {
            "format": "icflownet_single_cleantest_eval_v1",
            "scope": scope,
            "task": args.task,
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_artifact_format": artifact_format,
            "model_config": model_config,
            "threshold": 0.5,
            "threshold_source": "fixed_evaluator",
            "precision": "fp16_autocast" if amp_enabled else "fp32",
            "amp_enabled": amp_enabled,
            "clean_root": clean_root,
            "preflight": preflight_by_scope[scope],
            "available_eligible_records": len(eligible_by_scope[scope]),
            "eligible_records": accumulator["record_count"],
            "selected_record_indices": list(
                range(1, accumulator["record_count"] + 1)
            ),
            "cpu_fallback_records": [],
            "metrics": metrics,
            "elapsed_seconds": elapsed_seconds,
            "joint_scope_evaluation": True,
        }
        if args.save_details:
            result["record_details"] = accumulator["details"]
        results[scope] = result
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clean_root", required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument(
        "--scope", choices=("overall", "long_range", "both"), default="overall"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--start_record", type=int, default=1)
    parser.add_argument("--end_record", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=25)
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument(
        "--enable_amp",
        action="store_true",
        help="Explicitly opt into FP16 autocast; FP32 is the default.",
    )
    precision_group.add_argument(
        "--disable_amp",
        action="store_true",
        help="Deprecated compatibility flag; FP32 is already the default.",
    )
    parser.add_argument(
        "--cpu_fallback",
        action="store_true",
        help="Retry an individual record on CPU if its CUDA evaluation fails.",
    )
    parser.add_argument("--save_details", action="store_true")
    args = parser.parse_args()
    if args.enable_amp:
        print(
            "[WARNING] FP16 autocast was explicitly enabled; artifact results "
            "must use the default FP32 mode.",
            flush=True,
        )
    if args.start_record < 1:
        parser.error("--start_record must be at least 1")
    if args.end_record is not None and args.end_record < args.start_record:
        parser.error("--end_record must be greater than or equal to --start_record")

    output = os.path.abspath(args.output)
    if args.scope == "both":
        results = evaluate_both_scopes(args)
        os.makedirs(output, exist_ok=True)
        summaries = []
        for scope, result in results.items():
            scope_output = os.path.join(output, scope + ".json")
            with open(scope_output, "w") as handle:
                json.dump(result, handle, indent=2)
                handle.write("\n")
            summaries.append(
                {
                    "output": scope_output,
                    "task": result["task"],
                    "scope": result["scope"],
                    "eligible_records": result["eligible_records"],
                    "metrics": result["metrics"],
                    "elapsed_seconds": result["elapsed_seconds"],
                }
            )
        print(json.dumps(summaries, indent=2))
    else:
        result = evaluate(args)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(json.dumps({
            "output": output,
            "task": result["task"],
            "scope": result["scope"],
            "eligible_records": result["eligible_records"],
            "metrics": result["metrics"],
            "elapsed_seconds": result["elapsed_seconds"],
        }, indent=2))


if __name__ == "__main__":
    main()
