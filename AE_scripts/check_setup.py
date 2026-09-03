#!/usr/bin/env python3
"""Validate the ICFlowNet AE environment and extracted data package."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED_SHA256 = {
    "model/ic_hub_model.pt": "f39aa81f1b030daf796c254ad8dd8a0297ff953529ec71816849c2f4d5460cbd",
    "model/ic_nohub_model.pt": "08692bc76cd95222b2623900e06ffb46cace99c4ebb786b246887e7c6fad2de2",
    "model/mtl_model.pt": "40fb028de83182baed606e047f9088682b48ee4a083dfd77082f7fcd5e5a8ebb",
    "cleantest/binary_graph_mapping.jsonl": "47724e616d4b1226eee89fe1020650f83846542aa8ea5f388335112927c6771f",
    "train_sample/bintoindex.json": "3414463cb1ee29084c80fd22a11e2e1fa9232cd7c0b1f1e3bb01e64761e7daf3",
    "train_sample/indextobin.json": "53f8ecd66f93e0018cf61fb325820d5cc9060eeb9b2b37692d4d5bd83ffad7c2",
    "train_sample/indextograph.json": "8965aa91a58c23df9a7a5224da40688f904a7b5ce1530a00aa8e6de3c2fb9703",
    "train_sample/indextores.json": "0ecd5c77460a84585c8ad719d461aadd7481b08dede5857533699cc3ec91e581",
}
EXPECTED_CLEAN_RECORDS = 185
EXPECTED_TRAIN_RECORDS = 20
TASKS = ("ret", "jumptable", "indirectcall", "tailcall")
TRAIN_GT_SUFFIXES = {
    "ret": "_ret.json",
    "jumptable": "_correctjumptable.json",
    "indirectcall": "_icallbbtocallee.json",
    "tailcall": "_itcbbtofunc.json",
}


def fail(message):
    raise SystemExit("AE preflight FAIL: " + message)


def require(condition, message):
    if not condition:
        fail(message)


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                fail(f"invalid JSON at {path}:{line_number}: {error}")
    return rows


def read_json(path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as error:
        fail(f"cannot read {path}: {error}")
    except json.JSONDecodeError as error:
        fail(f"invalid JSON at {path}: {error}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dependencies(require_gpus):
    try:
        import angr
        import dgl
        import torch
    except ImportError as error:
        fail(f"missing Python dependency: {error}")

    require(torch.__version__.startswith("2.3."), f"expected PyTorch 2.3, got {torch.__version__}")
    require(dgl.__version__.startswith("2.4."), f"expected DGL 2.4, got {dgl.__version__}")
    require(angr.__version__.startswith("9.2.157"), f"expected angr 9.2.157, got {angr.__version__}")
    if require_gpus:
        require(torch.cuda.is_available(), "CUDA is unavailable")
        require(torch.cuda.device_count() >= 2, "E3 evaluation requires at least two visible GPUs")
    print(
        "Environment PASS: "
        f"torch={torch.__version__}, dgl={dgl.__version__}, angr={angr.__version__}, "
        f"visible_gpus={torch.cuda.device_count()}"
    )
    return torch


def validate_checkpoint(torch, path, expected_hubs, expected_task):
    artifact = torch.load(path, map_location="cpu")
    require(isinstance(artifact, dict), f"{path} is not a checkpoint dictionary")
    require(
        artifact.get("artifact_format") == "icflownet_model_v1",
        f"{path} has an unsupported artifact format",
    )
    require("model_state_dict" in artifact, f"{path} lacks model_state_dict")
    config = artifact.get("model_config")
    require(isinstance(config, dict), f"{path} lacks model_config")
    require(
        (bool(config.get("use_gch")), bool(config.get("use_gdh"))) == expected_hubs,
        f"{path} has the wrong hub configuration",
    )
    if expected_task is not None:
        require(config.get("task_type") == expected_task, f"{path} has the wrong task type")
    else:
        task_weights = config.get("task_weights", {})
        require(all(task in task_weights for task in TASKS), f"{path} lacks one or more MTL heads")


def validate_package(torch, data_root, check_hashes):
    require(data_root.is_dir(), f"data root does not exist: {data_root}")

    for relative in EXPECTED_SHA256:
        require((data_root / relative).is_file(), f"missing {data_root / relative}")

    if check_hashes:
        for relative, expected in EXPECTED_SHA256.items():
            actual = sha256(data_root / relative)
            require(actual == expected, f"checksum mismatch for {relative}")

    clean_root = data_root / "cleantest"
    clean_rows = read_jsonl(clean_root / "binary_graph_mapping.jsonl")
    require(
        len(clean_rows) == EXPECTED_CLEAN_RECORDS,
        f"expected {EXPECTED_CLEAN_RECORDS} clean-test records, found {len(clean_rows)}",
    )
    clean_keys = (
        "binary",
        "graph",
        "overall_positive",
        "overall_negative",
        "long_range_positive",
        "long_range_negative",
    )
    for row in clean_rows:
        binary_key = row.get("binary_key", "<unknown>")
        for key in clean_keys:
            relative = row.get(key)
            require(relative, f"clean-test record {binary_key} lacks {key}")
            require((clean_root / relative).is_file(), f"missing cleantest/{relative}")
        require(
            (clean_root / "resources" / binary_key).is_dir(),
            f"missing resources for clean-test binary {binary_key}",
        )

    sample_root = data_root / "train_sample"
    bintoindex = read_json(sample_root / "bintoindex.json")
    indextobin = read_json(sample_root / "indextobin.json")
    indextograph = read_json(sample_root / "indextograph.json")
    indextores = read_json(sample_root / "indextores.json")
    for name, mapping in (
        ("bintoindex", bintoindex),
        ("indextobin", indextobin),
        ("indextograph", indextograph),
        ("indextores", indextores),
    ):
        require(isinstance(mapping, dict), f"train_sample/{name}.json is not an object")

    sample_ids = {str(index) for index in bintoindex.values()}
    require(
        len(sample_ids) == EXPECTED_TRAIN_RECORDS,
        f"expected {EXPECTED_TRAIN_RECORDS} training records, found {len(sample_ids)}",
    )
    for name, mapping in (
        ("indextobin", indextobin),
        ("indextograph", indextograph),
        ("indextores", indextores),
    ):
        require(set(mapping) == sample_ids, f"train_sample/{name}.json has inconsistent indexes")

    task_coverage = Counter()
    for binary, index in bintoindex.items():
        index = str(index)
        require(indextobin[index] == binary, f"inconsistent binary mapping for index {index}")
        require(
            (sample_root / indextograph[index]).is_file(),
            f"missing sample graph for index {index}",
        )
        resource_root = sample_root / indextores[index]
        require(resource_root.is_dir(), f"missing sample resources for index {index}")
        resource_name = resource_root.name
        node_lookup = read_json(resource_root / f"{resource_name}_nodelookup.json")
        require(isinstance(node_lookup, dict), f"invalid node lookup for index {index}")
        for task, suffix in TRAIN_GT_SUFFIXES.items():
            ground_truth = read_json(resource_root / f"{resource_name}{suffix}")
            require(isinstance(ground_truth, dict), f"invalid {task} GT for index {index}")
            has_valid_pair = any(
                node_lookup.get(source, -1) != -1
                and any(node_lookup.get(target, -1) != -1 for target in targets)
                for source, targets in ground_truth.items()
            )
            if has_valid_pair:
                task_coverage[task] += 1
    require(all(task_coverage[task] > 0 for task in TASKS), "training sample does not cover all four tasks")

    validate_checkpoint(
        torch, data_root / "model" / "ic_hub_model.pt", (True, True), "indirectcall"
    )
    validate_checkpoint(
        torch, data_root / "model" / "ic_nohub_model.pt", (False, False), "indirectcall"
    )
    validate_checkpoint(torch, data_root / "model" / "mtl_model.pt", (True, True), None)
    coverage = ", ".join(f"{task}={task_coverage[task]}" for task in TASKS)
    print(
        f"Artifact data PASS: clean_records={len(clean_rows)}, "
        f"train_records={len(sample_ids)} ({coverage})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Validate files and packages without requiring two visible GPUs.",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip SHA-256 checks when doing a quick repeated preflight.",
    )
    args = parser.parse_args()
    torch = validate_dependencies(require_gpus=not args.skip_gpu_check)
    validate_package(torch, args.data_root.resolve(), check_hashes=not args.skip_checksums)
    print("AE preflight PASS")


if __name__ == "__main__":
    main()
