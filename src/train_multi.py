"""Train the multi-task ICFlowNet model from scratch."""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from tqdm import tqdm
import json
from collections import defaultdict
from loadgraph import JumpGraphDataset  # Your dataset
from common import (
    EDGE_TYPE_MAPPING,
    EDGE_TYPE_TO_FNHASH,
    load_model_artifact,
    save_model_artifact,
)
from multi_task_model import MultiTaskGAT  # Your multi-task model
from util.removeduplicate import (
    count_function_hash_sidecars,
    extract_funchash_sets,
    filter_and_save_val_jsons,
)
from collections import defaultdict
import random
import os
import json
from collections import Counter

global_removed_hash_counter = Counter()

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Dictionary to store per-edge-type counters
site_stats = {
    "total_sites": {},
    "removed_sites": {}
}

TRAIN_GT_SUFFIX_BY_EDGE_TYPE = {
    "ret": "_ret.json",
    "jumptable": "_correctjumptable.json",
    "indirectcall": "_icallbbtocallee.json",
    "tailcall": "_itcbbtofunc.json",
}

def get_dataset_graph_ids(dataset):
    """Return persisted graph ids rather than positional dataset offsets."""
    graph_files = getattr(dataset, "graph_files", None)
    if graph_files is None:
        return [str(idx) for idx in range(len(dataset))]
    return [str(graph_id) for graph_id in graph_files]

def infer_package_from_binary_path(path):
    """Infer a package key from /.../binary/<package>/<binary> paths."""
    parts = os.path.normpath(path).split(os.sep)
    if "binary" in parts:
        binary_pos = parts.index("binary")
        if binary_pos + 1 < len(parts):
            return parts[binary_pos + 1]
    return os.path.basename(os.path.dirname(path))

def package_for_binary_path(path, package_mapping):
    """Resolve a package from the canonical mapping, or infer it from the path."""
    if not path:
        return None

    binname = os.path.basename(path)
    for key in (path, binname):
        pkg = package_mapping.get(key)
        if pkg:
            return pkg

    return infer_package_from_binary_path(path)

def load_binary_to_package_or_empty(basedir):
    """Load the canonical package mapping, or infer packages from binary paths."""
    binary_to_package_path = os.path.join(basedir, "binary_to_package.json")
    if os.path.exists(binary_to_package_path):
        with open(binary_to_package_path, 'r') as f:
            mapping = json.load(f)
        print(f"[INFO] Loaded package mapping from {binary_to_package_path}")
        return mapping

    print(
        f"[INFO] No binary_to_package.json under {basedir}; "
        "inferring package names from binary paths."
    )
    return {}

def graph_has_train_gt_for_edge_type(indextores, graph_id, edge_type):
    """Fast GT presence check for package splitting without loading the DGL graph."""
    gt_folder = indextores.get(str(graph_id))
    if not gt_folder:
        return False

    binary_name = os.path.basename(gt_folder)
    gt_path = os.path.join(gt_folder, binary_name + TRAIN_GT_SUFFIX_BY_EDGE_TYPE[edge_type])
    nodelookup_path = os.path.join(gt_folder, binary_name + "_nodelookup.json")
    if not os.path.exists(gt_path) or not os.path.exists(nodelookup_path):
        return False

    try:
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
        with open(nodelookup_path, "r") as f:
            node_lookup = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(gt_data, dict):
        return False

    for src, dsts in gt_data.items():
        if node_lookup.get(src, -1) == -1:
            continue
        for dst in dsts:
            if node_lookup.get(dst, -1) != -1:
                return True
    return False

def packages_for_indices(indices, indextobin, package_mapping):
    packages = set()
    missing = 0
    for idx in indices:
        path = indextobin.get(str(idx))
        pkg = package_for_binary_path(path, package_mapping) if path else None
        if pkg:
            packages.add(pkg)
        else:
            missing += 1
    return packages, missing

def log_package_split_summary(train_indices, val_indices, test_indices, indextobin, package_mapping):
    train_pkgs, train_missing = packages_for_indices(train_indices, indextobin, package_mapping)
    val_pkgs, val_missing = packages_for_indices(val_indices, indextobin, package_mapping)
    test_pkgs, test_missing = packages_for_indices(test_indices, indextobin, package_mapping)

    train_val_overlap = train_pkgs & val_pkgs
    train_test_overlap = train_pkgs & test_pkgs
    val_test_overlap = val_pkgs & test_pkgs

    print(
        f"[INFO] Package split summary: "
        f"train={len(train_pkgs)} pkgs, val={len(val_pkgs)} pkgs, test={len(test_pkgs)} pkgs"
    )
    if train_missing or val_missing or test_missing:
        print(
            f"[WARN] Package lookup missing for graphs: "
            f"train={train_missing}, val={val_missing}, test={test_missing}"
        )
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print(
            f"[WARN] Package overlap detected: "
            f"train/val={len(train_val_overlap)}, "
            f"train/test={len(train_test_overlap)}, "
            f"val/test={len(val_test_overlap)}"
        )

def global_counter_for_task(edge_type, total, removed):
    if edge_type not in site_stats["total_sites"]:
        site_stats["total_sites"][edge_type] = 0
        site_stats["removed_sites"][edge_type] = 0
    site_stats["total_sites"][edge_type] += total
    site_stats["removed_sites"][edge_type] += removed

def calculate_class_weights(pos_count, neg_count):
    """Calculate class weights for imbalanced datasets"""
    total = pos_count + neg_count
    pos_weight = total / (2.0 * pos_count) if pos_count > 0 else 1.0
    neg_weight = total / (2.0 * neg_count) if neg_count > 0 else 1.0
    return pos_weight, neg_weight

def filter_indices_by_edge_type(dataset, edge_type, args):
    """
    Filter graph indices that contain at least one positive edge of the specified type.
    Results are cached to avoid recomputation if the dataset hasn't changed.
    """
    graph_ids = get_dataset_graph_ids(dataset)
    total_graphs = len(graph_ids)
    
    os.makedirs(os.path.join(args.output_dir, "filter_cache"), exist_ok=True)
    cache_file = os.path.join(args.output_dir, "filter_cache", f"filtered_indices_{edge_type}.json")
    
    # Try loading from cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
                if cache.get("total_graphs") == total_graphs and cache.get("graph_ids") == graph_ids:
                    print(f"[CACHE] Loaded {len(cache['valid_indices'])} filtered indices for '{edge_type}' from cache.")
                    return cache["valid_indices"]
                else:
                    print(f"[CACHE] Graph count mismatch for '{edge_type}'. Recomputing filtered indices.")
        except Exception as e:
            print(f"[CACHE] Failed to load cache for '{edge_type}': {e}")
    
    # Recompute if cache not found or invalid
    print(f"Filtering graphs for edge type: {edge_type} (ID: {EDGE_TYPE_MAPPING[edge_type]})")
    valid_indices = []
    basedir = os.path.dirname(dataset.graph_dir)
    indextores_path = os.path.join(basedir, "indextores.json")
    with open(indextores_path, "r") as f:
        indextores = json.load(f)

    for idx in tqdm(graph_ids, desc=f"Filtering graphs for {edge_type}"):
        try:
            if graph_has_train_gt_for_edge_type(indextores, idx, edge_type):
                valid_indices.append(idx)
        except Exception as e:
            print(f"[WARNING] Skipping graph {idx} due to error: {e}")
            continue
    
    print(f"Found {len(valid_indices)} graphs with {edge_type} edges out of {total_graphs} total graphs")
    
    # Save cache
    try:
        with open(cache_file, "w") as f:
            json.dump({
                "total_graphs": total_graphs,
                "graph_ids": graph_ids,
                "valid_indices": valid_indices
            }, f)
        print(f"[CACHE] Saved filtered indices to {cache_file}")
    except Exception as e:
        print(f"[CACHE] Failed to save cache for '{edge_type}': {e}")
    
    return valid_indices

# def multi_label_proportional_split_and_save(edge_type_order, dataset, args, split_cache_path):
#     """
#     Splits graphs into train/val/test such that each edge type gets an ~80/10/10 split
#     across the entire dataset, and each graph is assigned to only one split.

#     The split is saved to a JSON file with per-type graph index lists like:
#     {
#       "tailcall_train_graph": [...],
#       "tailcall_val_graph": [...],
#       "tailcall_test_graph": [...],
#       ...
#     }
#     """
#     ratio_train = args.train_ratio
#     ratio_val = args.val_ratio
#     ratio_test = 1.0 - ratio_train - ratio_val

#     # Step 1: Collect per-type graph indices and build graph -> types map
#     graph_to_types = defaultdict(set)
#     type_to_graphs = defaultdict(set)

#     for edge_type in edge_type_order:
#         indices = filter_indices_by_edge_type(dataset, edge_type, args)
#         for idx in indices:
#             graph_to_types[idx].add(edge_type)
#             type_to_graphs[edge_type].add(idx)

#     all_graphs = list(graph_to_types.keys())
#     random.seed(args.seed)
#     random.shuffle(all_graphs)

#     # Step 2: Initialize counters
#     train_set, val_set, test_set = set(), set(), set()
#     assigned = set()

#     # Counters and target counts for each edge type
#     counts = {et: {"train": 0, "val": 0, "test": 0} for et in edge_type_order}
#     targets = {
#         et: {
#             "train": int(ratio_train * len(type_to_graphs[et])),
#             "val": int(ratio_val * len(type_to_graphs[et])),
#             "test": len(type_to_graphs[et]) - int(ratio_train * len(type_to_graphs[et])) - int(ratio_val * len(type_to_graphs[et]))
#         } for et in edge_type_order
#     }

#     # Step 3: Assignment and per-type tracking
#     per_type_split = {
#         et: {
#             "train": [],
#             "val": [],
#             "test": []
#         } for et in edge_type_order
#     }

#     for idx in all_graphs:
#         if idx in assigned:
#             continue

#         types = graph_to_types[idx]

#         # Decide best split based on edge type needs
#         scores = {"train": 0, "val": 0, "test": 0}
#         for et in types:
#             for split in ["train", "val", "test"]:
#                 if counts[et][split] < targets[et][split]:
#                     scores[split] += 1

#         best_split = max(scores, key=scores.get)
#         if scores[best_split] == 0:
#             continue  # Can't help any remaining target

#         # Assign graph to chosen split
#         if best_split == "train":
#             train_set.add(idx)
#         elif best_split == "val":
#             val_set.add(idx)
#         else:
#             test_set.add(idx)

#         assigned.add(idx)
#         for et in types:
#             counts[et][best_split] += 1
#             per_type_split[et][best_split].append(idx)

#     # Prepare data for saving to JSON
#     split_json = {}
#     for et in edge_type_order:
#         split_json[f"{et}_train_graph"] = per_type_split[et]["train"]
#         split_json[f"{et}_val_graph"] = per_type_split[et]["val"]
#         split_json[f"{et}_test_graph"] = per_type_split[et]["test"]

#     if not os.path.exists(split_cache_path):
#         os.makedirs(os.path.dirname(split_cache_path), exist_ok=True)
#         with open(split_cache_path, "w") as f:
#             json.dump(split_json, f, indent=2)
#         print(f"✅ Saved per-type split info to {split_cache_path}")
#     else:
#         print(f"ℹ️ Split file already exists at {split_cache_path}, not overwriting.")

#     # Log summary
#     print("\n📊 Per-edge-type actual split counts:")
#     for et in edge_type_order:
#         print(f"  {et}: Train={len(per_type_split[et]['train'])}, "
#               f"Val={len(per_type_split[et]['val'])}, "
#               f"Test={len(per_type_split[et]['test'])}, "
#               f"Total={len(type_to_graphs[et])}")

#     print(f"\n📊 Final dataset sizes: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

#     return list(train_set), list(val_set), list(test_set), per_type_split


# def load_or_create_split_by_package(
#     edge_type_order, 
#     dataset, 
#     args, 
#     split_cache_path, 
#     indextobin, 
#     package_mapping
# ):
#     """
#     Load split from split_cache_path if it exists;
#     otherwise call multi_label_proportional_split_by_package_and_save
#     to create and save it.
#     Returns:
#         train_indices, val_indices, test_indices, per_type_split
#     """
#     if os.path.exists(split_cache_path):
#         print(f"[INFO] ✅ Found existing split at {split_cache_path}, loading it...")
#         with open(split_cache_path, "r") as f:
#             split_json = json.load(f)

#         per_type_split = split_json

#         train_indices = set()
#         val_indices = set()
#         test_indices = set()

#         for et in edge_type_order:
#             train_indices.update(split_json.get(f"{et}_train_graph", []))
#             val_indices.update(split_json.get(f"{et}_val_graph", []))
#             test_indices.update(split_json.get(f"{et}_test_graph", []))

#         train_indices = list(train_indices)
#         val_indices = list(val_indices)
#         test_indices = list(test_indices)

#         print(f"[INFO] ✅ Loaded splits: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

#     else:
#         print(f"[INFO] ❌ No split found — creating new splits and saving to {split_cache_path} ...")
#         train_indices, val_indices, test_indices, per_type_split = multi_label_proportional_split_by_package_and_save(
#             edge_type_order, dataset, args, split_cache_path, indextobin, package_mapping
#         )
#         print(f"[INFO] ✅ New splits generated: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

#     return train_indices, val_indices, test_indices, per_type_split

import os
import json
import random

def load_or_create_split_by_package(
    edge_type_order, 
    dataset, 
    args, 
    split_cache_path, 
    indextobin, 
    package_mapping
):
    """
    Load split from split_cache_path if it exists;
    otherwise call multi_label_proportional_split_by_package_and_save
    to create and save it.

    Then, downsample each split (train/val/test) to a percentage specified by args.split_percentage.
    Default is 100% (no downsampling).

    Returns:
        train_indices, val_indices, test_indices, per_type_split
    """
    if os.path.exists(split_cache_path) and not getattr(args, "force_recreate_split", False):
        print(f"[INFO] ✅ Found existing split at {split_cache_path}, loading it...")
        with open(split_cache_path, "r") as f:
            split_json = json.load(f)

        per_type_split = split_json

        train_indices = set()
        val_indices = set()
        test_indices = set()

        if all(key in split_json for key in ("train_graph", "val_graph", "test_graph")):
            train_indices.update(split_json.get("train_graph", []))
            val_indices.update(split_json.get("val_graph", []))
            test_indices.update(split_json.get("test_graph", []))
        else:
            for et in edge_type_order:
                train_indices.update(split_json.get(f"{et}_train_graph", []))
                val_indices.update(split_json.get(f"{et}_val_graph", []))
                test_indices.update(split_json.get(f"{et}_test_graph", []))

        train_indices = list(train_indices)
        val_indices = list(val_indices)
        test_indices = list(test_indices)

        print(f"[INFO] ✅ Loaded splits: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    else:
        if os.path.exists(split_cache_path):
            print(f"[INFO] Recreating split at {split_cache_path} because --force_recreate_split was set.")
        else:
            print(f"[INFO] ❌ No split found — creating new splits and saving to {split_cache_path} ...")
        train_indices, val_indices, test_indices, per_type_split = multi_label_proportional_split_by_package_and_save(
            edge_type_order, dataset, args, split_cache_path, indextobin, package_mapping
        )
        print(f"[INFO] ✅ New splits generated: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    # Downsample the loaded package split. At very small percentages, retain at
    # least one graph for each requested task when the split contains one.
    data_fraction = getattr(args, "data_fraction", None)
    if data_fraction is not None:
        if not 0 < data_fraction <= 1:
            raise ValueError("data_fraction must be in (0, 1].")
        split_percentage = data_fraction * 100.0
    else:
        split_percentage = getattr(args, "data_percentage", 100)
    if not 0 < split_percentage <= 100:
        raise ValueError("data_percentage must be in (0, 100].")

    coverage_types = [
        edge_type
        for edge_type in getattr(args, "target_edge_types", edge_type_order)
        if edge_type in edge_type_order
    ]

    def task_split_indices(edge_type, split_name):
        split_key = {
            "train": "train",
            "validation": "val",
            "val": "val",
            "test": "test",
        }[split_name.lower()]
        direct_key = f"{edge_type}_{split_key}_graph"
        if isinstance(per_type_split, dict) and direct_key in per_type_split:
            return [str(value) for value in per_type_split[direct_key]]
        nested = per_type_split.get(edge_type, {}) if isinstance(per_type_split, dict) else {}
        values = nested.get(split_key, []) if isinstance(nested, dict) else []
        return [str(value) for value in values]

    def sample_percentage(indices, name):
        ordered = sorted({str(value) for value in indices}, key=lambda value: int(value))
        size = len(ordered)
        if size == 0:
            return []
        k = max(1, int((split_percentage / 100.0) * size))
        if k >= size:
            print(f"[INFO] 🔀 {name}: Keeping full {size} (100%)")
            return ordered

        rng = random.Random(f"{args.seed}:{name}:{split_percentage:.12g}")
        available = set(ordered)
        mandatory = set()
        for edge_type in coverage_types:
            candidates = sorted(
                available.intersection(task_split_indices(edge_type, name)),
                key=lambda value: int(value),
            )
            if candidates:
                mandatory.add(rng.choice(candidates))

        keep_count = min(size, max(k, len(mandatory)))
        remaining = [value for value in ordered if value not in mandatory]
        sampled = set(mandatory)
        sampled.update(rng.sample(remaining, keep_count - len(sampled)))
        result = sorted(sampled, key=lambda value: int(value))
        print(
            f"[INFO] 🔀 {name}: Downsampled to {len(result)} of {size} "
            f"({split_percentage}%; task-coverage floor={len(mandatory)})"
        )
        return result

    train_indices = sample_percentage(train_indices, "Train")
    val_indices = sample_percentage(val_indices, "Validation")
    test_indices = sample_percentage(test_indices, "Test")
    log_package_split_summary(
        train_indices,
        val_indices,
        test_indices,
        indextobin,
        package_mapping,
    )

    return train_indices, val_indices, test_indices, per_type_split


def multi_label_proportional_split_by_package_and_save(
    edge_type_order,
    dataset,
    args,
    split_cache_path,
    indextobin,
    package_mapping,
):
    """
    Performs a multi-label-aware 80/10/10 split at the *package* level, assigning entire packages
    to train/val/test to avoid leakage. Ensures each edge type gets its target ratio approximately.
    """

    ratio_train = args.train_ratio
    ratio_val = args.val_ratio
    ratio_test = 1.0 - ratio_train - ratio_val

    # Step 1: Collect per-type graph indices and build graph -> edge_types
    graph_to_types = defaultdict(set)
    type_to_graphs = defaultdict(set)
    for edge_type in edge_type_order:
        indices = filter_indices_by_edge_type(dataset, edge_type, args)
        for idx in indices:
            graph_to_types[idx].add(edge_type)
            type_to_graphs[edge_type].add(idx)

    # Step 2: Group graphs by package
    package_to_graphs = defaultdict(set)
    for idx in graph_to_types:
        if idx not in indextobin:
            continue
        path = indextobin[idx]
        pkg = package_for_binary_path(path, package_mapping)
        if pkg:
            package_to_graphs[pkg].add(idx)

    all_packages = list(package_to_graphs.items())
    random.seed(args.seed)
    random.shuffle(all_packages)

    # Step 3: Compute per-type targets
    counts = {et: {"train": 0, "val": 0, "test": 0} for et in edge_type_order}
    targets = {
        et: {
            "train": int(ratio_train * len(type_to_graphs[et])),
            "val": int(ratio_val * len(type_to_graphs[et])),
            "test": len(type_to_graphs[et]) - int(ratio_train * len(type_to_graphs[et])) - int(ratio_val * len(type_to_graphs[et]))
        } for et in edge_type_order
    }

    # Step 4: Greedy package assignment to splits
    train_set, val_set, test_set = set(), set(), set()
    per_type_split = {et: {"train": [], "val": [], "test": []} for et in edge_type_order}

    for pkg, graph_indices in all_packages:
        type_counts = {"train": 0, "val": 0, "test": 0}
        for idx in graph_indices:
            for et in graph_to_types[idx]:
                for split in ["train", "val", "test"]:
                    if counts[et][split] < targets[et][split]:
                        type_counts[split] += 1

        # Choose best split for this package
        best_split = max(type_counts, key=type_counts.get)
        if type_counts[best_split] == 0:
            continue  # No split benefits from this package anymore

        if best_split == "train":
            train_set.update(graph_indices)
        elif best_split == "val":
            val_set.update(graph_indices)
        else:
            test_set.update(graph_indices)

        for idx in graph_indices:
            for et in graph_to_types[idx]:
                if counts[et][best_split] < targets[et][best_split]:
                    counts[et][best_split] += 1
                    per_type_split[et][best_split].append(idx)

    # Step 5: Prepare output
    def sort_graph_ids(values):
        return sorted((str(idx) for idx in values), key=lambda x: int(x))

    split_json = {
        "train_graph": sort_graph_ids(train_set),
        "val_graph": sort_graph_ids(val_set),
        "test_graph": sort_graph_ids(test_set),
    }
    for et in edge_type_order:
        split_json[f"{et}_train_graph"] = per_type_split[et]["train"]
        split_json[f"{et}_val_graph"] = per_type_split[et]["val"]
        split_json[f"{et}_test_graph"] = per_type_split[et]["test"]

    os.makedirs(os.path.dirname(split_cache_path), exist_ok=True)
    with open(split_cache_path, "w") as f:
        json.dump(split_json, f, indent=2)
    print(f"✅ Saved per-type split info to {split_cache_path}")

    # Final stats
    print("\n📊 Per-edge-type actual split counts:")
    for et in edge_type_order:
        print(f"  {et}: Train={len(per_type_split[et]['train'])}, "
              f"Val={len(per_type_split[et]['val'])}, "
              f"Test={len(per_type_split[et]['test'])}, "
              f"Total={len(type_to_graphs[et])}")

    print(f"\n📊 Final dataset sizes: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

    return list(train_set), list(val_set), list(test_set), per_type_split

    
def proportional_split_per_type(edge_type_order, dataset, args):
    """
    Split indices based on priority of edge types.
    The graphs are disjoint between splits. Tail call is split first, then icall, then jumptable and ret.
    """
    train_set, val_set, test_set = set(), set(), set()
    total_assigned = set()
    ratio_train, ratio_val = args.train_ratio, args.val_ratio

    for edge_type in edge_type_order:
        valid_indices = filter_indices_by_edge_type(dataset, edge_type, args)
        # Only use indices not already assigned
        unassigned = [idx for idx in valid_indices if idx not in total_assigned]
        
        random.seed(args.seed)
        random.shuffle(unassigned)

        n_total = len(unassigned)
        n_train = int(ratio_train * n_total)
        n_val = int(ratio_val * n_total)
        n_test = n_total - n_train - n_val

        train_indices = unassigned[:n_train]
        val_indices = unassigned[n_train:n_train + n_val]
        test_indices = unassigned[n_train + n_val:]

        train_set.update(train_indices)
        val_set.update(val_indices)
        test_set.update(test_indices)
        total_assigned.update(unassigned)

        print(f"📊 {edge_type}: Total={n_total}, Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    return list(train_set), list(val_set), list(test_set)


def split_indices_for_task(valid_indices, train_ratio, val_ratio, seed, indextobin):
    """
    Split the filtered indices into train/val/test sets
    """
    package_to_indices = defaultdict(list)
    for idx_str in valid_indices:
        idx = str(idx_str)
        if idx not in indextobin:
            continue
        path = indextobin[idx]
        pkg = infer_package_from_binary_path(path)
        package_to_indices[pkg].append(idx)
        
    random.seed(seed)  # Ensure reproducible splits
    # Step 3: shuffle packages
    all_packages = list(package_to_indices.items())
    random.shuffle(all_packages)
    
    train, val, test = [], [], []
    train_count = val_count = test_count = 0
    total_count = sum(len(v) for _, v in all_packages)
    for pkg, binaries in all_packages:
        if train_count < train_ratio * total_count:
            train.extend(binaries)
            train_count += len(binaries)
        elif val_count < val_ratio * total_count:
            val.extend(binaries)
            val_count += len(binaries)
        else:
            test.extend(binaries)
            test_count += len(binaries)
    return train, val, test

def process_duplicate_removal_for_split(train_indices, val_indices, test_indices, indextores, edge_type):
    """
    Process duplicate removal for train/val/test splits
    """
    # Extract function hashes from training set
    trainjthash = set()
    trainitchash = set()
    trainicallhash = set()
    trainrethash = set()
    
    for i in tqdm(train_indices, desc=f"Extracting hashes from training set for {edge_type}"):
        jt_set, ret_set, icall_set, itc_set = extract_funchash_sets(i, indextores)
        trainjthash.update(jt_set)
        trainitchash.update(itc_set)
        trainicallhash.update(icall_set)
        trainrethash.update(ret_set)
    
    # Filter validation set
    for j in tqdm(val_indices, desc=f"Filtering validation set for {edge_type}"):
        filter_and_save_val_jsons(
            trainjthash,
            trainrethash,
            trainicallhash,
            trainitchash,
            j,
            indextores,
            counter=lambda total, removed: global_counter_for_task(edge_type, total, removed),
            global_counter=global_removed_hash_counter  # << add this param
        )
    
    # Filter test set
    for k in tqdm(test_indices, desc=f"Filtering test set for {edge_type}"):
        filter_and_save_val_jsons(
            trainjthash,
            trainrethash,
            trainicallhash,
            trainitchash,
            k,
            indextores,
            counter=lambda total, removed: global_counter_for_task(edge_type, total, removed),
            global_counter=global_removed_hash_counter  # << add this param
        )

def extract_hashes_for_all_types(args):
    """
    Pre-extract hashes for all edge types to avoid redundant computation during trials.
    Returns a dictionary mapping edge_type -> (train_hashes, train_indices, val_indices, test_indices)
    """
    print(f"\n{'='*80}")
    print("🔄 PRE-EXTRACTING HASHES FOR ALL EDGE TYPES...")
    print(f"{'='*80}")
    
    # Setup paths
    basedir = os.path.dirname(args.graph_dir)
    indextores_path = os.path.join(basedir, "indextores.json")
    bintoindex_path = os.path.join(basedir, "bintoindex.json")
    
    with open(indextores_path, 'r') as f:
        indextores = json.load(f)
    with open(bintoindex_path, 'r') as f:
        bintoindex = json.load(f)
    
    index_to_path = {v: k for k, v in bintoindex.items()}
    
    # Create dataset once
    dataset = JumpGraphDataset(
        graph_dir=args.graph_dir,
        neg_multiplier=args.neg_multiplier if args.oversample_negatives else 1,
        use_data_nodes=args.use_data_nodes,
        use_func_rel=args.use_func_rel,
        use_reverse_edges=args.use_reverse_edges,
        oversample_negatives=args.oversample_negatives
    )
    
    all_edge_types = args.target_edge_types or list(EDGE_TYPE_MAPPING.keys())
    type_data = {}
    
    for edge_type in all_edge_types:
        print(f"\n📊 Processing {edge_type}...")
        
        # Filter indices for this edge type
        valid_indices = filter_indices_by_edge_type(dataset, edge_type, args)
        if len(valid_indices) == 0:
            print(f"⚠️  No data found for {edge_type}, skipping...")
            type_data[edge_type] = None
            continue
        
        # Split indices
        train_indices, val_indices, test_indices = split_indices_for_task(
            valid_indices, args.train_ratio, args.val_ratio, args.seed, index_to_path
        )
        
        # Extract hashes from training set
        print(f"🔍 Extracting hashes from {len(train_indices)} training graphs...")
        trainjthash = set()
        trainitchash = set()
        trainicallhash = set()
        trainrethash = set()
        
        for i in tqdm(train_indices, desc=f"Extracting hashes for {edge_type}"):
            jt_set, ret_set, icall_set, itc_set = extract_funchash_sets(i, indextores)
            trainjthash.update(jt_set)
            trainitchash.update(itc_set)
            trainicallhash.update(icall_set)
            trainrethash.update(ret_set)
        
        # Store everything for this edge type
        type_data[edge_type] = {
            'train_hashes': {
                'jump': trainjthash,
                'itc': trainitchash,
                'icall': trainicallhash,
                'ret': trainrethash
            },
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'valid_indices_count': len(valid_indices)
        }
        
        print(f"✅ {edge_type}: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
    
    # Clean up dataset
    del dataset
    torch.cuda.empty_cache()
    
    print(f"\n✅ Hash extraction completed for all edge types!")
    return type_data, indextores

# def apply_duplicate_filtering(edge_type, type_data, indextores):
#     """
#     Apply duplicate filtering using pre-extracted hashes.
#     """
#     if type_data is None:
#         return
    
#     train_hashes = type_data['train_hashes']
#     val_indices = type_data['val_indices']
#     test_indices = type_data['test_indices']
    
#     # Filter validation set
#     for j in tqdm(val_indices, desc=f"Filtering validation set for {edge_type}", leave=False):
#         filter_and_save_val_jsons(
#             train_hashes['jump'],
#             train_hashes['ret'],
#             train_hashes['icall'],
#             train_hashes['itc'],
#             j,
#             indextores,
#             counter=lambda total, removed: global_counter_for_task(edge_type, total, removed)
#              global_counter=global_removed_hash_counter  # << add this param
#         )
    
#     # Filter test set
#     for k in tqdm(test_indices, desc=f"Filtering test set for {edge_type}", leave=False):
#         filter_and_save_val_jsons(
#             train_hashes['jump'],
#             train_hashes['ret'],
#             train_hashes['icall'],
#             train_hashes['itc'],
#             k,
#             indextores,
#             counter=lambda total, removed: global_counter_for_task(edge_type, total, removed)
#             global_counter=global_removed_hash_counter  # << add this param
#         )

def check_graph_features(dataset, indices, edge_types):
    """
    Check feature dimensions of graphs in the dataset
    """
    # Use the first graph to get feature dimensions
    data = dataset.load_item_multi_target(indices[0], "train", edge_types)
    graph = data['graph']
    
    # Check code node features
    code_features = {}
    if 'code' in graph.ntypes:
        for key in graph.nodes['code'].data.keys():
            shape = graph.nodes['code'].data[key].shape
            code_features[key] = shape
    
    # Check data node features
    data_features = {}
    if 'data' in graph.ntypes:
        for key in graph.nodes['data'].data.keys():
            shape = graph.nodes['data'].data[key].shape
            data_features[key] = shape
    
    return code_features, data_features if 'data' in graph.ntypes else None

def calculate_task_weights(dataset, train_indices, target_edge_types):
    """Calculate task weights based on data distribution"""
    task_counts = {et: 0 for et in target_edge_types}
    
    # Sample a subset of training data to estimate distributions
    sample_indices = train_indices[:min(100, len(train_indices))]
    
    for idx in sample_indices:
        try:
            data = dataset.load_item_multi_target(idx, "train", target_edge_types)
            for et in target_edge_types:
                if et in data['targets']:
                    task_counts[et] += data['targets'][et]['pos_count']
        except:
            continue
    
    # Calculate inverse frequency weights
    total_samples = sum(task_counts.values())
    if total_samples == 0:
        return {et: 1.0 for et in target_edge_types}
    
    task_weights = {}
    for et in target_edge_types:
        if task_counts[et] > 0:
            task_weights[et] = total_samples / (len(target_edge_types) * task_counts[et])
        else:
            task_weights[et] = 1.0
    
    print(f"📊 Task weights: {task_weights}")
    return task_weights

def calculate_per_task_macro_f1(task_metrics, target_edge_types):
    """
    Calculate macro F1 for each task and overall macro F1.
    
    Args:
        task_metrics: Dictionary with metrics for each task
        target_edge_types: List of target edge types
    
    Returns:
        dict: Contains per-task macro F1 and overall macro F1
    """
    per_task_f1 = {}
    valid_f1_scores = []
    
    for et in target_edge_types:
        if et in task_metrics and task_metrics[et]['support'] > 0:
            per_task_f1[et] = task_metrics[et]['f1']
            valid_f1_scores.append(task_metrics[et]['f1'])
        else:
            per_task_f1[et] = 0.0
    
    # Overall macro F1 (average of per-task F1 scores)
    overall_macro_f1 = sum(valid_f1_scores) / len(valid_f1_scores) if valid_f1_scores else 0.0
    
    return {
        'per_task_f1': per_task_f1,
        'overall_macro_f1': overall_macro_f1,
        'valid_tasks_count': len(valid_f1_scores)
    }

def load_hyperparameters_from_json(json_path):
    """
    Load hyperparameters from simplified JSON file.
    
    Expected JSON format:
    {
        "hidden_dim": 256,
        "num_heads": 8,
        "shared_layers": 4,
        "task_layers": 1,
        "dropout": 0.4,
        "weight_decay": 1e-5,
        "lr": 0.001,
        "grad_clip": 1.0,
        "epochs": 50
    }
    
    Args:
        json_path: Path to JSON file with hyperparameters
    
    Returns:
        dict: Hyperparameters
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Hyperparameter file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        hyperparams = json.load(f)
    
    print(f"✅ Loaded hyperparameters from {json_path}")
    return hyperparams

# def apply_hyperparameters_to_args(args, hyperparams):
#     """
#     Apply loaded hyperparameters to args object.
#     Supports JSON in Optuna format:
#     {
#       "best_f1": ...,
#       "params": {
#          "hidden_dim": ...,
#          ...
#       }
#     }
#     """
#     # If the JSON uses Optuna style with 'params' inside, extract it.
#     if "params" in hyperparams:
#         params = hyperparams["params"]
#     else:
#         params = hyperparams

#     param_mapping = {
#         'hidden_dim': 'hidden_dim',
#         'num_heads': 'num_heads',
#         'num_layers': 'num_layers',
#         'dropout': 'dropout',
#         'weight_decay': 'weight_decay',
#         'lr': 'lr',
#         'grad_clip': 'grad_clip',
#     }

#     print(f"📋 Applying hyperparameters:")
#     for json_key, value in params.items():
#         if json_key in param_mapping:
#             args_key = param_mapping[json_key]
#             old_value = getattr(args, args_key, None)
#             setattr(args, args_key, value)
#             print(f"  {args_key}: {old_value} → {value}")
#         else:
#             print(f"  ⚠️  Unknown hyperparameter: {json_key} = {value}")

#     # Make sure shared_layers and task_layers fall back to num_layers if needed
#     if not hasattr(args, 'shared_layers') or args.shared_layers is None:
#         args.shared_layers = args.num_layers
#         print(f"✅ shared_layers fallback: {args.shared_layers}")
    
#     if not hasattr(args, 'task_layers') or args.task_layers is None:
#         args.task_layers = args.num_layers
#         print(f"✅ task_layers fallback: {args.task_layers}")


#     return args

def apply_hyperparameters_to_args(args, hyperparams):
    """
    Apply loaded hyperparameters to args object.
    Supports JSON in Optuna format: { "best_f1": ..., "params": {...} }
    """

    if "params" in hyperparams:
        params = hyperparams["params"]
    else:
        params = hyperparams

    param_mapping = {
        'hidden_dim': 'hidden_dim',
        'num_heads': 'num_heads',
        'num_layers': 'num_layers',
        'dropout': 'dropout',
        'weight_decay': 'weight_decay',
        'lr': 'lr',
        'grad_clip': 'grad_clip',
        'shared_layers': 'shared_layers',
        'task_layers': 'task_layers'
    }

    print(f"📋 Applying hyperparameters:")
    for json_key, value in params.items():
        if json_key in param_mapping:
            args_key = param_mapping[json_key]
            old_value = getattr(args, args_key, None)
            setattr(args, args_key, value)
            print(f"  {args_key}: {old_value} → {value}")
        else:
            print(f"  ⚠️  Unknown hyperparameter: {json_key} = {value}")

    # ✅ NEW: force fallback for shared_layers + task_layers if not explicitly in JSON
    if 'shared_layers' not in params:
        args.shared_layers = args.num_layers
        print(f"✅ shared_layers fallback: {args.shared_layers}")

    if 'task_layers' not in params:
        # Preserve the command-line/model default (one layer per task) when a
        # hyperparameter file does not explicitly override task depth.
        args.task_layers = getattr(args, 'task_layers', 1)
        print(f"✅ task_layers fallback: {args.task_layers}")

    return args



def train_multi_task_model_joint(model, dataset, indices, target_edge_types, device, optimizer, epoch, args):
    """
    JOINT TRAINING: Train all edge types together in a single forward pass.
    This approach shares gradients across all tasks for better representation learning.
    """
    model.train()
    total_loss = 0
    processed_graphs = 0
    task_losses = {et: 0 for et in target_edge_types}
    task_losses["edge_type"] = 0
    
    random.shuffle(indices)
    
    for i, idx in enumerate(tqdm(indices, desc=f"Multi-task Joint Training - Epoch {epoch+1}", leave=False)):
        try:
            # Load data for all target edge types
            data = dataset.load_item_multi_target(idx, "train", target_edge_types)
            graph = data['graph']
            candidate_meta = data.get('candidate_meta')
            targets = data['targets']
            
            if graph is None or graph.num_nodes() == 0:
                continue
            
            graph = graph.to(device)
            
            # JOINT TRAINING: Combine all edge types together
            valid_tasks = []
            all_edges = []
            all_labels = []
            all_types = []
            
            for et in target_edge_types:
                if (et in targets and 
                    targets[et]['edges'].size(0) > 0 and 
                    targets[et]['pos_count'] > 0):
                    valid_tasks.append(et)
                    
                    target_data = targets[et]
                    edges = target_data['edges'].to(device)
                    labels = target_data['labels'].to(device)
                    
                    type_id = EDGE_TYPE_MAPPING[et]
                    
                    # Keep the task identity for both positives and that
                    # task's independently sampled negatives.  The positive
                    # mask for the auxiliary type head comes from labels.
                    edge_types = torch.full(
                        (edges.size(0),),
                        type_id,
                        dtype=torch.long,
                        device=device,
                    )
                    
                    all_edges.append(edges)
                    all_labels.append(labels)
                    all_types.append(edge_types)
            
            if not all_edges:
                continue
            
            # Concatenate all edge data
            combined_edges = torch.cat(all_edges, dim=0)
            combined_labels = torch.cat(all_labels, dim=0)
            combined_types = torch.cat(all_types, dim=0)
            
            # Single forward pass for ALL edge types
            edge_logits, edge_type_logits = model(
                graph,
                combined_edges,
                candidate_meta=candidate_meta,
            )
            
            # Calculate loss using your model's get_loss method
            loss, type_losses_dict = model.get_loss(
                edge_logits, combined_labels, combined_types, 
                pos_weight=1.0, edge_type_logits=edge_type_logits
            )
            
            # Single backward pass
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            
            total_loss += loss.item()
            for task_name, task_loss in type_losses_dict.items():
                if task_name in task_losses:
                    task_losses[task_name] += float(task_loss.detach().item())
            
            processed_graphs += 1
        # except RuntimeError as e:
        #     if "CUDA out of memory" in str(e) or "CUBLAS_STATUS_EXECUTION_FAILED" in str(e):
        #         print(f"[CUDA ERROR] Graph {i} (Index {idx}) caused CUDA error ({e}), skipping.")
        #         torch.cuda.empty_cache()
        #         continue
        #     else:
        #         print(f"[ERROR] Graph {idx}: {e}")
        #         continue
        # except Exception as e:
        #     print(f"[ERROR] Graph {idx}: {e}")
        #     continue
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] Graph {i} (Index {idx}) caused OOM, skipping.")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"[ERROR] Graph {idx}: {e}")
            continue
        
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / max(processed_graphs, 1)
    avg_task_losses = {k: v / max(processed_graphs, 1) for k, v in task_losses.items()}
    
    print(f"[Multi-task Joint] Training - Processed: {processed_graphs}, Loss: {avg_loss:.4f}")
    for task_name, task_loss in avg_task_losses.items():
        if task_loss > 0:
            print(f"  {task_name}: {task_loss:.4f}")
    
    torch.cuda.empty_cache()
    return {
        'loss': avg_loss,
        'task_losses': avg_task_losses,
        'processed_graphs': processed_graphs
    }

def evaluate_multi_task_model_per_task_macro_f1(model, dataset, indices, target_edge_types, device, args):
    """
    Enhanced evaluation function:
    - Calculates Macro F1 for each task separately.
    - Computes average evaluation loss for the whole eval set.
    """
    model.eval()

    # Store predictions & labels per task
    task_scores = {et: [] for et in target_edge_types}
    task_labels = {et: [] for et in target_edge_types}

    total_loss = 0.0
    processed_graphs = 0

    with torch.no_grad():
        for i, idx in enumerate(tqdm(indices, desc="Evaluating multi-task", leave=False)):
            try:
                data = dataset.load_item_multi_target(idx, "eval", target_edge_types)
                graph = data['graph']
                candidate_meta = data.get('candidate_meta')
                targets = data['targets']

                if graph is None or graph.num_nodes() == 0:
                    continue

                graph = graph.to(device)

                valid_tasks = []
                all_edges = []
                all_labels = []
                all_types = []

                for et in target_edge_types:
                    if (
                        et not in targets or
                        targets[et]['edges'].size(0) == 0 or
                        targets[et]['pos_count'] == 0
                    ):
                        continue

                    target_data = targets[et]
                    edges = target_data['edges'].to(device)
                    labels = target_data['labels'].to(device)
                    type_id = EDGE_TYPE_MAPPING[et]
                    pos_count = target_data['pos_count']

                    edge_types = torch.full((edges.size(0),), -1, dtype=torch.long, device=device)
                    edge_types[:pos_count] = type_id

                    all_edges.append(edges)
                    all_labels.append(labels)
                    all_types.append(edge_types)

                if not all_edges:
                    continue

                combined_edges = torch.cat(all_edges, dim=0)
                combined_labels = torch.cat(all_labels, dim=0)
                combined_types = torch.cat(all_types, dim=0)

                edge_logits, edge_type_logits = model(
                    graph,
                    combined_edges,
                    candidate_meta=candidate_meta,
                )

                # Compute loss (same as in training)
                loss, _ = model.get_loss(
                    edge_logits,
                    combined_labels,
                    combined_types,
                    pos_weight=1.0,
                    edge_type_logits=edge_type_logits
                )
                total_loss += loss.item()
                processed_graphs += 1

                # For metrics: store predictions per task
                for et in target_edge_types:
                    if (
                        et not in targets or
                        targets[et]['edges'].size(0) == 0 or
                        targets[et]['pos_count'] == 0
                    ):
                        continue

                    target_data = targets[et]
                    edges = target_data['edges'].to(device)
                    labels = target_data['labels'].to(device)

                    edge_logits_et, _ = model(graph, edges, candidate_meta=candidate_meta)
                    task_logits = edge_logits_et[et]
                    scores = torch.sigmoid(task_logits.squeeze())

                    if scores.dim() == 0:
                        scores = scores.unsqueeze(0)
                    if labels.dim() == 0:
                        labels = labels.unsqueeze(0)

                    task_scores[et].append(scores.detach())
                    task_labels[et].append(labels.detach())

            except torch.cuda.OutOfMemoryError:
                print(f"[OOM] Graph {i} (Index {idx}) caused OOM, skipping.")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"[ERROR] Graph {idx}: {e}")
                continue

            if (i + 1) % 20 == 0:
                torch.cuda.empty_cache()

    # Compute per-task metrics
    task_results = {}
    # individual_f1_scores = []
    individual_f1_scores = {et: [] for et in target_edge_types} 

    for et in target_edge_types:
        if len(task_scores[et]) == 0:
            task_results[et] = {
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'tp': 0, 'fp': 0, 'fn': 0, 'support': 0,
                'total_examples': 0, 'optimal_threshold': 0.5
            }
            continue

        all_scores = torch.cat(task_scores[et], dim=0)
        all_labels = torch.cat(task_labels[et], dim=0)

        # Use default 0.5 or sweep if you want
        thresholds = [0.5]
        best_f1 = 0.0
        best_threshold = 0.5
        best_metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'support': int(all_labels.sum().item()),
            'total_examples': len(all_labels),
            'optimal_threshold': best_threshold
        }

        for threshold in thresholds:
            predictions = (all_scores >= threshold).float()

            tp = ((predictions == 1) & (all_labels == 1)).sum().item()
            fp = ((predictions == 1) & (all_labels == 0)).sum().item()
            fn = ((predictions == 0) & (all_labels == 1)).sum().item()

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'support': tp + fn,
                    'total_examples': len(all_labels),
                    'optimal_threshold': threshold
                }

        task_results[et] = best_metrics
        if best_f1 > 0:
            individual_f1_scores[et].append(best_f1)

    macro_f1_results = calculate_per_task_macro_f1(task_results, target_edge_types)
    avg_eval_loss = total_loss / max(processed_graphs, 1)

    torch.cuda.empty_cache()

    return {
        'task_metrics': task_results,
        'per_task_f1': macro_f1_results['per_task_f1'],
        'overall_macro_f1': macro_f1_results['overall_macro_f1'],
        'valid_tasks_count': macro_f1_results['valid_tasks_count'],
        'total_support': sum(task_results[et]['support'] for et in target_edge_types if et in task_results),
        'individual_f1_scores': individual_f1_scores,
        'eval_loss': avg_eval_loss
    }

def run_multi_task_joint_training(args, target_edge_types, type_data=None, indextores=None):
    """
    Multi-task training with joint training and per-task macro F1 evaluation.
    Now logs train loss per task and eval loss per epoch.
    """
    use_cuda = torch.cuda.is_available() and args.gpu >= 0
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(args.gpu)
    print(f"Using device: {device}")
    set_seed(args.seed)

    print(f"\n{'='*80}")
    print(f"JOINT MULTI-TASK TRAINING FOR EDGE TYPES: {target_edge_types}")
    print(f"EVALUATION: PER-TASK MACRO F1")
    print(f"{'='*80}")

    # === Prepare dataset ===
    dataset = JumpGraphDataset(
        graph_dir=args.graph_dir,
        neg_multiplier=args.neg_multiplier if args.oversample_negatives else 1,
        use_data_nodes=args.use_data_nodes,
        use_func_rel=args.use_func_rel,
        use_reverse_edges=args.use_reverse_edges,
        oversample_negatives=args.oversample_negatives
    )

    # === Load splits ===
    edge_type_order = ['tailcall', 'indirectcall', 'jumptable', 'ret']
    basedir = os.path.dirname(args.graph_dir)
    outputbase = os.path.dirname(args.output_dir)
    split_cache_path = args.split_cache_path or os.path.join(outputbase, "split_by_type.json")
    package_mapping = load_binary_to_package_or_empty(basedir)

    bintoindex_path = os.path.join(basedir, "bintoindex.json")
    with open(bintoindex_path, 'r') as f:
        bintoindex = json.load(f)

    indextopath = {str(v): str(k) for k, v in bintoindex.items()}

    # train_indices, val_indices, test_indices, per_type_split = multi_label_proportional_split_by_package_and_save(
    #     edge_type_order, dataset, args, split_cache_path, indextopath, package_mapping
    # )
    if args.train_all_task_graphs:
        train_indices = list(dataset.graph_files)
        val_indices = []
        test_indices = []
        per_type_split = {}
        print(
            f"[INFO] Using all {len(train_indices)} sample graphs for training; "
            "internal validation/test splits are disabled."
        )
    else:
        train_indices, val_indices, test_indices, per_type_split = load_or_create_split_by_package(
            edge_type_order, dataset, args, split_cache_path, indextopath, package_mapping
        )


    # === Duplicate removal ===
    basedir = os.path.dirname(args.graph_dir)
    indextores_path = os.path.join(basedir, "indextores.json")
    with open(indextores_path, 'r') as f:
        indextores = json.load(f)

    if args.skip_duplicate_filtering:
        print("[INFO] Skipping duplicate filtering for this run by request.")
    else:
        selected_indices = train_indices + val_indices + test_indices
        sidecar_count = count_function_hash_sidecars(selected_indices, indextores)
        if sidecar_count == 0:
            print(
                "[INFO] No function-hash sidecars found for the selected splits; "
                "duplicate filtering is skipped automatically."
            )
        else:
            print(f"[INFO] Found {sidecar_count} function-hash sidecars in the selected splits.")
            for edge_type in target_edge_types:
                process_duplicate_removal_for_split(
                    train_indices, val_indices, test_indices, indextores, edge_type
                )

    # === Model feature dims ===
    sample_data = dataset.load_item_multi_target(train_indices[0], "train", target_edge_types)
    sample_graph = sample_data['graph']

    code_feat_dim = None
    data_feat_dim = 1

    if 'code' in sample_graph.ntypes:
        if 'featmean' in sample_graph.nodes['code'].data:
            code_feat_dim = sample_graph.nodes['code'].data['featmean'].shape[1]
        elif 'feat' in sample_graph.nodes['code'].data:
            code_feat_dim = sample_graph.nodes['code'].data['feat'].shape[1]

    if 'data' in sample_graph.ntypes and sample_graph.num_nodes('data') > 0:
        data_feat_dim = sample_graph.nodes['data'].data['feat'].shape[1]

    # === Model ===
    task_weights = {et: 1.0 for et in target_edge_types}

    model = MultiTaskGAT(
        code_feat_dim=code_feat_dim,
        data_feat_dim=data_feat_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        shared_layers=getattr(args, 'shared_layers', args.num_layers),
        task_layers=getattr(args, 'task_layers', args.num_layers),
        dropout=args.dropout,
        use_reverse_edges=args.use_reverse_edges,
        task_weights=task_weights,
        use_gch=args.use_gch,
        use_gdh=args.use_gdh,
        task_aware_routing=not args.disable_task_aware_routing,
        gdh_radius=args.gdh_radius,
    ).to(device)
    
    
    print("\n🚀 [INFO] Model hyperparameters in use:")
    print(f"  code_feat_dim: {code_feat_dim}")
    print(f"  data_feat_dim: {data_feat_dim}")
    print(f"  hidden_dim: {args.hidden_dim}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  shared_layers: {getattr(args, 'shared_layers', args.num_layers)}")
    print(f"  task_layers: {getattr(args, 'task_layers', args.num_layers)}")
    print(f"  dropout: {args.dropout}")
    print(f"  use_reverse_edges: {args.use_reverse_edges}")
    print(f"  use_gch: {args.use_gch}")
    print(f"  use_gdh: {args.use_gdh}")
    print(f"  task_aware_routing: {not args.disable_task_aware_routing}")
    print(f"  task_weights: {task_weights}")
    if getattr(args, "skip_validation", False):
        print("  train_only_skip_validation: True")
    print("")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=args.scheduler_patience, verbose=True, threshold=0.01
    )

    trial_id = getattr(args, "trial_id", "preset")
    task_output_dir = os.path.join(args.output_dir, f"trial_{trial_id}", "multi_task_" + "_".join(target_edge_types))
    os.makedirs(task_output_dir, exist_ok=True)

    best_model_path = os.path.join(task_output_dir, f"best_model_trial_{trial_id}.pt")
    latest_model_path = os.path.join(task_output_dir, f"latest_model_trial_{trial_id}.pt")
    checkpoint_path = os.path.join(task_output_dir, f"checkpoint_trial_{trial_id}.pt")

    start_epoch = 0
    best_overall_macro_f1 = 0.0

    if os.path.exists(checkpoint_path):
        print(f"[Resume] Found checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        stored_gdh_radius = checkpoint.get('gdh_radius')
        if (
            stored_gdh_radius is not None
            and int(stored_gdh_radius) != int(args.gdh_radius)
        ):
            raise RuntimeError(
                f"Cannot resume GDH radius {stored_gdh_radius} checkpoint with "
                f"radius {args.gdh_radius}."
            )
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_overall_macro_f1 = checkpoint.get('best_overall_macro_f1', 0.0)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"[Resume] Resuming from epoch {start_epoch} with best Macro F1 so far {best_overall_macro_f1:.4f}")
    else:
        print("[Resume] No checkpoint found — starting fresh.")

    # === Files ===
    metrics_path = os.path.join(task_output_dir, "metrics.txt")
    losses_csv_path = os.path.join(task_output_dir, "losses.csv")
    best_f1_path = os.path.join(task_output_dir, "best_f1.json")
    train_only_status_path = os.path.join(task_output_dir, "train_only_status.json")

    # Init CSV header
    with open(losses_csv_path, 'a') as f:
        f.write("epoch," + ",".join([f"{et}_loss" for et in target_edge_types]) + ",eval_loss" + ",train_loss\n")

    # Init metrics.txt header
    if not os.path.exists(metrics_path):
        with open(metrics_path, 'a') as f:
            header = "epoch,train_loss," + \
                     ",".join([f"{et}_f1,{et}_precision,{et}_recall,{et}_support,{et}_threshold" for et in target_edge_types]) + \
                     ",overall_macro_f1,valid_tasks_count,eval_loss\n"
            f.write(header)

    patience_counter = 0

    if not args.eval_only:
        for epoch in range(start_epoch, args.epochs):
            train_metrics = train_multi_task_model_joint(
                model, dataset, train_indices, target_edge_types, device, optimizer, epoch, args
            )

            if getattr(args, "skip_validation", False):
                epoch_model_path = os.path.join(
                    task_output_dir,
                    f"model_epoch_{epoch + 1:03d}_trial_{trial_id}.pt"
                )
                total_train_loss = 0.0
                with open(losses_csv_path, 'a') as f:
                    row = [str(epoch + 1)]
                    for et in target_edge_types:
                        loss_val = train_metrics['task_losses'].get(et, 0.0)
                        row.append(f"{loss_val:.6f}")
                        total_train_loss += loss_val
                    row.append("nan")
                    row.append(f"{total_train_loss:.6f}")
                    f.write(",".join(row) + "\n")

                save_model_artifact(epoch_model_path, model)
                save_model_artifact(latest_model_path, model)
                save_model_artifact(best_model_path, model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_overall_macro_f1': best_overall_macro_f1,
                    'skip_validation': True,
                    'gdh_radius': args.gdh_radius,
                }, checkpoint_path)

                train_only_status = {
                    'mode': 'train_only_skip_validation',
                    'epoch': epoch + 1,
                    'epochs': args.epochs,
                    'train_loss': train_metrics['loss'],
                    'processed_graphs': train_metrics['processed_graphs'],
                    'epoch_model': os.path.basename(epoch_model_path),
                    'latest_model': os.path.basename(latest_model_path),
                    'compat_model': os.path.basename(best_model_path),
                    'checkpoint': os.path.basename(checkpoint_path),
                    'target_edge_types': target_edge_types,
                    'hyperparameters': {
                        'hidden_dim': args.hidden_dim,
                        'num_heads': args.num_heads,
                        'shared_layers': getattr(args, 'shared_layers', args.num_layers),
                        'task_layers': getattr(args, 'task_layers', args.num_layers),
                        'dropout': args.dropout,
                        'weight_decay': args.weight_decay,
                        'lr': args.lr,
                        'gdh_radius': args.gdh_radius,
                    }
                }
                with open(train_only_status_path, 'w') as f:
                    json.dump(train_only_status, f, indent=2)

                print(
                    f"[Train Only] Epoch {epoch + 1}/{args.epochs} saved without validation. "
                    f"Loss: {train_metrics['loss']:.6f}"
                )
                continue

            eval_results = evaluate_multi_task_model_per_task_macro_f1(
                model, dataset, val_indices, target_edge_types, device, args
            )

            overall_macro_f1 = eval_results['overall_macro_f1']
            eval_loss = eval_results['eval_loss']
            scheduler.step(overall_macro_f1)

            # === Write losses.csv ===
            with open(losses_csv_path, 'a') as f:
                row = [str(epoch + 1)]
                total_train_loss = 0.0
                for et in target_edge_types:
                    loss_val = train_metrics['task_losses'].get(et, 0.0)
                    row.append(f"{loss_val:.6f}")
                    total_train_loss += loss_val
                row.append(f"{eval_loss:.6f}")
                row.append(f"{total_train_loss:.6f}")   
                f.write(",".join(row) + "\n")

            # === Write metrics.txt ===
            with open(metrics_path, 'a') as f:
                row = [f"{epoch+1}", f"{train_metrics['loss']:.6f}"]
                for et in target_edge_types:
                    m = eval_results['task_metrics'].get(et, {})
                    row.extend([
                        f"{m.get('f1', 0):.4f}",
                        f"{m.get('precision', 0):.4f}",
                        f"{m.get('recall', 0):.4f}",
                        f"{m.get('support', 0)}",
                        f"{m.get('optimal_threshold', 0.5):.3f}"
                    ])
                row.extend([f"{overall_macro_f1:.4f}", f"{eval_results['valid_tasks_count']}", f"{eval_loss:.6f}"])
                f.write(",".join(row) + "\n")

            # === Logging ===
            print(f"[Epoch {epoch+1}] Overall Macro F1: {overall_macro_f1:.4f} | Eval Loss: {eval_loss:.6f}")
            for et in target_edge_types:
                m = eval_results['task_metrics'].get(et, {})
                print(f"  {et}: F1={m.get('f1', 0):.4f}, P={m.get('precision', 0):.4f}, R={m.get('recall', 0):.4f}, Sup={m.get('support', 0)}, Thresh={m.get('optimal_threshold', 0.5):.2f}")

            # === Save best ===
            is_best_epoch = overall_macro_f1 > best_overall_macro_f1
            if is_best_epoch:
                best_overall_macro_f1 = overall_macro_f1
                patience_counter = 0

                best_f1_record = {
                    "best_overall_macro_f1": best_overall_macro_f1,
                    "best_per_task_f1": eval_results['per_task_f1']
                }
                with open(best_f1_path, 'w') as f:
                    json.dump(best_f1_record, f, indent=2)

                save_model_artifact(best_model_path, model)
                print(f"[Best Model] 🎉 New best F1: {overall_macro_f1:.4f} — saved!")

            else:
                patience_counter += 1
                if patience_counter >= args.early_stopping_patience:
                    print("[Early Stop] Patience exceeded.")
                    break

            if getattr(args, "save_all_epoch_models", False):
                epoch_model_path = os.path.join(
                    task_output_dir,
                    f"model_epoch_{epoch + 1:03d}_trial_{trial_id}.pt",
                )
                epoch_metrics_path = os.path.join(
                    task_output_dir,
                    f"metrics_epoch_{epoch + 1:03d}_trial_{trial_id}.json",
                )
                save_model_artifact(epoch_model_path, model)
                with open(epoch_metrics_path, "w") as f:
                    json.dump(
                        {
                            "epoch": epoch + 1,
                            "is_best_validation_epoch": is_best_epoch,
                            "best_validation_macro_f1_so_far": best_overall_macro_f1,
                            "train": train_metrics,
                            "validation": eval_results,
                        },
                        f,
                        indent=2,
                        default=lambda value: value.item()
                        if hasattr(value, "item")
                        else str(value),
                    )
                print(
                    f"[Epoch Archive] Saved epoch {epoch + 1} model and metrics: "
                    f"{epoch_model_path}"
                )

            # === Always save checkpoint ===
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_overall_macro_f1': best_overall_macro_f1,
                'gdh_radius': args.gdh_radius,
            }, checkpoint_path)
            print(f"[Checkpoint] ✅ Saved at epoch {epoch + 1}")

    if getattr(args, "skip_validation", False):
        print("[Train Only] Skipping final test evaluation. Use the saved model for external evaluation.")
        return {
            'status': 'completed_train_only',
            'num_graphs': {
                'train': len(train_indices),
                'val': len(val_indices),
                'test': len(test_indices)
            },
            'target_edge_types': target_edge_types,
            'latest_model_path': latest_model_path,
            'checkpoint_path': checkpoint_path
        }

    # === Final test ===
    if os.path.exists(best_model_path):
        load_model_artifact(model, best_model_path, map_location=device)
        print("[Final Eval] Loaded best model weights for test set.")

    print("[Final Evaluation] Running test set evaluation...")
    test_results = evaluate_multi_task_model_per_task_macro_f1(
        model, dataset, test_indices, target_edge_types, device, args
    )

    return {
        'status': 'completed',
        'test_metrics': test_results['task_metrics'],
        'overall_macro_f1': test_results['overall_macro_f1'],
        'per_task_f1': test_results['per_task_f1'],
        'valid_tasks_count': test_results['valid_tasks_count'],
        'total_support': test_results['total_support'],
        'individual_f1_scores': test_results['individual_f1_scores'],
        'num_graphs': {
            'train': len(train_indices),
            'val': len(val_indices),
            'test': len(test_indices)
        },
        'best_overall_macro_f1': best_overall_macro_f1,
        'target_edge_types': target_edge_types
    }



def run_multi_task_training_with_preset_hyperparams(args):
    """
    Run multi-task training with preset hyperparameters from JSON file.
    """
    print(f"\n{'='*80}")
    print(f"🔧 LOADING PRESET HYPERPARAMETERS")
    print(f"{'='*80}")
    
    # Load hyperparameters
    try:
        hyperparams = load_hyperparameters_from_json(args.hyperparams_json)
        
        # Apply to args
        args = apply_hyperparameters_to_args(args, hyperparams)
        
        print(f"✅ Hyperparameters loaded and applied successfully!")
        
    except Exception as e:
        print(f"❌ Error loading hyperparameters: {e}")
        print(f"🔄 Falling back to command-line arguments...")
    
    # Run training with the loaded hyperparameters
    print(f"\n{'='*80}")
    print(f"🚀 STARTING JOINT TRAINING WITH PRESET HYPERPARAMETERS")
    print(f"Target Edge Types: {args.target_edge_types}")
    print(f"{'='*80}")
    
    result = run_multi_task_joint_training(args, args.target_edge_types)
    
    # Enhanced result logging
    print(f"\n{'='*80}")
    print(f"🎯 TRAINING COMPLETED WITH PRESET HYPERPARAMETERS")
    print(f"{'='*80}")
    print(f"📊 Final Overall Macro F1: {result.get('overall_macro_f1', 0):.4f}")
    print(f"📊 Target Edge Types: {args.target_edge_types}")
    
    # Save hyperparameters used
    trial_id = getattr(args, "trial_id", "preset")
    hyperparams_used_path = os.path.join(args.output_dir, f"trial_{trial_id}", "hyperparams_used.json")
    os.makedirs(os.path.dirname(hyperparams_used_path), exist_ok=True)
    
    hyperparams_used = {
        'source_file': args.hyperparams_json,
        'target_edge_types': args.target_edge_types,
        'training_method': 'joint',
        'evaluation_method': 'per_task_macro_f1',
        'hyperparameters': {
            'hidden_dim': args.hidden_dim,
            'num_heads': args.num_heads,
            'shared_layers': args.shared_layers,
            'task_layers': args.task_layers,
            'dropout': args.dropout,
            'weight_decay': args.weight_decay,
            'lr': args.lr,
            'grad_clip': args.grad_clip,
            'neg_multiplier': args.neg_multiplier,
            'epochs': args.epochs
        },
        'results': {
            'overall_macro_f1': result.get('overall_macro_f1', 0),
            'per_task_f1': result.get('per_task_f1', {}),
            'status': result.get('status', 'unknown')
        }
    }
    
    with open(hyperparams_used_path, 'w') as f:
        json.dump(hyperparams_used, f, indent=2)
    
    print(f"💾 Hyperparameters and results saved to: {hyperparams_used_path}")
    
    return result

def run_multi_task_training(args):
    """
    Main multi-task training function
    """
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    torch.cuda.set_device(args.gpu)
    print(f"Using device: {device}")
    set_seed(args.seed)
    
    # Validate target edge types
    all_edge_types = list(EDGE_TYPE_MAPPING.keys())
    if args.target_edge_types:
        invalid_types = [et for et in args.target_edge_types if et not in all_edge_types]
        if invalid_types:
            print(f"ERROR: Invalid edge types specified: {invalid_types}")
            return {"f1": 0.0}
        if len(args.target_edge_types) < 2:
            print("ERROR: Multi-task training requires at least 2 edge types")
            return {"f1": 0.0}
        target_edge_types = args.target_edge_types
    else:
        print("ERROR: Must specify target edge types for multi-task training")
        return {"f1": 0.0}
    
    print(f"Joint multi-task training for edge types: {target_edge_types}")
    print(f"Evaluation method: Per-task Macro F1")
    
    # Run multi-task training
    result = run_multi_task_joint_training(args, target_edge_types)
    
    # Save summary
    trial_id = getattr(args, "trial_id", "default")
    summary_path = os.path.join(args.output_dir, f"trial_{trial_id}", "multi_task_summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'a') as f:
        f.write(f"Joint Multi-Task Training Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Training Method: Joint\n")
        f.write(f"Evaluation Method: Per-Task Macro F1\n")
        f.write(f"Target Edge Types: {target_edge_types}\n")
        f.write(f"Status: {result.get('status', 'unknown')}\n")
        f.write(f"Final Overall Macro F1: {result.get('overall_macro_f1', 0):.4f}\n")
        f.write(f"Best Overall Macro F1: {result.get('best_overall_macro_f1', 0):.4f}\n")
        f.write(f"Valid Tasks: {result.get('valid_tasks_count', 0)}/{len(target_edge_types)}\n")
        f.write(f"\nPer-Task F1 Scores:\n")
        
        if 'per_task_f1' in result:
            for et in target_edge_types:
                f1_score = result['per_task_f1'].get(et, 0.0)
                f.write(f"{et}: {f1_score:.4f}\n")
        
        f.write(f"\nDataset Sizes:\n")
        if 'num_graphs' in result:
            ng = result['num_graphs']
            f.write(f"Train: {ng.get('train', 0)}\n")
            f.write(f"Val: {ng.get('val', 0)}\n")
            f.write(f"Test: {ng.get('test', 0)}\n")
    
    print(f"Multi-task training summary saved to {summary_path}")
    return result

# def main(args):
#     """Main function supporting joint multi-task training with per-task macro F1 and preset hyperparameters"""
#     if hasattr(args, 'multi_task') and args.multi_task:
#         print(f"[INFO] Running joint multi-task training with per-task Macro F1 evaluation...")
#         if not args.target_edge_types or len(args.target_edge_types) < 2:
#             print("❌ Multi-task training requires at least 2 target edge types")
#             return
        
#         # Check if preset hyperparameters are provided
#         if hasattr(args, 'hyperparams_json') and args.hyperparams_json:
#             print(f"[INFO] Using preset hyperparameters from: {args.hyperparams_json}")
#             result = run_multi_task_training_with_preset_hyperparams(args)
#         else:
#             print(f"[INFO] Running joint multi-task training with command-line hyperparameters...")
#             result = run_multi_task_training(args)
#             print(f"Joint multi-task training completed.")
#             print(f"📊 Final Overall Macro F1: {result.get('overall_macro_f1', 0):.4f}")
#     else:
#         print("❌ This script is for multi-task training only. Use --multi_task flag.")
#         print("For single-task training, use train_single_each.py")

def main(args):
    """
    Main entry for ONE joint multi-task training using hyperparameters JSON if provided.
    """
    print(f"[INFO] Starting joint multi-task training with per-task Macro F1 evaluation.")

    # === Use hyperparameters from JSON if provided ===
    if getattr(args, "hyperparams_json", None):
        print(f"[INFO] Loading preset hyperparameters from: {args.hyperparams_json}")
        hyperparams = load_hyperparameters_from_json(args.hyperparams_json)
        args = apply_hyperparameters_to_args(args, hyperparams)
    else:
        print(f"[INFO] No preset hyperparameters given — using defaults from command line.")

    # === Validate target edge types ===
    if not args.target_edge_types or len(args.target_edge_types) < 2:
        print(f"[ERROR] Multi-task training needs at least 2 edge types.")
        return

    # === Run training once ===
    result = run_multi_task_joint_training(args, args.target_edge_types)

    # === Save final summary ===
    trial_id = getattr(args, "trial_id", "summary")
    summary_path = os.path.join(args.output_dir, f"trial_{trial_id}", "multi_task_summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    # with open(summary_path, "w") as f:
    #     f.write(f"Joint Multi-Task Training Summary\n")
    #     f.write(f"{'='*50}\n")
    #     f.write(f"Target Edge Types: {args.target_edge_types}\n")
    #     f.write(f"Best Overall Macro F1: {result.get('best_overall_macro_f1', 0):.4f}\n")
    #     f.write(f"Per-task F1:\n")
    #     for et, f1 in result.get("per_task_f1", {}).items():
    #         f.write(f"{et}: {f1:.4f}\n")

    # print(f"[INFO] Training done. Summary saved to {summary_path}.")

    with open(summary_path, "w") as f:
        f.write(f"Joint Multi-Task Training Summary\n")
        f.write(f"{'='*50}\n\n")

        # Target edge types
        f.write(f"Target Edge Types:\n")
        for et in result.get('target_edge_types', []):
            f.write(f"  - {et}\n")
        f.write("\n")

        # Macro F1
        f.write(f"Best Overall Macro F1: {result.get('best_overall_macro_f1', 0):.4f}\n\n")

        # Per-task F1
        f.write(f"Per-task F1:\n")
        for et, f1 in result.get("per_task_f1", {}).items():
            f.write(f"  {et}: {f1:.4f}\n")
        f.write("\n")

        # Detailed test metrics per task
        f.write(f"Detailed Test Metrics per Task:\n")
        for et, metrics in result.get("test_metrics", {}).items():
            f.write(f"  {et}:\n")
            for metric_name, value in metrics.items():
                f.write(f"    {metric_name}: {value:.4f}\n")
        f.write("\n")

        # Individual F1 scores (all sites)
        f.write(f"Individual F1 Scores:\n")
        for et, scores in result.get("individual_f1_scores", {}).items():
            f.write(f"  {et}:\n")
            for i, score in enumerate(scores):
                f.write(f"    [{i}]: {score:.4f}\n")
        f.write("\n")

        # Valid tasks count
        f.write(f"Valid Tasks Count: {result.get('valid_tasks_count', 0)}\n\n")

        # Total support (number of edges considered)
        f.write(f"Total Support (Edge Count): {result.get('total_support', 0)}\n\n")

        # Number of graphs by split
        f.write(f"Number of Graphs:\n")
        for split, count in result.get("num_graphs", {}).items():
            f.write(f"  {split}: {count}\n")
        f.write("\n")

    print(f"[INFO] Training done. Summary saved to {summary_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint multi-task training with per-task Macro F1 evaluation")
    
    # Data options
    parser.add_argument('--graph_dir', type=str, required=True,
                      help='Directory containing graph files')
    parser.add_argument('--output_dir', type=str, default='./multi-task',
                      help='Directory to save output files')
    parser.add_argument('--use_data_nodes', action='store_true', default=True,
                      help='Use data nodes in the graph')
    parser.add_argument('--use_func_rel', action='store_true', default=False,
                      help='Use code2funchead relations (disabled by default)')
    parser.add_argument('--use_reverse_edges', action='store_true', default=False,
                      help='Use reverse edges in the graph to improve message passing')
    parser.add_argument('--use_gch', action=argparse.BooleanOptionalAction, default=True,
                      help='Use Global Code Hub routing (enabled by default; disabling is an ablation)')
    parser.add_argument('--use_gdh', action=argparse.BooleanOptionalAction, default=True,
                      help='Use Global Data Hub routing (enabled by default; disabling is an ablation)')
    parser.add_argument('--gdh_radius', type=int, choices=(2,), default=2,
                      help='xRef-only GDH neighborhood radius (fixed to 2)')
    parser.add_argument('--disable_task_aware_routing', action='store_true', default=False,
                      help='Disable task-aware routing and include jumptable destination candidates')
    parser.add_argument('--split_cache_path', type=str, default=None,
                      help='Optional split_by_type.json path to reuse for train/val/test splits')
    parser.add_argument('--force_recreate_split', action='store_true', default=False,
                      help='Recreate the package-level split even if split_cache_path already exists')
    parser.add_argument('--skip_duplicate_filtering', action='store_true', default=False,
                      help='Skip function-hash duplicate filtering even when sidecars are available')
    
    # Model options
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Hidden dimension for the model')
    parser.add_argument('--num_heads', type=int, default=8,
                      help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                      help='Number of GAT layers (used as shared_layers if not specified)')
    parser.add_argument('--shared_layers', type=int, default=4,
                      help='Number of shared encoder layers')
    parser.add_argument('--task_layers', type=int, default=1,
                      help='Number of task-specific layers')
    parser.add_argument('--dropout', type=float, default=0.4,
                      help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay for regularization')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                      help='Gradient clipping threshold (0 to disable)')
    parser.add_argument('--oversample_negatives', action='store_true', default=False,
                      help='Use more negative samples than positive ones')
    parser.add_argument('--neg_multiplier', type=int, default=2,
                      help='Multiplier for negative samples when oversampling')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                      help='Ratio of data to use for training')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                      help='Ratio of data to use for validation')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                      help='Ratio of data to use for testing')
    parser.add_argument(
    "--data_percentage",
    type=float,
    default=100.0,
    help="Percentage of each split to use, in (0, 100]; tiny values retain at least one graph per requested task (default: 100%%)"
    )
    parser.add_argument(
    "--data_fraction",
    type=float,
    default=None,
    help="Fraction of each split to use, in (0, 1]; overrides --data_percentage"
    )
 
    # Early stopping and scheduling
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                      help='Epochs to wait before early stopping')
    parser.add_argument('--scheduler_patience', type=int, default=3,
                      help='Epochs to wait before reducing learning rate')
    
    # Edge type selection - REQUIRED for multi-task
    # parser.add_argument('--target_edge_types', type=str, nargs='+', required=True,
                    #   help='Target edge types for multi-task training (e.g., --target_edge_types call jump ret)')
    parser.add_argument(
    "--target_edge_types",
    nargs="+",
    default=["ret", "jumptable","indirectcall","tailcall"],
    help="List of edge types to predict"
    )
    
    # Multi-task specific
    parser.add_argument('--multi_task', action='store_true', default=True,
                      help='Enable multi-task learning mode (default for this script)')
    
    # PRESET HYPERPARAMETERS
    parser.add_argument('--hyperparams_json', type=str, default=None,
                      help='Path to JSON file with preset hyperparameters')
    
    # Evaluation options
    parser.add_argument('--eval_only', action='store_true',
                      help='Only evaluate the model, no training')
    parser.add_argument('--skip_validation', action='store_true',
                      help='Train only: skip per-epoch validation and final test evaluation, saving latest model each epoch')
    parser.add_argument('--train_all_task_graphs', action='store_true', default=False,
                      help='Use every graph in graph_dir for training without creating validation/test splits; use with --skip_validation')
    parser.add_argument('--save_all_epoch_models', action='store_true', default=False,
                      help='Save a versioned model artifact and validation metrics for every normal training epoch')
    
    # Other options
    parser.add_argument('--gpu', type=int, default=1,
                      help='GPU device ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)
