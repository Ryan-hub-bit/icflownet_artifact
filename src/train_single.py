"""Train SingleTaskGAT models, one per selected edge type."""

import os
import argparse
import random
import subprocess
import sys
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from tqdm import tqdm
import json
import hashlib
from collections import defaultdict
from loadgraph import JumpGraphDataset  # Your new single-task dataset
from common import (
    EDGE_TYPE_MAPPING,
    EDGE_TYPE_TO_FNHASH,
    load_model_artifact,
    save_model_artifact,
)
from single_task_model import SingleTaskGAT
from util.removeduplicate import (
    count_function_hash_sidecars,
    extract_funchash_sets,
    filter_and_save_val_jsons,
)

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Dictionary to store per-edge-type counters
site_stats = {
    "total_sites": {},
    "removed_sites": {}
}

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

def dataset_graph_ids(dataset):
    """Return the real graph file ids rather than assuming ids are contiguous."""
    graph_files = getattr(dataset, "graph_files", None)
    if not graph_files:
        return list(range(len(dataset)))

    graph_ids = []
    for graph_id in graph_files:
        graph_id_str = str(graph_id)
        graph_ids.append(int(graph_id_str) if graph_id_str.isdigit() else graph_id_str)
    return graph_ids

def graph_id_signature(graph_ids):
    """Build a compact cache signature for sparse graph-id datasets."""
    graph_ids_as_str = [str(graph_id) for graph_id in graph_ids]
    joined = "\n".join(graph_ids_as_str)
    return {
        "count": len(graph_ids_as_str),
        "first": graph_ids_as_str[0] if graph_ids_as_str else None,
        "last": graph_ids_as_str[-1] if graph_ids_as_str else None,
        "sha256": hashlib.sha256(joined.encode("utf-8")).hexdigest(),
    }


def effective_hub_flags(args, edge_type):
    """Resolve paper defaults while preserving explicit hub ablations."""
    paper_default = edge_type != "jumptable"
    return (
        paper_default if args.use_gch is None else args.use_gch,
        paper_default if args.use_gdh is None else args.use_gdh,
    )

def save_train_only_snapshot(model, optimizer, task_output_dir, edge_type, epoch, train_metrics, args):
    """Save train-only checkpoints without relying on validation metrics."""
    epoch_num = epoch + 1
    use_gch, use_gdh = effective_hub_flags(args, edge_type)
    epoch_model = os.path.join(task_output_dir, f"model_epoch_{epoch_num:03d}.pt")
    latest_model = os.path.join(task_output_dir, "latest_model.pt")
    best_model = os.path.join(task_output_dir, "best_model.pt")
    checkpoint_path = os.path.join(task_output_dir, "checkpoint.pt")
    status_path = os.path.join(task_output_dir, "train_only_status.json")

    save_model_artifact(epoch_model, model)
    epoch_models_only = getattr(args, "save_epoch_models_only", False)
    if not epoch_models_only:
        save_model_artifact(latest_model, model)
        save_model_artifact(best_model, model)
        torch.save({
            "epoch": epoch_num,
            "task_type": edge_type,
            "gdh_radius": args.gdh_radius,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "hyperparameters": {
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "lr": args.lr,
                "use_gch": use_gch,
                "use_gdh": use_gdh,
                "task_aware_routing": not args.disable_task_aware_routing,
                "gdh_radius": args.gdh_radius,
            },
        }, checkpoint_path)

    with open(status_path, "w") as f:
        json.dump({
            "status": "train_only",
            "task_type": edge_type,
            "latest_epoch": epoch_num,
            "latest_model": os.path.basename(epoch_model),
            "best_model_alias": None if epoch_models_only else os.path.basename(best_model),
            "checkpoint": None if epoch_models_only else os.path.basename(checkpoint_path),
            "last_epoch_model": os.path.basename(epoch_model),
            "epoch_models_only": epoch_models_only,
            "train_metrics": train_metrics,
        }, f, indent=2)

    return {
        "epoch_model": os.path.basename(epoch_model),
        "latest_model": os.path.basename(epoch_model),
        "best_model_alias": None if epoch_models_only else os.path.basename(best_model),
        "checkpoint": None if epoch_models_only else os.path.basename(checkpoint_path),
    }


def run_clean_test_evaluation(checkpoint_path, task_output_dir, edge_type, epoch, args):
    """Evaluate one epoch checkpoint on the external clean test."""
    epoch_num = epoch + 1
    eval_dir = os.path.join(task_output_dir, "clean_eval", f"epoch_{epoch_num:03d}")
    os.makedirs(eval_dir, exist_ok=True)
    if args.clean_eval_scope == "both":
        output_path = eval_dir
    else:
        output_path = os.path.join(eval_dir, "result.json")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    evaluator_path = os.path.join(repo_root, "scripts", "eval_single_cleantest.py")
    command = [
        sys.executable,
        evaluator_path,
        "--checkpoint", os.path.abspath(checkpoint_path),
        "--clean_root", os.path.abspath(args.clean_eval_root),
        "--task", edge_type,
        "--scope", args.clean_eval_scope,
        "--output", output_path,
        "--gpu", str(args.clean_eval_gpu),
        "--log_every", str(args.clean_eval_log_every),
    ]
    if args.clean_eval_cpu_fallback:
        command.append("--cpu_fallback")

    print(
        f"[{edge_type}] CLEAN TEST EVALUATION AFTER EPOCH {epoch_num} "
        f"(scope={args.clean_eval_scope}, gpu={args.clean_eval_gpu})",
        flush=True,
    )
    evaluator_env = os.environ.copy()
    # The training process has already loaded libgomp through DGL/PyTorch.  A
    # child evaluator can otherwise inherit an incompatible Intel MKL threading
    # choice even though a fresh evaluator process starts normally.
    evaluator_env["MKL_THREADING_LAYER"] = "GNU"
    subprocess.run(command, check=True, env=evaluator_env)

    scope_paths = (
        {
            "overall": os.path.join(eval_dir, "overall.json"),
            "long_range": os.path.join(eval_dir, "long_range.json"),
        }
        if args.clean_eval_scope == "both"
        else {args.clean_eval_scope: output_path}
    )
    summaries = []
    for scope, result_path in scope_paths.items():
        with open(result_path, "r") as handle:
            result = json.load(handle)
        summary = {
            "epoch": epoch_num,
            "task": edge_type,
            "scope": scope,
            "checkpoint": os.path.abspath(checkpoint_path),
            "result": os.path.abspath(result_path),
            "eligible_records": result["eligible_records"],
            "metrics": result["metrics"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
        summaries.append(summary)
        print(
            f"[{edge_type}] CLEAN EVAL EPOCH {epoch_num} {scope}: "
            f"F1={result['metrics']['f1']:.6f}, "
            f"precision={result['metrics']['precision']:.6f}, "
            f"recall={result['metrics']['recall']:.6f}, "
            f"support={result['metrics']['support']}",
            flush=True,
        )

    history_path = os.path.join(task_output_dir, "clean_eval_history.jsonl")
    with open(history_path, "a") as handle:
        for summary in summaries:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
    return summaries

def filter_indices_by_edge_type(dataset, edge_type, args):
    """
    Filter graph indices that contain at least one positive edge of the specified type.
    Results are cached to avoid recomputation if the dataset hasn't changed.
    """
    edge_type_id = EDGE_TYPE_MAPPING[edge_type]
    graph_ids = dataset_graph_ids(dataset)
    graph_signature = graph_id_signature(graph_ids)
    total_graphs = len(graph_ids)
    
    os.makedirs(os.path.join(args.output_dir, "filter_cache"), exist_ok=True)
    cache_file = os.path.join(args.output_dir, "filter_cache", f"filtered_indices_{edge_type}.json")
    
    # Try loading from cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
                if cache.get("graph_signature") == graph_signature:
                    print(f"[CACHE] Loaded {len(cache['valid_indices'])} filtered indices for '{edge_type}' from cache.")
                    return cache["valid_indices"]
                else:
                    print(f"[CACHE] Graph id signature mismatch for '{edge_type}'. Recomputing filtered indices.")
        except Exception as e:
            print(f"[CACHE] Failed to load cache for '{edge_type}': {e}")
    
    # Recompute if cache not found or invalid
    print(f"Filtering graphs for edge type: {edge_type} (ID: {edge_type_id})")
    valid_indices = []
    for idx in tqdm(graph_ids, desc=f"Filtering graphs for {edge_type}"):
        try:
            # UPDATED: Pass target_edge_type to load_item
            data = dataset.load_item(idx, "train", edge_type)
            # UPDATED: Check the new return structure
            if data["pos_count"] > 0:  # Check if there are positive edges for this type
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
                "graph_signature": graph_signature,
                "valid_indices": valid_indices
            }, f)
        print(f"[CACHE] Saved filtered indices to {cache_file}")
    except Exception as e:
        print(f"[CACHE] Failed to save cache for '{edge_type}': {e}")
    
    return valid_indices

def infer_package_from_binary_path(path):
    """Infer a package key from paths shaped like .../binary/<package>/<binary>."""
    parts = os.path.normpath(path).split(os.sep)
    if "binary" in parts:
        binary_pos = parts.index("binary")
        if binary_pos + 1 < len(parts):
            return parts[binary_pos + 1]
    return os.path.basename(os.path.dirname(path))

def split_indices_for_task(valid_indices, train_ratio, val_ratio,seed, indextobin):
    """
    Split the filtered indices into train/val/test sets
    """
    
    package_to_indices = defaultdict(list)
    for idx_str in valid_indices:
        idx = int(idx_str)
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


def sample_indices_by_percentage(indices, data_percentage, seed, label):
    """Deterministically downsample one split while keeping at least one graph."""
    if not 0 < data_percentage <= 100:
        raise ValueError("data_percentage must be in (0, 100].")
    if not indices:
        return []

    ordered = sorted(indices, key=lambda value: int(value))
    keep_count = max(1, int(len(ordered) * data_percentage / 100.0))
    if keep_count >= len(ordered):
        return ordered

    rng = random.Random(f"{seed}:{label}:{data_percentage:.12g}")
    return sorted(rng.sample(ordered, keep_count), key=lambda value: int(value))


def resolve_data_percentage(args):
    """Resolve either a user-facing fraction or the legacy percentage option."""
    data_fraction = getattr(args, "data_fraction", None)
    if data_fraction is not None:
        if not 0 < data_fraction <= 1:
            raise ValueError("data_fraction must be in (0, 1].")
        return data_fraction * 100.0
    return args.data_percentage

def evaluate_single_task_model_with_threshold(model, dataset, indices, device, args, task_type, threshold):
    """
    Evaluate a single-task model with a specific threshold
    """
    model.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for i, idx in enumerate(tqdm(indices, desc=f"Evaluating {task_type} with threshold {threshold:.2f}")):
            try:
                data = dataset.load_item(idx, "eval", task_type)
                graph = data['graph'].to(device)
                candidate_meta = data.get('candidate_meta')
                
                if data['pos_count'] == 0:
                    continue
                
                all_edges = data['edges'].to(device)
                labels = data['labels'].to(device)
                
                edge_logits = model(graph, all_edges, candidate_meta=candidate_meta)
                scores = torch.sigmoid(edge_logits.squeeze())
                
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                torch.cuda.empty_cache()
                
            except Exception as e:
                continue
    
    if len(all_scores) == 0:
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'tp': 0, 'fp': 0, 'fn': 0, 'support': 0
        }
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Use specified threshold
    predictions = (all_scores >= threshold).astype(int)
    
    tp = np.sum((predictions == 1) & (all_labels == 1))
    fp = np.sum((predictions == 1) & (all_labels == 0))
    fn = np.sum((predictions == 0) & (all_labels == 1))
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'support': tp + fn
    }

def evaluate_single_task_model(model, dataset, indices, device, args, task_type, find_threshold=False):
    """
    GPU-only evaluation - no CPU transfers until final metrics
    """
    model.eval()
    
    # Store tensors on GPU
    all_scores_gpu = []
    all_labels_gpu = []
    
    with torch.no_grad():
        for i, idx in enumerate(tqdm(indices, desc=f"Evaluating {task_type}")):
            try:
                data = dataset.load_item(idx, "eval", task_type)
                graph = data['graph'].to(device)
                candidate_meta = data.get('candidate_meta')
                
                if data['pos_count'] == 0:
                    continue
                
                all_edges = data['edges'].to(device)  # Shape: [num_edges, 2]
                labels = data['labels'].to(device)
                
                # Skip if no edges
                if all_edges.size(0) == 0:
                    continue
                
                # Bounds checking for [num_edges, 2] format
                num_nodes = graph.num_nodes()
                
                if all_edges.dim() == 2 and all_edges.shape[1] == 2:
                    # Standard format: [num_edges, 2]
                    valid_mask = (all_edges[:, 0] >= 0) & (all_edges[:, 0] < num_nodes) & \
                                (all_edges[:, 1] >= 0) & (all_edges[:, 1] < num_nodes)
                else:
                    print(f"[ERROR] Unexpected edge tensor format: {all_edges.shape}")
                    continue
                
                if not valid_mask.all():
                    all_edges = all_edges[valid_mask]
                    labels = labels[valid_mask]
                    if all_edges.size(0) == 0:
                        continue
                
                # Model forward pass
                edge_logits = model(graph, all_edges, candidate_meta=candidate_meta)
                scores = torch.sigmoid(edge_logits.squeeze())
                
                # Shape fixes
                if scores.dim() == 0:
                    scores = scores.unsqueeze(0)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                
                # Store on GPU - NO CPU conversion
                all_scores_gpu.append(scores.detach())
                all_labels_gpu.append(labels.detach())
                
            except Exception as e:
                print(f"[ERROR] Graph {i} (Index {idx}): {e}")
                continue
            
            # Memory cleanup
            if (i + 1) % 20 == 0:
                torch.cuda.empty_cache()
    
    if len(all_scores_gpu) == 0:
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'tp': 0, 'fp': 0, 'fn': 0, 'support': 0,
            'total_examples': 0, 'optimal_threshold': 0.5
        }
    
    # Concatenate on GPU
    all_scores = torch.cat(all_scores_gpu, dim=0)
    all_labels = torch.cat(all_labels_gpu, dim=0)
    

    optimal_threshold = 0.5
    
    # Final metrics on GPU
    predictions = (all_scores >= optimal_threshold).float()
    tp = ((predictions == 1) & (all_labels == 1)).sum().item()
    fp = ((predictions == 1) & (all_labels == 0)).sum().item()
    fn = ((predictions == 0) & (all_labels == 1)).sum().item()
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'support': tp + fn,
        'total_examples': len(all_labels),
        'optimal_threshold': optimal_threshold
    }

def check_graph_features(dataset, indices, edge_type):
    """
    Check feature dimensions of graphs in the dataset
    """
    # Use the first graph to get feature dimensions
    # UPDATED: Pass target_edge_type to load_item
    data = dataset.load_item(indices[0], "train", edge_type)
    graph = data['graph']
    
    #print(f"Graph info:")
    # Check code node features
    code_features = {}
    if 'code' in graph.ntypes:
        for key in graph.nodes['code'].data.keys():
            shape = graph.nodes['code'].data[key].shape
            code_features[key] = shape
        #print(f"Code node features: {code_features}")
    
    # Check data node features
    data_features = {}
    if 'data' in graph.ntypes:
        for key in graph.nodes['data'].data.keys():
            shape = graph.nodes['data'].data[key].shape
            data_features[key] = shape
        #print(f"Data node features: {data_features}")
    
    # Check edge numbers  
    for etype in graph.canonical_etypes:
        num_edges = graph.number_of_edges(etype)
        #print(f"Edge type {etype}: {num_edges} edges")
    
    return code_features, data_features if 'data' in graph.ntypes else None

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
            counter=lambda total, removed: global_counter_for_task(edge_type, total, removed)
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
            counter=lambda total, removed: global_counter_for_task(edge_type, total, removed)
        )

def train_single_task_model(model, dataset, indices, device, optimizer, epoch, args, task_type):
    """
    Train a single-task model for a specific edge type with enhanced F1 optimization
    """
    model.train()
    total_loss = 0
    total_edge_acc = 0
    processed_graphs = 0
    total_pos_edges = 0
    total_neg_edges = 0

    def move_optimizer_state(target_device):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(target_device)
    
    # Shuffle indices for this epoch
    random.shuffle(indices)
    
    # Process each graph individually with progress bar
    for i, idx in enumerate(tqdm(indices, desc=f"Training {task_type} - Epoch {epoch+1}")):
        start_time = time.time()
        
        # Get single graph data
        try:
            data = dataset.load_item(idx, "train", task_type)
        except Exception as e:
            print(f"Error loading graph {idx}: {e}")
            continue
            
        # ``featmean`` is the model input whenever it is present. Some large
        # graphs also retain the full token matrix under ``feat``; avoid
        # transferring that unused tensor to the GPU.
        graph = data['graph']
        if 'code' in graph.ntypes:
            code_data = graph.nodes['code'].data
            if 'featmean' in code_data and 'feat' in code_data:
                del code_data['feat']
        skip_code_nodes = getattr(args, 'skip_code_nodes', 0)
        code_node_count = graph.num_nodes('code') if 'code' in graph.ntypes else 0
        if skip_code_nodes > 0 and code_node_count >= skip_code_nodes:
            print(
                f"[OOM-RISK SKIP] Graph {idx}: {code_node_count} code nodes "
                f">= threshold {skip_code_nodes}."
            )
            try:
                skip_log = os.path.join(args.output_dir, "skipped_oom_risk_graphs.txt")
                with open(skip_log, "a") as f:
                    f.write(
                        f"epoch={epoch + 1} task={task_type} graph={idx} "
                        f"code_nodes={code_node_count}\n"
                    )
            except Exception as log_err:
                print(f"[LOG ERROR] Failed to log OOM-risk graph {idx}: {log_err}")
            del graph, data
            continue
        cpu_fallback_threshold = getattr(args, 'cpu_fallback_code_nodes', 0)
        use_cpu_fallback = (
            device.type == 'cuda'
            and cpu_fallback_threshold > 0
            and graph.num_nodes('code') >= cpu_fallback_threshold
        )
        step_device = torch.device('cpu') if use_cpu_fallback else device
        if use_cpu_fallback:
            print(
                f"[CPU FALLBACK] Graph {idx}: {graph.num_nodes('code')} code nodes "
                f">= threshold {cpu_fallback_threshold}."
            )
            model.to(step_device)
            move_optimizer_state(step_device)
            torch.cuda.empty_cache()
        try:
            graph = graph.to(step_device)
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] Graph {i} (Index {idx}) could not be transferred to GPU, skipping.")
            try:
                oom_log = os.path.join(args.output_dir, "oom_graphs.txt")
                with open(oom_log, "a") as f:
                    f.write(f"{task_type} {idx}  # Graph {i}, GPU transfer OOM\n")
            except Exception as log_err:
                print(f"[LOG ERROR] Failed to log OOM graph {idx}: {log_err}")
            if use_cpu_fallback:
                model.to(device)
                move_optimizer_state(device)
            torch.cuda.empty_cache()
            continue
        candidate_meta = data.get('candidate_meta')
        
        # Check if we have any edges for this task type
        if data['pos_count'] == 0:
            print(f"[WARNING] No {task_type} edges in graph {idx} (this shouldn't happen after filtering)")
            continue
        
        # Get edges and labels - now consistently in [num_edges, 2] format
        all_edges = data['edges'].to(step_device)  # Shape: [num_edges, 2]
        all_labels = data['labels'].to(step_device)
        
        # Skip if no edges
        if all_edges.size(0) == 0:
            continue
        
        # Bounds checking for [num_edges, 2] format
        num_nodes = graph.num_nodes()
        
        # Validate edge indices
        if all_edges.dim() == 2 and all_edges.shape[1] == 2:
            # Standard format: [num_edges, 2] where each row is [source, target]
            valid_mask = (all_edges[:, 0] >= 0) & (all_edges[:, 0] < num_nodes) & \
                        (all_edges[:, 1] >= 0) & (all_edges[:, 1] < num_nodes)
        else:
            print(f"[ERROR] Unexpected edge tensor format: {all_edges.shape}")
            continue
        
        # Apply valid mask if needed
        if not valid_mask.all():
            print(f"[WARNING] Found {(~valid_mask).sum().item()} invalid edges in graph {idx}")
            all_edges = all_edges[valid_mask]
            all_labels = all_labels[valid_mask]
            
            if all_edges.size(0) == 0:
                print(f"[WARNING] All edges out of bounds in graph {idx}, skipping.")
                continue
        
        total_pos_edges += data['pos_count']
        total_neg_edges += data['neg_count']
        
        # Calculate class weights for this batch
        pos_weight, neg_weight = calculate_class_weights(data['pos_count'], data['neg_count'])
        
        # Very large graphs can have millions of labelled pairs. Encode the
        # graph once, then score pair chunks while accumulating the exact
        # mean-loss gradient.
        optimizer.zero_grad()
        edge_batch_size = getattr(args, 'edge_batch_size', 0)
        chunk_edges = edge_batch_size > 0 and all_edges.size(0) > edge_batch_size
        node_embeddings = None
        code_embedding_leaf = None
        scoring_embeddings = None
        model_output = None
        edge_logits = None
        loss = None
        chunk_logits = None
        chunk_loss = None
        chunk_loss_sum = None
        try:
            if chunk_edges:
                node_embeddings = model(graph, edge_pairs=None, candidate_meta=candidate_meta)
                # Detach the final code embeddings while scoring chunks. This
                # lets each scorer graph be freed immediately instead of
                # retaining millions of pair-level MLP activations. Accumulate
                # the exact gradient on the leaf, then propagate it through the
                # encoder once after all chunks.
                code_embedding_leaf = node_embeddings['code'].detach().requires_grad_(True)
                scoring_embeddings = {'code': code_embedding_leaf}
                loss_value = 0.0
                correct_edges = 0
                total_edges = all_edges.size(0)
                for chunk_start in range(0, total_edges, edge_batch_size):
                    chunk_end = min(chunk_start + edge_batch_size, total_edges)
                    chunk_logits = model.score_edges(
                        scoring_embeddings, all_edges[chunk_start:chunk_end]
                    ).squeeze()
                    chunk_labels = all_labels[chunk_start:chunk_end].float()
                    if args.use_focal_loss:
                        ce_loss = nn.BCEWithLogitsLoss(reduction='none')(
                            chunk_logits, chunk_labels
                        )
                        chunk_loss_sum = (
                            args.focal_alpha
                            * (1 - torch.exp(-ce_loss)) ** args.focal_gamma
                            * ce_loss
                        ).sum()
                    else:
                        pos_weight_tensor = torch.tensor(pos_weight, device=step_device)
                        chunk_loss_sum = nn.BCEWithLogitsLoss(
                            pos_weight=pos_weight_tensor,
                            reduction='sum',
                        )(chunk_logits, chunk_labels)
                    chunk_loss = chunk_loss_sum / total_edges
                    chunk_loss.backward()
                    loss_value += chunk_loss.item()
                    with torch.no_grad():
                        chunk_preds = (torch.sigmoid(chunk_logits) >= 0.5).float()
                        correct_edges += int((chunk_preds == chunk_labels).sum().item())
                node_embeddings['code'].backward(code_embedding_leaf.grad)
                edge_acc_value = correct_edges / total_edges
            else:
                model_output = model(graph, all_edges, candidate_meta=candidate_meta)
                if isinstance(model_output, tuple):
                    *_, edge_logits = model_output
                else:
                    edge_logits = model_output
                if args.use_focal_loss:
                    alpha = args.focal_alpha
                    gamma = args.focal_gamma
                    ce_loss = nn.BCEWithLogitsLoss(reduction='none')(
                        edge_logits.squeeze(), all_labels.float()
                    )
                    pt = torch.exp(-ce_loss)
                    loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()
                else:
                    pos_weight_tensor = torch.tensor(pos_weight, device=step_device)
                    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)(
                        edge_logits.squeeze(), all_labels.float()
                    )
                loss.backward()
                loss_value = loss.item()
                with torch.no_grad():
                    edge_preds = (torch.sigmoid(edge_logits.squeeze()) >= 0.5).float()
                    edge_acc_value = (
                        edge_preds == all_labels.float()
                    ).float().mean().item()
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] Graph {i} (Index {idx}) caused OOM, skipping.")
            optimizer.zero_grad()
            try:
                oom_log = os.path.join(args.output_dir, "oom_graphs.txt")
                with open(oom_log, "a") as f:
                    f.write(f"{task_type} {idx}  # Graph {i}, {graph.num_nodes()} nodes, {graph.num_edges()} edges\n")
            except Exception as log_err:
                print(f"[LOG ERROR] Failed to log OOM graph {idx}: {log_err}")
            if node_embeddings is not None:
                del node_embeddings
            if code_embedding_leaf is not None:
                del code_embedding_leaf, scoring_embeddings
            if edge_logits is not None:
                del edge_logits, model_output, loss
            if chunk_logits is not None:
                del chunk_logits, chunk_loss, chunk_loss_sum
            del graph, all_edges, all_labels
            if use_cpu_fallback:
                model.to(device)
                move_optimizer_state(device)
            torch.cuda.empty_cache()
            continue
        except Exception as model_error:
            print(f"[MODEL ERROR] Graph {i} (Index {idx}): {model_error}")
            print(f"[DEBUG] all_edges shape: {all_edges.shape}")
            print(f"[DEBUG] graph nodes: {graph.num_nodes()}, edges: {graph.num_edges()}")
            optimizer.zero_grad()
            if node_embeddings is not None:
                del node_embeddings
            if code_embedding_leaf is not None:
                del code_embedding_leaf, scoring_embeddings
            if edge_logits is not None:
                del edge_logits, model_output, loss
            if chunk_logits is not None:
                del chunk_logits, chunk_loss, chunk_loss_sum
            del graph, all_edges, all_labels
            if use_cpu_fallback:
                model.to(device)
                move_optimizer_state(device)
                torch.cuda.empty_cache()
            continue
        
        # Gradient clipping to prevent exploding gradients
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()

        if use_cpu_fallback:
            try:
                fallback_log = os.path.join(args.output_dir, "cpu_fallback_graphs.txt")
                with open(fallback_log, "a") as f:
                    f.write(
                        f"epoch={epoch + 1} task={task_type} graph={idx} "
                        f"code_nodes={data['graph'].num_nodes('code')}\n"
                    )
            except Exception as log_err:
                print(f"[LOG ERROR] Failed to log CPU fallback graph {idx}: {log_err}")
            model.to(device)
            move_optimizer_state(device)
        
        try:
            torch.cuda.empty_cache()
        except Exception as cleanup_error:
            print(f"[WARNING] Failed to clean up GPU memory: {cleanup_error}")
        
        # Accumulate metrics
        total_loss += loss_value
        total_edge_acc += edge_acc_value
        processed_graphs += 1
        if chunk_edges:
            del node_embeddings, code_embedding_leaf, scoring_embeddings
            del chunk_logits, chunk_loss, chunk_loss_sum
        else:
            del edge_logits, model_output, loss
        del graph, all_edges, all_labels
    
    # Calculate average metrics
    if processed_graphs > 0:
        avg_loss = total_loss / processed_graphs
        avg_edge_acc = total_edge_acc / processed_graphs
        print(f"[{task_type}] Training summary - Pos edges: {total_pos_edges}, Neg edges: {total_neg_edges}, Ratio: {total_neg_edges/max(total_pos_edges,1):.2f}")
    else:
        avg_loss = float('inf')
        avg_edge_acc = 0
    
    return {
        'loss': avg_loss,
        'edge_acc': avg_edge_acc,
        'processed_graphs': processed_graphs
    }

# Update your main() function to create task-specific datasets
def main(args):
    set_seed(args.seed)
    effective_data_percentage = resolve_data_percentage(args)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)
    print(f"Using device: {device}")
    print(f"train_only_skip_validation: {args.skip_validation}")
    
    # Define edge types from common mapping
    all_edge_types = list(EDGE_TYPE_MAPPING.keys())
    
    # Filter edge types based on user input
    if args.target_edge_types:
        # Validate that specified edge types exist
        invalid_types = [et for et in args.target_edge_types if et not in all_edge_types]
        if invalid_types:
            print(f"ERROR: Invalid edge types specified: {invalid_types}")
            print(f"Available edge types: {all_edge_types}")
            return
        
        args.edge_types = args.target_edge_types
        print(f"Training only on specified edge types: {args.edge_types}")
    else:
        args.edge_types = all_edge_types
        print(f"Training on all edge types: {args.edge_types}")
    
    args.edge_types_dict = EDGE_TYPE_MAPPING
    args.num_edge_types = len(args.edge_types)
    args.et_fnhash = EDGE_TYPE_TO_FNHASH
    
    basedir = os.path.dirname(args.graph_dir)
    print(f"basedir: {basedir}")
    indextores_path = os.path.join(basedir, "indextores.json")
    with open(indextores_path, 'r') as f:
        indextores = json.load(f)

    cached_split = None
    if args.split_cache_path:
        with open(args.split_cache_path, 'r') as f:
            cached_split = json.load(f)
        print(f"[INFO] Reusing package-level split from {args.split_cache_path}")
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    # Store results for each edge type
    all_task_results = {}
    
    # Process each edge type separately
    for edge_type in tqdm(args.edge_types, desc="Processing edge types"):
        print(f"\n{'='*80}")
        print(f"PROCESSING EDGE TYPE: {edge_type}")
        print(f"{'='*80}")
        
        # UPDATED: Create task-specific dataset for this edge type using new constructor
        print(f"Creating task-specific dataset for {edge_type}...")
        dataset = JumpGraphDataset(
            graph_dir=args.graph_dir,
            neg_multiplier=args.neg_multiplier if args.oversample_negatives else 1,
            use_data_nodes=args.use_data_nodes,
            use_func_rel=args.use_func_rel,
            use_reverse_edges=args.use_reverse_edges,
            oversample_negatives=args.oversample_negatives
        )
        
        if cached_split is not None:
            split_keys = {
                split_name: f"{edge_type}_{split_name}_graph"
                for split_name in ("train", "val", "test")
            }
            missing_keys = [key for key in split_keys.values() if key not in cached_split]
            if missing_keys:
                raise KeyError(
                    f"Split cache is missing keys for {edge_type}: {missing_keys}"
                )
            train_indices = [str(value) for value in cached_split[split_keys["train"]]]
            val_indices = [str(value) for value in cached_split[split_keys["val"]]]
            test_indices = [str(value) for value in cached_split[split_keys["test"]]]
            valid_indices = sorted(
                set(train_indices) | set(val_indices) | set(test_indices),
                key=lambda value: int(value),
            )
        else:
            # Filter graphs that contain this task, then create a package split.
            valid_indices = filter_indices_by_edge_type(dataset, edge_type, args)
            if len(valid_indices) == 0:
                print(f"No graphs found for edge type {edge_type}, skipping...")
                all_task_results[edge_type] = {
                    'status': 'no_data',
                    'message': 'No graphs contain this edge type'
                }
                continue
            basedir = os.path.dirname(args.graph_dir)
            bintoindex_path = os.path.join(basedir, "bintoindex.json")
            with open(bintoindex_path, 'r') as f:
                bintoindex = json.load(f)
            index_to_path = {v: k for k, v in bintoindex.items()}
            if args.train_all_task_graphs:
                train_indices = list(valid_indices)
                val_indices = []
                test_indices = []
            else:
                train_indices, val_indices, test_indices = split_indices_for_task(
                    valid_indices, args.train_ratio, args.val_ratio, args.seed, index_to_path
                )

        if args.train_all_task_graphs:
            train_indices = list(valid_indices)
            val_indices = []
            test_indices = []
            print(
                f"[INFO] Using all {len(train_indices)} eligible {edge_type} graphs "
                "for training; internal validation/test splits are disabled."
            )
        train_indices = sample_indices_by_percentage(
            train_indices, effective_data_percentage, args.seed, f"{edge_type}:train"
        )
        val_indices = sample_indices_by_percentage(
            val_indices, effective_data_percentage, args.seed, f"{edge_type}:val"
        )
        test_indices = sample_indices_by_percentage(
            test_indices, effective_data_percentage, args.seed, f"{edge_type}:test"
        )
        
        print(f"Dataset split for {edge_type}:")
        print(f"  Training: {len(train_indices)} graphs")
        print(f"  Validation: {len(val_indices)} graphs")
        print(f"  Testing: {len(test_indices)} graphs")
        
        # Step 3: Process duplicate removal when function-hash sidecars exist.
        if args.skip_duplicate_filtering:
            print(f"[INFO] Skipping duplicate filtering for {edge_type} by request.")
        else:
            selected_indices = train_indices + val_indices + test_indices
            sidecar_count = count_function_hash_sidecars(selected_indices, indextores)
            if sidecar_count == 0:
                print(
                    f"[INFO] No function-hash sidecars found for {edge_type}; "
                    "duplicate filtering is skipped automatically."
                )
            else:
                print(f"[INFO] Found {sidecar_count} function-hash sidecars for {edge_type}.")
                process_duplicate_removal_for_split(
                    train_indices, val_indices, test_indices, indextores, edge_type
                )
        
        # Step 4: Check graph features (use first training graph)
        if len(train_indices) > 0:
            print(f"Checking graph features for {edge_type}...")
            # UPDATED: Pass edge_type to check_graph_features
            code_features, data_features = check_graph_features(dataset, train_indices, edge_type)
            
            # Determine feature dimensions
            code_feat_dim = code_features.get('featmean', code_features.get('feat'))[1]
            data_feat_dim = data_features.get('feat')[1] if data_features else 1
            print(f"Code feature dimension: {code_feat_dim}")
            print(f"Data feature dimension: {data_feat_dim}")
        else:
            print(f"No training graphs available for {edge_type}")
            continue

        use_gch, use_gdh = effective_hub_flags(args, edge_type)
        
        # Step 5: Initialize model for this edge type
        model = SingleTaskGAT(
            code_feat_dim=code_feat_dim,
            data_feat_dim=data_feat_dim,
            task_type=edge_type,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_reverse_edges=args.use_reverse_edges,
            use_gch=use_gch,
            use_gdh=use_gdh,
            task_aware_routing=not args.disable_task_aware_routing,
            gdh_radius=args.gdh_radius,
        ).to(device)
        
        task_output_dir = os.path.join(args.output_dir, edge_type)
        os.makedirs(task_output_dir, exist_ok=True)
        model_path = os.path.join(task_output_dir, 'best_model.pt')
        
        if os.path.exists(model_path):
            load_model_artifact(model, model_path, map_location=device)
            print(f"[INFO] Loaded existing model for {edge_type} from {model_path}")
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        
        # Enhanced scheduler with F1-based monitoring
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',  # Maximize F1 score
            factor=0.5, 
            patience=args.scheduler_patience, 
            verbose=True,
            threshold=0.01
        )
        
        print(f"Initialized model for {edge_type} with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Step 6: Create tracking files for this edge type
        if args.skip_validation:
            train_history_path = os.path.join(task_output_dir, "train_history.csv")
            with open(train_history_path, 'w') as f:
                f.write("epoch,loss,edge_acc,processed_graphs\n")
        else:
            f1_path = os.path.join(task_output_dir, "f1_scores.txt")
            with open(f1_path, 'w') as f:
                f.write("epoch,f1,precision,recall,best_f1,threshold\n")
        
        # Step 7: Training for this edge type
        best_f1 = 0.0
        optimal_threshold = 0.5
        patience_counter = 0
        last_train_metrics = None
        latest_snapshot = None
        
        if not args.eval_only:
            print(f"\nStarting training for {edge_type}...")
            
            for epoch in tqdm(range(args.epochs), desc=f"Training {edge_type}"):
                print(f"\n[{edge_type}] Epoch {epoch+1}/{args.epochs}")
                
                # Train
                train_metrics = train_single_task_model(
                    model=model,
                    dataset=dataset,
                    indices=train_indices,
                    device=device,
                    optimizer=optimizer,
                    epoch=epoch,
                    args=args,
                    task_type=edge_type
                )
                
                # Skip if no graphs were processed
                if train_metrics['processed_graphs'] == 0:
                    print(f"No graphs processed for {edge_type}, skipping evaluation")
                    continue

                last_train_metrics = train_metrics

                if args.skip_validation:
                    latest_snapshot = save_train_only_snapshot(
                        model=model,
                        optimizer=optimizer,
                        task_output_dir=task_output_dir,
                        edge_type=edge_type,
                        epoch=epoch,
                        train_metrics=train_metrics,
                        args=args,
                    )
                    with open(train_history_path, 'a') as f:
                        f.write(f"{epoch+1},{train_metrics['loss']},{train_metrics['edge_acc']},{train_metrics['processed_graphs']}\n")
                    print(f"[{edge_type}] Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['edge_acc']:.4f}")
                    print(f"[{edge_type}] Saved train-only snapshot for epoch {epoch+1}: {latest_snapshot['epoch_model']}")
                    if args.clean_eval_root:
                        run_clean_test_evaluation(
                            checkpoint_path=os.path.join(
                                task_output_dir, latest_snapshot["epoch_model"]
                            ),
                            task_output_dir=task_output_dir,
                            edge_type=edge_type,
                            epoch=epoch,
                            args=args,
                        )
                    continue
                
                # Debug validation before evaluation
                print(f"[DEBUG] About to evaluate {len(val_indices)} validation graphs for {edge_type}")
                print(f"[DEBUG] First 5 validation indices: {val_indices[:5]}")
                
                # Evaluate with threshold optimization
                eval_metrics = evaluate_single_task_model(
                    model=model,
                    dataset=dataset,
                    indices=val_indices,
                    device=device,
                    args=args,
                    task_type=edge_type,
                    find_threshold=args.optimize_threshold
                )
                
                # Update scheduler based on F1 score
                scheduler.step(eval_metrics['f1'])
                
                # Get current F1 score and threshold
                current_f1 = eval_metrics['f1']
                current_threshold = eval_metrics['optimal_threshold']
                
                # Record F1 score
                with open(f1_path, 'a') as f:
                    f.write(f"{epoch+1},{current_f1},{eval_metrics['precision']},{eval_metrics['recall']},{best_f1},{current_threshold}\n")
                
                # Save best model with early stopping
                is_best_epoch = current_f1 > best_f1
                if is_best_epoch:
                    best_f1 = current_f1
                    optimal_threshold = current_threshold
                    patience_counter = 0
                    
                    # Save best F1 info
                    best_f1_path = os.path.join(task_output_dir, "best_f1.txt")
                    with open(best_f1_path, 'w') as f:
                        f.write(f"best_epoch,best_f1,precision,recall,threshold\n")
                        f.write(f"{epoch+1},{current_f1},{eval_metrics['precision']},{eval_metrics['recall']},{current_threshold}\n")
                    
                    # Save model
                    model_path = os.path.join(task_output_dir, 'best_model.pt')
                    save_model_artifact(model_path, model)
                    print(f"New best model for {edge_type} saved with F1: {current_f1:.4f}, Threshold: {current_threshold:.3f}")
                else:
                    patience_counter += 1

                if args.save_all_epoch_models:
                    epoch_model_path = os.path.join(
                        task_output_dir,
                        f"model_epoch_{epoch + 1:03d}.pt",
                    )
                    epoch_metrics_path = os.path.join(
                        task_output_dir,
                        f"metrics_epoch_{epoch + 1:03d}.json",
                    )
                    save_model_artifact(epoch_model_path, model)
                    with open(epoch_metrics_path, "w") as f:
                        json.dump(
                            {
                                "epoch": epoch + 1,
                                "task": edge_type,
                                "is_best_validation_epoch": is_best_epoch,
                                "best_validation_f1_so_far": best_f1,
                                "train": train_metrics,
                                "validation": eval_metrics,
                            },
                            f,
                            indent=2,
                        )
                    print(
                        f"[{edge_type}] Saved epoch {epoch + 1} model and metrics: "
                        f"{epoch_model_path}"
                    )

                if args.clean_eval_root:
                    if not args.save_all_epoch_models:
                        epoch_model_path = os.path.join(
                            task_output_dir,
                            f"model_epoch_{epoch + 1:03d}.pt",
                        )
                        save_model_artifact(epoch_model_path, model)
                    run_clean_test_evaluation(
                        checkpoint_path=epoch_model_path,
                        task_output_dir=task_output_dir,
                        edge_type=edge_type,
                        epoch=epoch,
                        args=args,
                    )
                
                # Print metrics
                print(f"[{edge_type}] Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['edge_acc']:.4f}")
                print(f"[{edge_type}] Val - F1: {eval_metrics['f1']:.4f}, Precision: {eval_metrics['precision']:.4f}, Recall: {eval_metrics['recall']:.4f}, Threshold: {current_threshold:.3f}")
                print(f"[{edge_type}] Best F1 so far: {best_f1:.4f}")
                
                # Early stopping
                if patience_counter >= args.early_stopping_patience:
                    print(f"[{edge_type}] Early stopping triggered after {args.early_stopping_patience} epochs without improvement")
                    break

        if args.skip_validation:
            task_results = {
                'status': 'train_only_completed',
                'num_graphs': {
                    'total_valid': len(valid_indices),
                    'train': len(train_indices),
                    'val_unused': len(val_indices),
                    'test_unused': len(test_indices)
                },
                'latest_epoch': args.epochs,
                'latest_snapshot': latest_snapshot,
                'last_train_metrics': last_train_metrics,
            }
            all_task_results[edge_type] = task_results
            print(f"[{edge_type}] Train-only complete; skipped validation and final test evaluation.")
            del model, optimizer, scheduler, dataset
            torch.cuda.empty_cache()
            continue
        
        # Step 8: Final evaluation on test set
        print(f"\n{'='*60}")
        print(f"FINAL TEST EVALUATION FOR {edge_type.upper()}")
        print(f"{'='*60}")
        
        best_model_path = os.path.join(task_output_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            load_model_artifact(model, best_model_path, map_location=device)
            print(f"✅ Restored best model for {edge_type} from {best_model_path}")
        else:
            print(f"⚠️ [WARNING] No best model found for {edge_type}, evaluating last model instead.")
        
        print(f"Evaluating on {len(test_indices)} test graphs...")
        print(f"[DEBUG] About to evaluate {len(test_indices)} test graphs for {edge_type}")
        print(f"[DEBUG] First 5 test indices: {test_indices[:5]}")
        
        test_metrics = evaluate_single_task_model(
            model=model,
            dataset=dataset,
            indices=test_indices,
            device=device,
            args=args,
            task_type=edge_type,
            find_threshold=False  # Use the optimal threshold found during validation
        )
        
        # Print detailed test results
        print(f"\n🎯 TEST RESULTS FOR {edge_type.upper()}:")
        print(f"   F1 Score:     {test_metrics['f1']:.4f}")
        print(f"   Precision:    {test_metrics['precision']:.4f}")
        print(f"   Recall:       {test_metrics['recall']:.4f}")
        print(f"   True Positives:  {test_metrics['tp']}")
        print(f"   False Positives: {test_metrics['fp']}")
        print(f"   False Negatives: {test_metrics['fn']}")
        print(f"   Support (TP+FN): {test_metrics['support']}")
        print(f"   Total Examples:  {test_metrics['total_examples']}")
        print(f"   Threshold Used:  {optimal_threshold:.3f}")
        print(f"{'='*60}")
        
        # Additional evaluation - validation set performance for comparison
        if len(val_indices) > 0:
            print(f"\n📊 VALIDATION SET PERFORMANCE (for comparison):")
            val_final_metrics = evaluate_single_task_model(
                model=model,
                dataset=dataset,
                indices=val_indices,
                device=device,
                args=args,
                task_type=edge_type,
                find_threshold=False
            )
            print(f"   Val F1:       {val_final_metrics['f1']:.4f}")
            print(f"   Val Precision: {val_final_metrics['precision']:.4f}")
            print(f"   Val Recall:   {val_final_metrics['recall']:.4f}")
            print(f"   Best Val F1:  {best_f1:.4f} (from training)")
        
        # Optional: Detailed evaluation with different thresholds
        if args.detailed_eval:
            print(f"\n🔍 DETAILED THRESHOLD ANALYSIS FOR {edge_type.upper()}:")
            thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7, optimal_threshold]
            threshold_results = []
            
            for thresh in tqdm(thresholds_to_test, desc=f"Threshold analysis for {edge_type}"):
                temp_metrics = evaluate_single_task_model_with_threshold(
                    model=model,
                    dataset=dataset,
                    indices=test_indices,
                    device=device,
                    args=args,
                    task_type=edge_type,
                    threshold=thresh
                )
                threshold_results.append((thresh, temp_metrics))
                print(f"   Threshold {thresh:.2f}: F1={temp_metrics['f1']:.4f}, "
                      f"P={temp_metrics['precision']:.4f}, R={temp_metrics['recall']:.4f}")
            
            # Save threshold analysis
            thresh_analysis_path = os.path.join(task_output_dir, 'threshold_analysis.txt')
            with open(thresh_analysis_path, 'w') as f:
                f.write(f"Threshold Analysis for {edge_type}\n")
                f.write("="*40 + "\n")
                f.write("Threshold,F1,Precision,Recall,TP,FP,FN\n")
                for thresh, metrics in threshold_results:
                    f.write(f"{thresh:.2f},{metrics['f1']:.4f},{metrics['precision']:.4f},"
                           f"{metrics['recall']:.4f},{metrics['tp']},{metrics['fp']},{metrics['fn']}\n")
        
        # Step 9: Save results for this edge type
        task_results = {
            'status': 'completed',
            'num_graphs': {
                'total_valid': len(valid_indices),
                'train': len(train_indices),
                'val': len(val_indices),
                'test': len(test_indices)
            },
            'test_metrics': test_metrics,
            'best_f1': best_f1,
            'optimal_threshold': optimal_threshold
        }
        
        # Record site stats for this edge type
        total_s = site_stats['total_sites'].get(edge_type, 0)
        removed_s = site_stats['removed_sites'].get(edge_type, 0)
        
        all_task_results[edge_type] = task_results
        
        # Save individual task results
        task_results_path = os.path.join(task_output_dir, 'results.txt')
        with open(task_results_path, 'w') as f:
            f.write(f"Results for {edge_type}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Dataset split:\n")
            f.write(f"  Total valid graphs: {len(valid_indices)}\n")
            f.write(f"  Training: {len(train_indices)}\n")
            f.write(f"  Validation: {len(val_indices)}\n")
            f.write(f"  Testing: {len(test_indices)}\n\n")
            
            f.write(f"Site Stats:\n")
            f.write(f"  Total candidate sites: {total_s}\n")
            f.write(f"  Removed (duplicate) sites: {removed_s}\n\n")
            
            f.write(f"Test Results:\n")
            f.write(f"  F1: {test_metrics['f1']:.4f}\n")
            f.write(f"  Precision: {test_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {test_metrics['recall']:.4f}\n")
            f.write(f"  Support: {test_metrics['support']}\n")
            f.write(f"  Optimal Threshold: {optimal_threshold:.3f}\n\n")
            
            f.write(f"Best validation F1: {best_f1:.4f}\n")
        
        print(f"Results for {edge_type} saved to {task_results_path}")
        
        # Clean up GPU memory and delete the dataset for this task
        del model, optimizer, scheduler, dataset
        torch.cuda.empty_cache()
    
    # Step 10: Generate overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    summary_path = os.path.join(args.output_dir, 'overall_summary.txt')
    if args.skip_validation:
        with open(summary_path, 'w') as f:
            f.write("Train-only Overall Summary\n")
            f.write("="*50 + "\n\n")
            header = f"{'Edge Type':<15} {'Status':<22} {'Valid Graphs':<12} {'Train Graphs':<12} {'Latest Epoch':<12} {'Latest Model'}"
            print(header)
            print("-" * 95)
            f.write(header + "\n")
            f.write("-" * 95 + "\n")

            completed_tasks = 0
            for edge_type in args.edge_types:
                result = all_task_results.get(edge_type, {'status': 'unknown'})
                if result['status'] == 'train_only_completed':
                    valid_graphs = result['num_graphs']['total_valid']
                    train_graphs = result['num_graphs']['train']
                    latest_epoch = result.get('latest_epoch', 'N/A')
                    latest_snapshot = result.get('latest_snapshot') or {}
                    latest_model = latest_snapshot.get('latest_model', 'N/A')
                    completed_tasks += 1
                    line = f"{edge_type:<15} {'Train-only completed':<22} {valid_graphs:<12} {train_graphs:<12} {latest_epoch:<12} {latest_model}"
                elif result['status'] == 'no_data':
                    line = f"{edge_type:<15} {'No Data':<22} {'0':<12} {'0':<12} {'N/A':<12} {'N/A'}"
                else:
                    line = f"{edge_type:<15} {result['status']:<22} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A'}"
                print(line)
                f.write(line + "\n")

            f.write(f"\nCompleted train-only tasks: {completed_tasks}/{len(args.edge_types)}\n")

        print(f"\nTrain-only summary saved to {summary_path}")
        print(f"Individual train-only checkpoints saved in subdirectories of {args.output_dir}")
        return

    with open(summary_path, 'w') as f:
        f.write("Overall Summary\n")
        f.write("="*50 + "\n\n")
        
        print(f"{'Edge Type':<15} {'Status':<12} {'Valid Graphs':<12} {'Test F1':<10} {'Test Precision':<15} {'Test Recall':<12} {'Threshold':<10} {'Support'}")
        print("-" * 110)
        f.write(f"{'Edge Type':<15} {'Status':<12} {'Valid Graphs':<12} {'Test F1':<10} {'Test Precision':<15} {'Test Recall':<12} {'Threshold':<10} {'Support'}\n")
        f.write("-" * 110 + "\n")
        
        total_weighted_f1 = 0
        total_support = 0
        completed_tasks = 0
        
        for edge_type in args.edge_types:
            result = all_task_results.get(edge_type, {'status': 'unknown'})
            
            if result['status'] == 'completed':
                test_metrics = result['test_metrics']
                valid_graphs = result['num_graphs']['total_valid']
                threshold = result.get('optimal_threshold', 0.5)
                
                print(f"{edge_type:<15} {'Completed':<12} {valid_graphs:<12} "
                      f"{test_metrics['f1']:.4f}{' ':3} "
                      f"{test_metrics['precision']:.4f}{' ':8} "
                      f"{test_metrics['recall']:.4f}{' ':5} "
                      f"{threshold:.3f}{' ':4} "
                      f"{test_metrics['support']}")
                
                f.write(f"{edge_type:<15} {'Completed':<12} {valid_graphs:<12} "
                       f"{test_metrics['f1']:.4f}{' ':3} "
                       f"{test_metrics['precision']:.4f}{' ':8} "
                       f"{test_metrics['recall']:.4f}{' ':5} "
                       f"{threshold:.3f}{' ':4} "
                       f"{test_metrics['support']}\n")
                
                # Accumulate for weighted average
                total_weighted_f1 += test_metrics['f1'] * test_metrics['support']
                total_support += test_metrics['support']
                completed_tasks += 1
                
            elif result['status'] == 'no_data':
                print(f"{edge_type:<15} {'No Data':<12} {'0':<12} {'N/A':<10} {'N/A':<15} {'N/A':<12} {'N/A':<10} {'0'}")
                f.write(f"{edge_type:<15} {'No Data':<12} {'0':<12} {'N/A':<10} {'N/A':<15} {'N/A':<12} {'N/A':<10} {'0'}\n")
        
        print("-" * 110)
        f.write("-" * 110 + "\n")
        
        if total_support > 0:
            overall_f1 = total_weighted_f1 / total_support
            print(f"{'Overall':<15} {'':<12} {'':<12} {overall_f1:.4f}")
            f.write(f"{'Overall':<15} {'':<12} {'':<12} {overall_f1:.4f}\n")
        else:
            print(f"{'Overall':<15} {'':<12} {'':<12} {'N/A':<10}")
            f.write(f"{'Overall':<15} {'':<12} {'':<12} {'N/A':<10}\n")
        
        f.write(f"\nCompleted tasks: {completed_tasks}/{len(args.edge_types)}\n")
        f.write(f"Total support (test examples): {total_support}\n")
        
        # Write F1 improvement recommendations
        f.write(f"\n" + "="*50 + "\n")
        f.write("F1 IMPROVEMENT RECOMMENDATIONS\n")
        f.write("="*50 + "\n\n")
        f.write("Based on the results, consider these improvements:\n\n")
        
        for edge_type in args.edge_types:
            result = all_task_results.get(edge_type, {'status': 'unknown'})
            if result['status'] == 'completed':
                test_metrics = result['test_metrics']
                precision = test_metrics['precision']
                recall = test_metrics['recall']
                
                f.write(f"{edge_type}:\n")
                if precision < 0.5 and recall > 0.8:
                    f.write("  - High recall, low precision: Increase negative sampling ratio or use focal loss\n")
                    f.write("  - Consider adding more regularization or reducing model complexity\n")
                elif precision > 0.8 and recall < 0.5:
                    f.write("  - High precision, low recall: Reduce classification threshold\n")
                    f.write("  - Consider data augmentation or oversampling positive examples\n")
                elif precision < 0.5 and recall < 0.5:
                    f.write("  - Both low: Check data quality, increase model capacity, or adjust loss function\n")
                    f.write("  - Consider ensemble methods or different architectures\n")
                else:
                    f.write("  - Performance looks balanced\n")
                f.write("\n")
    
    print(f"\nOverall summary saved to {summary_path}")
    print(f"Individual task results saved in subdirectories of {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-task training for edge prediction with F1 optimization")
    
    # Data options
    parser.add_argument('--graph_dir', type=str, required=True,
                      help='Directory containing graph files')
    parser.add_argument('--output_dir', type=str, default='./output_single_task',
                      help='Directory to save output files')
    parser.add_argument('--split_cache_path', type=str, default=None,
                      help='Optional package-level split_by_type.json to reuse')
    parser.add_argument('--skip_duplicate_filtering', action='store_true', default=False,
                      help='Skip function-hash duplicate filtering even when sidecars are available')
    parser.add_argument('--use_data_nodes', action='store_true', default=True,
                      help='Use data nodes in the graph')
    parser.add_argument('--use_func_rel', action='store_true', default=False,
                      help='Use code2funchead relations (disabled by default)')
    parser.add_argument('--use_reverse_edges', action='store_true', default=False,
                      help='Use reverse edges in the graph to improve message passing')
    parser.add_argument('--use_gch', action=argparse.BooleanOptionalAction, default=None,
                      help='Use Global Code Hub routing (paper default: disabled for single-task jumptable, enabled for other single tasks; either form overrides it)')
    parser.add_argument('--use_gdh', action=argparse.BooleanOptionalAction, default=None,
                      help='Use Global Data Hub routing (paper default: disabled for single-task jumptable, enabled for other single tasks; either form overrides it)')
    parser.add_argument('--gdh_radius', type=int, choices=(2,), default=2,
                      help='xRef-only GDH neighborhood radius (fixed to 2)')
    parser.add_argument('--disable_task_aware_routing', action='store_true', default=False,
                      help='Disable task-aware routing and include jumptable destination candidates')
    
    # Model options
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Hidden dimension for the model')
    parser.add_argument('--num_heads', type=int, default=8,
                      help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                      help='Number of GAT layers')
    parser.add_argument('--edge_batch_size', type=int, default=0,
                      help='Score this many labelled pairs at a time after one graph encoding; 0 disables chunking')
    parser.add_argument('--cpu_fallback_code_nodes', type=int, default=0,
                      help='Train graphs at or above this code-node count on CPU; 0 disables fallback')
    parser.add_argument('--skip_code_nodes', type=int, default=0,
                      help='Skip and log graphs at or above this code-node count before GPU transfer; 0 disables')
    parser.add_argument('--dropout', type=float, default=0.4,
                      help='Dropout rate (increased default for better regularization)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay for regularization (increased)')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate (reduced for more stable training)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                      help='Gradient clipping threshold (0 to disable)')
    parser.add_argument('--oversample_negatives', action='store_true', default=False,
                      help='Use more negative samples than positive ones')
    parser.add_argument('--neg_multiplier', type=int, default=2,
                      help='Multiplier for negative samples when oversampling')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs (increased for better convergence)')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                      help='Ratio of data to use for training')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                      help='Ratio of data to use for validation')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                      help='Ratio of data to use for testing')
    parser.add_argument(
        '--data_percentage',
        type=float,
        default=100.0,
        help='Percentage of each task split to use, in (0, 100] (default: 100)',
    )
    parser.add_argument(
        '--data_fraction',
        type=float,
        default=None,
        help='Fraction of each task split to use, in (0, 1]; overrides --data_percentage',
    )

    
    # F1 optimization options
    parser.add_argument('--optimize_threshold', action='store_true', default=True,
                      help='Find optimal classification threshold to maximize F1')
    parser.add_argument('--use_focal_loss', action='store_true', default=False,
                      help='Use focal loss instead of weighted BCE loss')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                      help='Alpha parameter for focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                      help='Gamma parameter for focal loss')

    
    # Early stopping and scheduling
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                      help='Epochs to wait before early stopping')
    parser.add_argument('--scheduler_patience', type=int, default=3,
                      help='Epochs to wait before reducing learning rate')
    
    # Edge type selection - NEW ARGUMENT
    parser.add_argument('--target_edge_types', type=str, nargs='+', 
                      help='Specific edge types to train (e.g., --target_edge_types jump ret). If not specified, trains all edge types.')
    
    # Evaluation options
    parser.add_argument('--detailed_eval', action='store_true', default=False,
                      help='Perform detailed evaluation with multiple thresholds')
    parser.add_argument('--skip_validation', action='store_true', default=False,
                      help='Skip validation and final test evaluation; save train-only snapshots each epoch')
    parser.add_argument('--save_epoch_models_only', action='store_true', default=False,
                      help='With --skip_validation, save only model_epoch_NNN.pt and metadata; omit duplicate latest/best models and optimizer checkpoint')
    parser.add_argument('--save_all_epoch_models', action='store_true', default=False,
                      help='Save a versioned model artifact and validation metrics for every normal training epoch')
    parser.add_argument('--train_all_task_graphs', action='store_true', default=False,
                      help='Train on the union of the cached task train/val/test graphs; intended for evaluation against a separate external test')
    parser.add_argument('--clean_eval_root', type=str, default=None,
                      help='External clean-test root to evaluate after every training epoch')
    parser.add_argument('--clean_eval_scope', choices=('overall', 'long_range', 'both'), default='overall',
                      help='External clean-test scope evaluated after every epoch')
    parser.add_argument('--clean_eval_gpu', type=int, default=1,
                      help='GPU used by the per-epoch external clean-test evaluator (-1 for CPU)')
    parser.add_argument('--clean_eval_log_every', type=int, default=25,
                      help='Clean-test evaluator progress interval in records')
    parser.add_argument('--clean_eval_cpu_fallback', action=argparse.BooleanOptionalAction, default=True,
                      help='Retry a clean-test record on CPU if CUDA evaluation fails')
    
    # Other options
    parser.add_argument('--eval_only', action='store_true',
                      help='Only evaluate the model, no training')
    parser.add_argument('--gpu', type=int, default=0,
                      help='GPU device ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--print_every', type=int, default=100,
                      help='Print training progress every N graphs')
    
    args = parser.parse_args()
    main(args)
