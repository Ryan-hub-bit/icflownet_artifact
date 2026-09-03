"""Shared utilities for distributed clean-test evaluation."""

import gzip
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from datetime import timedelta

import dgl
import torch
import torch.distributed as dist
import torch.nn.functional as F

from common import EDGE_TYPE_MAPPING, build_paper_mtl_augmented_graph


TASKS = ("ret", "jumptable", "indirectcall", "tailcall")


def distributed_setup():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        backend = os.environ.get("ICFLOWNET_DIST_BACKEND", "nccl")
        dist.init_process_group(backend=backend, timeout=timedelta(hours=12))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def cleanup_distributed(world_size):
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce(tensor, world_size):
    if world_size > 1:
        if dist.get_backend() == "gloo" and tensor.is_cuda:
            cpu_tensor = tensor.cpu()
            dist.all_reduce(cpu_tensor, op=dist.ReduceOp.SUM)
            tensor.copy_(cpu_tensor)
        else:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def seed_everything(seed, rank):
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def read_jsonl(path):
    rows = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_compressed_graph(path, use_func_rel=False):
    cache_dir = os.environ.get("ICFLOWNET_CLEAN_GRAPH_CACHE_DIR")
    cached_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path_digest = hashlib.sha256(os.path.abspath(path).encode("utf-8")).hexdigest()[:16]
        graph_name = os.path.basename(path)
        if graph_name.endswith(".gz"):
            graph_name = graph_name[:-3]
        cached_path = os.path.join(cache_dir, path_digest + "_" + graph_name)
        if os.path.exists(cached_path):
            graph = dgl.load_graphs(cached_path)[0][0]
            if not use_func_rel and "code2funchead" in graph.etypes:
                edge_ids = graph.edges(form="eid", etype="code2funchead")
                graph = dgl.remove_edges(graph, edge_ids, etype="code2funchead")
            return graph

    with tempfile.NamedTemporaryFile(
        prefix="icflownet_eval_",
        suffix=".graph",
        dir=cache_dir,
        delete=False,
    ) as temporary:
        temporary_path = temporary.name
    try:
        with gzip.open(path, "rb") as source, open(temporary_path, "wb") as destination:
            shutil.copyfileobj(source, destination)
        if cached_path:
            os.replace(temporary_path, cached_path)
            temporary_path = cached_path
        graph = dgl.load_graphs(temporary_path)[0][0]
        if not use_func_rel and "code2funchead" in graph.etypes:
            edge_ids = graph.edges(form="eid", etype="code2funchead")
            graph = dgl.remove_edges(graph, edge_ids, etype="code2funchead")
        return graph
    finally:
        if not cached_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def discard_unused_full_code_features(graph):
    """Keep pooled code features when both pooled and full tensors are stored."""
    if "code" in graph.ntypes:
        code_data = graph.nodes["code"].data
        if "featmean" in code_data and "feat" in code_data:
            del code_data["feat"]
    return graph


def prepare_clean_records(clean_root, resource_root, scope="overall"):
    if scope not in ("overall", "long_range"):
        raise ValueError("Unsupported clean-test scope: {}".format(scope))

    records = read_jsonl(os.path.join(clean_root, "binary_graph_mapping.jsonl"))
    prepared = []
    pair_totals = Counter()
    task_totals = Counter()
    zero_pair_records = []

    for record in records:
        binary_key = record["binary_key"]
        resource_dir = os.path.join(resource_root, binary_key)
        lookup_path = os.path.join(resource_dir, binary_key + "_nodelookup.json")
        nidtoaddr_path = os.path.join(resource_dir, binary_key + "_nidtoaddr.json")
        hubmeta_path = os.path.join(resource_dir, binary_key + "_hubmeta.json")
        graphstats_path = os.path.join(resource_dir, binary_key + "_graphstats.json")
        graph_path = os.path.join(clean_root, record["graph"])
        required = (lookup_path, hubmeta_path, graphstats_path, graph_path)
        missing = [path for path in required if not os.path.exists(path)]
        if missing:
            raise RuntimeError(
                "Missing clean-test resources for {}: {}".format(binary_key, missing)
            )

        graphstats = load_json(graphstats_path)
        expected_code_nodes = int(graphstats["code_nodes"])
        # New clean-test resources explicitly preserve separate code/data node-id
        # namespaces in ``nidtoaddr``.  Their legacy nodelookup can therefore
        # contain overlapping local IDs from both node types and must not be
        # interpreted as a code-only lookup.  Prefer the unambiguous code map
        # when it is available, while retaining compatibility with older data.
        if os.path.exists(nidtoaddr_path):
            nidtoaddr = load_json(nidtoaddr_path)
            code_nidtoaddr = nidtoaddr.get("code")
        else:
            code_nidtoaddr = None
        if isinstance(code_nidtoaddr, dict):
            lookup = {
                str(address): int(node_id)
                for node_id, address in code_nidtoaddr.items()
            }
        else:
            raw_lookup = load_json(lookup_path)
            lookup = {
                str(address): int(node_id)
                for address, node_id in raw_lookup.items()
            }
        if set(lookup.values()) != set(range(expected_code_nodes)):
            raise RuntimeError(
                "Node lookup does not exactly cover graph code ids for {}.".format(
                    binary_key
                )
            )

        task_rows = defaultdict(list)
        record_pair_count = 0
        for kind in (scope + "_positive", scope + "_negative"):
            rows = read_jsonl(os.path.join(clean_root, record[kind]))
            pair_totals[kind] += len(rows)
            record_pair_count += len(rows)
            for row in rows:
                src_key = str(row["src"])
                dst_key = str(row["dst"])
                if src_key not in lookup or dst_key not in lookup:
                    raise RuntimeError(
                        "Unmapped clean pair in {}: {}".format(binary_key, row)
                    )
                task = row["task"]
                task_rows[task].append(
                    {
                        "src": lookup[src_key],
                        "dst": lookup[dst_key],
                        "label": int(row["label"]),
                    }
                )
                task_totals[(task, int(row["label"]))] += 1

        if record_pair_count == 0:
            zero_pair_records.append(
                {"graph_id": str(record["graph_id"]), "binary_key": binary_key}
            )

        prepared.append(
            {
                "binary_key": binary_key,
                "graph_id": str(record["graph_id"]),
                "graph_path": graph_path,
                "candidate_meta": load_json(hubmeta_path),
                "task_rows": dict(task_rows),
                "expected_code_nodes": expected_code_nodes,
                "expected_data_nodes": int(graphstats["data_nodes"]),
            }
        )

    return prepared, {
        "records": len(prepared),
        "pair_totals": dict(pair_totals),
        "nonzero_pair_records": len(prepared) - len(zero_pair_records),
        "zero_pair_records": zero_pair_records,
        "task_label_totals": {
            "{}:{}".format(task, label): count
            for (task, label), count in sorted(task_totals.items())
        },
    }


def confusion_metrics(tn, fp, fn, tp):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": tp + fn,
        "pairs": tn + fp + fn + tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def evaluate_clean_full(
    ddp_model,
    clean_records,
    device,
    rank,
    world_size,
    output_dir,
    log_every,
    amp_enabled,
):
    module = ddp_model.module
    module.eval()
    local_records = clean_records[rank::world_size]
    confusions = torch.zeros((len(TASKS), 2, 2), dtype=torch.long, device=device)
    loss_sums = torch.zeros(len(TASKS), dtype=torch.float64, device=device)
    pair_counts = torch.zeros(len(TASKS), dtype=torch.float64, device=device)
    type_confusion = torch.zeros((4, 4), dtype=torch.long, device=device)
    evaluated = torch.zeros((), dtype=torch.long, device=device)
    local_details = []
    start_time = time.time()

    for item_index, item in enumerate(local_records, start=1):
        combined_pairs = []
        task_slices = {}
        actual_types = []
        actual_labels = []
        for task in TASKS:
            rows = item["task_rows"].get(task, [])
            if not rows:
                continue
            start = len(combined_pairs)
            combined_pairs.extend((row["src"], row["dst"]) for row in rows)
            actual_labels.extend(row["label"] for row in rows)
            actual_types.extend([EDGE_TYPE_MAPPING[task]] * len(rows))
            task_slices[task] = (start, len(combined_pairs), rows)

        if not combined_pairs:
            local_details.append(
                {
                    "binary": item["binary_key"],
                    "graph_id": item["graph_id"],
                    "pairs": 0,
                    "tasks": {},
                    "metric_contribution": "none_no_labeled_pairs",
                }
            )
            evaluated += 1
            _log_progress(rank, item_index, len(local_records), log_every, start_time)
            continue

        base_graph = discard_unused_full_code_features(
            load_compressed_graph(item["graph_path"])
        )
        if (
            base_graph.num_nodes("code") != item["expected_code_nodes"]
            or base_graph.num_nodes("data") != item["expected_data_nodes"]
        ):
            raise RuntimeError(
                "Clean graph node count mismatch for {}.".format(item["binary_key"])
            )
        model_graph = build_paper_mtl_augmented_graph(
            base_graph,
            item["candidate_meta"],
            use_gch=True,
            use_gdh=True,
            task_aware_routing=True,
            gdh_radius=2,
        )
        graph_gpu = model_graph.to(device)
        pairs = torch.tensor(combined_pairs, dtype=torch.long, device=device)
        labels = torch.tensor(actual_labels, dtype=torch.float32, device=device)
        types = torch.tensor(actual_types, dtype=torch.long, device=device)

        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            task_logits, edge_type_logits = module(
                graph_gpu, pairs, candidate_meta=None
            )

        detail = {
            "binary": item["binary_key"],
            "graph_id": item["graph_id"],
            "pairs": len(combined_pairs),
            "tasks": {},
        }
        for task_index, task in enumerate(TASKS):
            if task not in task_slices:
                continue
            start, end, rows = task_slices[task]
            logits = task_logits[task].reshape(-1)[start:end]
            task_labels = labels[start:end]
            scores = torch.sigmoid(logits)
            predictions = scores >= 0.5
            positives = task_labels == 1
            negatives = ~positives
            tn = int(((~predictions) & negatives).sum().item())
            fp = int((predictions & negatives).sum().item())
            fn = int(((~predictions) & positives).sum().item())
            tp = int((predictions & positives).sum().item())
            confusions[task_index, 0, 0] += tn
            confusions[task_index, 0, 1] += fp
            confusions[task_index, 1, 0] += fn
            confusions[task_index, 1, 1] += tp
            loss_sums[task_index] += F.binary_cross_entropy_with_logits(
                logits, task_labels, reduction="sum"
            ).double()
            pair_counts[task_index] += len(rows)
            task_detail = {
                "pairs": len(rows),
                "confusion_matrix": [[tn, fp], [fn, tp]],
            }
            detail["tasks"][task] = task_detail

        positive_mask = labels == 1
        predicted_types = edge_type_logits[positive_mask].argmax(dim=1)
        true_types = types[positive_mask]
        for true_type, predicted_type in zip(
            true_types.tolist(), predicted_types.tolist()
        ):
            type_confusion[true_type, predicted_type] += 1

        local_details.append(detail)
        evaluated += 1
        del graph_gpu, base_graph, model_graph, pairs, labels, types
        del task_logits, edge_type_logits
        torch.cuda.empty_cache()
        _log_progress(rank, item_index, len(local_records), log_every, start_time)

    detail_dir = os.path.join(output_dir, "clean_details")
    os.makedirs(detail_dir, exist_ok=True)
    detail_path = os.path.join(detail_dir, "rank_{}.jsonl".format(rank))
    with open(detail_path, "w") as handle:
        for detail in local_details:
            handle.write(json.dumps(detail) + "\n")

    for tensor in (confusions, loss_sums, pair_counts, type_confusion, evaluated):
        all_reduce(tensor, world_size)

    confusions = confusions.cpu()
    loss_sums = loss_sums.cpu()
    pair_counts = pair_counts.cpu()
    task_metrics = {}
    for task_index, task in enumerate(TASKS):
        tn = int(confusions[task_index, 0, 0].item())
        fp = int(confusions[task_index, 0, 1].item())
        fn = int(confusions[task_index, 1, 0].item())
        tp = int(confusions[task_index, 1, 1].item())
        task_metrics[task] = confusion_metrics(tn, fp, fn, tp)
        task_metrics[task]["loss"] = float(
            loss_sums[task_index].item() / max(pair_counts[task_index].item(), 1)
        )

    type_confusion = type_confusion.cpu()
    type_total = int(type_confusion.sum().item())
    type_correct = int(type_confusion.diag().sum().item())
    return {
        "evaluated_binaries": int(evaluated.item()),
        "task_metrics": task_metrics,
        "macro_f1": sum(value["f1"] for value in task_metrics.values()) / len(TASKS),
        "edge_type": {
            "class_order": ["ret", "jumptable", "tailcall", "indirectcall"],
            "confusion_matrix": type_confusion.tolist(),
            "accuracy": type_correct / type_total if type_total else 0.0,
            "positive_pairs": type_total,
        },
        "detail_rank_files": [
            os.path.join(detail_dir, "rank_{}.jsonl".format(worker_rank))
            for worker_rank in range(world_size)
        ],
        "elapsed_seconds": time.time() - start_time,
    }


def _log_progress(rank, current, total, log_every, start_time):
    if current == 1 or current % log_every == 0 or current == total:
        print(
            "[clean][rank {}] graphs={}/{} elapsed={:.1f}s".format(
                rank, current, total, time.time() - start_time
            ),
            flush=True,
        )
