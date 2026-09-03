"""
Standalone script to generate split_by_type.json for static graph data.
Used to create train/val/test splits WITHOUT running full training.

Output: --split_cache_path when provided; otherwise
        <output_dir_parent>/split_by_type.json for backward compatibility.

Usage:
    python generate_split.py \
        --graph_dir <WORK_ROOT>/total/graph \
        --output_dir outputs/multi/run_100pct \
        --target_edge_types ret jumptable indirectcall tailcall \
        --train_ratio 0.8 --val_ratio 0.1 --seed 42
"""
import argparse
import json
import os
import sys

# Allow importing from the same src/ directory
sys.path.insert(0, os.path.dirname(__file__))

from loadgraph import JumpGraphDataset
from train_multi import (
    load_binary_to_package_or_empty,
    load_or_create_split_by_package,
)


def main(args):
    basedir = os.path.dirname(args.graph_dir)
    outputbase = os.path.dirname(args.output_dir)

    # Load required index files (same logic as train_multi.py)
    bintoindex_path = os.path.join(basedir, "bintoindex.json")

    binary_to_package = load_binary_to_package_or_empty(basedir)

    print(f"[INFO] Loading bintoindex from {bintoindex_path} ...")
    with open(bintoindex_path, "r") as f:
        bintoindex = json.load(f)

    indextopath = {str(v): str(k) for k, v in bintoindex.items()}

    # Build dataset (needed to scan which graphs have which edge types)
    print(f"[INFO] Loading dataset from {args.graph_dir} ...")
    dataset = JumpGraphDataset(
        graph_dir=args.graph_dir,
        neg_multiplier=1,
        use_data_nodes=True,
        use_func_rel=False,
        use_reverse_edges=True,
        oversample_negatives=False,
    )

    edge_type_order = args.target_edge_types
    split_cache_path = args.split_cache_path or os.path.join(
        outputbase, "split_by_type.json"
    )

    print(f"[INFO] Generating split -> {split_cache_path}")
    print(f"[INFO] Edge types: {edge_type_order}")
    print(f"[INFO] Ratios: train={args.train_ratio}, val={args.val_ratio}, test={1-args.train_ratio-args.val_ratio:.2f}")

    split_parent = os.path.dirname(os.path.abspath(split_cache_path))
    os.makedirs(split_parent, exist_ok=True)

    train_idx, val_idx, test_idx, _ = load_or_create_split_by_package(
        edge_type_order,
        dataset,
        args,
        split_cache_path,
        indextopath,
        binary_to_package,
    )

    print(f"\n[DONE] Split saved to: {split_cache_path}")
    print(f"       Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate split_by_type.json for static graph data")
    parser.add_argument("--graph_dir", type=str, required=True,
                        help="Directory containing .graph.gz files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output dir (split file will be saved to its parent directory)")
    parser.add_argument(
        "--split_cache_path",
        type=str,
        default=None,
        help="Explicit split_by_type.json path; overrides the legacy output_dir-parent path",
    )
    parser.add_argument("--target_edge_types", type=str, nargs="+",
                        default=["ret", "jumptable", "indirectcall", "tailcall"],
                        help="Edge types to include in split")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_percentage", type=float, default=100.0,
                        help="Percentage of data to use (default: 100)")
    # These are required by JumpGraphDataset / filter_indices_by_edge_type
    parser.add_argument("--use_data_nodes", action="store_true", default=True)
    parser.add_argument("--use_func_rel", action="store_true", default=False)
    parser.add_argument("--use_reverse_edges", action="store_true", default=True)
    parser.add_argument("--oversample_negatives", action="store_true", default=False)
    parser.add_argument("--neg_multiplier", type=int, default=1)

    args = parser.parse_args()
    main(args)
