"""Dataset utilities for loading compressed DGL graphs and per-task edge labels."""

import os
import random
import pickle
import json
import dgl
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from loadgt import loadgt
from common import EDGE_TYPE_MAPPING, normalize_task_name
import gzip
import shutil
import tempfile

HUBMETAEX = '_hubmeta.json'
TASK_TO_CANDIDATE_KEY = {
    'indirectcall': 'icall',
    'tailcall': 'itailcall',
    'ret': 'ret',
    'jumptable': 'jumptable',
    'icall': 'icall',
    'itailcall': 'itailcall',
}


class JumpGraphDataset(Dataset):
    """Dataset for loading graph data with single-task focus on specific edge types."""
    
    def __init__(self, graph_dir, neg_multiplier=1, sampling_strategy='mixed', 
                 use_data_nodes=True, use_reverse_edges=False, 
                 use_func_rel=False, oversample_negatives=False):
        """
        Initialize the JumpGraphDataset for single-task learning.
        
        Args:
            graph_dir: Directory containing graph files
            neg_multiplier: Multiplier for negative samples (1 = balanced, 2 = 2x negatives, etc.)
            sampling_strategy: Strategy for negative sampling ('random', 'hard', 'mixed')
            use_data_nodes: Whether to use data nodes in the graph
            use_reverse_edges: Whether to add reverse edges to the graph
            use_func_rel: Whether to use function relation edges
            oversample_negatives: Whether to use more negative samples than positive ones
        """
        self.graph_dir = graph_dir
        self.neg_multiplier = neg_multiplier if oversample_negatives else 1
        self.oversample_negatives = oversample_negatives
        self.sampling_strategy = sampling_strategy
        self.use_data_nodes = use_data_nodes
        self.use_func_rel = use_func_rel
        self.use_reverse_edges = use_reverse_edges
        
        # Get all graph file ids
        self.graph_files = []
        for filename in os.listdir(graph_dir):
            if filename.endswith('.graph.gz'):
                graph_id = filename.split('.')[0]
                self.graph_files.append(graph_id)
        
        self.graph_files.sort(key=lambda x: int(x))
        
        print(f"Found {len(self.graph_files)} valid graph files.")
        print(f"Using data nodes: {self.use_data_nodes}")
        print(f"Single-task learning mode")
        print(f"Negative sampling multiplier: {self.neg_multiplier}")
        print(f"Oversample negatives: {self.oversample_negatives}")
        
        # Define edge type mapping
        self.gt_edges_type = EDGE_TYPE_MAPPING
        
        # Create reverse mapping
        self.type_id_to_name = {v: k for k, v in self.gt_edges_type.items()}
    
    def __len__(self):
        return len(self.graph_files)

    def _load_candidate_meta(self, basedir, idx):
        """Load persisted angr-only candidate metadata for one graph index."""
        indextores_path = os.path.join(basedir, "indextores.json")
        if not os.path.exists(indextores_path):
            return {}

        try:
            with open(indextores_path, 'r') as f:
                indextores = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

        gt_folder = indextores.get(str(idx))
        if not gt_folder:
            return {}

        binary_name = os.path.basename(gt_folder)
        hubmeta_path = os.path.join(gt_folder, binary_name + HUBMETAEX)
        if not os.path.exists(hubmeta_path):
            return {}

        try:
            with open(hubmeta_path, 'r') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _normalize_task_name(self, task_name):
        """Map training task names to the sidecar's candidate keys."""
        return normalize_task_name(task_name)

    def _validate_edges(self, edges_tensor, labels, graph, idx, target_edge_type):
        """Validate edge tensor to prevent device-side assertions."""
        
        if edges_tensor.size(0) == 0:
            return edges_tensor, labels
        
        num_nodes = graph.num_nodes('code') if 'code' in graph.ntypes else graph.num_nodes()
        
        # Check edge tensor properties
        if edges_tensor.dim() != 2 or edges_tensor.shape[1] != 2:
            print(f"[ERROR] Graph {idx}: Invalid edge tensor shape {edges_tensor.shape}")
            return torch.zeros((0, 2), dtype=torch.long), torch.zeros(0, dtype=torch.float)
        
        # Ensure correct data type
        if edges_tensor.dtype != torch.long:
            edges_tensor = edges_tensor.long()

        
        # Check for negative indices
        if torch.any(edges_tensor < 0):
            print(f"[ERROR] Graph {idx}: Negative edge indices found")
            return torch.zeros((0, 2), dtype=torch.long), torch.zeros(0, dtype=torch.float)
        
        # Check bounds
        max_src = torch.max(edges_tensor[:, 0]).item() if edges_tensor.size(0) > 0 else -1
        max_dst = torch.max(edges_tensor[:, 1]).item() if edges_tensor.size(0) > 0 else -1
        
        if max_src >= num_nodes or max_dst >= num_nodes:
            print(f"[ERROR] Graph {idx} ({target_edge_type}): Edge index out of bounds. "
                f"Max src: {max_src}, Max dst: {max_dst}, Num nodes: {num_nodes}")
            return torch.zeros((0, 2), dtype=torch.long), torch.zeros(0, dtype=torch.float)
        
        # Check edge-label count match
        if edges_tensor.size(0) != labels.size(0):
            print(f"[ERROR] Graph {idx}: Edge-label count mismatch: {edges_tensor.size(0)} vs {labels.size(0)}")
            return torch.zeros((0, 2), dtype=torch.long), torch.zeros(0, dtype=torch.float)
        
        return edges_tensor, labels
    
    
    def load_item_multi_target(self, idx, mode, target_edge_types):
        """
        Load a single graph item and prepare target edge-specific data for multiple edge types.
        Args:
            idx: Graph file index
            mode: Loading mode ('train', 'eval', etc.)
            target_edge_types: List of edge types to load
        Returns:
            Dictionary with:
                - 'graph': shared DGLGraph object
                - 'targets': { edge_type: edge+label dict as in load_item() }
        """
        for edge_type in target_edge_types:
            if edge_type not in self.gt_edges_type:
                raise ValueError(f"Unknown target edge type: {edge_type}. "
                                f"Available types: {list(self.gt_edges_type.keys())}")
        
        # === Load and preprocess graph ===
        compressed_graph_path = os.path.join(self.graph_dir, f"{idx}.graph.gz")
        # DDP ranks can load the same graph concurrently (for example during
        # feature-dimension probing), so a fixed ``<id>.graph`` temporary path
        # is unsafe.  Use one private file per call and always clean it up.
        with tempfile.NamedTemporaryFile(
            prefix=f"neujump_{idx}_",
            suffix=".graph",
            delete=False,
        ) as temporary:
            graph_path = temporary.name
        try:
            with gzip.open(compressed_graph_path, 'rb') as f_in, open(graph_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            graph_list, *_ = dgl.load_graphs(graph_path)
        finally:
            if os.path.exists(graph_path):
                os.remove(graph_path)
        graph = graph_list[0]
        
        basedir = os.path.dirname(self.graph_dir)
        candidate_meta = self._load_candidate_meta(basedir, idx)
        gt_edges, gt_type, duplicate_gt, duplicate_type = loadgt(basedir, str(idx), mode)
        
        # === Optional filtering ===
        if not self.use_data_nodes and 'data' in graph.ntypes:
            graph = dgl.node_type_subgraph(graph, ['code'])
        
        if not self.use_func_rel and 'code2funchead' in graph.etypes:
            graph = dgl.remove_edges(graph, graph.edges(form='eid', etype='code2funchead'), etype='code2funchead')
        
        if self.use_reverse_edges:
            AddReverse = dgl.transforms.AddReverse(sym_new_etype=True)
            graph = AddReverse(graph)
        
        # === Group GT and duplicates by type once ===
        pos_edges_by_type = defaultdict(list)
        num_nodes = graph.num_nodes('code') if 'code' in graph.ntypes else graph.num_nodes()
        
        # Add edge validation during grouping
        for i, (src, dst) in enumerate(gt_edges):
            # Skip invalid edges
            if src >= num_nodes or dst >= num_nodes or src < 0 or dst < 0:
                continue
            edge_type = gt_type[i]
            pos_edges_by_type[edge_type].append((src, dst, i))
        
        duplicate_edges_by_type = []
        for i, (src, dst) in enumerate(duplicate_gt):
            # Skip invalid duplicate edges
            if src >= num_nodes or dst >= num_nodes or src < 0 or dst < 0:
                continue
            edge_type = duplicate_type[i]
            duplicate_edges_by_type.append((src, dst, edge_type))
        
        # === Prepare output per edge type ===
        results_by_edge_type = {}
        
        for edge_type in target_edge_types:
            type_id = self.gt_edges_type[edge_type]
            target_pos_edges = [(src, dst) for src, dst, *_ in pos_edges_by_type[type_id]]
            target_pos_indices = [idx for *_, idx in pos_edges_by_type[type_id]]
            
            if len(target_pos_edges) == 0:
                results_by_edge_type[edge_type] = {
                    'edges': torch.zeros((0, 2), dtype=torch.long),
                    'labels': torch.zeros(0, dtype=torch.float),
                    'pos_count': 0,
                    'neg_count': 0,
                    'target_edge_type': edge_type,
                    'gt_count': 0,
                    'pos_neg_ratio': 0.0,
                    'neg_multiplier': self.neg_multiplier
                }
                continue
            
            # Generate negative samples for this edge type
            neg_edges = self._generate_negative_samples_for_target(
                gt_edges,
                pos_edges_by_type,
                duplicate_edges_by_type,
                type_id,
                num_nodes
            )
            
            # Prepare final dataset
            num_pos = len(target_pos_edges)
            desired_neg = min(num_pos * self.neg_multiplier, len(neg_edges))
            selected_neg_edges = neg_edges[:desired_neg]
            
            # Convert to tensors in [num_edges, 2] format consistently
            all_edges = target_pos_edges + selected_neg_edges
            
            if all_edges:
                edges_tensor = torch.tensor(all_edges, dtype=torch.long)
            else:
                edges_tensor = torch.zeros((0, 2), dtype=torch.long)
            
            # Create labels (1 for positive, 0 for negative)
            pos_labels = torch.ones(len(target_pos_edges), dtype=torch.float)
            neg_labels = torch.zeros(len(selected_neg_edges), dtype=torch.float)
            labels = torch.cat([pos_labels, neg_labels])
            
            # ADDED: Validate edges before storing
            edges_tensor, labels = self._validate_edges(edges_tensor, labels, graph, idx, edge_type)
            
            results_by_edge_type[edge_type] = {
                'edges': edges_tensor,  # Shape: [num_edges, 2]
                'labels': labels,
                'pos_count': len(target_pos_edges) if edges_tensor.size(0) > 0 else 0,
                'neg_count': len(selected_neg_edges) if edges_tensor.size(0) > 0 else 0,
                'target_edge_type': edge_type,
                'gt_count': len(target_pos_edges) if edges_tensor.size(0) > 0 else 0,
                'pos_neg_ratio': len(selected_neg_edges) / max(len(target_pos_edges), 1) if edges_tensor.size(0) > 0 else 0.0,
                'neg_multiplier': self.neg_multiplier
            }
        
        return {
            'graph': graph,
            'candidate_meta': candidate_meta,
            'targets': results_by_edge_type
        }
    
    # def load_item_multi_target(self, idx, mode, target_edge_types):
    #     """
    #     Load a single graph item and prepare target edge-specific data for multiple edge types.

    #     Args:
    #         idx: Graph file index
    #         mode: Loading mode ('train', 'eval', etc.)
    #         target_edge_types: List of edge types to load

    #     Returns:
    #         Dictionary with:
    #             - 'graph': shared DGLGraph object
    #             - 'targets': { edge_type: edge+label dict as in load_item() }
    #     """
    #     for edge_type in target_edge_types:
    #         if edge_type not in self.gt_edges_type:
    #             raise ValueError(f"Unknown target edge type: {edge_type}. "
    #                             f"Available types: {list(self.gt_edges_type.keys())}")

    #     # === Load and preprocess graph ===
    #     compressed_graph_path = os.path.join(self.graph_dir, f"{idx}.graph.gz")
    #     graph_path = os.path.join(self.graph_dir, f"{idx}.graph")

    #     with gzip.open(compressed_graph_path, 'rb') as f_in, open(graph_path, 'wb') as f_out:
    #         shutil.copyfileobj(f_in, f_out)

    #     graph_list, *_ = dgl.load_graphs(graph_path)
    #     os.remove(graph_path)
    #     graph = graph_list[0]

    #     basedir = os.path.dirname(self.graph_dir)
    #     gt_edges, gt_type, duplicate_gt, duplicate_type = loadgt(basedir, str(idx), mode)

    #     # === Optional filtering ===
    #     if not self.use_data_nodes and 'data' in graph.ntypes:
    #         graph = dgl.node_type_subgraph(graph, ['code'])

    #     if not self.use_func_rel and 'code2funchead' in graph.etypes:
    #         graph = dgl.remove_edges(graph, graph.edges(form='eid', etype='code2funchead'), etype='code2funchead')

    #     if self.use_reverse_edges:
    #         AddReverse = dgl.transforms.AddReverse(sym_new_etype=True)
    #         graph = AddReverse(graph)

    #     # === Group GT and duplicates by type once ===
    #     pos_edges_by_type = defaultdict(list)
    #     for i, (src, dst) in enumerate(gt_edges):
    #         edge_type = gt_type[i]
    #         pos_edges_by_type[edge_type].append((src, dst, i))

    #     duplicate_edges_by_type = []
    #     for i, (src, dst) in enumerate(duplicate_gt):
    #         edge_type = duplicate_type[i]
    #         duplicate_edges_by_type.append((src, dst, edge_type))

    #     # === Prepare output per edge type ===
    #     results_by_edge_type = {}
    #     for edge_type in target_edge_types:
    #         type_id = self.gt_edges_type[edge_type]
    #         target_pos_edges = [(src, dst) for src, dst, *_ in pos_edges_by_type[type_id]]
    #         target_pos_indices = [idx for *_, idx in pos_edges_by_type[type_id]]

    #         if len(target_pos_edges) == 0:
    #             results_by_edge_type[edge_type] = {
    #                 'edges': torch.zeros((0, 2), dtype=torch.long),
    #                 'labels': torch.zeros(0, dtype=torch.float),
    #                 'pos_count': 0,
    #                 'neg_count': 0,
    #                 'target_edge_type': edge_type,
    #                 'gt_count': 0,
    #                 'pos_neg_ratio': 0.0,
    #                 'neg_multiplier': self.neg_multiplier
    #             }
    #             continue

    #         neg_edges = self._generate_negative_samples_for_target(
    #             gt_edges,
    #             pos_edges_by_type,
    #             duplicate_edges_by_type,
    #             type_id,
    #             graph.num_nodes('code')
    #         )

    #         num_pos = len(target_pos_edges)
    #         desired_neg = min(num_pos * self.neg_multiplier, len(neg_edges))
    #         selected_neg_edges = neg_edges[:desired_neg]

    #         all_edges = target_pos_edges + selected_neg_edges
    #         edges_tensor = torch.tensor(all_edges, dtype=torch.long) if all_edges else torch.zeros((0, 2), dtype=torch.long)

    #         pos_labels = torch.ones(len(target_pos_edges))
    #         neg_labels = torch.zeros(len(selected_neg_edges))
    #         labels = torch.cat([pos_labels, neg_labels])

    #         results_by_edge_type[edge_type] = {
    #             'edges': edges_tensor,
    #             'labels': labels,
    #             'pos_count': num_pos,
    #             'neg_count': len(selected_neg_edges),
    #             'target_edge_type': edge_type,
    #             'gt_count': num_pos,
    #             'pos_neg_ratio': len(selected_neg_edges) / max(num_pos, 1),
    #             'neg_multiplier': self.neg_multiplier
    #         }

    #     return {
    #         'graph': graph,
    #         'targets': results_by_edge_type
    #     }

    # def load_item(self, idx, mode, target_edge_type):
    #     """
    #     Load a single graph item with focus on the specified target edge type.
        
    #     Args:
    #         idx: Graph file index
    #         mode: Loading mode (e.g., 'train', 'eval')
    #         target_edge_type: The specific edge type to focus on (e.g., 'call', 'data_dep')
            
    #     Returns:
    #         Dictionary containing graph data focused on the target edge type
    #     """
    #     if target_edge_type not in self.gt_edges_type:
    #         raise ValueError(f"Unknown target edge type: {target_edge_type}. "
    #                        f"Available types: {list(self.gt_edges_type.keys())}")
        
    #     target_type_id = self.gt_edges_type[target_edge_type]
        
    #     # Load graph file
    #     compressed_graph_path = os.path.join(self.graph_dir, f"{idx}.graph.gz")
    #     graph_path = os.path.join(self.graph_dir, f"{idx}.graph")
        
    #     # Decompress the graph file
    #     with gzip.open(compressed_graph_path, 'rb') as f_in, open(graph_path, 'wb') as f_out:
    #         shutil.copyfileobj(f_in, f_out)
                
    #     graph_list, *_ = dgl.load_graphs(graph_path)
    #     os.remove(graph_path)
    #     graph = graph_list[0]
    #     basedir = os.path.dirname(self.graph_dir)
    #     gt_edges, gt_type, duplicate_gt, duplicate_type = loadgt(basedir, str(idx), mode)
        
    #     # Filter graph based on configuration
    #     if not self.use_data_nodes and 'data' in graph.ntypes:
    #         # Remove data nodes and related edges
    #         graph = dgl.node_type_subgraph(graph, ['code'])
        
    #     if not self.use_func_rel and 'code2funchead' in graph.etypes:
    #         graph = dgl.remove_edges(graph, graph.edges(form='eid', etype='code2funchead'), etype='code2funchead')
        
    #     if self.use_reverse_edges:
    #         AddReverse = dgl.transforms.AddReverse(sym_new_etype=True)
    #         graph = AddReverse(graph)
        
    #     # Group positive edges by type
    #     pos_edges_by_type = defaultdict(list)
    #     for i, (src, dst) in enumerate(gt_edges):
    #         edge_type = gt_type[i]
    #         pos_edges_by_type[edge_type].append((src, dst, i))  # Include original index
        
    #     # Get edges for the target type only
    #     target_pos_edges = [(src, dst) for src, dst, *_ in pos_edges_by_type[target_type_id]]
    #     target_pos_indices = [idx for *_, idx in pos_edges_by_type[target_type_id]]
        
    #     if len(target_pos_edges) == 0:
    #         # Return empty tensors in consistent format
    #         return {
    #             'graph': graph,
    #             'edges': torch.zeros((0, 2), dtype=torch.long),  # FIXED: [num_edges, 2] format
    #             'labels': torch.zeros(0, dtype=torch.float),
    #             'pos_count': 0,
    #             'neg_count': 0,
    #             'target_edge_type': target_edge_type,
    #             'gt_count': 0,
    #             'pos_neg_ratio': 0.0
    #         }
        
    #     # Group duplicate edges by type
    #     duplicate_edges_by_type = []
    #     for i, (src, dst) in enumerate(duplicate_gt):
    #         edge_type = duplicate_type[i]
    #         duplicate_edges_by_type.append((src, dst, edge_type))
        
    #     # Generate negative samples for the target type
    #     neg_edges = self._generate_negative_samples_for_target(
    #         gt_edges,
    #         pos_edges_by_type,
    #         duplicate_edges_by_type,
    #         target_type_id,
    #         graph.num_nodes('code')
    #     )
        
    #     # Prepare final dataset
    #     num_pos = len(target_pos_edges)
    #     desired_neg = min(num_pos * self.neg_multiplier, len(neg_edges))
    #     selected_neg_edges = neg_edges[:desired_neg]
        
    #     # FIXED: Convert to tensors in [num_edges, 2] format consistently
    #     all_edges = target_pos_edges + selected_neg_edges
        
    #     if all_edges:
    #         # Convert list of (src, dst) tuples to tensor of shape [num_edges, 2]
    #         edges_tensor = torch.tensor(all_edges, dtype=torch.long)  # Shape: [num_edges, 2]
    #     else:
    #         edges_tensor = torch.zeros((0, 2), dtype=torch.long)
        
    #     # Create labels (1 for positive, 0 for negative)
    #     pos_labels = torch.ones(len(target_pos_edges), dtype=torch.float)
    #     neg_labels = torch.zeros(len(selected_neg_edges), dtype=torch.float)

    #     labels = torch.cat([pos_labels, neg_labels])
        
    #     # Debug: Print tensor shapes
    #     # print(f"[DEBUG] {target_edge_type} - edges shape: {edges_tensor.shape}, labels shape: {labels.shape}")
    #     # print(f"[DEBUG] {target_edge_type} - pos: {num_pos}, neg: {len(selected_neg_edges)}")
        
    #     return {
    #         'graph': graph,
    #         'edges': edges_tensor,  # Shape: [num_edges, 2]
    #         'labels': labels,
    #         'pos_count': num_pos,
    #         'neg_count': len(selected_neg_edges),
    #         'target_edge_type': target_edge_type,
    #         'gt_count': num_pos,
    #         'pos_neg_ratio': len(selected_neg_edges) / max(num_pos, 1),
    #         'neg_multiplier': self.neg_multiplier
        # }
    def load_item(self, idx, mode, target_edge_type):
        """
        Load a single graph item with focus on the specified target edge type.
        """
        if target_edge_type not in self.gt_edges_type:
            raise ValueError(f"Unknown target edge type: {target_edge_type}. "
                        f"Available types: {list(self.gt_edges_type.keys())}")
        
        target_type_id = self.gt_edges_type[target_edge_type]
        
        # Load graph file
        compressed_graph_path = os.path.join(self.graph_dir, f"{idx}.graph.gz")
        cache_dir = os.environ.get("ICFLOWNET_TRAIN_GRAPH_CACHE_DIR")
        remove_graph_path = False
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            graph_path = os.path.join(cache_dir, f"{idx}.graph")
            if not os.path.exists(graph_path):
                with tempfile.NamedTemporaryFile(
                    prefix=f"neujump_{idx}_",
                    suffix=".graph",
                    dir=cache_dir,
                    delete=False,
                ) as temporary:
                    temporary_path = temporary.name
                try:
                    with gzip.open(compressed_graph_path, 'rb') as f_in, open(temporary_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.replace(temporary_path, graph_path)
                finally:
                    if os.path.exists(temporary_path):
                        os.remove(temporary_path)
        else:
            with tempfile.NamedTemporaryFile(
                prefix=f"neujump_{idx}_",
                suffix=".graph",
                delete=False,
            ) as temporary:
                graph_path = temporary.name
            remove_graph_path = True
            with gzip.open(compressed_graph_path, 'rb') as f_in, open(graph_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        try:
            graph_list, *_ = dgl.load_graphs(graph_path)
        finally:
            if remove_graph_path and os.path.exists(graph_path):
                os.remove(graph_path)
        graph = graph_list[0]
        basedir = os.path.dirname(self.graph_dir)
        candidate_meta = self._load_candidate_meta(basedir, idx)
        gt_edges, gt_type, duplicate_gt, duplicate_type = loadgt(basedir, str(idx), mode)
        
        # Filter graph based on configuration
        if not self.use_data_nodes and 'data' in graph.ntypes:
            # Remove data nodes and related edges
            graph = dgl.node_type_subgraph(graph, ['code'])
        
        if not self.use_func_rel and 'code2funchead' in graph.etypes:
            graph = dgl.remove_edges(graph, graph.edges(form='eid', etype='code2funchead'), etype='code2funchead')
        
        if self.use_reverse_edges:
            AddReverse = dgl.transforms.AddReverse(sym_new_etype=True)
            graph = AddReverse(graph)
        
        # Group positive edges by type
        pos_edges_by_type = defaultdict(list)
        for i, (src, dst) in enumerate(gt_edges):
            edge_type = gt_type[i]
            pos_edges_by_type[edge_type].append((src, dst, i))  # Include original index
        
        # Get edges for the target type only
        target_pos_edges = [(src, dst) for src, dst, *_ in pos_edges_by_type[target_type_id]]
        target_pos_indices = [idx for *_, idx in pos_edges_by_type[target_type_id]]
        
        if len(target_pos_edges) == 0:
            # Return empty tensors in consistent format
            return {
                'graph': graph,
                'candidate_meta': candidate_meta,
                'edges': torch.zeros((0, 2), dtype=torch.long),
                'labels': torch.zeros(0, dtype=torch.float),
                'pos_count': 0,
                'neg_count': 0,
                'target_edge_type': target_edge_type,
                'gt_count': 0,
                'pos_neg_ratio': 0.0
            }
        
        # Group duplicate edges by type
        duplicate_edges_by_type = []
        for i, (src, dst) in enumerate(duplicate_gt):
            edge_type = duplicate_type[i]
            duplicate_edges_by_type.append((src, dst, edge_type))
        
        # Generate negative samples for the target type
        neg_edges = self._generate_negative_samples_for_target(
            gt_edges,
            pos_edges_by_type,
            duplicate_edges_by_type,
            target_type_id,
            graph.num_nodes('code')
        )
        
        # Prepare final dataset
        num_pos = len(target_pos_edges)
        desired_neg = min(num_pos * self.neg_multiplier, len(neg_edges))
        selected_neg_edges = neg_edges[:desired_neg]
        
        # Convert to tensors in [num_edges, 2] format consistently
        all_edges = target_pos_edges + selected_neg_edges
        
        if all_edges:
            # Convert list of (src, dst) tuples to tensor of shape [num_edges, 2]
            edges_tensor = torch.tensor(all_edges, dtype=torch.long)
        else:
            edges_tensor = torch.zeros((0, 2), dtype=torch.long)
        
        # Create labels (1 for positive, 0 for negative)
        pos_labels = torch.ones(len(target_pos_edges), dtype=torch.float)
        neg_labels = torch.zeros(len(selected_neg_edges), dtype=torch.float)
        labels = torch.cat([pos_labels, neg_labels])
        
        # ADDED: Validate edges before returning
        edges_tensor, labels = self._validate_edges(edges_tensor, labels, graph, idx, target_edge_type)
        
        return {
            'graph': graph,
            'candidate_meta': candidate_meta,
            'edges': edges_tensor,  # Shape: [num_edges, 2]
            'labels': labels,
            'pos_count': len(target_pos_edges) if edges_tensor.size(0) > 0 else 0,
            'neg_count': len(selected_neg_edges) if edges_tensor.size(0) > 0 else 0,
            'target_edge_type': target_edge_type,
            'gt_count': len(target_pos_edges) if edges_tensor.size(0) > 0 else 0,
            'pos_neg_ratio': len(selected_neg_edges) / max(len(target_pos_edges), 1) if edges_tensor.size(0) > 0 else 0.0,
            'neg_multiplier': self.neg_multiplier
        }
    
    def _generate_negative_samples_for_target(self, gt_edges, pos_edges_by_type, 
                                            duplicate_edges_by_type, target_type_id, num_code_nodes):
        """
        Generate negative samples specifically for the target edge type.
        
        Args:
            gt_edges: List of all positive edges (src, dst)
            pos_edges_by_type: Dictionary of positive edges grouped by type
            duplicate_edges_by_type: List of duplicate edges to avoid
            target_type_id: The ID of the target edge type
            num_code_nodes: Total number of code nodes
            
        Returns:
            List of negative edges for the target type
        """
        # Create sets for fast lookup
        all_pos_edge_set = set((src, dst) for src, dst in gt_edges)
        duplicate_edge_set = set((src, dst, edge_type) for src, dst, edge_type in duplicate_edges_by_type)
        target_pos_edges = [(src, dst) for src, dst, *_ in pos_edges_by_type[target_type_id]]
        target_pos_set = set(target_pos_edges)
        
        if not target_pos_edges:
            return []
        
        num_pos = len(target_pos_edges)
        desired_neg_count = num_pos * self.neg_multiplier
        src_nodes = [src for src, *_ in target_pos_edges]
        
        all_neg_samples_set = set()
        neg_edges = []
        
        # Generate cross-type negative edges (use edges from other types as negatives)
        cross_type_neg_edges = []
        desired_cross_count = desired_neg_count // 2  # Try to get half from cross-type
        
        # Get edges from other types to use as negatives
        available_other_types = [other_type for other_type in pos_edges_by_type.keys() 
                               if other_type != target_type_id and len(pos_edges_by_type[other_type]) > 0]
        
        cross_type_stats = {}
        
        if len(available_other_types) > 0:
            for other_type in available_other_types:
                if len(cross_type_neg_edges) >= desired_cross_count:
                    break
                
                other_type_name = self.type_id_to_name.get(other_type, f"unknown_{other_type}")
                other_edges = pos_edges_by_type[other_type]
                
                # Track stats for this type
                cross_type_stats[other_type_name] = {'attempted': 0, 'selected': 0, 'rejected': 0}
                
                # Shuffle the other edges to get random selection
                other_edges_shuffled = list(other_edges)
                random.shuffle(other_edges_shuffled)
                
                for src, dst, *_ in other_edges_shuffled:
                    edge_tuple = (src, dst)
                    cross_type_stats[other_type_name]['attempted'] += 1
                    
                    # Check if this edge can be used as a negative
                    if (edge_tuple not in all_neg_samples_set and 
                        edge_tuple not in target_pos_set and
                        (src, dst, target_type_id) not in duplicate_edge_set):
                        
                        cross_type_neg_edges.append(edge_tuple)
                        all_neg_samples_set.add(edge_tuple)
                        cross_type_stats[other_type_name]['selected'] += 1
                        
                        if len(cross_type_neg_edges) >= desired_cross_count:
                            break
                    else:
                        cross_type_stats[other_type_name]['rejected'] += 1
        
        # Generate random negative samples for the remaining needed samples
        remaining_needed = desired_neg_count - len(cross_type_neg_edges)
        random_neg_edges = self._generate_random_neg_samples(
            target_type_id,
            all_pos_edge_set,
            duplicate_edge_set,
            all_neg_samples_set,
            src_nodes,
            remaining_needed,
            num_code_nodes
        )
        
        # Combine cross-type and random negatives
        neg_edges = cross_type_neg_edges + random_neg_edges
        
        return neg_edges
    
    def _generate_random_neg_samples(self, type_id, pos_edge_set, duplicate_edge_set, 
                                   all_neg_samples_set, src_nodes, num_samples, num_code_nodes):
        """Generate completely random negative samples."""
        samples = []
        
        # Get unique source nodes
        unique_src_nodes = list(set(src_nodes))
        
        if not unique_src_nodes:
            # If no source nodes, use random nodes
            unique_src_nodes = list(range(min(100, num_code_nodes)))
        
        # Generate random samples
        attempts = 0
        max_attempts = num_samples * 20  # Increased attempts for higher multipliers
        
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Random source and destination
            src = random.choice(unique_src_nodes)
            dst = random.randint(0, num_code_nodes - 1)
            
            # Check edge uniqueness
            if ((src, dst) not in pos_edge_set and 
                (src, dst) not in all_neg_samples_set and 
                (src, dst, type_id) not in duplicate_edge_set):
                samples.append((src, dst))
                all_neg_samples_set.add((src, dst))
        
        if len(samples) < num_samples:
            print(f"[WARNING] Could only generate {len(samples)}/{num_samples} random negative samples after {max_attempts} attempts")
        
        return samples
    
    def get_collate_fn(self):
        """Return a collate function for single-task learning."""
        return self.collate_single_task
    
    def collate_single_task(self, batch):
        """
        Collate function for single-task learning.
        
        Args:
            batch: List of data items from __getitem__
            
        Returns:
            Batched data dictionary
        """
        graphs = [item['graph'] for item in batch]
        edges = [item['edges'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Batch graphs
        batched_graph = dgl.batch(graphs)
        
        # Concatenate edges and labels
        if len(edges) > 0 and all(e.size(0) > 0 for e in edges):
            batched_edges = torch.cat(edges, dim=0)  # Concatenate along first dimension
            batched_labels = torch.cat(labels)
        else:
            batched_edges = torch.zeros((0, 2), dtype=torch.long)
            batched_labels = torch.zeros(0, dtype=torch.float)
        
        # Collect other info
        pos_counts = [item['pos_count'] for item in batch]
        neg_counts = [item['neg_count'] for item in batch]
        target_edge_types = [item['target_edge_type'] for item in batch]
        
        return {
            'graph': batched_graph,
            'edges': batched_edges,
            'labels': batched_labels,
            'pos_counts': pos_counts,
            'neg_counts': neg_counts,
            'target_edge_types': target_edge_types,
            'batch_size': len(batch)
        }
