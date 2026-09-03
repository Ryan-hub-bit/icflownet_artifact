"""Multi-task GAT for joint indirect-control-flow prediction."""

import torch
import torch.nn as nn
import dgl
from common import (
    EDGE_TYPE_MAPPING,
    PaperHeteroGATLayer,
    add_paper_data_structural_features,
    build_paper_mtl_augmented_graph,
    extract_node_features,
)

class MultiTaskGAT(nn.Module):
    def __init__(
        self,
        code_feat_dim,
        data_feat_dim,
        hidden_dim=256,
        num_heads=8,
        shared_layers=4,
        task_layers=1,
        dropout=0.4,
        use_reverse_edges=False,
        task_weights=None,
        use_gch=True,
        use_gdh=True,
        task_aware_routing=True,
        gdh_radius=2,
        **checkpoint_compatibility,
    ):
        super(MultiTaskGAT, self).__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.edge_type_mapping = EDGE_TYPE_MAPPING
        self.code_feat_dim = code_feat_dim
        self.data_feat_dim = data_feat_dim
        self.shared_layers = shared_layers
        self.use_reverse_edges = use_reverse_edges
        self.use_gch = use_gch
        self.use_gdh = use_gdh
        self.task_aware_routing = task_aware_routing
        # Accept the redundant field written by older checkpoints.
        checkpoint_compatibility.pop("architecture_version", None)
        if checkpoint_compatibility:
            raise TypeError(
                "Unexpected model configuration keys: {}".format(
                    ", ".join(sorted(checkpoint_compatibility))
                )
            )
        self.gdh_radius = int(gdh_radius)
        self.task_layers = task_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Task weights for loss balancing
        self.task_weights = task_weights or {et: 1.0 for et in self.edge_type_mapping}
        self.task_weights["edge_type"] = self.task_weights.get("edge_type", 1.0)
        
        # Input projections
        self.code_projection = nn.Linear(code_feat_dim, hidden_dim)
        projected_data_dim = data_feat_dim + 2 if data_feat_dim > 0 else data_feat_dim
        self.data_projection = nn.Linear(projected_data_dim, hidden_dim) if projected_data_dim > 0 else None
        self.hub_embedding = nn.Embedding(2, hidden_dim)
        
        # Shared encoder layers
        self.shared_encoder = nn.ModuleList([
            PaperHeteroGATLayer(
                hidden_dim,
                hidden_dim,
                num_heads,
                dropout,
                use_reverse_edges=use_reverse_edges,
            )
            for _ in range(shared_layers)
        ])
        
        # Task-specific encoder layers
        self.task_encoders = nn.ModuleDict({
            task: nn.ModuleList([
                PaperHeteroGATLayer(
                    hidden_dim,
                    hidden_dim,
                    num_heads,
                    dropout,
                    use_reverse_edges=use_reverse_edges,
                )
                for _ in range(task_layers)
            ]) for task in self.edge_type_mapping
        })
        
        # Both endpoint matching and edge-type prediction always use two-layer
        # MLPs. Their hidden width follows the encoder width, leaving no
        # alternative scorer architecture to configure.
        self.edge_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for task in self.edge_type_mapping
        })
        self.edge_type_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.edge_type_mapping)),
        )
    
    def encode(self, g, task=None):
        """Encode graph nodes with optional task-specific layers"""
        feat_dict = extract_node_features(g)
        feat_dict = add_paper_data_structural_features(g, feat_dict)
        h_dict = {
            'code': self.code_projection(feat_dict['code'])
        }
        if 'data' in feat_dict and self.data_projection is not None:
            h_dict['data'] = self.data_projection(feat_dict['data'])
        if 'hub' in g.ntypes:
            h_dict['hub'] = self.hub_embedding(g.nodes['hub'].data['hub_kind'].long())
        
        # Apply shared layers
        for layer in self.shared_encoder:
            h_dict = layer(g, h_dict)
        
        # Apply task-specific layers if specified
        if task is not None and task in self.task_encoders:
            for layer in self.task_encoders[task]:
                h_dict = layer(g, h_dict)
        
        return h_dict
    
    def forward(self, g, edge_pairs=None, candidate_meta=None, return_embeddings=False):
        """
        Forward pass supporting multiple input formats like SingleTaskGAT
        
        Args:
            g: DGL graph
            edge_pairs: Edge pairs to predict - can be:
                       - Tuple of (src_ids, dst_ids)
                       - Tensor of shape [num_edges, 2] where each row is [src, dst]
                       - Tensor of shape [2, num_edges] where first row is src, second is dst
            return_embeddings: If True, return embeddings along with predictions
        
        Returns:
            If edge_pairs is None: node embeddings
            If return_embeddings is True: (node_embeddings, edge_logits, edge_type_logits)
            Else: (edge_logits, edge_type_logits)
        """
        model_graph = build_paper_mtl_augmented_graph(
            g,
            candidate_meta,
            use_gch=self.use_gch,
            use_gdh=self.use_gdh,
            task_aware_routing=self.task_aware_routing,
            gdh_radius=self.gdh_radius,
        )

        # Run the shared encoder once on the unified graph.
        node_embs_shared = self.encode(model_graph)
        
        if edge_pairs is None:
            return node_embs_shared
        
        # FIXED: Handle different edge_pairs formats like SingleTaskGAT
        if isinstance(edge_pairs, tuple):
            # Format: (src_ids, dst_ids)
            src_ids, dst_ids = edge_pairs
        elif isinstance(edge_pairs, torch.Tensor):
            if edge_pairs.dim() == 2:
                if edge_pairs.shape[1] == 2:
                    # Format: [num_edges, 2] where each row is [src, dst]
                    src_ids = edge_pairs[:, 0]
                    dst_ids = edge_pairs[:, 1]
                elif edge_pairs.shape[0] == 2:
                    # Format: [2, num_edges] where first row is src, second is dst
                    src_ids = edge_pairs[0, :]
                    dst_ids = edge_pairs[1, :]
                else:
                    raise ValueError(f"Unexpected edge_pairs tensor shape: {edge_pairs.shape}")
            else:
                raise ValueError(f"Unexpected edge_pairs tensor dimensions: {edge_pairs.dim()}")
        else:
            raise ValueError(f"Unexpected edge_pairs type: {type(edge_pairs)}")
        
        # Shared embeddings for edge type classification
        shared_edge_embs = torch.cat([
            node_embs_shared['code'][src_ids], 
            node_embs_shared['code'][dst_ids]
        ], dim=1)
        
        # Task-specific edge predictions
        edge_logits = {}
        for task in self.edge_type_mapping:
            node_embs_task = dict(node_embs_shared)
            for layer in self.task_encoders[task]:
                node_embs_task = layer(model_graph, node_embs_task)
            task_edge_embs = torch.cat([
                node_embs_task['code'][src_ids], 
                node_embs_task['code'][dst_ids]
            ], dim=1)
            edge_logits[task] = self.edge_heads[task](task_edge_embs)
        
        # Edge type classification
        edge_type_logits = self.edge_type_head(shared_edge_embs)
        
        if return_embeddings:
            return node_embs_shared, edge_logits, edge_type_logits
        else:
            return edge_logits, edge_type_logits
    
    def get_loss(
        self,
        edge_logits,
        edge_labels,
        edge_types,
        pos_weight=1.0,
        edge_type_logits=None,
        negative_class_weights=None,
        hard_negative_focus=None,
    ):
        """Compute multi-task loss with proper weighting"""
        bce = nn.BCEWithLogitsLoss(reduction='none')
        total_loss = 0
        losses = {}
        
        for task_name, type_id in self.edge_type_mapping.items():
            # Each classifier consumes only its own positives and independently
            # sampled negatives, as required by Section 4.4.
            type_mask = (edge_types == type_id)
            if not type_mask.any():
                continue

            task_labels = edge_labels[type_mask].float()
            logits = edge_logits[task_name].reshape(-1)[type_mask]
            
            # Calculate class weights
            n_pos = task_labels.sum().float()
            n_neg = len(task_labels) - n_pos
            
            if n_pos > 0:
                weight = (n_neg / n_pos) * pos_weight
                loss = bce(logits, task_labels)
                loss[task_labels == 1] *= weight
            else:
                loss = bce(logits, task_labels)

            negative_mask = task_labels == 0
            if negative_mask.any():
                class_weight = float(
                    (negative_class_weights or {}).get(task_name, 1.0)
                )
                if class_weight != 1.0:
                    loss[negative_mask] *= class_weight
                focus = (hard_negative_focus or {}).get(task_name)
                if focus is not None:
                    alpha, gamma = map(float, focus)
                    negative_scores = torch.sigmoid(
                        logits[negative_mask].detach()
                    )
                    loss[negative_mask] *= (
                        1.0 + alpha * negative_scores.pow(gamma)
                    )
            
            losses[task_name] = loss.mean()
            total_loss += self.task_weights[task_name] * losses[task_name]
        
        # Edge type classification loss (only for positive edges)
        if edge_type_logits is not None:
            pos_mask = edge_labels == 1
            if pos_mask.sum() > 0:
                type_loss = nn.CrossEntropyLoss()(
                    edge_type_logits[pos_mask], 
                    edge_types[pos_mask]
                )
                losses["edge_type"] = type_loss
                total_loss += self.task_weights["edge_type"] * type_loss
        
        return total_loss, losses
    
    def predict_edges(self, g, edge_pairs, threshold=0.5, candidate_meta=None):
        """Predict edges for all tasks"""
        with torch.no_grad():
            edge_logits, edge_type_logits = self.forward(
                g,
                edge_pairs,
                candidate_meta=candidate_meta,
            )
            
            predictions = {}
            probabilities = {}
            
            for task in self.edge_type_mapping:
                probs = torch.sigmoid(edge_logits[task].squeeze())
                preds = (probs >= threshold)
                predictions[task] = preds
                probabilities[task] = probs
            
            # Edge type predictions
            type_probs = torch.softmax(edge_type_logits, dim=1)
            type_preds = torch.argmax(type_probs, dim=1)
            
            return predictions, probabilities, type_preds, type_probs
    
    def evaluate(self, g, pos_edges, pos_types, neg_edges, threshold=0.5, candidate_meta=None):
        """Evaluate model performance on all tasks"""
        # Handle different input formats like SingleTaskGAT
        if isinstance(pos_edges, torch.Tensor):
            pos_edges_tensor = pos_edges
        else:
            pos_edges_tensor = torch.tensor(pos_edges, dtype=torch.long, device=g.device)
            
        if isinstance(neg_edges, torch.Tensor):
            neg_edges_tensor = neg_edges
        else:
            neg_edges_tensor = torch.tensor(neg_edges, dtype=torch.long, device=g.device)
        
        # Combine all edges
        all_edges = torch.cat([pos_edges_tensor, neg_edges_tensor], dim=0)
        all_labels = torch.cat([
            torch.ones(len(pos_edges), device=g.device),
            torch.zeros(len(neg_edges), device=g.device)
        ])
        all_types = torch.cat([
            torch.tensor(pos_types, dtype=torch.long, device=g.device),
            torch.full((len(neg_edges),), -1, dtype=torch.long, device=g.device)
        ])
        
        # Get predictions
        predictions, probabilities, type_preds, type_probs = self.predict_edges(
            g,
            all_edges,
            threshold,
            candidate_meta=candidate_meta,
        )
        
        # Calculate metrics for each task
        results = {}
        for task_name, type_id in self.edge_type_mapping.items():
            type_gt = (all_types == type_id) & (all_labels == 1)
            task_preds = predictions[task_name]
            
            tp = (task_preds & type_gt).sum().item()
            fp = (task_preds & ~type_gt).sum().item()
            fn = (~task_preds & type_gt).sum().item()
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)
            
            results[task_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": type_gt.sum().item()
            }
        
        return results
    
    def train_step(self, g, pos_edges, pos_types, neg_edges, optimizer, pos_weight=1.0, candidate_meta=None):
        """Training step with improved input handling"""
        # Handle different input formats like SingleTaskGAT
        if isinstance(pos_edges, torch.Tensor):
            pos_edges_tensor = pos_edges
        else:
            pos_edges_tensor = torch.tensor(pos_edges, dtype=torch.long, device=g.device)
            
        if isinstance(neg_edges, torch.Tensor):
            neg_edges_tensor = neg_edges
        else:
            neg_edges_tensor = torch.tensor(neg_edges, dtype=torch.long, device=g.device)
        
        # Combine edges - use [num_edges, 2] format
        edge_pairs = torch.cat([pos_edges_tensor, neg_edges_tensor], dim=0)
        
        edge_labels = torch.cat([
            torch.ones(len(pos_edges), device=g.device),
            torch.zeros(len(neg_edges), device=g.device)
        ])
        edge_types = torch.cat([
            torch.tensor(pos_types, device=g.device),
            torch.full((len(neg_edges),), -1, dtype=torch.long, device=g.device)
        ])
        
        # Forward pass
        edge_logits, edge_type_logits = self.forward(
            g,
            edge_pairs,
            candidate_meta=candidate_meta,
        )
        
        # Compute loss
        loss, type_losses = self.get_loss(edge_logits, edge_labels, edge_types, pos_weight, edge_type_logits)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item(), {k: v.item() for k, v in type_losses.items()}
