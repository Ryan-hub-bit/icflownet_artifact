"""Single-task GAT for one indirect-control-flow relation."""

import torch
import torch.nn as nn
from common import (
    EDGE_TYPE_MAPPING,
    PaperHeteroGATLayer,
    add_paper_data_structural_features,
    build_paper_task_augmented_graph,
    extract_node_features,
)

class SingleTaskGAT(nn.Module):
    def __init__(
        self,
        code_feat_dim,
        data_feat_dim,
        task_type,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.4,
        use_reverse_edges=False,
        use_gch=True,
        use_gdh=True,
        task_aware_routing=True,
        gdh_radius=2,
        **checkpoint_compatibility,
    ):
        super(SingleTaskGAT, self).__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.task_type = task_type
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_reverse_edges = use_reverse_edges
        self.use_gch = use_gch
        self.use_gdh = use_gdh
        self.task_aware_routing = task_aware_routing
        # Accept metadata written by older checkpoints. The scorer architecture
        # itself is fixed to an MLP and is no longer configurable.
        checkpoint_compatibility.pop("architecture_version", None)
        checkpoint_compatibility.pop("edge_predictor_hidden_dim", None)
        if checkpoint_compatibility:
            raise TypeError(
                "Unexpected model configuration keys: {}".format(
                    ", ".join(sorted(checkpoint_compatibility))
                )
            )
        self.gdh_radius = int(gdh_radius)
        # Versioned artifacts may replace this after construction.  Keeping it
        # on the model makes the default binary decision self-contained.
        self.decision_threshold = 0.5
        self.edge_type_mapping = EDGE_TYPE_MAPPING
        
        # Input projections
        self.code_projection = nn.Linear(code_feat_dim, hidden_dim)
        projected_data_dim = data_feat_dim + 2 if data_feat_dim > 0 else data_feat_dim
        self.data_projection = nn.Linear(projected_data_dim, hidden_dim) if projected_data_dim > 0 else None
        self.hub_embedding = nn.Embedding(2, hidden_dim)
        
        # GAT layers
        self.encoder = nn.ModuleList([
            PaperHeteroGATLayer(hidden_dim, hidden_dim, num_heads, dropout, use_reverse_edges=use_reverse_edges)
            for _ in range(num_layers)
        ])
        
        # Endpoint scoring is always a two-layer MLP. The MLP width follows the
        # encoder width so there is no separate scorer-architecture option.
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, g, feat_dict):
        h_dict = {
            'code': self.code_projection(feat_dict['code'])
        }
        if 'data' in feat_dict and self.data_projection is not None:
            h_dict['data'] = self.data_projection(feat_dict['data'])
        if 'hub' in g.ntypes:
            h_dict['hub'] = self.hub_embedding(g.nodes['hub'].data['hub_kind'].long())
            
        for layer in self.encoder:
            h_dict = layer(g, h_dict)
            
        return h_dict

    def forward(self, g, edge_pairs=None, candidate_meta=None, return_embeddings=False):
        """
        Forward pass of the model.
        
        Args:
            g: DGL graph
            edge_pairs: Edge pairs to predict - can be:
                       - Tuple of (src_ids, dst_ids)
                       - Tensor of shape [num_edges, 2] where each row is [src, dst]
                       - Tensor of shape [2, num_edges] where first row is src, second is dst
            return_embeddings: If True, return (embeddings, logits), else return only logits
            
        Returns:
            If edge_pairs is None: node embeddings
            If return_embeddings is True: (node_embeddings, edge_logits)
            Else: edge_logits only
        """
        g = build_paper_task_augmented_graph(
            g,
            candidate_meta,
            self.task_type,
            use_gch=self.use_gch,
            use_gdh=self.use_gdh,
            task_aware_routing=self.task_aware_routing,
            gdh_radius=self.gdh_radius,
        )
        feat_dict = extract_node_features(g)
        feat_dict = add_paper_data_structural_features(g, feat_dict)

        node_embeddings = self.encode(g, feat_dict)

        if edge_pairs is None:
            return node_embeddings

        edge_logits = self.score_edges(node_embeddings, edge_pairs)

        if return_embeddings:
            return node_embeddings, edge_logits
        else:
            # Return only edge logits for training/evaluation
            return edge_logits

    def score_edges(self, node_embeddings, edge_pairs):
        """Score endpoint pairs from one reusable graph encoding."""

        # FIXED: Handle different edge_pairs formats
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

        # Get embeddings for source and destination nodes
        src_embs = node_embeddings['code'][src_ids]
        dst_embs = node_embeddings['code'][dst_ids]
        
        # Concatenate source and destination embeddings
        edge_emb = torch.cat([src_embs, dst_embs], dim=1)
        return self.edge_predictor(edge_emb)

    def get_loss(self, edge_logits, edge_labels, edge_types, pos_weight=1.0):
        task_type_id = self.edge_type_mapping[self.task_type]
        type_labels = (edge_types == task_type_id) & (edge_labels == 1)
        type_logits = edge_logits.squeeze()
        
        n_pos = type_labels.sum().float()
        n_neg = len(type_labels) - n_pos
        type_pos_weight = (n_neg / n_pos) * pos_weight if n_pos > 0 else torch.tensor(pos_weight, device=type_logits.device)
        
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        losses = loss_fn(type_logits, type_labels.float())
        
        if n_pos > 0:
            losses[type_labels] *= type_pos_weight
            
        return losses.mean()

    def predict_edges(self, g, edge_pairs, threshold=None, candidate_meta=None):
        with torch.no_grad():
            edge_logits = self.forward(g, edge_pairs, candidate_meta=candidate_meta)
            probs = torch.sigmoid(edge_logits.squeeze())
            if threshold is None:
                threshold = self.decision_threshold
            return (probs >= threshold), probs

    def evaluate(self, g, pos_edges, pos_types, neg_edges, threshold=None, candidate_meta=None):
        """
        Args:
            pos_edges: List of tuples [(src, dst), ...] OR tensor of shape [num_edges, 2]
            pos_types: List of edge types
            neg_edges: List of tuples [(src, dst), ...] OR tensor of shape [num_edges, 2]
        """
        task_type_id = self.edge_type_mapping[self.task_type]
        
        # Handle different input formats
        if isinstance(pos_edges, torch.Tensor):
            # Already tensor format
            pos_edges_tensor = pos_edges
        else:
            # List of tuples format
            pos_edges_tensor = torch.tensor(pos_edges, dtype=torch.long, device=g.device)
            
        if isinstance(neg_edges, torch.Tensor):
            # Already tensor format  
            neg_edges_tensor = neg_edges
        else:
            # List of tuples format
            neg_edges_tensor = torch.tensor(neg_edges, dtype=torch.long, device=g.device)
        
        # Concatenate all edges
        all_edges = torch.cat([pos_edges_tensor, neg_edges_tensor], dim=0)
        
        all_labels = torch.cat([
            torch.ones(len(pos_edges), device=g.device),
            torch.zeros(len(neg_edges), device=g.device)
        ])
        all_types = torch.cat([
            torch.tensor(pos_types, dtype=torch.long, device=g.device),
            torch.full((len(neg_edges),), -1, dtype=torch.long, device=g.device)
        ])

        preds, probs = self.predict_edges(g, all_edges, threshold, candidate_meta=candidate_meta)
        type_gt = (all_types == task_type_id)
        
        tp = ((preds == 1) & type_gt).sum().item()
        fp = ((preds == 1) & ~type_gt).sum().item()
        fn = ((preds == 0) & type_gt).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": type_gt.sum().item()
        }

    def train_step(self, g, pos_edges, pos_types, neg_edges, optimizer, pos_weight=1.0, candidate_meta=None):
        """
        Args:
            pos_edges: List of tuples [(src, dst), ...] OR tensor of shape [num_edges, 2]
            pos_types: List of edge types
            neg_edges: List of tuples [(src, dst), ...] OR tensor of shape [num_edges, 2]
        """
        # Handle different input formats
        if isinstance(pos_edges, torch.Tensor):
            # Already tensor format
            pos_edges_tensor = pos_edges
        else:
            # List of tuples format
            pos_edges_tensor = torch.tensor(pos_edges, dtype=torch.long, device=g.device)
            
        if isinstance(neg_edges, torch.Tensor):
            # Already tensor format  
            neg_edges_tensor = neg_edges
        else:
            # List of tuples format
            neg_edges_tensor = torch.tensor(neg_edges, dtype=torch.long, device=g.device)
        
        # Concatenate all edges
        edge_pairs = torch.cat([pos_edges_tensor, neg_edges_tensor], dim=0)
            
        edge_labels = torch.cat([
            torch.ones(len(pos_edges), device=g.device),
            torch.zeros(len(neg_edges), device=g.device)
        ])
        edge_types = torch.cat([
            torch.tensor(pos_types, dtype=torch.long, device=g.device),
            torch.full((len(neg_edges),), -1, dtype=torch.long, device=g.device)
        ])

        edge_logits = self.forward(g, edge_pairs, candidate_meta=candidate_meta)
        loss = self.get_loss(edge_logits, edge_labels, edge_types, pos_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
