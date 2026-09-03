"""Shared model constants, hub routing helpers, and heterogeneous GAT blocks."""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv


EDGE_TYPE_MAPPING = {
    "ret": 0,
    "jumptable": 1,
    "indirectcall": 3,
    "tailcall": 2,
}

EDGE_TYPE_TO_FNHASH = {
    "ret": "trainrethash",
    "jumptable": "trainjthash",
    "tailcall": "trainitchash",
    "indirectcall": "trainicallhash",
}

TASK_TO_CANDIDATE_KEY = {
    "indirectcall": "icall",
    "tailcall": "itailcall",
    "ret": "ret",
    "jumptable": "jumptable",
    "icall": "icall",
    "itailcall": "itailcall",
}

BASE_RELATION_TYPES = (
    "code2code_edges",
    "code2funchead",
    "codecall_edges",
    "codejump_edges",
    "codexrefcode_edges",
    "codexrefdata_edges",
    "dataxrefcode_edges",
    "dataxrefdata_edges",
    "code_to_gch",
    "gch_to_code",
    "data_to_gdh",
    "gdh_to_data",
    "gch_to_gdh",
    "gdh_to_gch",
)

PAPER_HUB_RELATION_TYPES = (
    "src_to_gch",
    "gch_to_src",
    "dst_to_gch",
    "gch_to_dst",
    "data_to_gdh",
    "gdh_to_data",
    "gch_to_gdh",
    "gdh_to_gch",
)

PAPER_RELATION_TYPES = tuple(
    dict.fromkeys(
        tuple(rel for rel in BASE_RELATION_TYPES if rel not in {
            "code_to_gch",
            "gch_to_code",
            "data_to_gdh",
            "gdh_to_data",
            "gch_to_gdh",
            "gdh_to_gch",
        })
        + PAPER_HUB_RELATION_TYPES
    )
)

XREF_RELATION_TYPES = {
    "codexrefcode_edges",
    "codexrefdata_edges",
    "dataxrefcode_edges",
    "dataxrefdata_edges",
}

PAPER_HUB_INCOMING_RELATIONS = {
    "gch_to_src",
    "gch_to_dst",
    "gdh_to_data",
}

HUB_KIND_GCH = 0
HUB_KIND_GDH = 1
MODEL_ARTIFACT_FORMAT = "icflownet_model_v1"


def save_model_artifact(path, model, model_config=None):
    """Save weights together with the model configuration needed to load them."""
    model_config = dict(model_config or {})
    for attribute in (
        "code_feat_dim",
        "data_feat_dim",
        "gdh_radius",
        "use_gch",
        "use_gdh",
        "task_aware_routing",
        "use_reverse_edges",
        "hidden_dim",
        "num_heads",
        "num_layers",
        "shared_layers",
        "task_layers",
        "dropout",
        "task_weights",
        "task_type",
    ):
        if hasattr(model, attribute):
            model_config.setdefault(attribute, getattr(model, attribute))
    artifact = {
        "artifact_format": MODEL_ARTIFACT_FORMAT,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
    }
    decision_threshold = getattr(model, "decision_threshold", None)
    if decision_threshold is not None:
        artifact["decision_threshold"] = float(decision_threshold)
    torch.save(artifact, path)


def load_model_artifact(model, path, map_location=None):
    """Load a versioned model artifact or training checkpoint."""
    artifact = torch.load(path, map_location=map_location)

    metadata = {}
    if isinstance(artifact, dict) and artifact.get("artifact_format") == MODEL_ARTIFACT_FORMAT:
        model_config = dict(artifact.get("model_config", {}))
        # Older model files included this redundant field in model_config.
        model_config.pop("architecture_version", None)
        stored_gdh_radius = model_config.get("gdh_radius")
        requested_gdh_radius = getattr(model, "gdh_radius", None)
        if (
            stored_gdh_radius is not None
            and requested_gdh_radius is not None
            and int(stored_gdh_radius) != int(requested_gdh_radius)
        ):
            raise RuntimeError(
                "Checkpoint GDH radius {} does not match requested radius {}.".format(
                    stored_gdh_radius,
                    requested_gdh_radius,
                )
            )
        state_dict = artifact["model_state_dict"]
        decision_threshold = float(artifact.get("decision_threshold", 0.5))
        if not 0.0 <= decision_threshold <= 1.0:
            raise RuntimeError(
                "Checkpoint decision threshold must be between 0 and 1."
            )
        model.decision_threshold = decision_threshold
        metadata = {
            "model_config": model_config,
            "artifact_format": artifact.get("artifact_format"),
            "decision_threshold": decision_threshold,
        }
    elif isinstance(artifact, dict) and "model_state_dict" in artifact:
        state_dict = artifact["model_state_dict"]
        metadata = {
            "artifact_format": "training_checkpoint",
        }
    else:
        raise RuntimeError(
            "Unsupported unversioned checkpoint. Expected a versioned model artifact "
            "or training checkpoint."
        )

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load checkpoint weights: {}".format(exc)
        ) from exc
    return metadata


def normalize_task_name(task_name):
    """Map training task names to the candidate sidecar's keys."""
    return TASK_TO_CANDIDATE_KEY.get(task_name, task_name)


def extract_node_features(g):
    """Collect per-node-type input features using the repo's existing conventions."""
    feat_dict = {}

    for ntype in g.ntypes:
        node_data = g.nodes[ntype].data
        if "featmean" in node_data:
            feat_dict[ntype] = node_data["featmean"]
        elif "feat" in node_data:
            feat_dict[ntype] = node_data["feat"]
        elif ntype == "hub" and "hub_kind" in node_data:
            # Paper-exact hubs use nn.Embedding rather than binary-derived input
            # features.  The model consumes hub_kind directly.
            continue
        elif g.num_nodes(ntype) == 0:
            continue
        else:
            raise RuntimeError(
                f"Missing 'featmean' or 'feat' in node features for node type '{ntype}'."
            )

    if "code" not in feat_dict:
        raise RuntimeError("Missing required 'code' node features.")

    return feat_dict


def add_paper_data_structural_features(g, feat_dict):
    """Append normalized base-graph in/out degrees to saved data features.

    Released graphs persist the normalized-address feature only.  The paper also
    describes lightweight structural data features, so paper_exact derives the
    available in/out-degree signals at runtime without changing the saved graph.
    Hub edges are excluded so these features remain properties of the base ACFG.
    """
    if "data" not in feat_dict or "data" not in g.ntypes:
        return feat_dict

    data_features = feat_dict["data"]
    num_data_nodes = g.num_nodes("data")
    in_degree = data_features.new_zeros(num_data_nodes)
    out_degree = data_features.new_zeros(num_data_nodes)

    for canonical_etype in g.canonical_etypes:
        src_type, rel_type, dst_type = canonical_etype
        if rel_type in PAPER_HUB_RELATION_TYPES:
            continue
        if src_type == "data":
            out_degree += g.out_degrees(etype=canonical_etype).to(
                device=data_features.device,
                dtype=data_features.dtype,
            )
        if dst_type == "data":
            in_degree += g.in_degrees(etype=canonical_etype).to(
                device=data_features.device,
                dtype=data_features.dtype,
            )

    def normalize_degree(values):
        values = torch.log1p(values)
        maximum = values.max() if values.numel() else values.new_zeros(())
        return values / maximum.clamp_min(1.0)

    structural = torch.stack(
        (normalize_degree(in_degree), normalize_degree(out_degree)),
        dim=1,
    )
    augmented = dict(feat_dict)
    augmented["data"] = torch.cat((data_features, structural), dim=1)
    return augmented


def _valid_node_ids(base_graph, values, ntype):
    """Return in-range integer node ids without trusting sidecar contents."""
    if ntype not in base_graph.ntypes:
        return set()

    max_nodes = base_graph.num_nodes(ntype)
    valid_ids = set()
    for value in values or ():
        try:
            node_id = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= node_id < max_nodes:
            valid_ids.add(node_id)
    return valid_ids


def collect_xref_data_neighborhood(base_graph, code_candidates, radius=2):
    """Collect data nodes within an undirected xRef-only hop radius.

    The paper defines GDH pruning by shortest-path distance in the recovered
    xRef subgraph.  Existing artifact generation treated code->data and
    data->code xRefs symmetrically for the one-hop case, so this implementation
    preserves that convention while extending it to a real bounded BFS.
    """
    if radius < 1 or "code" not in base_graph.ntypes or "data" not in base_graph.ntypes:
        return set()

    start_nodes = {
        ("code", node_id)
        for node_id in _valid_node_ids(base_graph, code_candidates, "code")
    }
    if not start_nodes:
        return set()

    adjacency = {}

    def add_neighbor(left, right):
        adjacency.setdefault(left, set()).add(right)

    for canonical_etype in base_graph.canonical_etypes:
        src_type, rel_type, dst_type = canonical_etype
        if rel_type not in XREF_RELATION_TYPES:
            continue

        src_nodes, dst_nodes = base_graph.edges(etype=canonical_etype)
        src_values = src_nodes.detach().cpu().tolist()
        dst_values = dst_nodes.detach().cpu().tolist()
        for src_id, dst_id in zip(src_values, dst_values):
            src_key = (src_type, int(src_id))
            dst_key = (dst_type, int(dst_id))
            add_neighbor(src_key, dst_key)
            add_neighbor(dst_key, src_key)

    visited = set(start_nodes)
    frontier = set(start_nodes)
    data_nodes = set()

    for _depth in range(1, radius + 1):
        next_frontier = set()
        for node_key in frontier:
            for neighbor in adjacency.get(node_key, ()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                next_frontier.add(neighbor)
                if neighbor[0] == "data":
                    data_nodes.add(neighbor[1])
        frontier = next_frontier
        if not frontier:
            break

    return data_nodes


def _cached_xref_data_neighborhood(
    base_graph,
    candidate_meta,
    cache_name,
    code_candidates,
    radius,
):
    """Cache topology-only GDH membership inside an in-memory metadata dict."""
    runtime_cache = candidate_meta.setdefault("_paper_exact_runtime_cache", {})
    signature = (
        int(radius),
        int(base_graph.num_nodes("code")) if "code" in base_graph.ntypes else 0,
        int(base_graph.num_nodes("data")) if "data" in base_graph.ntypes else 0,
        tuple(sorted(int(node_id) for node_id in code_candidates)),
    )
    cached = runtime_cache.get(cache_name)
    if cached is not None and cached.get("signature") == signature:
        return set(cached.get("data_candidates", ()))

    data_candidates = collect_xref_data_neighborhood(
        base_graph,
        code_candidates,
        radius=radius,
    )
    runtime_cache[cache_name] = {
        "signature": signature,
        "data_candidates": sorted(data_candidates),
    }
    return data_candidates


def _task_code_candidate_sets(base_graph, candidate_meta, candidate_key):
    code_meta = candidate_meta.get("code_candidates", {})
    src_candidates = _valid_node_ids(
        base_graph,
        code_meta.get("src", {}).get(candidate_key, []),
        "code",
    )
    dst_candidates = _valid_node_ids(
        base_graph,
        code_meta.get("dst", {}).get(candidate_key, []),
        "code",
    )

    if candidate_key == "jumptable":
        dst_by_src = code_meta.get("dst_by_src", {}).get("jumptable", {})
        for dst_nodes in dst_by_src.values():
            dst_candidates.update(_valid_node_ids(base_graph, dst_nodes, "code"))

    return src_candidates, dst_candidates


def _build_paper_augmented_graph(
    base_graph,
    src_candidates,
    dst_candidates,
    data_candidates,
    use_gch,
    use_gdh,
):
    """Materialize the paper's role-separated Dual-Hub topology."""
    use_gch = bool(use_gch and "code" in base_graph.ntypes)
    use_gdh = bool(use_gdh and "data" in base_graph.ntypes)
    if not use_gch and not use_gdh:
        return base_graph
    if "hub" in base_graph.ntypes:
        raise ValueError("Paper hub augmentation expects an unaugmented base graph.")
    if not src_candidates and not dst_candidates and not data_candidates:
        return base_graph

    graph_device = base_graph.device
    graph_dtype = base_graph.idtype
    graph_data = {}
    for canonical_etype in base_graph.canonical_etypes:
        graph_data[canonical_etype] = base_graph.edges(etype=canonical_etype)

    num_nodes_dict = {ntype: base_graph.num_nodes(ntype) for ntype in base_graph.ntypes}
    hub_node_ids = {}
    hub_kinds = []

    if use_gch:
        hub_node_ids["gch"] = len(hub_kinds)
        hub_kinds.append(HUB_KIND_GCH)
    if use_gdh:
        hub_node_ids["gdh"] = len(hub_kinds)
        hub_kinds.append(HUB_KIND_GDH)

    num_nodes_dict["hub"] = len(hub_kinds)

    def ids_tensor(values):
        return torch.tensor(list(values), dtype=graph_dtype, device=graph_device)

    def add_code_hub_relation(nodes, forward_rel, reverse_rel):
        if not nodes:
            return
        sorted_nodes = sorted(int(node_id) for node_id in nodes)
        gch_id = hub_node_ids["gch"]
        graph_data[("code", forward_rel, "hub")] = (
            ids_tensor(sorted_nodes),
            ids_tensor([gch_id] * len(sorted_nodes)),
        )
        graph_data[("hub", reverse_rel, "code")] = (
            ids_tensor([gch_id] * len(sorted_nodes)),
            ids_tensor(sorted_nodes),
        )

    if use_gch:
        add_code_hub_relation(src_candidates, "src_to_gch", "gch_to_src")
        add_code_hub_relation(dst_candidates, "dst_to_gch", "gch_to_dst")

    if use_gdh and data_candidates:
        sorted_data = sorted(int(node_id) for node_id in data_candidates)
        gdh_id = hub_node_ids["gdh"]
        graph_data[("data", "data_to_gdh", "hub")] = (
            ids_tensor(sorted_data),
            ids_tensor([gdh_id] * len(sorted_data)),
        )
        graph_data[("hub", "gdh_to_data", "data")] = (
            ids_tensor([gdh_id] * len(sorted_data)),
            ids_tensor(sorted_data),
        )

    if use_gch and use_gdh:
        graph_data[("hub", "gch_to_gdh", "hub")] = (
            ids_tensor([hub_node_ids["gch"]]),
            ids_tensor([hub_node_ids["gdh"]]),
        )
        graph_data[("hub", "gdh_to_gch", "hub")] = (
            ids_tensor([hub_node_ids["gdh"]]),
            ids_tensor([hub_node_ids["gch"]]),
        )

    augmented_graph = dgl.heterograph(
        graph_data,
        num_nodes_dict=num_nodes_dict,
        idtype=graph_dtype,
        device=graph_device,
    )

    for ntype in base_graph.ntypes:
        for feat_name, feat_tensor in base_graph.nodes[ntype].data.items():
            augmented_graph.nodes[ntype].data[feat_name] = feat_tensor

    augmented_graph.nodes["hub"].data["hub_kind"] = torch.tensor(
        hub_kinds,
        dtype=torch.long,
        device=graph_device,
    )
    return augmented_graph


def build_paper_task_augmented_graph(
    base_graph,
    candidate_meta,
    task_name,
    use_gch=False,
    use_gdh=False,
    task_aware_routing=True,
    gdh_radius=2,
):
    """Build the paper-exact single-task topology from a reusable base graph."""
    if not use_gch and not use_gdh:
        return base_graph
    if not candidate_meta:
        return base_graph

    candidate_key = normalize_task_name(task_name)
    src_candidates, dst_candidates = _task_code_candidate_sets(
        base_graph,
        candidate_meta,
        candidate_key,
    )
    # Jump-table dispatch sites still route to the source side of the code
    # hub.  Only their destination candidates are excluded by the paper's
    # task-aware topology.  Keep this identical to the unified MTL path below.
    if task_aware_routing and candidate_key == "jumptable":
        dst_candidates = set()

    data_candidates = _cached_xref_data_neighborhood(
        base_graph,
        candidate_meta,
        "single:{}".format(candidate_key),
        src_candidates | dst_candidates,
        gdh_radius,
    ) if use_gdh else set()

    return _build_paper_augmented_graph(
        base_graph,
        src_candidates,
        dst_candidates,
        data_candidates,
        use_gch,
        use_gdh,
    )


def build_paper_mtl_augmented_graph(
    base_graph,
    candidate_meta,
    use_gch=False,
    use_gdh=False,
    task_aware_routing=True,
    gdh_radius=2,
):
    """Build one unified paper MTL graph shared by all task branches."""
    if not use_gch and not use_gdh:
        return base_graph
    if not candidate_meta:
        return base_graph

    src_candidates = set()
    dst_candidates = set()
    for candidate_key in ("icall", "itailcall", "ret", "jumptable"):
        task_src, task_dst = _task_code_candidate_sets(
            base_graph,
            candidate_meta,
            candidate_key,
        )
        # Keep jump-table sources: angr cannot reliably distinguish them from
        # indirect-tail-call sources, and both belong to the shared
        # ``jumptable_itailcall`` source group recorded in hub metadata.
        src_candidates.update(task_src)
        if candidate_key != "jumptable" or not task_aware_routing:
            dst_candidates.update(task_dst)

    data_candidates = _cached_xref_data_neighborhood(
        base_graph,
        candidate_meta,
        "mtl",
        src_candidates | dst_candidates,
        gdh_radius,
    ) if use_gdh else set()

    return _build_paper_augmented_graph(
        base_graph,
        src_candidates,
        dst_candidates,
        data_candidates,
        use_gch,
        use_gdh,
    )


class PaperHeteroGATLayer(nn.Module):
    """Paper-exact HGAT layer with local-conditioned hub-gated fusion."""

    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        dropout=0.4,
        use_residual=True,
        use_reverse_edges=False,
    ):
        super(PaperHeteroGATLayer, self).__init__()

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        # ``out_dim`` is the total hidden width.  Split it evenly across the
        # attention heads and concatenate their outputs back to ``out_dim``.
        self.head_dim = out_dim // num_heads
        self.use_reverse_edges = use_reverse_edges
        self.use_residual = use_residual

        relation_types = list(PAPER_RELATION_TYPES)
        if use_reverse_edges:
            relation_types.extend(
                "rev_" + rel_type
                for rel_type in PAPER_RELATION_TYPES
                if rel_type not in PAPER_HUB_RELATION_TYPES
            )
        relation_types = tuple(dict.fromkeys(relation_types))

        self.gat_layers = nn.ModuleDict({
            rel_type: GATConv(
                in_dim,
                self.head_dim,
                num_heads,
                dropout,
                dropout,
                allow_zero_in_degree=True,
            )
            for rel_type in relation_types
        })

        if use_residual:
            self.residuals = nn.ModuleDict({
                ntype: nn.Linear(in_dim, out_dim)
                for ntype in ("code", "data")
            })
            # The implementation stores both virtual nodes under DGL's compact
            # ``hub`` node type, but retains the paper's distinct gcode/gdata
            # residual transforms through their hub_kind identities.
            self.hub_residuals = nn.ModuleList([
                nn.Linear(in_dim, out_dim),
                nn.Linear(in_dim, out_dim),
            ])

        ordinary_types = ("code", "data")
        self.hub_gates = nn.ModuleDict({
            ntype: nn.Linear(out_dim + in_dim, 1)
            for ntype in ordinary_types
        })
        self.hub_scales = nn.ParameterDict({
            ntype: nn.Parameter(torch.ones(()))
            for ntype in ordinary_types
        })
        self.layer_norms = nn.ModuleDict({
            ntype: nn.LayerNorm(out_dim)
            for ntype in ordinary_types
        })

    def _zero_message(self, nfeat):
        return nfeat.new_zeros((nfeat.shape[0], self.out_dim))

    def _sum_messages(self, messages, nfeat):
        if not messages:
            return self._zero_message(nfeat)
        return torch.stack(messages, dim=0).sum(dim=0)

    def forward(self, g, feat_dict):
        local_messages = {ntype: [] for ntype in feat_dict}
        hub_messages = {ntype: [] for ntype in feat_dict}

        for canonical_etype in g.canonical_etypes:
            src_type, rel_type, dst_type = canonical_etype
            if src_type not in feat_dict or dst_type not in feat_dict:
                continue
            if rel_type.startswith("rev_") and not self.use_reverse_edges:
                continue
            if rel_type not in self.gat_layers or g.num_edges(canonical_etype) == 0:
                continue

            relation_heads = self.gat_layers[rel_type](
                g[canonical_etype],
                (feat_dict[src_type], feat_dict[dst_type]),
            )
            relation_message = relation_heads.flatten(start_dim=1)

            if dst_type in ("code", "data") and rel_type in PAPER_HUB_INCOMING_RELATIONS:
                hub_messages[dst_type].append(relation_message)
            else:
                local_messages[dst_type].append(relation_message)

        final_features = {}
        for ntype, nfeat in feat_dict.items():
            local_message = self._sum_messages(local_messages[ntype], nfeat)
            if self.use_residual and ntype in self.residuals:
                residual = self.residuals[ntype](nfeat)
            elif self.use_residual and ntype == "hub":
                hub_kinds = g.nodes["hub"].data["hub_kind"].long()
                residual = self._zero_message(nfeat)
                for kind, transform in enumerate(self.hub_residuals):
                    kind_mask = (hub_kinds == kind).to(nfeat.dtype).unsqueeze(-1)
                    residual = residual + kind_mask * transform(nfeat)
            else:
                residual = self._zero_message(nfeat)

            if ntype in ("code", "data"):
                hub_message = self._sum_messages(hub_messages[ntype], nfeat)
                gate = torch.sigmoid(
                    self.hub_gates[ntype](torch.cat([local_message, nfeat], dim=-1))
                )
                combined = (
                    local_message
                    + self.hub_scales[ntype] * gate * hub_message
                    + residual
                )
                final_features[ntype] = F.elu(self.layer_norms[ntype](combined))
            else:
                # Hub nodes use the standard relation-aware update without the
                # ordinary-node local/hub gate from Equations (3)-(4).
                final_features[ntype] = F.elu(local_message + residual)

        return final_features
