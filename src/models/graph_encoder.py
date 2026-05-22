"""
E3 — Graph Attention Network Encoder  (Stage 3 — I-CARE)

Learns DOC-specific connectivity features from the I-CARE dataset.
Nodes = EEG electrodes (19 × standard 10-20);  Edges = dwPLI connectivity.

Design rationale
────────────────
Cardiac-arrest coma patients (CPC=4) show characteristic breakdown of
whole-brain functional connectivity:
  • Delta/theta connectivity dominates in UWS / CPC=4 (Chennu 2014 principle)
  • Alpha-band "rich club" network preserved in CPC=1/2 recovering patients
  • MCS / CPC=3 shows partial alpha restoration

A GAT (Graph Attention Network) operating on electrode-level dwPLI naturally
captures these topology changes.  Multi-band (delta/theta/alpha/beta) node
features enrich the representation beyond single-band connectivity.

The learned graph embedding is one of three encoder streams in the PDI-CCS
fusion: its disagreement with E1 (pathology-focused) and E2 (affective
processing) operationalises the covert awareness hypothesis.

References:
  I-CARE: Kjaergaard et al. (2023). I-CARE: International Cardiac Arrest
    REsearch. PhysioNet 2023.  doi:10.13026/2ksm-4p10
  dwPLI: Vinck et al. (2011). NeuroImage 57(4):2161-2177.
  GAT: Veličković et al. (2018). Graph Attention Networks. ICLR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    _TG_AVAILABLE = True
except ImportError:
    _TG_AVAILABLE = False
    # Provide a minimal fallback so the module is importable without torch_geometric
    # The GAT layers will raise a clear error when actually called.

from src.config import (
    GNN_HIDDEN_DIM,
    GNN_LAYERS,
    GNN_HEADS,
    GNN_DROPOUT,
    CHENNU_EEG_CH,
    NUM_DOC_CLASSES,
    RANDOM_SEED,
)


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction from connectivity matrix
# ─────────────────────────────────────────────────────────────────────────────

def connectivity_to_graph(
    node_features: torch.Tensor,    # (n_nodes, n_band_features)
    adj_matrix:    torch.Tensor,    # (n_nodes, n_nodes)  dwPLI weights
    threshold:     float = 0.1,     # prune weak edges
) -> "Data":
    """
    Convert a connectivity matrix into a torch_geometric Data object.

    Nodes: EEG electrodes
    Edges: pairs with dwPLI > threshold (symmetric)
    Edge weights: dwPLI value
    """
    if not _TG_AVAILABLE:
        raise ImportError("torch_geometric is required for GraphEncoder. Install it first.")

    n      = adj_matrix.size(0)
    mask   = (adj_matrix > threshold) & (~torch.eye(n, dtype=torch.bool, device=adj_matrix.device))
    src, dst = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr  = adj_matrix[src, dst].unsqueeze(1)

    return Data(
        x          = node_features.float(),
        edge_index = edge_index,
        edge_attr  = edge_attr.float(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GAT Block
# ─────────────────────────────────────────────────────────────────────────────

class GATBlock(nn.Module):
    """
    Single GAT layer with residual + normalisation.
    """

    def __init__(
        self,
        in_dim:   int,
        out_dim:  int,
        heads:    int = GNN_HEADS,
        dropout:  float = GNN_DROPOUT,
        concat:   bool  = False,
    ) -> None:
        super().__init__()
        if not _TG_AVAILABLE:
            raise ImportError("torch_geometric is required for GATBlock.")
        self.gat = GATConv(
            in_dim, out_dim, heads=heads, dropout=dropout, concat=concat
        )
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.res  = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index, edge_attr=None):
        out = self.gat(x, edge_index, edge_attr=edge_attr)
        out = self.drop(F.elu(self.norm(out)))
        return out + self.res(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full Graph Encoder (E3)
# ─────────────────────────────────────────────────────────────────────────────

class GraphEncoder(nn.Module):
    """
    Shallow GAT encoder for DOC EEG connectivity graphs.

    Input  : torch_geometric Batch (or Data)
    Output : (B, embed_dim) — graph-level embeddings

    Node features expected: band power values per electrode.
    Default: 4 bands (delta/theta/alpha/beta) → in_node_dim=4.
    """

    def __init__(
        self,
        in_node_dim: int  = 4,          # 4 frequency bands
        hidden_dim:  int  = GNN_HIDDEN_DIM,
        n_layers:    int  = GNN_LAYERS,
        gat_heads:   int  = GNN_HEADS,
        dropout:     float = GNN_DROPOUT,
        embed_dim:   int  = 64,
        num_classes: int  = NUM_DOC_CLASSES,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_node_dim, hidden_dim)
        self.gat_layers = nn.ModuleList([
            GATBlock(hidden_dim, hidden_dim, gat_heads, dropout)
            for _ in range(n_layers)
        ])
        self.pool_fc    = nn.Sequential(
            nn.Linear(hidden_dim * 2, embed_dim),  # mean + max pooling
            nn.LayerNorm(embed_dim),
            nn.ELU(),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.embed_dim  = embed_dim

    def embed(self, batch) -> torch.Tensor:
        """Returns (B, embed_dim) graph embedding."""
        x = F.elu(self.input_proj(batch.x))
        for layer in self.gat_layers:
            x = layer(x, batch.edge_index, batch.edge_attr)
        # Readout: concatenate mean + max pool
        b = batch.batch if hasattr(batch, "batch") and batch.batch is not None \
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        g_mean = global_mean_pool(x, b)    # (B, H)
        g_max  = global_max_pool(x, b)     # (B, H)
        g      = torch.cat([g_mean, g_max], dim=-1)  # (B, 2H)
        return self.pool_fc(g)

    def forward(self, batch):
        """
        Returns:
            logits  : (B, num_classes)
            emb     : (B, embed_dim)
        """
        emb    = self.embed(batch)
        logits = self.classifier(emb)
        return logits, emb


# ─────────────────────────────────────────────────────────────────────────────
# Scaffold: Stage 3 training entry point (requires Chennu 2014)
# ─────────────────────────────────────────────────────────────────────────────

def train_graph_encoder(
    graph_loader,   # DataLoader yielding torch_geometric Batch objects
    device: str,
    epochs: int = 80,
    pretrained_state: Optional[dict] = None,
) -> GraphEncoder:
    """
    Fine-tune GraphEncoder on I-CARE DOC graphs.

    graph_loader : DataLoader where each item is a torch_geometric Batch
    """
    from src.stage2_emotion.train_emotion_encoder import GeneralisedCrossEntropyLoss

    model     = GraphEncoder().to(device)
    criterion = GeneralisedCrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

    model.train()
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for batch in graph_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            logits, _ = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            ep_loss += loss.item()
        if ep % 10 == 0:
            print(f"[GraphEncoder] Epoch {ep}/{epochs}  loss={ep_loss/len(graph_loader):.4f}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test  (no real data needed)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _TG_AVAILABLE:
        print("torch_geometric not installed — skipping smoke test.")
    else:
        from torch_geometric.data import Batch
        torch.manual_seed(RANDOM_SEED)

        n_nodes = 19   # I-CARE standard 10-20 (19 channels)
        # Synthetic single graph: random band-power node features + random adjacency
        node_feat = torch.randn(n_nodes, 4)
        adj  = torch.rand(n_nodes, n_nodes)
        adj  = (adj + adj.T) / 2
        data = connectivity_to_graph(node_feat, adj, threshold=0.5)
        data.y = torch.tensor([1])

        batch = Batch.from_data_list([data, data])
        model = GraphEncoder()
        model.eval()
        with torch.no_grad():
            logits, emb = model(batch)
        print(f"GraphEncoder logits: {logits.shape}   embed: {emb.shape}")
