from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


if torch is None:

    def build_normalized_adjacency(*args, **kwargs):
        raise ImportError("torch is required to build the LightGCN adjacency")


    class LightGCNBackbone:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("torch is required to instantiate LightGCNBackbone")

else:

    def build_normalized_adjacency(edge_index: torch.Tensor, n_nodes: int, device: torch.device) -> torch.Tensor:
        edge_index = edge_index.to(device=device, dtype=torch.long)
        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(n_nodes, device=device, dtype=torch.float32)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))
        deg = deg.clamp(min=1.0)
        deg_inv_sqrt = deg.pow(-0.5)
        values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj = torch.sparse_coo_tensor(edge_index, values, (n_nodes, n_nodes), device=device)
        return adj.coalesce()


    def _sparse_edge_dropout(adj: torch.Tensor, drop_rate: float) -> torch.Tensor:
        """Randomly drop edges from a sparse adjacency matrix during training."""
        if drop_rate <= 0.0:
            return adj
        indices = adj.indices()
        values = adj.values()
        mask = torch.bernoulli(torch.full_like(values, 1.0 - drop_rate)).bool()
        new_indices = indices[:, mask]
        new_values = values[mask] / (1.0 - drop_rate)  # rescale to preserve expectation
        return torch.sparse_coo_tensor(new_indices, new_values, adj.size(), device=adj.device).coalesce()


    class LightGCNBackbone(nn.Module):
        def __init__(self, n_users: int, n_items: int, embedding_dim: int, num_layers: int,
                     num_clusters: int = 0) -> None:
            super().__init__()
            self.n_users = int(n_users)
            self.n_items = int(n_items)
            self.embedding_dim = int(embedding_dim)
            self.num_layers = int(num_layers)
            self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
            self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.xavier_uniform_(self.user_embedding.weight)
            nn.init.xavier_uniform_(self.item_embedding.weight)

        def forward(
            self, norm_adj: torch.Tensor, edge_drop_rate: float = 0.0,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            initial = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            adj = norm_adj
            if self.training and edge_drop_rate > 0.0:
                adj = _sparse_edge_dropout(norm_adj, edge_drop_rate)
            layers = [initial]
            propagated = initial
            for _ in range(self.num_layers):
                propagated = torch.sparse.mm(adj, propagated)
                layers.append(propagated)
            stacked = torch.stack(layers, dim=0).mean(dim=0)
            return stacked[: self.n_users], stacked[self.n_users :]
