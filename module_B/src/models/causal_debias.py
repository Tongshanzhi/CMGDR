from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Function
except ImportError:
    torch = None
    nn = None
    F = None
    Function = object

from ..utils import mode_components
from .lightgcn_backbone import LightGCNBackbone


if torch is None:

    class CMGDRModel:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("torch is required to instantiate CMGDRModel")

else:

    class GradientReversal(Function):
        @staticmethod
        def forward(ctx, tensor: torch.Tensor, lambda_: float) -> torch.Tensor:
            ctx.lambda_ = lambda_
            return tensor.view_as(tensor)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
            return -ctx.lambda_ * grad_output, None


    def grad_reverse(tensor: torch.Tensor, lambda_: float) -> torch.Tensor:
        return GradientReversal.apply(tensor, lambda_)


    class CMGDRModel(nn.Module):
        def __init__(
            self,
            n_users: int,
            n_items: int,
            visual_dim: int,
            num_clusters: int,
            embedding_dim: int,
            num_layers: int,
            text_dim: int = 0,
            use_item_item_graph: bool = False,
        ) -> None:
            super().__init__()
            self.n_users = int(n_users)
            self.n_items = int(n_items)
            self.visual_dim = int(visual_dim)
            self.num_clusters = int(num_clusters)
            self.embedding_dim = int(embedding_dim)
            self.num_layers = int(num_layers)
            self.text_dim = int(text_dim)
            self.use_item_item_graph = use_item_item_graph

            self.backbone = LightGCNBackbone(
                n_users=self.n_users,
                n_items=self.n_items,
                embedding_dim=self.embedding_dim,
                num_layers=self.num_layers,
            )
            self.visual_projector = nn.Sequential(
                nn.Linear(self.visual_dim, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            )
            # Text projector (if text features available)
            if self.text_dim > 0:
                self.text_projector = nn.Sequential(
                    nn.Linear(self.text_dim, self.embedding_dim),
                    nn.LayerNorm(self.embedding_dim),
                    nn.ReLU(),
                    nn.Linear(self.embedding_dim, self.embedding_dim),
                )
            # Item-Item GNN layer (if copurchase graph available)
            if self.use_item_item_graph:
                self.item_item_gate = nn.Sequential(
                    nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                    nn.Sigmoid(),
                )
            self.cluster_embedding = nn.Embedding(self.num_clusters, self.embedding_dim)
            self.bias_encoder = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            )
            self.causal_encoder = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            )
            self.cluster_classifier = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.num_clusters),
            )
            # Prototype-conditioned GNN: learnable same-cluster vs cross-cluster weights
            self.proto_same_weight = nn.Parameter(torch.tensor(1.5))
            self.proto_cross_weight = nn.Parameter(torch.tensor(1.0))

        def _visual_latents(
            self,
            visual_features: torch.Tensor,
            visual_clusters: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            visual_latent = self.visual_projector(visual_features)
            cluster_latent = self.cluster_embedding(visual_clusters)
            bias_latent = self.bias_encoder(torch.cat([visual_latent, cluster_latent], dim=-1))
            return visual_latent, cluster_latent, bias_latent

        def _causal_latent(
            self,
            item_graph_embeddings: torch.Tensor,
            bias_latent: torch.Tensor,
        ) -> torch.Tensor:
            return self.causal_encoder(
                torch.cat([item_graph_embeddings, item_graph_embeddings - bias_latent], dim=-1)
            )

        def encode_all(
            self,
            norm_adj: torch.Tensor,
            visual_features: torch.Tensor,
            visual_clusters: torch.Tensor,
            cluster_prototypes: torch.Tensor,
            mode: str = "full",
            grl_lambda: float = 1.0,
            text_features: torch.Tensor | None = None,
            item_item_adj: torch.Tensor | None = None,
            edge_drop_rate: float = 0.0,
        ) -> dict[str, torch.Tensor | str | None]:
            flags = mode_components(mode)
            user_embeddings, item_graph_embeddings = self.backbone(norm_adj, edge_drop_rate=edge_drop_rate)

            # Enhance item embeddings with text features
            text_latent = None
            if text_features is not None and self.text_dim > 0:
                text_latent = self.text_projector(text_features)
                item_graph_embeddings = item_graph_embeddings + text_latent

            # Enhance item embeddings with item-item graph propagation
            # with prototype-conditioned weighting
            if item_item_adj is not None and self.use_item_item_graph:
                item_neighbor = torch.sparse.mm(item_item_adj, item_graph_embeddings)
                # Prototype-conditioned: same-cluster neighbors get higher weight
                if visual_clusters is not None:
                    neighbor_cluster_emb = torch.sparse.mm(item_item_adj, self.cluster_embedding(visual_clusters).float())
                    self_cluster_emb = self.cluster_embedding(visual_clusters)
                    similarity = F.cosine_similarity(self_cluster_emb, neighbor_cluster_emb, dim=-1)
                    # Interpolate between cross_weight and same_weight based on similarity
                    proto_weight = self.proto_cross_weight + (self.proto_same_weight - self.proto_cross_weight) * (similarity + 1) / 2
                    item_neighbor = item_neighbor * proto_weight.unsqueeze(-1)
                gate = self.item_item_gate(torch.cat([item_graph_embeddings, item_neighbor], dim=-1))
                item_graph_embeddings = item_graph_embeddings + gate * item_neighbor

            if not flags["visual"]:
                zeros = torch.zeros_like(item_graph_embeddings)
                return {
                    "mode": mode,
                    "user_embeddings": user_embeddings,
                    "item_graph_embeddings": item_graph_embeddings,
                    "item_causal_embeddings": item_graph_embeddings,
                    "item_bias_embeddings": zeros,
                    "item_total_embeddings": item_graph_embeddings,
                    "counterfactual_causal_embeddings": item_graph_embeddings,
                    "counterfactual_total_embeddings": item_graph_embeddings,
                    "cluster_logits": None,
                    "visual_latent": None,
                    "text_latent": text_latent,
                }

            visual_latent, cluster_latent, bias_latent = self._visual_latents(visual_features, visual_clusters)
            prototype_visual_latent = self.visual_projector(cluster_prototypes[visual_clusters])

            if mode == "visual_concat":
                item_causal_embeddings = item_graph_embeddings + visual_latent
                item_bias_embeddings = visual_latent
                item_total_embeddings = item_causal_embeddings
                counterfactual_causal_embeddings = item_graph_embeddings + prototype_visual_latent
                counterfactual_total_embeddings = counterfactual_causal_embeddings
            else:
                item_bias_embeddings = bias_latent
                item_causal_embeddings = self._causal_latent(item_graph_embeddings, item_bias_embeddings)
                item_total_embeddings = item_causal_embeddings + item_bias_embeddings

                counterfactual_bias = self.bias_encoder(
                    torch.cat([prototype_visual_latent, cluster_latent], dim=-1)
                )
                counterfactual_causal_embeddings = self._causal_latent(
                    item_graph_embeddings, counterfactual_bias
                )
                counterfactual_total_embeddings = counterfactual_causal_embeddings + counterfactual_bias

            cluster_logits = self.cluster_classifier(grad_reverse(item_causal_embeddings, grl_lambda))
            return {
                "mode": mode,
                "user_embeddings": user_embeddings,
                "item_graph_embeddings": item_graph_embeddings,
                "item_causal_embeddings": item_causal_embeddings,
                "item_bias_embeddings": item_bias_embeddings,
                "item_total_embeddings": item_total_embeddings,
                "counterfactual_causal_embeddings": counterfactual_causal_embeddings,
                "counterfactual_total_embeddings": counterfactual_total_embeddings,
                "cluster_logits": cluster_logits,
                "visual_latent": visual_latent,
                "text_latent": text_latent,
            }

        @staticmethod
        def score_pairs(
            user_embeddings: torch.Tensor,
            item_embeddings: torch.Tensor,
            user_indices: torch.Tensor,
            item_indices: torch.Tensor,
        ) -> torch.Tensor:
            users = user_embeddings[user_indices]
            items = item_embeddings[item_indices]
            return (users * items).sum(dim=-1)

        def forward(
            self,
            user_indices: torch.Tensor,
            pos_item_indices: torch.Tensor,
            neg_item_indices: torch.Tensor,
            norm_adj: torch.Tensor,
            visual_features: torch.Tensor,
            visual_clusters: torch.Tensor,
            cluster_prototypes: torch.Tensor,
            mode: str = "full",
            grl_lambda: float = 1.0,
            text_features: torch.Tensor | None = None,
            item_item_adj: torch.Tensor | None = None,
            edge_drop_rate: float = 0.0,
        ) -> dict[str, torch.Tensor | str | None]:
            outputs = self.encode_all(
                norm_adj=norm_adj,
                visual_features=visual_features,
                visual_clusters=visual_clusters,
                cluster_prototypes=cluster_prototypes,
                mode=mode,
                grl_lambda=grl_lambda,
                text_features=text_features,
                item_item_adj=item_item_adj,
                edge_drop_rate=edge_drop_rate,
            )
            user_embeddings = outputs["user_embeddings"]
            causal_embeddings = outputs["item_causal_embeddings"]
            total_embeddings = outputs["item_total_embeddings"]
            counterfactual_causal = outputs["counterfactual_causal_embeddings"]
            counterfactual_total = outputs["counterfactual_total_embeddings"]

            outputs.update(
                {
                    "pos_causal_scores": self.score_pairs(
                        user_embeddings, causal_embeddings, user_indices, pos_item_indices
                    ),
                    "neg_causal_scores": self.score_pairs(
                        user_embeddings, causal_embeddings, user_indices, neg_item_indices
                    ),
                    "pos_total_scores": self.score_pairs(
                        user_embeddings, total_embeddings, user_indices, pos_item_indices
                    ),
                    "neg_total_scores": self.score_pairs(
                        user_embeddings, total_embeddings, user_indices, neg_item_indices
                    ),
                    "pos_causal_cf_scores": self.score_pairs(
                        user_embeddings, counterfactual_causal, user_indices, pos_item_indices
                    ),
                    "neg_causal_cf_scores": self.score_pairs(
                        user_embeddings, counterfactual_causal, user_indices, neg_item_indices
                    ),
                    "pos_total_cf_scores": self.score_pairs(
                        user_embeddings, counterfactual_total, user_indices, pos_item_indices
                    ),
                    "neg_total_cf_scores": self.score_pairs(
                        user_embeddings, counterfactual_total, user_indices, neg_item_indices
                    ),
                }
            )
            return outputs
