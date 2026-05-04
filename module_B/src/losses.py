from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


if torch is None:

    def _torch_required() -> None:
        raise ImportError("torch is required to compute CMGDR losses")


    def bpr_loss(*args, **kwargs):
        _torch_required()


    def residual_loss(*args, **kwargs):
        _torch_required()


    def adversarial_loss(*args, **kwargs):
        _torch_required()


    def counterfactual_consistency_loss(*args, **kwargs):
        _torch_required()


    def orthogonality_loss(*args, **kwargs):
        _torch_required()

else:

    def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        return -F.logsigmoid(pos_scores - neg_scores).mean()


    def residual_loss(
        item_graph_embeddings: torch.Tensor,
        causal_item_embeddings: torch.Tensor,
        bias_item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(item_graph_embeddings, causal_item_embeddings + bias_item_embeddings)


    def adversarial_loss(cluster_logits: torch.Tensor, visual_clusters: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(cluster_logits, visual_clusters)


    def counterfactual_consistency_loss(
        causal_scores: torch.Tensor,
        counterfactual_causal_scores: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(causal_scores, counterfactual_causal_scores)


    def orthogonality_loss(causal_item_embeddings: torch.Tensor, bias_item_embeddings: torch.Tensor) -> torch.Tensor:
        causal_norm = F.normalize(causal_item_embeddings, dim=-1)
        bias_norm = F.normalize(bias_item_embeddings, dim=-1)
        cosine_overlap = (causal_norm * bias_norm).sum(dim=-1)
        return (cosine_overlap.pow(2)).mean()


    def cross_modal_contrastive_loss(
        visual_latent: torch.Tensor,
        text_latent: torch.Tensor,
        temperature: float = 0.2,
    ) -> torch.Tensor:
        """InfoNCE loss: same-item visual and text representations are positive pairs."""
        v_norm = F.normalize(visual_latent, dim=-1)
        t_norm = F.normalize(text_latent, dim=-1)
        logits = v_norm @ t_norm.T / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


    def text_consistency_loss(
        predicted_scores: torch.Tensor,
        review_ratings: torch.Tensor,
    ) -> torch.Tensor:
        """Penalise divergence between predicted preference and review sentiment."""
        pred_prob = torch.sigmoid(predicted_scores)
        return F.mse_loss(pred_prob, review_ratings)
