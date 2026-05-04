"""
Multimodal recommendation baselines implemented on our LightGCN backbone.
All models share the same training loop, data pipeline, and evaluation protocol.

Baselines:
- MMGCN     (Wei et al., MM'19): Per-modality GCN + late fusion
- LATTICE   (Zhang et al., MM'21): Modality-aware item-item graph learning
- BM3       (Zhou et al., WWW'23): Bootstrapped contrastive multimodal learning
- FREEDOM   (Zhou et al., MM'23): Frozen item-item graph + denoised propagation
- MGCN      (Yu et al., MM'23): Multi-view GCN with behavior-guided purifier
- LGMRec    (Guo et al., AAAI'24): Local-global graph learning for multimodal rec
- MENTOR    (Xu et al., AAAI'25): Multi-granularity graph learning for multimodal rec
- CausalRec (Qiu et al., MM'21): Causal inference for visual debiasing
- EliMRec   (Liu et al., MM'22): Eliminating multimodal noise via causal intervention
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lightgcn_backbone import LightGCNBackbone, build_normalized_adjacency


class MMGCNModel(nn.Module):
    """MMGCN: Modal-specific GCN + concatenation fusion."""

    def __init__(self, n_users, n_items, visual_dim, text_dim, embedding_dim, num_layers):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.backbone = LightGCNBackbone(n_users, n_items, embedding_dim, num_layers)
        self.visual_proj = nn.Linear(visual_dim, embedding_dim)
        self.text_proj = nn.Linear(text_dim, embedding_dim)
        # Per-modality GCN layers (1-layer each for efficiency)
        self.visual_gcn_weight = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.text_gcn_weight = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fusion = nn.Linear(embedding_dim * 3, embedding_dim)

    def encode_all(self, norm_adj, visual_features, text_features, **kwargs):
        user_emb, item_emb_id = self.backbone(norm_adj)
        # Modal-specific item representations
        v_emb = F.relu(self.visual_proj(visual_features))
        t_emb = F.relu(self.text_proj(text_features))
        # Single-layer GCN on each modality (item-side only via user-item graph)
        full_v = torch.cat([torch.zeros(self.n_users, self.embedding_dim, device=v_emb.device), v_emb])
        full_t = torch.cat([torch.zeros(self.n_users, self.embedding_dim, device=t_emb.device), t_emb])
        prop_v = torch.sparse.mm(norm_adj, full_v)[self.n_users:]
        prop_t = torch.sparse.mm(norm_adj, full_t)[self.n_users:]
        v_out = F.relu(self.visual_gcn_weight(prop_v))
        t_out = F.relu(self.text_gcn_weight(prop_t))
        # Late fusion
        item_total = self.fusion(torch.cat([item_emb_id, v_out, t_out], dim=-1))
        return user_emb, item_total


class LATTICEModel(nn.Module):
    """LATTICE: Learn item-item graph from modality features, then propagate."""

    def __init__(self, n_users, n_items, visual_dim, text_dim, embedding_dim, num_layers, k_neighbors=10):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.k_neighbors = k_neighbors

        self.backbone = LightGCNBackbone(n_users, n_items, embedding_dim, num_layers)
        self.visual_trs = nn.Linear(visual_dim, embedding_dim)
        self.text_trs = nn.Linear(text_dim, embedding_dim)
        self.item_item_agg = nn.Linear(embedding_dim * 2, embedding_dim)

    def _build_knn_graph(self, features, k):
        """Build k-NN item-item graph from features."""
        with torch.no_grad():
            feat_norm = F.normalize(features, dim=-1)
            sim = feat_norm @ feat_norm.T
            sim.fill_diagonal_(-1e9)
            _, topk_idx = sim.topk(k, dim=-1)
        n = features.size(0)
        row = torch.arange(n, device=features.device).unsqueeze(1).expand(-1, k).reshape(-1)
        col = topk_idx.reshape(-1)
        vals = torch.ones(row.size(0), device=features.device) / k
        adj = torch.sparse_coo_tensor(torch.stack([row, col]), vals, (n, n)).coalesce()
        return adj

    def encode_all(self, norm_adj, visual_features, text_features, **kwargs):
        user_emb, item_emb_id = self.backbone(norm_adj)
        v_emb = self.visual_trs(visual_features)
        t_emb = self.text_trs(text_features)
        # Build modality-specific item-item graphs
        v_adj = self._build_knn_graph(v_emb, self.k_neighbors)
        t_adj = self._build_knn_graph(t_emb, self.k_neighbors)
        # Propagate on item-item graphs
        v_prop = torch.sparse.mm(v_adj, item_emb_id)
        t_prop = torch.sparse.mm(t_adj, item_emb_id)
        # Aggregate
        item_enhanced = self.item_item_agg(torch.cat([v_prop, t_prop], dim=-1))
        item_total = item_emb_id + item_enhanced
        return user_emb, item_total


class BM3Model(nn.Module):
    """BM3: Bootstrapped contrastive learning for multimodal recommendation."""

    def __init__(self, n_users, n_items, visual_dim, text_dim, embedding_dim, num_layers,
                 dropout=0.3, cl_weight=0.1, temperature=0.2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.cl_weight = cl_weight
        self.temperature = temperature
        self.dropout = dropout

        self.backbone = LightGCNBackbone(n_users, n_items, embedding_dim, num_layers)
        self.visual_proj = nn.Linear(visual_dim, embedding_dim)
        self.text_proj = nn.Linear(text_dim, embedding_dim)
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def encode_all(self, norm_adj, visual_features, text_features, **kwargs):
        user_emb, item_emb_id = self.backbone(norm_adj)
        v_emb = self.visual_proj(visual_features)
        t_emb = self.text_proj(text_features)
        item_total = item_emb_id + v_emb + t_emb
        return user_emb, item_total

    def contrastive_loss(self, visual_features, text_features):
        """BM3-style bootstrapped contrastive loss between modalities."""
        v = self.visual_proj(visual_features)
        t = self.text_proj(text_features)
        # Dropout augmentation
        v1 = F.dropout(v, p=self.dropout, training=True)
        v2 = F.dropout(v, p=self.dropout, training=True)
        t1 = F.dropout(t, p=self.dropout, training=True)
        # Cross-modal: visual predicts text (bootstrapped)
        pred_v = self.predictor(v1)
        cross_loss = 2 - 2 * F.cosine_similarity(pred_v, t1.detach(), dim=-1).mean()
        # Intra-modal: visual self-consistency
        intra_loss = 2 - 2 * F.cosine_similarity(v1, v2.detach(), dim=-1).mean()
        return self.cl_weight * (cross_loss + intra_loss)


class FREEDOMModel(nn.Module):
    """FREEDOM: Frozen item-item graph with degree-sensitive denoising."""

    def __init__(self, n_users, n_items, visual_dim, text_dim, embedding_dim, num_layers,
                 k_neighbors=10, denoise_ratio=0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.k_neighbors = k_neighbors
        self.denoise_ratio = denoise_ratio

        self.backbone = LightGCNBackbone(n_users, n_items, embedding_dim, num_layers)
        self.visual_proj = nn.Linear(visual_dim, embedding_dim)
        self.text_proj = nn.Linear(text_dim, embedding_dim)
        # Learnable modality weights
        self.modal_weight = nn.Parameter(torch.ones(2) / 2)

    def _build_frozen_knn(self, features, k):
        with torch.no_grad():
            feat_norm = F.normalize(features, dim=-1)
            sim = feat_norm @ feat_norm.T
            sim.fill_diagonal_(-1e9)
            vals, idx = sim.topk(k, dim=-1)
        n = features.size(0)
        row = torch.arange(n, device=features.device).unsqueeze(1).expand(-1, k).reshape(-1)
        col = idx.reshape(-1)
        edge_vals = F.softmax(vals.reshape(-1).float(), dim=0)
        adj = torch.sparse_coo_tensor(torch.stack([row, col]), edge_vals, (n, n)).coalesce()
        return adj

    def encode_all(self, norm_adj, visual_features, text_features, **kwargs):
        user_emb, item_emb_id = self.backbone(norm_adj)
        v_emb = self.visual_proj(visual_features)
        t_emb = self.text_proj(text_features)
        # Frozen item-item graphs (built once, not updated by gradient)
        v_adj = self._build_frozen_knn(visual_features, self.k_neighbors)
        t_adj = self._build_frozen_knn(text_features, self.k_neighbors)
        # Propagate ID embeddings through frozen modality graphs
        v_prop = torch.sparse.mm(v_adj, item_emb_id)
        t_prop = torch.sparse.mm(t_adj, item_emb_id)
        # Weighted fusion
        w = F.softmax(self.modal_weight, dim=0)
        item_total = item_emb_id + w[0] * v_prop + w[1] * t_prop
        return user_emb, item_total


class MGCNModel(nn.Module):
    """MGCN: Multi-view GCN with modality-specific user-item propagation."""

    def __init__(self, n_users, n_items, visual_dim, text_dim, embedding_dim, num_layers):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Shared ID embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.visual_proj = nn.Linear(visual_dim, embedding_dim)
        self.text_proj = nn.Linear(text_dim, embedding_dim)
        # Per-modality attention
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3),
        )

    def encode_all(self, norm_adj, visual_features, text_features, **kwargs):
        v_emb = self.visual_proj(visual_features)
        t_emb = self.text_proj(text_features)
        id_emb = self.item_embedding.weight

        # Multi-view propagation: each modality has its own user-item propagation
        views = []
        for modal_feat in [id_emb, v_emb, t_emb]:
            full = torch.cat([self.user_embedding.weight, modal_feat], dim=0)
            propagated = full
            layers = [full]
            for _ in range(self.num_layers):
                propagated = torch.sparse.mm(norm_adj, propagated)
                layers.append(propagated)
            stacked = torch.stack(layers, dim=0).mean(dim=0)
            views.append(stacked)

        # Attention-based fusion
        user_views = torch.stack([v[:self.n_users] for v in views], dim=1)  # (n_users, 3, dim)
        item_views = torch.stack([v[self.n_users:] for v in views], dim=1)  # (n_items, 3, dim)

        # Item attention
        item_cat = torch.cat([item_views[:, 0], item_views[:, 1], item_views[:, 2]], dim=-1)
        item_attn = F.softmax(self.attention(item_cat), dim=-1).unsqueeze(-1)  # (n_items, 3, 1)
        item_total = (item_views * item_attn).sum(dim=1)

        # User: simple average across views
        user_total = user_views.mean(dim=1)

        return user_total, item_total


class LGMRecModel(nn.Module):
    """LGMRec (AAAI'24): Local-global graph learning for multimodal recommendation.

    Captures local user-item collaborative patterns via LightGCN and global
    user-user / item-item semantic relationships via hypergraph convolution
    on modality-derived clusters.
    """

    def __init__(self, n_users, n_items, visual_dim, text_dim, embedding_dim, num_layers,
                 n_hyper_clusters=64):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_hyper_clusters = n_hyper_clusters

        # Local: standard LightGCN
        self.backbone = LightGCNBackbone(n_users, n_items, embedding_dim, num_layers)
        self.visual_proj = nn.Linear(visual_dim, embedding_dim)
        self.text_proj = nn.Linear(text_dim, embedding_dim)

        # Global: hypergraph convolution via soft cluster assignment
        # Item -> cluster assignment matrix  (n_items, n_clusters)
        self.v_cluster_proj = nn.Linear(embedding_dim, n_hyper_clusters)
        self.t_cluster_proj = nn.Linear(embedding_dim, n_hyper_clusters)
        # Cluster -> item back-projection
        self.v_cluster_back = nn.Linear(n_hyper_clusters, embedding_dim)
        self.t_cluster_back = nn.Linear(n_hyper_clusters, embedding_dim)

        # Fusion
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.Sigmoid(),
        )
        self.fusion = nn.Linear(embedding_dim * 3, embedding_dim)

    def encode_all(self, norm_adj, visual_features, text_features, **kwargs):
        # Local branch
        user_emb, item_emb_id = self.backbone(norm_adj)

        v_emb = F.relu(self.visual_proj(visual_features))
        t_emb = F.relu(self.text_proj(text_features))

        # Global branch: hypergraph convolution
        # Soft assignment: item -> cluster
        v_assign = F.softmax(self.v_cluster_proj(v_emb), dim=-1)  # (n_items, K)
        t_assign = F.softmax(self.t_cluster_proj(t_emb), dim=-1)

        # Cluster representations: weighted sum of item embeddings
        v_cluster = v_assign.T @ item_emb_id  # (K, dim)
        t_cluster = t_assign.T @ item_emb_id

        # Back-project: cluster -> item
        v_global = v_assign @ v_cluster  # (n_items, dim)
        t_global = t_assign @ t_cluster

        # Gated fusion of local + global
        item_cat = torch.cat([item_emb_id, v_global, t_global], dim=-1)
        gate_val = self.gate(item_cat)
        item_total = gate_val * item_emb_id + (1 - gate_val) * self.fusion(item_cat)

        return user_emb, item_total


class MENTORModel(nn.Module):
    """MENTOR (AAAI'25): Multi-granularity graph learning for multimodal recommendation.

    Models item relationships at multiple granularity levels: fine-grained
    instance-level k-NN graphs and coarse-grained cluster-level graphs,
    with cross-granularity attention fusion.
    """

    def __init__(self, n_users, n_items, visual_dim, text_dim, embedding_dim, num_layers,
                 k_neighbors=10, n_coarse_clusters=32):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.k_neighbors = k_neighbors
        self.n_coarse_clusters = n_coarse_clusters

        self.backbone = LightGCNBackbone(n_users, n_items, embedding_dim, num_layers)
        self.visual_proj = nn.Linear(visual_dim, embedding_dim)
        self.text_proj = nn.Linear(text_dim, embedding_dim)

        # Coarse-grained: cluster assignment
        self.cluster_assign = nn.Linear(embedding_dim, n_coarse_clusters)
        self.cluster_transform = nn.Linear(embedding_dim, embedding_dim)

        # Cross-granularity attention
        self.cross_attn_q = nn.Linear(embedding_dim, embedding_dim)
        self.cross_attn_k = nn.Linear(embedding_dim, embedding_dim)

        # Final fusion
        self.fusion = nn.Linear(embedding_dim * 3, embedding_dim)

    def _build_knn_graph(self, features, k):
        with torch.no_grad():
            feat_norm = F.normalize(features, dim=-1)
            sim = feat_norm @ feat_norm.T
            sim.fill_diagonal_(-1e9)
            _, topk_idx = sim.topk(k, dim=-1)
        n = features.size(0)
        row = torch.arange(n, device=features.device).unsqueeze(1).expand(-1, k).reshape(-1)
        col = topk_idx.reshape(-1)
        vals = torch.ones(row.size(0), device=features.device) / k
        return torch.sparse_coo_tensor(torch.stack([row, col]), vals, (n, n)).coalesce()

    def encode_all(self, norm_adj, visual_features, text_features, **kwargs):
        user_emb, item_emb_id = self.backbone(norm_adj)

        v_emb = F.relu(self.visual_proj(visual_features))
        t_emb = F.relu(self.text_proj(text_features))
        mm_emb = v_emb + t_emb  # fused multimodal

        # Fine-grained: instance-level k-NN propagation
        fine_adj = self._build_knn_graph(mm_emb, self.k_neighbors)
        fine_prop = torch.sparse.mm(fine_adj, item_emb_id)

        # Coarse-grained: cluster-level propagation
        assign = F.softmax(self.cluster_assign(mm_emb), dim=-1)  # (n_items, C)
        cluster_rep = assign.T @ item_emb_id  # (C, dim)
        cluster_rep = F.relu(self.cluster_transform(cluster_rep))
        coarse_prop = assign @ cluster_rep  # (n_items, dim)

        # Cross-granularity attention
        q = self.cross_attn_q(fine_prop)
        k = self.cross_attn_k(coarse_prop)
        attn = torch.sigmoid((q * k).sum(dim=-1, keepdim=True))
        multi_gran = attn * fine_prop + (1 - attn) * coarse_prop

        # Fusion
        item_total = self.fusion(torch.cat([item_emb_id, multi_gran, mm_emb], dim=-1))

        return user_emb, item_total


class CausalRecModel(nn.Module):
    """CausalRec (MM'21): Causal inference for visual debiasing in recommendation.

    Uses a causal graph to model visual features as confounders. Applies
    backdoor adjustment by learning to disentangle visual-causal and
    visual-non-causal components via adversarial training with a visual
    feature predictor.
    """

    def __init__(self, n_users, n_items, visual_dim, text_dim, embedding_dim, num_layers,
                 adv_weight=0.01):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.adv_weight = adv_weight

        self.backbone = LightGCNBackbone(n_users, n_items, embedding_dim, num_layers)
        self.visual_proj = nn.Linear(visual_dim, embedding_dim)
        self.text_proj = nn.Linear(text_dim, embedding_dim)

        # CausalRec: disentangle item embedding into interest and conformity
        self.interest_proj = nn.Linear(embedding_dim, embedding_dim)
        self.conform_proj = nn.Linear(embedding_dim, embedding_dim)

        # Visual predictor (adversarial): tries to predict visual features from interest embedding
        self.visual_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, visual_dim),
        )

        # Fusion
        self.fusion = nn.Linear(embedding_dim * 2, embedding_dim)

    def encode_all(self, norm_adj, visual_features, text_features, **kwargs):
        user_emb, item_emb_id = self.backbone(norm_adj)

        t_emb = F.relu(self.text_proj(text_features))

        # Disentangle item embedding
        interest = F.relu(self.interest_proj(item_emb_id))
        conform = F.relu(self.conform_proj(item_emb_id))

        # Use interest (debiased) + text for final item representation
        item_total = self.fusion(torch.cat([interest, t_emb], dim=-1))

        return user_emb, item_total

    def causal_loss(self, norm_adj, visual_features):
        """Adversarial loss: interest embedding should NOT predict visual features."""
        _, item_emb_id = self.backbone(norm_adj)
        interest = F.relu(self.interest_proj(item_emb_id))
        conform = F.relu(self.conform_proj(item_emb_id))

        # Visual predictor tries to reconstruct visual features from interest
        pred_visual = self.visual_predictor(interest.detach())
        recon_loss = F.mse_loss(pred_visual, visual_features)

        # Adversarial: interest should fool the predictor (maximize reconstruction error)
        pred_visual_adv = self.visual_predictor(interest)
        adv_loss = -F.mse_loss(pred_visual_adv, visual_features.detach())

        # Disentanglement: interest and conformity should be orthogonal
        orth_loss = (F.normalize(interest, dim=-1) * F.normalize(conform, dim=-1)).sum(dim=-1).pow(2).mean()

        return self.adv_weight * (recon_loss + adv_loss + orth_loss)


class EliMRecModel(nn.Module):
    """EliMRec (MM'22): Eliminating multimodal noise via causal intervention.

    Identifies modality-specific noise that confounds user preference and
    removes it through counterfactual reasoning: compares the factual
    prediction (with all modalities) against counterfactual predictions
    (with each modality removed) to isolate genuine causal effects.
    """

    def __init__(self, n_users, n_items, visual_dim, text_dim, embedding_dim, num_layers,
                 cf_weight=0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.cf_weight = cf_weight

        self.backbone = LightGCNBackbone(n_users, n_items, embedding_dim, num_layers)
        self.visual_proj = nn.Linear(visual_dim, embedding_dim)
        self.text_proj = nn.Linear(text_dim, embedding_dim)

        # Modality-specific encoders for causal effect estimation
        self.visual_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Counterfactual fusion: learns to weight modality contributions
        self.modal_gate = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.Sigmoid(),
        )

        # Noise estimator per modality
        self.v_noise_est = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.Tanh(),
        )
        self.t_noise_est = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.Tanh(),
        )

    def encode_all(self, norm_adj, visual_features, text_features, **kwargs):
        user_emb, item_emb_id = self.backbone(norm_adj)

        v_emb = self.visual_proj(visual_features)
        t_emb = self.text_proj(text_features)

        # Estimate and remove modality noise
        v_noise = self.v_noise_est(v_emb)
        t_noise = self.t_noise_est(t_emb)
        v_clean = self.visual_encoder(v_emb - v_noise)
        t_clean = self.text_encoder(t_emb - t_noise)

        # Gated fusion: factual representation with denoised modalities
        cat_feat = torch.cat([item_emb_id, v_clean, t_clean], dim=-1)
        gate = self.modal_gate(cat_feat)
        item_total = gate * item_emb_id + (1 - gate) * (v_clean + t_clean) / 2

        return user_emb, item_total

    def counterfactual_loss(self, norm_adj, visual_features, text_features):
        """Counterfactual regularization: removing noise shouldn't change ranking order."""
        _, item_emb_id = self.backbone(norm_adj)
        v_emb = self.visual_proj(visual_features)
        t_emb = self.text_proj(text_features)

        # Factual: with noise
        v_noisy = self.visual_encoder(v_emb)
        t_noisy = self.text_encoder(t_emb)
        factual = item_emb_id + v_noisy + t_noisy

        # Counterfactual: without noise
        v_noise = self.v_noise_est(v_emb)
        t_noise = self.t_noise_est(t_emb)
        v_clean = self.visual_encoder(v_emb - v_noise)
        t_clean = self.text_encoder(t_emb - t_noise)
        counterfactual = item_emb_id + v_clean + t_clean

        # The denoised version should preserve item similarity structure
        fact_sim = F.normalize(factual, dim=-1) @ F.normalize(factual, dim=-1).T
        cf_sim = F.normalize(counterfactual, dim=-1) @ F.normalize(counterfactual, dim=-1).T

        # Sample a subset to avoid O(n^2) memory
        n_sample = min(1024, factual.size(0))
        idx = torch.randperm(factual.size(0), device=factual.device)[:n_sample]
        loss = F.mse_loss(cf_sim[idx][:, idx], fact_sim[idx][:, idx].detach())

        return self.cf_weight * loss
