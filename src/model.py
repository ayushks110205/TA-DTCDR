"""
Phase 5 — TA-DTCDR Model
===========================
Paper reference: Section 4.2 (Cross-domain Graph Convolution, Eq. 2-3),
                  Section 4.3 (Tag Semantic Alignment, Eq. 6-9),
                  Section 4.4 (Contrastive Learning for Tags, Eq. 10-11).

This is the core module. Structure mirrors the paper's own breakdown:
  - CrossDomainGCN: the heterogeneous graph convolution + cross-domain
    transfer (this is the backbone, extends BiTGCF-style propagation).
  - TagSemanticAlignment: consistency + ordering losses using tag context.
  - TagContrastiveLearning: keeps tag embeddings distinct (anti-over-smoothing).
  - TADTCDR: top-level module wiring all three together + final scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossDomainGCN(nn.Module):
    """Implements Equations 2 and 3.

    Eq. 2 (per-domain aggregation):
        a_u^(l) = e_u^(l) + N(u,v) * Aggr(u,v)
        a_t^(l) = e_t^(l) + N(t,v) * Aggr(t,v)
        a_v^(l) = e_v^(l) + N(u,v,t) * [Aggr(v,u) + Aggr(v,t)]
      where Aggr(x,y) = sum over y in N(x) of (e_y + e_y (elementwise*) e_x)
            N(x,y) = 1 / sqrt(|N_x| |N_y|)

    Eq. 3 (cross-domain transfer, applied to shared users/tags only):
        c_u = l_uA * a_uA + (1 - l_uA) * a_uB
        e_uA^(l+1) = f(a_uA, c_u) = 0.5 * (a_uA + c_u)     [similarly for uB, t]
      where l_uA = |N_uA| / (|N_uA| + |N_uB|)

    Non-shared users/items just take e^(l+1) = a^(l+1) directly (no transfer).

    After L layers, concatenate all layer outputs:
        e_u = concat(e_u^(0), ..., e_u^(L))
    """

    def __init__(self, embedding_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

    def aggregate(self, self_emb, neighbor_embs, norm_term):
        """Implements Aggr(x,y) + N(x,y) scaling — Eq. 2 building block."""
        raise NotImplementedError

    def transfer(self, emb_a, emb_b, weight_a):
        """Implements Eq. 3's shared-node blending step."""
        raise NotImplementedError

    def forward(self, graph_a, graph_b, shared_node_info, init_embeddings):
        """Run num_layers rounds of aggregate + transfer for both domains.
        Returns final concatenated embeddings for users, items, tags in
        both domains: dict with keys like 'user_a', 'item_a', 'tag_a',
        'user_b', 'item_b', 'tag_b'.
        """
        raise NotImplementedError


class TagSemanticAlignment(nn.Module):
    """Implements Section 4.3, Equations 6-9.

    Splits each embedding into e_f (fused half) and e_s (independent half).
    Concatenates tag context with user/item embeddings (Fig. 4), computes
    context-aware scores S(u,v,t), then combines consistency + ordering
    objectives into L_TSA (Eq. 9). Computed separately per domain, summed.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        # e_full = e_f (+) e_s -- both halves length embedding_dim // 2
        self.half_dim = embedding_dim // 2

    def score(self, u_emb, v_emb, tag_emb):
        """S(u, v, t) — inner product of tag-fused user/item embeddings."""
        raise NotImplementedError

    def forward(self, u, v_pos, v_neg, tags_pos_a, tags_neg_a, tags_pos_b, tags_neg_b):
        """Compute L_TSA per Eq. 9 (consistency + ordering terms), for one domain.
        Caller sums L_TSA^A + L_TSA^B for the total TSA loss.
        """
        raise NotImplementedError


class TagContrastiveLearning(nn.Module):
    """Implements Section 4.4, Equations 10-11.

    Pools user/item embeddings (average), concatenates with each tag
    embedding, passes through MLP to get user-view and item-view tag
    representations (Eq. 10). Contrastive (InfoNCE-style) loss pulls
    same-tag views together, pushes different tags apart (Eq. 11).
    """

    def __init__(self, embedding_dim: int, mlp_dims=(128, 256, 128)):
        super().__init__()
        d0, d1, d2 = mlp_dims
        self.mlp = nn.Sequential(
            nn.Linear(d0, d1), nn.ReLU(),
            nn.Linear(d1, d2),
        )

    def pooled_view(self, tag_embs, pooled_node_emb):
        """Eq. 10: e_{t,pooled} = MLP(e_t + mean(node embeddings))."""
        raise NotImplementedError

    def contrastive_loss(self, view_x, view_y):
        """Eq. 11's L_{x,y}: InfoNCE loss over all tags in vocab."""
        raise NotImplementedError

    def forward(self, tag_embs_a, tag_embs_b, user_embs_a, item_embs_a,
                user_embs_b, item_embs_b):
        """Compute L_TCL = L_{uA,vA} + L_{uB,vB} + L_{uA,uB} + L_{vA,vB} (Eq. 11)."""
        raise NotImplementedError


class TADTCDR(nn.Module):
    """Top-level model wiring GCN backbone + TSA + TCL together.
    See src/train.py for how losses get combined (Eq. 12).
    """

    def __init__(self, num_users_a, num_items_a, num_users_b, num_items_b,
                 num_tags, config: dict):
        super().__init__()
        dim = config["model"]["embedding_size"]

        self.user_emb_a = nn.Embedding(num_users_a, dim)
        self.item_emb_a = nn.Embedding(num_items_a, dim)
        self.user_emb_b = nn.Embedding(num_users_b, dim)
        self.item_emb_b = nn.Embedding(num_items_b, dim)
        # tag embeddings initialized from Phase 3's AutoEncoder output,
        # loaded and copied into this table at training-script level.
        self.tag_emb = nn.Embedding(num_tags, dim)

        self.gcn = CrossDomainGCN(
            embedding_dim=dim,
            num_layers=config["model"]["num_gcn_layers"],
            dropout=config["model"]["dropout"],
        )
        self.tsa = TagSemanticAlignment(embedding_dim=dim)
        self.tcl = TagContrastiveLearning(
            embedding_dim=dim, mlp_dims=config["model"]["tcl_mlp_dims"]
        )

    def predict(self, user_emb, item_emb):
        """Final score = inner product (Section 3, used for BCE + ranking)."""
        return (user_emb * item_emb).sum(dim=-1)

    def forward(self, graph_a, graph_b, shared_node_info, batch):
        """Full forward pass: run GCN, compute BCE-ready scores, and return
        everything needed by src/losses.py to compute all 4 loss terms.
        """
        raise NotImplementedError
