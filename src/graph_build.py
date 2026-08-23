"""
Phase 4 — Heterogeneous Graph Construction
=============================================
Paper reference: Section 4.2 (intro), Section 3 (Preliminary).

Responsibilities:
1. Build per-domain heterogeneous graphs with 3 node types: user, item, tag.
   Edges: user-item (from interactions), item-tag (from Phase 2 tags).
2. Identify the "shared" nodes across a domain pair:
   - overlapping users (same user_id in both domains)
   - shared tags (same tag string, survived Phase 2 filtering)
3. Represent adjacency efficiently — recommend torch_geometric's
   HeteroData or a custom sparse adjacency dict, since the paper's
   aggregation (Eq. 2) is a custom weighted-neighbor-average, not a
   standard GCNConv.

Output: an in-memory graph object (or pickled) consumed directly by
        src/model.py during training — no need to persist to disk unless
        graph construction becomes a bottleneck.
"""

from dataclasses import dataclass
from typing import Dict, Set

import torch


@dataclass
class DomainGraph:
    """Holds node id mappings and edge lists for a single domain.

    user_ids, item_ids, tag_ids: contiguous integer id spaces (0..n-1)
    user_item_edges: LongTensor [2, num_edges] (interactions)
    item_tag_edges:  LongTensor [2, num_edges]
    """

    num_users: int
    num_items: int
    num_tags: int
    user_item_edges: torch.Tensor
    item_tag_edges: torch.Tensor
    user_id_map: Dict[str, int]
    item_id_map: Dict[str, int]
    tag_id_map: Dict[str, int]


def build_domain_graph(interactions_df, item_tags: dict, tag_vocab: dict) -> DomainGraph:
    """Construct a single domain's graph from processed interactions
    (Phase 1 output) and item->tags mapping (Phase 2 output).
    """
    raise NotImplementedError


def compute_neighbor_counts(graph: DomainGraph) -> dict:
    """Precompute |N_x| for every node — needed for the normalization
    term N(x,y) = 1/sqrt(|N_x||N_y|) in Equation 2.
    """
    raise NotImplementedError


def align_shared_nodes(graph_a: DomainGraph, graph_b: DomainGraph,
                        overlapping_users: Set[str]) -> dict:
    """Build index mappings so that a shared user/tag's row in graph_a's
    embedding table and graph_b's embedding table can be looked up
    consistently — needed for Equation 3's cross-domain transfer step.

    Returns dict with keys like 'shared_user_idx_a', 'shared_user_idx_b',
    'shared_tag_idx_a', 'shared_tag_idx_b' (aligned LongTensors, same order).
    """
    raise NotImplementedError
