"""
Phase 7 — Evaluation
=======================
Paper reference: Section 5.1.2 (Evaluation Metrics).

Leave-one-out ranking evaluation: for each user, rank their 1 true positive
test item against 99 sampled negatives. Compute:
  - HR@10  : is the positive item in the top-10 of the ranked list?
  - NDCG@10: same, but weighted by rank position (log-discounted).
"""

import math

import torch


def hit_rate_at_k(ranked_item_ids: list, positive_item_id, k: int = 10) -> float:
    """Return 1.0 if positive_item_id is within the top-k of ranked_item_ids,
    else 0.0. ranked_item_ids should already be sorted by predicted score,
    descending.
    """
    raise NotImplementedError


def ndcg_at_k(ranked_item_ids: list, positive_item_id, k: int = 10) -> float:
    """NDCG@k for a single positive item (ideal DCG = 1, since there's only
    one relevant item). If positive is at rank position `pos` (1-indexed)
    within top-k: NDCG = 1 / log2(pos + 1). Else 0.
    """
    raise NotImplementedError


def evaluate_model(model, eval_loader, graph_a, graph_b, shared_node_info,
                    domain: str, k: int = 10) -> dict:
    """Run the model over all evaluation users, compute average HR@k and
    NDCG@k for the given domain ('a' or 'b'). Returns {'HR@10': ..., 'NDCG@10': ...}.
    """
    raise NotImplementedError
