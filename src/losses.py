"""
Loss Functions
================
Paper reference: Section 4.5, Equation 12 for the combination.
  L = L_BCE + lambda_Reg * L_Reg + lambda_TSA * L_TSA + lambda_TCL * L_TCL

Individual loss definitions:
  - L_BCE  : Equation 4  (main recommendation loss)
  - L_Reg  : Equation 5  (L2 regularization on layer-0 embeddings)
  - L_TSA  : Equation 9  (computed inside model.TagSemanticAlignment)
  - L_TCL  : Equation 11 (computed inside model.TagContrastiveLearning)

This file holds the simple, standalone ones (BCE, Reg); TSA/TCL live as
methods on their respective modules in model.py since they need access
to the model's internal split/fusion logic.
"""

import torch
import torch.nn.functional as F


def bce_loss(pred_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Equation 4. pred_scores are raw inner products -- apply sigmoid
    internally (use F.binary_cross_entropy_with_logits for numerical stability
    instead of manually calling sigmoid then BCE).
    """
    return F.binary_cross_entropy_with_logits(pred_scores, labels.float())


def l2_reg_loss(embeddings_layer0: list) -> torch.Tensor:
    """Equation 5. Sum of squared L2 norms of layer-0 (initial, trainable)
    user/item/tag embeddings involved in the current batch.
    embeddings_layer0: list of tensors, e.g. [e_u0, e_v0, e_t0] for the batch.
    """
    total = 0.0
    for emb in embeddings_layer0:
        total = total + emb.pow(2).sum(dim=-1)
    return total.mean()


def combine_losses(l_bce, l_reg, l_tsa, l_tcl, cfg: dict) -> torch.Tensor:
    """Equation 12."""
    return (
        l_bce
        + cfg["loss"]["lambda_reg"] * l_reg
        + cfg["loss"]["lambda_tsa"] * l_tsa
        + cfg["loss"]["lambda_tcl"] * l_tcl
    )
