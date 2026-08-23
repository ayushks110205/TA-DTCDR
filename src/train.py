"""
Phase 6 — Training Loop
==========================
Paper reference: Section 5.1.4 (Implementation Details).

Adam optimizer, lr=0.001, batch_size=1024, 50 epochs, negative sample
ratio=1. Evaluate on validation set each epoch (or every N), checkpoint
best model by validation NDCG@10, early stop with patience.
"""

import argparse
from pathlib import Path

import torch
import yaml
from torch.optim import Adam

# from graph_build import build_domain_graph, align_shared_nodes
# from model import TADTCDR
# from losses import bce_loss, l2_reg_loss, combine_losses
# from evaluate import evaluate_model


def train_one_epoch(model, optimizer, train_loader, graph_a, graph_b,
                     shared_node_info, cfg) -> dict:
    """One epoch: for each batch, forward pass -> compute all 4 losses ->
    backward -> step. Return dict of average losses for logging.
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_pair", nargs=2, required=True,
                         help="e.g. --domain_pair book movie")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    domain_a, domain_b = args.domain_pair
    print(f"Training TA-DTCDR on domain pair: {domain_a} <-> {domain_b}")

    # TODO:
    # 1. Load processed data (Phase 1 output) for both domains
    # 2. Load filtered tags + compressed tag embeddings (Phase 2/3 output)
    # 3. Build graphs (Phase 4) for both domains + shared node alignment
    # 4. Instantiate TADTCDR model, copy tag embeddings into model.tag_emb
    # 5. Training loop: for epoch in range(cfg['train']['epochs']):
    #      train_one_epoch(...)
    #      if epoch % eval_every == 0: evaluate on validation set
    #      checkpoint if best NDCG@10 so far, else patience -= 1
    #      if patience == 0: break
    # 6. Load best checkpoint, run final evaluation on test set
    # 7. Save results to results/{domain_a}_{domain_b}_results.json


if __name__ == "__main__":
    main()
