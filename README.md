# TA-DTCDR Reimplementation

Reimplementation of **Tag-Augmented Dual-Target Cross-Domain Recommendation**
(Pan et al., RecSys '25) for [your course/professor name] project.

Paper reference: https://doi.org/10.1145/3705328.3748067

## Domains
Books, Movies (Movies_and_TV), Music (CDs_and_Vinyl) — from Amazon Reviews 2023
(McAuley Lab, HuggingFace: `McAuley-Lab/Amazon-Reviews-2023`).

## Project status

- [ ] Phase 1 — Data preparation
- [ ] Phase 2 — LLM tag distillation
- [ ] Phase 3 — Tag embedding (fastText + AutoEncoder)
- [ ] Phase 4 — Graph construction
- [ ] Phase 5 — Model (heterogeneous GCN + TSA + TCL)
- [ ] Phase 6 — Training loop
- [ ] Phase 7 — Evaluation (HR@10, NDCG@10)
- [ ] Phase 8 — Ablations (tag distillation impact, TSA/TCL impact, layer-count sweep, tag threshold sweep)

## Folder structure

```
ta-dtcdr/
├── data/
│   ├── raw/            # downloaded Amazon jsonl files (not committed)
│   ├── processed/       # filtered train/test/valid splits per domain pair
│   └── tags/            # LLM-generated tags + filtered tag vocab per domain
├── src/
│   ├── data_prep.py     # Phase 1
│   ├── tag_distill.py   # Phase 2
│   ├── tag_embed.py     # Phase 3
│   ├── graph_build.py   # Phase 4
│   ├── model.py         # Phase 5
│   ├── losses.py         # Phase 5 (loss functions)
│   ├── train.py          # Phase 6
│   └── evaluate.py       # Phase 7
├── configs/
│   └── config.yaml       # hyperparameters (Section 5.1.4 of paper)
├── notebooks/             # exploration + ablation plots (Phase 8)
├── checkpoints/           # saved model weights
└── results/               # metrics, logs, result tables
```

## How to run (fill in as you build)

```bash
# Phase 1
python src/data_prep.py --domains book movie --config configs/config.yaml

# Phase 2
python src/tag_distill.py --domain book --config configs/config.yaml

# Phase 3
python src/tag_embed.py --config configs/config.yaml

# Phase 6 (training, runs phases 4-7 internally per Section 5.1.4 setup)
python src/train.py --domain_pair book movie --config configs/config.yaml
```

## Key paper values to reproduce (Section 5.1.4)

| Hyperparameter | Value |
|---|---|
| Embedding size | 128 (300 for raw tag vectors) |
| GCN layers (L) | 2 |
| Dropout | 0.3 |
| lambda_TSA | 1 |
| lambda_TCL | 0.001 |
| lambda_Reg | 0.001 |
| Optimizer | Adam, lr=0.001 |
| Batch size | 1024 |
| Epochs | 50 |
| Negative sample ratio | 1 |
| Tag frequency threshold | 80 |
