"""
Phase 3 — Tag Embedding (fastText + AutoEncoder)
====================================================
Paper reference: Section 4.1, Equation 1 (e_ti = AutoEncoder(f(t_i))).

Responsibilities:
1. Load pretrained fastText word vectors (300-dim).
2. Map every filtered tag -> its 300-dim fastText vector.
3. Train a small AutoEncoder (300 -> 256 -> 128 -> 256 -> 300) with MSE
   reconstruction loss to compress tags into the 128-dim space used by
   the GCN (Section 5.1.4: encoder dims 300 -> 256 -> 128).
4. Save the 128-dim compressed embeddings as initial tag node features.

Output: data/tags/tag_embeddings_128d.pt   (tag_str -> tensor[128])
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml


class TagAutoEncoder(nn.Module):
    """Encoder: 300 -> 256 -> 128. Decoder: 128 -> 256 -> 300.
    Trained with MSE reconstruction loss on fastText vectors only
    (unsupervised — no labels needed).
    """

    def __init__(self, dims=(300, 256, 128)):
        super().__init__()
        d0, d1, d2 = dims
        self.encoder = nn.Sequential(
            nn.Linear(d0, d1), nn.ReLU(),
            nn.Linear(d1, d2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d2, d1), nn.ReLU(),
            nn.Linear(d1, d0),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon


def load_fasttext_vectors(tags: list, fasttext_model_path: str) -> dict:
    """Load pretrained fastText model, return {tag: 300-dim np.array}.
    fasttext-wheel or gensim both work; fastText handles OOV via subwords,
    which matters since some distilled tags may be unusual words.
    """
    raise NotImplementedError


def train_autoencoder(vectors: dict, dims=(300, 256, 128), epochs=200, lr=1e-3) -> TagAutoEncoder:
    """Train the AutoEncoder on all tag vectors (unsupervised).
    Return trained model.
    """
    raise NotImplementedError


def compress_tags(model: TagAutoEncoder, vectors: dict) -> dict:
    """Run trained encoder on all tag vectors, return {tag: 128-dim tensor}."""
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # TODO: load filtered shared tag vocab (from tag_distill.py output),
    #       load fastText vectors, train autoencoder, save compressed embeddings


if __name__ == "__main__":
    main()
