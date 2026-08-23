"""
Generates fake review + meta parquet files mimicking the real Amazon
Reviews 2023 schema, then runs the full Phase 1 pipeline against them
to catch bugs BEFORE spending Kaggle compute on the real dataset.
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(0)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def make_fake_domain(domain: str, n_users=500, n_items=300, n_interactions=4000, shared_users=None):
    user_ids = [f"user_{i}" for i in range(n_users)]
    if shared_users:
        user_ids = shared_users + user_ids[len(shared_users):]
    item_ids = [f"{domain}_item_{i}" for i in range(n_items)]

    rows = []
    for _ in range(n_interactions):
        u = np.random.choice(user_ids)
        it = np.random.choice(item_ids)
        rating = np.random.choice([1,2,3,4,5], p=[0.05,0.05,0.1,0.3,0.5])
        ts = np.random.randint(1_000_000_000, 1_700_000_000)
        rows.append((u, it, rating, ts))
    df_reviews = pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"])

    meta_rows = []
    for it in item_ids:
        meta_rows.append((it, f"Fake Title for {it}", f"A thrilling {domain} synopsis about adventure and survival."))
    df_meta = pd.DataFrame(meta_rows, columns=["item_id", "item_title", "item_synopsis"])

    df_reviews.to_parquet(RAW_DIR / f"{domain}_reviews.parquet")
    df_meta.to_parquet(RAW_DIR / f"{domain}_meta.parquet")
    return user_ids

# create overlapping users between book and movie domains (simulate real overlap)
shared = [f"shared_user_{i}" for i in range(50)]
make_fake_domain("book", shared_users=shared)
make_fake_domain("movie", shared_users=shared)

print("Fake data generated.")

# ---- now run the real pipeline against it ----
import yaml
cfg = {
    "data": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "target_n_users": 300,  # force subsampling to trigger in this test
        "min_user_interactions": 3,
        "min_item_interactions": 3,
        "negative_samples_eval": 10,  # smaller for fast test
    }
}

from data_prep import (
    load_raw_reviews, to_implicit_feedback, process_domain,
    find_overlapping_users,
)

# ---- mimic main()'s two-pass flow, including the critical fix: cap the
# overlap set ONCE with a single shared random draw before it's used by
# either domain's subsampling ----
book_raw = to_implicit_feedback(load_raw_reviews("book", Path("data/raw")))
movie_raw = to_implicit_feedback(load_raw_reviews("movie", Path("data/raw")))

true_overlap = find_overlapping_users(book_raw, movie_raw)
print(f"\nTrue overlap before subsampling: {len(true_overlap)} users")

target_n_users = cfg["data"]["target_n_users"]
rng = np.random.default_rng(42)
if len(true_overlap) > target_n_users:
    capped_overlap = set(rng.choice(list(true_overlap), size=target_n_users, replace=False))
    print(f"Capped overlap to {target_n_users} (single shared draw)")
else:
    capped_overlap = true_overlap

book_data = process_domain("book", cfg, book_raw, forced_users=capped_overlap)
movie_data = process_domain("movie", cfg, movie_raw, forced_users=capped_overlap)

surviving_overlap = find_overlapping_users(book_data["full"], movie_data["full"])
print(f"Surviving overlap after subsampling to {target_n_users} users: "
      f"{len(surviving_overlap)} users")

# with the fix, surviving overlap should closely match capped_overlap
# (small differences only from k-core filtering possibly dropping some
# low-activity forced users afterward)
retention_rate = len(surviving_overlap) / len(capped_overlap) if capped_overlap else 0
print(f"Overlap retention rate (vs capped target): {retention_rate:.1%}")
assert retention_rate > 0.8, "Forced-user subsampling isn't preserving overlap as intended!"

# sanity checks
assert len(book_data["train"]) > 0, "train split is empty!"
assert len(book_data["test"]) > 0, "test split is empty!"
assert "negative_items" in book_data["test"].columns, "negative sampling didn't run!"
assert len(book_data["test"].iloc[0]["negative_items"]) == 10, "wrong number of negatives!"

# check no leakage: negatives shouldn't include the positive item itself
for _, row in book_data["test"].iterrows():
    assert row["item_id"] not in row["negative_items"], "LEAKAGE: positive item found in negatives!"

print("\nAll sanity checks passed.")
print(f"Book train/valid/test sizes: {len(book_data['train'])}/{len(book_data['valid'])}/{len(book_data['test'])}")
