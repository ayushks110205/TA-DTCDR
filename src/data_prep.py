"""
Phase 1 — Data Preparation
===========================
Paper reference: Section 5.1.1 (Datasets and Tasks), Table 1.

Responsibilities:
1. Load raw Amazon Reviews 2023 data for a domain (book / movie / music).
2. Apply k-core filtering (iteratively drop users/items below interaction
   threshold until stable) to control dataset size — paper doesn't give
   exact thresholds, so document whatever you choose in your report.
3. Convert explicit star ratings -> implicit binary feedback (r_uv in {0,1}).
4. Identify overlapping users between each domain pair (needed later for
   Eq. 3 cross-domain transfer).
5. Leave-one-out split: last interaction -> test, second-last -> valid,
   rest -> train (Section 5.1.2).
6. Generate 99 negative samples per user for test/valid evaluation.

Output: data/processed/{domain}_train.csv, _valid.csv, _test.csv
        data/processed/{domain_pair}_overlap_users.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_raw_reviews(domain: str, raw_dir: Path) -> pd.DataFrame:
    """Load ONLY the interaction data for a domain (user_id, item_id,
    rating, timestamp) -- deliberately does NOT merge item_title/
    item_synopsis here.

    Why: merging text metadata onto every interaction row duplicates each
    item's synopsis once per interaction (e.g. a book with 200 reviews
    would carry 200 separate copies of its synopsis in memory). At real
    scale (9M+ interactions) this caused excessive RAM usage for no
    benefit -- none of k_core_filter / subsample_users / split / negative
    sampling need the text at all. Metadata is joined in separately, once
    per surviving item, via load_item_meta() at the end of process_domain().
    """
    reviews_path = raw_dir / f"{domain}_reviews.parquet"
    df = pd.read_csv(reviews_path) if reviews_path.suffix == ".csv" else pd.read_parquet(reviews_path)
    df = df.rename(columns={"parent_asin": "item_id"})
    df = df[["user_id", "item_id", "rating", "timestamp"]]
    return df


def load_item_meta(domain: str, raw_dir: Path) -> pd.DataFrame:
    """Load item metadata (item_id, item_title, item_synopsis) separately
    from interactions. Called once at the end of process_domain() to
    attach titles/synopses only to the final surviving item set, not to
    every raw interaction row.
    """
    meta_path = raw_dir / f"{domain}_meta.parquet"
    return pd.read_parquet(meta_path)


def subsample_users(df: pd.DataFrame, target_n_users: int, forced_users: set = None,
                     seed: int = 42) -> pd.DataFrame:
    """Randomly subsample down to target_n_users, ALWAYS including every
    user in `forced_users` (e.g. users known to overlap with another
    domain), then filling the remaining quota with a random sample of
    everyone else.

    IMPORTANT: `forced_users` must already be capped to <= target_n_users
    by the CALLER using a single, consistent decision shared across every
    domain that needs the same overlap set preserved (see main()'s
    two-pass flow). Deciding which overlap users to keep independently
    per domain would produce different random subsets per domain and
    defeat the entire purpose of forcing them — two separate random draws
    from the same pool don't coincide.
    """
    rng = np.random.default_rng(seed)
    all_users = df["user_id"].unique()

    forced_users = forced_users or set()
    forced_present = [u for u in all_users if u in forced_users]

    if len(forced_present) > target_n_users:
        raise ValueError(
            f"forced_users ({len(forced_present)}) exceeds target_n_users "
            f"({target_n_users}) for this domain — cap forced_users ONCE "
            f"in the caller before calling subsample_users per domain, "
            f"not independently inside each call."
        )

    remaining_pool = [u for u in all_users if u not in forced_users]
    n_remaining_needed = min(target_n_users - len(forced_present), len(remaining_pool))
    sampled_remaining = rng.choice(remaining_pool, size=n_remaining_needed, replace=False)
    selected = np.concatenate([forced_present, sampled_remaining])

    return df[df["user_id"].isin(set(selected))].reset_index(drop=True)


def k_core_filter(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Iteratively remove users/items with fewer interactions than the threshold,
    until the dataset stabilizes (no more removals needed).

    This must be iterative: removing sparse items can drop a user below
    min_user (and vice versa), so a single pass isn't enough.
    """
    prev_len = -1
    while len(df) != prev_len:
        prev_len = len(df)

        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user].index
        df = df[df["user_id"].isin(valid_users)]

        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= min_item].index
        df = df[df["item_id"].isin(valid_items)]

    return df.reset_index(drop=True)


def to_implicit_feedback(df: pd.DataFrame, positive_threshold: int = 4) -> pd.DataFrame:
    """Convert rating column to binary implicit feedback.

    Convention (paper doesn't specify exact cutoff): rating >= positive_threshold
    -> keep as a positive interaction (r_uv = 1). Rows below threshold are DROPPED
    (not set to 0) — standard practice for implicit-feedback recsys, since we only
    observe positives; negatives are sampled separately during training/eval.
    """
    df = df[df["rating"] >= positive_threshold].copy()
    df["label"] = 1
    # De-duplicate: keep only the most recent interaction per (user, item) pair
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["user_id", "item_id"], keep="last"
    )
    return df.reset_index(drop=True)


def find_overlapping_users(df_a: pd.DataFrame, df_b: pd.DataFrame) -> set:
    """Return set of user_ids present in both domain dataframes."""
    return set(df_a["user_id"].unique()) & set(df_b["user_id"].unique())


def leave_one_out_split(df: pd.DataFrame):
    """Per user, sort interactions chronologically:
    - last interaction -> test
    - second-to-last -> valid
    - rest -> train
    Returns (train_df, valid_df, test_df).

    Users with < 3 interactions can't be split this way (need at least
    1 for train, 1 for valid, 1 for test) — such users go entirely to train.
    """
    df = df.sort_values(["user_id", "timestamp"])
    grouped = df.groupby("user_id")

    train_rows, valid_rows, test_rows = [], [], []

    for user_id, group in grouped:
        n = len(group)
        if n < 3:
            train_rows.append(group)
        else:
            train_rows.append(group.iloc[:-2])
            valid_rows.append(group.iloc[[-2]])
            test_rows.append(group.iloc[[-1]])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    valid_df = pd.concat(valid_rows).reset_index(drop=True) if valid_rows else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_rows).reset_index(drop=True) if test_rows else pd.DataFrame(columns=df.columns)

    return train_df, valid_df, test_df


def sample_negatives(eval_df: pd.DataFrame, user_positive_items: dict,
                      all_items: list, n_neg: int = 99, seed: int = 42) -> pd.DataFrame:
    """For each (user, positive_item) row in test/valid, sample n_neg items
    the user has NOT interacted with anywhere in the dataset (train+valid+test
    combined), for ranking evaluation (Section 5.1.2).

    user_positive_items: dict of user_id -> set of ALL item_ids that user has
    ever interacted with (across train/valid/test) — needed so sampled
    negatives are true negatives, not items the user liked but in another split.
    """
    rng = np.random.default_rng(seed)
    all_items_arr = np.array(all_items)

    neg_lists = []
    for _, row in eval_df.iterrows():
        user_id = row["user_id"]
        seen = user_positive_items.get(user_id, set())
        negatives = []
        # rejection sampling; fine at this scale, item pools are thousands not millions
        while len(negatives) < n_neg:
            candidates = rng.choice(all_items_arr, size=n_neg * 2, replace=False)
            for c in candidates:
                if c not in seen and c not in negatives:
                    negatives.append(c)
                if len(negatives) >= n_neg:
                    break
        neg_lists.append(negatives[:n_neg])

    eval_df = eval_df.copy()
    eval_df["negative_items"] = neg_lists
    return eval_df


def process_domain(domain: str, cfg: dict, df: pd.DataFrame, forced_users: set = None) -> dict:
    """Full Phase 1 pipeline for one domain, given an already-loaded and
    implicit-converted INTERACTIONS-ONLY dataframe (from main()'s pass 1,
    via load_raw_reviews -- no title/synopsis text attached): subsample ->
    k-core filter -> split -> negative sample -> save.

    forced_users must already be capped to <= target_n_users by the
    caller (see main()) — see subsample_users' docstring for why.
    """
    processed_dir = Path(cfg["data"]["processed_dir"])
    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    target_n_users = cfg["data"].get("target_n_users")
    if target_n_users and df["user_id"].nunique() > target_n_users:
        df = subsample_users(df, target_n_users, forced_users=forced_users)
        print(f"[{domain}] after subsampling to ~{target_n_users} users: "
              f"{len(df)} interactions, {df['user_id'].nunique()} users")

    df = k_core_filter(
        df,
        min_user=cfg["data"]["min_user_interactions"],
        min_item=cfg["data"]["min_item_interactions"],
    )
    print(f"[{domain}] after k-core filter: {len(df)} interactions, "
          f"{df['user_id'].nunique()} users, {df['item_id'].nunique()} items")

    train_df, valid_df, test_df = leave_one_out_split(df)
    print(f"[{domain}] split -> train: {len(train_df)}, valid: {len(valid_df)}, test: {len(test_df)}")

    # every item a user has EVER touched, across all splits -> used to
    # guarantee sampled negatives are true negatives
    user_positive_items = df.groupby("user_id")["item_id"].apply(set).to_dict()
    all_items = df["item_id"].unique().tolist()

    n_neg = cfg["data"]["negative_samples_eval"]
    valid_df = sample_negatives(valid_df, user_positive_items, all_items, n_neg=n_neg)
    test_df = sample_negatives(test_df, user_positive_items, all_items, n_neg=n_neg)

    train_df.to_parquet(processed_dir / f"{domain}_train.parquet")
    valid_df.to_parquet(processed_dir / f"{domain}_valid.parquet")
    test_df.to_parquet(processed_dir / f"{domain}_test.parquet")

    # Join title/synopsis ONLY against the final surviving item set (a few
    # hundred/thousand items) instead of every raw interaction row --
    # this is the fix for the excessive RAM usage caused by duplicating
    # text once per interaction instead of once per item.
    surviving_item_ids = set(all_items)
    meta_df = load_item_meta(domain, raw_dir)
    item_meta = meta_df[meta_df["item_id"].isin(surviving_item_ids)].drop_duplicates("item_id")
    item_meta.to_parquet(processed_dir / f"{domain}_item_meta.parquet")

    return {"full": df, "train": train_df, "valid": valid_df, "test": test_df}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pass 1: load + implicit-convert every domain WITHOUT subsampling,
    # so overlapping users can be computed on the true full sets first.
    # Subsampling independently per domain (without this) would randomly
    # wipe out most cross-domain overlap by sheer chance, since overlap
    # is already a small % of a huge user base (see 5-core stats).
    prelim = {}
    for domain in args.domains:
        print(f"[{domain}] (pass 1) loading + implicit-converting for overlap computation...")
        df = load_raw_reviews(domain, raw_dir)
        df = to_implicit_feedback(df)
        prelim[domain] = df

    domains_list = list(prelim.keys())
    all_overlaps = {}
    for i in range(len(domains_list)):
        for j in range(i + 1, len(domains_list)):
            d_a, d_b = domains_list[i], domains_list[j]
            overlap = find_overlapping_users(prelim[d_a], prelim[d_b])
            all_overlaps[(d_a, d_b)] = overlap
            print(f"True overlap {d_a}-{d_b} (pre-subsample): {len(overlap)} users")

    # union of all overlap sets a domain participates in -> its forced-keep set.
    # If a domain's target_n_users is smaller than its overlap set, the
    # overlap must be capped ONCE here (shared across every domain in that
    # pair) using a single random draw -- NOT independently inside each
    # domain's subsampling, which would produce different random subsets
    # per domain and destroy the overlap instead of preserving it.
    target_n_users = cfg["data"].get("target_n_users")
    rng = np.random.default_rng(42)

    capped_overlaps = {}
    for (d_a, d_b), overlap in all_overlaps.items():
        if target_n_users and len(overlap) > target_n_users:
            overlap = set(rng.choice(list(overlap), size=target_n_users, replace=False))
            print(f"Capped overlap {d_a}-{d_b} to {target_n_users} (single shared draw)")
        capped_overlaps[(d_a, d_b)] = overlap

    forced_by_domain = {d: set() for d in domains_list}
    for (d_a, d_b), overlap in capped_overlaps.items():
        forced_by_domain[d_a] |= overlap
        forced_by_domain[d_b] |= overlap

    # ---- Pass 2: subsample + k-core filter + split + save, per domain,
    # using the forced-keep sets computed above. Reuse the dataframes
    # already loaded in pass 1 instead of re-reading from disk.
    domain_data = {}
    for domain in args.domains:
        domain_data[domain] = process_domain(
            domain, cfg, prelim[domain], forced_users=forced_by_domain[domain]
        )

    # ---- recompute overlap on the FINAL post-subsample data (this is what
    # actually gets used by graph_build.py later — report both numbers in
    # your write-up: true overlap vs. surviving overlap after shrinking)
    for i in range(len(domains_list)):
        for j in range(i + 1, len(domains_list)):
            d_a, d_b = domains_list[i], domains_list[j]
            final_overlap = find_overlapping_users(domain_data[d_a]["full"], domain_data[d_b]["full"])
            print(f"Surviving overlap {d_a}-{d_b} (post-subsample): {len(final_overlap)} users")
            with open(processed_dir / f"{d_a}_{d_b}_overlap_users.json", "w") as f:
                json.dump(list(final_overlap), f)


if __name__ == "__main__":
    main()
