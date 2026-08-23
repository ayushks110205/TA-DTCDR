# ============================================================
# CELL 1 — Install dependencies
# ============================================================
!pip install -q -U huggingface_hub pandas pyarrow

# ============================================================
# CELL 2 — Download PRE-BUILT 5-core interaction files
# ============================================================
# Why this approach instead of raw review jsonl:
#   - McAuley Lab's `datasets` loading script is broken with current
#     `datasets` library versions (script-based datasets deprecated).
#   - The raw review files are also huge (Books.jsonl alone = 20.1GB)
#     and contain full review text we don't need.
#   - McAuley Lab separately hosts pre-filtered "5-core" files
#     (users/items with >=5 interactions, already deduplicated) as
#     small direct-download .csv.gz files. This IS a form of k-core
#     filtering already done for us — see amazon-reviews-2023.github.io
#     /data_processing/5core.html for the full stats table.
#
# NOTE: 5-core scale is still much bigger than the paper's Table 1
# numbers (e.g. Books: 776K users vs paper's ~17K) — we'll apply
# ADDITIONAL filtering on top in data_prep.py to bring it down to a
# comparable, tractable scale for training on Kaggle GPUs.

import os
import urllib.request

os.makedirs("/kaggle/working/raw", exist_ok=True)

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only"

DOMAIN_CATEGORY_MAP = {
    "book": "Books",
    "movie": "Movies_and_TV",
    "music": "CDs_and_Vinyl",
}

def download_5core_ratings(domain_key: str, category: str):
    url = f"{BASE_URL}/{category}.csv.gz"
    out_path = f"/kaggle/working/raw/{domain_key}_ratings_5core.csv.gz"
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, out_path)
    print(f"Saved to {out_path}")
    return out_path

# Sanity check with book first
book_ratings_path = download_5core_ratings("book", DOMAIN_CATEGORY_MAP["book"])

import pandas as pd
df_check = pd.read_csv(book_ratings_path)
print(df_check.shape)
print(df_check.head())

# ============================================================
# CELL 3 — Download the rest once book domain looks right
# ============================================================
for domain_key, category in [("movie", "Movies_and_TV"), ("music", "CDs_and_Vinyl")]:
    download_5core_ratings(domain_key, category)

# ============================================================
# CELL 4 — Metadata: download + stream-filter to items we actually need
# ============================================================
# meta_{Category}.jsonl covers the ENTIRE Amazon catalog for that
# category (millions of items), most of which aren't in our 5-core
# interaction set. We stream line-by-line and only keep matches,
# to avoid holding the whole thing in memory as a DataFrame.

from huggingface_hub import hf_hub_download
import json

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

def fetch_and_filter_meta(domain_key: str, category: str, needed_item_ids: set):
    print(f"Downloading metadata for {category} (this file covers the full catalog, may take a while)...")
    meta_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"raw/meta_categories/meta_{category}.jsonl",
        local_dir="/kaggle/working/hf_cache",
    )

    records = []
    with open(meta_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("parent_asin") in needed_item_ids:
                records.append({
                    "item_id": obj.get("parent_asin"),
                    "item_title": obj.get("title", ""),
                    "item_synopsis": " ".join(obj.get("description", []) or []),
                })

    df_meta = pd.DataFrame(records)
    df_meta.to_parquet(f"/kaggle/working/raw/{domain_key}_meta.parquet")
    print(f"{domain_key}: kept {len(df_meta)} / needed {len(needed_item_ids)} items with metadata")
    return df_meta

# Example for book domain:
book_ids_needed = set(df_check["parent_asin"].unique())
book_meta = fetch_and_filter_meta("book", DOMAIN_CATEGORY_MAP["book"], book_ids_needed)

# Repeat for movie and music once book works:
# movie_df = pd.read_csv("/kaggle/working/raw/movie_ratings_5core.csv.gz")
# movie_meta = fetch_and_filter_meta("movie", "Movies_and_TV", set(movie_df["parent_asin"].unique()))
# music_df = pd.read_csv("/kaggle/working/raw/music_ratings_5core.csv.gz")
# music_meta = fetch_and_filter_meta("music", "CDs_and_Vinyl", set(music_df["parent_asin"].unique()))

# ============================================================
# CELL 5 — Rename ratings files to match data_prep.py's expected schema
# and save as parquet (data_prep.py reads {domain}_reviews.parquet)
# ============================================================
for domain_key in ["book", "movie", "music"]:
    csv_path = f"/kaggle/working/raw/{domain_key}_ratings_5core.csv.gz"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"parent_asin": "parent_asin"})  # already correct name
        df.to_parquet(f"/kaggle/working/raw/{domain_key}_reviews.parquet")
        print(f"{domain_key}: {len(df)} interactions saved as parquet")

# ============================================================
# CELL 6 — (Optional) clean up cache to free disk
# ============================================================
# import shutil
# shutil.rmtree("/kaggle/working/hf_cache", ignore_errors=True)
