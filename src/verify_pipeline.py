"""
verify_pipeline.py -- Post-pipeline data integrity verification for TA-DTCDR.

Loads the parquet / JSON outputs produced by the data pipeline and runs a
comprehensive suite of sanity checks (existence, cold-start, leakage,
negative-sample correctness, item-meta coverage, null checks, overlap
consistency).  Prints a PASS/FAIL per check, a Table-1-style statistics
summary, and exits with code 1 if any check fails.

Usage
-----
    python src/verify_pipeline.py \
        --domains book movie \
        --domain-pairs book_movie \
        --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
class CheckResults:
    """Accumulates PASS / FAIL results and prints a final summary."""

    def __init__(self) -> None:
        self.passed: int = 0
        self.failed: int = 0

    def ok(self, msg: str) -> None:
        print(f"  [PASS] {msg}")
        self.passed += 1

    def fail(self, msg: str, detail: str = "") -> None:
        suffix = f": {detail}" if detail else ""
        print(f"  [FAIL] {msg}{suffix}")
        self.failed += 1

    def summary(self) -> None:
        total = self.passed + self.failed
        status = "ALL PASSED" if self.failed == 0 else "FAILURES DETECTED"
        print()
        print("=" * 64)
        print(f"  Summary: {self.passed}/{total} checks passed, "
              f"{self.failed}/{total} failed  --  {status}")
        print("=" * 64)

    @property
    def exit_code(self) -> int:
        return 0 if self.failed == 0 else 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _safe_read_parquet(path: Path) -> pd.DataFrame | None:
    """Return a DataFrame or None if the file does not exist / is unreadable."""
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _safe_read_json(path: Path) -> list | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-domain checks
# ---------------------------------------------------------------------------

def verify_domain(
    domain: str,
    processed_dir: Path,
    neg_samples_eval: int,
    results: CheckResults,
) -> dict[str, pd.DataFrame]:
    """Run all per-domain checks.  Returns loaded split DataFrames for reuse."""

    print(f"\n{'-' * 40}")
    print(f"  Domain: {domain}")
    print(f"{'-' * 40}")

    splits: dict[str, pd.DataFrame] = {}

    # -- Check 1: existence & non-empty ------------------------------------
    for split_name in ("train", "valid", "test"):
        fpath = processed_dir / f"{domain}_{split_name}.parquet"
        df = _safe_read_parquet(fpath)
        if df is None:
            results.fail(
                f"{domain}_{split_name}.parquet exists and is non-empty",
                f"file not found or unreadable at {fpath}",
            )
            continue
        if df.empty:
            results.fail(
                f"{domain}_{split_name}.parquet exists and is non-empty",
                "file exists but contains 0 rows",
            )
            continue
        results.ok(f"{domain}_{split_name}.parquet exists and is non-empty "
                    f"({len(df):,} rows)")
        splits[split_name] = df

    # If any core split is missing we cannot run the remaining checks.
    if not all(s in splits for s in ("train", "valid", "test")):
        results.fail(
            f"{domain}: skipping remaining checks",
            "one or more split files are missing",
        )
        return splits

    train_df = splits["train"]
    valid_df = splits["valid"]
    test_df = splits["test"]

    # -- Check 2: cold-start -- no user in valid/test absent from train -----
    train_users = set(train_df["user_id"])
    valid_users = set(valid_df["user_id"])
    test_users = set(test_df["user_id"])
    cold_valid = valid_users - train_users
    cold_test = test_users - train_users
    cold = cold_valid | cold_test
    if cold:
        results.fail(
            f"{domain}: no cold-start users in valid/test",
            f"{len(cold)} user(s) appear in valid/test but not train "
            f"(e.g. {list(cold)[:5]})",
        )
    else:
        results.ok(f"{domain}: no cold-start users in valid/test")

    # -- Check 3: split leakage -- no (user_id, item_id) in >1 split -------
    def _pairs(df: pd.DataFrame) -> set:
        # Returns set of (user_id, item_id) tuples
        return set(zip(df["user_id"], df["item_id"]))

    train_pairs = _pairs(train_df)
    valid_pairs = _pairs(valid_df)
    test_pairs = _pairs(test_df)

    leak_tv = train_pairs & valid_pairs
    leak_tt = train_pairs & test_pairs
    leak_vt = valid_pairs & test_pairs
    total_leak = len(leak_tv) + len(leak_tt) + len(leak_vt)
    if total_leak:
        results.fail(
            f"{domain}: no (user, item) pair in multiple splits",
            f"train&valid={len(leak_tv)}, train&test={len(leak_tt)}, "
            f"valid&test={len(leak_vt)}",
        )
    else:
        results.ok(f"{domain}: no (user, item) pair in multiple splits")

    # -- Checks 4 & 5: negative_items correctness -------------------------
    for split_name in ("valid", "test"):
        sdf = splits[split_name]

        # Check that negative_items column exists
        if "negative_items" not in sdf.columns:
            results.fail(
                f"{domain}/{split_name}: negative_items column exists",
                "column not found",
            )
            continue

        # Check 4: positive item not in negative list
        pos_in_neg = sdf.apply(
            lambda r: r["item_id"] in r["negative_items"], axis=1
        )
        n_bad = int(pos_in_neg.sum())
        if n_bad:
            results.fail(
                f"{domain}/{split_name}: positive item_id not in negative_items",
                f"{n_bad} row(s) have the positive item inside negatives",
            )
        else:
            results.ok(
                f"{domain}/{split_name}: positive item_id not in negative_items"
            )

        # Check 5: negative_items length == negative_samples_eval
        lengths = sdf["negative_items"].apply(len)
        bad_len = lengths[lengths != neg_samples_eval]
        if len(bad_len):
            results.fail(
                f"{domain}/{split_name}: negative_items length == "
                f"{neg_samples_eval}",
                f"{len(bad_len)} row(s) have wrong length "
                f"(min={int(lengths.min())}, max={int(lengths.max())})",
            )
        else:
            results.ok(
                f"{domain}/{split_name}: negative_items length == "
                f"{neg_samples_eval}"
            )

    # -- Check 6: item_id subset of item_meta --------------------------------
    meta_path = processed_dir / f"{domain}_item_meta.parquet"
    meta_df = _safe_read_parquet(meta_path)
    if meta_df is None:
        results.fail(
            f"{domain}: item_ids are a subset of item_meta",
            f"item_meta file not found at {meta_path}",
        )
    else:
        meta_items = set(meta_df["item_id"])
        all_items = (
            set(train_df["item_id"])
            | set(valid_df["item_id"])
            | set(test_df["item_id"])
        )
        missing = all_items - meta_items
        if missing:
            results.fail(
                f"{domain}: item_ids are a subset of item_meta",
                f"{len(missing)} item(s) in splits but not in meta "
                f"(e.g. {list(missing)[:5]})",
            )
        else:
            results.ok(f"{domain}: item_ids are a subset of item_meta")

    # -- Check 7: no nulls in user_id, item_id, item_title ----------------
    null_cols = ["user_id", "item_id", "item_title"]
    null_issues: list[str] = []
    for split_name in ("train", "valid", "test"):
        sdf = splits[split_name]
        for col in null_cols:
            if col not in sdf.columns:
                null_issues.append(f"{split_name} missing column '{col}'")
                continue
            n_null = int(sdf[col].isna().sum())
            if n_null:
                null_issues.append(f"{split_name}.{col} has {n_null} null(s)")
    if null_issues:
        results.fail(
            f"{domain}: no nulls in user_id / item_id / item_title",
            "; ".join(null_issues),
        )
    else:
        results.ok(f"{domain}: no nulls in user_id / item_id / item_title")

    return splits


# ---------------------------------------------------------------------------
# Cross-domain checks
# ---------------------------------------------------------------------------

def verify_cross_domain(
    domain_a: str,
    domain_b: str,
    processed_dir: Path,
    all_splits: dict[str, dict[str, pd.DataFrame]],
    results: CheckResults,
) -> None:
    """Run cross-domain overlap checks for a given pair."""

    pair_label = f"{domain_a}-{domain_b}"
    print(f"\n{'-' * 40}")
    print(f"  Cross-domain: {pair_label}")
    print(f"{'-' * 40}")

    # Try both orderings for the overlap file.
    overlap: list | None = None
    tried: list[Path] = []
    for a, b in [(domain_a, domain_b), (domain_b, domain_a)]:
        fpath = processed_dir / f"{a}_{b}_overlap_users.json"
        tried.append(fpath)
        overlap = _safe_read_json(fpath)
        if overlap is not None:
            break

    if overlap is None:
        results.fail(
            f"{pair_label}: overlap JSON exists",
            f"tried {[str(p) for p in tried]}",
        )
        return

    overlap_set = set(str(uid) for uid in overlap)

    # Collect all user_ids from both domains' splits.
    def _domain_users(domain: str) -> set[str]:
        users: set[str] = set()
        for split_name in ("train", "valid", "test"):
            if domain in all_splits and split_name in all_splits[domain]:
                users.update(
                    str(u) for u in all_splits[domain][split_name]["user_id"]
                )
        return users

    users_a = _domain_users(domain_a)
    users_b = _domain_users(domain_b)

    # -- Check 9: every overlap user exists in both domains ----------------
    missing_a = overlap_set - users_a
    missing_b = overlap_set - users_b
    if missing_a or missing_b:
        parts: list[str] = []
        if missing_a:
            parts.append(f"{len(missing_a)} missing from {domain_a}")
        if missing_b:
            parts.append(f"{len(missing_b)} missing from {domain_b}")
        results.fail(
            f"{pair_label}: all overlap users exist in both domains",
            "; ".join(parts),
        )
    else:
        results.ok(
            f"{pair_label}: all overlap users exist in both domains "
            f"({len(overlap_set):,} users)"
        )

    # -- Info 10: overlap percentage ---------------------------------------
    pct_a = (len(overlap_set & users_a) / len(users_a) * 100) if users_a else 0
    pct_b = (len(overlap_set & users_b) / len(users_b) * 100) if users_b else 0
    print(f"  [INFO] Overlap: {len(overlap_set):,} users  "
          f"({pct_a:.1f}% of {domain_a}, {pct_b:.1f}% of {domain_b})")


# ---------------------------------------------------------------------------
# Table 1 -- dataset statistics
# ---------------------------------------------------------------------------

def print_table1(
    domains: list[str],
    all_splits: dict[str, dict[str, pd.DataFrame]],
) -> None:
    """Print a neatly formatted Table-1-style statistics summary."""

    print(f"\n{'=' * 64}")
    print("  Table 1 -- Dataset Statistics After Processing")
    print(f"{'=' * 64}")

    header = (
        f"  {'Domain':<10} {'#Users':>10} {'#Items':>10} "
        f"{'#Inter.':>10} {'Density':>12}"
    )
    print(header)
    print(f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 12}")

    for domain in domains:
        if domain not in all_splits:
            print(f"  {domain:<10} {'(data unavailable)':>44}")
            continue

        frames = [
            all_splits[domain][s]
            for s in ("train", "valid", "test")
            if s in all_splits[domain]
        ]
        if not frames:
            print(f"  {domain:<10} {'(data unavailable)':>44}")
            continue

        combined = pd.concat(frames, ignore_index=True)
        n_users = combined["user_id"].nunique()
        n_items = combined["item_id"].nunique()
        n_interactions = len(combined)
        density = (
            n_interactions / (n_users * n_items) * 100
            if n_users and n_items
            else 0.0
        )

        print(
            f"  {domain:<10} {n_users:>10,} {n_items:>10,} "
            f"{n_interactions:>10,} {density:>11.4f}%"
        )

    print(f"{'=' * 64}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify TA-DTCDR data-pipeline outputs.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        required=True,
        help="List of domain names (e.g. book movie).",
    )
    parser.add_argument(
        "--domain-pairs",
        nargs="*",
        default=None,
        help=(
            "Domain pairs formatted as domainA_domainB. "
            "Defaults to all combinations of --domains."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config.yaml (default: configs/config.yaml).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # -- Load config -------------------------------------------------------
    try:
        cfg = _load_config(args.config)
    except Exception as exc:
        print(f"[FATAL] Cannot load config at {args.config}: {exc}")
        return 1

    processed_dir = Path(cfg["data"]["processed_dir"])
    neg_samples_eval: int = int(cfg["data"]["negative_samples_eval"])

    # -- Resolve domain pairs ----------------------------------------------
    if args.domain_pairs is not None:
        pairs = [tuple(p.split("_", 1)) for p in args.domain_pairs]
    else:
        pairs = list(itertools.combinations(args.domains, 2))

    print("=" * 64)
    print("  TA-DTCDR Pipeline Verification")
    print(f"  Domains       : {', '.join(args.domains)}")
    print(f"  Domain pairs  : {[f'{a}_{b}' for a, b in pairs]}")
    print(f"  Processed dir : {processed_dir}")
    print(f"  Neg samples   : {neg_samples_eval}")
    print("=" * 64)

    results = CheckResults()

    # -- Per-domain checks -------------------------------------------------
    all_splits: dict[str, dict[str, pd.DataFrame]] = {}
    for domain in args.domains:
        all_splits[domain] = verify_domain(
            domain, processed_dir, neg_samples_eval, results
        )

    # -- Cross-domain checks -----------------------------------------------
    for domain_a, domain_b in pairs:
        verify_cross_domain(
            domain_a, domain_b, processed_dir, all_splits, results
        )

    # -- Table 1 stats (info only) ----------------------------------------
    print_table1(args.domains, all_splits)

    # -- Summary -----------------------------------------------------------
    results.summary()
    return results.exit_code


if __name__ == "__main__":
    sys.exit(main())
