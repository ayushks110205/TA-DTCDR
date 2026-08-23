"""
Phase 2 — LLM-based Tag Distillation
======================================
Paper reference: Section 4.1, Figure 3, Equation 1.

Responsibilities:
1. Build the 3-part prompt: INSTRUCTION + NOTICE + 3 few-shot EXAMPLES
   (one example per domain: book, movie, music) — copy structure from Fig. 3.
2. For every item (title + synopsis), call the LLM to get exactly 5
   single-word tags.
3. Validate output format (comma-separated, exactly 5 tags, single words,
   no genre words) — retry on malformed output.
4. Filter: keep tags that (a) appear in BOTH domains of a pair, and
   (b) have frequency >= threshold (default 80, see config).

Output: data/tags/{domain}_tags_raw.json      (item_id -> list[str])
        data/tags/{domain_pair}_tags_filtered.json (shared tag vocab)
"""

import argparse
import json
import re
from pathlib import Path

import yaml
from openai import OpenAI

# ---- Prompt components (mirrors Figure 3 exactly) ----

INSTRUCTION = (
    "You are an expert tagger. Based on the input domain, title, and synopsis "
    "below, use your knowledge to generate at least 5 tags for the item. "
    "Each tag should be given using only one word."
)

NOTICE = """\
1. Do not output tags related to the item's genre, such as book, novel, poetry, movie, comedy, music, etc.
2. Do not output tags referring to specific people, place names, organizations, etc.
3. This is a cross-domain task, so consider all mentioned domains below: 'Book', 'Movie', and 'Music'. Ensure each tag works and expresses the same meaning in both domains.
4. Return your answer as a comma-separated list: tag1, tag2, tag3, ...
5. Do not include any additional text, just the tags.
"""

# Few-shot examples — one per domain, taken directly from Figure 3.
# You may swap in your own filtered dataset's representative items instead.
FEWSHOT_EXAMPLES = [
    {
        "domain": "BOOK",
        "title": "Island of the Blue Dolphins",
        "synopsis": "In the Pacific there is an island that looks like...",
        "tags": "island, survival, dolphins, solitude, discovery",
    },
    {
        "domain": "MOVIE",
        "title": "Robocop",
        "synopsis": "There's a new law enforcer in town and he's half...",
        "tags": "cyborg, justice, revenge, technology, crime",
    },
    {
        "domain": "MUSIC",
        "title": "On The Threshold Of A Dream",
        "synopsis": "Product description Audio CD Amazon.com...",
        "tags": "cosmic, futuristic, poetry, melancholy, melody",
    },
]


def build_prompt(domain: str, title: str, synopsis: str) -> str:
    """Concatenate INSTRUCTION + NOTICE + few-shot EXAMPLES + actual item input,
    matching the structure shown in Figure 3.
    """
    raise NotImplementedError


def call_llm(client: OpenAI, prompt: str, model: str) -> str:
    """Call the LLM and return raw text response."""
    raise NotImplementedError


def parse_tags(raw_response: str, expected_n: int = 5) -> list:
    """Parse comma-separated tag string into a clean list of lowercase,
    single-word tags. Return None if malformed (caller should retry).
    """
    raise NotImplementedError


def distill_domain_tags(domain: str, items_df, client: OpenAI, model: str) -> dict:
    """Loop over all items in a domain, distill tags for each.
    Returns dict: item_id -> list[str] (tags).
    Consider batching / async calls to reduce wall-clock time and cost.
    """
    raise NotImplementedError


def filter_shared_tags(tags_a: dict, tags_b: dict, freq_threshold: int) -> set:
    """Keep only tags that appear in items from BOTH domains, with
    frequency >= freq_threshold in each domain (Section 4.1).
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # TODO: load processed items for domain, run distill_domain_tags,
    #       save raw tags. Filtering across domain pairs happens once both
    #       domains in a pair are distilled — run as a separate merge step.


if __name__ == "__main__":
    main()
