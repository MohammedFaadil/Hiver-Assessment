"""Build the working dataset for one brand: reconstruct first-contact pairs,
keep English-language / reasonable-length messages, weak-label intents, and
split into three DISJOINT-by-conversation pools:

  - golden_candidates : stratified-by-weak-label sample for hand labeling
                         (data/golden/golden_eval_set.csv is drawn from this)
  - retrieval_kb       : historical (customer msg -> brand reply) pairs used
                         to ground both the template baseline and the LLM
  - train_pool         : weakly-labeled examples used to train the classical
                         "simple baseline" intent classifier

golden_candidates is carved out FIRST and excluded from the other two pools,
so nothing the agent is graded on ever leaked into its own grounding data or
classifier training set.

We start from a bounded random subsample of the ~165k reconstructable
AmazonHelp conversations rather than all of them — the assignment explicitly
expects and encourages subsampling, and it keeps this script fast.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from agent import config
from agent.data_ingest import load_raw
from agent.intent_rules import weak_label
from agent.text_utils import is_english, redact_for_reuse

RAW_SUBSAMPLE_SIZE = 45_000   # candidate first-contact pairs before quality filtering
GOLDEN_CANDIDATES_SIZE = 550  # oversample vs. the 150-250 we'll finally hand-label
MIN_CHARS = 15
MAX_CHARS = 320


def main() -> None:
    rng = np.random.default_rng(config.RANDOM_SEED)
    t0 = time.time()

    print(f"[1/6] Loading raw dataset (this downloads ~170MB on first run)...")
    from agent.thread_builder import build_first_contact_pairs

    df = load_raw()
    print(f"      {len(df):,} raw rows loaded in {time.time()-t0:.1f}s")

    print(f"[2/6] Reconstructing first-contact pairs for brand={config.BRAND!r}...")
    pairs = build_first_contact_pairs(df, config.BRAND)
    print(f"      {len(pairs):,} reconstructable (customer -> {config.BRAND} reply) pairs")
    del df

    print(f"[3/6] Subsampling {RAW_SUBSAMPLE_SIZE:,} candidates for quality filtering...")
    if len(pairs) > RAW_SUBSAMPLE_SIZE:
        pairs = pairs.sample(n=RAW_SUBSAMPLE_SIZE, random_state=config.RANDOM_SEED).reset_index(drop=True)

    print("[4/6] Filtering by length + English language (this runs langdetect per row)...")
    len_ok = pairs["customer_text"].str.len().between(MIN_CHARS, MAX_CHARS)
    pairs = pairs[len_ok].reset_index(drop=True)
    print(f"      {len(pairs):,} remain after length filter")

    tqdm.pandas(desc="      langdetect")
    is_en = pairs["customer_text"].progress_apply(is_english)
    pairs = pairs[is_en].reset_index(drop=True)
    print(f"      {len(pairs):,} remain after English-language filter")

    print("[5/6] Weak-labeling intents (keyword rules; training signal only, not gold)...")
    pairs["weak_intent"] = pairs["customer_text"].apply(weak_label)
    pairs["brand_reply_clean"] = pairs["brand_reply"].apply(redact_for_reuse)
    print(pairs["weak_intent"].value_counts())

    print("[6/6] Splitting into golden_candidates / retrieval_kb / train_pool (disjoint golden slice)...")
    idx = rng.permutation(len(pairs))
    pairs = pairs.iloc[idx].reset_index(drop=True)

    golden_candidates = pairs.iloc[:GOLDEN_CANDIDATES_SIZE].copy()
    remainder = pairs.iloc[GOLDEN_CANDIDATES_SIZE:].copy()

    kb_size = min(config.RETRIEVAL_KB_MAX_SIZE, len(remainder))
    retrieval_kb = remainder.sample(n=kb_size, random_state=config.RANDOM_SEED)

    train_size = min(config.TRAIN_POOL_MAX_SIZE, len(remainder))
    train_pool = remainder.sample(n=train_size, random_state=config.RANDOM_SEED + 1)

    golden_candidates.to_csv(config.GOLDEN_CANDIDATES_PATH, index=False)
    retrieval_kb.rename(columns={"brand_reply_clean": "brand_reply_redacted"}).to_csv(
        config.RETRIEVAL_KB_PATH, index=False
    )
    train_pool.to_csv(config.TRAIN_POOL_PATH, index=False)
    pairs.to_csv(config.CONVERSATIONS_PATH, index=False)

    print(f"\nSaved:")
    print(f"  golden candidates : {len(golden_candidates):>6,} -> {config.GOLDEN_CANDIDATES_PATH}")
    print(f"  retrieval KB      : {len(retrieval_kb):>6,} -> {config.RETRIEVAL_KB_PATH}")
    print(f"  train pool        : {len(train_pool):>6,} -> {config.TRAIN_POOL_PATH}")
    print(f"  full clean pool   : {len(pairs):>6,} -> {config.CONVERSATIONS_PATH}")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
