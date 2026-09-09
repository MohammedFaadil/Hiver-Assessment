"""Stratified sample of golden_candidates.csv for hand-labeling.

We deliberately over-sample rare intents relative to their natural frequency
(target counts below) rather than sampling proportionally: with only 150-250
slots total, proportional sampling would leave classes like account_access
or billing_payment with too few examples for a meaningful per-class
precision/recall estimate. The true (skewed) class frequency is reported
separately from the full 33.8k pool in REPORT.md, so this choice doesn't
misrepresent real-world prevalence — see DECISION_LOG.md.

weak_intent here is only used to make sure the reviewed sample is topically
diverse; the actual gold_intent assigned during hand-labeling (see
LABELING_GUIDE.md) is an independent judgment call, not copied from it.
"""
from __future__ import annotations

import pandas as pd

from agent import config

TARGET_COUNTS = {
    "order_delivery": 45,
    "complaint_escalation": 30,
    "unclear_other": 25,
    "returns_refund": 25,
    "compliment": 20,
    "billing_payment": 20,
    "product_inquiry": 20,
    "account_access": 15,
}

OUT_PATH = config.PROCESSED_DIR / "golden_sample_to_label.csv"


def main() -> None:
    df = pd.read_csv(config.GOLDEN_CANDIDATES_PATH)
    parts = []
    for intent, n in TARGET_COUNTS.items():
        bucket = df[df["weak_intent"] == intent]
        n_take = min(n, len(bucket))
        parts.append(bucket.sample(n=n_take, random_state=config.RANDOM_SEED))
        if n_take < n:
            print(f"WARNING: only {n_take}/{n} available for {intent}")
    sample = pd.concat(parts).sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    sample.insert(0, "row_id", range(1, len(sample) + 1))
    sample.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(sample)} rows to {OUT_PATH}")
    print(sample["weak_intent"].value_counts())


if __name__ == "__main__":
    main()
