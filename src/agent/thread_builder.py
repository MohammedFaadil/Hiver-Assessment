"""Reconstruct first-contact (customer root tweet -> brand's first reply)
pairs for one brand out of the raw flat tweet table.

Scope decision (see REPORT.md / DECISION_LOG.md): the agent triages *new*
incoming customer messages, i.e. the customer's opening tweet in a thread —
not later follow-ups (those are a dialogue-state-tracking problem, out of
scope here). This keeps "classify this incoming message" well-defined.
"""
from __future__ import annotations

import pandas as pd


def build_first_contact_pairs(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    inbound = df[df["inbound"]]
    roots = inbound[inbound["in_response_to_tweet_id"].isna()][
        ["tweet_id", "text", "created_at"]
    ].rename(columns={
        "tweet_id": "root_tweet_id",
        "text": "customer_text",
        "created_at": "customer_created_at",
    })

    brand_replies = df[
        (df["author_id"] == brand) & df["in_response_to_tweet_id"].notna()
    ].copy()
    brand_replies["in_response_to_tweet_id"] = pd.to_numeric(
        brand_replies["in_response_to_tweet_id"], errors="coerce"
    ).astype("Int64")

    merged = brand_replies.merge(
        roots,
        left_on="in_response_to_tweet_id",
        right_on="root_tweet_id",
        how="inner",
    )

    # Rare: two brand tweets both claim to reply to the same root (duplicate
    # agent response / race condition on the brand's side) — keep the first.
    merged = merged.sort_values("created_at").drop_duplicates(
        subset="root_tweet_id", keep="first"
    )

    out = merged[
        ["root_tweet_id", "customer_text", "customer_created_at", "text", "created_at"]
    ].rename(columns={"text": "brand_reply", "created_at": "brand_reply_created_at"})

    out = out.sort_values("customer_created_at").reset_index(drop=True)
    return out
