"""Download (or reuse a cached copy of) the Kaggle "Customer Support on
Twitter" dataset and load it with sane dtypes.

Uses kagglehub's anonymous public-dataset download (no KAGGLE_USERNAME/
KAGGLE_KEY required — Kaggle opened this up for public datasets in April
2024). If that ever regresses, point RAW_CSV_OVERRIDE at a manually
downloaded copy of twcs.csv.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DATASET_SLUG = "thoughtvector/customer-support-on-twitter"
RAW_CSV_OVERRIDE = os.environ.get("RAW_CSV_OVERRIDE", "")

DTYPES = {
    "tweet_id": "int64",
    "author_id": "string",
    "inbound": "bool",
    "text": "string",
    "response_tweet_id": "string",
    "in_response_to_tweet_id": "string",
}
USECOLS = list(DTYPES.keys()) + ["created_at"]


def download_raw() -> Path:
    if RAW_CSV_OVERRIDE:
        p = Path(RAW_CSV_OVERRIDE)
        if not p.exists():
            raise FileNotFoundError(f"RAW_CSV_OVERRIDE set but not found: {p}")
        return p

    import kagglehub

    dataset_dir = Path(kagglehub.dataset_download(DATASET_SLUG))
    direct = dataset_dir / "twcs" / "twcs.csv"
    if direct.exists():
        return direct
    candidates = list(dataset_dir.rglob("twcs.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find twcs.csv under {dataset_dir}. "
            "Set RAW_CSV_OVERRIDE to a manual download if this persists."
        )
    return candidates[0]


def load_raw(csv_path: Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Load the full (or nrows-limited) raw tweet table with parsed timestamps."""
    if csv_path is None:
        csv_path = download_raw()
    df = pd.read_csv(csv_path, dtype=DTYPES, usecols=USECOLS, nrows=nrows)
    df["created_at"] = pd.to_datetime(
        df["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce"
    )
    return df
