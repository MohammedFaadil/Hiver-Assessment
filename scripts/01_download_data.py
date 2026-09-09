"""Download the Kaggle "Customer Support on Twitter" dataset (anonymous
public download via kagglehub — no API key needed) and report basic stats.
Safe to re-run: kagglehub caches the download under ~/.cache/kagglehub.
"""
from __future__ import annotations

from agent.data_ingest import download_raw, load_raw


def main() -> None:
    csv_path = download_raw()
    print(f"Dataset available at: {csv_path}")
    df = load_raw(csv_path)
    print(f"Total rows: {len(df):,}")
    print("\nTop 10 brand accounts by outbound tweet volume:")
    print(df.loc[~df["inbound"], "author_id"].value_counts().head(10))


if __name__ == "__main__":
    main()
