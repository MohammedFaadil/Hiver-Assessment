import pandas as pd

from agent.thread_builder import build_first_contact_pairs


def _row(tweet_id, author_id, inbound, text, in_response_to, created_at="2020-01-01"):
    return {
        "tweet_id": tweet_id,
        "author_id": author_id,
        "inbound": inbound,
        "text": text,
        "in_response_to_tweet_id": in_response_to,
        "created_at": pd.Timestamp(created_at),
    }


def test_build_first_contact_pairs_matches_root_customer_message_to_brand_reply():
    df = pd.DataFrame([
        # a genuine first-contact thread: customer root -> brand reply
        _row(1, "1001", True, "my package is late", None),
        _row(2, "BrandCo", False, "sorry to hear, checking now", 1, "2020-01-01 00:01:00"),
        # a second brand reply further down the same thread -- should NOT be
        # picked up as the "first contact" pair (only tweet_id=1's root matters)
        _row(3, "1001", True, "any update?", 2, "2020-01-01 00:02:00"),
        _row(4, "BrandCo", False, "still checking", 3, "2020-01-01 00:03:00"),
    ])
    pairs = build_first_contact_pairs(df, "BrandCo")
    assert len(pairs) == 1
    assert pairs.iloc[0]["customer_text"] == "my package is late"
    assert pairs.iloc[0]["brand_reply"] == "sorry to hear, checking now"


def test_build_first_contact_pairs_excludes_brand_initiated_threads():
    df = pd.DataFrame([
        # root is the BRAND, not a customer -- should be excluded entirely
        _row(1, "BrandCo", False, "we noticed an issue with your order", None),
        _row(2, "1001", True, "oh really, what happened", 1, "2020-01-01 00:01:00"),
        _row(3, "BrandCo", False, "let us look into it", 2, "2020-01-01 00:02:00"),
    ])
    pairs = build_first_contact_pairs(df, "BrandCo")
    assert len(pairs) == 0


def test_build_first_contact_pairs_ignores_other_brands():
    df = pd.DataFrame([
        _row(1, "2001", True, "hello is anyone there", None),
        _row(2, "OtherBrand", False, "yes we are here", 1, "2020-01-01 00:01:00"),
    ])
    pairs = build_first_contact_pairs(df, "BrandCo")
    assert len(pairs) == 0


def test_build_first_contact_pairs_keeps_earliest_reply_on_duplicate_root():
    df = pd.DataFrame([
        _row(1, "3001", True, "help please", None),
        _row(2, "BrandCo", False, "second (later) reply", 1, "2020-01-01 00:05:00"),
        _row(3, "BrandCo", False, "first (earlier) reply", 1, "2020-01-01 00:01:00"),
    ])
    pairs = build_first_contact_pairs(df, "BrandCo")
    assert len(pairs) == 1
    assert pairs.iloc[0]["brand_reply"] == "first (earlier) reply"


def test_build_first_contact_pairs_handles_no_matches_gracefully():
    df = pd.DataFrame([
        _row(1, "4001", True, "just a tweet, no reply", None),
    ])
    pairs = build_first_contact_pairs(df, "BrandCo")
    assert len(pairs) == 0
    assert list(pairs.columns) == [
        "root_tweet_id", "customer_text", "customer_created_at", "brand_reply", "brand_reply_created_at",
    ]
