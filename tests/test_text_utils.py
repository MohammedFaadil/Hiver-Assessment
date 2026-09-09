from agent.text_utils import (
    clean_for_vectorization,
    redact_for_reuse,
    strip_agent_signoff,
    strip_mentions,
    strip_urls,
    tokenize,
)


def test_strip_urls_replaces_with_placeholder():
    out = strip_urls("check this out https://t.co/abc123 thanks")
    assert "https://" not in out
    assert "URLTOKEN" in out.upper() or "urltoken" in out


def test_strip_mentions_removes_handle():
    assert strip_mentions("@AmazonHelp hello there") == " hello there"


def test_strip_agent_signoff_removes_trailing_initials():
    assert strip_agent_signoff("We can help with that ^GG") == "We can help with that"
    assert strip_agent_signoff("We can help with that - AA") == "We can help with that"
    assert strip_agent_signoff("No signoff here") == "No signoff here"


def test_clean_for_vectorization_lowercases_and_strips_noise():
    out = clean_for_vectorization("@AmazonHelp my order #12345 is LATE! see https://t.co/abc #fail")
    assert out == out.lower()
    assert "@" not in out
    assert "http" not in out
    assert "12345" not in out  # numbers become a shared placeholder token
    assert "late" in out


def test_tokenize_removes_stopwords_and_short_tokens():
    tokens = tokenize("the package is at my house and it is late")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "package" in tokens
    assert "late" in tokens


def test_tokenize_can_keep_stopwords():
    tokens = tokenize("the package is late", remove_stopwords=False)
    assert "the" in tokens
    assert "is" in tokens


def test_redact_for_reuse_strips_leading_mention_url_and_signoff():
    raw = "@382016 One-Day refers to transit time once it ships. https://t.co/xyz ^VB"
    out = redact_for_reuse(raw)
    assert not out.startswith("@")
    assert "https://" not in out
    assert "[link]" in out
    assert not out.endswith("^VB")
    assert "One-Day refers to transit time" in out
