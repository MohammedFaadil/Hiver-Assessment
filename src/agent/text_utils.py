"""Text cleaning / tokenization shared by the classifier, retrieval index, and
weak-labeling rules. No compiled dependencies (regex + stdlib only) so it runs
anywhere — see DECISION_LOG.md for why scipy/scikit-learn were dropped.
"""
from __future__ import annotations

import re

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
AGENT_SIGNOFF_RE = re.compile(r"\s*[\^\-]\s*[A-Z]{2,3}\s*$")
NUMBER_RE = re.compile(r"\b\d+\b")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
WHITESPACE_RE = re.compile(r"\s+")

# Small built-in stopword list (avoids an nltk dependency / data download).
_STOPWORDS_TEXT = """
a an the this that these those is are was were be been being am
i you he she it we they me him her us them my your his its our their
mine yours hers ours theirs to of in on at for with from by about as into
over after under above below up down out off again further then once
and or but if because until while so than too very can will just don
do does did doing have has had having not no nor only own same
"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


def strip_urls(text: str, placeholder: str = " URLTOKEN ") -> str:
    return URL_RE.sub(placeholder, text)


def strip_mentions(text: str, placeholder: str = "") -> str:
    return MENTION_RE.sub(placeholder, text)


def strip_agent_signoff(text: str) -> str:
    """Remove trailing agent-initials sign-offs like ' ^GG' or '- AA'."""
    return AGENT_SIGNOFF_RE.sub("", text).strip()


def clean_for_vectorization(text: str) -> str:
    """Lowercase, drop mentions/urls/numbers, strip punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = strip_urls(t, " urltoken ")
    t = strip_mentions(t, " ")
    t = HASHTAG_RE.sub(r"\1", t)
    t = NUMBER_RE.sub(" numtoken ", t)
    t = NON_ALNUM_RE.sub(" ", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    cleaned = clean_for_vectorization(text)
    tokens = cleaned.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def is_english(text: str) -> bool:
    """Best-effort English filter. AmazonHelp serves many locales (see
    REPORT.md); this assessment scopes to English-language traffic only —
    multilingual support is listed as future work in DECISION_LOG.md."""
    if not isinstance(text, str):
        return False
    stripped = strip_mentions(strip_urls(text)).strip()
    if len(stripped) < 8:
        return False
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0  # deterministic across runs (see DECISION_LOG.md)
        return detect(stripped) == "en"
    except Exception:
        return False


def redact_for_reuse(text: str) -> str:
    """Prepare a historical brand reply for reuse as a template: drop the
    leading customer @mention, strip agent sign-off initials, and replace
    tracking/help links with a generic placeholder (a stale t.co link would
    be actively wrong if reused verbatim for a different customer)."""
    if not isinstance(text, str):
        return ""
    t = MENTION_RE.sub("", text, count=1).strip()
    t = URL_RE.sub("[link]", t)
    t = strip_agent_signoff(t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t
