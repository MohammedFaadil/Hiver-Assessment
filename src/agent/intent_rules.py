"""Weak/distant supervision: keyword-rule labeling functions used ONLY to
build a noisy training pool for the classical "simple baseline" classifier.

This is deliberate weak supervision, not gold labels: rules are applied in
priority order (first match wins) to a large unlabeled pool, the classifier
learns from that noisy signal, and it is then evaluated against the
hand-labeled golden set (data/golden/golden_eval_set.csv), which these rules
never touch. See REPORT.md "Problem framing" and DECISION_LOG.md for why.
"""
from __future__ import annotations

import re

# Order matters: first pattern that matches wins.
#
# Note the liberal use of \w* suffix wildcards on stems (e.g. "order\w*"
# rather than "order"): an earlier version anchored a trailing \b directly
# after literal stems like "order" / "charge", which silently failed to match
# their own inflections ("ordered", "charging") because \b requires a
# word-boundary immediately after the literal — there isn't one when a suffix
# follows. Caught by auditing a sample of the "unclear_other" bucket, which
# was ~35% of the pool before this fix (see DECISION_LOG.md).
_RULES: list[tuple[str, re.Pattern]] = [
    ("complaint_escalation", re.compile(
        r"\b(escalat\w*|supervisor|manager|lawyer|lawsuit|sue|fraud\w*|"
        r"scam\w*|ridiculous|unacceptable|disgusting|shame on|never (buy|shop|order)\w*|"
        r"still waiting|again and again|for the (\d+|second|third|last) time|"
        r"no one (is |has )?respond\w*|ignor\w*|worst (customer )?service|"
        r"shocking|terrible|horrible|awful|appalling|pathetic|disappoint\w*|"
        r"call me|do something|fix this now|complain\w*|fed ?up)\b", re.IGNORECASE)),
    ("compliment", re.compile(
        r"\b(thank\w*|thx|appreciat\w*|awesome|great (service|job|help|work)|"
        r"love\w* (you|amazon|this)|excellent|amazing|kudos|well done|fantastic)\b", re.IGNORECASE)),
    ("billing_payment", re.compile(
        r"\b(charg\w*|overcharg\w*|payment\w*|billing|invoic\w*|gift ?card\w*|promo\w*|"
        r"voucher\w*|coupon\w*|price (match|error)|subscription\w*|debit\w*|deduct\w*)\b", re.IGNORECASE)),
    ("returns_refund", re.compile(
        r"\b(return\w*|refund\w*|exchang\w*|cancel\w*|money back|rma)\b", re.IGNORECASE)),
    ("account_access", re.compile(
        r"\b(log ?in\w*|log ?out\w*|password\w*|my account|account\w*.{0,15}(access|locked|hacked|"
        r"suspended|closed)|locked|hacked|suspended|"
        r"prime membership|two[- ]factor|2fa|verification code|email preferences|"
        r"unsubscrib\w*|notification\w*|privacy|security (concern|issue)|sign ?in\w*|household)\b", re.IGNORECASE)),
    ("order_delivery", re.compile(
        r"\b(order\w*|pre-?order\w*|package\w*|deliver\w*|ship\w*|dispatch\w*|track\w*|arriv\w*|"
        r"lost|los(e|ing) (my|the|it)|where is my|hasn'?t (arrived|shown up|come)|"
        r"missing (item|package|order)|damaged|"
        r"wrong (item|order|product)|sent me (a|the) wrong|late (delivery|package)|"
        r"out for delivery|courier|parcel\w*)\b", re.IGNORECASE)),
    ("product_inquiry", re.compile(
        r"\b(kindle\w*|echo|alexa|fire ?(tv|stick|tablet|hd)\w*|app\w* (crash\w*|bug\w*|issue\w*)|"
        r"not working|isn'?t working|doesn'?t work|won'?t (work|play|load)|"
        r"device\w*|how do i|how to|instruction\w*|seller\w*|"
        r"marketplace|third[- ]party|listing\w*|createspace|app store|firmware|"
        r"update\w* (issue|problem)|feature request|screen mirroring|trade[- ]?in\w*)\b", re.IGNORECASE)),
]


def weak_label(text: str) -> str:
    if not isinstance(text, str) or len(text.split()) < 2:
        return "unclear_other"
    for label, pattern in _RULES:
        if pattern.search(text):
            return label
    return "unclear_other"


def weak_label_with_rule(text: str) -> tuple[str, str | None]:
    """Same as weak_label but also returns which regex matched (for audits)."""
    if not isinstance(text, str) or len(text.split()) < 2:
        return "unclear_other", None
    for label, pattern in _RULES:
        m = pattern.search(text)
        if m:
            return label, m.group(0)
    return "unclear_other", None


def majority_class(labels) -> str:
    vals, counts = _unique_counts(labels)
    return vals[counts.argmax()]


def _unique_counts(labels):
    import numpy as np
    arr = np.asarray(list(labels))
    vals, counts = np.unique(arr, return_counts=True)
    return vals, counts
