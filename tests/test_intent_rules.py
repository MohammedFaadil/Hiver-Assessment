import pytest

from agent.intent_rules import majority_class, weak_label

CASES = [
    ("my order is late and hasn't arrived yet", "order_delivery"),
    ("I preordered this and it still hasn't shipped", "order_delivery"),
    ("I want to return this item and get a refund", "returns_refund"),
    ("you charged me twice for this order", "billing_payment"),
    ("I can't log in to my account, it says locked", "account_access"),
    ("how do I set up my new echo dot", "product_inquiry"),
    ("thank you so much for the quick help, amazing service", "compliment"),
    ("this is the worst service ever, I want a supervisor", "complaint_escalation"),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_weak_label_matches_expected_category(text, expected):
    assert weak_label(text) == expected


def test_weak_label_falls_back_to_unclear_for_short_or_unmatched_text():
    assert weak_label("ok") == "unclear_other"
    assert weak_label("") == "unclear_other"
    assert weak_label(None) == "unclear_other"


def test_majority_class_returns_most_frequent_label():
    labels = ["a", "b", "a", "a", "c"]
    assert majority_class(labels) == "a"
