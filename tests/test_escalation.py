from agent import config
from agent.escalation import decide_escalation


def test_high_risk_intent_always_escalates_regardless_of_confidence():
    decision = decide_escalation(
        intent="billing_payment", classifier_confidence=0.99,
        retrieval_similarity=0.9, customer_text="what payment methods do you accept",
    )
    assert decision.escalate is True
    assert any("billing_payment" in r for r in decision.reasons)


def test_low_risk_intent_with_high_confidence_and_similarity_auto_handles():
    decision = decide_escalation(
        intent="order_delivery", classifier_confidence=0.95,
        retrieval_similarity=0.6, customer_text="where is my package",
    )
    assert decision.escalate is False


def test_low_confidence_forces_escalation_even_for_low_risk_intent():
    decision = decide_escalation(
        intent="order_delivery",
        classifier_confidence=config.CLASSIFIER_CONFIDENCE_THRESHOLD - 0.1,
        retrieval_similarity=0.6, customer_text="where is my package",
    )
    assert decision.escalate is True
    assert any("confidence" in r for r in decision.reasons)


def test_low_retrieval_similarity_forces_escalation():
    decision = decide_escalation(
        intent="order_delivery", classifier_confidence=0.95,
        retrieval_similarity=config.RETRIEVAL_SIMILARITY_THRESHOLD - 0.01,
        customer_text="a totally novel kind of message",
    )
    assert decision.escalate is True
    assert any("similar historical resolution" in r for r in decision.reasons)


def test_urgency_lexicon_triggers_escalation_for_low_risk_intent():
    decision = decide_escalation(
        intent="product_inquiry", classifier_confidence=0.95, retrieval_similarity=0.6,
        customer_text="this is a fire hazard and I am worried about my kids",
    )
    assert decision.escalate is True
    assert any("urgency" in r for r in decision.reasons)


def test_auto_handle_reason_mentions_all_checks_passed():
    decision = decide_escalation(
        intent="compliment", classifier_confidence=0.9,
        retrieval_similarity=0.5, customer_text="thanks so much for the help",
    )
    assert decision.escalate is False
    assert "low-risk" in decision.reason_text


def test_none_confidence_and_similarity_do_not_crash():
    decision = decide_escalation(
        intent="order_delivery", classifier_confidence=None,
        retrieval_similarity=None, customer_text="where is my order",
    )
    assert decision.escalate is False
    assert "n/a" in decision.reason_text
