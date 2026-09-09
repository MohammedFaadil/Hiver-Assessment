"""Deterministic escalation rule engine.

Design choice (see DECISION_LOG.md): escalation gating is a business-policy
decision that must be auditable and consistent, so it is NOT delegated to the
LLM. It is a small set of transparent rules over (a) intent risk category,
(b) intent-classifier confidence, (c) retrieval grounding strength, and
(d) an urgency/frustration lexicon. The LLM is only used to draft reply text
and to judge reply quality — never to gate the escalation decision.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent import config

_URGENCY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in config.URGENCY_LEXICON]


@dataclass
class EscalationDecision:
    escalate: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def reason_text(self) -> str:
        return "; ".join(self.reasons)


def decide_escalation(
    intent: str,
    classifier_confidence: float | None,
    retrieval_similarity: float | None,
    customer_text: str,
) -> EscalationDecision:
    reasons: list[str] = []

    if intent in config.HIGH_RISK_INTENTS:
        reasons.append(
            f"intent '{intent}' is policy-flagged high-risk "
            "(financial impact, explicit dissatisfaction, or ambiguity)"
        )

    if classifier_confidence is not None and classifier_confidence < config.CLASSIFIER_CONFIDENCE_THRESHOLD:
        reasons.append(
            f"intent-classifier confidence {classifier_confidence:.2f} is below the "
            f"auto-handle threshold ({config.CLASSIFIER_CONFIDENCE_THRESHOLD})"
        )

    if retrieval_similarity is not None and retrieval_similarity < config.RETRIEVAL_SIMILARITY_THRESHOLD:
        reasons.append(
            f"no sufficiently similar historical resolution found "
            f"(best match similarity {retrieval_similarity:.2f} < {config.RETRIEVAL_SIMILARITY_THRESHOLD})"
        )

    matched = [p.pattern for p in _URGENCY_PATTERNS if p.search(customer_text or "")]
    if matched:
        reasons.append(
            f"urgency/frustration language detected ({len(matched)} signal(s), e.g. {matched[0]!r})"
        )

    escalate = len(reasons) > 0
    if not escalate:
        conf_str = f"{classifier_confidence:.2f}" if classifier_confidence is not None else "n/a"
        sim_str = f"{retrieval_similarity:.2f}" if retrieval_similarity is not None else "n/a"
        reasons = [
            f"intent '{intent}' is low-risk, classifier confidence {conf_str} is at/above threshold, "
            f"a grounded historical match was found (similarity {sim_str}), and no urgency flags fired"
        ]

    return EscalationDecision(escalate=escalate, reasons=reasons)
