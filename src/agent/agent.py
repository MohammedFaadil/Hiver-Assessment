"""Orchestrator: wires classifier + retrieval + escalation rules + (optional)
LLM into three run modes that share one interface (AgentResult):

  - trivial  : majority-class intent, one canned reply, never escalates.
  - simple   : TF-IDF/NaiveBayes intent + nearest-neighbor templated reply +
               rule-based escalation. No LLM calls — the "simple baseline".
  - llm      : same retrieval grounding, but the reply is drafted by the Groq
               LLM (RAG-style, conditioned on retrieved precedent) and the
               LLM also proposes an intent + its own escalation opinion,
               which we log for comparison. The actual escalate/no-escalate
               decision always comes from the deterministic rule engine
               (see escalation.py) so it stays auditable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import joblib

from agent import config, llm_client
from agent.escalation import decide_escalation
from agent.nlp_model import MultinomialNB, Vectorizer
from agent.retrieval import RetrievalIndex
from agent.text_utils import redact_for_reuse

CANNED_REPLY = (
    "Thanks for reaching out! We've received your message and a member of "
    "our team will follow up shortly."
)

AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": config.INTENT_IDS},
        "drafted_reply": {"type": "string"},
        "confidence": {"type": "number"},
        "needs_escalation": {"type": "boolean"},
        "escalation_reason": {"type": "string"},
    },
    "required": ["intent", "drafted_reply", "confidence", "needs_escalation", "escalation_reason"],
    "additionalProperties": False,
}

AGENT_SYSTEM_PROMPT = (
    f"You are a support-triage assistant for {config.BRAND} on Twitter. "
    "Given a customer's tweet and similar past cases showing how this brand "
    "actually resolved them, do three things: "
    "(1) classify the customer's intent into exactly one of the allowed categories, "
    "(2) draft a concise, empathetic, on-brand reply that is GROUNDED in the past "
    "cases — do not invent order numbers, refund amounts, or policies that are not "
    "shown or implied by the examples; ask the customer for details you don't have, "
    "(3) flag whether this needs a human agent, with a short reason.\n\n"
    "Allowed intents:\n" + "\n".join(f"- {i.id}: {i.description}" for i in config.INTENTS)
)


@dataclass
class AgentResult:
    customer_text: str
    mode: str
    intent: str
    intent_confidence: float | None
    drafted_reply: str
    retrieval_similarity: float | None
    retrieved_examples: list[dict] = field(default_factory=list)
    escalate: bool = False
    escalation_reason: str = ""
    llm_suggested_escalate: bool | None = None
    llm_escalation_reason: str | None = None
    latency_s: float | None = None
    usage: dict | None = None


def _build_examples_block(retrieved) -> list[dict]:
    return [
        {"customer_text": r["customer_text"], "brand_reply": r["brand_reply"], "similarity": float(r["similarity"])}
        for r in retrieved.to_dict("records")
    ]


@dataclass
class Pipeline:
    vectorizer: Vectorizer
    classifier: MultinomialNB
    retrieval_index: RetrievalIndex
    majority_intent: str
    canned_reply: str = CANNED_REPLY

    @classmethod
    def load(cls, classifier_path=None, retrieval_path=None) -> "Pipeline":
        classifier_path = classifier_path or config.CLASSIFIER_PATH
        retrieval_path = retrieval_path or (config.MODELS_DIR / "retrieval_index.joblib")
        bundle = joblib.load(classifier_path)
        retrieval_index = RetrievalIndex.load(retrieval_path)
        return cls(
            vectorizer=bundle["vectorizer"],
            classifier=bundle["classifier"],
            retrieval_index=retrieval_index,
            majority_intent=bundle["majority_intent"],
        )

    def save(self, classifier_path=None, retrieval_path=None) -> None:
        classifier_path = classifier_path or config.CLASSIFIER_PATH
        retrieval_path = retrieval_path or (config.MODELS_DIR / "retrieval_index.joblib")
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "majority_intent": self.majority_intent,
            },
            classifier_path,
        )
        self.retrieval_index.save(retrieval_path)

    def classify(self, text: str) -> tuple[str, float]:
        X = self.vectorizer.transform_counts([text])
        proba = self.classifier.predict_proba(X)[0]
        idx = int(proba.argmax())
        return str(self.classifier.classes_[idx]), float(proba[idx])

    # -- run modes ---------------------------------------------------------

    def run_trivial(self, customer_text: str) -> AgentResult:
        decision = decide_escalation(
            intent=self.majority_intent, classifier_confidence=None,
            retrieval_similarity=None, customer_text=customer_text,
        )
        # The trivial baseline is intentionally "do nothing smart": it never
        # escalates, regardless of what the (unused) rule engine would say.
        return AgentResult(
            customer_text=customer_text,
            mode="trivial",
            intent=self.majority_intent,
            intent_confidence=None,
            drafted_reply=self.canned_reply,
            retrieval_similarity=None,
            retrieved_examples=[],
            escalate=False,
            escalation_reason="trivial baseline never escalates by definition",
        )

    def run_simple(self, customer_text: str, k: int = config.RETRIEVAL_TOP_K) -> AgentResult:
        intent, conf = self.classify(customer_text)
        retrieved = self.retrieval_index.query(customer_text, k=k, intent_filter=intent)
        top_sim = float(retrieved["similarity"].iloc[0]) if len(retrieved) else 0.0
        reply = redact_for_reuse(retrieved["brand_reply"].iloc[0]) if len(retrieved) else self.canned_reply

        decision = decide_escalation(
            intent=intent, classifier_confidence=conf,
            retrieval_similarity=top_sim, customer_text=customer_text,
        )
        return AgentResult(
            customer_text=customer_text,
            mode="simple",
            intent=intent,
            intent_confidence=conf,
            drafted_reply=reply,
            retrieval_similarity=top_sim,
            retrieved_examples=_build_examples_block(retrieved),
            escalate=decision.escalate,
            escalation_reason=decision.reason_text,
        )

    def run_llm(self, customer_text: str, k: int = config.RETRIEVAL_TOP_K, model: str | None = None) -> AgentResult:
        model = model or config.LLM_AGENT_MODEL
        intent_guess, conf = self.classify(customer_text)
        retrieved = self.retrieval_index.query(customer_text, k=k, intent_filter=intent_guess)
        top_sim = float(retrieved["similarity"].iloc[0]) if len(retrieved) else 0.0
        examples = _build_examples_block(retrieved)

        examples_block = "\n".join(
            f'{i+1}. "{e["customer_text"]}" -> "{e["brand_reply"]}"' for i, e in enumerate(examples)
        ) or "(no similar past cases found)"
        user_prompt = (
            f'CUSTOMER MESSAGE:\n"{customer_text}"\n\n'
            f"SIMILAR PAST CASES (customer message -> how {config.BRAND} replied):\n{examples_block}\n\n"
            "Respond with the JSON fields requested."
        )

        t0 = time.time()
        llm_out = llm_client.chat_structured(
            system_prompt=AGENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=AGENT_SCHEMA,
            schema_name="agent_response",
            model=model,
            max_tokens=500,
        )
        latency = time.time() - t0

        final_intent = llm_out.get("intent") if llm_out.get("intent") in config.INTENT_IDS else intent_guess

        decision = decide_escalation(
            intent=final_intent, classifier_confidence=conf,
            retrieval_similarity=top_sim, customer_text=customer_text,
        )
        return AgentResult(
            customer_text=customer_text,
            mode="llm",
            intent=final_intent,
            intent_confidence=conf,
            drafted_reply=llm_out.get("drafted_reply", "").strip(),
            retrieval_similarity=top_sim,
            retrieved_examples=examples,
            escalate=decision.escalate,
            escalation_reason=decision.reason_text,
            llm_suggested_escalate=bool(llm_out.get("needs_escalation", False)),
            llm_escalation_reason=llm_out.get("escalation_reason", ""),
            latency_s=latency,
            usage=llm_out.get("_usage"),
        )
