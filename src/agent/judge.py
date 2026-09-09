"""LLM-as-judge: scores a drafted reply against a 5-point rubric across four
dimensions plus an overall score. Deliberately uses a different (larger)
model than the one that drafts replies (config.LLM_JUDGE_MODEL vs
LLM_AGENT_MODEL) to reduce — not eliminate — same-model self-preference bias.
See REPORT.md "what's misleading about my headline number" for the caveat
that both still share a model family/provider, and eval/judge_human_agreement
for how well this judge tracks a manual read of the same replies.
"""
from __future__ import annotations

from agent import config, llm_client

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "grounding_score": {"type": "integer"},
        "helpfulness_score": {"type": "integer"},
        "tone_score": {"type": "integer"},
        "clarity_score": {"type": "integer"},
        "overall_score": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": [
        "grounding_score", "helpfulness_score", "tone_score",
        "clarity_score", "overall_score", "rationale",
    ],
    "additionalProperties": False,
}

JUDGE_SYSTEM_PROMPT = f"""You are a strict quality auditor grading draft replies written by an AI \
support agent for {config.BRAND}'s Twitter support account. Score the DRAFTED REPLY on a 1-5 \
integer scale (5 = excellent, 1 = very poor) on each dimension:

- grounding_score: Does it avoid inventing facts (order numbers, refund amounts, promises, \
policies) not present in the customer message or the shown past-case precedent, and does it \
sensibly draw on that precedent when relevant? Fabrication or contradiction of precedent = 1-2.
- helpfulness_score: Does it move the customer toward resolution with a correct, actionable \
next step or answer?
- tone_score: Professional, empathetic, on-brand, appropriate to the customer's sentiment.
- clarity_score: Concise and easy to understand, no rambling or ambiguity.
- overall_score: Your holistic 1-5 judgment of reply quality, weighing all of the above.

Be a skeptical grader: reserve 5s for genuinely excellent replies. Justify with a 1-2 sentence \
rationale citing the specific reason for the scores."""


def judge_reply(
    customer_text: str,
    drafted_reply: str,
    retrieved_examples: list[dict] | None = None,
    model: str | None = None,
) -> dict:
    model = model or config.LLM_JUDGE_MODEL
    examples = retrieved_examples or []
    examples_block = "\n".join(
        f'{i+1}. "{e["customer_text"]}" -> "{e["brand_reply"]}"' for i, e in enumerate(examples)
    ) or "(none provided)"

    user_prompt = (
        f'CUSTOMER MESSAGE:\n"{customer_text}"\n\n'
        f"PAST-CASE PRECEDENT SHOWN TO THE AGENT:\n{examples_block}\n\n"
        f'DRAFTED REPLY TO GRADE:\n"{drafted_reply}"\n\n'
        "Respond with the JSON fields requested."
    )

    out = llm_client.chat_structured(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=JUDGE_SCHEMA,
        schema_name="judge_response",
        model=model,
        max_tokens=400,
    )
    for k in ("grounding_score", "helpfulness_score", "tone_score", "clarity_score", "overall_score"):
        out[k] = max(1, min(5, int(out.get(k, 3))))
    return out
