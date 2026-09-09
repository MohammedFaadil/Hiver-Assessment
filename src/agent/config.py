"""Central configuration: paths, brand, intent taxonomy, thresholds, model names.

Everything that another module might need to agree on (intent ids, escalation
thresholds, file locations) lives here so scripts/*.py stay thin.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Brand + paths
# ---------------------------------------------------------------------------
BRAND = "AmazonHelp"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
GOLDEN_DIR = DATA_DIR / "golden"
EVAL_DIR = PROJECT_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
MODELS_DIR = PROJECT_ROOT / "models"

for _d in (RAW_DIR, CACHE_DIR, PROCESSED_DIR, GOLDEN_DIR, RESULTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CONVERSATIONS_PATH = PROCESSED_DIR / "conversations.csv"
TRAIN_POOL_PATH = PROCESSED_DIR / "train_pool.csv"          # weakly-labeled training pool (classifier)
RETRIEVAL_KB_PATH = PROCESSED_DIR / "retrieval_kb.csv"        # grounding knowledge base
GOLDEN_CANDIDATES_PATH = PROCESSED_DIR / "golden_candidates.csv"  # stratified sample shown to the labeler
GOLDEN_SET_PATH = GOLDEN_DIR / "golden_eval_set.csv"          # final hand-labeled 150-250 examples
CLASSIFIER_PATH = MODELS_DIR / "intent_classifier.joblib"

# ---------------------------------------------------------------------------
# Intent taxonomy (defined bottom-up from reading ~150 real AmazonHelp threads,
# see REPORT.md "Problem framing" for the derivation). Kept small on purpose.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Intent:
    id: str
    label: str
    description: str
    risk: str  # "low" | "medium" | "high" — feeds the escalation rule engine


INTENTS: list[Intent] = [
    Intent(
        "order_delivery",
        "Order / Delivery Status",
        "Where an order or package is, tracking not updating, late / missing / "
        "damaged / wrong item delivered.",
        risk="low",
    ),
    Intent(
        "returns_refund",
        "Returns / Refunds / Cancellations",
        "Wants to return an item, asks about refund status, or wants to cancel "
        "an order.",
        risk="medium",
    ),
    Intent(
        "billing_payment",
        "Billing / Payment Issue",
        "Incorrect or unauthorized charge, failed payment method, promo/voucher/"
        "gift-card problems.",
        risk="high",
    ),
    Intent(
        "account_access",
        "Account / Access / Subscription",
        "Login/password trouble, Prime membership, notification/email "
        "preferences, privacy or security concerns.",
        risk="medium",
    ),
    Intent(
        "product_inquiry",
        "Product / Technical / Policy Inquiry",
        "How-to questions, device or app technical issues, digital content "
        "problems, marketplace/seller/listing policy questions.",
        risk="low",
    ),
    Intent(
        "complaint_escalation",
        "Complaint / Explicit Escalation",
        "General dissatisfaction, explicit request for a supervisor/callback, "
        "repeated unresolved contact, or anger demanding action.",
        risk="high",
    ),
    Intent(
        "compliment",
        "Compliment / Positive Feedback",
        "Praise, thanks, or positive feedback with no open request.",
        risk="low",
    ),
    Intent(
        "unclear_other",
        "Unclear / Other",
        "Spam, unintelligible, non-English, or unrelated to support.",
        risk="high",
    ),
]

INTENT_IDS: list[str] = [i.id for i in INTENTS]
INTENT_BY_ID: dict[str, Intent] = {i.id: i for i in INTENTS}
RISK_BY_INTENT: dict[str, str] = {i.id: i.risk for i in INTENTS}

# ---------------------------------------------------------------------------
# Escalation policy thresholds (see src/agent/escalation.py for the rules
# that use these; see DECISION_LOG.md for why these values)
# ---------------------------------------------------------------------------
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.45
RETRIEVAL_SIMILARITY_THRESHOLD = 0.12
HIGH_RISK_INTENTS = {i.id for i in INTENTS if i.risk == "high"}

URGENCY_LEXICON = [
    r"\blawyer\b", r"\bsue\b", r"\blawsuit\b", r"\bfraud\b", r"\bscam(med)?\b",
    r"\bunacceptable\b", r"\bdisgusting\b", r"\bshame on\b", r"\bnever (buy|shop|order)ing?\b",
    r"\bstill waiting\b", r"\bagain and again\b", r"\bfor the (\d+|second|third|last) time\b",
    r"\bno one (is |has )?respond(ing|ed)?\b", r"\bignored\b", r"\bescalate\b", r"\bsupervisor\b",
    r"\bmanager\b", r"\bcomplaint\b", r"\bworst (customer )?service\b", r"\bcall me\b",
    r"\bright now\b", r"\bimmediately\b",
    # safety signals — found during golden-set labeling (rows about a fire hazard and a
    # child-safety concern) that the original lexicon missed entirely; see DECISION_LOG.md
    r"\bfire hazard\b", r"\bhazard(ous)?\b", r"\bunsafe\b", r"\bdangerous\b",
    r"\binjur(y|ed|ies)\b", r"\baccidentally activate\b", r"\bchoking\b",
]

# ---------------------------------------------------------------------------
# Retrieval / classifier hyperparameters
# ---------------------------------------------------------------------------
MAX_VOCAB_SIZE = 4000
MIN_DOC_FREQ = 2
RETRIEVAL_TOP_K = 3
RETRIEVAL_KB_MAX_SIZE = 9000   # subsample of historical resolved pairs used for grounding
TRAIN_POOL_MAX_SIZE = 6000     # weakly-labeled pool used to train the "simple baseline" classifier

# ---------------------------------------------------------------------------
# LLM (Groq) configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_BASE = "https://api.groq.com/openai/v1"
LLM_AGENT_MODEL = os.environ.get("LLM_AGENT_MODEL", "openai/gpt-oss-20b")
LLM_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "openai/gpt-oss-120b")

# Free-tier-safe throughput budget. Measured empirically against this account
# (see DECISION_LOG.md): real cap is ~8000 tokens/min; we target below that to
# leave headroom for retries and avoid 429s.
LLM_TOKEN_BUDGET_PER_MIN = 6500
LLM_MAX_RETRIES = 5

def has_llm() -> bool:
    return bool(GROQ_API_KEY)
