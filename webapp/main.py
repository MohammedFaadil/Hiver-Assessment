"""FastAPI backend for the Hiver AI Support Agent demo.

Serves a static frontend (webapp/static/) plus a small JSON API that runs the
real pipeline (src/agent) against user-typed messages and exposes the
evaluation artifacts already computed under eval/results/.

Run with:  uvicorn webapp.main:app --reload --port 8000
(see scripts/run_webapp.py for a plain `python` entry point too)
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import config
from agent.agent import Pipeline

app = FastAPI(title="Hiver AI Support Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline.load()
    return _pipeline


EXAMPLES = [
    {
        "label": "Order status",
        "text": "I had made the payment in advance against order 407-2602912-8773954. "
                "Please provide me a replacement if the product is lost by you",
    },
    {
        "label": "Return delay",
        "text": "What sort of service is this? 4 messages about a scheduled pickup on the "
                "way and suddenly a message saying it's cancelled with no alternative!",
    },
    {
        "label": "Overcharge dispute",
        "text": "Amazon have overcharged me for my order and I need this sorted out ASAP "
                "with a rep please!",
    },
    {
        "label": "Locked out (8 months)",
        "text": "It has been 8 months that I can't login into my account. Without access "
                "I can't even ask for help.",
    },
    {
        "label": "Repeat product defect",
        "text": "My daughter's kindle is broken for the 2nd time. Same issue, and it's no "
                "longer under warranty. Can you help? Very disappointed.",
    },
    {
        "label": "Escalation request",
        "text": "How do I get a direct email or phone number for your senior legal team? "
                "Urgent matter, please follow & DM me.",
    },
    {
        "label": "Compliment",
        "text": "I'm like a 5 year old playing with my new fire stick! It has an Alexa "
                "Voice Remote... it is brilliant. I love technology!",
    },
    {
        "label": "Vague / ambiguous",
        "text": "How long did the priority email take to come for the tour? I've still "
                "not had mine.",
    },
]


class AnalyzeRequest(BaseModel):
    message: str
    modes: list[str] = ["trivial", "simple", "llm"]


def _result_to_dict(result) -> dict:
    d = dataclasses.asdict(result)
    d.pop("usage", None)
    return d


@app.get("/api/health")
def health():
    return {"status": "ok", "llm_configured": config.has_llm(), "brand": config.BRAND}


@app.get("/api/examples")
def get_examples():
    return EXAMPLES


@app.get("/api/intents")
def get_intents():
    return [
        {"id": i.id, "label": i.label, "description": i.description, "risk": i.risk}
        for i in config.INTENTS
    ]


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    text = req.message.strip()
    if not text:
        raise HTTPException(400, "message must not be empty")
    if len(text) > 500:
        raise HTTPException(400, "message too long (max 500 chars for this demo)")

    pipeline = get_pipeline()
    out: dict = {}

    if "trivial" in req.modes:
        out["trivial"] = _result_to_dict(pipeline.run_trivial(text))
    if "simple" in req.modes:
        out["simple"] = _result_to_dict(pipeline.run_simple(text))
    if "llm" in req.modes:
        if not config.has_llm():
            out["llm"] = {"error": "GROQ_API_KEY not configured on the server"}
        else:
            try:
                out["llm"] = _result_to_dict(pipeline.run_llm(text))
            except Exception as e:  # noqa: BLE001
                out["llm"] = {"error": str(e)}
    return out


@app.get("/api/metrics")
def get_metrics():
    path = config.RESULTS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "metrics.json not found -- run scripts/08_evaluate.py")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/judge-summary")
def judge_summary():
    path = config.RESULTS_DIR / "judge_scores.csv"
    if not path.exists():
        raise HTTPException(404, "judge_scores.csv not found -- run scripts/09_llm_judge.py")
    df = pd.read_csv(path)
    dims = ["grounding_score", "helpfulness_score", "tone_score", "clarity_score", "overall_score"]
    summary = {}
    for mode, g in df.groupby("mode"):
        summary[mode] = {"n": len(g), **{d: float(g[d].mean()) for d in dims}}
    return summary


@app.get("/api/judge-agreement")
def judge_agreement():
    path = config.RESULTS_DIR / "judge_human_agreement.json"
    if not path.exists():
        raise HTTPException(404, "judge_human_agreement.json not found -- run scripts/10_judge_agreement.py")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/golden-sample")
def golden_sample(n: int = 12):
    df = pd.read_csv(config.GOLDEN_SET_PATH)
    n = max(1, min(n, len(df)))
    sample = df.sample(n=n, random_state=None)
    return sample[["id", "customer_text", "gold_intent", "gold_escalate", "gold_escalation_reason"]].to_dict("records")


@app.get("/api/failure-examples")
def failure_examples(mode: str = "llm", n: int = 6):
    preds_path = config.RESULTS_DIR / f"predictions_{'llm' if mode == 'llm' else 'baselines'}.csv"
    if not preds_path.exists():
        raise HTTPException(404, f"{preds_path.name} not found")
    preds = pd.read_csv(preds_path)
    if mode != "llm":
        preds = preds[preds["mode"] == mode]
    golden = pd.read_csv(config.GOLDEN_SET_PATH)
    merged = preds.merge(golden, on="id", suffixes=("", "_gold"))
    wrong_intent = merged[merged["pred_intent"] != merged["gold_intent"]]
    wrong_escalate = merged[merged["pred_escalate"] != merged["gold_escalate"]]
    cols = ["id", "customer_text", "pred_intent", "gold_intent", "pred_escalate",
            "gold_escalate", "gold_escalation_reason", "drafted_reply"]
    return {
        "wrong_intent": wrong_intent[cols].head(n).to_dict("records"),
        "wrong_escalate": wrong_escalate[cols].head(n).to_dict("records"),
    }


# Static frontend last so /api/* routes above take precedence.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))
