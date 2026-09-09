# Hiver AI Support Agent — AmazonHelp Triage

An AI support-triage agent for **AmazonHelp**'s Twitter customer support, built for the
Hiver SDE Intern take-home assignment. Given an incoming customer message, it:

1. **Classifies intent** into one of 8 categories derived from reading real AmazonHelp threads.
2. **Drafts a reply** grounded in how AmazonHelp has actually resolved similar issues before
   (retrieval-augmented generation over ~9,000 historical resolutions).
3. **Decides auto-handle vs. escalate to a human**, with a stated, auditable reason.

Three system tiers are built and compared head-to-head: a **trivial** baseline, a **simple**
classical-ML baseline, and the **headline** LLM+RAG system — see [`REPORT.md`](REPORT.md) for the
full write-up (problem framing, baselines, failure analysis, and what's misleading about the
headline number) and [`DECISION_LOG.md`](DECISION_LOG.md) for the 21 non-obvious decisions behind it.

A live interactive demo (try messages yourself, browse the evaluation dashboard, inspect real
failure cases) ships with the repo — see [Web demo](#web-demo) below.

---

## TL;DR results

Automated metrics against a 200-example hand-labeled golden set (never used for training or
retrieval grounding). LLM-as-judge reply-quality scores are 1-5, independently graded by a
larger/different model than the one that drafts replies.

| System               | Intent accuracy | Intent macro-F1 | Escalation F1 | Escalation recall | ROUGE-L vs. historical reply |
|----------------------|:---------------:|:---------------:|:-------------:|:------------------:|:-----------------------------:|
| Trivial baseline      | 38.5%           | 0.069           | 0.000         | 0.000               | 0.083 |
| Simple (TF-IDF+NB+retrieval) | 60.5%    | 0.545           | 0.523         | 0.600               | 0.159 |
| **Headline (LLM + RAG)** | **71.5%**    | **0.693**       | 0.465         | 0.508               | **0.171** |

LLM-as-judge and judge/human-agreement numbers are filled in from
`eval/results/judge_scores.csv` and `eval/results/judge_human_agreement.json` once
`scripts/09_llm_judge.py` / `10_judge_agreement.py` have run (see [REPORT.md](REPORT.md) for the
committed, full-run numbers — this table only tracks what's cheap to regenerate live).

**The one number that will surprise you**: the headline system has *better* intent accuracy but
*worse* escalation recall than the simple baseline. Improving the classifier doesn't automatically
improve the business metric you actually care about, because the escalation policy is coupled to
specific intent labels — see REPORT.md, "What's misleading about my headline number."

---

## Quick start — reproduce in under 15 minutes

**Prerequisites**: Python 3.10+, ~1GB free disk (dataset cache), internet access. No Kaggle
account needed (anonymous public-dataset download). A Groq API key is optional but recommended
— see below.

```bash
git clone <this-repo-url>
cd Hiver-Assessment

python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

> **Windows + OneDrive-synced folder note**: if `pip install` fails with `ImportError: DLL load
> failed ... Application Control policy has blocked this file`, your machine's security policy is
> blocking native extensions written into a synced folder. Create the venv *outside* the OneDrive
> path instead (e.g. `python -m venv %USERPROFILE%\.venvs\hiver-assessment`) and point your shell
> at that interpreter — the project code itself doesn't need to move. See DECISION_LOG.md #8 for
> why the classical-ML pieces avoid scikit-learn/scipy entirely as a result.

**(Optional, recommended) Configure an LLM provider:**

```bash
cp .env.example .env
# edit .env and set GROQ_API_KEY=... (https://console.groq.com — free tier, no card required)
```

Without a key, everything still runs — the trivial and simple baselines are pure numpy/rule-based
(no API calls), and the LLM-headline numbers are read from the full run already committed under
`eval/results/`.

**Run the whole pipeline with one command:**

```bash
python scripts/run_pipeline.py
```

This rebuilds the dataset, trains the classifier + retrieval index, runs both baselines, runs a
**real, live** small sample against the Groq API (proving the integration genuinely works, not
mocked), scores it with the LLM judge, and prints/writes final metrics — **typically 5-8 minutes**,
comfortably under the 15-minute budget. It reports the full 200-example headline numbers from the
committed `eval/results/` artifacts (see below for why a *live* full run doesn't fit 15 minutes).

To reproduce those artifacts **live from scratch** yourself (takes 40-90 minutes due to Groq's
free-tier throughput — see [DECISION_LOG.md](DECISION_LOG.md) #16):

```bash
python scripts/run_pipeline.py --full
```

**Launch the interactive demo:**

```bash
uvicorn webapp.main:app --reload --port 8000
# open http://127.0.0.1:8000
```

---

## Web demo

A small FastAPI + vanilla-JS app (`webapp/`) that runs the real pipeline, not mock data:

- **Try it live** — type or click a preset customer message; see the trivial, simple, and
  LLM+RAG systems' intent classification, confidence, drafted reply, escalation decision +
  reason, and the actual historical cases each reply is grounded in, side by side.
- **Evaluation dashboard** — live charts over `eval/results/metrics.json` and `judge_scores.csv`:
  intent-F1 / escalation-F1 comparison, an intent confusion matrix, and LLM-judge quality scores
  by system tier.
- **Failure analysis** — real LLM-vs-gold-label disagreements pulled live from the evaluation
  artifacts, not cherry-picked screenshots.

Runs entirely on your machine; the only network call it makes is to the Groq API when you tick
the "LLM + RAG" mode in the live tester.

---

## Repository structure

```
src/agent/            Core library (installed editable via pyproject.toml)
  config.py             Brand, intent taxonomy, thresholds, model names -- single source of truth
  text_utils.py         Cleaning/tokenizing (no compiled deps)
  nlp_model.py          TF-IDF vectorizer + Multinomial Naive Bayes, pure numpy
  metrics.py            Precision/recall/F1/kappa/ROUGE-L, pure numpy
  data_ingest.py        Kaggle dataset download + load
  thread_builder.py     Raw tweets -> (customer message, brand reply) pairs
  intent_rules.py       Weak-supervision keyword rules (training signal, not gold)
  retrieval.py          TF-IDF nearest-neighbor grounding index
  llm_client.py         Rate-limited Groq API client, structured JSON output
  agent.py              Orchestrator: trivial / simple / llm run modes
  escalation.py         Deterministic, auditable escalation rule engine
  judge.py              LLM-as-judge rubric scorer

scripts/               Numbered pipeline stages (run via run_pipeline.py, or individually)
data/
  golden/                 golden_eval_set.csv (the 200 hand-labeled examples) + LABELING_GUIDE.md
  processed/               Intermediate pipeline artifacts (gitignored raw data, committed small ones)
eval/results/           Committed metrics, predictions, judge scores, charts (the evidence)
webapp/                 FastAPI backend + static frontend for the live demo
tests/                  47 unit tests (pytest) for the numpy ML code, rules, and escalation logic
REPORT.md               Problem framing, baselines, failure analysis, what's misleading, next steps
DECISION_LOG.md         21 non-obvious decisions and why
```

---

## The agent, in three tiers

| | Trivial | Simple | Headline (LLM + RAG) |
|---|---|---|---|
| Intent | majority class (constant) | TF-IDF + Naive Bayes | LLM (structured JSON output), classifier as fallback/cross-check |
| Reply | one fixed canned message | nearest-neighbor historical reply, redacted for reuse | LLM-drafted, conditioned on top-3 retrieved historical resolutions |
| Escalation | never | rule engine (see below) | **same rule engine** — the LLM's own opinion is logged for comparison, never authoritative |
| API calls | none | none | Groq (`openai/gpt-oss-20b` draft, `openai/gpt-oss-120b` judge) |

**Escalation is a deterministic rule engine, not an LLM judgment call** (`src/agent/escalation.py`):
a message escalates if its intent is in a policy-flagged high-risk category (billing, explicit
complaints, unclear), if classifier confidence is below threshold, if no sufficiently similar
historical resolution was found, or if urgency/frustration/safety language is detected. This is a
deliberate choice — see DECISION_LOG.md #14 — an escalation gate has to be auditable and
consistent, which an LLM's judgment isn't guaranteed to be run-to-run.

## Data

[Kaggle "Customer Support on Twitter"](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
(thoughtvector), ~2.8M tweets. `scripts/02_build_dataset.py` reconstructs ~77k first-contact
(customer's opening tweet → AmazonHelp's first direct reply) pairs, filters to English messages
15-320 characters, and weak-labels intent via keyword rules — a random 45,000-pair subsample is
used throughout, per the assignment's explicit "subsample expected and encouraged" note. See
DECISION_LOG.md for why AmazonHelp was chosen over other brands in the dataset.

## Golden evaluation set

`data/golden/golden_eval_set.csv` — **200 hand-labeled examples**, stratified across the 8
intents (so rare categories still get a meaningful per-class score), carved out of the pool
*before* the retrieval knowledge base or classifier training pool were built, so there is zero
leakage. Full sampling and labeling methodology, including the exact escalation policy applied
during labeling, is in [`data/golden/LABELING_GUIDE.md`](data/golden/LABELING_GUIDE.md).

## Evaluation harness

- **Automated metrics** (`scripts/08_evaluate.py`, pure numpy — see `src/agent/metrics.py`):
  per-class precision/recall/F1 and confusion matrix for intent; precision/recall/F1 for the
  binary escalate/auto-handle decision; ROUGE-L and unigram-overlap-F1 against the brand's actual
  historical reply as a cheap, non-LLM reply-quality proxy.
- **LLM-as-judge** (`scripts/09_llm_judge.py`, `src/agent/judge.py`): a 1-5 rubric across
  grounding, helpfulness, tone, and clarity, graded by a different (larger) model than the one
  that drafted the reply.
- **Judge-human agreement** (`scripts/10_judge_agreement.py`): a subsample independently re-graded
  by hand on the same rubric, with mean absolute difference, % within ±1 point, and Spearman
  correlation reported against the LLM judge — see REPORT.md for the actual numbers and honest
  discussion of what they do and don't prove.

## Configuration

All tunable values (brand, intent taxonomy, escalation thresholds, model names, retrieval sizes)
live in `src/agent/config.py`. LLM provider config is via `.env` (see `.env.example`):

```
GROQ_API_KEY=...
LLM_AGENT_MODEL=openai/gpt-oss-20b     # optional override
LLM_JUDGE_MODEL=openai/gpt-oss-120b    # optional override
```

## Testing

```bash
python -m pytest tests/ -v
```

47 unit tests covering the TF-IDF/Naive Bayes implementation (verified against hand-computed
values), text cleaning, weak-labeling rules, the escalation rule engine, thread reconstruction,
and all custom metrics.

## Known limitations

- English-only; first-contact messages only (no multi-turn dialogue state) — see REPORT.md and
  DECISION_LOG.md for the reasoning and what a follow-up version would add.
- TF-IDF retrieval, not dense embeddings (a deliberate dependency-footprint trade-off — see
  DECISION_LOG.md #10).
- Single-annotator golden labels; the LLM judge and the model that drafts replies share a
  provider (Groq) and model family, which can inflate agreement relative to a fully independent
  judge — both caveats are discussed explicitly in REPORT.md.

## Citations / borrowed material

- Dataset: [Kaggle "Customer Support on Twitter"](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (thoughtvector), downloaded via [`kagglehub`](https://github.com/Kaggle/kagglehub)'s anonymous public-dataset access.
- LLM inference: [Groq API](https://console.groq.com) (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`).
- Frontend chart library: [Chart.js](https://www.chartjs.org/) via CDN.
- Chart color palette and dashboard layout method: this session's `dataviz` skill reference palette (see `webapp/static/styles.css` header comment for the source values).
- No other code, prompts, or datasets were copied from external sources; all `src/agent/*` and `webapp/*` code was written for this assignment.
- Built with Claude Code (Anthropic) as an AI coding assistant, per the assignment's explicit rules allowing this.
