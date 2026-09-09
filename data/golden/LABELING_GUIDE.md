# Golden Evaluation Set — Sampling & Labeling Guide

## What this set is

200 hand-labeled examples of real, first-contact AmazonHelp customer tweets, used as
the held-out ground truth for every metric in this repo. Nothing in this set is used
to train the classifier, build the retrieval knowledge base, or tune thresholds.

## Sampling procedure

1. Started from 76,799 reconstructed (customer's first tweet -> AmazonHelp's first
   direct reply) pairs (`src/agent/thread_builder.py`), out of ~2.81M raw rows in the
   Kaggle "Customer Support on Twitter" dataset.
2. Took a random 45,000-pair subsample (fixed seed 42) and filtered to English-language
   messages (langdetect) between 15-320 characters -> 33,816 clean pairs.
3. Shuffled (seed 42) and set aside the **first 550** as "golden candidates" **before**
   building the retrieval KB (9,000 pairs) or the classifier training pool (6,000
   pairs) from the remainder. This ordering is what guarantees zero leakage: the
   golden set was carved out first, everything else was built from what was left over.
4. Drew a **stratified** (not proportional) sample of 200 from those 550 candidates,
   across 8 keyword-rule "weak intent" buckets (`src/agent/intent_rules.py`), with
   target counts of 45/30/25/25/20/20/20/15 (`scripts/03_select_golden_sample.py`).
   Rare categories (e.g. `account_access`) are deliberately over-represented relative
   to their true frequency (~4% of the full pool) so each has enough examples (15+)
   for a meaningful per-class precision/recall estimate. The true, skewed class
   frequency is reported separately in REPORT.md from the full 33.8k pool — this
   sample is for *evaluation power*, not for describing real-world prevalence.
5. The weak-intent label only decided which bucket a candidate was drawn from, so the
   reviewer would see a topically diverse mix — it was never copied into the gold
   label (step below re-reads every message from scratch).

## Labeling procedure

Each of the 200 sampled tweets was read individually — together with AmazonHelp's
actual historical reply, for resolution context only — and assigned:

| Field                    | Meaning |
|---------------------------|---------|
| `gold_intent`              | One of the 8 taxonomy categories (`src/agent/config.py`), judged from the customer's message alone. |
| `gold_escalate`            | `True`/`False` — see escalation policy below. |
| `gold_escalation_reason`   | One-line justification for that call. |
| `notes`                    | What a correct/helpful reply needs to contain — used later during failure analysis and to sanity-check the LLM-as-judge's scores. |

### Escalation policy applied while labeling

Label `gold_escalate = True` if **any** of the following hold, else `False`:

- **Money at risk**: unauthorized/incorrect charge, refund dispute, or anything beyond
  a routine status question with financial exposure.
- **Explicit escalation ask or high frustration**: supervisor/callback/legal threats,
  or strong repeated-contact frustration ("still waiting", "3rd time", etc.).
- **Unverifiable-by-bot account action with real consequences**: e.g. unlocking or
  closing an account, where a wrong automated action is costly and identity can't be
  confirmed over public Twitter.
- **Too ambiguous/unintelligible to safely act on.**

Routine status/how-to/informational requests and compliments are `False`.

## Who labeled it, and how

Labeled by the assignment author in a single pass, using Claude Code as an AI
reading/drafting assistant: each tweet was read against the rubric above and a label
was drafted and then reviewed before being written to `golden_eval_set.csv`. This is
disclosed per the assignment's own rules ("You may use AI coding assistants freely...
not knowing what you borrowed is not [fine]").

This means the labels reflect **one** annotator's judgment calls — there is no
inter-annotator agreement number for the gold labels themselves. See REPORT.md,
"What's misleading about my headline number," for what this does and doesn't let us
claim, and `eval/results/judge_human_agreement.csv` for the separate check of whether
the LLM-as-judge (which grades reply *quality*, a different task from this labeling)
agrees with a manual re-grade.

## Known limitations of this set

- English-only (AmazonHelp serves many locales; multilingual support is out of scope —
  see DECISION_LOG.md).
- First-contact messages only — no multi-turn dialogue state (a customer's 2nd/3rd
  reply in a thread is a different, harder labeling problem: distinguishing "new
  issue" from "answering the agent's follow-up question").
- Single labeler, single labeling pass.
- Drawn from a brand whose Twitter support is unusually resolution-heavy on the public
  timeline (only ~0.7% of replies deflect to DM) — conclusions about "how much can be
  grounded from public data" may not transfer to brands that resolve mostly in DMs.
