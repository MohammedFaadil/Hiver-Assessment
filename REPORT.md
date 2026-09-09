# Report: AI Support-Triage Agent for AmazonHelp

## 1. Problem framing

### What "good" means for this brand

AmazonHelp handles first-contact customer messages spanning eight recurring needs (derived by
reading ~200 real threads and confirmed against a 33.8k-message pool — see
[`LABELING_GUIDE.md`](data/golden/LABELING_GUIDE.md)): order/delivery status, returns/refunds,
billing disputes, account access, product/technical questions, complaints/escalation requests,
compliments, and a residual unclear bucket. For a brand at this volume, "good" is not "the AI
answers everything" — it's:

1. **Never confidently wrong on money or safety.** A wrong refund promise or a dismissed safety
   report (a fire-hazard report showed up in the golden set — id 71) is far more costly than an
   extra human handoff.
2. **Auto-handle the boring majority safely.** ~38% of first-contact messages are routine
   order/delivery status questions with a well-established resolution pattern; these should not
   consume human agent time.
3. **Escalate with a reason a human can act on immediately**, not a black-box flag.
4. **Never invent facts** — order numbers, refund amounts, or policies not evidenced by the
   customer's message or a genuinely similar historical case.

### What I chose not to build

- **Multi-turn dialogue state.** The agent triages a customer's *opening* message. A 3rd-turn
  reply ("here's my order number") is a different, harder problem — distinguishing "new issue"
  from "answering the agent's question" needs conversation-state tracking, not just message
  classification. Scoping to first-contact keeps "classify this incoming message" well-defined
  and covers the highest-volume triage decision (what happens the moment a ticket arrives).
- **Multilingual support.** AmazonHelp serves many locales (Japanese, Spanish, German, and
  Hindi-English code-switching all appear in the raw data); non-English messages are routed to
  `unclear_other` and escalated. This is a real, disclosed limitation, not silently dropped data.
- **Fine-tuning anything.** Everything here is zero/few-shot (retrieval-grounded prompting) or a
  from-scratch classical classifier trained on weak labels. Fine-tuning an LLM was out of scope
  for the time budget and would fight the "reproduce in 15 minutes" constraint directly.
- **Dense/embedding retrieval.** TF-IDF cosine similarity, not sentence embeddings — a deliberate
  dependency-footprint trade documented in `DECISION_LOG.md` #10.

## 2. Results vs. two baselines

All numbers are on the 200-example hand-labeled golden set (`data/golden/golden_eval_set.csv`),
which was carved out before the retrieval knowledge base or classifier training pool existed —
zero leakage. Full methodology in `LABELING_GUIDE.md`.

| Metric | Trivial | Simple (TF-IDF+NB+retrieval) | **Headline (LLM+RAG)** |
|---|---:|---:|---:|
| Intent accuracy | 38.5% | 60.5% | **71.5%** |
| Intent macro-F1 | 0.069 | 0.545 | **0.693** |
| Escalation F1 | 0.000 | **0.523** | 0.465 |
| Escalation precision | 0.000 | 0.464 | 0.429 |
| Escalation recall | 0.000 | **0.600** | 0.508 |
| ROUGE-L vs. historical reply | 0.083 | 0.159 | **0.171** |

- **Trivial** = majority-class intent (`order_delivery`), one fixed canned reply, never escalates.
- **Simple** = TF-IDF + Multinomial Naive Bayes (trained on weak/keyword-rule labels, evaluated
  against hand labels) + nearest-neighbor retrieval-templated reply + the same rule-based
  escalation engine as the headline system.
- **Headline** = same retrieval grounding, reply drafted by Groq (`openai/gpt-oss-20b`)
  conditioned on the top-3 retrieved historical resolutions; escalation decided by the identical
  deterministic rule engine (see `src/agent/escalation.py`) — the LLM's own escalation opinion is
  logged but never authoritative.

**Bootstrap validation (5,000 resamples of the 200 golden examples, fixed seed):** the intent
accuracy gain (headline vs. simple) is statistically solid — mean +11.0pp, 95% CI [+3.5pp,
+18.5pp], present in 99.7% of resamples. The escalation F1 *regression* is directionally
consistent (present in 94.4% of resamples) but its 95% CI on the difference, [-0.132, +0.013],
touches zero — with only 65 escalate-positive examples in the golden set, I can't claim this is
definitively real at conventional significance. See §4.

**LLM-as-judge reply-quality scores** (1-5, graded by `openai/gpt-oss-120b`, a different/larger
model than the drafting model), full raw data in `eval/results/judge_scores.csv`:

| Dimension | Simple (n=200) | **Headline LLM+RAG (n=120)** |
|---|---:|---:|
| Grounding | 3.08 | **4.60** |
| Helpfulness | 2.50 | **4.40** |
| Tone | 3.30 | **4.70** |
| Clarity | 3.94 | **4.82** |
| **Overall** | **2.82** | **4.53** |

The gap is large and consistent across every dimension, not just the overall score — the simple
baseline's biggest weakness is `helpfulness` (2.50/5): reusing a historical reply verbatim
frequently answers a *similar* past case rather than *this* customer's specific situation, which
the judge penalizes heavily and consistently. (The LLM-mode judge run scored 120 of the golden
set's 200 examples rather than all 200 — Groq's free-tier throughput, discussed in §4, makes
judging 400 replies with a large model a genuinely long-running batch job; 120 is already a large,
stable sample, not a preliminary one, and the full run continues in the background.)

Judge-vs-human agreement was checked on a fixed, hand-graded sample of 15 simple-mode replies
(chosen before the full LLM-mode run was judged, so it isn't cherry-picked toward either system):
**Spearman ρ = 0.95, 100%
of scores within ±1 point, 60% exact match** — the judge tracks relative reply quality very well
and is systematically about half a point more generous on borderline-good replies, essentially
perfect on clear failures.

## 3. Failure analysis: top 5 modes, with real examples

**1. Tone overrides topic for angry-but-routine messages.**
*Example (id 5):* "Thanks for ruining my 3 year old's Christmas present by delivering her Barbie
house like this" — gold intent is `order_delivery` (a damaged-item report with a clear resolution
path); the LLM predicted `complaint_escalation` and flagged escalation, purely from the sarcastic/
angry framing. *Hypothesis:* the model (and, to be fair, the weak-label training signal it's
partly cross-checked against) over-weights sentiment words. *Fix direction:* the escalation rule
engine already separates "is this a complaint" from "does the customer need a human" via the
urgency lexicon and confidence checks — but intent classification itself has no such separation.

**2. Retrieval grabs topically-similar but sentiment-mismatched precedent.**
*Example (id 57):* "MY MAGAZINE JUST ARRIVED ALONG WITH THE CD THEY GAVE ME A FREE HANDBAG...
THANK YOU" — a compliment. The retrieved "similar" precedent was a damaged-item complaint (shared
vocabulary: delivery-adjacent words), so the drafted reply opens with "I'm sorry for the
experience with the delivery" to a happy customer. *Hypothesis:* TF-IDF matches surface words, not
sentiment; a positive message with delivery vocabulary retrieves negative delivery precedent.
*Fix direction:* filter retrieval candidates by a cheap sentiment signal before the topic match,
or weight the compliment/complaint distinction into the vector itself.

**3. Billing concepts get conflated.**
*Example (id 73):* "it's coming up on my bank account that I've been charged twice" (a genuine
double-charge complaint) drew a reply about a routine "£1 authorization hold" — a real but
*different* billing concept from a different historical case. This is the most concerning failure
mode found: it's not just unhelpful, it's a factually confident, wrong reassurance about money.
*Hypothesis:* "billing_payment" is one intent bucket covering several distinct sub-issues (auth
holds, double charges, refund timing, gift cards) with surface-level vocabulary overlap.
*Fix direction:* split `billing_payment` into finer sub-intents, or require the retrieval step to
match on a normalized "billing sub-topic" rather than raw text.

**4. Improving the classifier can *hurt* the business metric.** (Detailed in §4 — the headline
system's better intent accuracy comes with *worse* escalation recall than the simple baseline,
because the escalation policy is categorically coupled to specific intent labels.)

**5. Reused/retrieved text can leak another customer's name.**
*Examples (ids 26, 52, 77):* replies addressed the current customer as "Harmani," "Sarah," and
"Andrew" respectively — names pulled verbatim from the *retrieved historical reply*, not the
current customer's message. `redact_for_reuse()` strips URLs, leading @mentions, and agent
sign-off initials, but not names embedded mid-reply, and the LLM prompt doesn't explicitly forbid
this either. *Fix direction (concrete, not yet implemented — see §5):* add "never address the
customer by a name that does not appear in their own message" to the agent system prompt, and
audit whether it's worth a cheap regex/NER pass over the retrieval KB itself.

## 4. What's misleading about my headline number

**"71.5% intent accuracy, headline system" hides a real trade-off, not just a caveat.** The
LLM+RAG system beats the simple baseline on intent accuracy by a statistically solid +11pp — but
its escalation F1 is *lower* (0.465 vs. 0.523), and escalation recall notably lower (0.508 vs.
0.600). This isn't noise in the intuitive sense: it happens because the escalation rule engine
categorically escalates three "high-risk" intents (`billing_payment`, `complaint_escalation`,
`unclear_other`). When the more-accurate LLM reclassifies a message *out* of one of those buckets
and into a more specific, lower-risk one, it can lose an escalation that the simple baseline's
noisier classifier "accidentally" caught. **A more accurate classifier is not automatically a
better triage system** when downstream policy depends on the classifier's exact category
boundaries. I did not fix this before writing the report, because the honest fix requires deciding
whether the escalation policy should trigger on *finer-grained risk signals* than the 8-way
intent label — a design change, not a bug fix (see §5).

**The reported effect size for the escalation regression is not fully nailed down.** A 5,000-
resample bootstrap over the 200 golden examples puts the escalation-F1 difference's 95% CI at
[-0.132, +0.013] — it touches zero. 94.4% of resamples favor the simple baseline, which is
suggestive, not proof, with only 65 escalate-positive examples in a 200-row golden set. I'm
reporting the direction because I believe the causal story above, not because the interval alone
would meet a formal significance bar.

**The full-200-example LLM numbers are not what a live 15-minute reproduction run computes.**
Groq's free tier measured out to ~8,000 tokens/minute against this account (via response headers,
not blog-post guesses) — a live run over all 200 examples takes ~25-40 minutes for drafting and
~40-70 for judging. `scripts/run_pipeline.py`'s default path runs a small *live* smoke-test slice
(proving the real API integration works) and reports the headline numbers from a full run executed
once and committed to `eval/results/`. This is disclosed, not hidden, but it does mean "run the
quick-start command" and "regenerate exactly these numbers from a cold start" are not the same
claim — only `--full` mode is the latter, and it takes 40-90 minutes.

**The golden set has one labeler and no inter-annotator agreement number for the labels
themselves.** I have evidence for how well the *LLM judge* agrees with a human re-grader (§2), but
not for how well a second independent human would agree with my own `gold_intent` /
`gold_escalate` calls. Ambiguous cases exist — e.g. id 56 ("Totally ridiculous n disgusting
customer response, i request never purchase anything from [competitor]. [Amazon] is best
eCommerce site") is genuinely a backhanded compliment about Amazon via comparison to a rival, and
a different labeler might reasonably read it differently on a fast pass.

**The LLM judge and the LLM that drafts replies share a provider and model family** (Groq's
`gpt-oss-20b`/`120b`). Using a larger, different-size model for judging is a real mitigation, not
a full fix — a same-family judge may still be more forgiving of same-family stylistic quirks than
a fully independent model or a human would be across the board (the human-agreement check in §2
addresses calibration on individual scores, not this systemic-bias question).

## 5. What I'd do next with one more week

1. **Decouple escalation risk from the 8-way intent label.** Add a lightweight secondary
   classifier (or a few more regex signals) for "financial amount mentioned," "repeated-contact
   language," and "safety/security keyword" directly, so escalation isn't solely gated by which
   of 8 buckets the primary intent classifier chose (addresses §3.4/§4 directly).
2. **Split `billing_payment` into sub-intents** (auth holds vs. double charges vs. refund timing
   vs. gift cards/vouchers) to fix the conflation failure in §3.3, and re-measure.
3. **Add an explicit no-name-fabrication instruction to the agent prompt**, re-run the full
   200-example LLM pass, and check whether the id-26/52/77-style failures in §3.5 actually drop —
   this is a one-line prompt change but needs the full re-run + re-judge to validate, which is why
   it's "next week" and not "already done."
4. **A second labeler on a 50-example overlap subset of the golden set**, to get a real
   inter-annotator agreement number for the gold labels themselves (currently the biggest
   unaddressed gap flagged in §4).
5. **Sentiment-aware retrieval** (even a cheap lexicon-based positive/negative score used as a
   retrieval filter) to fix the compliment-retrieves-complaint failure in §3.2.
6. **Try dense embeddings for retrieval** (behind a config flag, `sentence-transformers` or
   similar) and measure whether it meaningfully improves grounding quality over TF-IDF, now that
   the dependency-footprint concern (`DECISION_LOG.md` #10) could be evaluated empirically rather
   than assumed.
7. **A held-out temporal split** (train/retrieve on earlier tweets, evaluate on later ones) to
   check whether performance holds up over time, not just over a random split of the same period.
