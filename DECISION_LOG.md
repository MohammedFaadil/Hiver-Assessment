# Decision Log

Non-obvious decisions made while building this, and why. Chronological-ish, grouped by
area. See `REPORT.md` for the fuller narrative and `LABELING_GUIDE.md` for the golden-set
methodology specifically.

## Scope

1. **Brand: AmazonHelp, not Apple/Delta/etc.** It's the largest brand in the dataset
   (169,840 outbound tweets) and, critically, only ~0.7% of its replies deflect to DM —
   AppleSupport by contrast resolves almost everything via "please DM us," which would
   have starved the retrieval-grounding knowledge base of visible resolutions. AmazonHelp
   also maps directly onto the e-commerce/retail support domain, which is closer to a
   typical Hiver customer than an airline or telecom brand would be.

2. **Scoped to first-contact messages only, not full multi-turn threads.** "Classify this
   incoming message" is well-defined for a customer's opening tweet. A 3rd-turn reply
   ("here's my order number") is a different problem — distinguishing "new issue" from
   "answering the agent's prior question" — that needs dialogue-state tracking, which is
   out of scope. See REPORT.md "what I chose not to build."

3. **English-only.** AmazonHelp serves many locales (saw Japanese, Spanish, German,
   Hindi-English code-switching in the raw data); evaluating multilingual quality would
   roughly double the scope. Non-English messages are treated as `unclear_other` /
   auto-escalated, which is itself a real, disclosed limitation (see golden example with
   the German-language tweet, id 189 in the eval set).

## Data pipeline

4. **Subsampled 45,000 candidate pairs before filtering, not the full ~77k.** The
   assignment explicitly expects and encourages subsampling; capping the input keeps
   `scripts/02_build_dataset.py` fast (~2.5 min including a fresh Kaggle download) without
   materially narrowing intent coverage — the resulting 33.8k-row clean pool still yields
   9,000+ retrieval-KB rows and enough per-intent volume for classifier training.

5. **Golden set carved out *before* building the retrieval KB / train pool, from the same
   shuffled pool.** This is what guarantees zero leakage: the golden candidates are the
   first slice after a fixed-seed shuffle, and the KB/train pool are sampled only from the
   remainder. Order of operations matters here, not just "no exact duplicate rows."

6. **Weak (keyword-rule) labels for classifier training; hand labels only for
   evaluation.** There's no ground truth in the raw data. Rather than hand-labeling
   thousands of training examples, `intent_rules.py` applies precision-oriented regex
   rules to build a noisy training pool, and the classifier is judged purely against the
   independently hand-labeled golden set. Weak-label accuracy vs. the gold set is itself
   reported (62%) as a measure of how noisy the training signal actually is.

7. **Caught and fixed a real regex bug via data auditing, not code review.** The first
   version of the weak-labeling rules anchored a trailing `\b` directly after literal verb
   stems (e.g. `\border\b`, `charge(d)?`), which silently fails to match their own common
   inflections — `\border\b` does not match "ordered" because `\b` needs a word boundary
   immediately after "order," and "ordered" has "e" there instead. This made
   `unclear_other` ~35% of the pool. Rewriting stems as `order\w*` etc. dropped it to
   ~23%. Lesson kept for next time: audit a random sample of the catch-all bucket, don't
   just trust that "some fraction will be genuinely unclear" without checking why.

## Modeling

8. **No scikit-learn / scipy.** On the machine this was built on, a Windows Application
   Control policy blocks their compiled DLLs specifically (numpy, pandas, and matplotlib's
   compiled bits were unaffected — this isn't a blanket "block everything" policy, which
   made it hard to predict in advance). Rather than fight a local security policy, the
   classical "simple baseline" — TF-IDF vectorization, Multinomial Naive Bayes, cosine
   retrieval — is implemented directly in `src/agent/nlp_model.py` (~150 lines, verified
   against hand-computed examples). This also means the repo has a smaller, more
   auditable dependency footprint, which is a reasonable trade independent of the original
   reason.

9. **Multinomial Naive Bayes over logistic regression for the simple baseline.** NB has a
   closed-form fit (no gradient descent / learning-rate tuning needed), is a standard,
   respectable text-classification baseline, and is trivial to explain and re-derive live
   if asked. Softmax regression would need an optimizer loop reimplemented in numpy too,
   for no clear accuracy win at this scale.

10. **TF-IDF retrieval, not sentence embeddings.** Embeddings (e.g. sentence-transformers)
    would likely improve semantic retrieval quality, but pull in `torch` — a large,
    slow-to-install, compiled-extension-heavy dependency that risks both the 15-minute
    reproduction budget and the same DLL-blocking issue that ruled out scipy. TF-IDF
    cosine similarity is fast, dependency-light, and the retrieval sanity checks
    (`scripts/05_train_classifier.py`) show it finds genuinely relevant precedent.
    Documented as a clear "what I'd try next" item.

## LLM integration (Groq)

11. **Groq, via a user-supplied key, using `openai/gpt-oss-20b` (drafting) and
    `openai/gpt-oss-120b` (judging) — not Llama-3.x.** A generic web search suggested
    `llama-3.3-70b-versatile` as Groq's flagship, but querying this account's actual
    `/v1/models` endpoint showed Llama 3.x is not in the active model list at all — Groq's
    lineup changes and secondhand sources go stale. Always verify against the live API,
    not search results, before hardcoding a model id.

12. **`reasoning_effort: "low"` + `include_reasoning: false` on every call.** The gpt-oss
    models emit hidden chain-of-thought as separate "reasoning tokens" *before* content.
    A first test with `max_tokens=5` came back with empty `content` because the entire
    budget was consumed by reasoning — this is a real failure mode, not a hypothetical
    one. Low reasoning effort keeps latency and token spend down for a task (classify +
    template-fill) that doesn't need multi-step reasoning.

13. **Structured output via `json_schema` strict mode, not prompt-and-hope.** Both
    gpt-oss sizes support strict schema-constrained decoding on Groq, which eliminates a
    whole class of "model returned prose instead of JSON" failures and constrains
    `intent` to exactly the 8 taxonomy values via an enum, rather than relying on
    string-matching a free-text label back to the taxonomy.

14. **Escalation is decided by a deterministic rule engine, never by the LLM.** The LLM's
    own `needs_escalation` opinion is logged for comparison but never used for the actual
    decision (`src/agent/escalation.py`). Escalation gating is a business-policy decision
    that has to be auditable, explainable, and consistent run-to-run — an LLM's judgment
    call on that isn't guaranteed to be any of those three. The rule engine composes:
    intent risk category, classifier confidence, retrieval-similarity strength, and a
    small urgency/frustration/safety lexicon.

15. **Gold `escalate` labels were assigned by independent judgment, not derived from the
    rule engine's own logic — on purpose, to avoid grading my own homework.** Early in
    labeling it was tempting to just say "high-risk intent category => escalate=True" for
    every gold example, since that's literally what the rule engine does. That would have
    made escalation "accuracy" close to 100% by construction and measured nothing. Instead
    each of the 200 examples got a real judgment call (money at risk / explicit human ask
    / stated repeated-contact history / security consequence / safety / unintelligible).
    The result is genuinely informative: e.g. gold `unclear_other` messages only warrant
    escalation 16.7% of the time, while the rule engine currently escalates that category
    100% of the time by policy — a real, reportable calibration gap, not a tautology.

16. **Cached-replay + live-smoke-test hybrid for the LLM stages, not a from-scratch live
    run every time.** Groq's free tier measured out to roughly 8,000 tokens/minute on this
    account (confirmed via response headers, not assumed from blog posts) — at ~950
    tokens/agent-call, a live run across the full 200-example golden set takes ~25-40
    minutes, which cannot fit a 15-minute reproduction budget alongside data prep and
    training. `scripts/07_run_llm_agent.py` and `09_llm_judge.py` default to a small
    `--limit` (fast, proves the integration genuinely calls the real API) while the full
    200-example runs are executed once and their outputs committed under
    `eval/results/`, with a `--full` flag to reproduce them live at the real ~30-70 minute
    cost. This is disclosed as a "what's misleading about my headline number" item.

## Golden set

17. **Stratified, not proportional, sampling for the 200-example golden set.** Sampling
    proportional to natural frequency would have left rare categories like
    `account_access` or `billing_payment` with too few examples (~8-15) for a meaningful
    per-class F1. The sample instead targets 45/30/25/25/20/20/20/15 per category
    (`scripts/03_select_golden_sample.py`), which trades "represents true prevalence" for
    "every category has enough n to trust its number" — true prevalence is reported
    separately from the full 33.8k pool.

18. **Every gold label was independently re-derived from the raw tweet text, never copied
    from the weak-label guess used to select the sample.** Confirmed by measuring
    agreement between `gold_intent` and `weak_intent_guess`: only 62%. That gap is the
    actual evidence the labeling was independent, not rubber-stamped.

## Web demo

19. **Vanilla HTML/CSS/JS + FastAPI, not React/a build toolchain.** Avoids Node-tooling
    risk on a machine that already has one unpredictable native-dependency blocker, keeps
    the "reproduce this" story to one `pip install` + one `uvicorn` command, and a
    hand-built, small (~400-line) frontend is fully auditable in one sitting.

20. **The confusion-matrix "heatmap" is a plain HTML/CSS grid, not a Chart.js canvas.** An
    early version used Chart.js `scatter` with square point markers, but fixed-pixel point
    radii don't reliably tile a variable-size container into a true grid (gaps/overlaps
    depending on viewport width). A `display: grid` table with per-cell background-color
    tiles perfectly by construction and doubles as an accessible data table.

21. **Judge scores are shown as a grouped bar chart, not a radar chart**, despite radar
    being the "obvious" choice for "5 dimensions x 3 systems." Radar charts distort
    area/angle perception and make non-adjacent axes hard to compare directly — a plain
    grouped bar keeps every score on one shared, honest axis.

## Assignment-specific

22. **Did not use Banking77, the assignment's optional secondary dataset.** It's explicitly
    scoped to "intent work only," but its 77 categories are banking-specific (card disputes,
    exchange rates, top-ups) with no clean mapping onto an e-commerce brand's support
    traffic. The taxonomy here was derived directly from reading ~200 real AmazonHelp
    threads instead, which the assignment separately asks for ("intents that you define
    from the data") — recorded here so this reads as a considered choice, not an oversight.
