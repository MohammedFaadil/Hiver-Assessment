"""Run the LLM (RAG-grounded) agent over the golden set.

Groq's free tier caps out around ~8000 tokens/min measured against this
account (see DECISION_LOG.md) -- at ~1000-1100 tokens per agent call, a live
run over the full 200-example golden set takes roughly 25-30 minutes, which
does not fit the README's 15-minute reproduction budget. So:

  --limit N   (default 20)   fast, live smoke-test slice -- proves the Groq
                              integration genuinely works end-to-end. This is
                              what the default `make reproduce` path runs.
  --full                     the complete 200-example run used to compute the
                              headline numbers in REPORT.md. Takes ~25-30 min;
                              intended to be run once and the output committed
                              to eval/results/, not re-run on every clone.

Progress is saved incrementally so a killed/interrupted run doesn't lose
completed work -- rerun with the same args and it resumes past what's already
in the output file.
"""
from __future__ import annotations

import argparse
import time
import traceback

import pandas as pd
from tqdm import tqdm

from agent import config, llm_client
from agent.agent import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if not config.has_llm():
        raise SystemExit("GROQ_API_KEY not set -- see .env.example. LLM agent mode requires it.")

    golden = pd.read_csv(config.GOLDEN_SET_PATH)
    if not args.full:
        golden = golden.head(args.limit)

    out_path = config.RESULTS_DIR / (args.out or ("predictions_llm.csv" if args.full else "predictions_llm_smoke.csv"))

    done_ids: set[int] = set()
    existing_rows: list[dict] = []
    if out_path.exists():
        prev = pd.read_csv(out_path)
        existing_rows = prev.to_dict("records")
        done_ids = set(prev["id"].tolist())
        print(f"Resuming: {len(done_ids)} examples already done in {out_path}")

    pipeline = Pipeline.load()
    todo = golden[~golden["id"].isin(done_ids)]
    print(f"Running LLM agent on {len(todo)} examples (model={config.LLM_AGENT_MODEL})...")

    rows = existing_rows
    t0 = time.time()
    n_errors = 0
    for _, ex in tqdm(list(todo.iterrows()), total=len(todo)):
        try:
            result = pipeline.run_llm(ex["customer_text"])
            rows.append({
                "id": ex["id"],
                "mode": "llm",
                "customer_text": ex["customer_text"],
                "pred_intent": result.intent,
                "intent_confidence": result.intent_confidence,
                "drafted_reply": result.drafted_reply,
                "retrieval_similarity": result.retrieval_similarity,
                "pred_escalate": result.escalate,
                "escalation_reason": result.escalation_reason,
                "llm_suggested_escalate": result.llm_suggested_escalate,
                "llm_escalation_reason": result.llm_escalation_reason,
                "latency_s": result.latency_s,
            })
        except Exception as e:  # noqa: BLE001 -- keep going, log, don't lose progress
            n_errors += 1
            print(f"\n  [error] id={ex['id']}: {e}")
            traceback.print_exc(limit=1)
        # save incrementally every few rows so a crash/interrupt doesn't lose work
        if len(rows) % 5 == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    usage = llm_client.summarize_usage()
    dt = time.time() - t0
    print(f"\nWrote {len(rows)} predictions -> {out_path}")
    print(f"Errors: {n_errors}")
    print(f"Elapsed: {dt:.1f}s for {len(todo)} new calls this run")
    print(f"Cumulative LLM usage this process: {usage}")


if __name__ == "__main__":
    main()
