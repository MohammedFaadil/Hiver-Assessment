"""Score drafted replies with the LLM-as-judge rubric (src/agent/judge.py).

Judges replies from ALL modes present in eval/results/predictions_*.csv
(trivial, simple, llm) so reply-quality comparisons in REPORT.md aren't
limited to the headline system. Uses config.LLM_JUDGE_MODEL (a different,
larger model than the agent's drafting model) and the same incremental-save/
resume pattern as scripts/07 since this is another rate-limited, long-running
batch (~all rows x ~700 tokens/call against the ~6500 tokens/min budget).

  --limit N   judge only the first N rows per mode (fast smoke test)
  --modes     comma-separated subset of modes to judge (default: all found)
"""
from __future__ import annotations

import argparse
import time
import traceback

import pandas as pd
from tqdm import tqdm

from agent import config, llm_client
from agent.agent import Pipeline
from agent.judge import judge_reply


def load_all_predictions() -> pd.DataFrame:
    frames = [pd.read_csv(config.RESULTS_DIR / "predictions_baselines.csv")]
    llm_path = config.RESULTS_DIR / "predictions_llm.csv"
    smoke_path = config.RESULTS_DIR / "predictions_llm_smoke.csv"
    if llm_path.exists():
        frames.append(pd.read_csv(llm_path))
    elif smoke_path.exists():
        frames.append(pd.read_csv(smoke_path))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--modes", default=None)
    parser.add_argument("--out", default="judge_scores.csv")
    args = parser.parse_args()

    if not config.has_llm():
        raise SystemExit("GROQ_API_KEY not set -- see .env.example. Judging requires it.")

    preds = load_all_predictions()
    if args.modes:
        preds = preds[preds["mode"].isin(args.modes.split(","))]
    if args.limit:
        preds = preds.groupby("mode", group_keys=False).head(args.limit)

    out_path = config.RESULTS_DIR / args.out
    done_keys: set[tuple] = set()
    rows: list[dict] = []
    if out_path.exists():
        prev = pd.read_csv(out_path)
        rows = prev.to_dict("records")
        done_keys = set(zip(prev["id"], prev["mode"]))
        print(f"Resuming: {len(done_keys)} (id, mode) pairs already judged in {out_path}")

    todo = preds[~preds.apply(lambda r: (r["id"], r["mode"]) in done_keys, axis=1)]
    print(f"Judging {len(todo)} replies (model={config.LLM_JUDGE_MODEL})...")

    # Retrieval is a deterministic, free (no-API) function of the classifier +
    # index, neither of which have changed since generation -- so we recompute
    # "what precedent was this reply grounded in" here rather than having to
    # thread a retrieved_examples column through every prediction script.
    # (trivial mode retrieves nothing, so the judge correctly sees none for it.)
    pipeline = Pipeline.load()

    def retrieved_for(mode: str, text: str) -> list[dict]:
        if mode == "trivial":
            return []
        intent_guess, _conf = pipeline.classify(text)
        hits = pipeline.retrieval_index.query(text, k=config.RETRIEVAL_TOP_K, intent_filter=intent_guess)
        return hits.to_dict("records")

    t0 = time.time()
    n_errors = 0
    for _, row in tqdm(list(todo.iterrows()), total=len(todo)):
        try:
            retrieved = retrieved_for(row["mode"], row["customer_text"])
            score = judge_reply(
                customer_text=row["customer_text"],
                drafted_reply=row["drafted_reply"],
                retrieved_examples=retrieved,
            )
            rows.append({
                "id": row["id"],
                "mode": row["mode"],
                "customer_text": row["customer_text"],
                "drafted_reply": row["drafted_reply"],
                "grounding_score": score["grounding_score"],
                "helpfulness_score": score["helpfulness_score"],
                "tone_score": score["tone_score"],
                "clarity_score": score["clarity_score"],
                "overall_score": score["overall_score"],
                "rationale": score["rationale"],
            })
        except Exception as e:  # noqa: BLE001
            n_errors += 1
            print(f"\n  [error] id={row['id']} mode={row['mode']}: {e}")
            traceback.print_exc(limit=1)
        if len(rows) % 5 == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    dt = time.time() - t0
    print(f"\nWrote {len(rows)} judged replies -> {out_path}")
    print(f"Errors: {n_errors}")
    print(f"Elapsed: {dt:.1f}s for {len(todo)} new calls this run")
    print(f"Cumulative LLM usage this process: {llm_client.summarize_usage()}")


if __name__ == "__main__":
    main()
