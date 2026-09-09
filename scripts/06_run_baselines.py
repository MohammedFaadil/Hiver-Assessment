"""Run the trivial and simple baselines over the full golden set. Both are
pure numpy / rule-based -- no API calls, no rate limits, finishes in seconds.
Output feeds scripts/08_evaluate.py.
"""
from __future__ import annotations

import time

import pandas as pd
from tqdm import tqdm

from agent import config
from agent.agent import Pipeline


def main() -> None:
    t0 = time.time()
    golden = pd.read_csv(config.GOLDEN_SET_PATH)
    pipeline = Pipeline.load()

    rows = []
    for mode, run_fn in [("trivial", pipeline.run_trivial), ("simple", pipeline.run_simple)]:
        for _, ex in tqdm(list(golden.iterrows()), desc=mode):
            result = run_fn(ex["customer_text"])
            rows.append({
                "id": ex["id"],
                "mode": mode,
                "customer_text": ex["customer_text"],
                "pred_intent": result.intent,
                "intent_confidence": result.intent_confidence,
                "drafted_reply": result.drafted_reply,
                "retrieval_similarity": result.retrieval_similarity,
                "pred_escalate": result.escalate,
                "escalation_reason": result.escalation_reason,
            })

    out = pd.DataFrame(rows)
    out_path = config.RESULTS_DIR / "predictions_baselines.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} predictions ({golden.shape[0]} examples x 2 modes) -> {out_path}")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
