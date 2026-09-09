"""How well does the LLM judge agree with a human?

HUMAN_SCORES below are a manual re-grade -- read independently, before
transcribing -- of a fixed sample of (id, mode) pairs already scored by
scripts/09_llm_judge.py, using the identical rubric (src/agent/judge.py). The
sample is a hardcoded, stable set (not re-sampled at runtime) so this script
gives the same answer regardless of how much of judge_scores.csv exists when
it's run; see LABELING_GUIDE.md / DECISION_LOG.md for why golden-set labeling
uses the same "hardcode + verify a snippet" pattern.

Each entry stores a `check` substring asserted against the real customer_text
so a mismatched id fails loudly instead of silently comparing the wrong row.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from agent import config
from agent.metrics import spearman_corr

# (id, mode) -> (check_substring, my_grounding, my_helpfulness, my_tone, my_clarity, my_overall)
HUMAN_SCORES: dict[tuple[int, str], tuple[str, int, int, int, int, int]] = {
    (1, "simple"): ("3 forms", 3, 2, 4, 5, 3),
    (5, "simple"): ("Barbie house", 5, 3, 5, 5, 4),
    (20, "simple"): ("Wal-Mart", 4, 3, 4, 5, 3),
    (26, "simple"): ("refund mail", 1, 1, 2, 2, 1),
    (28, "simple"): ("prioritizing an order", 4, 4, 3, 5, 4),
    (48, "simple"): ("shipping department", 4, 4, 4, 5, 4),
    (50, "simple"): ("microwave", 4, 3, 3, 4, 3),
    (52, "simple"): ("KYC", 1, 1, 2, 1, 1),
    (57, "simple"): ("FREE HANDBAG", 1, 1, 1, 3, 1),
    (67, "simple"): ("guaranteed", 3, 2, 2, 3, 2),
    (73, "simple"): ("charged twice. Order no", 1, 1, 2, 3, 1),
    (76, "simple"): ("Dishonest delivery setup", 4, 2, 3, 4, 2),
    (77, "simple"): ("wwe2k18", 2, 1, 3, 3, 2),
    (81, "simple"): ("delivered but you didn", 5, 4, 5, 5, 4),
    (85, "simple"): ("1yr prime subscptin", 5, 3, 4, 5, 4),
}

DIMS = ["grounding_score", "helpfulness_score", "tone_score", "clarity_score", "overall_score"]


def main() -> None:
    path = config.RESULTS_DIR / "judge_scores.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run scripts/09_llm_judge.py first")
    judge_df = pd.read_csv(path).set_index(["id", "mode"])

    rows = []
    for (row_id, mode), (check, g, h, t, c, o) in HUMAN_SCORES.items():
        if (row_id, mode) not in judge_df.index:
            print(f"  [skip] ({row_id}, {mode}) not yet in {path.name} -- judge run may still be in progress")
            continue
        judge_row = judge_df.loc[(row_id, mode)]
        text = judge_row["customer_text"]
        if check.lower() not in str(text).lower():
            raise SystemExit(f"Verification failed for id={row_id} mode={mode}: {check!r} not found in {text!r}")
        rows.append({
            "id": row_id, "mode": mode,
            "human_grounding": g, "judge_grounding": judge_row["grounding_score"],
            "human_helpfulness": h, "judge_helpfulness": judge_row["helpfulness_score"],
            "human_tone": t, "judge_tone": judge_row["tone_score"],
            "human_clarity": c, "judge_clarity": judge_row["clarity_score"],
            "human_overall": o, "judge_overall": judge_row["overall_score"],
        })

    if not rows:
        raise SystemExit("No human-scored rows matched judge_scores.csv yet.")

    df = pd.DataFrame(rows)
    diffs = (df["human_overall"] - df["judge_overall"]).abs()
    exact_match_rate = float((diffs == 0).mean())
    within_one = float((diffs <= 1).mean())
    mean_abs_diff = float(diffs.mean())
    corr = spearman_corr(df["human_overall"], df["judge_overall"])

    per_dim = {}
    for dim in DIMS:
        short = dim.replace("_score", "")
        d = (df[f"human_{short}"] - df[f"judge_{short}"]).abs()
        per_dim[short] = {"mean_abs_diff": float(d.mean()), "within_one_point_rate": float((d <= 1).mean())}

    result = {
        "description": (
            f"{len(df)} (customer message, drafted reply) pairs, independently re-graded by hand "
            "on the same 1-5 rubric the LLM judge uses, before comparing to the judge's stored "
            "scores. Sample is fixed (see scripts/10_judge_agreement.py), not cherry-picked after "
            "seeing the judge's scores."
        ),
        "n": len(df),
        "exact_match_rate_overall": exact_match_rate,
        "mean_abs_diff_overall": mean_abs_diff,
        "within_one_point_rate": within_one,
        "spearman_corr": corr,
        "per_dimension": per_dim,
    }

    out_path = config.RESULTS_DIR / "judge_human_agreement.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"n = {len(df)}")
    print(f"exact match (overall score): {exact_match_rate:.1%}")
    print(f"mean |human - judge| (overall): {mean_abs_diff:.2f}")
    print(f"within +/-1 point: {within_one:.1%}")
    print(f"Spearman correlation: {corr:.2f}")
    print("\nper-dimension mean abs diff:")
    for dim, stats in per_dim.items():
        print(f"  {dim:<14} {stats['mean_abs_diff']:.2f}  (within +/-1: {stats['within_one_point_rate']:.0%})")
    print(f"\nWrote {out_path}")

    df.to_csv(config.RESULTS_DIR / "judge_human_agreement_detail.csv", index=False)


if __name__ == "__main__":
    main()
