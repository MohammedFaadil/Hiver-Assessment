"""Automated evaluation: intent classification metrics, escalation-decision
metrics, and text-overlap reply-quality proxies, computed per mode
(trivial / simple / llm) against the golden set. Writes eval/results/
metrics.json plus two charts. LLM-as-judge scoring is a separate script
(09_llm_judge.py) since it needs API calls.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent import config
from agent.metrics import (
    binary_precision_recall_f1,
    precision_recall_f1,
    rouge_l_f1,
    unigram_overlap_f1,
)


def load_predictions() -> pd.DataFrame:
    frames = [pd.read_csv(config.RESULTS_DIR / "predictions_baselines.csv")]
    llm_path = config.RESULTS_DIR / "predictions_llm.csv"
    smoke_path = config.RESULTS_DIR / "predictions_llm_smoke.csv"
    if llm_path.exists():
        frames.append(pd.read_csv(llm_path))
    elif smoke_path.exists():
        print(f"NOTE: full LLM run not found, using smoke-test subset ({smoke_path.name})")
        frames.append(pd.read_csv(smoke_path))
    else:
        print("NOTE: no LLM predictions found -- run scripts/07_run_llm_agent.py first. "
              "Evaluating trivial/simple only.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    golden = pd.read_csv(config.GOLDEN_SET_PATH)[
        ["id", "gold_intent", "gold_escalate", "brand_historical_reply"]
    ]
    preds = load_predictions()
    merged = preds.merge(golden, on="id", how="inner")

    results = {}
    for mode, df in merged.groupby("mode"):
        n = len(df)
        intent_metrics = precision_recall_f1(
            df["gold_intent"].tolist(), df["pred_intent"].tolist(), config.INTENT_IDS
        )
        esc_metrics = binary_precision_recall_f1(
            df["gold_escalate"].tolist(), df["pred_escalate"].tolist()
        )
        rouge_scores = [
            rouge_l_f1(ref, cand) for ref, cand in zip(df["brand_historical_reply"], df["drafted_reply"])
        ]
        unigram_scores = [
            unigram_overlap_f1(ref, cand) for ref, cand in zip(df["brand_historical_reply"], df["drafted_reply"])
        ]
        reply_lengths = df["drafted_reply"].str.split().str.len()

        results[mode] = {
            "n": n,
            "intent": {
                "accuracy": intent_metrics["accuracy"],
                "macro_f1": intent_metrics["macro_f1"],
                "weighted_f1": intent_metrics["weighted_f1"],
                "per_class_f1": {k: v["f1"] for k, v in intent_metrics["per_class"].items()},
                "confusion_matrix": intent_metrics["confusion_matrix"],
                "labels": intent_metrics["labels"],
            },
            "escalation": esc_metrics,
            "reply_quality_proxy": {
                "rouge_l_vs_historical_mean": float(np.mean(rouge_scores)),
                "unigram_f1_vs_historical_mean": float(np.mean(unigram_scores)),
                "reply_len_words_mean": float(reply_lengths.mean()),
                "reply_len_words_median": float(reply_lengths.median()),
            },
        }

    out_path = config.RESULTS_DIR / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}\n")

    print(f"{'mode':<10} {'n':>4} {'intent_acc':>11} {'intent_macroF1':>15} "
          f"{'esc_f1':>7} {'esc_prec':>9} {'esc_recall':>11} {'rougeL':>8}")
    for mode, r in results.items():
        print(f"{mode:<10} {r['n']:>4} {r['intent']['accuracy']:>11.3f} "
              f"{r['intent']['macro_f1']:>15.3f} {r['escalation']['f1']:>7.3f} "
              f"{r['escalation']['precision']:>9.3f} {r['escalation']['recall']:>11.3f} "
              f"{r['reply_quality_proxy']['rouge_l_vs_historical_mean']:>8.3f}")

    # --- charts -------------------------------------------------------
    modes = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    x = np.arange(len(modes))
    axes[0].bar(x, [results[m]["intent"]["macro_f1"] for m in modes], color="#4C72B0", label="intent macro-F1")
    axes[0].bar(x, [results[m]["escalation"]["f1"] for m in modes], width=0.4, color="#DD8452",
                alpha=0.85, label="escalation F1")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(modes)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Intent macro-F1 vs Escalation F1 by mode")
    axes[0].legend()

    last_mode = modes[-1]
    cm = np.array(results[last_mode]["intent"]["confusion_matrix"])
    labels = results[last_mode]["intent"]["labels"]
    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[1].set_yticklabels(labels, fontsize=7)
    axes[1].set_xlabel("predicted")
    axes[1].set_ylabel("gold")
    axes[1].set_title(f"Confusion matrix ({last_mode})")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    chart_path = config.RESULTS_DIR / "metrics_summary.png"
    fig.savefig(chart_path, dpi=150)
    print(f"\nWrote chart -> {chart_path}")


if __name__ == "__main__":
    main()
