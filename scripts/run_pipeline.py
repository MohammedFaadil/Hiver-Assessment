"""One-command reproduction of the headline results, start to finish.

Quick mode (default) fits well under 15 minutes: it rebuilds the dataset,
classifier, retrieval index, baselines, evaluation, and runs a small LIVE
sample against the real Groq API (proving the integration genuinely works)
-- then reports the FULL 200-example headline numbers from the artifacts
already committed under eval/results/ (see DECISION_LOG.md for why: Groq's
free-tier throughput makes a full 200-example live run take 25-70 minutes,
which cannot fit a 15-minute budget alongside everything else).

    python scripts/run_pipeline.py            # quick mode, ~5-8 minutes
    python scripts/run_pipeline.py --full      # full live re-run, ~40-90 minutes

Steps that don't need an LLM (data build, classifier, baselines, metrics)
always run for real. Steps that do (scripts 07, 09, 10) are skipped with a
clear message if GROQ_API_KEY isn't set, and the rest of the pipeline still
completes using whatever cached eval/results/*.csv are already committed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    t0 = time.time()
    result = subprocess.run([PY, "-u", *cmd], cwd=ROOT)
    dt = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"--- {label}: {status} in {dt:.1f}s ---")
    if result.returncode != 0:
        print(f"\nStopping pipeline: {label} failed. See output above.")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="live re-run of the full 200-example LLM + judge stages")
    parser.add_argument("--smoke-limit", type=int, default=15, help="examples per mode for the quick-mode live smoke test")
    args = parser.parse_args()

    from agent import config  # local import so argparse --help doesn't need the venv fully set up

    t_start = time.time()

    run(["scripts/02_build_dataset.py"], "1/8: Build dataset (download + filter + weak-label)")
    run(["scripts/03_select_golden_sample.py"], "2/8: Select stratified golden sample")
    run(["scripts/04_apply_gold_labels.py"], "3/8: Apply hand-written gold labels")
    run(["scripts/05_train_classifier.py"], "4/8: Train classifier + build retrieval index")
    run(["scripts/06_run_baselines.py"], "5/8: Run trivial + simple baselines")

    if config.has_llm():
        if args.full:
            run(["scripts/07_run_llm_agent.py", "--full"], "6/8: LLM agent -- FULL live run (~25-40 min)")
            run(["scripts/09_llm_judge.py", "--modes", "llm,simple"], "7/8: LLM judge -- FULL live run (~40-70 min)")
            run(["scripts/09_llm_judge.py", "--modes", "trivial", "--limit", "40"], "7b/8: LLM judge -- trivial sample")
        else:
            run(["scripts/07_run_llm_agent.py", "--limit", str(args.smoke_limit)],
                "6/8: LLM agent -- quick LIVE smoke test (proves the real API works)")
            run(["scripts/09_llm_judge.py", "--limit", str(args.smoke_limit)],
                "7/8: LLM judge -- quick LIVE smoke test")
        run(["scripts/10_judge_agreement.py"], "8/8: Judge-vs-human agreement check")
    else:
        print("\n" + "=" * 70)
        print("GROQ_API_KEY not set -- skipping steps 6-8 (LLM agent, judge, agreement).")
        print("Headline numbers below still use the FULL 200-example results already")
        print("committed under eval/results/ from the original run. Set GROQ_API_KEY in")
        print(".env and re-run to also exercise the live path yourself.")
        print("=" * 70)

    run(["scripts/08_evaluate.py"], "Final: Compute metrics over whatever predictions exist")

    print(f"\nTotal pipeline time: {time.time()-t_start:.1f}s")
    print("\nHeadline numbers: see eval/results/metrics.json, eval/results/metrics_summary.png,")
    print("and REPORT.md. Launch the interactive demo with:")
    print("  uvicorn webapp.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
