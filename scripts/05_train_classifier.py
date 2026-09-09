"""Train the classical "simple baseline" intent classifier (TF-IDF counts +
Multinomial Naive Bayes) on the weakly-labeled train_pool, and separately
build the TF-IDF retrieval index over retrieval_kb. Both are pure numpy (see
src/agent/nlp_model.py for why) and get saved for reuse by every downstream
script (baselines, the LLM agent, evaluation).

Also computes the majority-class intent from the weak-labeled pool, used by
the trivial baseline.
"""
from __future__ import annotations

import time

import joblib
import pandas as pd

from agent import config
from agent.intent_rules import majority_class
from agent.nlp_model import MultinomialNB, Vectorizer
from agent.retrieval import RetrievalIndex


def main() -> None:
    t0 = time.time()

    print("[1/3] Training intent classifier on weakly-labeled train_pool...")
    train_df = pd.read_csv(config.TRAIN_POOL_PATH)
    texts = train_df["customer_text"].tolist()
    labels = train_df["weak_intent"].to_numpy()

    vectorizer = Vectorizer(max_features=config.MAX_VOCAB_SIZE, min_df=config.MIN_DOC_FREQ)
    vectorizer.fit(texts)
    X_counts = vectorizer.transform_counts(texts)

    classifier = MultinomialNB(alpha=1.0)
    classifier.fit(X_counts, labels)
    maj_intent = majority_class(labels)
    print(f"      vocab size: {len(vectorizer.vocab_)}, majority class: {maj_intent}")

    joblib.dump(
        {"vectorizer": vectorizer, "classifier": classifier, "majority_intent": maj_intent},
        config.CLASSIFIER_PATH,
    )
    print(f"      saved -> {config.CLASSIFIER_PATH}")

    print("\n[2/3] Building retrieval index over retrieval_kb...")
    kb_df = pd.read_csv(config.RETRIEVAL_KB_PATH)
    # Retrieval matches on the *raw* customer text (closer to what a real
    # incoming message looks like). It serves the *redacted* reply as the
    # groundable template (no stale links / other customers' @mentions) —
    # keep the original under brand_reply_raw for audit/reference only.
    kb_df["brand_reply_raw"] = kb_df["brand_reply"]
    kb_df["brand_reply"] = kb_df["brand_reply_redacted"]
    kb_df["intent"] = kb_df["weak_intent"]  # used for intent_filter in RetrievalIndex.query

    retrieval_index = RetrievalIndex.build(
        kb_df, max_features=config.MAX_VOCAB_SIZE, min_df=config.MIN_DOC_FREQ
    )
    retrieval_path = config.MODELS_DIR / "retrieval_index.joblib"
    retrieval_index.save(retrieval_path)
    print(f"      KB size: {len(kb_df)}, saved -> {retrieval_path}")

    print("\n[3/3] Quick sanity check on a held-out-style query...")
    sample = retrieval_index.query("my package is 5 days late and tracking is stuck", k=3)
    for _, r in sample.iterrows():
        print(f"      sim={r['similarity']:.3f} | {r['customer_text'][:70]!r} -> {r['brand_reply'][:70]!r}")

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
