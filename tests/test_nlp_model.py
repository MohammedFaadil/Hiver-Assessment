import numpy as np
import pytest

from agent.nlp_model import MultinomialNB, Vectorizer, cosine_topk

DOCS = [
    "my order is late where is my package",
    "package never arrived tracking not updating",
    "i want a refund for this item return it",
    "please cancel my order and refund me",
    "my card was charged twice billing error",
    "unauthorized charge on my account payment issue",
    "thanks so much great service love amazon",
    "awesome support really appreciate the help",
]
LABELS = np.array([
    "order_delivery", "order_delivery",
    "returns_refund", "returns_refund",
    "billing_payment", "billing_payment",
    "compliment", "compliment",
])


def test_vectorizer_builds_vocab_and_respects_min_df():
    vec = Vectorizer(max_features=200, min_df=2).fit(DOCS)
    assert len(vec.vocab_) > 0
    # "order" appears in >=2 docs, should be in vocab; a single-doc-only rare
    # token should not be, given min_df=2.
    assert "order" in vec.vocab_


def test_tfidf_rows_are_unit_normalized():
    vec = Vectorizer(max_features=200, min_df=1).fit(DOCS)
    tfidf = vec.transform_tfidf(DOCS)
    norms = np.linalg.norm(tfidf, axis=1)
    nonzero = norms[norms > 0]
    assert np.allclose(nonzero, 1.0, atol=1e-6)


def test_naive_bayes_classifies_toy_data_correctly():
    vec = Vectorizer(max_features=200, min_df=1).fit(DOCS)
    X = vec.transform_counts(DOCS)
    nb = MultinomialNB(alpha=1.0).fit(X, LABELS)

    test_docs = ["where is my package it never showed up", "thank you amazon for the great help"]
    X_test = vec.transform_counts(test_docs)
    preds = nb.predict(X_test)
    assert preds[0] == "order_delivery"
    assert preds[1] == "compliment"


def test_naive_bayes_predict_proba_sums_to_one():
    vec = Vectorizer(max_features=200, min_df=1).fit(DOCS)
    X = vec.transform_counts(DOCS)
    nb = MultinomialNB(alpha=1.0).fit(X, LABELS)
    proba = nb.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-8)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_cosine_topk_returns_best_match_first():
    vec = Vectorizer(max_features=200, min_df=1).fit(DOCS)
    tfidf = vec.transform_tfidf(DOCS)
    query = vec.transform_tfidf(["my package is late"])[0]
    idx, sims = cosine_topk(query, tfidf, k=3)
    assert len(idx) == 3
    # similarities should be sorted descending
    assert list(sims) == sorted(sims, reverse=True)
    # the best match should be one of the two order_delivery docs
    assert LABELS[idx[0]] == "order_delivery"


def test_vectorizer_roundtrip_via_dict():
    vec = Vectorizer(max_features=50, min_df=1).fit(DOCS)
    d = vec.to_dict()
    restored = Vectorizer.from_dict(d)
    a = vec.transform_tfidf(DOCS)
    b = restored.transform_tfidf(DOCS)
    assert np.allclose(a, b)


def test_naive_bayes_roundtrip_via_dict():
    vec = Vectorizer(max_features=50, min_df=1).fit(DOCS)
    X = vec.transform_counts(DOCS)
    nb = MultinomialNB().fit(X, LABELS)
    restored = MultinomialNB.from_dict(nb.to_dict())
    assert np.array_equal(nb.predict(X), restored.predict(X))
