"""Minimal TF-IDF + Multinomial Naive Bayes, implemented directly on numpy.

Why not scikit-learn: on the machine this was built on, a Windows Application
Control policy blocks scipy/scikit-learn's compiled DLLs (numpy/pandas/
matplotlib are unaffected — see DECISION_LOG.md). Rather than fight a local
security policy, the classical-ML "simple baseline" is implemented directly:
it's ~150 lines, has zero compiled dependencies, and is fully auditable line
by line — which is a reasonable trade for a component whose whole purpose is
to be the transparent, inspectable alternative to the LLM.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from agent.text_utils import tokenize


@dataclass
class Vectorizer:
    """Bag-of-words vectorizer producing both raw counts and TF-IDF vectors
    over a shared vocabulary (counts feed Naive Bayes, TF-IDF feeds retrieval)."""

    max_features: int = 4000
    min_df: int = 2

    vocab_: dict | None = None
    idf_: np.ndarray | None = None

    def fit(self, docs: list[str]) -> "Vectorizer":
        doc_freq: dict[str, int] = {}
        tokenized = [tokenize(d) for d in docs]
        for toks in tokenized:
            for t in set(toks):
                doc_freq[t] = doc_freq.get(t, 0) + 1

        candidates = [(t, df) for t, df in doc_freq.items() if df >= self.min_df]
        candidates.sort(key=lambda x: (-x[1], x[0]))
        candidates = candidates[: self.max_features]

        self.vocab_ = {t: i for i, (t, _) in enumerate(candidates)}
        n_docs = len(docs)
        df_arr = np.array([df for _, df in candidates], dtype=np.float64)
        # smoothed idf, always positive
        self.idf_ = np.log((1 + n_docs) / (1 + df_arr)) + 1.0
        return self

    def _counts_matrix(self, docs: list[str]) -> np.ndarray:
        assert self.vocab_ is not None, "call fit() first"
        n = len(docs)
        v = len(self.vocab_)
        X = np.zeros((n, v), dtype=np.float64)
        for row, doc in enumerate(docs):
            for tok in tokenize(doc):
                col = self.vocab_.get(tok)
                if col is not None:
                    X[row, col] += 1.0
        return X

    def transform_counts(self, docs: list[str]) -> np.ndarray:
        return self._counts_matrix(docs)

    def transform_tfidf(self, docs: list[str]) -> np.ndarray:
        X = self._counts_matrix(docs)
        X = X * self.idf_[None, :]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def to_dict(self) -> dict:
        return {
            "max_features": self.max_features,
            "min_df": self.min_df,
            "vocab": self.vocab_,
            "idf": self.idf_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Vectorizer":
        v = cls(max_features=d["max_features"], min_df=d["min_df"])
        v.vocab_ = d["vocab"]
        v.idf_ = np.array(d["idf"], dtype=np.float64)
        return v


@dataclass
class MultinomialNB:
    """Textbook multinomial Naive Bayes with Laplace smoothing."""

    alpha: float = 1.0
    classes_: np.ndarray | None = None
    class_log_prior_: np.ndarray | None = None
    feature_log_prob_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNB":
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(max(X_c.shape[0], 1) / X.shape[0])
            feat_count = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(feat_count / feat_count.sum())
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        return X @ self.feature_log_prob_.T + self.class_log_prior_[None, :]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logp = self.predict_log_proba(X)
        logp = logp - logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = np.argmax(self.predict_log_proba(X), axis=1)
        return self.classes_[idx]

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "classes": self.classes_.tolist(),
            "class_log_prior": self.class_log_prior_.tolist(),
            "feature_log_prob": self.feature_log_prob_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MultinomialNB":
        m = cls(alpha=d["alpha"])
        m.classes_ = np.array(d["classes"])
        m.class_log_prior_ = np.array(d["class_log_prior"])
        m.feature_log_prob_ = np.array(d["feature_log_prob"])
        return m


def cosine_topk(query_vec: np.ndarray, kb_matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Both query_vec and kb_matrix rows are assumed L2-normalized (TF-IDF
    output of Vectorizer already is), so cosine similarity == dot product."""
    sims = kb_matrix @ query_vec
    k = min(k, len(sims))
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return top_idx, sims[top_idx]


def save_json(path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def load_json(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
