"""Nearest-neighbor retrieval over a knowledge base of historical
(customer message -> brand reply) pairs. This is what "grounds" both the
template baseline and the LLM's drafted replies in how the brand has
actually resolved similar issues before.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from agent.nlp_model import Vectorizer, cosine_topk


@dataclass
class RetrievalIndex:
    vectorizer: Vectorizer
    kb_df: pd.DataFrame
    kb_tfidf: np.ndarray

    @classmethod
    def build(cls, kb_df: pd.DataFrame, max_features: int = 4000, min_df: int = 2) -> "RetrievalIndex":
        texts = kb_df["customer_text"].tolist()
        vec = Vectorizer(max_features=max_features, min_df=min_df).fit(texts)
        tfidf = vec.transform_tfidf(texts)
        return cls(vectorizer=vec, kb_df=kb_df.reset_index(drop=True), kb_tfidf=tfidf)

    def query(self, text: str, k: int = 3, intent_filter: str | None = None) -> pd.DataFrame:
        qvec = self.vectorizer.transform_tfidf([text])[0]

        if intent_filter and "intent" in self.kb_df.columns:
            mask = (self.kb_df["intent"] == intent_filter).to_numpy()
            idx_pool = np.where(mask)[0]
            if len(idx_pool) < k:
                idx_pool = np.arange(len(self.kb_df))
        else:
            idx_pool = np.arange(len(self.kb_df))

        sub_matrix = self.kb_tfidf[idx_pool]
        local_idx, sims = cosine_topk(qvec, sub_matrix, k)
        top_idx = idx_pool[local_idx]

        results = self.kb_df.iloc[top_idx].copy()
        results["similarity"] = sims
        return results.reset_index(drop=True)

    def save(self, path) -> None:
        joblib.dump(
            {"vectorizer": self.vectorizer, "kb_df": self.kb_df, "kb_tfidf": self.kb_tfidf},
            path,
        )

    @classmethod
    def load(cls, path) -> "RetrievalIndex":
        d = joblib.load(path)
        return cls(vectorizer=d["vectorizer"], kb_df=d["kb_df"], kb_tfidf=d["kb_tfidf"])
