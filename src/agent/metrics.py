"""Classification metrics implemented directly on numpy (no scikit-learn --
see DECISION_LOG.md) plus a couple of lightweight, dependency-free text
overlap metrics used as an automated (non-LLM) proxy for reply quality.
"""
from __future__ import annotations

import numpy as np

from agent.text_utils import tokenize


def confusion_matrix(y_true, y_pred, labels: list[str]) -> np.ndarray:
    idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm


def precision_recall_f1(y_true, y_pred, labels: list[str]) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels)
    per_class = {}
    precisions, recalls, f1s, supports = [], [], [], []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[label] = {
            "precision": precision, "recall": recall, "f1": f1, "support": int(support),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    accuracy = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
    macro_f1 = float(np.mean(f1s))
    weights = np.array(supports) / max(sum(supports), 1)
    weighted_f1 = float(np.sum(np.array(f1s) * weights))

    return {
        "per_class": per_class,
        "accuracy": float(accuracy),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def binary_precision_recall_f1(y_true: list[bool], y_pred: list[bool]) -> dict:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


def cohens_kappa(y_true, y_pred, labels: list[str]) -> float:
    """Agreement above chance -- used for judge-vs-human agreement, and could
    equally be used for two independent intent labelers (see REPORT.md)."""
    cm = confusion_matrix(y_true, y_pred, labels)
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    row_marg = cm.sum(axis=1) / n
    col_marg = cm.sum(axis=0) / n
    pe = float(np.sum(row_marg * col_marg))
    if pe == 1.0:
        return 1.0
    return float((po - pe) / (1 - pe))


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)] if len(tokens) >= n else []


def rouge_l_f1(reference: str, candidate: str) -> float:
    """ROUGE-L (longest common subsequence) F1 between two texts, tokenized
    with our own tokenizer. Pure-python LCS -- fine at tweet-reply lengths."""
    ref = tokenize(reference, remove_stopwords=False)
    cand = tokenize(candidate, remove_stopwords=False)
    if not ref or not cand:
        return 0.0
    m, n = len(ref), len(cand)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == cand[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / n
    recall = lcs / m
    return 2 * precision * recall / (precision + recall)


def spearman_corr(x, y) -> float:
    """Spearman rank correlation, implemented directly (no scipy). Average
    ranks are used for ties, which is the standard convention."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0

    def rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(len(a), dtype=float)
        sorted_a = a[order]
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
                j += 1
            avg_rank = (i + j) / 2 + 1  # 1-indexed average rank for the tie block
            ranks[order[i:j + 1]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def unigram_overlap_f1(reference: str, candidate: str) -> float:
    ref = set(tokenize(reference))
    cand = set(tokenize(candidate))
    if not ref or not cand:
        return 0.0
    overlap = len(ref & cand)
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)
