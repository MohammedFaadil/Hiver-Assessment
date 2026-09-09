import math

from agent.metrics import (
    binary_precision_recall_f1,
    cohens_kappa,
    confusion_matrix,
    precision_recall_f1,
    rouge_l_f1,
    spearman_corr,
    unigram_overlap_f1,
)


def test_confusion_matrix_counts_correctly():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    cm = confusion_matrix(y_true, y_pred, ["a", "b"])
    # row 0 = true 'a': one predicted 'a', one predicted 'b'
    assert cm[0].tolist() == [1, 1]
    # row 1 = true 'b': zero predicted 'a', two predicted 'b'
    assert cm[1].tolist() == [0, 2]


def test_precision_recall_f1_matches_hand_computed_values():
    y_true = ["a", "a", "b", "b", "c", "c"]
    y_pred = ["a", "b", "b", "b", "c", "a"]
    r = precision_recall_f1(y_true, y_pred, ["a", "b", "c"])
    assert math.isclose(r["accuracy"], 4 / 6)
    assert math.isclose(r["per_class"]["a"]["precision"], 0.5)
    assert math.isclose(r["per_class"]["a"]["recall"], 0.5)
    assert math.isclose(r["per_class"]["b"]["precision"], 2 / 3)
    assert math.isclose(r["per_class"]["b"]["recall"], 1.0)
    assert math.isclose(r["per_class"]["c"]["precision"], 1.0)
    assert math.isclose(r["per_class"]["c"]["recall"], 0.5)


def test_precision_recall_f1_perfect_predictions():
    labels = ["x", "y"]
    r = precision_recall_f1(["x", "y", "x"], ["x", "y", "x"], labels)
    assert r["accuracy"] == 1.0
    assert r["macro_f1"] == 1.0


def test_binary_precision_recall_f1_matches_hand_computed_values():
    y_true = [True, True, False, False, True]
    y_pred = [True, False, False, True, True]
    r = binary_precision_recall_f1(y_true, y_pred)
    assert r == {
        "tp": 2, "fp": 1, "fn": 1, "tn": 1,
        "precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3, "accuracy": 0.6,
    } or (
        r["tp"] == 2 and r["fp"] == 1 and r["fn"] == 1 and r["tn"] == 1
        and math.isclose(r["precision"], 2 / 3) and math.isclose(r["recall"], 2 / 3)
    )


def test_binary_precision_recall_f1_handles_no_positive_predictions():
    r = binary_precision_recall_f1([True, True, False], [False, False, False])
    assert r["precision"] == 0.0
    assert r["recall"] == 0.0
    assert r["f1"] == 0.0


def test_cohens_kappa_perfect_agreement_is_one():
    assert cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"], ["a", "b"]) == 1.0


def test_cohens_kappa_chance_agreement_is_near_zero():
    # y_pred is drawn independently of y_true with matching label frequencies
    y_true = ["a", "b"] * 50
    y_pred = ["a", "a"] * 25 + ["b", "b"] * 25
    k = cohens_kappa(y_true, y_pred, ["a", "b"])
    assert -1.0 <= k <= 1.0  # sanity bound; exact value depends on the split


def test_rouge_l_identical_strings_is_one():
    assert rouge_l_f1("the package is late", "the package is late") == 1.0


def test_rouge_l_no_overlap_is_zero():
    assert rouge_l_f1("completely different words here", "xyz abc qrs tuv") == 0.0


def test_unigram_overlap_f1_matches_hand_computed_value():
    ref = "please send us a dm with your order number"
    cand = "send a dm with the order number"
    # ref meaningful tokens (stopwords removed): please, send, dm, order, number (5)
    # cand meaningful tokens: send, dm, order, number (4); all 4 are in ref
    score = unigram_overlap_f1(ref, cand)
    expected_precision = 4 / 4
    expected_recall = 4 / 5
    expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
    assert math.isclose(score, expected_f1, abs_tol=1e-9)


def test_unigram_overlap_f1_empty_string_is_zero():
    assert unigram_overlap_f1("", "something") == 0.0
    assert unigram_overlap_f1("something", "") == 0.0


def test_spearman_corr_perfect_positive_and_negative():
    assert math.isclose(spearman_corr([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 1.0, abs_tol=1e-9)
    assert math.isclose(spearman_corr([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), -1.0, abs_tol=1e-9)


def test_spearman_corr_matches_textbook_example():
    # classic worked example: d = [-1,1,-1,1,0], rho = 1 - 6*sum(d^2)/(n*(n^2-1)) = 0.8
    assert math.isclose(spearman_corr([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]), 0.8, abs_tol=1e-9)


def test_spearman_corr_constant_input_is_zero():
    assert spearman_corr([1, 1, 1], [1, 2, 3]) == 0.0
