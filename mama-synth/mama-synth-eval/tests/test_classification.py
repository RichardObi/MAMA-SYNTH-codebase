"""Unit tests for classification module."""

import numpy as np
import pytest

from mama_sia_eval.classification import (
    compute_auroc,
    compute_balanced_accuracy,
    evaluate_classification,
)


class TestAUROC:
    """Tests for AUROC computation."""

    def test_perfect_prediction(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])
        assert compute_auroc(y_true, y_prob) == pytest.approx(1.0)

    def test_random_prediction(self) -> None:
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)
        y_prob = rng.rand(100)
        auc = compute_auroc(y_true, y_prob)
        # Random predictions → AUROC ~ 0.5
        assert 0.3 < auc < 0.7

    def test_inverse_prediction(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        assert compute_auroc(y_true, y_prob) == pytest.approx(0.0)

    def test_single_class_returns_nan(self) -> None:
        y_true = np.array([0, 0, 0])
        y_prob = np.array([0.1, 0.5, 0.9])
        result = compute_auroc(y_true, y_prob)
        assert result is None or np.isnan(result)


class TestBalancedAccuracy:
    """Tests for balanced accuracy computation."""

    def test_perfect_prediction(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert compute_balanced_accuracy(y_true, y_pred) == pytest.approx(1.0)

    def test_all_wrong(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        assert compute_balanced_accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_imbalanced_classes(self) -> None:
        # 90 negatives, 10 positives; predict all negative
        y_true = np.array([0] * 90 + [1] * 10)
        y_pred = np.array([0] * 100)
        ba = compute_balanced_accuracy(y_true, y_pred)
        # TPR=0, TNR=1 → balanced accuracy = 0.5
        assert ba == pytest.approx(0.5)


class TestEvaluateClassification:
    """Tests for evaluate_classification end-to-end."""

    def test_returns_expected_keys(self) -> None:
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 50)
        y_prob = rng.rand(50)
        result = evaluate_classification(y_true, y_prob)
        assert "auroc" in result
        assert "balanced_accuracy" in result

    def test_good_predictions_high_metrics(self) -> None:
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 50 + [1] * 50)
        # Good predictions with some noise
        y_prob = np.where(y_true == 1, 0.8, 0.2) + rng.randn(100) * 0.05
        y_prob = np.clip(y_prob, 0, 1)
        result = evaluate_classification(y_true, y_prob)
        assert result["auroc"] > 0.9
        assert result["balanced_accuracy"] > 0.85

    def test_threshold_parameter(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.3, 0.4, 0.6, 0.7])
        result_default = evaluate_classification(y_true, y_prob, threshold=0.5)
        result_high = evaluate_classification(y_true, y_prob, threshold=0.9)
        # With high threshold, all predicted as 0 → worse balanced accuracy
        assert result_default["balanced_accuracy"] > result_high["balanced_accuracy"]
