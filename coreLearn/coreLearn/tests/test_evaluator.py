"""Unit tests for evaluator metric functions and Evaluator class."""

import pytest
from coreLearn import (
    Evaluator,
    accuracy, mae, mse, rmse,
    precision, recall, f1_score,
)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def test_mae():
    assert mae([1.0, 2.0, 3.0], [1.5, 2.5, 3.5]) == pytest.approx(0.5)


def test_mse():
    assert mse([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]) == pytest.approx(1.0)


def test_rmse():
    assert rmse([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]) == pytest.approx(1.0)


def test_mae_perfect():
    assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def test_accuracy():
    assert accuracy([0, 1, 1, 0, 1], [0, 1, 0, 0, 1]) == pytest.approx(0.8)


def test_accuracy_perfect():
    assert accuracy([1, 0, 1], [1, 0, 1]) == pytest.approx(1.0)


def test_precision_perfect():
    assert precision([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_recall_binary():
    assert recall([1, 1, 0], [1, 0, 0]) == pytest.approx(0.75)


def test_f1_perfect():
    assert f1_score([0, 1, 0, 1], [0, 1, 0, 1]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_empty_raises():
    with pytest.raises(ValueError):
        accuracy([], [])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        mae([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

def test_evaluate_regression_keys():
    result = Evaluator.evaluate_regression([1.0, 2.0], [1.0, 2.0])
    assert set(result.keys()) == {"mae", "mse", "rmse"}


def test_evaluate_regression_perfect():
    result = Evaluator.evaluate_regression([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert result["mae"] == pytest.approx(0.0)
    assert result["mse"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)


def test_evaluate_classification_keys():
    result = Evaluator.evaluate_classification([0, 1, 0, 1], [0, 1, 0, 1])
    assert set(result.keys()) == {"accuracy", "precision", "recall", "f1"}


def test_evaluate_classification_perfect():
    result = Evaluator.evaluate_classification([0, 1, 0, 1], [0, 1, 0, 1])
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_register_regression():
    Evaluator.register("max_err", lambda t, p: max(abs(a - b) for a, b in zip(t, p)), kind="regression")
    result = Evaluator.evaluate_regression([1.0, 2.0, 3.0], [1.5, 2.0, 3.5])
    assert result["max_err"] == pytest.approx(0.5)


def test_register_unknown_kind_raises():
    with pytest.raises(ValueError):
        Evaluator.register("bad", lambda t, p: 0.0, kind="unknown")
