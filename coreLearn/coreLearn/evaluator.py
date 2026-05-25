import numpy as np


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    yt: np.ndarray = np.array(list(y_true), dtype=float)
    yp: np.ndarray = np.array(list(y_pred), dtype=float)
    if len(yt) == 0:
        raise ValueError("y_true must not be empty.")
    if len(yt) != len(yp):
        raise ValueError(f"y_true and y_pred must have the same length: {len(yt)} != {len(yp)}")
    return float(np.mean(np.abs(yt - yp)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    yt: np.ndarray = np.array(list(y_true), dtype=float)
    yp: np.ndarray = np.array(list(y_pred), dtype=float)
    if len(yt) == 0:
        raise ValueError("y_true must not be empty.")
    if len(yt) != len(yp):
        raise ValueError(f"y_true and y_pred must have the same length: {len(yt)} != {len(yp)}")
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly predicted labels."""
    yt: list = list(y_true)
    yp: list = list(y_pred)
    if len(yt) == 0:
        raise ValueError("y_true must not be empty.")
    if len(yt) != len(yp):
        raise ValueError(f"y_true and y_pred must have the same length: {len(yt)} != {len(yp)}")
    return sum(t == p for t, p in zip(yt, yp)) / len(yt)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged precision across all classes."""
    yt: np.ndarray = np.array(y_true)
    yp: np.ndarray = np.array(y_pred)
    classes: np.ndarray = np.unique(yt)
    scores: list[float] = []
    for c in classes:
        tp: int = int(np.sum((yp == c) & (yt == c)))
        fp: int = int(np.sum((yp == c) & (yt != c)))
        scores.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return float(np.mean(scores))


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged recall across all classes."""
    yt: np.ndarray = np.array(y_true)
    yp: np.ndarray = np.array(y_pred)
    classes: np.ndarray = np.unique(yt)
    scores: list[float] = []
    for c in classes:
        tp: int = int(np.sum((yp == c) & (yt == c)))
        fn: int = int(np.sum((yp != c) & (yt == c)))
        scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return float(np.mean(scores))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 score."""
    yt: np.ndarray = np.array(y_true)
    yp: np.ndarray = np.array(y_pred)
    classes: np.ndarray = np.unique(yt)
    scores: list[float] = []
    for c in classes:
        tp: int = int(np.sum((yp == c) & (yt == c)))
        fp: int = int(np.sum((yp == c) & (yt != c)))
        fn: int = int(np.sum((yp != c) & (yt == c)))
        p: float = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r: float = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Runs registered metrics by kind: regression or classification.

    Built-in metrics are pre-registered per kind. New metrics can be added
    at runtime via register() without modifying this class (Open/Closed Principle).
    """

    _regression_metrics: dict[str, callable] = {
        "mae":  mae,
        "mse":  mse,
        "rmse": rmse,
    }

    _classification_metrics: dict[str, callable] = {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1_score,
    }

    @classmethod
    def register(cls, name: str, fn: callable, kind: str = "regression") -> None:
        """
        Register a new metric function.

        Parameters
        ----------
        name : metric name used as dict key
        fn   : callable with signature (y_true, y_pred) -> float
        kind : 'regression' (default) or 'classification'
        """
        if kind == "regression":
            cls._regression_metrics[name] = fn
        elif kind == "classification":
            cls._classification_metrics[name] = fn
        else:
            raise ValueError(f"Unknown kind '{kind}'. Use 'regression' or 'classification'.")

    @classmethod
    def evaluate_regression(cls, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Run all registered regression metrics (mae, mse, rmse, ...)."""
        return {name: fn(y_true, y_pred) for name, fn in cls._regression_metrics.items()}

    @classmethod
    def evaluate_classification(cls, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Run all registered classification metrics (accuracy, precision, recall, f1, ...)."""
        return {name: fn(y_true, y_pred) for name, fn in cls._classification_metrics.items()}
