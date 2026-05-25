"""my_ml_library — public API."""

from .knn import KNNClassifier
from .linear_regression import LinearRegression
from .evaluator import Evaluator, accuracy, mae, mse, rmse, precision, recall, f1_score
from .distances import DistanceMetric, DistanceMetricFactory

__version__ = "0.1.0"

__all__ = [
    "KNNClassifier",
    "LinearRegression",
    "Evaluator",
    "accuracy", "mae", "mse", "rmse",
    "precision", "recall", "f1_score",
    "DistanceMetric", "DistanceMetricFactory",
]
