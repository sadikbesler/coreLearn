from abc import ABC, abstractmethod

import numpy as np

from .base import BaseModel


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------

class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit weights to the given design matrix X and target vector y."""


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class NormalEquationStrategy(OptimizationStrategy):
    """
    Closed-form solution: w = argmin ||Xw - y||^2

    Uses np.linalg.lstsq (SVD-based) instead of the textbook (X^T X)^-1 X^T y
    formula. Direct inversion squares the condition number of X and causes
    numerical warnings on ill-conditioned data; lstsq avoids this entirely.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        weights, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return weights


class GradientDescentStrategy(OptimizationStrategy):
    """
    Batch gradient descent: w <- w - lr * (1/n) * X^T (Xw - y)

    Numerical stability:
    - np.errstate(all='ignore') suppresses NumPy overflow/invalid warnings
      that would otherwise fire during the matmul before any check runs.
    - np.isfinite guards catch divergence and stop the loop early.

    Tip: normalize X and y with StandardScaler before training on
    large or unscaled datasets (see example.py section 2b).
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000) -> None:
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        w: np.ndarray = np.zeros(X.shape[1])
        n: int = len(y)

        with np.errstate(all='ignore'):
            for _ in range(self.epochs):
                residual: np.ndarray = X @ w - y
                if not np.isfinite(residual).all():
                    break
                grad: np.ndarray = X.T @ residual / n
                if not np.isfinite(grad).all():
                    break
                w -= self.learning_rate * grad

        return w


# ---------------------------------------------------------------------------
# LinearRegression model
# ---------------------------------------------------------------------------

class LinearRegression(BaseModel):
    """
    Linear Regression model: y = X * w + b

    The optimization algorithm is selected by name and encapsulated in an
    OptimizationStrategy object — callers never interact with the strategy
    directly (Strategy Pattern).

    Parameters
    ----------
    strategy      : 'normal' (default) or 'gradient_descent'
    learning_rate : learning rate — gradient_descent only
    epochs        : number of iterations — gradient_descent only
    """

    def __init__(
        self,
        strategy: str = "normal",
        learning_rate: float = 0.01,
        epochs: int = 1000,
    ) -> None:
        if strategy == "normal":
            self._strategy: OptimizationStrategy = NormalEquationStrategy()
        elif strategy == "gradient_descent":
            self._strategy = GradientDescentStrategy(
                learning_rate=learning_rate, epochs=epochs
            )
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Use 'normal' or 'gradient_descent'."
            )
        self._weights: np.ndarray | None = None
        self._n_features: int = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Fit the model to training data X and target vector y."""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)   # (n,) -> (n, 1) single-feature convenience
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-D (n_samples, n_features), got shape {X.shape}."
            )
        if len(X) == 0:
            raise ValueError("Training data X must not be empty.")
        if len(y) == 0:
            raise ValueError("Target vector y must not be empty.")
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same number of samples: {len(X)} != {len(y)}."
            )

        self._n_features = X.shape[1]
        X_b: np.ndarray = np.c_[np.ones(len(X)), X]  
        self._weights = self._strategy.fit(X_b, y)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> list[float]:
        """
        Return predictions as a plain Python list.

        Returns list (not np.ndarray) to stay consistent with
        BaseModel.predict() — satisfying the Liskov Substitution Principle.
        Use predict_array() when NumPy output is needed.
        """
        if self._weights is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-D (n_samples, n_features), got shape {X.shape}."
            )
        if len(X) == 0:
            return []
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Feature count mismatch: model was trained with {self._n_features} "
                f"features, but X has {X.shape[1]}."
            )
        X_b: np.ndarray = np.c_[np.ones(len(X)), X]
        with np.errstate(all='ignore'): 
            return (X_b @ self._weights).tolist()

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        """Return predictions as a NumPy array."""
        if self._weights is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-D (n_samples, n_features), got shape {X.shape}."
            )
        if len(X) == 0:
            return np.array([])
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Feature count mismatch: model was trained with {self._n_features} "
                f"features, but X has {X.shape[1]}."
            )
        X_b: np.ndarray = np.c_[np.ones(len(X)), X]
        with np.errstate(all='ignore'): 
            return X_b @ self._weights

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def coef_(self) -> np.ndarray:
        """Learned feature coefficients (excludes the bias term)."""
        return self._weights[1:]

    @property
    def intercept_(self) -> float:
        """Learned bias (intercept) term."""
        return float(self._weights[0])
