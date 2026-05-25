from abc import ABC, abstractmethod

import numpy as np


class DistanceMetric(ABC):
    """
    Abstract distance metric.

    Subclasses only need to implement ``compute()``.
    ``__call__`` support lets an instance be used like a function:
    ``metric(a, b)``  →  ``metric.compute(a, b)``
    """

    @abstractmethod
    def compute(self, a: list, b: list) -> float:
        """Compute and return the distance between two points."""

    def __call__(self, a: list, b: list) -> float:
        """Allow the object to be called as a function."""
        return self.compute(a, b)


class EuclideanDistance(DistanceMetric):
    """
    Euclidean (L2) distance: √Σ(aᵢ − bᵢ)²

    General-purpose; suitable for continuous, scaled data.
    """

    def compute(self, a: list, b: list) -> float:
        a_arr, b_arr = np.array(a), np.array(b)
        return float(np.sqrt(np.sum((a_arr - b_arr) ** 2)))


class ManhattanDistance(DistanceMetric):
    """
    Manhattan (L1) distance: Σ|aᵢ − bᵢ|

    Preferred over Euclidean for grid-like spaces or when
    robustness to outliers is desired.
    """

    def compute(self, a: list, b: list) -> float:
        a_arr, b_arr = np.array(a), np.array(b)
        return float(np.sum(np.abs(a_arr - b_arr)))


class DistanceMetricFactory:
    """
    **Factory Pattern** — creates ``DistanceMetric`` instances by name.

    ``KNNClassifier`` does not import concrete classes (``EuclideanDistance``,
    etc.) directly; it simply passes a name to this factory.  Benefits:

    * Adding a new metric requires no changes to ``KNNClassifier``.
    * Decouples metric classes from consumer code (loose coupling).

    Usage::

        metric = DistanceMetricFactory.create("euclidean")
        dist   = metric([1, 2], [4, 6])   # → 5.0

    Registering a new metric::

        class ChebyshevDistance(DistanceMetric):
            def compute(self, a, b):
                return float(max(abs(x - y) for x, y in zip(a, b)))

        DistanceMetricFactory.register("chebyshev", ChebyshevDistance)
    """

    _registry: dict[str, type[DistanceMetric]] = {
        "euclidean": EuclideanDistance,
        "manhattan": ManhattanDistance,
    }

    @classmethod
    def create(cls, name: str) -> "DistanceMetric":
        """
        Instantiate and return a ``DistanceMetric`` for the given name.

        Parameters
        ----------
        name : str
            Metric name — see ``available()`` for registered options.

        Raises
        ------
        ValueError
            If an unknown name is provided.
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown distance metric: '{name}'. "
                f"Available: {cls.available()}"
            )
        return cls._registry[name]()

    @classmethod
    def available(cls) -> list[str]:
        """Return a list of all registered metric names."""
        return list(cls._registry)

    @classmethod
    def register(cls, name: str, metric_class: type[DistanceMetric]) -> None:
        """
        Add a new distance metric class to the registry.

        Parameters
        ----------
        name         : str                    — Lookup key.
        metric_class : type[DistanceMetric]   — Concrete class (not yet instantiated).

        Raises
        ------
        TypeError
            If ``metric_class`` is not a subclass of ``DistanceMetric``.
        """
        if not (isinstance(metric_class, type) and
                issubclass(metric_class, DistanceMetric)):
            raise TypeError(
                f"{metric_class} must be a subclass of DistanceMetric."
            )
        cls._registry[name] = metric_class
