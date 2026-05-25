from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .base import BaseModel
from .distances import DistanceMetricFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _majority_vote(labels: list):
    """Return the most frequent label in the list."""
    counts: dict = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=lambda k: counts[k])


def _predict_worker(args: tuple):
    """
    Module-level worker for ProcessPoolExecutor.

    Must be defined at module level (not nested) so that Python's
    multiprocessing can pickle it and send it to worker processes.

    Each worker process receives its own copy of the KD-Tree and metric
    via pickle — no shared memory, no data race.
    """
    tree, sample, k, metric = args
    neighbours = tree.nearest_k(sample, k, metric)
    return _majority_vote(neighbours)


# ---------------------------------------------------------------------------
# KD-Tree
# ---------------------------------------------------------------------------

class KDNode:
    """A single node in a KD-Tree."""

    def __init__(self, point: list, label, left=None, right=None):
        self.point = point
        self.label = label
        self.left = left
        self.right = right


class KDTree:
    """
    Binary space-partitioning tree for fast nearest-neighbour lookups.

    Both ``_build`` and ``_search`` are **recursive**, satisfying the
    Recursion learning outcome.
    """

    def __init__(self, points: list, labels: list):
        data = list(zip(points, labels))
        self.root = self._build(data, depth=0)

    # -- recursive build --

    def _build(self, data: list, depth: int):
        """Split data along alternating axes and return the root KDNode."""
        if not data:
            return None
        k = len(data[0][0])
        axis = depth % k
        data.sort(key=lambda item: item[0][axis])
        mid = len(data) // 2
        return KDNode(
            point=data[mid][0],
            label=data[mid][1],
            left=self._build(data[:mid], depth + 1),
            right=self._build(data[mid + 1:], depth + 1),
        )

    # -- public query --

    def nearest_k(self, target: list, k: int, metric) -> list:
        """Return labels of the k nearest neighbours to *target*."""
        best: list = []
        self._search(self.root, target, k, metric, depth=0, best=best)
        best.sort(key=lambda x: x[0])
        return [label for _, label in best[:k]]

    # -- recursive search --

    def _search(self, node, target: list, k: int,
                metric, depth: int, best: list) -> None:
        """Recursively prune branches using the splitting-plane distance."""
        if node is None:
            return
        dist = metric(target, node.point)

        if len(best) < k:
            best.append((dist, node.label))
        elif dist < best[-1][0]:
            best[-1] = (dist, node.label)
        best.sort(key=lambda x: x[0])

        axis = depth % len(target)
        diff = target[axis] - node.point[axis]
        near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)

        self._search(near, target, k, metric, depth + 1, best)
        if len(best) < k or abs(diff) < best[-1][0]:
            self._search(far, target, k, metric, depth + 1, best)


# ---------------------------------------------------------------------------
# KNN Classifier
# ---------------------------------------------------------------------------

class KNNClassifier(BaseModel):
    """
    K-Nearest Neighbours classifier.

    Uses a KD-Tree for efficient lookups (**Recursion**) and
    ``ProcessPoolExecutor`` for **parallel** prediction across test samples
    (**Concurrency**). Unlike threading, each worker runs in a separate
    process with its own GIL — enabling true CPU-bound parallelism.

    Parameters
    ----------
    k        : number of neighbours
    distance : 'euclidean' (default) or 'manhattan'
    n_jobs   : number of parallel worker processes during predict
               (1 = no multiprocessing, sequential)
    """

    def __init__(self, k: int = 5, distance: str = "euclidean", n_jobs: int = 1):
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive integer, got: {k!r}.")
        if not isinstance(n_jobs, int) or n_jobs < 1:
            raise ValueError(f"n_jobs must be a positive integer, got: {n_jobs!r}.")
        self.k = k
        self.n_jobs = n_jobs
        self._metric = DistanceMetricFactory.create(distance) 
        self._tree: KDTree | None = None
        self._n_train: int = 0
        self._n_features: int = 0

    def fit(self, X, y) -> "KNNClassifier":
        """Build the internal KD-Tree from training data."""
        X = np.array(X, dtype=float)
        y = list(y)

        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-D (n_samples, n_features), got shape {X.shape}."
            )
        if len(X) == 0:
            raise ValueError("Training data X must not be empty.")
        if len(y) == 0:
            raise ValueError("Label vector y must not be empty.")
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same number of samples: {len(X)} != {len(y)}."
            )
        if self.k > len(X):
            raise ValueError(
                f"k ({self.k}) cannot be greater than the number of "
                f"training samples ({len(X)})."
            )

        self._n_train = len(X)
        self._n_features = X.shape[1]
        self._tree = KDTree(X.tolist(), y)
        return self

    def _predict_one(self, x: list):
        """Classify a single sample by majority vote among k neighbours."""
        neighbours = self._tree.nearest_k(x, self.k, self._metric)
        return _majority_vote(neighbours)

    def predict(self, X) -> list:
        """
        Predict class labels for all samples in X.

        n_jobs=1 : sequential prediction (no process overhead).
        n_jobs>1 : samples are distributed across ``n_jobs`` worker processes
                   via ``ProcessPoolExecutor``. Each process receives a
                   pickled copy of the KD-Tree and metric — no shared memory,
                   no data race, true CPU-level parallelism.
        """
        if self._tree is None:
            raise RuntimeError("Call fit() before predict().")
        raw = list(X) if not isinstance(X, np.ndarray) else X
        if len(raw) == 0:
            return []
        X = np.array(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-D (n_samples, n_features), got shape {X.shape}."
            )
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Feature count mismatch: model was trained with {self._n_features} "
                f"features, but X has {X.shape[1]}."
            )
        samples: list = X.tolist()

        if self.n_jobs == 1:
            return [self._predict_one(x) for x in samples]

        args = [(self._tree, x, self.k, self._metric) for x in samples]
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            return list(executor.map(_predict_worker, args))
