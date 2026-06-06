# CoreLearn

A lightweight Python machine learning library built from scratch using only **NumPy**.  
Implements KNN classification and Linear Regression.

---

## Installation

```bash
# Download the project using pip:
pip install coreLearn 

```

After installation, import from anywhere:

```python
from coreLearn import KNNClassifier, LinearRegression, Evaluator
```

---

## Quick Start

```python
from coreLearn import KNNClassifier, LinearRegression, Evaluator, accuracy, mae

# --- KNN Classification ---
knn = KNNClassifier(k=5, distance="euclidean", n_jobs=2)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(accuracy(y_test, predictions))

# --- Linear Regression ---
lr = LinearRegression(strategy="normal")
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print(mae(y_test, predictions))

# --- Evaluator ---
print(Evaluator.evaluate_regression(y_test, predictions))
# {'mae': ..., 'mse': ..., 'rmse': ...}

print(Evaluator.evaluate_classification(y_test, knn_preds))
# {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ...}
```

---

## Package Structure

```
coreLearn/
├── __init__.py          ← Public API
├── base.py              ← Abstract base class — Template Method Pattern
├── distances.py         ← Distance metrics — Factory Pattern
├── knn.py               ← KNN Classifier — Recursion + Concurrency + OOP
├── linear_regression.py ← Linear Regression — Strategy Pattern + OOP
├── evaluator.py         ← Metric engine — Functional Programming
├── examples/
│   ├── demo_notebook.ipynb
│   ├── housing.csv
│   └── penguin.csv
└── tests/
    ├── test_knn.py
    ├── test_linear_regression.py
    ├── test_distances.py
    └── test_evaluator.py
```

---

## Running Tests

```bash
cd coreLearn/
pytest coreLearn/tests/ -v
```

---

## Learning Outcomes

### 1 — Object-Oriented Programming (OOP)

**File:** `base.py`, `knn.py`, `linear_regression.py`, `distances.py`

#### Abstract Base Class & Inheritance

`BaseModel` is an abstract class that defines the contract every model must follow.  
`KNNClassifier` and `LinearRegression` both inherit from it:

```python
# base.py
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y) -> "BaseModel": ...

    @abstractmethod
    def predict(self, X) -> list: ...

# knn.py
class KNNClassifier(BaseModel):   # ← inheritance
    def fit(self, X, y): ...
    def predict(self, X): ...

# linear_regression.py
class LinearRegression(BaseModel):  # ← inheritance
    def fit(self, X, y): ...
    def predict(self, X): ...
```

#### Polymorphism

Both models share the same interface — they can be used interchangeably:

```python
for model in [KNNClassifier(k=3), LinearRegression()]:
    model.fit(X_train, y_train)   # same call, different behaviour
    model.predict(X_test)         # same call, different behaviour
```

#### Encapsulation

Internal state is hidden with `_` prefixes. Users interact only through the public API:

```python
# knn.py
self._metric = DistanceMetricFactory.create(distance)  # private
self._tree   = None                                     # private

# linear_regression.py — controlled read access via properties
@property
def coef_(self) -> np.ndarray:
    return self._weights[1:]

@property
def intercept_(self) -> float:
    return float(self._weights[0])
```

`OptimizationStrategy`, `NormalEquationStrategy`, and `GradientDescentStrategy` inside  
`linear_regression.py` form an additional hierarchy demonstrating inheritance within the library.

---

### 2 — Functional Programming

**File:** `evaluator.py`

#### Functions as First-Class Objects

Metric functions are stored in dictionaries as values and called dynamically:

```python
# evaluator.py
_regression_metrics: dict[str, callable] = {
    "mae":  mae,
    "mse":  mse,
    "rmse": rmse,
}

@classmethod
def evaluate_regression(cls, y_true, y_pred) -> dict:
    # applies every registered function — no if/elif chain
    return {name: fn(y_true, y_pred) for name, fn in cls._regression_metrics.items()}
```

#### Higher-Order Function — `register()`

`Evaluator.register()` accepts any callable and plugs it in at runtime.  
This is the classic higher-order function pattern: a function (or method) that takes another function as an argument.

```python
# Add a custom metric without modifying the Evaluator class
Evaluator.register(
    "max_error",
    lambda y_true, y_pred: max(abs(a - b) for a, b in zip(y_true, y_pred)),
    kind="regression",
)
result = Evaluator.evaluate_regression(y_test, y_pred)
print(result["max_error"])   # available immediately
```

#### Pure Functions

`mae`, `mse`, `rmse`, `accuracy`, `precision`, `recall`, `f1_score` are all pure functions:
- No side effects
- No mutation of inputs
- Same inputs always produce the same output

```python
from coreLearn import mae, accuracy
mae([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])   # → 0.5  (always)
accuracy([0, 1, 1], [0, 1, 0])           # → 0.666 (always)
```

---

### 3 — Concurrency

**File:** `knn.py` — `KNNClassifier.predict()`

`KNNClassifier` uses `ProcessPoolExecutor` to classify test samples in parallel across  
multiple CPU processes. Unlike threads, each worker runs in its own process with its  
own GIL — enabling true CPU-bound parallelism.

```python
# knn.py
def predict(self, X) -> list:
    ...
    if self.n_jobs == 1:
        # sequential — no overhead for small datasets
        return [self._predict_one(x) for x in samples]

    # parallel — distribute samples across n_jobs worker processes
    args = [(self._tree, x, self.k, self._metric) for x in samples]
    with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
        return list(executor.map(_predict_worker, args))
```

```python
# n_jobs=1  → sequential (default, safe for notebooks)
knn = KNNClassifier(k=5, n_jobs=1)

# n_jobs=4  → 4 parallel worker processes
knn = KNNClassifier(k=5, n_jobs=4)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
```

> **Note:** `ProcessPoolExecutor` requires the `if __name__ == "__main__":` guard on  
> Windows/macOS when used in scripts. The `n_jobs=1` default is safe everywhere.

---

### 4 — Recursion

**File:** `knn.py` — `KDTree`

The KD-Tree data structure is built and searched using **mutual recursion**.  
Both `_build` and `_search` call themselves with a strictly smaller subproblem each time.

#### `_build` — Recursive Tree Construction

**Base case:** empty data → return `None`.  
**Recursive case:** split on the median, call `_build` on each half with `depth + 1`.

```python
# knn.py
def _build(self, data: list, depth: int):
    if not data:          # ← base case
        return None
    axis = depth % len(data[0][0])
    data.sort(key=lambda item: item[0][axis])
    mid = len(data) // 2
    return KDNode(
        point = data[mid][0],
        label = data[mid][1],
        left  = self._build(data[:mid],     depth + 1),  # ← recursion
        right = self._build(data[mid + 1:], depth + 1),  # ← recursion
    )
```

#### `_search` — Recursive Nearest-Neighbour Search

**Base case:** node is `None` → return.  
**Recursive case:** visit the near branch, then prune and optionally visit the far branch.

```python
# knn.py
def _search(self, node, target, k, metric, depth, best):
    if node is None:      # ← base case
        return
    dist = metric(target, node.point)
    # update best list ...
    self._search(near, target, k, metric, depth + 1, best)  # ← recursion
    if len(best) < k or abs(diff) < best[-1][0]:
        self._search(far, target, k, metric, depth + 1, best)  # ← recursion (pruned)
```

**Pruning:** the `abs(diff) < best[-1][0]` condition skips the far branch when it cannot  
contain a closer neighbour — achieving O(log n) average search complexity.

---

### 5 — SOLID Principles

**Files:** all modules

#### S — Single Responsibility

Every class has exactly one reason to change:

| Class | Sole Responsibility |
|-------|-------------------|
| `BaseModel` | Define the common model contract |
| `KDTree` | Spatial nearest-neighbour search |
| `KNNClassifier` | KNN classification logic |
| `LinearRegression` | Linear regression logic |
| `NormalEquationStrategy` | Closed-form weight computation |
| `GradientDescentStrategy` | Iterative gradient-based weight computation |
| `DistanceMetricFactory` | Instantiate distance metric objects by name |
| `Evaluator` | Compute and manage evaluation metrics |

#### O — Open/Closed

Classes are open for extension, closed for modification.  
New metrics and distance functions can be added **without editing any existing class**:

```python
# Add a new metric — Evaluator source code untouched
Evaluator.register("r2", lambda t, p: ..., kind="regression")

# Add a new distance — KNNClassifier source code untouched
DistanceMetricFactory.register("chebyshev", ChebyshevDistance)
knn = KNNClassifier(k=3, distance="chebyshev")
```

#### L — Liskov Substitution

Any `BaseModel` subclass can replace `BaseModel` without breaking callers:

```python
def train_and_score(model: BaseModel, X_train, y_train, X_test, y_test):
    preds = model.fit_predict(X_train, y_train, X_test)
    return accuracy(y_test, preds)

train_and_score(KNNClassifier(k=3), ...)   # works
train_and_score(LinearRegression(), ...)   # works
```

#### I — Interface Segregation

`DistanceMetric` exposes only what is needed — a single `compute()` method.  
Implementors are not forced to implement anything they do not use:

```python
# distances.py
class DistanceMetric(ABC):
    @abstractmethod
    def compute(self, a: list, b: list) -> float: ...
    # nothing else required
```

#### D — Dependency Inversion

`LinearRegression` depends on the **abstraction** `OptimizationStrategy`,  
not on any concrete strategy class:

```python
# linear_regression.py
self._weights = self._strategy.fit(X_b, y)
#               ↑ OptimizationStrategy interface — concrete class unknown here
```

---

### 6 — Architectural & Design Patterns

- **Core layer** (`base.py`, `distances.py`): abstractions and shared contracts  
- **Algorithm layer** (`knn.py`, `linear_regression.py`): concrete ML algorithms  
- **Evaluation layer** (`evaluator.py`): metric computation  
- **Public API** (`__init__.py`): single entry point, re-exports everything  

#### Pattern 1 — Template Method (`base.py`)

`fit_predict` defines the fixed skeleton (fit → predict).  
Subclasses fill in each step without altering the sequence:

```python
# base.py
def fit_predict(self, X_train, y_train, X_test) -> list:
    self.fit(X_train, y_train)   # ← step 1: implemented by subclass
    return self.predict(X_test)  # ← step 2: implemented by subclass
```

Every model gets `fit_predict` for free through inheritance.

#### Pattern 2 — Strategy (`linear_regression.py`)

The optimisation algorithm is swapped at construction time.  
`LinearRegression.fit()` never knows which concrete strategy it is using:

```python
lr_ne = LinearRegression(strategy="normal")           # uses NormalEquationStrategy
lr_gd = LinearRegression(strategy="gradient_descent") # uses GradientDescentStrategy

# Both models have the same interface — caller code is identical
lr_ne.fit(X_train, y_train)
lr_gd.fit(X_train, y_train)
```

To add a third optimiser (e.g. Adam), only a new `OptimizationStrategy` subclass is needed.

#### Pattern 3 — Factory (`distances.py`)

`DistanceMetricFactory` centralises object creation.  
`KNNClassifier` never imports `EuclideanDistance` or `ManhattanDistance` directly:

```python
# distances.py
class DistanceMetricFactory:
    _registry = {"euclidean": EuclideanDistance, "manhattan": ManhattanDistance}

    @classmethod
    def create(cls, name: str) -> DistanceMetric:
        return cls._registry[name]()   # create and return

    @classmethod
    def register(cls, name: str, metric_class: type) -> None:
        cls._registry[name] = metric_class  # extend without modifying

# knn.py — only depends on the factory, not the concrete classes
self._metric = DistanceMetricFactory.create(distance)
```

---

## API Reference

### `KNNClassifier`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | `int` | `5` | Number of neighbours |
| `distance` | `str` | `"euclidean"` | `"euclidean"` or `"manhattan"` (or any registered name) |
| `n_jobs` | `int` | `1` | Worker processes for prediction (`1` = sequential) |

### `LinearRegression`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | `str` | `"normal"` | `"normal"` (closed-form) or `"gradient_descent"` |
| `learning_rate` | `float` | `0.01` | Learning rate — gradient descent only |
| `epochs` | `int` | `1000` | Iterations — gradient descent only |

### `Evaluator`

| Method | Description |
|--------|-------------|
| `evaluate_regression(y_true, y_pred)` | Returns `{"mae", "mse", "rmse"}` |
| `evaluate_classification(y_true, y_pred)` | Returns `{"accuracy", "precision", "recall", "f1"}` |
| `register(name, fn, kind)` | Add a custom metric at runtime |

### Standalone metric functions

```python
from coreLearn import accuracy, mae, mse, rmse, precision, recall, f1_score
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Matrix operations, vectorised arithmetic |
| `pytest` | Unit testing |
| `scikit-learn` | Datasets and preprocessing in examples only |
| `pandas` | Data loading in examples only |
| `matplotlib` | Visualisation in examples only |
