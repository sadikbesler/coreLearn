"""Unit tests for KNNClassifier."""

import pytest
from coreLearn import KNNClassifier


def test_basic_classification():
    X_train = [[1, 1], [2, 2], [3, 3], [7, 7], [8, 8], [9, 9]]
    y_train = [0, 0, 0, 1, 1, 1]
    model = KNNClassifier(k=3)
    model.fit(X_train, y_train)
    assert model.predict([[2, 2]]) == [0]
    assert model.predict([[8, 8]]) == [1]


def test_manhattan_distance():
    X_train = [[0, 0], [1, 0], [0, 1], [5, 5], [6, 5], [5, 6]]
    y_train = ["A", "A", "A", "B", "B", "B"]
    model = KNNClassifier(k=3, distance="manhattan")
    model.fit(X_train, y_train)
    preds = model.predict([[0.5, 0.5], [5.5, 5.5]])
    assert preds == ["A", "B"]


def test_parallel_predict():
    X_train = [[i] for i in range(20)]
    y_train = [0 if i < 10 else 1 for i in range(20)]
    model = KNNClassifier(k=3, n_jobs=2)
    model.fit(X_train, y_train)
    preds = model.predict([[4], [15]])
    assert preds == [0, 1]


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        KNNClassifier().predict([[1, 2]])


def test_unknown_distance_raises():
    with pytest.raises(ValueError):
        KNNClassifier(distance="cosine")


def test_fit_predict_template_method():
    model = KNNClassifier(k=1)
    preds = model.fit_predict([[0], [10]], [0, 1], [[1], [9]])
    assert preds == [0, 1]


# ---------------------------------------------------------------------------
# Parametre doğrulama — fit()
# ---------------------------------------------------------------------------

def test_k_greater_than_n_samples_raises():
    """k, eğitim örnek sayısından büyük olamaz."""
    model = KNNClassifier(k=10)
    with pytest.raises(ValueError, match="k"):
        model.fit([[1], [2], [3]], [0, 1, 0])


def test_k_zero_raises():
    """k sıfır veya negatif olamaz."""
    with pytest.raises(ValueError):
        KNNClassifier(k=0)


def test_k_negative_raises():
    """k negatif olamaz."""
    with pytest.raises(ValueError):
        KNNClassifier(k=-1)


def test_n_jobs_zero_raises():
    """n_jobs sıfır olamaz."""
    with pytest.raises(ValueError):
        KNNClassifier(n_jobs=0)


def test_n_jobs_negative_raises():
    """n_jobs negatif olamaz."""
    with pytest.raises(ValueError):
        KNNClassifier(n_jobs=-2)


# ---------------------------------------------------------------------------
# Parametre doğrulama — predict()
# ---------------------------------------------------------------------------

def test_feature_mismatch_raises():
    """Tahmin sırasında özellik sayısı eğitimle uyuşmalı."""
    model = KNNClassifier(k=1)
    model.fit([[1, 2], [3, 4]], [0, 1])
    with pytest.raises(ValueError, match="[Ff]eature"):
        model.predict([[1]])          # 1 özellik; model 2 ile eğitildi


def test_predict_empty_returns_empty():
    """Boş X verildiğinde predict boş liste döndürür."""
    model = KNNClassifier(k=1)
    model.fit([[1], [2]], [0, 1])
    assert model.predict([]) == []
