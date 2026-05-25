"""Unit tests for LinearRegression."""

import pytest
from coreLearn import LinearRegression


def test_normal_equation_perfect_line():
    X = [[1], [2], [3], [4], [5]]
    y = [2, 4, 6, 8, 10]        # y = 2x
    model = LinearRegression()
    model.fit(X, y)
    assert abs(model.predict([[6]])[0] - 12) < 0.01


def test_coef_and_intercept():
    X = [[1], [2], [3]]
    y = [3, 5, 7]               # y = 2x + 1
    model = LinearRegression()
    model.fit(X, y)
    assert abs(model.intercept_ - 1.0) < 0.01
    assert abs(model.coef_[0] - 2.0) < 0.01


def test_gradient_descent_strategy():
    X = [[1], [2], [3], [4], [5]]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    model = LinearRegression(strategy="gradient_descent", learning_rate=0.1, epochs=2000)
    model.fit(X, y)
    assert abs(model.predict([[6]])[0] - 12) < 0.5


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        LinearRegression().predict([[1]])


def test_fit_empty_raises():
    with pytest.raises(ValueError):
        LinearRegression().fit([], [])


def test_feature_mismatch_raises():
    model = LinearRegression()
    model.fit([[1, 2], [3, 4]], [1.0, 2.0])
    with pytest.raises(ValueError):
        model.predict([[1]])          # 1 feature, model trained on 2


def test_predict_empty_returns_empty():
    model = LinearRegression()
    model.fit([[1], [2]], [1.0, 2.0])
    assert model.predict([]) == []


def test_fit_predict_template_method():
    model = LinearRegression()
    preds = model.fit_predict([[1], [2]], [1.0, 2.0], [[3]])
    assert abs(preds[0] - 3.0) < 0.1


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        LinearRegression(strategy="unknown")


# ---------------------------------------------------------------------------
# predict_array()
# ---------------------------------------------------------------------------

def test_predict_array_returns_ndarray():
    """predict_array() numpy dizisi döndürmeli."""
    import numpy as np
    model = LinearRegression()
    model.fit([[1], [2], [3]], [2.0, 4.0, 6.0])
    result = model.predict_array([[4]])
    assert isinstance(result, np.ndarray)
    assert abs(result[0] - 8.0) < 0.01


def test_predict_array_empty_returns_empty():
    """Boş X → boş numpy dizisi."""
    import numpy as np
    model = LinearRegression()
    model.fit([[1], [2]], [1.0, 2.0])
    result = model.predict_array([])
    assert isinstance(result, np.ndarray)
    assert len(result) == 0


def test_predict_array_before_fit_raises():
    """fit() öncesi predict_array() çağrısı RuntimeError fırlatmalı."""
    with pytest.raises(RuntimeError):
        LinearRegression().predict_array([[1]])


def test_predict_array_feature_mismatch_raises():
    """predict_array() özellik sayısı uyuşmazlığında ValueError fırlatmalı."""
    model = LinearRegression()
    model.fit([[1, 2], [3, 4]], [1.0, 2.0])
    with pytest.raises(ValueError):
        model.predict_array([[1]])    # 1 özellik; model 2 ile eğitildi


# ---------------------------------------------------------------------------
# Çok özellikli (multi-feature) regresyon
# ---------------------------------------------------------------------------

def test_multi_feature_normal_equation():
    """İki özellikli: y = x1 + 2*x2."""
    X = [[1, 1], [2, 1], [1, 2], [2, 2]]
    y = [3.0, 4.0, 5.0, 6.0]          # 1*x1 + 2*x2
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict([[3, 3]])[0]  # beklenen: 1*3 + 2*3 = 9
    assert abs(pred - 9.0) < 0.1
    assert len(model.coef_) == 2


def test_multi_feature_gradient_descent():
    """İki özellikli gradient descent."""
    X = [[1, 0], [0, 1], [2, 0], [0, 2]]
    y = [2.0, 3.0, 4.0, 6.0]          # 2*x1 + 3*x2
    model = LinearRegression(strategy="gradient_descent", learning_rate=0.05, epochs=5000)
    model.fit(X, y)
    pred = model.predict([[1, 1]])[0]  # beklenen: 2 + 3 = 5
    assert abs(pred - 5.0) < 0.5


# ---------------------------------------------------------------------------
# Girdi doğrulama — fit()
# ---------------------------------------------------------------------------

def test_fit_xy_length_mismatch_raises():
    """X ve y farklı uzunluktaysa ValueError fırlatmalı."""
    with pytest.raises(ValueError):
        LinearRegression().fit([[1], [2], [3]], [1.0, 2.0])


# ---------------------------------------------------------------------------
# coef_ / intercept_ — fit() öncesi erişim
# ---------------------------------------------------------------------------

def test_coef_before_fit_raises():
    """fit() çağrılmadan coef_'e erişim TypeError/AttributeError fırlatmalı."""
    model = LinearRegression()
    with pytest.raises((TypeError, AttributeError)):
        _ = model.coef_


def test_intercept_before_fit_raises():
    """fit() çağrılmadan intercept_'e erişim TypeError/AttributeError fırlatmalı."""
    model = LinearRegression()
    with pytest.raises((TypeError, AttributeError)):
        _ = model.intercept_
