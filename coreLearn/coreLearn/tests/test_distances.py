"""Unit tests for DistanceMetric subclasses and DistanceMetricFactory."""

import math

import pytest
from coreLearn import DistanceMetric, DistanceMetricFactory


# ---------------------------------------------------------------------------
# EuclideanDistance
# ---------------------------------------------------------------------------

def test_euclidean_basic():
    """3-4-5 üçgeni: sqrt(3²+4²) = 5."""
    metric = DistanceMetricFactory.create("euclidean")
    assert metric.compute([0, 0], [3, 4]) == pytest.approx(5.0)


def test_euclidean_same_point():
    """Aynı noktalar arası mesafe sıfır olmalı."""
    metric = DistanceMetricFactory.create("euclidean")
    assert metric.compute([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_euclidean_1d():
    """Tek boyutlu: |7 - 3| = 4."""
    metric = DistanceMetricFactory.create("euclidean")
    assert metric.compute([3], [7]) == pytest.approx(4.0)


def test_euclidean_symmetry():
    """Mesafe simetriktir: d(a,b) == d(b,a)."""
    metric = DistanceMetricFactory.create("euclidean")
    assert metric.compute([1, 2], [5, 6]) == pytest.approx(metric.compute([5, 6], [1, 2]))


def test_euclidean_callable():
    """DistanceMetric nesnesi fonksiyon gibi çağrılabilmeli (__call__)."""
    metric = DistanceMetricFactory.create("euclidean")
    assert metric([0, 0], [3, 4]) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# ManhattanDistance
# ---------------------------------------------------------------------------

def test_manhattan_basic():
    """|3-0| + |4-0| = 7."""
    metric = DistanceMetricFactory.create("manhattan")
    assert metric.compute([0, 0], [3, 4]) == pytest.approx(7.0)


def test_manhattan_same_point():
    """Aynı noktalar arası mesafe sıfır olmalı."""
    metric = DistanceMetricFactory.create("manhattan")
    assert metric.compute([2, 5], [2, 5]) == pytest.approx(0.0)


def test_manhattan_1d():
    """Tek boyutlu: |1 - 9| = 8."""
    metric = DistanceMetricFactory.create("manhattan")
    assert metric.compute([1], [9]) == pytest.approx(8.0)


def test_manhattan_symmetry():
    """Mesafe simetriktir: d(a,b) == d(b,a)."""
    metric = DistanceMetricFactory.create("manhattan")
    assert metric.compute([0, 1], [4, 6]) == pytest.approx(metric.compute([4, 6], [0, 1]))


def test_manhattan_callable():
    """DistanceMetric nesnesi fonksiyon gibi çağrılabilmeli (__call__)."""
    metric = DistanceMetricFactory.create("manhattan")
    assert metric([0, 0], [3, 4]) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Euclidean vs Manhattan farkı
# ---------------------------------------------------------------------------

def test_euclidean_less_or_equal_manhattan():
    """L2 ≤ L1 her zaman doğrudur (norm ilişkisi)."""
    eu = DistanceMetricFactory.create("euclidean")
    ma = DistanceMetricFactory.create("manhattan")
    a, b = [1, 3, 5], [4, 7, 2]
    assert eu.compute(a, b) <= ma.compute(a, b) + 1e-9


# ---------------------------------------------------------------------------
# DistanceMetricFactory — create()
# ---------------------------------------------------------------------------

def test_factory_create_euclidean_type():
    """Factory 'euclidean' için doğru tip döndürmeli."""
    from coreLearn.distances import EuclideanDistance
    metric = DistanceMetricFactory.create("euclidean")
    assert isinstance(metric, EuclideanDistance)


def test_factory_create_manhattan_type():
    """Factory 'manhattan' için doğru tip döndürmeli."""
    from coreLearn.distances import ManhattanDistance
    metric = DistanceMetricFactory.create("manhattan")
    assert isinstance(metric, ManhattanDistance)


def test_factory_create_unknown_raises():
    """Bilinmeyen metrik adı ValueError fırlatmalı."""
    with pytest.raises(ValueError, match="Unknown"):
        DistanceMetricFactory.create("chebyshev")


def test_factory_create_returns_new_instance():
    """Her create() çağrısı bağımsız bir nesne döndürmeli."""
    m1 = DistanceMetricFactory.create("euclidean")
    m2 = DistanceMetricFactory.create("euclidean")
    assert m1 is not m2


# ---------------------------------------------------------------------------
# DistanceMetricFactory — available()
# ---------------------------------------------------------------------------

def test_factory_available_contains_defaults():
    """Varsayılan metrikler listelenebilmeli."""
    names = DistanceMetricFactory.available()
    assert "euclidean" in names
    assert "manhattan" in names


def test_factory_available_returns_list():
    """available() bir liste döndürmeli."""
    assert isinstance(DistanceMetricFactory.available(), list)


# ---------------------------------------------------------------------------
# DistanceMetricFactory — register()
# ---------------------------------------------------------------------------

def test_factory_register_and_use():
    """Yeni metrik kaydedilip kullanılabilmeli."""

    class ChebyshevDistance(DistanceMetric):
        def compute(self, a, b):
            return float(max(abs(x - y) for x, y in zip(a, b)))

    DistanceMetricFactory.register("chebyshev_tmp", ChebyshevDistance)
    metric = DistanceMetricFactory.create("chebyshev_tmp")
    # [1,2,3] ile [4,6,5] → max(3,4,2) = 4
    assert metric.compute([1, 2, 3], [4, 6, 5]) == pytest.approx(4.0)


def test_factory_register_appears_in_available():
    """Kaydedilen metrik available() listesinde görünmeli."""

    class DummyDistance(DistanceMetric):
        def compute(self, a, b):
            return 0.0

    DistanceMetricFactory.register("dummy_tmp", DummyDistance)
    assert "dummy_tmp" in DistanceMetricFactory.available()


def test_factory_register_invalid_class_raises():
    """DistanceMetric'ten türemeyen sınıf kaydı TypeError fırlatmalı."""

    class NotAMetric:
        pass

    with pytest.raises(TypeError):
        DistanceMetricFactory.register("bad", NotAMetric)


def test_factory_register_non_class_raises():
    """Sınıf olmayan bir nesne kaydı TypeError fırlatmalı."""
    with pytest.raises(TypeError):
        DistanceMetricFactory.register("bad_fn", lambda a, b: 0.0)


# ---------------------------------------------------------------------------
# DistanceMetric — soyut sınıf doğrudan örneklenemez
# ---------------------------------------------------------------------------

def test_distance_metric_is_abstract():
    """DistanceMetric doğrudan örneklenemez (abstract)."""
    with pytest.raises(TypeError):
        DistanceMetric()  # type: ignore[abstract]
